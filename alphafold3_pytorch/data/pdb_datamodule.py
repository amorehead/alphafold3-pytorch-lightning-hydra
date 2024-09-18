import multiprocessing
import os
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from beartype.typing import Any, Callable, Dict, List, Literal, Tuple, Union
from lightning import LightningDataModule
from torch import tensor
from torch.utils.data import DataLoader, Dataset

from alphafold3_pytorch.common import (
    amino_acid_constants,
    dna_constants,
    ligand_constants,
    rna_constants,
)
from alphafold3_pytorch.common.biomolecule import Biomolecule, get_residue_constants
from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.models.components.attention import (
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed,
)
from alphafold3_pytorch.models.components.inputs import (
    ATOM_DEFAULT_PAD_VALUES,
    UNCOLLATABLE_ATOM_INPUT_FIELDS,
    Alphafold3Input,
    AtomInput,
    BatchedAtomInput,
    PDBDataset,
    PDBInput,
    alphafold3_input_to_molecule_lengthed_molecule_input,
    maybe_transform_to_atom_inputs,
    molecule_lengthed_molecule_input_to_atom_input,
)
from alphafold3_pytorch.utils.model_utils import pad_at_dim
from alphafold3_pytorch.utils.pylogger import RankedLogger
from alphafold3_pytorch.utils.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists, not_exists

log = RankedLogger(__name__, rank_zero_only=False)

# dataloader and collation fn


@typecheck
def collate_inputs_to_batched_atom_input(
    inputs: List,
    int_pad_value=-1,
    atoms_per_window: int | None = None,
    map_input_fn: Callable | None = None,
    transform_to_atom_inputs: bool = True,
) -> BatchedAtomInput | None:
    """Collate function for a list of AtomInput objects.

    :param inputs: A list of AtomInput objects.
    :param int_pad_value: The padding value for integer tensors.
    :param atoms_per_window: The number of atoms per window.
    :param map_input_fn: A function to apply to each input before collation.
    :param transform_to_atom_inputs: Whether to transform the inputs to AtomInput objects.
    :return: A collated BatchedAtomInput object.
    """
    if exists(map_input_fn):
        inputs = [map_input_fn(i) for i in inputs]

    # go through all the inputs
    # and for any that is not AtomInput, try to transform it with the registered input type to corresponding registered function

    if transform_to_atom_inputs:
        atom_inputs = maybe_transform_to_atom_inputs(inputs)

        if len(atom_inputs) < len(inputs):
            # if some of the `inputs` could not be converted into `atom_inputs`,
            # randomly select a subset of the `atom_inputs` to duplicate to match
            # the expected number of `atom_inputs`
            assert (
                len(atom_inputs) > 0
            ), "No `AtomInput` objects could be created for the current batch."
            atom_inputs = random.choices(atom_inputs, k=len(inputs))  # nosec
    else:
        assert all(isinstance(i, AtomInput) for i in inputs), (
            "When `transform_to_atom_inputs=False`, all provided "
            "inputs must be of type `AtomInput`."
        )
        atom_inputs = inputs

    assert all(isinstance(i, AtomInput) for i in atom_inputs), (
        "All inputs must be of type `AtomInput`. "
        "If you want to transform the inputs to `AtomInput`, "
        "set `transform_to_atom_inputs=True`."
    )

    # take care of windowing the atompair_inputs and atompair_ids if they are not windowed already

    if exists(atoms_per_window):
        for atom_input in atom_inputs:
            atompair_inputs = atom_input.atompair_inputs
            atompair_ids = atom_input.atompair_ids

            atompair_inputs_is_windowed = atompair_inputs.ndim == 4

            if not atompair_inputs_is_windowed:
                atom_input.atompair_inputs = full_pairwise_repr_to_windowed(
                    atompair_inputs, window_size=atoms_per_window
                )

            if exists(atompair_ids):
                atompair_ids_is_windowed = atompair_ids.ndim == 3

                if not atompair_ids_is_windowed:
                    atom_input.atompair_ids = full_attn_bias_to_windowed(
                        atompair_ids, window_size=atoms_per_window
                    )

    # separate input dictionary into keys and values

    keys = list(atom_inputs[0].dict().keys())
    atom_inputs = [i.dict().values() for i in atom_inputs]

    outputs = []

    for key, grouped in zip(keys, zip(*atom_inputs)):
        # if all None, just return None

        not_none_grouped = [*filter(exists, grouped)]

        if len(not_none_grouped) == 0:
            outputs.append(None)
            continue

        # collate lists for uncollatable fields

        if key in UNCOLLATABLE_ATOM_INPUT_FIELDS:
            outputs.append(grouped)
            continue

        # default to empty tensor for any Nones

        one_tensor = not_none_grouped[0]

        dtype = one_tensor.dtype
        ndim = one_tensor.ndim

        # use -1 for padding int values, for assuming int are labels - if not, handle within alphafold3

        if key in ATOM_DEFAULT_PAD_VALUES:
            pad_value = ATOM_DEFAULT_PAD_VALUES[key]
        elif dtype in (torch.int, torch.long):
            pad_value = int_pad_value
        elif dtype == torch.bool:
            pad_value = False
        else:
            pad_value = 0.0

        # get the max lengths across all dimensions

        shapes_as_tensor = torch.stack(
            [tensor(tuple(g.shape) if exists(g) else ((0,) * ndim)).int() for g in grouped],
            dim=-1,
        )

        max_lengths = shapes_as_tensor.amax(dim=-1)

        default_tensor = torch.full(max_lengths.tolist(), pad_value, dtype=dtype)

        # pad across all dimensions

        padded_inputs = []

        for inp in grouped:
            if not_exists(inp):
                padded_inputs.append(default_tensor)
                continue

            for dim, max_length in enumerate(max_lengths.tolist()):
                inp = pad_at_dim(inp, (0, max_length - inp.shape[dim]), value=pad_value, dim=dim)

            padded_inputs.append(inp)

        # stack

        stacked = torch.stack(padded_inputs)

        outputs.append(stacked)

    # batched atom input dictionary

    batched_atom_input_dict = dict(tuple(zip(keys, outputs)))

    # reconstitute dictionary

    batched_atom_inputs = BatchedAtomInput(**batched_atom_input_dict)
    return batched_atom_inputs


@typecheck
def alphafold3_inputs_to_batched_atom_input(
    inp: Alphafold3Input | List[Alphafold3Input], **collate_kwargs
) -> BatchedAtomInput:
    """Convert a list of Alphafold3Input objects to a BatchedAtomInput object.

    :param inp: A list of Alphafold3Input objects.
    :param collate_kwargs: Additional keyword arguments for collation.
    :return: A BatchedAtomInput object.
    """
    if isinstance(inp, Alphafold3Input):
        inp = [inp]

    atom_inputs = maybe_transform_to_atom_inputs(inp)
    return collate_inputs_to_batched_atom_input(atom_inputs, **collate_kwargs)


@typecheck
def alphafold3_input_to_biomolecule(
    af3_input: Alphafold3Input, atom_positions: np.ndarray
) -> Biomolecule:
    """Convert an `Alphafold3Input` object into a corresponding `Biomolecule` object.

    :param af3_input: The `AlphaFold3Input` object for multi-domain `Biomolecules`.
    :param atom_positions: The sampled or reference atom coordinates of shape [num_res, repr_dimension (47), 3].
    :return: A `Biomolecule` object for data handling within the rest of the codebase.
    """
    af3_atom_input = molecule_lengthed_molecule_input_to_atom_input(
        alphafold3_input_to_molecule_lengthed_molecule_input(af3_input)
    )

    # Ensure that the atom positions are of the correct shape
    if atom_positions is not None:
        assert atom_positions.shape[0] == len(
            af3_atom_input.molecule_ids
        ), "Please ensure that the atoms are of the shape [num_res, repr, 3]"
        assert (
            atom_positions.shape[-1] == 3
        ), "Please ensure that the atoms are of the shape [num_res, repr, 3]"

    # Step 1. Get the various intermediate inputs
    # Hacky solution: Need to double up on ligand because metal constants dont exist yet
    ALL_restypes = np.concatenate(
        [
            amino_acid_constants.restype_atom47_to_compact_atom,
            rna_constants.restype_atom47_to_compact_atom,
            dna_constants.restype_atom47_to_compact_atom,
            ligand_constants.restype_atom47_to_compact_atom,
            ligand_constants.restype_atom47_to_compact_atom,
        ],
        axis=0,
    )
    molecule_ids = af3_atom_input.molecule_ids.cpu().numpy()
    restype_to_atom = np.array([ALL_restypes[mol_idx] for mol_idx in molecule_ids])
    molecule_types = np.nonzero(af3_atom_input.is_molecule_types)[:, 1]
    res_rep_atom_indices = [
        get_residue_constants(res_chem_index=molecule_type.item()).res_rep_atom_index
        for molecule_type in molecule_types
    ]

    # Step 2. Atom Names
    # atom_names: array of strings [num_res], each residue is denoted by representative atom name
    atom_names = []

    for res_idx in range(len(molecule_ids)):
        molecule_type = molecule_types[res_idx].item()
        residue = molecule_ids[res_idx].item()
        residue_offset = get_residue_constants(res_chem_index=molecule_type).min_restype_num
        residue_idx = residue - residue_offset
        atom_idx = res_rep_atom_indices[res_idx]
        # If the molecule type is a protein, RNA, or DNA
        if molecule_type < 3:
            # Dictionary of Residue to Atoms
            res_to_atom = get_residue_constants(
                res_chem_index=molecule_type
            ).restype_name_to_compact_atom_names
            residue_name = get_residue_constants(res_chem_index=molecule_type).resnames[
                residue_idx
            ]
            atom_names.append(res_to_atom[residue_name][atom_idx])
        else:
            # TODO: See if there is a way to add in metals as separate to the ligands
            atom_name = get_residue_constants(
                res_chem_index=molecule_type
            ).restype_name_to_compact_atom_names["UNL"][atom_idx]
            atom_names.append(atom_name)

    # Step 3. Restypes
    # restypes: np.array [num_res] w/ values from 0 to 32
    res_types = molecule_ids.copy()

    # Step 4. Atom Masks
    # atom_masks: np.array [num_res, num_atom_types (47)]
    # Due to the first Atom that's present being a zero due to zero indexed counts we force it to be a one.
    atom_masks = np.stack(
        [
            np.array(np.concatenate([np.array([1]), r2a[1:]]) != 0).astype(int)
            for r2a in restype_to_atom
        ]
    )

    # Step 5. Residue Index
    # residue_index: np.array [num_res], 1-indexed
    residue_index = af3_atom_input.additional_molecule_feats.cpu().numpy()[:, 0] + 1

    # Step 6. Chain Index
    # chain_index: np.array [num_res], borrow the entity IDs (vs sym_ids, idx3) as chain IDs
    chain_index = af3_atom_input.additional_molecule_feats.cpu().numpy()[:, 2]

    # Step 7. Chain IDs
    # chain_ids: list of strings [num_res], each residue is denoted by chain ID
    chain_ids = [str(x) for x in chain_index]

    # Step 8. B-Factors
    # b_factors: np.ndarray [num_res, num_atom_type]
    b_factors = np.ones_like(atom_masks)

    # Step 9. ChemIDs
    # TODO: The individual Ligand Molecules when split up by RDKit are not being assigned a specific chemical ID
    # chemids: list of strings [num_res], each residue is denoted by chemical ID
    chemids = []

    for idx in range(len(molecule_ids)):
        mt = molecule_types[idx].item()
        restypes = get_residue_constants(res_chem_index=mt).restypes
        min_res_offset = get_residue_constants(res_chem_index=mt).min_restype_num
        restype_dict = {min_res_offset + i: restype for i, restype in enumerate(restypes)}
        try:
            one_letter = restype_dict[molecule_ids[idx].item()]
            chemids.append(get_residue_constants(res_chem_index=mt).restype_1to3[one_letter])
        except KeyError:
            chemids.append("UNK")

    chemids = np.array(chemids)

    # Step 10. ChemTypes
    # chemtypes: np.array [num_res], each residue is denoted by chemical type 0-4
    chemtypes = np.nonzero(af3_atom_input.is_molecule_types.cpu().numpy())[1]

    # Step 11. Entity to Chain
    # entity_to_chain: dict, entity ID to chain ID
    # quick and dirty assignment
    entity_to_chain = {int(x): [int(x)] for x in np.unique(chain_index)}

    # Step 12. Biomolecule Object
    biomol = Biomolecule(
        atom_positions=atom_positions,
        atom_name=atom_names,
        restype=res_types,
        atom_mask=atom_masks,
        residue_index=residue_index,
        chain_index=chain_index,
        chain_id=chain_ids,
        b_factors=b_factors,
        chemid=chemids,
        chemtype=chemtypes,
        bonds=None,
        unique_res_atom_names=None,  # TODO: Need to find how to use the ligand information here
        author_cri_to_new_cri=None,
        chem_comp_table=None,
        entity_to_chain=entity_to_chain,
        mmcif_to_author_chain=None,
        mmcif_metadata={
            "_pdbx_audit_revision_history.revision_date": [
                f"{datetime.today().strftime('%Y-%m-%d')}"
            ]
        },
    )

    return biomol


@typecheck
def pdb_inputs_to_batched_atom_input(
    inp: PDBInput | List[PDBInput], **collate_kwargs
) -> BatchedAtomInput:
    """Convert a list of PDBInput objects to a BatchedAtomInput object.

    :param inp: A list of PDBInput objects.
    :param collate_kwargs: Additional keyword arguments for collation.
    :return: A BatchedAtomInput object.
    """
    if isinstance(inp, PDBInput):
        inp = [inp]

    atom_inputs = maybe_transform_to_atom_inputs(inp)
    return collate_inputs_to_batched_atom_input(atom_inputs, **collate_kwargs)


@typecheck
def AF3DataLoader(
    *args,
    atoms_per_window: int | None = None,
    map_input_fn: Callable | None = None,
    transform_to_atom_inputs: bool = True,
    **kwargs,
):
    """Create a `torch.utils.data.DataLoader` with the `collate_inputs_to_batched_atom_input` or
    `map_input_fn` function for data collation.

    :param args: The arguments to pass to `torch.utils.data.DataLoader`.
    :param atoms_per_window: The number of atoms per window.
    :param map_input_fn: A function to apply to each input before collation.
    :param transform_to_atom_inputs: Whether to transform the inputs to AtomInput objects.
    :param kwargs: The keyword arguments to pass to `torch.utils.data.DataLoader`.
    :return: A `torch.utils.data.DataLoader` with a custom AF3 collate function.
    """
    collate_fn = partial(
        collate_inputs_to_batched_atom_input,
        atoms_per_window=atoms_per_window,
        transform_to_atom_inputs=transform_to_atom_inputs,
    )

    if exists(map_input_fn):
        collate_fn = partial(collate_fn, map_input_fn=map_input_fn)

    return DataLoader(*args, collate_fn=collate_fn, **kwargs)


class PDBDataModule(LightningDataModule):
    """`LightningDataModule` for a PDB-based dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = os.path.join("data", "pdb_data"),
        msa_dir: str = os.path.join("data", "pdb_data", "data_caches", "msa"),
        templates_dir: str = os.path.join("data", "pdb_data", "data_caches", "template"),
        sample_type: Literal["default", "clustered"] = "default",
        contiguous_weight: float = 0.2,
        spatial_weight: float = 0.4,
        spatial_interface_weight: float = 0.4,
        crop_size: int = 384,
        max_msas_per_chain: int | None = None,
        max_num_msa_tokens: int | None = None,
        max_templates_per_chain: int | None = None,
        num_templates_per_chain: int | None = None,
        max_num_template_tokens: int | None = None,
        max_train_length: int | None = None,
        max_val_length: int | None = None,
        train_cutoff_date: str | None = None,
        kalign_binary_path: str | None = None,
        sampling_weight_for_disorder_pdb_distillation: float = 0.02,
        train_on_transcription_factor_distillation_sets: bool = False,
        pdb_distillation: bool | None = None,
        constraints: List[str] | None = None,
        constraints_ratio: float = 0.1,
        max_number_of_chains: int = 20,
        atoms_per_window: int | None = None,
        map_dataset_input_fn: Callable | None = None,
        train_val_test_split: Tuple[int, int, int] | None = None,
        shuffle_train_val_test_subsets: bool = True,
        overfitting_train_examples: bool = False,
        ablate_weighted_pdb_sampler: bool = False,
        filter_out_pdb_ids: List[str] | None = None,
        sample_only_pdb_ids: List[str] | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        multiprocessing_context: Union[str, multiprocessing.context.BaseContext] | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()

        assert (
            sum([contiguous_weight, spatial_weight, spatial_interface_weight]) == 1.0
        ), "The sum of contiguous_weight, spatial_weight, and spatial_interface_weight must be equal to 1.0."

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

        # if map dataset function given, curry into DataLoader

        self.dataloader_class = partial(
            AF3DataLoader,
            atoms_per_window=atoms_per_window,
            transform_to_atom_inputs=False,
        )

        if exists(map_dataset_input_fn):
            self.dataloader_class = partial(
                self.dataloader_class, map_input_fn=map_dataset_input_fn
            )

        # load dataset splits

        sample_only_pdb_ids = (
            # sample only specific PDB IDs as requested
            set(self.hparams.sample_only_pdb_ids)
            if exists(self.hparams.sample_only_pdb_ids)
            else None
        )
        sample_only_pdb_ids_list = (
            list(sample_only_pdb_ids) if exists(sample_only_pdb_ids) else None
        )

        filter_out_pdb_ids = (
            # filter out specific PDB IDs as requested
            set(self.hparams.filter_out_pdb_ids)
            if exists(self.hparams.filter_out_pdb_ids)
            else None
        )
        filter_out_pdb_ids_list = list(filter_out_pdb_ids) if exists(filter_out_pdb_ids) else None

        # data paths for each split
        for split in ("train", "val", "test"):
            path_split = split
            if self.hparams.overfitting_train_examples:
                # NOTE: when overfitting to a subset of examples,
                # we want to load the training set multiple times
                # to ensure that the model sees the same examples
                # across training, validation, and testing
                path_split = "train"

            setattr(
                self,
                f"{split}_mmcifs_dir",
                os.path.join(self.hparams.data_dir, f"{path_split}_mmcifs"),
            )
            setattr(
                self,
                f"{split}_clusterings_dir",
                os.path.join(self.hparams.data_dir, "data_caches", f"{path_split}_clusterings"),
            )
            setattr(
                self,
                f"{split}_msa_dir",
                os.path.join(self.hparams.msa_dir, f"{path_split}_msas"),
            )
            setattr(
                self,
                f"{split}_templates_dir",
                os.path.join(self.hparams.templates_dir, f"{path_split}_templates"),
            )
            setattr(
                self,
                f"{split}_chain_mapping_paths",
                [
                    os.path.join(
                        getattr(self, f"{split}_clusterings_dir"),
                        "ligand_chain_cluster_mapping.csv",
                    ),
                    os.path.join(
                        getattr(self, f"{split}_clusterings_dir"),
                        "nucleic_acid_chain_cluster_mapping.csv",
                    ),
                    os.path.join(
                        getattr(self, f"{split}_clusterings_dir"),
                        "peptide_chain_cluster_mapping.csv",
                    ),
                    os.path.join(
                        getattr(self, f"{split}_clusterings_dir"),
                        "protein_chain_cluster_mapping.csv",
                    ),
                ],
            )
            setattr(
                self,
                f"{split}_interface_mapping_path",
                os.path.join(
                    getattr(self, f"{path_split}_clusterings_dir"),
                    "interface_cluster_mapping.csv",
                ),
            )

        # training set
        sampler_train = (
            None
            if self.hparams.ablate_weighted_pdb_sampler
            else WeightedPDBSampler(
                chain_mapping_paths=self.train_chain_mapping_paths,
                interface_mapping_path=self.train_interface_mapping_path,
                batch_size=1,
                pdb_ids_to_skip=filter_out_pdb_ids_list,
                pdb_ids_to_keep=sample_only_pdb_ids_list,
            )
        )
        self.data_train = PDBDataset(
            folder=self.train_mmcifs_dir,
            sampler=sampler_train,
            sample_type=self.hparams.sample_type,
            contiguous_weight=self.hparams.contiguous_weight,
            spatial_weight=self.hparams.spatial_weight,
            spatial_interface_weight=self.hparams.spatial_interface_weight,
            crop_size=self.hparams.crop_size,
            max_msas_per_chain=self.hparams.max_msas_per_chain,
            max_num_msa_tokens=self.hparams.max_num_msa_tokens,
            max_templates_per_chain=self.hparams.max_templates_per_chain,
            num_templates_per_chain=self.hparams.num_templates_per_chain,
            max_num_template_tokens=self.hparams.max_num_template_tokens,
            max_length=self.hparams.max_train_length,
            cutoff_date=self.hparams.train_cutoff_date,
            kalign_binary_path=self.hparams.kalign_binary_path,
            training=True,
            inference=False,
            constraints=self.hparams.constraints,
            constraints_ratio=self.hparams.constraints_ratio,
            filter_out_pdb_ids=filter_out_pdb_ids,
            sample_only_pdb_ids=sample_only_pdb_ids,
            return_atom_inputs=True,
            msa_dir=self.train_msa_dir,
            templates_dir=self.train_templates_dir,
        )

        # validation set
        sampler_val = (
            None
            if self.hparams.ablate_weighted_pdb_sampler
            else WeightedPDBSampler(
                chain_mapping_paths=self.val_chain_mapping_paths,
                interface_mapping_path=self.val_interface_mapping_path,
                batch_size=1,
                pdb_ids_to_skip=filter_out_pdb_ids_list,
                pdb_ids_to_keep=sample_only_pdb_ids_list,
            )
        )
        self.data_val = PDBDataset(
            folder=self.val_mmcifs_dir,
            sampler=sampler_val,
            sample_type=self.hparams.sample_type,
            contiguous_weight=self.hparams.contiguous_weight,
            spatial_weight=self.hparams.spatial_weight,
            spatial_interface_weight=self.hparams.spatial_interface_weight,
            crop_size=self.hparams.crop_size,
            max_msas_per_chain=self.hparams.max_msas_per_chain,
            max_num_msa_tokens=self.hparams.max_num_msa_tokens,
            max_templates_per_chain=self.hparams.max_templates_per_chain,
            num_templates_per_chain=self.hparams.num_templates_per_chain,
            max_num_template_tokens=self.hparams.max_num_template_tokens,
            max_length=self.hparams.max_val_length,
            kalign_binary_path=self.hparams.kalign_binary_path,
            training=False,
            inference=False,
            constraints=self.hparams.constraints,
            constraints_ratio=self.hparams.constraints_ratio,
            filter_out_pdb_ids=filter_out_pdb_ids,
            sample_only_pdb_ids=sample_only_pdb_ids,
            return_atom_inputs=True,
            msa_dir=self.val_msa_dir,
            templates_dir=self.val_templates_dir,
        )

        # evaluation set
        sampler_test = (
            None
            if self.hparams.ablate_weighted_pdb_sampler
            else WeightedPDBSampler(
                chain_mapping_paths=self.test_chain_mapping_paths,
                interface_mapping_path=self.test_interface_mapping_path,
                batch_size=1,
                pdb_ids_to_skip=filter_out_pdb_ids_list,
                pdb_ids_to_keep=sample_only_pdb_ids_list,
            )
        )
        self.data_test = PDBDataset(
            folder=self.test_mmcifs_dir,
            sampler=sampler_test,
            sample_type=self.hparams.sample_type,
            contiguous_weight=self.hparams.contiguous_weight,
            spatial_weight=self.hparams.spatial_weight,
            spatial_interface_weight=self.hparams.spatial_interface_weight,
            crop_size=self.hparams.crop_size,
            max_msas_per_chain=self.hparams.max_msas_per_chain,
            max_num_msa_tokens=self.hparams.max_num_msa_tokens,
            max_templates_per_chain=self.hparams.max_templates_per_chain,
            num_templates_per_chain=self.hparams.num_templates_per_chain,
            max_num_template_tokens=self.hparams.max_num_template_tokens,
            kalign_binary_path=self.hparams.kalign_binary_path,
            training=False,
            inference=False,
            constraints=self.hparams.constraints,
            constraints_ratio=self.hparams.constraints_ratio,
            filter_out_pdb_ids=filter_out_pdb_ids,
            sample_only_pdb_ids=sample_only_pdb_ids,
            return_atom_inputs=True,
            msa_dir=self.test_msa_dir,
            templates_dir=self.test_templates_dir,
        )

        # subsample dataset splits as requested

        if exists(self.hparams.train_val_test_split):
            train_count, val_count, test_count = self.hparams.train_val_test_split

            train_indices = list(range(len(self.data_train)))
            val_indices = list(range(len(self.data_val)))
            test_indices = list(range(len(self.data_test)))

            if (
                self.hparams.shuffle_train_val_test_subsets
                and not self.hparams.overfitting_train_examples
            ):
                random.shuffle(train_indices)  # nosec
                random.shuffle(val_indices)  # nosec
                random.shuffle(test_indices)  # nosec

            self.data_train = torch.utils.data.Subset(self.data_train, train_indices[:train_count])
            self.data_val = torch.utils.data.Subset(self.data_val, val_indices[:val_count])
            self.data_test = torch.utils.data.Subset(self.data_test, test_indices[:test_count])

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if exists(self.trainer):
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
            if self.batch_size_per_device > 1:
                raise RuntimeError(
                    "Only a `batch_size_per_device` of 1 is supported for now "
                    "due to the input requirements of the `MultiChainPermutationAlignment` algorithm."
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.dataloader_class(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            multiprocessing_context=self.hparams.multiprocessing_context,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return self.dataloader_class(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            multiprocessing_context=self.hparams.multiprocessing_context,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self.dataloader_class(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            multiprocessing_context=self.hparams.multiprocessing_context,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            drop_last=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = PDBDataModule()
