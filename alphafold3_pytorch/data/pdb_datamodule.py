import os
import random
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.models.components.attention import (
    full_attn_bias_to_windowed,
    full_pairwise_repr_to_windowed,
)
from alphafold3_pytorch.models.components.inputs import (
    ATOM_DEFAULT_PAD_VALUES,
    UNCOLLATABLE_ATOM_INPUT_FIELDS,
    Alphafold3Input,
    BatchedAtomInput,
    PDBDataset,
    PDBInput,
    maybe_transform_to_atom_inputs,
)
from alphafold3_pytorch.utils.model_utils import pad_at_dim
from alphafold3_pytorch.utils.tensor_typing import typecheck
from alphafold3_pytorch.utils.utils import exists

# dataloader and collation fn


@typecheck
def collate_inputs_to_batched_atom_input(
    inputs: List,
    int_pad_value=-1,
    atoms_per_window: int | None = None,
    map_input_fn: Callable | None = None,
) -> BatchedAtomInput | None:
    """Collate function for a list of AtomInput objects.

    :param inputs: A list of AtomInput objects.
    :param int_pad_value: The padding value for integer tensors.
    :param atoms_per_window: The number of atoms per window.
    :param map_input_fn: A function to apply to each input before collation.
    :return: A collated BatchedAtomInput object.
    """
    if exists(map_input_fn):
        inputs = [map_input_fn(i) for i in inputs]

    # go through all the inputs
    # and for any that is not AtomInput, try to transform it with the registered input type to corresponding registered function

    atom_inputs = maybe_transform_to_atom_inputs(inputs)
    if not atom_inputs:
        return None

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
            [Tensor(tuple(g.shape) if exists(g) else ((0,) * ndim)).int() for g in grouped],
            dim=-1,
        )

        max_lengths = shapes_as_tensor.amax(dim=-1)

        default_tensor = torch.full(max_lengths.tolist(), pad_value, dtype=dtype)

        # pad across all dimensions

        padded_inputs = []

        for inp in grouped:
            if not exists(inp):
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
    **kwargs,
):
    """Create a `torch.utils.data.DataLoader` with the `collate_inputs_to_batched_atom_input` or
    `map_input_fn` function for data collation.

    :param args: The arguments to pass to `torch.utils.data.DataLoader`.
    :param atoms_per_window: The number of atoms per window.
    :param map_input_fn: A function to apply to each input before collation.
    :param kwargs: The keyword arguments to pass to `torch.utils.data.DataLoader`.
    :return: A `torch.utils.data.DataLoader` with a custom AF3 collate function.
    """
    collate_fn = partial(collate_inputs_to_batched_atom_input, atoms_per_window=atoms_per_window)

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
        max_templates_per_chain: int | None = None,
        num_templates_per_chain: int | None = None,
        kalign_binary_path: str | None = None,
        sampling_weight_for_disorder_pdb_distillation: float = 0.02,
        train_on_transcription_factor_distillation_sets: bool = False,
        pdb_distillation: bool | None = None,
        max_number_of_chains: int = 20,
        atoms_per_window: int | None = None,
        map_dataset_input_fn: Callable | None = None,
        train_val_test_split: Tuple[int, int, int] | None = None,
        shuffle_train_val_test_subsets: bool = True,
        overfitting_train_examples: bool = False,
        sample_only_pdb_ids: List[str] | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
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

        self.dataloader_class = partial(AF3DataLoader, atoms_per_window=atoms_per_window)

        if exists(map_dataset_input_fn):
            self.dataloader_class = partial(
                self.dataloader_class, map_input_fn=map_dataset_input_fn
            )

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
        if self.trainer is not None:
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

        # load dataset splits only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # sample only specific PDB IDs as requested

            sample_only_pdb_ids = (
                set(self.hparams.sample_only_pdb_ids)
                if exists(self.hparams.sample_only_pdb_ids)
                else None
            )
            sample_only_pdb_ids_list = (
                list(sample_only_pdb_ids) if exists(sample_only_pdb_ids) else None
            )

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
                    os.path.join(
                        self.hparams.data_dir, "data_caches", f"{path_split}_clusterings"
                    ),
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

            self.data_train = PDBDataset(
                folder=self.train_mmcifs_dir,
                sampler=WeightedPDBSampler(
                    chain_mapping_paths=self.train_chain_mapping_paths,
                    interface_mapping_path=self.train_interface_mapping_path,
                    batch_size=1,
                    pdb_ids_to_keep=sample_only_pdb_ids_list,
                ),
                sample_type=self.hparams.sample_type,
                contiguous_weight=self.hparams.contiguous_weight,
                spatial_weight=self.hparams.spatial_weight,
                spatial_interface_weight=self.hparams.spatial_interface_weight,
                crop_size=self.hparams.crop_size,
                max_msas_per_chain=self.hparams.max_msas_per_chain,
                max_templates_per_chain=self.hparams.max_templates_per_chain,
                num_templates_per_chain=self.hparams.num_templates_per_chain,
                kalign_binary_path=self.hparams.kalign_binary_path,
                training=True,
                sample_only_pdb_ids=sample_only_pdb_ids,
                msa_dir=self.train_msa_dir,
                templates_dir=self.train_templates_dir,
            )

            # validation set

            self.data_val = PDBDataset(
                folder=self.val_mmcifs_dir,
                sampler=WeightedPDBSampler(
                    chain_mapping_paths=self.val_chain_mapping_paths,
                    interface_mapping_path=self.val_interface_mapping_path,
                    batch_size=1,
                    pdb_ids_to_keep=sample_only_pdb_ids_list,
                ),
                sample_type=self.hparams.sample_type,
                contiguous_weight=self.hparams.contiguous_weight,
                spatial_weight=self.hparams.spatial_weight,
                spatial_interface_weight=self.hparams.spatial_interface_weight,
                crop_size=self.hparams.crop_size,
                max_msas_per_chain=self.hparams.max_msas_per_chain,
                max_templates_per_chain=self.hparams.max_templates_per_chain,
                num_templates_per_chain=self.hparams.num_templates_per_chain,
                kalign_binary_path=self.hparams.kalign_binary_path,
                training=False,
                sample_only_pdb_ids=sample_only_pdb_ids,
                msa_dir=self.val_msa_dir,
                templates_dir=self.val_templates_dir,
            )

            # evaluation set

            self.data_test = PDBDataset(
                folder=self.test_mmcifs_dir,
                sampler=WeightedPDBSampler(
                    chain_mapping_paths=self.test_chain_mapping_paths,
                    interface_mapping_path=self.test_interface_mapping_path,
                    batch_size=1,
                    pdb_ids_to_keep=sample_only_pdb_ids_list,
                ),
                sample_type=self.hparams.sample_type,
                contiguous_weight=self.hparams.contiguous_weight,
                spatial_weight=self.hparams.spatial_weight,
                spatial_interface_weight=self.hparams.spatial_interface_weight,
                crop_size=self.hparams.crop_size,
                max_msas_per_chain=self.hparams.max_msas_per_chain,
                max_templates_per_chain=self.hparams.max_templates_per_chain,
                num_templates_per_chain=self.hparams.num_templates_per_chain,
                kalign_binary_path=self.hparams.kalign_binary_path,
                training=False,
                sample_only_pdb_ids=sample_only_pdb_ids,
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

                self.data_train = torch.utils.data.Subset(
                    self.data_train, train_indices[:train_count]
                )
                self.data_val = torch.utils.data.Subset(self.data_val, val_indices[:val_count])
                self.data_test = torch.utils.data.Subset(self.data_test, test_indices[:test_count])

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self.dataloader_class(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
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
