

from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.models.components.inputs import PDBDataset,BatchedAtomInput,AtomInput
from torch.utils.data import DataLoader
import os
from alphafold3_pytorch.models.components.alphafold3 import Alphafold3
from alphafold3_pytorch.models.components.inputs import pdb_input_to_molecule_input,molecule_to_atom_input
from alphafold3_pytorch.data.atom_datamodule import (collate_inputs_to_batched_atom_input,
    alphafold3_inputs_to_batched_atom_input,
    pdb_inputs_to_batched_atom_input,
)

def test_data_input():

    interface_mapping_path = os.path.join(
        "data", "test", "interface_cluster_mapping.csv"
    )
    chain_mapping_paths = [
        os.path.join(
            "data", "test", "ligand_chain_cluster_mapping.csv"
        ),
        os.path.join(
            "data",
            "test",
            "nucleic_acid_chain_cluster_mapping.csv",
        ),
        os.path.join(
            "data", "test", "peptide_chain_cluster_mapping.csv"
        ),
        os.path.join(
            "data", "test", "protein_chain_cluster_mapping.csv"
        ),
    ]

    sampler = WeightedPDBSampler(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=64,
    )


    dataset = PDBDataset(
    folder='/Users/cody/mmcif',
    sampler=sampler,
    sample_type='clustered',
    crop_size=10
        )

    mol=pdb_input_to_molecule_input(biomol=dataset[0])
    atom_input=molecule_to_atom_input(mol)
    batched_atom_input=collate_inputs_to_batched_atom_input([atom_input],atoms_per_window=27)

    alphafold3 = Alphafold3(
        dim_atom_inputs=3,
        dim_atompair_inputs=1,
        atoms_per_window=27,
        dim_template_feats=44,
        num_dist_bins=38,
        confidence_head_kwargs=dict(pairformer_depth=1),
        template_embedder_kwargs=dict(pairformer_stack_depth=1),
        msa_module_kwargs=dict(depth=1),
        pairformer_stack=dict(depth=2),
        diffusion_module_kwargs=dict(
            atom_encoder_depth=1,
            token_transformer_depth=1,
            atom_decoder_depth=1,
        ),
    )
    
    loss = alphafold3(**batched_atom_input.dict())
    loss.backward()

