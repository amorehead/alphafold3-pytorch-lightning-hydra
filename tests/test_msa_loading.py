import os

import pytest

from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler
from alphafold3_pytorch.models.components.alphafold3 import Alphafold3
from alphafold3_pytorch.models.components.inputs import (
    PDBDataset,
    molecule_to_atom_input,
    pdb_input_to_molecule_input,
)
from alphafold3_pytorch.data.pdb_datamodule import pdb_inputs_to_batched_atom_input

def test_msa_loading():
    """Test a PDBDataset constructed using a WeightedPDBSampler."""
    data_test = os.path.join("data", "test")
    if not os.path.exists(os.path.join("data", "test", "mmcif")):
        pytest.skip("The directory `data/test/mmcif` is not populated yet.")

    interface_mapping_path = os.path.join(data_test, "interface_cluster_mapping.csv")
    chain_mapping_paths = [
        os.path.join(data_test, "ligand_chain_cluster_mapping.csv"),
        os.path.join(
            data_test,
            "nucleic_acid_chain_cluster_mapping.csv",
        ),
        os.path.join(data_test, "peptide_chain_cluster_mapping.csv"),
        os.path.join(data_test, "protein_chain_cluster_mapping.csv"),
    ]

    sampler = WeightedPDBSampler(
        chain_mapping_paths=chain_mapping_paths,
        interface_mapping_path=interface_mapping_path,
        batch_size=64,
    )

    pdb_input = PDBDataset(
        folder=os.path.join("data", "test","mmcif"), sampler=sampler, sample_type="default", crop_size=128,
        msa_dir=os.path.join("data", "val_msa","protein","val_protein")
    ,training=False)


    #mol_input = pdb_input_to_molecule_input(pdb_input=dataset[0])
    batched_atom_input = pdb_inputs_to_batched_atom_input(pdb_input[0], atoms_per_window=27)

    # training
    

test_msa_loading()   
