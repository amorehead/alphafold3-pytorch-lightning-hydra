from pathlib import Path

import pytest
from torch.utils.data import Sampler

from alphafold3_pytorch.data.weighted_pdb_sampler import WeightedPDBSampler

TEST_FOLDER = Path("./data/test/data_caches/clusterings/")

INTERFACE_MAPPING_PATH = str(TEST_FOLDER / "interface_cluster_mapping.csv")

CHAIN_MAPPING_PATHS = [
    str(TEST_FOLDER / "ligand_chain_cluster_mapping.csv"),
    str(TEST_FOLDER / "nucleic_acid_chain_cluster_mapping.csv"),
    str(TEST_FOLDER / "peptide_chain_cluster_mapping.csv"),
    str(TEST_FOLDER / "protein_chain_cluster_mapping.csv"),
]


@pytest.fixture
def sampler():
    """Return a `WeightedPDBSampler` object."""
    return WeightedPDBSampler(
        chain_mapping_paths=CHAIN_MAPPING_PATHS,
        interface_mapping_path=INTERFACE_MAPPING_PATH,
        batch_size=4,
    )


def test_sample(sampler: Sampler):
    """Test the sampling method of the `WeightedSamplerPDB` class."""
    assert len(sampler.sample(4)) == 4, "The sampled batch size does not match the expected size."


def test_cluster_based_sample(sampler: Sampler):
    """Test the cluster-based sampling method of the `WeightedSamplerPDB` class."""
    assert (
        len(sampler.cluster_based_sample(4)) == 4
    ), "The cluster-based sampled batch size does not match the expected size."
