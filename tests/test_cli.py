import os
from pathlib import Path
from shutil import rmtree

from alphafold3_pytorch.cli import cli
from alphafold3_pytorch.models.components.alphafold3 import Alphafold3

os.environ["TYPECHECK"] = "True"
os.environ["DEBUG"] = "True"


def test_cli():
    alphafold3 = Alphafold3(dim_atom_inputs=3, dim_template_feats=44, num_molecule_mods=0)

    checkpoint_path = os.path.join("test-folder", "test-cli-alphafold3.pt")
    alphafold3.save(checkpoint_path, overwrite=True)

    output_mmcif_filepath = os.path.join("test-folder", "output.cif")

    cli(
        [
            "--checkpoint",
            checkpoint_path,
            "-prot",
            "AG",
            "-prot",
            "TC",
            "--output",
            output_mmcif_filepath,
        ],
        standalone_mode=False,
    )

    assert Path(output_mmcif_filepath).exists()

    rmtree("test-folder")
