from __future__ import annotations

from pathlib import Path

import click
import torch
from Bio.PDB.mmcifio import MMCIFIO

from alphafold3_pytorch.models.components.alphafold3 import Alphafold3
from alphafold3_pytorch.models.components.inputs import Alphafold3Input

# simple cli using click


@click.command()
@click.option("-ckpt", "--checkpoint", type=str, help="Path to AlphaFold 3 checkpoint")
@click.option("-prot", "--protein", type=str, multiple=True, help="Protein sequences")
@click.option("-rna", "--rna", type=str, multiple=True, help="Single-stranded RNA sequences")
@click.option("-dna", "--dna", type=str, multiple=True, help="Single-stranded DNA sequences")
@click.option("-steps", "--num-sample-steps", type=int, help="Number of sampling steps to take")
@click.option("-cuda", "--use-cuda", type=bool, help="Use CUDA if available")
@click.option("-o", "--output", type=str, help="Output path", default="output.cif")
def cli(
    checkpoint: str,
    protein: list[str],
    rna: list[str],
    dna: list[str],
    num_sample_steps: int,
    use_cuda: bool,
    output: str,
):
    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f"AlphaFold 3 checkpoint must exist at {str(checkpoint_path)}"

    alphafold3_input = Alphafold3Input(
        proteins=protein,
        ss_rna=rna,
        ss_dna=dna,
    )

    alphafold3 = Alphafold3.init_and_load(checkpoint_path)

    if use_cuda and torch.cuda.is_available():
        alphafold3 = alphafold3.cuda()

    alphafold3.eval()

    (structure,) = alphafold3.forward_with_alphafold3_inputs(
        alphafold3_input, return_bio_pdb_structures=True, num_sample_steps=num_sample_steps
    )

    output_path = Path(output)
    output_path.parents[0].mkdir(exist_ok=True, parents=True)

    pdb_writer = MMCIFIO()
    pdb_writer.set_structure(structure)
    pdb_writer.save(str(output_path))

    print(f"mmCIF file saved to {str(output_path)}")
