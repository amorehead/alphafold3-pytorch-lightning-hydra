#!/usr/bin/env python
"""
Implementation of the evaluation procedure described in sections 6.3 - 6.4 of the 
Alphafold3 supplementary information document.

https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
"""

import argparse
import itertools
import math
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set 
from typing import Tuple

import torch

from alphafold3_pytorch import Alphafold3
from alphafold3_pytorch.utils.model_utils import dict_to_device
from alphafold3_pytorch.models.components.inputs import PDBDataset
from alphafold3_pytorch.models.components.inputs import pdb_dataset_to_atom_inputs
from alphafold3_pytorch.models.components.inputs import IS_RNA_INDEX 
from alphafold3_pytorch.models.components.inputs import IS_DNA_INDEX
from alphafold3_pytorch.models.components.alphafold3 import ComputeModelSelectionScore
from alphafold3_pytorch.data.pdb_datamodule import AF3DataLoader
from alphafold3_pytorch.models.alphafold3_module import Sample


# Constants.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NUM_IDENTICAL_ENTITIES_FOR_EXHAUSTIVE_SEARCH: int = 3
EXPECTED_DIM_ATOM_INPUT: int = 3
INT_PAD_VALUE: int = -1
FLOAT_PAD_VALUE: float = 0.0

# Supporting utils.
lddt_calc_module = ComputeModelSelectionScore()


def group_identical_entity_asym_ids(
    token_idx_to_asym_ids_map,
    token_idx_to_entity_ids_map,
) -> Set[Tuple[int]]:
    """ 
    Evaluate the token to asym and token to entity mappings of a complex. Assign the 
    complex's asym ids into 'identical' entity groups. Entity groups are defined as sets 
    of asym ids that have all equal counts of tokens from the same entity ids.
    """

    identical_asym_ids = {}

    for asym_id in [x.item() for x in token_idx_to_asym_ids_map.unique()]:

        # Fetch the token indices that make up the asym_id/chain.
        asym_id_idxs = torch.where(token_idx_to_asym_ids_map == asym_id)

        # Get the entities that compose the chain and their counts (in chain).
        entity_ids_in_asym_id, entity_id_counts_in_asym_id = torch.unique(
            token_idx_to_entity_ids_map[asym_id_idxs], 
            return_counts=True
        )

        # This composition of distinct entity ids and their frequencies in the asym id
        # characterizes which chains are "identical". We can use an immutable set as
        # an order-free key to match "identical" asyms together
        chain_identifier = frozenset(
            (value.item(), count.item()) for value, count in 
            zip(entity_ids_in_asym_id, entity_id_counts_in_asym_id)
        )

        identical_asym_ids.setdefault(chain_identifier, []).append(asym_id)

    return set([tuple(x) for x in identical_asym_ids.values()])


def exhaustive_search_for_optimal_asym_id_assignments(
    entity_group_asym_ids: Tuple[int],
    atom_to_asym_id_map,
    is_molecule_types,
    predicted_atom_pos,    
    true_atom_pos,
):
    """ Exhaustive search of all possible identical asym_id swaps to maximizes lddt. """

    # Identify all permutations of the ordering of the listed asym ids.
    entity_group_asym_id_permutations = itertools.permutations(entity_group_asym_ids)
    max_lddt = float('-inf')
    best_permutation = None
    best_predicted_atom_pos = None

    # Iterate through each permutation of possible asym_id swaps to find the one
    # that maximizes the lddt of the complex.
    for entity_group_asym_id_permutation in entity_group_asym_id_permutations:

        permutations_predicted_atom_pos = torch.clone(predicted_atom_pos)

        # Create a new predicted atom pos tensor by performing the asym id pos swaps.
        for idx, swap_asym_id in enumerate(entity_group_asym_id_permutation):

            # The asym id to be swapped with.
            base_asym_id = entity_group_asym_ids[idx]

            # Skip self-swaps.
            if base_asym_id == swap_asym_id:
                continue

            # Isolate the atomic idxs to swap values for.
            original_atom_position_idxs = atom_to_asym_id_map == base_asym_id
            swap_atom_position_idxs = atom_to_asym_id_map == swap_asym_id

            # Ensure the number of atoms composing the two asym ids are identical. If 
            # they're not, something went wrong.
            if original_atom_position_idxs.sum() != swap_atom_position_idxs.sum():
                raise ValueError(
                    "Something went wrong, when trying to swap asym_ids to find the "
                    "closest matching assignment to the ground truth, the num atoms "
                    "beloinging to both asym ids do not match. Thus they must not be "
                    f"identical. Found shapes [{original_atom_position_idxs.size(0)}] "
                    f"and [{swap_atom_position_idxs.size(0)}]."
                )

            # Perform the swap on a copy of the predictions.
            permutations_predicted_atom_pos[original_atom_position_idxs] = (
                predicted_atom_pos[swap_atom_position_idxs]
            )
            permutations_predicted_atom_pos[swap_atom_position_idxs] = (
                predicted_atom_pos[original_atom_position_idxs]
            )

        # Caluclate the lddt between the positions of the actual and predicted 
        # asym id.
        lddt = lddt_calc_module.compute_lddt(
            true_coords=true_atom_pos.unsqueeze(0),
            pred_coords=permutations_predicted_atom_pos.unsqueeze(0),
            is_dna=is_molecule_types[..., IS_DNA_INDEX].unsqueeze(0),
            is_rna=is_molecule_types[..., IS_RNA_INDEX].unsqueeze(0),
            pairwise_mask=(torch.ones(len(true_atom_pos), len(true_atom_pos)) * True).unsqueeze(0),
        )
        if lddt.item() > max_lddt:
            best_predicted_atom_pos = permutations_predicted_atom_pos
            best_permutation = entity_group_asym_id_permutation
            max_lddt = lddt.item()

    return best_predicted_atom_pos, best_permutation, max_lddt


def sa_search_for_optimal_asym_id_assignments(
    entity_group_asym_ids: Tuple[int],
    atom_to_asym_id_map,
    is_molecule_types,
    predicted_atom_pos,    
    true_atom_pos,
    starting_temp: float = 0.25,
    cooling_rate: float = 0.99,
    num_steps: int = 50,
):
    """ Simulated annealing search for identical asym_id swaps that maximize lddt. """

    temp: float = starting_temp
    iter_num: int = 0


    cstate: Tuple[int] = entity_group_asym_ids
    cstate_atom_pos = torch.clone(predicted_atom_pos)
    
    
    cstate_lddt = lddt_calc_module.compute_lddt(
        true_coords=true_atom_pos.unsqueeze(0),
        pred_coords=predicted_atom_pos.unsqueeze(0),
        is_dna=is_molecule_types[..., IS_DNA_INDEX].unsqueeze(0),
        is_rna=is_molecule_types[..., IS_RNA_INDEX].unsqueeze(0),
        pairwise_mask=(torch.ones(len(true_atom_pos), len(true_atom_pos)) * True).unsqueeze(0),
    )

    while iter_num < num_steps:
        iter_num += 1

        neighboring_assignment_predicted_atom_pos = torch.clone(predicted_atom_pos)
        nstate = list(cstate)

        # Generate a 'neighboring' state by swapping two asym_ids.
        swap_idxs = random.sample(range(0, len(cstate)), 2)
        nstate[swap_idxs[0]] = cstate[swap_idxs[1]]
        nstate[swap_idxs[1]] = cstate[swap_idxs[0]]


        # Create a new predicted atom pos tensor by performing the asym id pos swaps.
        for idx, swap_asym_id in enumerate(nstate):

            # The asym id to be swapped with.
            base_asym_id = entity_group_asym_ids[idx]

            # Skip self-swaps.
            if base_asym_id == swap_asym_id:
                continue

            # Isolate the atomic idxs to swap values for.
            original_atom_position_idxs = atom_to_asym_id_map == base_asym_id
            swap_atom_position_idxs = atom_to_asym_id_map == swap_asym_id

            # Ensure the number of atoms composing the two asym ids are identical. If 
            # they're not, something went wrong.
            if original_atom_position_idxs.sum() != swap_atom_position_idxs.sum():
                raise ValueError(
                    "Something went wrong, when trying to swap asym_ids to find the "
                    "closest matching assignment to the ground truth, the num atoms "
                    "beloinging to both asym ids do not match. Thus they must not be "
                    f"identical. Found shapes [{original_atom_position_idxs.size(0)}] "
                    f"and [{swap_atom_position_idxs.size(0)}]."
                )

            # Perform the swap on a copy of the predictions.
            neighboring_assignment_predicted_atom_pos[original_atom_position_idxs] = (
                predicted_atom_pos[swap_atom_position_idxs]
            )
            neighboring_assignment_predicted_atom_pos[swap_atom_position_idxs] = (
                predicted_atom_pos[original_atom_position_idxs]
            )

        # Caluclate the lddt between the positions of the actual and the new
        # neighboring swap assignment.
        nstate_lddt = lddt_calc_module.compute_lddt(
            # We don't need different masks for the same asym id.
            true_coords=true_atom_pos.unsqueeze(0),
            pred_coords=neighboring_assignment_predicted_atom_pos.unsqueeze(0),
            is_dna= is_molecule_types[..., IS_DNA_INDEX].unsqueeze(0),
            is_rna=is_molecule_types[..., IS_RNA_INDEX].unsqueeze(0),
            pairwise_mask=(torch.ones(len(true_atom_pos), len(true_atom_pos)) * True).unsqueeze(0),
        )

        # Probabilistically accept a new state.
        acceptance_threshold = (math.e)**((nstate_lddt - cstate_lddt) / temp)
        if random.random() < acceptance_threshold:
            cstate = nstate
            cstate_lddt = nstate_lddt

        # Update the temperature.
        temp = temp * cooling_rate**(iter_num)

    return neighboring_assignment_predicted_atom_pos, cstate, cstate_lddt 


# Analogs of methods from lightining/hydra code so eval can be run outside a lightning context.

def prepare_batch_dict(af3: Alphafold3, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the input batch dictionary for the model.

    :param batch_dict: The input batch dictionary.
    :return: The prepared batch dictionary.
    """
    if not af3.has_molecule_mod_embeds:
        batch_dict["is_molecule_mod"] = None
    return batch_dict


# Core benchmarking script.
if __name__ == "__main__":

    # Define the command line interface.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-model-weights",
        dest="inference_model_weights",
        type=Path,
        help=(
            "File path containing alphafold3 model weights to be used in the predictions "
            "that will be evaluated."
        ),
    )
    parser.add_argument(
        "--pdb-dataset",
        dest="pdb_dataset",
        type=Path,
        help="A path to a directory containing mmcif files to evaluate",
    )
    parser.add_argument(
        "--inference-batch-size",
        dest="inference_batch_size",
        default=2,
        required=False,
        type=int,
        help="The size of batches to use for inference.",
    )
    parser.add_argument(
        "--inference-num-dataloading-workers",
        dest="inference_num_dataloading_workers",
        default=int(mp.cpu_count() * .5),
        type=int,
        help="The number of worker processes to use for dataloading.",
    )
    parser.add_argument(
        "--inference-atoms-per-window",
        dest="inference_atoms_per_window",
        type=Optional[int],
        default=None,
        required=False,
        help="The window size to use (None) if no windowing for atom pair tensor reprs.",
    )


    # Parse the user-provided args.
    args = parser.parse_args()

    # Load the model-weights into an Alphafold3 obj.
    alphafold3 = Alphafold3.init_and_load(path=args.inference_model_weights)
    alphafold3.to(DEVICE)
    alphafold3.eval()

    # Validate the alphafold3 dimensions meet the script's expectations.
    if alphafold3.dim_atom_inputs != EXPECTED_DIM_ATOM_INPUT:
        raise ValueError(
            f"The alphafold3 model loaded from [{args.inference_model_weights}] uses a "
            "atom input shape 'dim_atom_inputs' [{alphafold3.dim_atom_inputs}] that "
            "differs from what this script expects [{EXPECTED_DIM_ATOM_INPUT}]"
        )

    # Load the pdb file dataset into a reference obj.
    pdb_dataset = PDBDataset(folder=args.pdb_dataset)
    atom_dataset = pdb_dataset_to_atom_inputs(
        pdb_dataset=pdb_dataset,
        return_atom_dataset=True,
    )

    dataloader = AF3DataLoader(
        dataset=atom_dataset,
        batch_size=args.inference_batch_size,
        num_workers=args.inference_num_dataloading_workers,
        atoms_per_window=args.inference_atoms_per_window,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
        int_pad_value=INT_PAD_VALUE,
    )
    
    for batched_atom_input in dataloader:

        batch_dict = prepare_batch_dict(
            af3=alphafold3,
            batch_dict=batched_atom_input.dict()
        )
        batch_dict = dict_to_device(batch_dict, device=alphafold3.device) 

        with torch.no_grad():
            # Perform inference.
            batch_sampled_atom_pos, logits = alphafold3(
                **batch_dict,
                return_loss=False,
                return_confidence_head_logits=True,
                return_distogram_head_logits=True,
            )

        # Note: A for loop here because complexs will have irregular numbers and types of 
        # enities, meaning descriptive tensors would have irregular dimensional shapes. 
        for batch_idx in range(batched_atom_input.atom_inputs.size(0)):

            # Pull out tensorized mappings between tensor indices and parent entities
            # for a single sample.
            padded_atom_lens_per_token_idx = batched_atom_input.molecule_atom_lens[batch_idx]
            padded_token_idx_to_is_molecule_types = batched_atom_input.is_molecule_types[batch_idx]
            _, _, padded_token_idx_to_asym_ids, padded_token_idx_to_entity_id, _= (
                batched_atom_input.additional_molecule_feats[batch_idx].unbind(dim=-1)
            )

            # Find the unpadded token indices of this sample.
            sample_unpadded_token_idxs = padded_token_idx_to_asym_ids != INT_PAD_VALUE

            # Remove the padding from all the descriptive batched tensors.
            token_idx_to_asym_ids = padded_token_idx_to_asym_ids[sample_unpadded_token_idxs]
            token_idx_to_entity_ids = padded_token_idx_to_entity_id[sample_unpadded_token_idxs]
            atom_lens_per_token_idx = padded_atom_lens_per_token_idx[sample_unpadded_token_idxs]
            token_idx_to_is_molecule_types = padded_token_idx_to_is_molecule_types[
                sample_unpadded_token_idxs
            ]

            # Group all the asym_ids that are "identical" together at the token level.
            identical_entity_asym_id_groups = group_identical_entity_asym_ids(
                token_idx_to_asym_ids_map=token_idx_to_asym_ids,
                token_idx_to_entity_ids_map=token_idx_to_entity_ids
            )

            # Find the unpadded atomic indices of this sample.
            sample_unpadded_atom_idxs = torch.arange(0, atom_lens_per_token_idx.sum()).long()

            # Construct unpadded atom level mappings to various atomic attributes.
            atom_to_asym_id_map = token_idx_to_asym_ids.repeat_interleave(
                atom_lens_per_token_idx,
            )
            atom_to_is_molecule_types = token_idx_to_is_molecule_types.repeat_interleave(
                atom_lens_per_token_idx,
                dim=0,
            )

            # For each group of matching asym_ids, swap their positions to find the
            # arrangement that most closely matches the obersrved arrangement.
            for entity_group_asym_ids in identical_entity_asym_id_groups:

                if len(entity_group_asym_ids) <= 1:
                    # No swapping arrangements to search through.
                    continue

                elif len(entity_group_asym_ids) <= MAX_NUM_IDENTICAL_ENTITIES_FOR_EXHAUSTIVE_SEARCH:
                    # Few enough matching entities to do an exhaustive search.
                    best_predicted_atom_pos, best_permutation, max_lddt = (
                        exhaustive_search_for_optimal_asym_id_assignments(
                            entity_group_asym_ids=entity_group_asym_ids,
                            atom_to_asym_id_map=atom_to_asym_id_map,
                            is_molecule_types=atom_to_is_molecule_types,
                            predicted_atom_pos=batch_sampled_atom_pos[batch_idx][sample_unpadded_atom_idxs, ...],
                            true_atom_pos=batched_atom_input.atom_pos[batch_idx][sample_unpadded_atom_idxs, ...],
                        )
                    )

                else:
                    best_predicted_atom_pos, best_permutation, max_lddt = (
                        sa_search_for_optimal_asym_id_assignments(
                            entity_group_asym_ids=entity_group_asym_ids,
                            atom_to_asym_id_map=atom_to_asym_id_map,
                            is_molecule_types=atom_to_is_molecule_types,
                            predicted_atom_pos=batch_sampled_atom_pos[batch_idx][sample_unpadded_atom_idxs, ...],
                            true_atom_pos=batched_atom_input.atom_pos[batch_idx][sample_unpadded_atom_idxs, ...],
                        )
                    )

                # Update the token and atom mappings to use the new asym id assignments.

                # Create copies so the partial updates don't remove information needed
                # to finish the mapping.
                atom_to_asym_id_map_copy = torch.clone(atom_to_asym_id_map)
                token_idx_to_asym_ids_copy = torch.clone(token_idx_to_asym_ids)
                token_idx_to_entity_ids_copy = torch.clone(token_idx_to_entity_ids)
                atom_lens_per_token_idx_copy = torch.clone(atom_lens_per_token_idx)
                token_idx_to_is_molecule_types_copy = torch.clone(token_idx_to_is_molecule_types)
                atom_to_is_molecule_types_copy = torch.clone(atom_to_is_molecule_types)

                for t, new_asym_id in enumerate(best_permutation):
                    old_asym_id = entity_group_asym_ids[t]

                    is_old_asym_id_token = token_idx_to_asym_ids == old_asym_id
                    is_new_asym_id_token = token_idx_to_asym_ids == new_asym_id

                    is_old_asym_id_atom = atom_to_asym_id_map == old_asym_id
                    is_new_asym_id_atom = atom_to_asym_id_map == new_asym_id

                    # Perform the old asym id to new asym id swaps in all supporting tensor copies.
                    atom_to_asym_id_map_copy[is_old_asym_id_atom] = new_asym_id
                    token_idx_to_asym_ids_copy[is_old_asym_id_token] = new_asym_id
                    token_idx_to_is_molecule_types_copy[is_old_asym_id_token, ...] = token_idx_to_is_molecule_types[
                        is_new_asym_id_token
                    ]
                    atom_to_is_molecule_types_copy[is_old_asym_id_atom, ...] = atom_to_is_molecule_types[
                        is_new_asym_id_atom
                    ]
                    atom_lens_per_token_idx_copy[is_old_asym_id_token] = atom_lens_per_token_idx[
                        is_new_asym_id_token
                    ]
                    token_idx_to_entity_ids_copy[is_old_asym_id_token] = token_idx_to_entity_ids[
                        is_new_asym_id_token
                    ]
            
                # Reassign the tensors to the copies that have been successfully swapped.
                atom_to_asym_id_map = atom_to_asym_id_map_copy
                token_idx_to_asym_ids = token_idx_to_asym_ids_copy
                token_idx_to_entity_ids = token_idx_to_entity_ids_copy
                atom_lens_per_token_ids = atom_lens_per_token_idx_copy
                token_idx_to_is_molecule_types_copy = token_idx_to_is_molecule_types
                atom_to_is_molecule_types_copy = atom_to_is_molecule_types

            # Ligand symmetry resolution logic?

            # Compute evaluation metrics.



        
