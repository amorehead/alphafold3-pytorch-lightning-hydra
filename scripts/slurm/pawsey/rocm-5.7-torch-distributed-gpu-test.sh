#!/bin/bash

######################### Batch Headers #########################
#SBATCH --partition=gpu                                       # use partition `gpu` for GPU nodes
#SBATCH --account=pawsey1018-gpu                              # IMPORTANT: use your own project and the -gpu suffix
#SBATCH --nodes=2                                             # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --ntasks-per-node=1                                   # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --time 0-00:05:00                                     # time limit for the job (up to 24 hours: `0-24:00:00`)
#SBATCH --job-name=torch-distributed-gpu-test                 # job name
#SBATCH --output=J-%x.%j.out                                  # output log file
#SBATCH --error=J-%x.%j.err                                   # error log file
#SBATCH --exclusive                                           # request exclusive node access
#################################################################

# Load required modules
module load pytorch/2.2.0-rocm5.7.3
module load pawseyenv/2023.08
module load singularity/3.11.4-nohost

# Define the container image path
export SINGULARITY_CONTAINER="/scratch/pawsey1018/$USER/af3-pytorch-lightning-hydra/af3-pytorch-lightning-hydra_0.5.5_dev.sif"

# Configure torch.distributed
NUM_PYTORCH_PROCESSES=8
RDZV_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
RDZV_PORT=29400

# Run Singularity container
srun -c 64 --jobid "$SLURM_JOBID" singularity exec \
    --cleanenv \
    -H "$PWD":/home \
    -B alphafold3-pytorch-lightning-hydra:/alphafold3-pytorch-lightning-hydra \
    --pwd /alphafold3-pytorch-lightning-hydra \
    "$SINGULARITY_CONTAINER" \
    bash -c "
        torchrun \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $NUM_PYTORCH_PROCESSES \
        --rdzv_id=$RANDOM \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
        scripts/slurm/torch-distributed-gpu-test.py
    "
