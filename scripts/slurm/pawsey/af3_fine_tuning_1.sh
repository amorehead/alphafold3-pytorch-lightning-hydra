#!/bin/bash

######################### Batch Headers #########################
#SBATCH --partition=gpu                                       # use partition `gpu` for GPU nodes
#SBATCH --account=pawsey1018-gpu                              # IMPORTANT: use your own project and the -gpu suffix
#SBATCH --nodes=32                                            # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --ntasks-per-node=1                                   # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --time 0-24:00:00                                     # time limit for the job (up to 24 hours: `0-24:00:00`)
#SBATCH --job-name=af3_fine_tuning_1                          # job name
#SBATCH --output=J-%x.%j.out                                  # output log file
#SBATCH --error=J-%x.%j.err                                   # error log file
#SBATCH --exclusive                                           # request exclusive node access
#################################################################

# Load required modules
module load pawseyenv/2024.05
module load singularity/4.1.0-slurm

# Establish environment variables
TARGET_BATCH_SIZE=256

# Define the container image path
export SINGULARITY_CONTAINER="/scratch/pawsey1018/$USER/af3-pytorch-lightning-hydra/af3-pytorch-lightning-hydra_0.5.25_dev.sif"

# Set number of PyTorch (GPU) processes per node to be spawned by torchrun - NOTE: One for each GCD
NUM_PYTORCH_PROCESSES=8
# Set the number of threads to be generated for each PyTorch (GPU) process
export OMP_NUM_THREADS=8

# Define the compute node executing the batch script
RDZV_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export RDZV_HOST
export RDZV_PORT=29400

echo "Rendezvous Node IP: $RDZV_HOST"

# NOTE: The following `srun` command gives all the available resources to
# `torchrun` which will then distribute them internally to the processes
# it creates. Importantly, notice that processes are NOT created by srun!
# For what `srun` is concerned, only one task is created, the `torchrun` process.

# Define WandB run ID
RUN_ID="xausbbkt" # NOTE: Generate a unique ID for each run using `python3 scripts/generate_id.py`

# Run Singularity container
srun -c 64 singularity exec \
    --cleanenv \
    -H "$PWD":/home \
    -B alphafold3-pytorch-lightning-hydra:/alphafold3-pytorch-lightning-hydra \
    --pwd /alphafold3-pytorch-lightning-hydra \
    "$SINGULARITY_CONTAINER" \
    bash -c "
        /usr/bin/kalign --version \
        && WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID OMP_NUM_THREADS=$OMP_NUM_THREADS \
        torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$NUM_PYTORCH_PROCESSES \
        --rdzv_id=$RANDOM \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
        alphafold3_pytorch/train.py \
        data.batch_size=$((SLURM_JOB_NUM_NODES * NUM_PYTORCH_PROCESSES)) \
        data.kalign_binary_path=/usr/bin/kalign \
        experiment=af3_fine_tuning_1 \
        trainer.accumulate_grad_batches=$((TARGET_BATCH_SIZE / (SLURM_JOB_NUM_NODES * NUM_PYTORCH_PROCESSES))) \
        trainer.ckpt_path=logs/train/runs/stage_1/checkpoints/last.ckpt \
        trainer.num_nodes=$SLURM_JOB_NUM_NODES \
        trainer.devices=$NUM_PYTORCH_PROCESSES
    "

# Inform user of run completion
echo "Run completed for job: $SLURM_JOB_NAME"
