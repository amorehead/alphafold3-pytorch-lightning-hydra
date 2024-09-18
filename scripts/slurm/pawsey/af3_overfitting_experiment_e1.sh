#!/bin/bash

######################### Batch Headers #########################
#SBATCH --partition=gpu                                       # use partition `gpu` for GPU nodes
#SBATCH --account=pawsey1018-gpu                              # IMPORTANT: use your own project and the -gpu suffix
#SBATCH --nodes=1                                             # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gres=gpu:1                                          # NOTE: requests any GPU resource(s)
#SBATCH --ntasks-per-node=1                                   # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --time 0-01:00:00                                     # time limit for the job (up to 24 hours: `0-24:00:00`)
#SBATCH --job-name=af3_overfitting_e1                         # job name
#SBATCH --output=J-%x.%j.out                                  # output log file
#SBATCH --error=J-%x.%j.err                                   # error log file
#################################################################

# Load required modules
module load pawseyenv/2024.05
module load singularity/4.1.0-slurm

# Define the container image path
export SINGULARITY_CONTAINER="/scratch/pawsey1018/$USER/af3-pytorch-lightning-hydra/af3-pytorch-lightning-hydra_0.5.6_source_dev.sif"

# Set the number of threads to be generated for each PyTorch (GPU) process
export OMP_NUM_THREADS=8

# Define WandB run ID
RUN_ID="qyj6mki3"  # NOTE: Generate a unique ID for each run using `python3 scripts/generate_id.py`

# Run Singularity container
srun singularity exec --rocm \
    --cleanenv \
    -H "$PWD":/home \
    -B alphafold3-pytorch-lightning-hydra:/alphafold3-pytorch-lightning-hydra \
    --pwd /alphafold3-pytorch-lightning-hydra \
    "$SINGULARITY_CONTAINER" \
    bash -c "
        /usr/bin/kalign --version \
        && WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID OMP_NUM_THREADS=$OMP_NUM_THREADS NCCL_DEBUG=INFO \
        python3 alphafold3_pytorch/train.py \
        data.batch_size=1 \
        data.kalign_binary_path=/usr/bin/kalign \
        experiment=af3_overfitting_e1 \
        trainer.num_nodes=1 \
        trainer.devices=1
    "

# Inform user of run completion
echo "Run completed for job: $SLURM_JOB_NAME"
