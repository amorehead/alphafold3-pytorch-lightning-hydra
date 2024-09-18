#!/bin/bash

######################### Batch Headers #########################
#SBATCH --partition=gpu                                       # use partition `gpu` for GPU nodes
#SBATCH --account=pawsey1018-gpu                              # IMPORTANT: use your own project and the -gpu suffix
#SBATCH --nodes=2                                             # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --ntasks-per-node=1                                   # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --time 0-00:20:00                                     # time limit for the job (up to 24 hours: `0-24:00:00`)
#SBATCH --job-name=af3_overfitting_e16                        # job name
#SBATCH --output=J-%x.%j.out                                  # output log file
#SBATCH --error=J-%x.%j.err                                   # error log file
#SBATCH --exclusive                                           # request exclusive node access
#################################################################

# Load required modules
module load pawseyenv/2024.05
module load singularity/4.1.0-slurm

# Define the container image path
export SINGULARITY_CONTAINER="/scratch/pawsey1018/$USER/af3-pytorch-lightning-hydra/af3-pytorch-lightning-hydra_0.5.6_source_dev.sif"

# Set number of PyTorch (GPU) processes per node to be spawned by torchrun - NOTE: One for each GCD
NUM_PYTORCH_PROCESSES=8
# Set the number of threads to be generated for each PyTorch (GPU) process
export OMP_NUM_THREADS=8

# Define the compute node executing the batch script
RDZV_HOST=$(hostname)
export RDZV_HOST
export RDZV_PORT=29400

# NOTE: The following `srun` command gives all the available resources to
# `torchrun` which will then distribute them internally to the processes
# it creates. Importantly, notice that processes are NOT created by srun!
# For what `srun` is concerned, only one task is created, the `torchrun` process.

# Define WandB run ID
RUN_ID="s8lofo23"  # NOTE: Generate a unique ID for each run using `python3 scripts/generate_id.py`

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
        NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_SHOW_CPP_STACKTRACES=1 \
        torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$NUM_PYTORCH_PROCESSES \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$RDZV_HOST:$RDZV_PORT \
        alphafold3_pytorch/train.py \
        data.batch_size=$((SLURM_JOB_NUM_NODES*NUM_PYTORCH_PROCESSES)) \
        data.kalign_binary_path=/usr/bin/kalign \
        environment=torch_elastic \
        experiment=af3_overfitting_e16 \
        model.net.diffusion_num_augmentations=4 \
        +model.net.dim_atom=8 \
        +model.net.dim_pairwise=8 \
        +model.net.dim_single=8 \
        +model.net.dim_token=8 \
        +model.net.confidence_head_kwargs='{pairformer_depth: 1}' \
        +model.net.template_embedder_kwargs='{pairformer_stack_depth: 1}' \
        +model.net.msa_module_kwargs='{depth: 1, dim_msa: 8}' \
        +model.net.pairformer_stack='{depth: 1, pair_bias_attn_dim_head: 4, pair_bias_attn_heads: 2}' \
        +model.net.diffusion_module_kwargs='{atom_encoder_depth: 1, token_transformer_depth: 1, atom_decoder_depth: 1, atom_encoder_kwargs: {attn_pair_bias_kwargs: {dim_head: 4}}, atom_decoder_kwargs: {attn_pair_bias_kwargs: {dim_head: 4}}}' \
        trainer.num_nodes=$SLURM_JOB_NUM_NODES \
        trainer.devices=$NUM_PYTORCH_PROCESSES
    "

# Inform user of run completion
echo "Run completed for job: $SLURM_JOB_NAME"
