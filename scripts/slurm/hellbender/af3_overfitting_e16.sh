#!/bin/bash -l

######################### Batch Headers #########################
#SBATCH --partition=chengji-lab-gpu                           # use reserved partition `chengji-lab-gpu`
#SBATCH --account=chengji-lab                                 # NOTE: this must be specified to use the reserved partition above
#SBATCH --nodes=2                                             # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gres=gpu:4                                          # e.g., request A/H100 GPU resource(s)
#SBATCH --ntasks-per-node=4                                   # NOTE: this needs to be `1` on SLURM clusters when using Lightning's `ddp_spawn` strategy`; otherwise, set to match Lightning's quantity of `Trainer(devices=...)`
#SBATCH --mem=0                                               # NOTE: use `--mem=0` to request all memory "available" on the assigned node
#SBATCH -t 0-00:20:00                                         # time limit for the job (up to two days: `2-00:00:00`)
#SBATCH -J af3_overfitting_e16                                # job name
#SBATCH --output=J-%x.%j.out                                  # output log file
#SBATCH --error=J-%x.%j.err                                   # error log file
#SBATCH --exclusive                                           # request exclusive node access

# Load required modules
module purge
module load cuda/11.8.0_gcc_9.5.0

# Determine location of the project's directory
PROJECT_DIR="/cluster/pixstor/chengji-lab/$USER/Repositories/Lab_Repositories/alphafold3-pytorch-lightning-hydra"
cd "$PROJECT_DIR" || exit

# Activate the project's Conda environment
# shellcheck source=/dev/null
source "/home/$USER/mambaforge/etc/profile.d/conda.sh"
conda activate alphafold3-pytorch/

# Establish environment variables
export TORCH_HOME="/cluster/pixstor/chengji-lab/$USER/torch_cache"
export HF_HOME="/cluster/pixstor/chengji-lab/$USER/hf_cache"

mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"

# Define WandB run ID
RUN_ID="1vahdhf5" # NOTE: Generate a unique ID for each run using `python3 scripts/generate_id.py`

# Run script
bash -c "
    $CONDA_PREFIX/bin/kalign \
    && WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID \
    TORCH_HOME=$TORCH_HOME \
    HF_HOME=$HF_HOME \
    srun python3 alphafold3_pytorch/train.py \
    data.batch_size=$((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE)) \
    data.kalign_binary_path=$CONDA_PREFIX/bin/kalign \
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
    trainer.devices=$SLURM_NTASKS_PER_NODE
"

# Inform user of run completion
echo "Run completed for job: $SLURM_JOB_NAME"
