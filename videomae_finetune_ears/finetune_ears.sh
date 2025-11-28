#!/usr/bin/env bash


#SBATCH --job-name=videomae_finetune_ears
#SBATCH --partition=prioritized
#SBATCH --nodelist=i256-a10-10
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=videomae_finetune_ears.out

# ============================================================================
# Setup paths and environment
# ============================================================================

cd /home/create.aau.dk/az66ep/read-my-ears/videomae_finetune_ears || exit 1

# ============================================================================
# W&B Configuration
# ============================================================================
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_PROJECT="${WANDB_PROJECT:-videomae_finetune_ears}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-videomae_finetune_ears_${SLURM_JOB_ID}}"
export WANDB_ENTITY="${WANDB_ENTITY:-joaomalves}"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# ============================================================================
# Python and Torch Configuration
# ============================================================================
export PYTHONPATH=/home/create.aau.dk/az66ep/videomae_finetune_ears:$PYTHONPATH
export TORCH_HOME=/home/create.aau.dk/az66ep/videomae_finetune_ears/.cache/torch

# Set NCCL environment variables for multi-GPU training
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFFSIZE=2097152
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Training Configuration
# ============================================================================
LOG_DIR="./logs"

mkdir -p "${LOG_DIR}"

# ============================================================================
# Create sweep first
# ============================================================================
echo "Creating W&B sweep..."
SWEEP_OUTPUT=$(singularity exec \
    --nv \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    pytorch.sif python3 create_sweep.py)

echo "$SWEEP_OUTPUT"

# Extract sweep ID from output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "^Sweep ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Could not extract sweep ID. Exiting."
    exit 1
fi

echo "Successfully created sweep: $SWEEP_ID"
export WANDB_SWEEP_ID=$SWEEP_ID

# ============================================================================
# Launch distributed training
# ============================================================================
srun --ntasks=4 --ntasks-per-node=4 bash -c " singularity exec \
    --nv \
    --env TORCH_HOME=$TORCH_HOME \
    --env PYTHONPATH=$PYTHONPATH \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_PROJECT=$WANDB_PROJECT \
    --env WANDB_RUN_NAME=$WANDB_RUN_NAME \
    --env WANDB_ENTITY=$WANDB_ENTITY \
    --env WANDB_SWEEP_ID=$WANDB_SWEEP_ID \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env NCCL_DEBUG=$NCCL_DEBUG \
    --env NCCL_IB_DISABLE=$NCCL_IB_DISABLE \
    --env NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
    --env NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL \
    --env NCCL_BUFFSIZE=$NCCL_BUFFSIZE \
    --env CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env PYTHONUNBUFFERED=$PYTHONUNBUFFERED \
    --env PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF \
    pytorch.sif torchrun \
        --nnodes=1 \
        --nproc-per-node=4 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py
"
