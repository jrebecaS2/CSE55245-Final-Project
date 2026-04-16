#!/bin/bash
#SBATCH --job-name=sft-train
#SBATCH --account=PAS3272           # Replace with your OSC project code (e.g., PAS1234)
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/sft_train_%j.out
#SBATCH --error=logs/sft_train_%j.err

# ─── Environment setup ───
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/${SLURM_JOB_ACCOUNT}/${USER}/triton_cache
export HF_HOME=/fs/scratch/PAS3272/jrskohler/.cache/huggingface
export TRANSFORMERS_CACHE=/fs/scratch/PAS3272/jrskohler/.cache/huggingface

# Use scratch for HF cache so downloads don't fill home dir
export HF_HOME=/fs/scratch/${SLURM_JOB_ACCOUNT}/${USER}/.cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/transformers

# ─── Activate your environment ───
source /fs/scratch/PAS3272/jrskohler/.venv/bin/activate

# ─── Create logs dir if needed ───
mkdir -p logs

# ─── Navigate to project root ───
cd $SLURM_SUBMIT_DIR/..

# ─── Run training ───
# Pass --sanity for a quick 100-step test run
python scripts/run_sft.py "$@"
