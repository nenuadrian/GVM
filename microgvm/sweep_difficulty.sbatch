#!/bin/bash
#SBATCH --job-name=mgvm-diff
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# Accept rate swept via TASK DIFFICULTY. The earlier array swept --warmup-steps
# and barely moved it (p_mean 0.204 -> 0.266), because a 400K model does not
# learn mod-7 chains in 1600 steps. Modulus and chain length move it directly:
# random guessing alone is 1/modulus.
#
#   sbatch --array=0-23 microgvm/sweep_difficulty.sbatch
#
# 6 difficulty configs x 4 seeds. Each run takes ~3-6 min on 16 cores, so the
# whole grid finishes in one scheduling round.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
cd "$GVM_ROOT/microgvm"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load apps/binapps/conda/miniforge3/25.3.0_python3.10 2>/dev/null || module load miniforge/24.11.2
eval "$(conda shell.bash hook)"
conda activate "${GVM_CONDA_ENV:-$HOME/.conda/envs/gvm25}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export WANDB_DIR="$GVM_ROOT/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/.cache/wandb"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" results

# modulus:max_k:warmup, easiest first
CONFIGS=(5:2:1500 5:3:1500 7:2:1500 7:3:3000 7:4:1500 11:4:1500)
SEEDS=(0 1 2 3)

i="${SLURM_ARRAY_TASK_ID:-0}"
cfg="${CONFIGS[$(( i % ${#CONFIGS[@]} ))]}"
seed="${SEEDS[$(( i / ${#CONFIGS[@]} ))]}"
IFS=: read -r MOD MK W <<< "$cfg"
TAG="m${MOD}k${MK}"

echo "=== $TAG seed=$seed (modulus=$MOD max_k=$MK warmup=$W) | $(hostname) ==="
python run.py --mode estimator \
    --modulus "$MOD" --max-k "$MK" --n-prompts 512 \
    --batch-prompts 32 --n-base 8 --pilot 4 \
    --gt-rollouts 512 --repeats 128 \
    --warmup-steps "$W" --d 128 --n-layer 2 --seed "$seed" \
    --wandb-project "${GVM_WANDB_PROJECT:-microgvm}" \
    --run "diff-${TAG}-seed${seed}" \
    --out "results/diff_${TAG}_seed${seed}.json"

echo "MICROGVM_OK $TAG seed$seed"
