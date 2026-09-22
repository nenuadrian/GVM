#!/bin/bash
#SBATCH --job-name=microgvm
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

# Estimator comparison on the tiny CoT task. CPU only -- the model is ~400K
# parameters, and the cost is dominated by per-rollout backward passes, which
# parallelise across threads.
#
#   sbatch microgvm/run_cpu.sbatch                      # one config
#   sbatch --array=0-4 microgvm/run_cpu.sbatch          # sweep accept-rate spread
#
# The array sweeps --warmup-steps, which is what sets the accept-rate spread.
# That sweep is the question no LM experiment can answer cheaply: at what p
# spread does any allocator beat uniform?

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

WARMUPS=(100 200 400 800 1600)
i="${SLURM_ARRAY_TASK_ID:-2}"
W="${GVM_WARMUP:-${WARMUPS[$i]}}"
TAG="warmup${W}"

echo "=== microgvm | warmup=$W | threads=$OMP_NUM_THREADS | $(hostname) ==="
python run.py \
    --mode estimator \
    --modulus "${GVM_MODULUS:-7}" \
    --max-k "${GVM_MAX_K:-4}" \
    --n-prompts 512 \
    --batch-prompts "${GVM_BATCH_PROMPTS:-32}" \
    --n-base "${GVM_N_BASE:-8}" \
    --pilot "${GVM_PILOT:-4}" \
    --gt-rollouts "${GVM_GT_ROLLOUTS:-256}" \
    --repeats "${GVM_REPEATS:-64}" \
    --warmup-steps "$W" \
    --d 128 --n-layer 2 \
    --seed "${GVM_SEED:-0}" \
    --wandb-project "${GVM_WANDB_PROJECT:-microgvm}" \
    --run "$TAG-seed${GVM_SEED:-0}" \
    --out "results/${TAG}_seed${GVM_SEED:-0}.json"

echo "MICROGVM_OK $TAG"
