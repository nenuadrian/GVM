# Shared environment for the LMFlow RAFT replication on CSF3 (Manchester).
# Sourced by every sbatch script in runs/csf3/raft/.
#
# Deliberately a separate conda env from gvm25: LMFlow's RAFT aligner only
# imports under transformers < 4.35 (see README), and gvm25 is on a modern
# transformers for verl. The two stacks cannot share an environment.

export GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
export LMFLOW_ROOT="${LMFLOW_ROOT:-$HOME/scratch/LMFlow}"
export RAFT_S="${RAFT_S:-$GVM_ROOT/runs/csf3/raft}"

# Where SFT / RAFT checkpoints and generated data land. Not in $LMFLOW_ROOT, so
# the checkout stays a clean v0.0.6 tree that can be deleted and re-cloned.
export RAFT_OUT="${RAFT_OUT:-$HOME/scratch/raft_lmflow}"

export RAFT_CONDA_ENV="${RAFT_CONDA_ENV:-$HOME/.conda/envs/raftlm}"
export RAFT_CONDA_MODULE="${RAFT_CONDA_MODULE:-apps/binapps/conda/miniforge3/25.3.0_python3.10}"

# Caches on scratch: home is NAS-backed and slow for many small files.
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export XDG_CACHE_HOME="$HOME/scratch/.cache"
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
export TRITON_CACHE_DIR="$HOME/scratch/.cache/triton"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$RAFT_OUT"

export TOKENIZERS_PARALLELISM=false
# Without this, python buffers stdout when piped into tee, so a hang looks
# identical to a silent process and you learn nothing from the log.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN

# raft_aligner.py imports matplotlib and writes training_reward.png on rank 0.
# Compute nodes have no display; without this the import raises.
export MPLBACKEND=Agg

# Weights & Biases. Key comes from ~/.netrc (machine api.wandb.ai).
export WANDB_PROJECT="${WANDB_PROJECT:-raft-lmflow-hh}"
export WANDB_DIR="${WANDB_DIR:-$RAFT_OUT/wandb}"
export WANDB_CACHE_DIR="$HOME/scratch/.cache/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

raft_activate() {
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load "$RAFT_CONDA_MODULE" 2>/dev/null || module load miniforge/24.11.2
    eval "$(conda shell.bash hook)"
    conda activate "$RAFT_CONDA_ENV"
}
