# Shared environment for GVM replication on CSF3 (Manchester).
# Sourced by every sbatch script in runs/csf3/.

export GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
export GVM_CONDA_ENV="${GVM_CONDA_ENV:-$HOME/.conda/envs/gvm25}"
export GVM_CONDA_MODULE="${GVM_CONDA_MODULE:-apps/binapps/conda/miniforge3/25.3.0_python3.10}"

# Caches on scratch: home is NAS-backed and slow for many small files.
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TRITON_CACHE_DIR="$HOME/scratch/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$HOME/scratch/.cache/inductor"
export XDG_CACHE_HOME="$HOME/scratch/.cache"
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PIP_CACHE_DIR"

# vLLM 0.8.3 + verl: V0 engine is what this fork was pinned against.
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
# Single node, no InfiniBand needed between GPUs on one box.
export NCCL_P2P_DISABLE=0

# Ray's plasma store is an AF_UNIX socket, and the kernel caps that path at 107
# bytes. Ray appends ~68 bytes of its own (/ray/session_<timestamp>_<pid>/sockets/
# plasma_store), so RAY_TMPDIR gets ~39. The scratch path alone is 48, hence
# node-local /tmp -- which is where these sockets belong on a single-node job.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID:-$$}}"
mkdir -p "$RAY_TMPDIR"
if (( ${#RAY_TMPDIR} > 39 )); then
    echo "RAY_TMPDIR is ${#RAY_TMPDIR} chars ('$RAY_TMPDIR'); >39 overflows the" \
         "107-byte AF_UNIX limit once Ray appends its session path." >&2
    exit 1
fi

# Weights & Biases. Key comes from ~/.netrc (machine api.wandb.ai).
export WANDB_PROJECT="${WANDB_PROJECT:-gvm-replication}"
export WANDB_DIR="${WANDB_DIR:-$GVM_ROOT/wandb}"
export WANDB_CACHE_DIR="$HOME/scratch/.cache/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

gvm_activate() {
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load "$GVM_CONDA_MODULE" 2>/dev/null || module load miniforge/24.11.2
    # `conda activate` needs the shell hook; `conda run` is too noisy for long jobs.
    eval "$(conda shell.bash hook)"
    conda activate "$GVM_CONDA_ENV"
}
