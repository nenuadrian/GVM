#!/bin/bash
#SBATCH --job-name=raft-env
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Builds the pinned 2023-era stack LMFlow's RAFT aligner needs.
#
# The ceiling is transformers 4.34.x, and it is not a preference. LMFlow's
# raft_trainer.py is a fork of the transformers 4.28 Trainer and imports
# ShardedDDPOption from transformers.trainer_utils and is_fairscale_available
# from transformers.integrations. Both were deleted in 4.35.0, so the module
# stops importing -- which is why LMFlow's own auto_pipeline.py registers the
# raft_aligner pipeline only when transformers is older than 4.35.
#
# Everything else follows from that: accelerate/datasets/tokenizers of the same
# era, and numpy<2 because torch 2.1 predates the numpy 2 ABI break. torch
# 2.1.2+cu121 is the floor that still ships sm_90 kernels for the H200s.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/raft/env.sh"
source "$GVM_ROOT/runs/csf3/raft/config.sh"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$RAFT_CONDA_MODULE" 2>/dev/null || module load miniforge/24.11.2
eval "$(conda shell.bash hook)"

if [[ ! -x "$RAFT_CONDA_ENV/bin/python" ]]; then
    conda create -y -p "$RAFT_CONDA_ENV" python=3.10
fi
conda activate "$RAFT_CONDA_ENV"
python -V

pip install --upgrade "pip<25" setuptools wheel packaging ninja

# Pinned in one resolve so pip cannot quietly upgrade transformers past 4.34.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2 torchvision==0.16.2

pip install \
    "transformers==4.34.1" \
    "tokenizers==0.14.1" \
    "huggingface_hub==0.19.4" \
    "accelerate==0.24.1" \
    "datasets==2.14.6" \
    "peft==0.6.2" \
    "deepspeed==0.12.6" \
    "numpy==1.26.4" \
    "scipy==1.11.4" \
    "sentencepiece==0.1.99" \
    "protobuf==3.20.3" \
    "wandb==0.16.6" \
    "matplotlib==3.8.4" \
    "einops>=0.6.1" \
    "evaluate==0.4.0" \
    "scikit-learn==1.3.2"

# Best effort: lets deepspeed JIT-build any CUDA op it decides it wants. Not
# fatal -- ZeRO-2 with no offload and the client (torch) optimizer does not need
# nvcc, and the smoke test will say so either way.
conda install -y -p "$RAFT_CONDA_ENV" -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1 \
    || echo "WARN: no conda nvcc; deepspeed CUDA-op JIT will be unavailable"

# LMFlow v0.0.6 (Dec 2023). Not HEAD: on HEAD, AutoModel.get_model with
# arch_type=text_regression returns HFTextRegressionModel, whose own inference()
# shadows the reward function raft_align.py registers via
# register_inference_function -- so the reward model is silently ignored.
# v0.0.6 still returns the plain TextRegressionModel, where registration works.
if [[ ! -d "$LMFLOW_ROOT/.git" ]]; then
    git clone --branch v0.0.6 --depth 1 https://github.com/OptimalScale/LMFlow.git "$LMFLOW_ROOT"
fi
cd "$LMFLOW_ROOT"
echo "LMFlow at: $(git log -1 --format='%H %ad' --date=short) $(git describe --tags --always)"

# --no-deps: setup.py would re-resolve the whole 2023 pin set, including a
# transformers built from a git SHA.
pip install -e . --no-deps

python "$RAFT_S/patches/apply_patches.py" "$LMFLOW_ROOT"

echo "=== versions ==="
python - <<'PY'
import importlib
for m in ["torch","transformers","tokenizers","huggingface_hub","accelerate",
          "datasets","deepspeed","peft","numpy","wandb","lmflow"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:18s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:18s} FAILED: {type(e).__name__}: {e}")
PY

echo "=== the import that decides whether any of this works ==="
python - <<'PY'
# If transformers is >= 4.35 this raises, and LMFlow would have silently dropped
# raft_aligner from PIPELINE_MAPPING instead.
from lmflow.pipeline.auto_pipeline import AutoPipeline, PIPELINE_MAPPING
assert "raft_aligner" in PIPELINE_MAPPING, (
    "raft_aligner is not registered -- transformers is too new. "
    f"registered: {sorted(PIPELINE_MAPPING)}"
)
from lmflow.pipeline.raft_aligner import RaftAligner
print("raft_aligner registered and importable:", RaftAligner)
PY
echo "BUILD_OK"
