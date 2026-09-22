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

# setuptools is pinned below 81, which is where pkg_resources was removed.
# torch 2.1.2's torch/utils/cpp_extension.py opens with
# `from pkg_resources import packaging`, so on a current setuptools every
# deepspeed build and every JIT op compile dies with ModuleNotFoundError.
# ~/.local/lib/python3.10/site-packages holds 157 packages from other projects,
# including a torch. User site is on sys.path even inside a conda env, so
# without this a stale copy there can shadow a pin resolved here.
export PYTHONNOUSERSITE=1

pip install --upgrade "pip<25" "setuptools==69.5.1" wheel packaging ninja

# numpy before torch, and hard-pinned: torch 2.1.2 is built against the numpy 1.x
# C ABI and greets numpy 2 with "Failed to initialize NumPy: _ARRAY_API not
# found", after which every tensor<->array conversion fails at runtime.
# torchvision is not installed at all -- nothing here uses it, and it is the only
# thing in this stack that would drag numpy back up.
pip install "numpy==1.26.4"

pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2

python - <<'NPCHK'
import numpy, torch
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__} breaks torch 2.1"
assert torch.from_numpy(numpy.zeros(3)).sum().item() == 0.0
print(f"OK torch {torch.__version__} / numpy {numpy.__version__}, cuda={torch.version.cuda}")
NPCHK

# Pinned in one resolve so pip cannot quietly upgrade transformers past 4.34.
# huggingface_hub is 0.17.3, not something newer: tokenizers 0.14.1 -- the only
# range transformers 4.34.1 accepts -- requires huggingface_hub<0.18.
pip install \
    "transformers==4.34.1" \
    "tokenizers==0.14.1" \
    "huggingface_hub==0.17.3" \
    "accelerate==0.24.1" \
    "datasets==2.14.6" \
    "pyarrow==14.0.2" \
    "pandas==2.1.4" \
    "peft==0.6.2" \
    "numpy==1.26.4" \
    "scipy==1.11.4" \
    "sentencepiece==0.1.99" \
    "protobuf==3.20.3" \
    "wandb==0.16.6" \
    "matplotlib==3.8.4" \
    "einops>=0.6.1" \
    "evaluate==0.4.0" \
    "scikit-learn==1.3.2" \
    "pillow==10.3.0" \
    "bitsandbytes==0.43.3"
# pillow and bitsandbytes are not optional extras. lmflow/datasets/dataset.py
# imports multi_modal_dataset, which imports PIL, and
# lmflow/models/hf_decoder_model.py has a bare `import bitsandbytes` -- both run
# on the way to `from lmflow.pipeline.auto_pipeline import AutoPipeline`.
#
# pyarrow is held at 14: datasets 2.14.6 subclasses pa.PyExtensionType, which
# newer pyarrow removed, so `import datasets` dies on the current release.
#
# bitsandbytes is 0.43.3, not the era-matching 0.41.x, purely so the import does
# not raise. Neither version can find libcusparse.so.12 here (torch bundles
# cublas and cudart but not cusparse, and there is no CUDA toolkit module), but
# 0.41 treats that as fatal while 0.43 warns and carries on. Expect a loud
# CUDA SETUP banner in every log; nothing in this replication quantizes, so
# bitsandbytes only ever needs to be importable.

# Best effort: lets deepspeed JIT-build any CUDA op it decides it wants. Not
# fatal -- ZeRO-2 with no offload and the client (torch) optimizer does not need
# nvcc, and the smoke test will say so either way.
conda install -y -p "$RAFT_CONDA_ENV" -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1 \
    || echo "WARN: no conda nvcc; deepspeed CUDA-op JIT will be unavailable"

# deepspeed 0.12.6 is an sdist whose setup.py imports torch, so it must see the
# environment it is being installed into -- hence --no-build-isolation.
#
# This is required, not optional. RAFT_LAUNCHER=torchrun changes how ranks are
# spawned, but lmflow/pipeline/evaluator.py and inferencer.py both have a bare
# `import deepspeed` at module scope, and auto_pipeline imports them -- so
# without deepspeed installed, nothing in LMFlow imports at all.
pip install --no-build-isolation "deepspeed==0.12.6"

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
