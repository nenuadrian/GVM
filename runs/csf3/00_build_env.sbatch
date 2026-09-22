#!/bin/bash
#SBATCH --job-name=gvm-env
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/env.sh"
cd "$GVM_ROOT"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$GVM_CONDA_MODULE" 2>/dev/null || module load miniforge/24.11.2
eval "$(conda shell.bash hook)"

if [[ ! -x "$GVM_CONDA_ENV/bin/python" ]]; then
    conda create -y -p "$GVM_CONDA_ENV" python=3.10
fi
conda activate "$GVM_CONDA_ENV"
python -V

pip install --upgrade pip setuptools wheel packaging ninja

# requirements.txt is a pip freeze from the authors' box; line 201 is an editable
# install of this very repo pinned to /home/ubuntu/... Drop it and install the
# repo itself from the checkout below.
grep -v '^-e file://' requirements.txt > /tmp/gvm-requirements-$$.txt
pip install -r /tmp/gvm-requirements-$$.txt
rm -f /tmp/gvm-requirements-$$.txt

# Prebuilt wheel: building flash-attn from source takes hours and needs nvcc.
# torch 2.6.0 PyPI wheels are _GLIBCXX_USE_CXX11_ABI=0 -> cxx11abiFALSE.
pip install --no-deps --no-build-isolation \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# The verl fork in this repo, without letting pip re-resolve the pinned stack.
pip install -e . --no-deps

echo "=== versions ==="
python - <<'PY'
import importlib
for m in ["torch","transformers","vllm","ray","flash_attn","wandb","datasets","verl","tensordict"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:16s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:16s} FAILED: {type(e).__name__}: {e}")
PY
echo "BUILD_OK"
