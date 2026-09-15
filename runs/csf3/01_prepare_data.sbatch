#!/bin/bash
#SBATCH --job-name=gvm-data
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
gvm_activate

# MATH500 validation set (the paper's headline benchmark).
if [[ ! -f data/math500/test.parquet ]]; then
    python runs/data_preprocess/math500.py --local_dir data/math500
fi

# Numina-Math: 15 shards of ~9986 prompts (one per EM iteration) plus the
# concatenation of all 15 (149,882 prompts) used by the non-GVM baselines.
# No --train_end: the parquet row order must line up 1:1 with the HF split that
# em/stage_1_collect_data.py reads, because sample_sizes is indexed by row.
if [[ ! -f data/numina_math_15_all/train.parquet ]]; then
    python runs/data_preprocess/numina_process.py --local_dir ./data/numina_math
fi

# Warm the HF cache on a node with internet so training jobs never block on it.
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-Math-1.5B")
print("model cached")
PY

echo "=== parquet row counts ==="
python - <<'PY'
import glob, pyarrow.parquet as pq
for f in sorted(glob.glob("data/numina_math_*/train.parquet")) + ["data/math500/test.parquet"]:
    print(f"{pq.ParquetFile(f).metadata.num_rows:>7d}  {f}")
PY
echo "DATA_OK"
