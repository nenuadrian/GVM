#!/bin/bash
#SBATCH --job-name=raft-eval
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Mean reward on held-out HH-RLHF prompts for base / sft / raft, which is the
# comparison the paper reports for this experiment. One GPU is enough.
#
#   RAFT_EVAL_TAG=sft sbatch 04_eval.sbatch     # just one
#   sbatch 04_eval.sbatch                       # all three present on disk

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/raft/env.sh"
source "$GVM_ROOT/runs/csf3/raft/config.sh"
raft_activate
cd "$LMFLOW_ROOT"

slug="$(basename "$RAFT_BASE_MODEL" | tr '[:upper:]' '[:lower:]')"
sft_dir="${RAFT_SFT_DIR:-$RAFT_OUT/sft/$slug}"
raft_dir="$RAFT_OUT/raft/$RAFT_RUN_NAME"
prompts="$RAFT_OUT/data/eval_prompts/eval_prompts.json"
mkdir -p "$RAFT_OUT/eval"

export WANDB_PROJECT="$RAFT_PROJECT"
export WANDB_NAME="$RAFT_RUN_NAME"
export WANDB_RESUME=allow
export WANDB_RUN_ID="$(echo -n "$RAFT_PROJECT/$RAFT_RUN_NAME" | md5sum | cut -c1-16)"

run_one() {  # run_one <tag> <model path>
    local tag="$1" model="$2"
    if [[ "$model" != "$RAFT_BASE_MODEL" && ! -f "$model/config.json" ]]; then
        echo "-- skipping $tag: no model at $model"
        return 0
    fi
    echo "=== eval $tag: $model ==="
    python "$RAFT_S/reward_eval.py" \
        --model "$model" \
        --reward_model "$RAFT_REWARD_MODEL" \
        --prompts "$prompts" \
        --n "$RAFT_EVAL_N" \
        --temperature "$RAFT_EVAL_TEMP" \
        --max_new_tokens "$RAFT_MAX_NEW" \
        --batch_size "$RAFT_EVAL_BS" \
        --tag "$tag" \
        --out "$RAFT_OUT/eval/$RAFT_RUN_NAME.$tag.json" \
        --wandb
}

[[ -f "$prompts" ]] || { echo "no eval prompts at $prompts -- run 01_prepare_data.sbatch" >&2; exit 2; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ -n "${RAFT_EVAL_TAG:-}" ]]; then
    case "$RAFT_EVAL_TAG" in
        base) run_one base "$RAFT_BASE_MODEL" ;;
        sft)  run_one sft  "$sft_dir" ;;
        raft) run_one raft "$raft_dir" ;;
        *) echo "RAFT_EVAL_TAG must be base, sft or raft" >&2; exit 2 ;;
    esac
else
    run_one base "$RAFT_BASE_MODEL"
    run_one sft  "$sft_dir"
    run_one raft "$raft_dir"
fi

echo "=== summary ==="
python - <<PY
import glob, json, os
rows = []
for f in sorted(glob.glob(os.environ["RAFT_OUT"] + "/eval/" + os.environ["RAFT_RUN_NAME"] + ".*.json")):
    d = json.load(open(f))
    rows.append((d["tag"], d["n"], d["mean_reward"], d["sem_reward"], d["mean_response_tokens"]))
print(f"{'tag':<6} {'n':>6} {'mean reward':>13} {'+/- sem':>9} {'resp tok':>9}")
for t, n, m, s, L in rows:
    print(f"{t:<6} {n:>6} {m:>13.4f} {s:>9.4f} {L:>9.1f}")
PY
echo "EVAL_OK"
