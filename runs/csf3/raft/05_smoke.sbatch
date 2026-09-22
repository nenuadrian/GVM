#!/bin/bash
#SBATCH --job-name=raft-smoke
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:2
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# End-to-end shakedown of every moving part, in the order they can break.
# Run this before 02/03 -- it is ~20 minutes against a ~3 hour RAFT run.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/raft/env.sh"
source "$GVM_ROOT/runs/csf3/raft/config.sh"
raft_activate
cd "$LMFLOW_ROOT"

smoke_gpus="${RAFT_SMOKE_GPUS:-2}"
smoke_out="$RAFT_OUT/smoke"
rm -rf "$smoke_out"; mkdir -p "$smoke_out"
export WANDB_PROJECT="${RAFT_PROJECT}-smoke"
export WANDB_NAME="smoke-$(date +%m%d-%H%M)"
unset WANDB_RUN_ID
export RAFT_RM_BATCH_SIZE=8

echo "############ 1. does the pipeline exist on this transformers? ############"
python - <<'PY'
import transformers
from lmflow.pipeline.auto_pipeline import PIPELINE_MAPPING
print("transformers", transformers.__version__)
assert "raft_aligner" in PIPELINE_MAPPING, (
    "raft_aligner unregistered: LMFlow drops it on transformers >= 4.35. "
    f"got {transformers.__version__}")
from lmflow.pipeline.raft_aligner import RaftAligner          # noqa: F401
from lmflow.pipeline.utils.raft_trainer import RaftTrainer    # noqa: F401
print("OK: raft_aligner + raft_trainer import")
PY

echo "############ 2. is the patch actually applied? ############"
grep -q "csf3] one wandb point per RAFT iteration" src/lmflow/pipeline/raft_aligner.py \
    && echo "OK: aligner wandb patch present" \
    || { echo "FAIL: run patches/apply_patches.py"; exit 1; }
grep -q "csf3] reward pipeline" examples/raft_align.py \
    && echo "OK: reward pipeline patch present" \
    || { echo "FAIL: run patches/apply_patches.py"; exit 1; }

echo "############ 3. does the reward model rank HH-RLHF pairs correctly? ############"
# The strongest cheap check there is: if the RM cannot separate chosen from
# rejected on its own training distribution, every reward in the run is noise
# and the format or the dtype is wrong.
python - <<'PY'
import json, os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

rm_id = os.environ["RAFT_REWARD_MODEL"]
try:
    tok = AutoTokenizer.from_pretrained(rm_id)
except Exception as e:
    print(f"fast tokenizer unavailable ({e}); using slow"); tok = AutoTokenizer.from_pretrained(rm_id, use_fast=False)
tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id; tok.padding_side = "left"
rm = AutoModelForSequenceClassification.from_pretrained(
    rm_id, torch_dtype=torch.bfloat16, num_labels=1).to("cuda").eval()

pairs = json.load(open("data/hh_rlhf/rm/hh_rlhf_rm_data.json"))["instances"][:64]
pos = [p["positive"] for p in pairs]; neg = [p["negative"] for p in pairs]

def score(texts):
    out = []
    for i in range(0, len(texts), 8):
        enc = tok(texts[i:i+8], return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out += rm(**enc).logits.squeeze(-1).float().cpu().tolist()
    return out

sp, sn = score(pos), score(neg)
acc = sum(a > b for a, b in zip(sp, sn)) / len(sp)
print(f"pairwise accuracy on {len(sp)} held-in pairs: {acc:.3f}")
print(f"mean chosen {sum(sp)/len(sp):.3f}  mean rejected {sum(sn)/len(sn):.3f}")
# The card reports ~69% for the GPT-Neo RM and ~79% for LLaMA-7B; anything near
# chance means the model or the prompt format is wrong, not merely weak.
assert acc > 0.6, f"reward model is near chance ({acc:.3f}) -- check format/dtype"
print("OK: reward model discriminates")
PY

echo "############ 4. two RAFT iterations, real model, tiny batch ############"
deepspeed --num_gpus="$smoke_gpus" --master_port=29699 \
  examples/raft_align.py \
    --model_name_or_path "$RAFT_BASE_MODEL" \
    --reward_model_or_path "$RAFT_REWARD_MODEL" \
    --dataset_path data/hh_rlhf/rlhf/rlhf_prompt \
    --num_raft_iteration 2 \
    --raft_batch_size $(( 8 * smoke_gpus )) \
    --top_reward_percentage "$RAFT_TOP_PCT" \
    --collection_strategy "$RAFT_COLLECTION" \
    --output_min_length 32 --output_max_length 48 \
    --inference_batch_size_per_device 8 \
    --learning_rate "$RAFT_LR" --lr_scheduler_type constant \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --bf16 \
    --deepspeed "$RAFT_S/configs/ds_zero2_no_offload.json" \
    --output_dir "$smoke_out/model" --overwrite_output_dir \
    --output_reward_path "$smoke_out/reward.txt" \
    --run_name "$WANDB_NAME" --report_to none \
    --validation_split_percentage 0 --logging_steps 1 --do_train \
    --ddp_timeout 72000 --save_steps 7777 --save_strategy no \
    --dataloader_num_workers 1 --preprocessing_num_workers 8 \
  2>&1 | tee "$smoke_out/raft.log"

echo "--- reward trace ---"; cat "$smoke_out/reward.txt"
grep -q "mean reward" "$smoke_out/reward.txt" \
    || { echo "FAIL: no rewards recorded"; exit 1; }
[[ -f "$smoke_out/model/config.json" ]] \
    || { echo "FAIL: aligner did not save a model"; exit 1; }
echo "OK: RAFT loop ran and saved"

echo "############ 5. eval path ############"
python "$RAFT_S/reward_eval.py" \
    --model "$smoke_out/model" \
    --reward_model "$RAFT_REWARD_MODEL" \
    --prompts "$RAFT_OUT/data/eval_prompts/eval_prompts.json" \
    --n 32 --batch_size 8 --max_new_tokens 48 \
    --tag smoke --out "$smoke_out/eval.json" --wandb

echo "SMOKE_OK"
