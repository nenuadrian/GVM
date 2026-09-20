#!/bin/bash
#SBATCH --job-name=raft-sft
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Stage 1 of the replication: the SFT model RAFT starts from.
#
# The paper aligns "LLaMA-7B-SFT", which is LLaMA-7B fine-tuned on the preferred
# responses of HH-RLHF. This does the same thing to a 3B base model. It is not
# optional: RAFT ranks the policy's own samples with a reward model that was
# trained on ###Human/###Assistant text, so a base model that has never seen
# that format produces samples the reward model scores as noise.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/raft/env.sh"
source "$GVM_ROOT/runs/csf3/raft/config.sh"
raft_activate
cd "$LMFLOW_ROOT"

slug="$(basename "$RAFT_BASE_MODEL" | tr '[:upper:]' '[:lower:]')"
sft_dir="$RAFT_OUT/sft/$slug"
mkdir -p "$sft_dir" "$RAFT_OUT/logs"

export WANDB_PROJECT="$RAFT_PROJECT"
export WANDB_NAME="sft-$slug"
export WANDB_RUN_GROUP="sft"
export WANDB_RESUME=allow
export WANDB_RUN_ID="$(echo -n "$RAFT_PROJECT/sft-$slug" | md5sum | cut -c1-16)"

cap=()
[[ -n "$RAFT_SFT_MAX_SAMPLES" ]] && cap=(--max_train_samples "$RAFT_SFT_MAX_SAMPLES")

echo "=== SFT $RAFT_BASE_MODEL -> $sft_dir on $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

deepspeed --num_gpus="$RAFT_GPUS" --master_port=29610 \
  examples/finetune.py \
    --model_name_or_path "$RAFT_BASE_MODEL" \
    --dataset_path data/hh_rlhf/sft \
    --output_dir "$sft_dir" --overwrite_output_dir \
    --num_train_epochs "$RAFT_SFT_EPOCHS" \
    --learning_rate "$RAFT_SFT_LR" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --block_size "$RAFT_SFT_BLOCK" \
    --per_device_train_batch_size "$RAFT_SFT_BS" \
    --deepspeed "$RAFT_S/configs/ds_zero2_no_offload.json" \
    --bf16 \
    --run_name "$WANDB_NAME" \
    --report_to wandb \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_strategy no \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 12 \
    "${cap[@]}" \
  2>&1 | tee -a "$RAFT_OUT/logs/sft-$slug.log"

# LMFlow writes the merged model at the end of tune(); make sure it is loadable
# before a 20-iteration RAFT run tries to.
python - <<PY
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
d = os.environ["RAFT_OUT"] + "/sft/" + "$slug"
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16)
print("SFT model loads:", type(m).__name__, sum(p.numel() for p in m.parameters())/1e9, "B params")
PY
echo "SFT_OK  $sft_dir"
