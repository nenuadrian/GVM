#!/bin/bash
#SBATCH --job-name=raft-align
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Stage 2: RAFT itself, through LMFlow's own raft_aligner.
#
# Each iteration: every rank draws K samples for each of raft_batch_size/world_size
# prompts, scores all K with the reward model, keeps the argmax, and SFTs on the
# kept set for num_train_epochs. That is Algorithm 1 of arXiv:2304.06767.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/raft/env.sh"
source "$GVM_ROOT/runs/csf3/raft/config.sh"
raft_activate
cd "$LMFLOW_ROOT"

slug="$(basename "$RAFT_BASE_MODEL" | tr '[:upper:]' '[:lower:]')"
sft_dir="${RAFT_SFT_DIR:-$RAFT_OUT/sft/$slug}"
raft_dir="$RAFT_OUT/raft/$RAFT_RUN_NAME"

if [[ ! -f "$sft_dir/config.json" ]]; then
    echo "No SFT model at $sft_dir -- run 02_sft.sbatch first, or point" \
         "RAFT_SFT_DIR at an existing HH-RLHF SFT checkpoint." >&2
    exit 2
fi
mkdir -p "$raft_dir" "$RAFT_OUT/logs"

# K = 1/top_reward_percentage, the best-of-K width. The patched reward pipeline
# scores a whole group in one forward pass instead of K sequential ones.
K="$(python -c "print(int(round(1/float('$RAFT_TOP_PCT'))))")"
export RAFT_RM_BATCH_SIZE="$K"

export WANDB_PROJECT="$RAFT_PROJECT"
export WANDB_NAME="$RAFT_RUN_NAME"
export WANDB_RUN_GROUP="raft"
export WANDB_RESUME=allow
export WANDB_RUN_ID="$(echo -n "$RAFT_PROJECT/$RAFT_RUN_NAME" | md5sum | cut -c1-16)"

echo "=== RAFT: $sft_dir, best-of-$K, ${RAFT_ITERS} iterations | $(hostname) | $(date) ==="
echo "reward model: $RAFT_REWARD_MODEL"
echo "prompts per iteration: $RAFT_BATCH_SIZE ($((RAFT_BATCH_SIZE / RAFT_GPUS)) per rank x $RAFT_GPUS ranks)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

case "$RAFT_LAUNCHER" in
    deepspeed) launch=(deepspeed --num_gpus="$RAFT_GPUS" --master_port=29611)
               ds=(--deepspeed "$RAFT_S/configs/ds_zero2_no_offload.json") ;;
    torchrun)  launch=(torchrun --nproc_per_node="$RAFT_GPUS" --master_port=29611)
               ds=() ;;
    *) echo "RAFT_LAUNCHER must be deepspeed or torchrun, got '$RAFT_LAUNCHER'" >&2; exit 2 ;;
esac

# --report_to none is deliberate. raft_aligner calls trainer.train() once per
# RAFT iteration, so the trainer's global_step restarts at 0 every time and
# wandb silently drops every step after the first iteration. The patch in
# patches/apply_patches.py logs one row per iteration on a monotonic axis
# instead, which is also where the reward curve lives -- upstream only ever put
# it in a matplotlib PNG.
"${launch[@]}" examples/raft_align.py \
    --model_name_or_path "$sft_dir" \
    --reward_model_or_path "$RAFT_REWARD_MODEL" \
    --dataset_path data/hh_rlhf/rlhf/rlhf_prompt \
    --num_raft_iteration "$RAFT_ITERS" \
    --raft_batch_size "$RAFT_BATCH_SIZE" \
    --top_reward_percentage "$RAFT_TOP_PCT" \
    --collection_strategy "$RAFT_COLLECTION" \
    --output_min_length "$RAFT_MIN_NEW" \
    --output_max_length "$RAFT_MAX_NEW" \
    --inference_batch_size_per_device "$K" \
    --learning_rate "$RAFT_LR" \
    --lr_scheduler_type constant \
    --num_train_epochs "$RAFT_EPOCHS" \
    --per_device_train_batch_size "$RAFT_TRAIN_BS" \
    --per_device_eval_batch_size 1 \
    --bf16 \
    "${ds[@]}" \
    --output_dir "$raft_dir" --overwrite_output_dir \
    --output_reward_path "$raft_dir/reward.txt" \
    --run_name "$RAFT_RUN_NAME" \
    --report_to none \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 7777 \
    --save_strategy no \
    --dataloader_num_workers 1 \
    --preprocessing_num_workers 12 \
  2>&1 | tee -a "$RAFT_OUT/logs/$RAFT_RUN_NAME.log"

echo "=== reward trace ==="
cat "$raft_dir/reward.txt" 2>/dev/null || echo "(no reward.txt)"
echo "RAFT_OK  $raft_dir"
