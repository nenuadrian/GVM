#!/bin/bash
#SBATCH --job-name=gvm-base
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --time=4-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# GRPO / RAFT++ baseline on the full 15-shard Numina-Math split (149,882 prompts
# = 146 steps at batch 1024), validating on MATH500 avg@8.
#
#   GVM_METHOD=grpo   sbatch 02_train_baseline.sbatch
#   GVM_METHOD=raftpp sbatch 02_train_baseline.sbatch

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/env.sh"
source "$GVM_ROOT/runs/csf3/config.sh"
cd "$GVM_ROOT"
gvm_activate

method="${GVM_METHOD:?set GVM_METHOD to grpo or raftpp}"
case "$method" in
    grpo)   adv=grpo; policy_loss=plusplus; clip_high="$GVM_CLIP_HIGH_GRPO" ;;
    raftpp) adv=raft; policy_loss=plusplus; clip_high="$GVM_CLIP_HIGH_RAFT" ;;
    *) echo "GVM_METHOD must be grpo or raftpp, got '$method'" >&2; exit 2 ;;
esac

run_name="${method}-qwen15"
ckpt_dir="checkpoints/$GVM_PROJECT/$run_name"
mkdir -p "$ckpt_dir" logs

export WANDB_PROJECT="$GVM_PROJECT"
export WANDB_NAME="$run_name"
export WANDB_RUN_GROUP="baseline"
# Reuse one wandb run across slurm requeues so the curve stays continuous.
export WANDB_RESUME=allow
export WANDB_RUN_ID="$(echo -n "$GVM_PROJECT/$run_name" | md5sum | cut -c1-16)"

# If the 4-day wall clock is hit, just sbatch this script again: save_freq=5 plus
# resume_mode=auto continue from the last checkpoint into the same wandb run.

echo "=== $run_name on $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$adv" \
    algorithm.kl_ctrl.kl_coef="$GVM_KL_COEF" \
    data.train_files="['data/numina_math_15_all/train.parquet']" \
    data.val_files="['data/math500/test.parquet']" \
    data.train_batch_size="$GVM_TRAIN_BATCH" \
    data.max_prompt_length="$GVM_MAX_PROMPT_LEN" \
    data.max_response_length="$GVM_MAX_RESP_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$GVM_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="$GVM_LR" \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$GVM_PPO_MINI_BATCH" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$GVM_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.policy_loss="$policy_loss" \
    actor_rollout_ref.actor.clip_ratio_low="$GVM_CLIP_LOW" \
    actor_rollout_ref.actor.clip_ratio_high="$clip_high" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$GVM_KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GVM_GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$GVM_MAX_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.n="$GVM_ROLLOUT_N" \
    actor_rollout_ref.rollout.val_kwargs.n="$GVM_VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$GVM_VAL_TEMP" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="$GVM_PROJECT" \
    trainer.experiment_name="$run_name" \
    trainer.n_gpus_per_node="$GVM_GPUS" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq="$GVM_SAVE_FREQ" \
    trainer.test_freq="$GVM_TEST_FREQ" \
    trainer.resume_mode=auto \
    trainer.default_local_dir="$ckpt_dir" \
    trainer.total_epochs=1 \
    ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-32}" \
    2>&1 | tee -a "logs/$run_name.log"

echo "TRAIN_OK $run_name"
