#!/bin/bash
#SBATCH --job-name=gvm-smoke
#SBATCH --partition=gpuH_short
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# End-to-end shakedown on a 256-prompt slice: a couple of GRPO steps, then the
# full GVM E-step + M-step path. Catches config, vLLM, FSDP, reward and wandb
# problems in minutes instead of three days into a real run.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/env.sh"
source "$GVM_ROOT/runs/csf3/config.sh"
cd "$GVM_ROOT"
gvm_activate

export WANDB_PROJECT="${GVM_PROJECT}-smoke"
smoke_rows=256
smoke_batch=64
smoke_dir="$GVM_ROOT/data/smoke"
mkdir -p "$smoke_dir" logs

python - "$smoke_rows" <<'PY'
import sys, pyarrow.parquet as pq
n = int(sys.argv[1])
t = pq.read_table("data/numina_math_1/train.parquet").slice(0, n)
pq.write_table(t, "data/smoke/train.parquet")
# Row order is preserved, so extra_info.index still matches the sample_sizes list.
print("smoke train rows:", t.num_rows)
v = pq.read_table("data/math500/test.parquet").slice(0, 64)
pq.write_table(v, "data/smoke/test.parquet")
print("smoke val rows:", v.num_rows)
PY

common=(
    data.train_files="['data/smoke/train.parquet']"
    data.val_files="['data/smoke/test.parquet']"
    data.train_batch_size="$smoke_batch"
    data.max_prompt_length="$GVM_MAX_PROMPT_LEN"
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="$GVM_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr="$GVM_LR"
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$GVM_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="$GVM_KL_COEF"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization="$GVM_GPU_MEM_UTIL"
    actor_rollout_ref.rollout.max_num_batched_tokens="$GVM_MAX_BATCHED_TOKENS"
    actor_rollout_ref.rollout.val_kwargs.n=2
    actor_rollout_ref.rollout.val_kwargs.temperature="$GVM_VAL_TEMP"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.kl_ctrl.kl_coef="$GVM_KL_COEF"
    trainer.critic_warmup=0
    trainer.logger="['console','wandb']"
    trainer.project_name="$WANDB_PROJECT"
    trainer.n_gpus_per_node="$GVM_GPUS"
    trainer.nnodes=1
    trainer.val_before_train=True
    trainer.save_freq=-1
    trainer.test_freq=2
    trainer.resume_mode=disable
    trainer.total_epochs=1
    trainer.total_training_steps=2
    ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-32}"
)

echo "################ smoke 1/2: GRPO baseline path ################"
python3 -m verl.trainer.main_ppo "${common[@]}" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss=plusplus \
    actor_rollout_ref.rollout.n=4 \
    trainer.experiment_name=smoke-grpo \
    trainer.default_local_dir="$GVM_ROOT/checkpoints/smoke/grpo" \
    2>&1 | tee "logs/smoke-grpo.log"

echo "################ smoke 2/2: GVM-RAFT++ E-step + M-step ################"
model_prefix="$(basename "$GVM_MODEL")"
suffix="smoke_n${GVM_STAGE1_SAMPLES}"
em_dir="$GVM_ROOT/em/data/$model_prefix/$suffix/data_1"
mkdir -p "$em_dir"

pushd "$GVM_ROOT/em" >/dev/null
for (( g=0; g<GVM_GPUS; g++ )); do
    CUDA_VISIBLE_DEVICES=$g python stage_1_collect_data.py \
        --local_index "$g" --world_size "$GVM_GPUS" \
        --model_name_or_path "$GVM_MODEL" --iter 1 \
        --data_path "ScaleML-RLHF/numina_math_1" --end "$smoke_rows" \
        --model_prefix "$model_prefix" --suffix "$suffix" \
        --stage_1_samples "$GVM_STAGE1_SAMPLES" --max_length 1024 \
        --system_prompt qwen25-math-cot &
done
wait
for (( g=0; g<GVM_GPUS; g++ )); do
    CUDA_VISIBLE_DEVICES=$g python stage_2_calc_acceptRates_grads.py \
        --local_index "$g" --num_collect_files "$GVM_GPUS" \
        --model_name_or_path "$GVM_MODEL" --iter 1 --act_params embed_tokens \
        --model_prefix "$model_prefix" --suffix "$suffix" \
        --stage_1_samples "$GVM_STAGE1_SAMPLES" --system_prompt qwen25-math-cot &
done
wait
python stage_2_calc_sample_size.py \
    --num_collect_files "$GVM_GPUS" --iter 1 \
    --model_prefix "$model_prefix" --suffix "$suffix" \
    --stage_1_samples "$GVM_STAGE1_SAMPLES" \
    --stage_2_samples $(( smoke_rows * GVM_STAGE1_SAMPLES )) \
    --alpha "$GVM_ALPHA" --beta "$GVM_BETA" \
    --filter_threshold 1.0 --filter_insufficient 0.0
popd >/dev/null

sizes_json="$em_dir/sample_sizes_sample${GVM_STAGE1_SAMPLES}_a${GVM_ALPHA}_b${GVM_BETA}.json"
python - "$sizes_json" "$smoke_rows" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
n = int(sys.argv[2])
assert len(s) == n, f"sample_sizes has {len(s)} entries, expected {n} (index alignment is broken)"
print(f"sample sizes: n={len(s)} total={sum(s)} max={max(s)} zeros={sum(1 for x in s if x == 0)}")
PY

python3 -m verl.trainer.main_ppo "${common[@]}" \
    algorithm.adv_estimator=raft \
    actor_rollout_ref.actor.policy_loss=plusplus \
    actor_rollout_ref.actor.clip_ratio_low="$GVM_CLIP_LOW" \
    actor_rollout_ref.actor.clip_ratio_high="$GVM_CLIP_HIGH_RAFT" \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.use_em=True \
    +actor_rollout_ref.rollout.sample_sizes_data="$sizes_json" \
    trainer.experiment_name=smoke-gvm-raftpp \
    trainer.default_local_dir="$GVM_ROOT/checkpoints/smoke/gvm-raftpp" \
    2>&1 | tee "logs/smoke-gvm-raftpp.log"

echo "SMOKE_OK"
