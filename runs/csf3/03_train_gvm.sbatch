#!/bin/bash
#SBATCH --job-name=gvm-em
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=4-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# GVM-GRPO / GVM-RAFT++ : the EM loop of Algorithm 1.
# Each iteration = 1 E-step (GVM budget scheduling on Numina shard i) followed by
# 9 M-steps (verl update, batch 1024 over ~9986 prompts). 15 iterations = 135 steps.
#
#   GVM_METHOD=gvm-grpo   sbatch 03_train_gvm.sbatch
#   GVM_METHOD=gvm-raftpp sbatch 03_train_gvm.sbatch
#
# Every stage is skipped if its output already exists, so the job is safe to
# resubmit after a wall-clock kill.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/env.sh"
source "$GVM_ROOT/runs/csf3/config.sh"
cd "$GVM_ROOT"
gvm_activate

method="${GVM_METHOD:?set GVM_METHOD to gvm-grpo or gvm-raftpp}"
case "$method" in
    gvm-grpo)
        adv=grpo; policy_loss=plusplus; clip_high="$GVM_CLIP_HIGH_GRPO"
        rollout_n="$GVM_GRPO_ROLLOUT_N"   # 4: budget is divided by 4, prompts copied n_i/4 times
        ;;
    gvm-raftpp)
        adv=raft; policy_loss=plusplus; clip_high="$GVM_CLIP_HIGH_RAFT"
        rollout_n="$GVM_RAFT_ROLLOUT_N"   # 1: prompts copied n_i times, one rollout each
        ;;
    *) echo "GVM_METHOD must be gvm-grpo or gvm-raftpp, got '$method'" >&2; exit 2 ;;
esac

run_name="${method}-qwen15"
model_prefix="$(basename "$GVM_MODEL")"
suffix="${method}_n${GVM_STAGE1_SAMPLES}_a${GVM_ALPHA}_b${GVM_BETA}"
steps_per_iter=9

export WANDB_PROJECT="$GVM_PROJECT"
export WANDB_NAME="$run_name"
export WANDB_RUN_GROUP="gvm"
export WANDB_RESUME=allow
export WANDB_RUN_ID="$(echo -n "$GVM_PROJECT/$run_name" | md5sum | cut -c1-16)"

# verl writes <ckpt>/actor/huggingface/{config.json,tokenizer} on every save, with
# no weights -- only the merge step below puts weights there. So "is this
# iteration finished" has to test for weights, not for config.json.
has_weights() { compgen -G "$1/*.safetensors" >/dev/null || [[ -f "$1/pytorch_model.bin" ]]; }
# The number of steps in an iteration follows from the shard size; don't assume 9.
last_step_dir() {
    local base="$1" n
    n=$(ls -d "$base"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    [[ -n "$n" ]] && echo "$base/global_step_$n"
}

echo "=== $run_name on $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for (( i=1; i<=GVM_EM_ITERS; i++ )); do
    iter_run="${run_name}-iter${i}"
    ckpt_dir="$GVM_ROOT/checkpoints/$GVM_PROJECT/$iter_run"

    done_marker="$ckpt_dir/.gvm_iteration_done"
    if [[ -f "$done_marker" ]] && has_weights "$(cat "$done_marker")"; then
        echo "--- iteration $i already complete, skipping ---"
        continue
    fi

    # Policy that generates this iteration's samples: base model on iter 1,
    # otherwise the merged HF export of the previous iteration's last step.
    if (( i == 1 )); then
        policy="$GVM_MODEL"
    else
        prev_marker="$GVM_ROOT/checkpoints/$GVM_PROJECT/${run_name}-iter$((i-1))/.gvm_iteration_done"
        [[ -f "$prev_marker" ]] || { echo "iteration $((i-1)) never finished" >&2; exit 3; }
        policy="$(cat "$prev_marker")"
        has_weights "$policy" || { echo "no weights at $policy" >&2; exit 3; }
    fi

    em_dir="$GVM_ROOT/em/data/$model_prefix/$suffix/data_${i}"
    sizes_json="$em_dir/sample_sizes_sample${GVM_STAGE1_SAMPLES}_a${GVM_ALPHA}_b${GVM_BETA}.json"
    mkdir -p "$em_dir"

    echo "=============== iteration $i/$GVM_EM_ITERS | policy=$policy ==============="

    if [[ ! -f "$sizes_json" ]]; then
        pushd "$GVM_ROOT/em" >/dev/null

        # --- E-step part 1: N'=8 pilot rollouts per prompt, sharded over 4 GPUs
        if [[ ! -f "$em_dir/stage_1_collected_data_$((GVM_GPUS-1)).json" ]]; then
            echo "--- stage 1: pilot sampling (iter $i) ---"
            for (( g=0; g<GVM_GPUS; g++ )); do
                CUDA_VISIBLE_DEVICES=$g python stage_1_collect_data.py \
                    --local_index "$g" --world_size "$GVM_GPUS" \
                    --model_name_or_path "$policy" \
                    --iter "$i" --data_path "ScaleML-RLHF/numina_math_${i}" \
                    --model_prefix "$model_prefix" --suffix "$suffix" \
                    --stage_1_samples "$GVM_STAGE1_SAMPLES" \
                    --max_length "$GVM_MAX_RESP_LEN" \
                    --system_prompt qwen25-math-cot &
            done
            wait
        fi

        # --- E-step part 2: accept rates p_i and Lipschitz/gradient norms G_i
        if [[ ! -f "$em_dir/grads_$((GVM_GPUS-1)).json" ]]; then
            echo "--- stage 2a: accept rates + gradient norms (iter $i) ---"
            for (( g=0; g<GVM_GPUS; g++ )); do
                CUDA_VISIBLE_DEVICES=$g python stage_2_calc_acceptRates_grads.py \
                    --local_index "$g" --num_collect_files "$GVM_GPUS" \
                    --model_name_or_path "$policy" \
                    --iter "$i" --act_params embed_tokens \
                    --model_prefix "$model_prefix" --suffix "$suffix" \
                    --stage_1_samples "$GVM_STAGE1_SAMPLES" \
                    --system_prompt qwen25-math-cot &
            done
            wait
        fi

        # --- E-step part 3: Proposition 1 closed-form budget allocation.
        # N = 8 * (#prompts) overall. GVM-GRPO asks for N/4 copies and then draws
        # 4 rollouts per copy, so the realised budget is the same N either way.
        n_prompts=$(python -c "
import json, sys
d, w = sys.argv[1], int(sys.argv[2])
print(sum(len(json.load(open(f'{d}/accept_rates_{k}.json'))) for k in range(w)))
" "$em_dir" "$GVM_GPUS")
        budget=$(( n_prompts * GVM_STAGE1_SAMPLES / rollout_n ))
        echo "--- stage 2b: budget scheduling | prompts=$n_prompts N=$budget rollout_n=$rollout_n ---"
        python stage_2_calc_sample_size.py \
            --num_collect_files "$GVM_GPUS" --iter "$i" \
            --model_prefix "$model_prefix" --suffix "$suffix" \
            --stage_1_samples "$GVM_STAGE1_SAMPLES" \
            --stage_2_samples "$budget" \
            --alpha "$GVM_ALPHA" --beta "$GVM_BETA" \
            --filter_threshold 1.0 --filter_insufficient 0.0
        popd >/dev/null
    fi

    [[ -f "$sizes_json" ]] || { echo "budget file not produced: $sizes_json" >&2; exit 4; }

    # Log this E-step's allocation to the same wandb run as the M-steps.
    VERL_LOG_STEP_OFFSET=$(( (i - 1) * steps_per_iter )) \
        python runs/csf3/log_estep.py --iter "$i" --em_dir "$em_dir" \
            --sizes "$sizes_json" --world_size "$GVM_GPUS" \
            --step $(( (i - 1) * steps_per_iter )) || echo "(wandb E-step log failed, continuing)"

    # --- M-step: 9 verl updates over this shard.
    echo "--- M-steps: verl (iter $i) ---"
    VERL_LOG_STEP_OFFSET=$(( (i - 1) * steps_per_iter )) \
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator="$adv" \
        algorithm.kl_ctrl.kl_coef="$GVM_KL_COEF" \
        data.train_files="['data/numina_math_${i}/train.parquet']" \
        data.val_files="['data/math500/test.parquet']" \
        data.train_batch_size="$GVM_TRAIN_BATCH" \
        data.max_prompt_length="$GVM_MAX_PROMPT_LEN" \
        data.max_response_length="$GVM_MAX_RESP_LEN" \
        data.filter_overlong_prompts=True \
        data.filter_overlong_prompts_workers=8 \
        data.truncation=error \
        data.shuffle=True \
        actor_rollout_ref.model.path="$policy" \
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
        actor_rollout_ref.rollout.n="$rollout_n" \
        +actor_rollout_ref.rollout.use_em=True \
        +actor_rollout_ref.rollout.sample_sizes_data="$sizes_json" \
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
        trainer.val_before_train=$([[ $i -eq 1 ]] && echo True || echo False) \
        trainer.save_freq="$GVM_SAVE_FREQ" \
        trainer.test_freq="$GVM_TEST_FREQ" \
        trainer.resume_mode=auto \
        trainer.default_local_dir="$ckpt_dir" \
        trainer.total_epochs=1 \
        ray_init.num_cpus="${SLURM_CPUS_PER_TASK:-16}" \
        2>&1 | tee -a "logs/${iter_run}.log"

    # --- Export to HF format so the next E-step can load it with vLLM.
    final_ckpt="$(last_step_dir "$ckpt_dir")"
    [[ -n "$final_ckpt" ]] || { echo "no checkpoint written for iter $i" >&2; exit 5; }
    hf_ckpt="$final_ckpt/actor/huggingface"
    echo "--- merging FSDP shards to HF: $final_ckpt ---"
    python scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir "$final_ckpt/actor" \
        --target_dir "$hf_ckpt"

    has_weights "$hf_ckpt" || { echo "merge produced no weights for iter $i" >&2; exit 5; }
    echo "$hf_ckpt" > "$done_marker"
done

echo "TRAIN_OK $run_name"
