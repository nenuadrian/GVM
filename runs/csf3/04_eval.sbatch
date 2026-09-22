#!/bin/bash
#SBATCH --job-name=gvm-eval
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Table 1 evaluation: avg@8 at temperature 1.0 over MATH500, Minerva Math,
# Olympiad Bench, AIME24 and AMC23, using the repo's own eval/ pipeline.
#
#   GVM_EVAL_MODEL=<hf path or hub id> GVM_EVAL_NAME=<label> sbatch 04_eval.sbatch
#
# With GVM_METHOD set instead, the final checkpoint of that run is located
# automatically; GVM_EVAL_NAME=base evaluates the untrained Qwen2.5-Math-1.5B.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
source "$GVM_ROOT/runs/csf3/env.sh"
source "$GVM_ROOT/runs/csf3/config.sh"
cd "$GVM_ROOT"
gvm_activate

if [[ -n "${GVM_METHOD:-}" ]]; then
    # Must match 03_train_gvm.sbatch, which appends GVM_RUN_SUFFIX. Without it an
    # ablation arm's eval silently resolves to the base arm's checkpoint tree.
    run_name="${GVM_METHOD}-qwen15${GVM_RUN_SUFFIX:-}"
    case "$GVM_METHOD" in
        grpo|raftpp)
            # Baselines keep one checkpoint tree; take the highest global_step_N.
            base="$GVM_ROOT/checkpoints/$GVM_PROJECT/$run_name"
            last=$(ls -d "$base"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
            [[ -n "$last" ]] || { echo "no checkpoint under $base" >&2; exit 2; }
            src="$base/global_step_$last/actor"
            ;;
        gvm-grpo|gvm-raftpp)
            # The EM loop leaves a .gvm_iteration_done marker pointing at each
            # iteration's merged HF export; take the last one.
            last_iter=$(ls -d "$GVM_ROOT/checkpoints/$GVM_PROJECT/${run_name}-iter"* 2>/dev/null \
                        | sed 's/.*-iter//' | sort -n | tail -1)
            [[ -n "$last_iter" ]] || { echo "no iterations for $run_name" >&2; exit 2; }
            marker="$GVM_ROOT/checkpoints/$GVM_PROJECT/${run_name}-iter${last_iter}/.gvm_iteration_done"
            [[ -f "$marker" ]] || { echo "iteration $last_iter never finished merging" >&2; exit 2; }
            src="$(dirname "$(cat "$marker")")"
            ;;
        *) echo "unknown GVM_METHOD '$GVM_METHOD'" >&2; exit 2 ;;
    esac
    GVM_EVAL_NAME="${GVM_EVAL_NAME:-$run_name}"
    # verl saves config.json + tokenizer under actor/huggingface on every step but
    # no weights, so only a *.safetensors there means the merge has happened.
    if compgen -G "$src/huggingface/*.safetensors" >/dev/null; then
        GVM_EVAL_MODEL="$src/huggingface"
    else
        echo "--- merging FSDP shards for evaluation ---"
        GVM_EVAL_MODEL="$src/huggingface"
        # See 03_train_gvm.sbatch: --hf_model_path is required despite the
        # deprecation notice, because actor/huggingface/ is written empty.
        python scripts/legacy_model_merger.py merge --backend fsdp \
            --hf_model_path "$GVM_MODEL" \
            --local_dir "$src" --target_dir "$GVM_EVAL_MODEL"
        compgen -G "$GVM_EVAL_MODEL/*.safetensors" >/dev/null \
            || { echo "merge produced no weights" >&2; exit 2; }
    fi
fi

: "${GVM_EVAL_MODEL:?set GVM_EVAL_MODEL or GVM_METHOD}"
: "${GVM_EVAL_NAME:?set GVM_EVAL_NAME or GVM_METHOD}"
data="${GVM_EVAL_DATA:-math500,minerva_math,olympiad_bench,aime24,amc23}"

echo "=== eval $GVM_EVAL_NAME <- $GVM_EVAL_MODEL ==="
cd "$GVM_ROOT/eval"

# One vLLM process per GPU, each taking a contiguous slice of every benchmark.
for (( g=0; g<GVM_GPUS; g++ )); do
    CUDA_VISIBLE_DEVICES=$g python gen.py \
        --model_name_or_path "$GVM_EVAL_MODEL" \
        --model_prefix "$GVM_EVAL_NAME" \
        --data "$data" \
        --n "$GVM_VAL_N" \
        --temperature "$GVM_VAL_TEMP" \
        --max_length "$GVM_MAX_RESP_LEN" \
        --num_gpu "$GVM_GPUS" --local_rank "$g" &
done
wait

python aggregate.py --model_prefix "$GVM_EVAL_NAME" --data "$data" --num_gpu "$GVM_GPUS"

cd "$GVM_ROOT"
python runs/csf3/log_eval.py --name "$GVM_EVAL_NAME" --data "$data" \
    || echo "(wandb eval log failed, results are still in eval/result/)"

echo "EVAL_OK $GVM_EVAL_NAME"
