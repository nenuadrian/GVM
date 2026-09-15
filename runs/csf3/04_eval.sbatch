#!/bin/bash
#SBATCH --job-name=gvm-eval
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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
    run_name="${GVM_METHOD}-qwen15"
    case "$GVM_METHOD" in
        grpo|raftpp)
            # Baselines keep one checkpoint tree; take the highest global_step_N.
            base="$GVM_ROOT/checkpoints/$GVM_PROJECT/$run_name"
            last=$(ls -d "$base"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
            [[ -n "$last" ]] || { echo "no checkpoint under $base" >&2; exit 2; }
            src="$base/global_step_$last/actor"
            ;;
        gvm-grpo|gvm-raftpp)
            last_iter=$(ls -d "$GVM_ROOT/checkpoints/$GVM_PROJECT/${run_name}-iter"* 2>/dev/null \
                        | sed 's/.*-iter//' | sort -n | tail -1)
            [[ -n "$last_iter" ]] || { echo "no iterations for $run_name" >&2; exit 2; }
            src="$GVM_ROOT/checkpoints/$GVM_PROJECT/${run_name}-iter${last_iter}/global_step_9/actor"
            ;;
        *) echo "unknown GVM_METHOD '$GVM_METHOD'" >&2; exit 2 ;;
    esac
    GVM_EVAL_NAME="${GVM_EVAL_NAME:-$run_name}"
    if [[ -f "$src/huggingface/config.json" && -f "$src/huggingface/model.safetensors" ]]; then
        GVM_EVAL_MODEL="$src/huggingface"
    else
        GVM_EVAL_MODEL="$src/hf_export"
        if [[ ! -f "$GVM_EVAL_MODEL/config.json" ]]; then
            echo "--- merging FSDP shards for evaluation ---"
            python scripts/legacy_model_merger.py merge --backend fsdp \
                --hf_model_path "$GVM_MODEL" --local_dir "$src" --target_dir "$GVM_EVAL_MODEL"
        fi
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
