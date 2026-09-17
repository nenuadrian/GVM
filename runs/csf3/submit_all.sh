#!/usr/bin/env bash
# Queue the whole Table 1 replication as one dependency chain.
#
# The gpu-h200-fse QOS caps this account at 4 H200s in total, so nothing here can
# overlap: every GPU stage waits for the previous one with afterok. A failure
# leaves the rest of the chain pending on an unsatisfiable dependency rather than
# running on top of a broken state -- scancel them, fix, and rerun.
#
#   bash runs/csf3/submit_all.sh                        # everything, on H200
#   bash runs/csf3/submit_all.sh grpo                   # one method (plus its eval)
#   GVM_TARGET=l40s bash runs/csf3/submit_all.sh        # on gpuL instead
#   GVM_PRESET=small GVM_TARGET=l40s bash runs/csf3/submit_all.sh
#   GVM_SKIP_SMOKE=1 bash runs/csf3/submit_all.sh
#
# GVM_TARGET picks the partition/account/gres (see targets.sh). h200 and l40s
# draw on different QOS pools, so two chains -- one per target -- can run at the
# same time; splitting the four methods across them roughly halves wall clock.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
cd "$GVM_ROOT"
mkdir -p logs
S="runs/csf3"

# GVM_PRESET=small shortens the sweep; the exported vars reach the jobs through
# sbatch --export=ALL, and config.sh only fills in defaults for what is unset.
if [[ -n "${GVM_PRESET:-}" ]]; then
    preset="$GVM_ROOT/$S/preset_${GVM_PRESET}.sh"
    [[ -f "$preset" ]] || { echo "no such preset: $preset" >&2; exit 2; }
    source "$preset"
    echo "preset '$GVM_PRESET': ${GVM_EM_ITERS} EM iterations, baselines capped at" \
         "${GVM_BASELINE_STEPS} steps, eval on ${GVM_EVAL_DATA}, project ${GVM_PROJECT}"
fi

only="${1:-all}"
GVM_MODEL_ID="${GVM_MODEL:-Qwen/Qwen2.5-Math-1.5B}"

source "$GVM_ROOT/$S/targets.sh"
gvm_target "${GVM_TARGET:-h200}"
echo "target '${GVM_TARGET:-h200}': $GVM_SB_PARTITION / $GVM_SB_ACCOUNT / $GVM_SB_GRES / $GVM_SB_MEM"

# sbatch CLI flags override the #SBATCH directives baked into each script, so the
# scripts stay readable with an H200 default and still retarget cleanly.
place=(--partition="$GVM_SB_PARTITION" --account="$GVM_SB_ACCOUNT"
       --gres="$GVM_SB_GRES" --mem="$GVM_SB_MEM")

dep=""
n_dry=0
sub() {  # sub <description> <sbatch args...>
    local desc="$1"; shift
    if [[ -n "${GVM_DRY_RUN:-}" ]]; then
        n_dry=$((n_dry + 1))
        printf '%-34s sbatch %s%s\n' "$desc" \
            "${dep:+--dependency=afterok:$dep }" "$*"
        dep="dry$n_dry"
        return 0
    fi
    local out
    out=$(sbatch --parsable ${dep:+--dependency=afterok:$dep} "$@")
    dep="${out%%;*}"
    printf '%-34s job %s\n' "$desc" "$dep"
}

cd logs

if [[ "$only" == all ]]; then
    sub "prepare data"           "$GVM_ROOT/$S/01_prepare_data.sbatch"
    [[ -n "${GVM_SKIP_SMOKE:-}" ]] || sub "smoke test" "${place[@]}" "$GVM_ROOT/$S/05_smoke.sbatch"
    sub "eval base model" "${place[@]}" \
        --export="ALL,GVM_EVAL_MODEL=$GVM_MODEL_ID,GVM_EVAL_NAME=base" \
        "$GVM_ROOT/$S/04_eval.sbatch"
fi

run_method() {
    local m="$1" script
    case "$m" in
        grpo|raftpp)          script="02_train_baseline.sbatch" ;;
        gvm-grpo|gvm-raftpp)  script="03_train_gvm.sbatch" ;;
        *) echo "unknown method '$m'" >&2; exit 2 ;;
    esac
    sub "train $m"  "${place[@]}" --export="ALL,GVM_METHOD=$m" "$GVM_ROOT/$S/$script"
    sub "eval $m"   "${place[@]}" --export="ALL,GVM_METHOD=$m" "$GVM_ROOT/$S/04_eval.sbatch"
}

if [[ "$only" == all ]]; then
    for m in grpo raftpp gvm-grpo gvm-raftpp; do run_method "$m"; done
else
    run_method "$only"
fi

echo
echo "queued. watch with:  squeue -u \$USER"
