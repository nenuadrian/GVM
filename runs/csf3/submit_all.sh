#!/usr/bin/env bash
# Queue the whole Table 1 replication as one dependency chain.
#
# The gpu-h200-fse QOS caps this account at 4 H200s in total, so nothing here can
# overlap: every GPU stage waits for the previous one with afterok. A failure
# leaves the rest of the chain pending on an unsatisfiable dependency rather than
# running on top of a broken state -- scancel them, fix, and rerun.
#
#   bash runs/csf3/submit_all.sh            # everything
#   bash runs/csf3/submit_all.sh grpo       # one method (plus its eval)
#   GVM_SKIP_SMOKE=1 bash runs/csf3/submit_all.sh

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
cd "$GVM_ROOT"
mkdir -p logs
S="runs/csf3"

only="${1:-all}"
dep=""
sub() {  # sub <description> <sbatch args...>
    local desc="$1"; shift
    local out
    out=$(sbatch --parsable ${dep:+--dependency=afterok:$dep} "$@")
    dep="${out%%;*}"
    printf '%-34s job %s\n' "$desc" "$dep"
}

cd logs

if [[ "$only" == all ]]; then
    sub "prepare data"           "$GVM_ROOT/$S/01_prepare_data.sbatch"
    [[ -n "${GVM_SKIP_SMOKE:-}" ]] || sub "smoke test" "$GVM_ROOT/$S/05_smoke.sbatch"
    sub "eval base model" --export=ALL,GVM_EVAL_MODEL=Qwen/Qwen2.5-Math-1.5B,GVM_EVAL_NAME=base \
        "$GVM_ROOT/$S/04_eval.sbatch"
fi

run_method() {
    local m="$1" script
    case "$m" in
        grpo|raftpp)          script="02_train_baseline.sbatch" ;;
        gvm-grpo|gvm-raftpp)  script="03_train_gvm.sbatch" ;;
        *) echo "unknown method '$m'" >&2; exit 2 ;;
    esac
    sub "train $m"  --export="ALL,GVM_METHOD=$m" "$GVM_ROOT/$S/$script"
    sub "eval $m"   --export="ALL,GVM_METHOD=$m" "$GVM_ROOT/$S/04_eval.sbatch"
}

if [[ "$only" == all ]]; then
    for m in grpo raftpp gvm-grpo gvm-raftpp; do run_method "$m"; done
else
    run_method "$only"
fi

echo
echo "queued. watch with:  squeue -u \$USER"
