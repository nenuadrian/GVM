#!/usr/bin/env bash
# Queue the RAFT replication as one dependency chain.
#
#   bash runs/csf3/raft/submit_all.sh            # data -> smoke -> sft -> raft -> eval
#   bash runs/csf3/raft/submit_all.sh --no-smoke
#   bash runs/csf3/raft/submit_all.sh --from raft
#   bash runs/csf3/raft/submit_all.sh --target l40s   # gpu-free pool, no H200 wait
#   RAFT_DEP=12345 bash runs/csf3/raft/submit_all.sh --from smoke
#   RAFT_DRY_RUN=1 bash runs/csf3/raft/submit_all.sh
#
# 00_build_env.sbatch is NOT in the chain: submit it once by hand and read its
# output, because everything downstream depends on the transformers pin holding.
#
# The gpu-h200-fse QOS caps this account at 4 H200s in total, so this competes
# with runs/csf3/submit_all.sh. Do not run both chains at once.

set -euo pipefail
GVM_ROOT="${GVM_ROOT:-$HOME/scratch/GVM}"
S="$GVM_ROOT/runs/csf3/raft"
source "$S/env.sh"
source "$S/config.sh"
cd "$GVM_ROOT"; mkdir -p logs

smoke=1; from=data; target="${RAFT_TARGET:-h200}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-smoke) smoke=0 ;;
        --from) from="$2"; shift ;;
        --target) target="$2"; shift ;;
        *) echo "unknown option '$1'" >&2; exit 2 ;;
    esac
    shift
done

# gpu-h200-fse's gres/gpu:h200=4 is a QOS *group* limit shared across every user
# on it, not a per-account one, so gpuH can be full with none of your own jobs
# running. gpu-free (gpuL/gpuA) is a separate pool, so an L40S chain neither
# waits on nor consumes the H200 quota.
#
# A 3B policy plus a 3B reward model in bf16 is ~15 GB before optimizer state, so
# this fits an L40S's 48 GB with room; the H200's 141 GB is not needed here.
case "$target" in
    h200) PART=gpuH; ACCT=gpu-h200-fse-pgdr; DEV=h200;     MEM=200G ;;
    l40s) PART=gpuL; ACCT=gpu-free;          DEV=l40s;     MEM=160G ;;
    a100) PART=gpuA; ACCT=gpu-free;          DEV=a100_80g; MEM=160G ;;
    *) echo "--target must be h200, l40s or a100" >&2; exit 2 ;;
esac
place=(--partition="$PART" --account="$ACCT" --mem="$MEM")
echo "target '$target': $PART / $ACCT / gpu:$DEV / $MEM"

# Chain onto a job that is already queued (e.g. 01_prepare_data submitted by
# hand alongside 00_build_env): RAFT_DEP=<jobid> bash submit_all.sh --from smoke
dep="${RAFT_DEP:-}"; n_dry=0
sub() {  # sub <afterok|afterany> <description> <sbatch args...>
    local kind="$1" desc="$2"; shift 2
    if [[ -n "${RAFT_DRY_RUN:-}" ]]; then
        n_dry=$((n_dry + 1))
        printf '%-26s sbatch %s%s\n' "$desc" "${dep:+--dependency=$kind:$dep }" "$*"
        dep="dry$n_dry"; return 0
    fi
    local out
    out=$(sbatch --parsable ${dep:+--dependency=$kind:$dep} --export=ALL "$@")
    dep="${out%%;*}"
    printf '%-26s job %s\n' "$desc" "$dep"
}

STAGES=(data smoke sft raft eval)
idx() { local i; for i in "${!STAGES[@]}"; do [[ "${STAGES[$i]}" == "$1" ]] && { echo "$i"; return 0; }; done
        echo "--from must be one of: ${STAGES[*]}" >&2; exit 2; }
from_i="$(idx "$from")"

want() { (( $(idx "$1") >= from_i )); }  # is stage $1 at or after --from?

cd logs
if want data; then sub afterany "prepare data" "$S/01_prepare_data.sbatch"; fi
if (( smoke )) && want smoke; then
    sub afterok "smoke test" "${place[@]}" --gres="gpu:$DEV:${RAFT_SMOKE_GPUS:-2}" "$S/05_smoke.sbatch"
fi
if want sft; then
    sub afterok "sft" "${place[@]}" --gres="gpu:$DEV:$RAFT_GPUS" "$S/02_sft.sbatch"
    # Reference points for the RAFT curve, cheap and independent of what follows.
    sub afterok "eval base+sft" "${place[@]}" --gres="gpu:$DEV:1" --job-name=e-sft "$S/04_eval.sbatch"
fi
# afterany, not afterok: RAFT needs the SFT checkpoint but not the eval that
# follows it, and a failed 20-minute eval should not strand a 3-hour run.
if want raft; then
    sub afterany "raft align" "${place[@]}" --gres="gpu:$DEV:$RAFT_GPUS" "$S/03_raft.sbatch"
fi
if want eval; then
    sub afterok "eval raft" "${place[@]}" --gres="gpu:$DEV:1" --job-name=e-raft \
        --export="ALL,RAFT_EVAL_TAG=raft" "$S/04_eval.sbatch"
fi

echo
echo "queued. watch with:  squeue -u \$USER"
echo "wandb:               https://wandb.ai/<entity>/$RAFT_PROJECT"
