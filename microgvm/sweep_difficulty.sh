#!/usr/bin/env bash
# Sweep the accept rate via TASK DIFFICULTY rather than warmup steps.
#
# The first sweep varied --warmup-steps, on the assumption that more supervised
# pretraining would raise the accept rate. It barely moved it: p_mean went
# 0.204 -> 0.266 across 100..1600 steps, because a 400K model does not learn
# mod-7 chains that fast. So that sweep measured one narrow, low-p regime and
# said nothing about where allocators actually differ.
#
# Modulus and chain length move difficulty directly: random guessing alone is
# 1/modulus, and each extra operation is another chance to err.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p results

# "modulus:max_k:warmup", easiest first
CONFIGS=${CONFIGS:-"5:2:1500 5:3:1500 7:2:1500 7:3:3000 7:4:1500 11:4:1500"}

for seed in ${SEEDS:-0 1}; do
  for cfg in $CONFIGS; do
    IFS=: read -r MOD MK W <<< "$cfg"
    tag="m${MOD}k${MK}"
    echo "################ modulus=$MOD max_k=$MK warmup=$W seed=$seed ################"
    python3 run.py --mode estimator \
        --modulus "$MOD" --max-k "$MK" --n-prompts 512 \
        --batch-prompts 32 --n-base 8 --pilot 4 \
        --gt-rollouts "${GT:-512}" --repeats "${R:-128}" \
        --warmup-steps "$W" --d 128 --n-layer 2 --seed "$seed" \
        --wandb-project "${WANDB_PROJECT:-microgvm}" \
        --run "diff-${tag}-seed${seed}" \
        --out "results/diff_${tag}_seed${seed}.json"
  done
done
echo "DIFFICULTY_SWEEP_DONE"
