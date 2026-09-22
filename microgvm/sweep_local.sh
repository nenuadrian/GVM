#!/usr/bin/env bash
# Local sweep. The cluster array does the same thing; this just skips the queue.
# Run names are prefixed "local-" so the two never collide in wandb.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p results

for seed in ${SEEDS:-0 1}; do
  for W in ${WARMUPS:-100 200 400 800 1600}; do
    echo "################ warmup=$W seed=$seed ################"
    python3 run.py --mode estimator \
        --modulus 7 --max-k 4 --n-prompts 512 \
        --batch-prompts 32 --n-base 8 --pilot 4 \
        --gt-rollouts "${GT:-512}" --repeats "${R:-128}" \
        --warmup-steps "$W" --d 128 --n-layer 2 --seed "$seed" \
        --wandb-project "${WANDB_PROJECT:-microgvm}" \
        --run "${RUN_PREFIX:-local}-warmup${W}-seed${seed}" \
        --out "results/local_warmup${W}_seed${seed}.json"
  done
done
echo "SWEEP_DONE"
