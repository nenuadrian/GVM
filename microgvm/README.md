# microgvm — measuring what GVM and VIP actually do

A ~400K-parameter transformer on a verifiable CoT task, small enough that the
**true gradient is computable**. That single property is the point: at 1.5B you
can only judge an allocator by downstream accuracy, which confounds variance
reduction with everything else. Here you measure the estimator directly.

## The task

Modular arithmetic chains. `a1 op a2 op ... op ak =`, the model emits the `k`
running partial results, reward is 1 if the last one is right. This is literally
the paper's latent-variable setup: `z` = reasoning steps, `y` = answer, and
rejection sampling on correctness is the E-step.

Two knobs that move **independently**:

- `k` — number of operations, so output length
- modulus / op mix — difficulty at fixed `k`

Sequence reversal, the obvious alternative, ties difficulty and length to the
same variable. That would make a length effect and a difficulty effect
indistinguishable — and one of GVM's issues is precisely that `G_i` sums over
response tokens and so partly measures length.

## What is compared

All allocators get the **same budget** `C = m * n_base`:

| allocator | `p_i` from | allocation |
|---|---|---|
| `uniform` | — | `C/m` |
| `gvm` | measured, `N'` pilot rollouts | `∝ G_i / sqrt(p_i + α/p_i^(β-1))` (Prop. 1) |
| `vip` | predicted by a GP | minimise `Σ a_i (n_i-1)/n_i²` (Thm 5.1) |
| `neyman` | — | `∝ σ_i`, measured per-prompt noise |

crossed with two weightings of the per-prompt contributions:

| weighting | what it is |
|---|---|
| `per_prompt` | each prompt contributes its own mean, prompts averaged equally — GVM Algorithm 1 line 8 |
| `per_sample` | flat mean over all rollouts, so a prompt's weight grows with its budget — **what verl's `token-mean` does** |

`gvm|per_sample` is the released code. `gvm|per_prompt` is the paper's estimator.
The gap between them is the mechanism question.

## Running

```bash
sbatch microgvm/run_cpu.sbatch                  # one config
sbatch --array=0-4 microgvm/run_cpu.sbatch      # sweep the accept-rate spread
```

The array sweeps `--warmup-steps`, which sets how much supervised pretraining the
model gets and therefore the spread of accept rates. **That sweep is the question
no LM experiment can answer cheaply**: at what `p` spread does any allocator beat
uniform? Our GSM8K runs sat at ~2% accuracy, where 83% of prompts are at `p=0`
and every allocator collapses to the same thing.

Locally, a fast sanity run:

```bash
cd microgvm && python run.py --modulus 5 --max-k 3 --n-prompts 64 \
    --batch-prompts 8 --n-base 6 --gt-rollouts 64 --repeats 24 \
    --warmup-steps 120 --d 64
```

## Reading the output

`mse = bias² + variance`, relative to `uniform|per_prompt`. Lower is better.
`total rollouts` includes the pilot pass, which the budget `C` does not — so
compare on that column, not on `C`.

The bias term subtracts `var/R`: `||mean − truth||²` over `R` repeats
overestimates bias² by exactly that, and at small `R` the correction dominates.
Both weightings are formed from the **same** rollouts, so the comparison between
them is paired rather than confounded by sampling noise.

## Caveats worth keeping

`neyman` is a strong reference, **not a true oracle**. It is optimal for a fixed
baseline, but the advantage here is `r - rbar` with `rbar` estimated from the same
`n_i` draws, so its noise depends on `n_i` in a way the derivation does not model.
And `σ_i` is itself measured with finite rollouts — at small `--gt-rollouts` it is
noisy enough to lose to a good heuristic. Raise `--gt-rollouts` until it stops
moving before trusting it.

A 400K model on modular arithmetic has different `G`/`p` structure than a 1.5B
model on Numina. This rejects bad ideas in minutes and answers mechanism
questions that are structurally unanswerable at scale. It does not replace the
cluster runs.

## What it makes cheap to try

- **Reusing the pilot rollouts.** The GVM authors wrote this
  (`em/stage_2_sample.py`, drawing only `n_i − N'` new samples) and never wired
  it up. It roughly halves GVM's cost.
- **`G_i` as RMS instead of mean-of-norms** — the Jensen gap between the paper's
  Section 3.1 definition and its Algorithm 2.
- **`G_i` summed vs averaged over tokens** — does the allocation track difficulty
  or just length? Hold `k` fixed to find out.
- **Sequential allocation** — draw rollouts one at a time, stop when a prompt's
  estimate is precise enough, instead of a fixed two-stage split.
- **Token-budget rather than rollout-budget** allocation, since long chains cost
  more.
