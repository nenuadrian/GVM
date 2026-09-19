# microgvm results

24 runs: 6 difficulty configs x 4 seeds, accept rate `p` spanning 0.22-0.73.
Each compares four allocators at **equal budget** `C = 32 x 8`, crossed with two
weightings, against a gradient measured from 512 rollouts per prompt.

Medians, not means: `gvm|per_sample` has a median of 4.54x and a mean of 157x
because one run hit 3302x. The means are outlier theatre.

## MSE relative to `uniform|per_prompt` (median of 24)

| arm | median | mean |
|---|---|---|
| `neyman\|per_prompt` | **0.53** | 0.49 |
| `gvm\|per_prompt` | **0.85** | 0.74 |
| `vip\|per_prompt` | 0.93 | 0.94 |
| `uniform\|per_prompt` | 1.00 | 1.00 |
| `uniform\|per_sample` | 1.00 | 1.00 |
| `vip\|per_sample` | 1.28 | 1.48 |
| `gvm\|per_sample` | **4.54** | 156.94 |
| `neyman\|per_sample` | 6.57 | 82.87 |

## 1. GVM's allocation works

`gvm|per_prompt` at 0.85x beats uniform. Proposition 1 does what it claims.

This **reverses** what the earlier warmup sweep showed (1.16x, worse than
uniform). That sweep was confined to `p_mean` 0.20-0.27 because warmup steps
turned out not to move the accept rate. Sweeping task difficulty instead reaches
`p` up to 0.73, and GVM improves as `p` rises:

| regime | `gvm\|per_prompt` | `neyman\|per_prompt` | share of available reduction captured |
|---|---|---|---|
| `p < 0.30` (n=11) | 0.90x | 0.56x | 23% |
| `p >= 0.30` (n=13) | 0.77x | 0.47x | 42% |

`corr(p_mean, gvm MSE) = -0.23`. The paper operates near `p ~ 0.4`, which is
where GVM looks best — and explains why a low-`p` regime showed nothing.

## 2. But it leaves most of the gain on the table

Neyman reaches 0.53x, so roughly half the variance is reducible at this budget.
GVM captures 23-42% of that. The analytic surrogate
`G_i / sqrt(p_i + alpha/p_i^(beta-1))` is a worse proxy for per-prompt noise than
the noise itself, which is unsurprising but had never been quantified.

## 3. The weighting destroys it

Paired per run, `per_sample` MSE divided by `per_prompt` MSE (median):

| allocator | ratio | worse in |
|---|---|---|
| `uniform` | **1.00x** | 0/24 |
| `vip` | 1.35x | 22/24 |
| `gvm` | **5.71x** | 23/24 |
| `neyman` | 12.22x | 24/24 |

`uniform` at exactly 1.00x in **24/24** is the control: with flat `n_i` the two
weightings are the same estimator, so anything else would be a bug.

The damage scales with how unevenly the allocator spends: uniform (flat) < vip
(bounded by L/U) < gvm (unbounded) < neyman (unbounded). That is the mechanism,
not a coincidence -- the weighting error grows with the spread in `n_i`, so **the
better the allocator, the more `token-mean` costs**.

For GVM this turns a 0.85x win into a 4.54x loss: the released configuration is
worse than not allocating at all.

## 4. VIP is roughly free but roughly flat

`vip|per_prompt` 0.93x, with no pilot pass at all -- GVM pays `N' x m` extra
rollouts for its 0.85x. On total-rollout cost VIP is ahead. Its bounded `[L, U]`
allocation also makes it far more robust to the weighting bug (1.35x vs 5.71x),
which is a real practical advantage of the constraint.

## Caveats

- 400K params on modular arithmetic. `G`/`p` structure differs from a 1.5B model
  on Numina; this answers mechanism questions, not transfer.
- `neyman` is a strong reference, not an oracle: optimal for a fixed baseline,
  while the advantage here is `r - rbar` estimated from the same draws.
- Estimator quality, not downstream accuracy. A noisier estimator that happens to
  act as a curriculum could still train better.
