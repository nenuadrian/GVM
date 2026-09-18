# Does the missing `1/(n_i p_i)` weight matter?

`gvm_estimator_check.py` tests the paper's central claim — that the Proposition 1
allocation reduces gradient-estimator variance at fixed budget — and whether that
reduction survives the way the allocation is actually wired into verl.

Input is real E-step output: `estep_qwen15_smoke_p_G.json` holds the accept rates
`p_i` and gradient norms `G_i` that `em/stage_2_calc_acceptRates_grads.py` measured
for 256 Numina prompts under Qwen2.5-Math-1.5B. So the allocation under test is the
one the pipeline actually produces, not a synthetic one.

```bash
python analysis/gvm_estimator_check.py --data analysis/estep_qwen15_smoke_p_G.json --rho 0.3
```

## Result

Four arms at identical budget `sum(n_i) = 8m`, MSE against the uniform-average
gradient (the EM objective of Algorithm 1 line 8), relative to `uniform+weighted`:

| rho | gvm + weighted (theory) | gvm + unweighted (code) | variance ratio code/theory |
|-----|------------------------|-------------------------|-----------------------------|
| 0.1 | 0.48x                  | 1.27x                   | 2.61x                       |
| 0.3 | 0.47x                  | 1.43x                   | 2.58x                       |
| 0.5 | 0.46x                  | 1.77x                   | 2.52x                       |
| 0.7 | 0.44x                  | 2.37x                   | 2.40x                       |
| 0.9 | 0.41x                  | 3.38x                   | 2.18x                       |

**Proposition 1 works.** With the `1/(n_i p_i)` weight, the allocation cuts MSE to
0.41-0.48x of uniform sampling — a consistent ~2.1-2.4x variance reduction.

**The integration negates it.** Without the weight, which is how verl runs it (see
the trace in the session notes: the actor's `select_keys` never receives `n_i` or
`p_i`), the same allocation is 1.3-3.4x *worse* than plain uniform sampling. Even
ignoring bias entirely and comparing variance alone, the unweighted arm carries
2.2-2.6x the variance of the weighted one.

## What is modelled, and what that does to the conclusion

`p_i` and `G_i` are measured. The per-sample gradient *vector* is not — the pipeline
only ever stores its norm — so it is modelled as `g = mu_i + sigma_i * e` with
`||mu_i|| = rho*G_i`. `rho` is the unknown, hence the sweep; the ordering of the four
arms is unchanged across it.

Two conservative omissions, both of which would widen the gap rather than narrow it:

- Response length is ignored. verl's `token-mean` weights by tokens, and GVM favours
  prompts with longer solutions, so the real weighting is more skewed than modelled.
- Prompts with `p_i = 0` are excluded from the target, since they can never yield an
  accepted sample under any allocation and so are not estimable either way.

Not modelled at all: multiple gradient steps per rollout batch, the KL term, and
PPO clipping. This measures the estimator, not the training run.
