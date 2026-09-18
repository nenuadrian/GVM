"""Does GVM's allocation actually reduce gradient-estimator variance, and does
that depend on the 1/(n_i p_i) weight the implementation omits?

Uses the real accept rates p_i and gradient norms G_i measured by the E-step
(em/stage_2_calc_acceptRates_grads.py) on Qwen2.5-Math-1.5B, so the allocation
under test is the one the pipeline actually produces. What has to be modelled is
the per-sample gradient vector itself, which the pipeline only ever reduces to a
norm -- see `simulate` for that model and `--rho` for the sensitivity sweep.

Four arms, all at identical total budget sum(n_i) = N:

    uniform  + weighted    n_i = N/m,  estimator (1/m) sum_i S_i/(n_i p_i)
    uniform  + unweighted  n_i = N/m,  estimator sum_i S_i / sum_i A_i
    gvm      + weighted    Proposition 1 allocation, weighted   <- the theory
    gvm      + unweighted  Proposition 1 allocation, unweighted <- the code

where S_i is the sum of accepted gradients for prompt i and A_i ~ Bin(n_i, p_i)
is how many were accepted.
"""

import argparse
import json

import numpy as np


def gvm_allocation(p, G, budget, alpha=1e-3, beta=2.0):
    """Proposition 1, verbatim from em/stage_2_calc_sample_size.py:83."""
    ratio = np.where(
        (G == 0) | (p == 0),
        0.0,
        G / np.sqrt(p + alpha / np.power(np.where(p > 0, p, 1.0), beta - 1.0)),
    )
    ratio = ratio / ratio.sum()
    # float_to_int_preserve_sum, same as the pipeline
    scaled = ratio * budget
    n = np.round(scaled).astype(int)
    err = budget - n.sum()
    if err != 0:
        resid = scaled - n
        idx = np.argsort(-resid if err > 0 else resid)[: abs(err)]
        n[idx] += int(np.sign(err))
    return np.clip(n, 0, None)


def simulate(p, G, n, rho, d, reps, rng):
    """Draw `reps` replicates of the estimator under allocation `n`.

    Per-sample gradient g = mu_i + sigma_i * e,  e ~ N(0, I_d / d), so
    E||g||^2 = ||mu_i||^2 + sigma_i^2. The measured G_i is split into a
    systematic part ||mu_i|| = rho*G_i and a noise part sigma_i = G_i*sqrt(1-rho^2);
    rho is what the pipeline cannot tell us, hence the sweep.
    """
    m = len(p)
    direction = rng.normal(size=(m, d))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    mu = (rho * G)[:, None] * direction          # (m, d)
    sigma = G * np.sqrt(1.0 - rho**2)            # (m,)

    # p_i = 0 prompts can never produce an accepted sample, so they contribute
    # nothing under any allocation; excluding them keeps 1/(n_i p_i) finite.
    live = (n > 0) & (p > 0)
    A = np.zeros((reps, m), dtype=float)
    A[:, live] = rng.binomial(n[live], p[live], size=(reps, live.sum()))

    # S_i = A_i*mu_i + sigma_i*sqrt(A_i)*e,  e ~ N(0, I_d/d)
    noise = rng.normal(size=(reps, m, d)) / np.sqrt(d)
    S = A[:, :, None] * mu[None, :, :] + (sigma[None, :, None] * np.sqrt(A)[:, :, None]) * noise

    est = {}
    # Theory (Lemma 1): weight each prompt by 1/(n_i p_i) -> equal weight per prompt.
    w = np.zeros(m)
    w[live] = 1.0 / (n[live] * p[live])
    est["weighted"] = (S * w[None, :, None]).sum(axis=1) / m
    # Code (token-mean): flat mean over accepted samples -> weight proportional to A_i.
    tot = A.sum(axis=1, keepdims=True)
    est["unweighted"] = S.sum(axis=1) / np.maximum(tot, 1.0)
    return est, mu


def report(name, draws, target):
    bias = draws.mean(axis=0) - target
    var = draws.var(axis=0).sum()          # trace of covariance
    mse = var + float(bias @ bias)
    return dict(name=name, var=var, bias2=float(bias @ bias), mse=mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="json with real p and G arrays")
    ap.add_argument("--rho", type=float, default=0.3, help="signal fraction of G_i")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = json.load(open(args.data))
    p = np.asarray(raw["p"], dtype=float)
    G = np.asarray(raw["G"], dtype=float)
    m = len(p)
    budget = 8 * m
    rng = np.random.default_rng(args.seed)

    n_uni = np.full(m, budget // m)
    n_gvm = gvm_allocation(p, G, budget)

    print(f"prompts={m}  budget={budget}  (uniform n_i=8)")
    print(f"gvm: sum={n_gvm.sum()}  max={n_gvm.max()}  zeros={(n_gvm == 0).sum()}")
    stranded = int(((p > 0) & (n_gvm == 0)).sum())
    print(f"prompts with p_i>0 that GVM gives zero budget: {stranded}")
    print()

    # A prompt with p_i = 0 can never yield an accepted sample under ANY allocation,
    # so it is outside the estimable set for both arms and is excluded from the target.
    target_set = p > 0
    rows = []
    for label, n in (("uniform", n_uni), ("gvm", n_gvm)):
        est, mu = simulate(p, G, n, args.rho, args.dim, args.reps, rng)
        target = mu[target_set].sum(axis=0) / m
        for kind in ("weighted", "unweighted"):
            rows.append(report(f"{label:8s} + {kind}", est[kind], target))

    base = next(r for r in rows if r["name"].startswith("uniform  + weighted"))
    print(f"{'arm':26s} {'variance':>12s} {'bias^2':>12s} {'MSE':>12s} {'MSE vs base':>12s}")
    for r in rows:
        print(f"{r['name']:26s} {r['var']:12.4g} {r['bias2']:12.4g} {r['mse']:12.4g}"
              f" {r['mse']/base['mse']:11.2f}x")
    print(f"\n(rho={args.rho}, d={args.dim}, reps={args.reps}; "
          f"'base' = uniform+weighted, the pre-GVM unbiased estimator)")


if __name__ == "__main__":
    main()
