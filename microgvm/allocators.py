"""Rollout budget allocators, plus the oracle that none of them can see.

    uniform   n_i = C/m
    gvm       n_i proportional to G_i / sqrt(p_i + alpha/p_i^(beta-1))
              arXiv:2505.02391 Prop. 1; p_i and G_i MEASURED by a pilot pass.
    vip       minimise sum_i a_i (n_i-1)/n_i^2 s.t. sum n_i = C, L<=n_i<=U,
              a_i = 4 sigma^2 p_i (1-p_i).  arXiv:2602.01601 Thm 5.1;
              p_i PREDICTED by a GP, no pilot pass.
    neyman    n_i proportional to the MEASURED standard deviation of prompt i's
              per-rollout gradient contribution.

The last one is the reference this testbed exists to provide. For the per-prompt
estimator, Var = (1/m^2) sum_i sigma_i^2 / n_i, and minimising that under
sum n_i = C gives n_i proportional to sigma_i -- classic Neyman allocation. It is
not a usable method (estimating sigma_i costs far more rollouts than it saves);
it is a yardstick, turning "does GVM beat uniform?" into "how much of the
achievable reduction does GVM capture?" -- a number unobtainable at 1.5B.

It is a strong reference, not a true oracle, for two reasons worth keeping in
mind when reading the results. The advantage baseline r - rbar is estimated from
the same n_i draws, so its noise depends on n_i in a way the Neyman derivation
(which assumes a fixed baseline) does not model; and sigma_i is itself measured
with finite rollouts, so at small --gt-rollouts it is noisy enough to lose to a
good heuristic. Raise --gt-rollouts until it stops moving before trusting it.
"""

import numpy as np

__all__ = ["round_preserving_sum", "uniform_alloc", "gvm_alloc", "vip_alloc",
           "neyman_alloc", "PromptSuccessGP", "alloc_stats"]


def round_preserving_sum(x, total, lo=0, hi=None):
    x = np.asarray(x, dtype=float)
    n = np.round(x).astype(int)
    n = np.clip(n, lo, hi if hi is not None else np.iinfo(np.int64).max)
    resid = x - n
    while n.sum() != total:
        err = total - n.sum()
        step = 1 if err > 0 else -1
        room = (n < hi) if (step > 0 and hi is not None) else (n > lo) if step < 0 else np.ones_like(n, bool)
        if not room.any():
            raise ValueError(f"cannot reach {total} within [{lo},{hi}]")
        for i in np.argsort(-resid if step > 0 else resid):
            if room[i]:
                n[i] += step; resid[i] -= step; break
    return n


def uniform_alloc(m, budget, lo=0, hi=None):
    return round_preserving_sum(np.full(m, budget / m), budget, lo, hi)


def _proportional(w, budget, lo, hi):
    w = np.asarray(w, dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return uniform_alloc(len(w), budget, lo, hi)
    return round_preserving_sum(w / w.sum() * budget, budget, lo, hi)


def gvm_alloc(p, G, budget, alpha=1e-3, beta=2.0, lo=0, hi=None):
    p, G = np.asarray(p, float), np.asarray(G, float)
    safe = np.where(p > 0, p, 1.0)
    w = G / np.sqrt(safe + alpha / np.power(safe, beta - 1.0))
    return _proportional(np.where((p <= 0) | (G <= 0), 0.0, w), budget, lo, hi)


def neyman_alloc(sigma, budget, lo=0, hi=None):
    """Variance-minimising allocation given measured per-prompt std."""
    return _proportional(np.asarray(sigma, float), budget, lo, hi)


def _vip_n_of_lambda(a, lam, lo, hi):
    f = lambda n: a * (n - 2.0) / n ** 3      # stationarity of a(n-1)/n^2
    if a <= 0:
        return float(lo)
    if lam <= f(hi):
        return float(hi)
    if lam >= f(lo):
        return float(lo)
    n_lo, n_hi = float(lo), float(hi)
    for _ in range(60):                        # f is decreasing for n > 3
        mid = 0.5 * (n_lo + n_hi)
        if f(mid) > lam:
            n_lo = mid
        else:
            n_hi = mid
    return 0.5 * (n_lo + n_hi)


def vip_alloc(p_hat, budget, lo=3, hi=None, sigma=None):
    p_hat = np.asarray(p_hat, float)
    m = len(p_hat)
    hi = budget if hi is None else hi
    if not (m * lo <= budget <= m * hi):
        raise ValueError(f"need m*L <= C <= m*U: {m}*{lo} <= {budget} <= {m}*{hi}")
    sig = np.ones(m) if sigma is None else np.asarray(sigma, float)
    a = 4.0 * sig ** 2 * p_hat * (1.0 - p_hat)
    lam_hi = max(float(np.max(a * (lo - 2.0) / lo ** 3)), 1e-12)
    lam_lo = 0.0
    for _ in range(80):                        # bisect the KKT multiplier
        lam = 0.5 * (lam_lo + lam_hi)
        if sum(_vip_n_of_lambda(ai, lam, lo, hi) for ai in a) > budget:
            lam_lo = lam
        else:
            lam_hi = lam
    lam = 0.5 * (lam_lo + lam_hi)
    return round_preserving_sum([_vip_n_of_lambda(ai, lam, lo, hi) for ai in a], budget, lo, hi)


class PromptSuccessGP:
    """VIP's GP over prompt features (arXiv:2602.01601 s5.1).

    p = sigmoid(g(x)), g ~ GP(m, RBF). Recursive posterior update; the full m x m
    kernel is never formed, only the m x |B| and |B| x |B| blocks.
    """

    def __init__(self, feats, bandwidth=None, eps=0.05, jitter=1e-4):
        X = np.asarray(feats, float)
        self.X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        self.n, self.eps, self.jitter = len(X), float(eps), float(jitter)
        if bandwidth is None:
            d2 = ((self.X[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)
            med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
            bandwidth = float(np.sqrt(max(med, 1e-9) / 2.0))
        self.h = float(bandwidth)
        self.m = np.zeros(self.n)
        self.updates = 0

    def _k(self, A, B):
        return np.exp(-((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / (2 * self.h ** 2))

    def predict(self, idx):
        return 1.0 / (1.0 + np.exp(-self.m[np.asarray(idx, int)]))

    def update(self, idx, realised_p):
        idx = np.asarray(idx, int)
        p = np.clip(np.asarray(realised_p, float), self.eps, 1 - self.eps)
        g = np.log(p / (1 - p))
        Xb = self.X[idx]
        K = self._k(Xb, Xb) + self.jitter * np.eye(len(idx))
        try:
            alpha = np.linalg.solve(K, g - self.m[idx])
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K, g - self.m[idx], rcond=None)[0]
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.m = self.m + self._k(self.X, Xb) @ alpha
        self.m[idx] = g
        bound = abs(np.log((1 - self.eps) / self.eps)) * 2.0
        self.m = np.clip(np.nan_to_num(self.m, nan=0.0), -bound, bound)
        self.updates += 1


def alloc_stats(n, p=None, prefix="alloc"):
    n = np.asarray(n, float)
    share = n / max(n.sum(), 1.0)
    out = {f"{prefix}/min": float(n.min()), f"{prefix}/max": float(n.max()),
           f"{prefix}/median": float(np.median(n)),
           f"{prefix}/frac_zero": float((n == 0).mean()),
           f"{prefix}/concentration": float((share ** 2).sum()),
           f"{prefix}/uniform_concentration": 1.0 / len(n)}
    if p is not None:
        p = np.asarray(p, float)
        out.update({f"{prefix}/p_mean": float(p.mean()),
                    f"{prefix}/p_spread": float(p.std()),
                    f"{prefix}/p_frac_degenerate": float(((p <= 0) | (p >= 1)).mean())})
    return out
