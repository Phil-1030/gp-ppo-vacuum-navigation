"""Lightweight statistical pipeline for pairwise policy comparisons.

Welch's t-test, Mann-Whitney U, two-proportion z-test, and bootstrap CIs.
Implemented locally (without scipy.stats) to avoid pulling a new dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ─── Normal CDF helpers ──────────────────────────────────────────────────────

def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_p_normal(z: float) -> float:
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


# ─── Welch's t-test (two-sample, unequal var) ────────────────────────────────

def _welch_t_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan"), float("nan")
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0.0:
        return float("inf") if ma != mb else 0.0, 0.0 if ma != mb else 1.0, ma - mb
    t = (ma - mb) / se
    # Welch–Satterthwaite df
    num = (va / na + vb / nb) ** 2
    denom = ((va / na) ** 2) / (na - 1) + ((vb / nb) ** 2) / (nb - 1)
    df = num / denom if denom > 0 else max(na + nb - 2, 1)
    p = _t_two_sided_p(t, df)
    return float(t), float(p), float(ma - mb)


def _t_two_sided_p(t: float, df: float) -> float:
    """Approximate two-sided p-value for a t-statistic via normal approx for
    df >= 30, else use a Hill-style approximation. Adequate for n_episodes
    on the order of 50-200.
    """
    if df >= 30:
        return _two_sided_p_normal(t)
    # Beta-incomplete-style approximation via numerical integration of the
    # t-density. For our sample sizes df >= ~50 always, so this branch is
    # rarely hit; we use a coarse but bounded fallback.
    x = df / (df + t * t)
    p = _regularised_inc_beta(x, df / 2.0, 0.5)
    return float(p)


def _regularised_inc_beta(x: float, a: float, b: float) -> float:
    # Lentz's algorithm. Translated from Numerical Recipes 6.4 (public-domain
    # equivalent). Adequate for our uses; we only call it with small df.
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Continued fraction
    eps = 1e-12
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin: d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin: c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin: d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin: c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return front * h


# ─── Mann-Whitney U (two-sided, no continuity correction) ────────────────────

def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    combined = np.concatenate([a, b])
    ranks = combined.argsort().argsort().astype(np.float64) + 1.0
    # Average ranks for ties.
    sorted_vals = np.sort(combined)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = ranks[combined.argsort()[i:j + 1]].mean()
            ranks[combined.argsort()[i:j + 1]] = avg
        i = j + 1
    R1 = ranks[:na].sum()
    U1 = R1 - na * (na + 1) / 2
    U2 = na * nb - U1
    U = min(U1, U2)
    mu_U = na * nb / 2
    sigma_U = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    if sigma_U == 0.0:
        return float(U), float("nan")
    z = (U - mu_U) / sigma_U
    return float(U), float(_two_sided_p_normal(z))


# ─── Two-proportion z-test ───────────────────────────────────────────────────

def _two_proportion_z(a_succ: int, a_n: int, b_succ: int, b_n: int) -> tuple[float, float, float]:
    if a_n == 0 or b_n == 0:
        return float("nan"), float("nan"), float("nan")
    p1 = a_succ / a_n
    p2 = b_succ / b_n
    p = (a_succ + b_succ) / (a_n + b_n)
    se = math.sqrt(p * (1 - p) * (1 / a_n + 1 / b_n))
    if se == 0.0:
        return 0.0, 1.0, p1 - p2
    z = (p1 - p2) / se
    return float(z), float(_two_sided_p_normal(z)), float(p1 - p2)


# ─── Bootstrap mean CI ───────────────────────────────────────────────────────

def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


# ─── Public dataclass ────────────────────────────────────────────────────────

@dataclass
class PairwiseStats:
    metric: str
    policy_a: str
    policy_b: str
    mean_a: float
    mean_b: float
    welch_t: float
    welch_p: float
    mw_U: float
    mw_p: float
    mean_diff: float
    bootstrap_lo: float
    bootstrap_hi: float


def compare_distributions(
    a: np.ndarray, b: np.ndarray, *,
    metric: str, policy_a: str, policy_b: str,
    bootstrap_seed: int = 0,
) -> PairwiseStats:
    t, p_t, mean_diff = _welch_t_p(a, b)
    U, p_u = _mann_whitney_u(a, b)
    diffs = []
    rng = np.random.default_rng(bootstrap_seed)
    for _ in range(10_000):
        sa = rng.choice(a, size=len(a), replace=True).mean() if len(a) > 0 else float("nan")
        sb = rng.choice(b, size=len(b), replace=True).mean() if len(b) > 0 else float("nan")
        diffs.append(sa - sb)
    diffs = np.array(diffs, dtype=np.float64)
    lo = float(np.quantile(diffs, 0.025)) if len(diffs) else float("nan")
    hi = float(np.quantile(diffs, 0.975)) if len(diffs) else float("nan")
    return PairwiseStats(
        metric=metric,
        policy_a=policy_a, policy_b=policy_b,
        mean_a=float(np.mean(a)) if len(a) else float("nan"),
        mean_b=float(np.mean(b)) if len(b) else float("nan"),
        welch_t=t, welch_p=p_t,
        mw_U=U, mw_p=p_u,
        mean_diff=mean_diff,
        bootstrap_lo=lo, bootstrap_hi=hi,
    )


def compare_proportions(
    a_succ: int, a_n: int, b_succ: int, b_n: int, *,
    metric: str, policy_a: str, policy_b: str,
) -> PairwiseStats:
    z, p, diff = _two_proportion_z(a_succ, a_n, b_succ, b_n)
    return PairwiseStats(
        metric=metric,
        policy_a=policy_a, policy_b=policy_b,
        mean_a=a_succ / a_n if a_n > 0 else float("nan"),
        mean_b=b_succ / b_n if b_n > 0 else float("nan"),
        welch_t=z, welch_p=p,
        mw_U=float("nan"), mw_p=float("nan"),
        mean_diff=diff,
        bootstrap_lo=float("nan"), bootstrap_hi=float("nan"),
    )
