"""Cluster-aware correlation statistics for cross-corpus GOP-lite validation.

Phone/word/utterance-level GOP-lite records are not independent draws: they
cluster within speakers (a speaker who mispronounces "TH" produces many
correlated error phones; an utterance's alignment quality affects every word
in it). A naive Fisher-z confidence interval computed on the raw item count
(e.g. n=118,455 phones) massively understates the true uncertainty, which is
bounded by the number of speakers (e.g. 24), not the number of phones. These
helpers resample whole clusters with replacement (a percentile bootstrap)
instead of individual items, so the reported interval reflects the actual
unit of independence.
"""
from collections import defaultdict

import numpy as np


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _cluster_index(clusters: np.ndarray) -> dict:
    idx = defaultdict(list)
    for i, c in enumerate(clusters):
        idx[c].append(i)
    return {c: np.array(v) for c, v in idx.items()}


def cluster_bootstrap_pearson(x, y, clusters, n_boot: int = 2000, seed: int = 0,
                               ci: float = 0.95) -> dict:
    """Percentile bootstrap CI for Pearson r, resampling clusters (e.g.
    speakers) with replacement rather than individual items.

    Returns the point estimate on the full (unresampled) sample plus a
    cluster-level CI; `n_items` and `n_clusters` are both reported since the
    CI width is governed by the latter, not the former.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)
    idx_by_cluster = _cluster_index(clusters)
    unique = np.array(list(idx_by_cluster))

    r_point = _pearson(x, y)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in sampled])
        r = _pearson(x[idx], y[idx])
        if not np.isnan(r):
            boots.append(r)
    boots = np.array(boots)
    alpha = (1 - ci) / 2
    lo, hi = (np.quantile(boots, [alpha, 1 - alpha]) if len(boots) else (float("nan"), float("nan")))

    return {
        "r": round(r_point, 4) if not np.isnan(r_point) else None,
        "ci_lo": round(float(lo), 4) if not np.isnan(lo) else None,
        "ci_hi": round(float(hi), 4) if not np.isnan(hi) else None,
        "ci_level": ci,
        "n_items": int(len(x)),
        "n_clusters": int(len(unique)),
        "n_boot_valid": int(len(boots)),
    }


def cluster_bootstrap_paired_diff(x1, x2, y, clusters, n_boot: int = 2000, seed: int = 0,
                                   ci: float = 0.95) -> dict:
    """Percentile bootstrap CI for r(x1, y) - r(x2, y) on the SAME items and
    clusters (e.g. two GOP engines scored against the same human labels), so
    the within-cluster item correspondence is preserved on every resample.
    This is the right test for an "engine A beats engine B" claim -- comparing
    two separately-bootstrapped marginal CIs for overlap is weaker and can
    both under- and over-state significance when x1/x2 are correlated (which
    two GOP engines scoring the same audio always are).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)
    idx_by_cluster = _cluster_index(clusters)
    unique = np.array(list(idx_by_cluster))

    r1_point = _pearson(x1, y)
    r2_point = _pearson(x2, y)
    diff_point = (r1_point - r2_point) if not (np.isnan(r1_point) or np.isnan(r2_point)) else float("nan")

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in sampled])
        r1 = _pearson(x1[idx], y[idx])
        r2 = _pearson(x2[idx], y[idx])
        if not (np.isnan(r1) or np.isnan(r2)):
            diffs.append(r1 - r2)
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    lo, hi = (np.quantile(diffs, [alpha, 1 - alpha]) if len(diffs) else (float("nan"), float("nan")))
    significant = bool(lo > 0 or hi < 0) if not (np.isnan(lo) or np.isnan(hi)) else None

    return {
        "r1": round(r1_point, 4) if not np.isnan(r1_point) else None,
        "r2": round(r2_point, 4) if not np.isnan(r2_point) else None,
        "diff": round(diff_point, 4) if not np.isnan(diff_point) else None,
        "diff_ci_lo": round(float(lo), 4) if not np.isnan(lo) else None,
        "diff_ci_hi": round(float(hi), 4) if not np.isnan(hi) else None,
        "ci_level": ci,
        "significant": significant,
        "n_items": int(len(x1)),
        "n_clusters": int(len(unique)),
        "n_boot_valid": int(len(diffs)),
    }


def kruskal_by_group(values, groups) -> dict:
    """Kruskal-Wallis H-test on `values` (e.g. per-speaker point-biserial r's)
    partitioned by `groups` (e.g. each speaker's L1) -- the test for "do the
    groups differ", with `values` at the cluster level (one r per speaker),
    not the raw item level.
    """
    from scipy.stats import kruskal

    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique_groups = sorted(set(groups.tolist()))
    samples = [values[groups == g] for g in unique_groups]
    h, p = kruskal(*samples)
    return {
        "h": round(float(h), 4),
        "p": float(p),
        "by_group": {
            g: {"n": int(len(s)), "mean": round(float(np.mean(s)), 4),
                "median": round(float(np.median(s)), 4)}
            for g, s in zip(unique_groups, samples)
        },
    }
