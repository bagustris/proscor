"""Offline tests for proscor/stats.py's cluster-bootstrap helpers."""
import numpy as np
from scipy.stats import kruskal

from proscor import stats


def test_cluster_bootstrap_pearson_point_estimate_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 2 * x + rng.normal(size=200) * 0.1
    clusters = rng.integers(0, 10, size=200)  # 10 clusters, arbitrary membership
    out = stats.cluster_bootstrap_pearson(x, y, clusters, n_boot=200, seed=2)
    assert out["r"] == round(float(np.corrcoef(x, y)[0, 1]), 4)
    assert out["n_items"] == 200
    assert out["n_clusters"] == len(set(clusters.tolist()))
    assert out["ci_lo"] < out["r"] < out["ci_hi"]


def _hierarchical_data(n_clusters: int, seed: int, n: int = 1000):
    """n items split into n_clusters equal clusters, each sharing a random
    cluster-level effect `u_c` added to both x and y -- the same structure a
    speaker random-effect creates in GOP data (a speaker who's just bad at a
    sound produces many correlated error phones)."""
    rng = np.random.default_rng(seed)
    size = n // n_clusters
    cluster_ids = np.repeat(np.arange(n_clusters), size)
    u = rng.normal(size=n_clusters)
    x = u[cluster_ids] + rng.normal(scale=0.3, size=len(cluster_ids))
    y = u[cluster_ids] + rng.normal(scale=0.3, size=len(cluster_ids))
    return x, y, cluster_ids


def test_cluster_bootstrap_pearson_widens_with_fewer_clusters():
    """Same item count (n=1000), same per-item noise -- only the number of
    independent cluster-level draws differs (4 vs. 200). The whole point of
    clustering the bootstrap is that CI width tracks the number of
    independent units, not the number of items; with only 4 units, resampling
    them with replacement is highly variable, so the few-cluster CI must be
    much wider despite having the exact same n."""
    wins = 0
    trials = 10
    for t in range(trials):
        x_few, y_few, c_few = _hierarchical_data(4, seed=1000 + t)
        x_many, y_many, c_many = _hierarchical_data(200, seed=2000 + t)
        out_few = stats.cluster_bootstrap_pearson(x_few, y_few, c_few, n_boot=300, seed=t)
        out_many = stats.cluster_bootstrap_pearson(x_many, y_many, c_many, n_boot=300, seed=t)
        if (out_few["ci_hi"] - out_few["ci_lo"]) > (out_many["ci_hi"] - out_many["ci_lo"]):
            wins += 1
    assert wins == trials  # holds in every trial at this effect size (verified 10/10)


def test_cluster_bootstrap_pearson_one_cluster_per_item_is_stable():
    rng = np.random.default_rng(5)
    x = rng.normal(size=300)
    y = rng.normal(size=300)  # no real relationship
    clusters = np.arange(300)
    out = stats.cluster_bootstrap_pearson(x, y, clusters, n_boot=500, seed=6)
    assert out["ci_lo"] < 0 < out["ci_hi"]  # true r=0 should be inside the CI


def test_cluster_bootstrap_paired_diff_recovers_point_estimates():
    rng = np.random.default_rng(7)
    n = 400
    y = rng.normal(size=n)
    x1 = y + rng.normal(size=n) * 0.1   # strong correlate of y
    x2 = rng.normal(size=n)             # unrelated to y
    clusters = rng.integers(0, 20, size=n)

    out = stats.cluster_bootstrap_paired_diff(x1, x2, y, clusters, n_boot=500, seed=8)
    assert out["r1"] > 0.9
    assert abs(out["r2"]) < 0.2
    assert out["diff"] == round(out["r1"] - out["r2"], 4)
    assert out["significant"] is True
    assert out["diff_ci_lo"] > 0


def test_cluster_bootstrap_paired_diff_not_significant_when_identical():
    """x1 == x2 exactly => r1 == r2 on every possible resample (not just the
    full sample), so the diff CI must collapse to exactly zero width at
    zero -- a deterministic base case, not a statistical one, so no flaky
    seed dependence."""
    rng = np.random.default_rng(9)
    n = 300
    y = rng.normal(size=n)
    x1 = y + rng.normal(size=n) * 0.5
    clusters = rng.integers(0, 15, size=n)

    out = stats.cluster_bootstrap_paired_diff(x1, x1, y, clusters, n_boot=200, seed=10)
    assert out["diff"] == 0.0
    assert out["diff_ci_lo"] == 0.0
    assert out["diff_ci_hi"] == 0.0
    assert out["significant"] is False


def test_kruskal_by_group_matches_scipy_direct_call():
    rng = np.random.default_rng(11)
    values = rng.normal(size=30)
    groups = np.array((["a"] * 10) + (["b"] * 10) + (["c"] * 10))

    out = stats.kruskal_by_group(values, groups)
    h_expected, p_expected = kruskal(values[:10], values[10:20], values[20:])
    assert out["h"] == round(float(h_expected), 4)
    assert out["p"] == float(p_expected)
    assert out["by_group"]["a"]["n"] == 10


def test_kruskal_by_group_detects_real_difference():
    rng = np.random.default_rng(12)
    low = rng.normal(loc=0.0, scale=0.05, size=10)
    high = rng.normal(loc=1.0, scale=0.05, size=10)
    values = np.concatenate([low, high])
    groups = np.array((["low"] * 10) + (["high"] * 10))
    out = stats.kruskal_by_group(values, groups)
    assert out["p"] < 0.01
