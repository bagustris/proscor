"""Offline tests for proscor/dtw_ssl.py's pure-numpy DTW logic (no model,
no audio -- WavLM embedding extraction itself isn't unit-tested here,
matching this project's convention of not exercising real model downloads
in the offline test suite; see tests/test_align_phone.py's equivalent
scope note)."""
import numpy as np
import pytest

from proscor import dtw_ssl


def test_dtw_cost_identical_sequences_is_zero():
    """Two identical (already-unit-norm) sequences must align perfectly
    along the diagonal with zero cosine distance everywhere -> cost 0."""
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert dtw_ssl.dtw_cost(a, a.copy()) == pytest.approx(0.0, abs=1e-6)


def test_dtw_cost_normalizes_by_path_length_not_sum_of_lengths():
    """Regression-shaped test for the exact bug the module docstring warns
    against: normalizing by `T1+T2` instead of the backtracked path's own
    length. With T1=1 (a single frame [1,0]) and T2=2 (template frames
    [1,0] then [0,1]), the only legal path is forced through both
    template frames (i stays 0, j goes 0->1): costs are dist[0,0]=0
    (identical) and dist[0,1]=1 (orthogonal), summing to 1, over a
    2-step path. Normalizing by path length gives 1/2=0.5; normalizing by
    T1+T2=3 would give 1/3 instead -- these are different enough (0.5 vs
    0.333) that either bug would fail this assertion."""
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert dtw_ssl.dtw_cost(a, b) == pytest.approx(0.5, abs=1e-6)


def test_dtw_cost_is_symmetric():
    """DTW's cost matrix and path should be identical whether `a` is the
    learner or the template -- the algorithm has no inherent direction,
    only the transposed distance matrix, which shouldn't change the
    optimal path's cost."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 4)).astype(np.float32)
    b = rng.normal(size=(7, 4)).astype(np.float32)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    assert dtw_ssl.dtw_cost(a, b) == pytest.approx(dtw_ssl.dtw_cost(b, a), abs=1e-6)


def test_dtw_cost_orthogonal_sequences_cost_one_per_frame():
    """Maximally dissimilar (orthogonal) unit vectors give cosine distance
    1.0 at every cell, so any path -- diagonal or not -- costs exactly
    (path length) * 1.0; for two same-length sequences the diagonal path
    (length = T) gives cost 1.0 after normalization."""
    a = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    assert dtw_ssl.dtw_cost(a, b) == pytest.approx(1.0, abs=1e-6)


def test_score_aggregates_mean_and_min_over_templates():
    learner = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    close_template = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # cost 0
    far_template = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)     # cost 1

    result = dtw_ssl.score(learner, [close_template, far_template])
    assert result["n_templates"] == 2
    assert result["min"] == pytest.approx(0.0, abs=1e-6)
    assert result["mean"] == pytest.approx(0.5, abs=1e-6)
    assert sorted(result["costs"]) == pytest.approx([0.0, 1.0], abs=1e-6)
