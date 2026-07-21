"""Offline tests for proscor/align.py pure functions (no model, no audio)."""
import numpy as np
import pytest

from proscor import align


VOCAB = {"▁sh": 1, "ip": 2, "▁ship": 3, "i": 4, "p": 5, "▁s": 6, "hip": 7}
BLANK = 8


def test_segmentations_enumerates_all_splits():
    segs = align._segmentations("ship", VOCAB, BLANK)
    assert [3] in segs            # whole-word piece
    assert [1, 2] in segs         # ▁sh + ip
    assert [1, 4, 5] in segs      # ▁sh + i + p
    assert [6, 7] in segs         # ▁s + hip
    for seg in segs:
        assert BLANK not in seg


def test_segmentations_unsegmentable_word():
    assert align._segmentations("zzz", VOCAB, BLANK) == []


def _lp_for_path(path, T, V, hot=0.0, cold=-10.0):
    """Log-probs where frame t strongly prefers token path[t]."""
    lp = np.full((T, V), cold)
    for t, tok in enumerate(path):
        lp[t, tok] = hot
    return lp


def test_ctc_loglik_prefers_matching_sequence():
    blank = 3
    # frames: a a blank b b  (tokens a=1, b=2)
    lp = _lp_for_path([1, 1, blank, 2, 2], T=5, V=4)
    good = align._ctc_loglik(lp, [1, 2], blank)
    bad = align._ctc_loglik(lp, [2, 1], blank)
    assert good > bad + 5


def test_ctc_loglik_repeated_token_needs_blank():
    blank = 3
    # "a a" requires a blank between the two a's on a collapsed path;
    # audio with one is likelier than audio without.
    with_gap = _lp_for_path([1, blank, 1], T=3, V=4)
    without_gap = _lp_for_path([1, 1, 1], T=3, V=4)
    assert align._ctc_loglik(with_gap, [1, 1], blank) > \
        align._ctc_loglik(without_gap, [1, 1], blank) + 5


def test_map_score_bounds_and_monotonicity():
    assert align._map_score(1.0, 0.0) == pytest.approx(100.0)
    assert align._map_score(0.0, -50.0) == pytest.approx(0.0, abs=1e-6)
    # better posterior -> better score; better gop -> better score
    assert align._map_score(0.9, -0.4) > align._map_score(0.2, -0.4)
    assert align._map_score(0.9, -0.1) > align._map_score(0.9, -1.5)


def test_strip_stress():
    assert align._strip_stress(["SH", "IH1", "P"]) == ["SH", "IH", "P"]
