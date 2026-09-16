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


def test_greedy_segmentation_prefers_longest_piece():
    toks = align._greedy_segmentation("ship", VOCAB, BLANK)
    assert toks == [3]  # "▁ship" is in-vocab whole-word, greedy takes it whole


def test_greedy_segmentation_falls_back_to_smaller_pieces():
    vocab = {"▁sh": 1, "ip": 2, "i": 3, "p": 4}
    assert align._greedy_segmentation("ship", vocab, blank=5) == [1, 2]


def test_sentence_tokens_spans_are_contiguous_and_ordered():
    flat, spans = align._sentence_tokens(["ship", "ship"], VOCAB, BLANK)
    assert flat == [3, 3]
    assert spans == [(0, 1), (1, 2)]


def test_ctc_viterbi_recovers_known_best_path():
    blank = 3
    lp = _lp_for_path([1, 1, blank, 2, 2], T=5, V=4)
    path, loglik = align._ctc_viterbi(lp, [1, 2], blank)
    assert path is not None
    # extended seq = [blank, 1, blank, 2, blank] -> states 1 and 3 are the tokens
    assert path[0] == 1 and path[1] == 1  # token 1 (a) for the first two frames
    assert path[3] == 3 and path[4] == 3  # token 2 (b) for the last two frames
    assert loglik > align._NEG / 2


def test_ctc_viterbi_loglik_never_exceeds_forward_loglik():
    """Viterbi is a max over paths; the forward algorithm sums over all paths
    (logsumexp), which can only be >= the max of the same terms."""
    rng = np.random.default_rng(0)
    blank = 4
    lp = np.log(rng.dirichlet(np.ones(5), size=12))  # random (T=12, V=5) log-probs
    tokens = [1, 2, 1, 3]
    _, viterbi_ll = align._ctc_viterbi(lp, tokens, blank)
    forward_ll = align._ctc_loglik(lp, tokens, blank)
    assert viterbi_ll <= forward_ll + 1e-6


def test_ctc_viterbi_reports_failure_when_audio_too_short():
    blank = 3
    lp = _lp_for_path([1], T=1, V=4)
    path, loglik = align._ctc_viterbi(lp, [1, 2], blank)  # needs >= 3 frames, only 1 given
    assert path is None
    assert loglik <= align._NEG / 2


def test_align_words_gop_scores_near_zero_for_perfectly_matching_audio(monkeypatch):
    """End-to-end (still offline): stub out the ONNX session/logprobs so
    align_words_gop's frame-span/GOP bookkeeping is exercised without a
    model file."""
    tok2id = {"▁a": 1, "▁b": 2}
    blank = 0
    monkeypatch.setattr(align, "_session", lambda model_dir=None, use_int8=None: None)
    monkeypatch.setattr(align, "_VOCAB", (tok2id, blank))
    # word "a" -> token 1 for frames 0-1, word "b" -> token 2 for frames 3-4,
    # with blanks elsewhere; each token frame is a confident (near-one-hot) prediction.
    lp = _lp_for_path([1, 1, blank, 2, 2], T=5, V=3, hot=0.0, cold=-30.0)
    monkeypatch.setattr(align, "_logprobs", lambda samples, model_dir=None, use_int8=None: lp)

    samples = np.zeros(1600, dtype=np.float32)
    results = align.align_words_gop(samples, ["a", "b"], sr=align.SAMPLE_RATE)

    assert len(results) == 2
    for r in results:
        assert r is not None
        assert r["gop"] == pytest.approx(0.0, abs=1e-6)  # aligned label *is* the argmax
    assert results[0]["n_frames"] == 2
    assert results[1]["n_frames"] == 2


def test_align_words_gop_returns_none_for_unsegmentable_word(monkeypatch):
    tok2id = {"▁a": 1}
    blank = 0
    monkeypatch.setattr(align, "_session", lambda model_dir=None, use_int8=None: None)
    monkeypatch.setattr(align, "_VOCAB", (tok2id, blank))
    lp = _lp_for_path([1], T=1, V=2)
    monkeypatch.setattr(align, "_logprobs", lambda samples, model_dir=None, use_int8=None: lp)

    samples = np.zeros(1600, dtype=np.float32)
    results = align.align_words_gop(samples, ["a", "zzz"], sr=align.SAMPLE_RATE)
    assert results[0] is not None
    assert results[1] is None
