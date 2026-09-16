"""Offline tests for proscor/align_phone.py pure logic (no model, no audio,
no espeak/phonemizer — the ONNX session, log-probs, and phonemizer calls are
all stubbed, matching tests/test_align.py's style for the BPE model)."""
import numpy as np
import pytest

from proscor import align_phone


def test_phonemize_word_is_case_insensitive():
    """Regression test for a real bug: espeak-ng's en-us phonemization
    spells out an all-caps word letter-by-letter when it also reads as a
    plausible acronym ("IT" -> "I-T", "US" -> "U-S"), instead of pronouncing
    it -- speechocean762's transcripts are all-caps, so this silently
    corrupted the target phones for every occurrence of "IT"/"US" (2.14% of
    test-split word occurrences) until _phonemize_word started lowercasing.
    Requires the real phonemizer/espeak-ng (not stubbed, unlike the other
    tests here) -- skipped if unavailable."""
    if not align_phone.available():
        pytest.skip("phonemizer/espeak-ng not installed")
    align_phone._phonemize_word.cache_clear()
    assert align_phone._phonemize_word("IT") == align_phone._phonemize_word("it")
    assert align_phone._phonemize_word("US") == align_phone._phonemize_word("us")
    assert align_phone._phonemize_word("IT") == ("ɪ", "t")


def test_word_phones_and_ids_filters_unknown_phones_together(monkeypatch):
    """A phone espeak emits that isn't in this model's vocab must be dropped
    from *both* the phone-text list and the id list, in lockstep -- an
    earlier version filtered only the id list, which silently desynced the
    two and misattributed later phones to the wrong frames."""
    monkeypatch.setattr(align_phone, "_TOK2ID", {"k": 1, "æ": 2})
    monkeypatch.setattr(align_phone, "_phonemize_word", lambda w: ("k", "??", "æ"))
    phones, ids = align_phone._word_phones_and_ids("cat")
    assert phones == ["k", "æ"]
    assert ids == [1, 2]


def _lp_for_path(path, T, V, hot=0.0, cold=-30.0):
    lp = np.full((T, V), cold)
    for t, tok in enumerate(path):
        lp[t, tok] = hot
    return lp


def test_align_words_gop_word_score_is_mean_of_its_phone_scores(monkeypatch):
    blank = 0
    monkeypatch.setattr(align_phone, "_TOK2ID", {"a": 1, "b": 2, "c": 3})
    monkeypatch.setattr(align_phone, "_BLANK", blank)
    monkeypatch.setattr(align_phone, "_session", lambda use_int8=True: None)
    monkeypatch.setattr(align_phone, "_phonemize_word",
                         lambda w: {"one": ("a",), "two": ("b", "c")}[w])
    # word "one" -> token a, frames 0-1; blank; word "two" -> b then c
    lp = _lp_for_path([1, 1, blank, 2, blank, 3], T=6, V=4)
    monkeypatch.setattr(align_phone, "_logprobs", lambda samples, use_int8=True: lp)

    samples = np.zeros(1600, dtype=np.float32)
    result = align_phone.align_words_gop(samples, ["one", "two"], sr=align_phone.SAMPLE_RATE)

    assert result["word_gop"][0]["gop"] == pytest.approx(0.0, abs=1e-6)
    assert result["word_gop"][1]["gop"] == pytest.approx(0.0, abs=1e-6)
    assert [p["phone"] for p in result["phone_gop"][0]] == ["a"]
    assert [p["phone"] for p in result["phone_gop"][1]] == ["b", "c"]


def test_align_words_gop_unsegmentable_word_returns_none(monkeypatch):
    blank = 0
    monkeypatch.setattr(align_phone, "_TOK2ID", {"a": 1})
    monkeypatch.setattr(align_phone, "_BLANK", blank)
    monkeypatch.setattr(align_phone, "_session", lambda use_int8=True: None)
    monkeypatch.setattr(align_phone, "_phonemize_word", lambda w: {"a": ("a",), "zzz": ("q",)}[w])
    lp = _lp_for_path([1], T=1, V=2)
    monkeypatch.setattr(align_phone, "_logprobs", lambda samples, use_int8=True: lp)

    samples = np.zeros(1600, dtype=np.float32)
    result = align_phone.align_words_gop(samples, ["a", "zzz"], sr=align_phone.SAMPLE_RATE)
    assert result["word_gop"][0] is not None
    assert result["word_gop"][1] is None
    assert result["phone_gop"][1] == []
