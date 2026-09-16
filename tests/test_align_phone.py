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


# --- reconcile_phones: ARPABET <-> espeak-IPA constrained alignment --------
# All cases below are real speechocean762 words (PLAN.md section 5a item 4);
# `espeak_gops` are distinguishable fake floats (0.0, 0.1, 0.2, ...) so each
# assertion can trace exactly which espeak phone's GOP a merge propagated.

def test_reconcile_rhotic_merge_two_phones():
    """"mark": ARPABET keeps AA0+R separate, espeak fuses them into "ɑːɹ"."""
    gops, ops = align_phone.reconcile_phones(
        ["M", "AA0", "R", "K"], ["m", "ɑːɹ", "k"], [0.0, 0.1, 0.2])
    assert ops == ["match", "merge2", "merge2", "match"]
    assert gops == [0.0, 0.1, 0.1, 0.2]  # AA0 and R both get the "ɑːɹ" GOP


def test_reconcile_rhotic_merge_no_r_phone_in_dataset():
    """"here": CMUdict has no R phone at all (HH IH AH0), but espeak still
    produces one rhotic token ("h ɪɹ") -- the merge rule keys off AH0 acting
    as a rhotic offglide, not off an explicit R phone being present."""
    gops, ops = align_phone.reconcile_phones(
        ["HH", "IH", "AH0"], ["h", "ɪɹ"], [0.0, 0.1])
    assert ops == ["match", "merge2", "merge2"]
    assert gops == [0.0, 0.1, 0.1]


def test_reconcile_syllabic_l_merge():
    """"difficult": espeak's vocab has a dedicated "əl" token for the
    syllabic L that ARPABET spells as two phones, AH0 L."""
    gops, ops = align_phone.reconcile_phones(
        ["D", "IH1", "F", "IH0", "K", "AH0", "L", "T"],
        ["d", "ɪ", "f", "ɪ", "k", "əl", "t"],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert ops == ["match", "match", "match", "match", "match", "merge2", "merge2", "match"]
    assert gops == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6]


def test_reconcile_diphthong_cluster_merge():
    """"idea": espeak treats the final vowel sequence as one complex-nucleus
    token ("iə") where ARPABET keeps two separate vowel symbols (IH AH1)."""
    gops, ops = align_phone.reconcile_phones(
        ["AY0", "D", "IH", "AH1"], ["aɪ", "d", "iə"], [0.0, 0.1, 0.2])
    assert ops == ["match", "match", "merge2", "merge2"]
    assert gops == [0.0, 0.1, 0.2, 0.2]


def test_reconcile_yod_dropping_is_a_deletion_not_a_bad_merge():
    """"new": espeak's en-us applies yod-dropping ("n uː", no /j/) where
    CMUdict's canonical entry keeps the historical glide (N Y UW0). This is
    a genuine dialectal disagreement, not a segmentation artifact -- no
    espeak phone corresponds to the Y, so it must come back as a deletion
    (None), never silently absorbed into a neighbouring merge."""
    gops, ops = align_phone.reconcile_phones(
        ["N", "Y", "UW0"], ["n", "uː"], [0.0, 0.1])
    assert ops == ["match", "del", "match"]
    assert gops == [0.0, None, 0.1]


def test_reconcile_player_known_imperfect_case():
    """"player": ARPABET P L EH1 IH AH0 R vs espeak's "p l eɪ ɚ" needs a
    3-merge of (IH, AH0, R) -> "ɚ", but "ɚ" doesn't start with any IH/IY
    equivalence-class representative, so _merge_compatible correctly
    declines it (see its docstring) rather than guessing. The DP still
    recovers *some* signal via its del/ins fallback instead of discarding
    the whole word (the pre-reconciliation status quo) -- this test pins
    that fallback behavior as known-imperfect, not silently broken: some
    dataset phones get an arbitrary (not semantically precise) substitution
    match rather than their "correct" counterpart, which is an accepted
    trade-off (see the module docstring on reconcile_phones)."""
    gops, ops = align_phone.reconcile_phones(
        ["P", "L", "EH1", "IH", "AH0", "R"], ["p", "l", "eɪ", "ɚ"],
        [0.0, 0.1, 0.2, 0.3])
    assert ops == ["match", "match", "del", "del", "sub", "sub"]  # pins exact current behavior
    assert gops.count(None) == 2  # not a total loss like before reconciliation (was 6/6 None)


def test_reconcile_already_matching_length_is_all_matches():
    """No mismatch -> every phone should resolve via plain 1:1 match, never
    a merge (merges must not fire on words that didn't need them)."""
    gops, ops = align_phone.reconcile_phones(
        ["K", "AE1", "T"], ["k", "æ", "t"], [0.0, 0.1, 0.2])
    assert ops == ["match", "match", "match"]
    assert gops == [0.0, 0.1, 0.2]
