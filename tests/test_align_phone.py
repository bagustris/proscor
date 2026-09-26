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
    monkeypatch.setattr(align_phone, "_session", lambda model_id=None, use_int8=True: None)
    monkeypatch.setattr(align_phone, "_phonemize_word",
                         lambda w: {"one": ("a",), "two": ("b", "c")}[w])
    # word "one" -> token a, frames 0-1; blank; word "two" -> b then c
    lp = _lp_for_path([1, 1, blank, 2, blank, 3], T=6, V=4)
    monkeypatch.setattr(align_phone, "_logprobs", lambda samples, model_id=None, use_int8=True: lp)

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
    monkeypatch.setattr(align_phone, "_session", lambda model_id=None, use_int8=True: None)
    monkeypatch.setattr(align_phone, "_phonemize_word", lambda w: {"a": ("a",), "zzz": ("q",)}[w])
    lp = _lp_for_path([1], T=1, V=2)
    monkeypatch.setattr(align_phone, "_logprobs", lambda samples, model_id=None, use_int8=True: lp)

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


# --- segmentation-free GOP (PLAN.md section 5f; Cao et al. 2025) -----------

def _rand_lp(T, V, seed):
    """Log-softmax-shaped but otherwise arbitrary log-probs (not hot/cold
    saturated), so a brute-force enumeration has something nontrivial to
    sum over -- needed for the wildcard-vs-brute-force tests below."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(T, V))
    m = logits.max(-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))


def _safe_candidates(tokens, position, blank, vocab_size):
    """The candidate set _ctc_loglik_wildcard's precondition requires:
    every vocab id except blank and the values immediately flanking
    `position` -- see its docstring for why the skip-transition math needs
    this. Mirrors gop_sf's own exclusion logic exactly."""
    exclude = {blank}
    if position > 0:
        exclude.add(tokens[position - 1])
    if position + 1 < len(tokens):
        exclude.add(tokens[position + 1])
    return np.array([c for c in range(vocab_size) if c not in exclude])


def test_ctc_loglik_wildcard_matches_brute_force_with_one_candidate():
    """logsumexp over a single-element candidate set must reduce to exactly
    that candidate's own _ctc_loglik -- a non-tautological check that the
    position slicing lines up, independent of the multi-candidate lane
    machinery below."""
    blank = 0
    tokens = [1, 2, 3]
    lp = _rand_lp(T=9, V=5, seed=1)

    wildcard_ll = align_phone._ctc_loglik_wildcard(lp, tokens, position=1, blank=blank,
                                                     candidate_ids=np.array([2]))
    assert wildcard_ll == pytest.approx(align_phone._ctc_loglik(lp, tokens, blank), abs=1e-6)


def test_ctc_loglik_wildcard_matches_brute_force_enumeration():
    """The whole point of the per-candidate-lane trick is doing in one
    forward pass what brute force would do in |candidates| separate passes:
    logP of "some candidate phone was produced here" should equal logsumexp
    over every candidate's own logP with that phone substituted in. This is
    the load-bearing correctness proof for gop_sf's substitution term, at
    the FULL candidate count gop_sf actually uses (not a restricted subset
    -- a metric that only checked a handful of candidates wouldn't be
    comparable to the paper's published numbers)."""
    blank = 0
    tokens = [1, 2, 3]
    lp = _rand_lp(T=9, V=5, seed=1)
    candidates = _safe_candidates(tokens, 1, blank, vocab_size=5)

    wildcard_ll = align_phone._ctc_loglik_wildcard(lp, tokens, position=1, blank=blank,
                                                     candidate_ids=candidates)
    brute = [align_phone._ctc_loglik(lp, tokens[:1] + [c] + tokens[2:], blank) for c in candidates]
    from scipy.special import logsumexp
    assert wildcard_ll == pytest.approx(logsumexp(brute), abs=1e-6)


def test_ctc_loglik_wildcard_overcounts_when_candidate_collides_with_neighbor():
    """Pin the bug the precondition exists to avoid: if the candidate set
    DOES include a neighbor's value, skip-transition legality genuinely
    differs per candidate and a shared topology can't represent that --
    the mismatch is large, not a rounding error. This is why gop_sf
    excludes neighbors rather than approximating around it."""
    blank = 0
    tokens = [1, 2, 3]
    lp = _rand_lp(T=9, V=5, seed=1)
    unsafe_candidates = np.array([1, 2, 3, 4])  # includes neighbors 1 and 3

    wildcard_ll = align_phone._ctc_loglik_wildcard(lp, tokens, position=1, blank=blank,
                                                     candidate_ids=unsafe_candidates)
    brute = [align_phone._ctc_loglik(lp, tokens[:1] + [c] + tokens[2:], blank) for c in unsafe_candidates]
    from scipy.special import logsumexp
    assert wildcard_ll != pytest.approx(logsumexp(brute), abs=1e-3)


def test_ctc_loglik_wildcard_matches_brute_force_at_every_position():
    """Same proof, repeated at each position in a longer sequence (which
    includes repeated tokens, so several positions' neighbor-exclusion sets
    differ) and a different seed, so the wildcard_state indexing and the
    per-candidate lanes aren't just accidentally right for one spot."""
    blank = 0
    tokens = [1, 2, 3, 1, 2]
    lp = _rand_lp(T=15, V=5, seed=2)
    from scipy.special import logsumexp

    for pos in range(len(tokens)):
        candidates = _safe_candidates(tokens, pos, blank, vocab_size=5)
        wildcard_ll = align_phone._ctc_loglik_wildcard(lp, tokens, position=pos, blank=blank,
                                                         candidate_ids=candidates)
        brute = [align_phone._ctc_loglik(lp, tokens[:pos] + [c] + tokens[pos + 1:], blank)
                 for c in candidates]
        assert wildcard_ll == pytest.approx(logsumexp(brute), abs=1e-6), f"position {pos}"


def test_gop_sf_matches_manual_sub_del_combination():
    """gop_sf's own combining logic (canonical - logaddexp(sub, del)) should
    match computing the three pieces by hand, at the same full (neighbor-
    excluded) candidate set gop_sf itself builds -- the second half of the
    correctness chain (the first half is the wildcard tests above)."""
    blank = 0
    tokens = [1, 2, 3]
    lp = _rand_lp(T=9, V=5, seed=3)
    vocab_size = 5

    scores = align_phone.gop_sf(lp, tokens, blank, vocab_size)

    canonical_ll = align_phone._ctc_loglik(lp, tokens, blank)
    for pos in range(len(tokens)):
        candidate_ids = _safe_candidates(tokens, pos, blank, vocab_size)
        sub_ll = align_phone._ctc_loglik_wildcard(lp, tokens, pos, blank, candidate_ids)
        del_tokens = tokens[:pos] + tokens[pos + 1:]
        del_ll = align_phone._ctc_loglik(lp, del_tokens, blank)
        expected = canonical_ll - float(np.logaddexp(sub_ll, del_ll))
        assert scores[pos] == pytest.approx(expected, abs=1e-6)


def test_gop_sf_is_never_positive():
    """Structural property, not just an empirical one: the wildcard's
    candidate set always includes the canonical phone itself (it's only
    the immediate neighbors that get excluded), so logP(L_SDI) >=
    logP(L_C) always (summing in a superset can only add mass), making
    GOP_SF = logP(L_C) - logP(L_SDI) <= 0 for every position on any audio
    -- same "0 = ceiling, negative = deficit" convention the existing
    posterior-deficit GOP already uses. tokens here have no adjacent
    repeats, so the known edge case (gop_sf's docstring) doesn't apply."""
    lp = _rand_lp(T=12, V=6, seed=4)
    scores = align_phone.gop_sf(lp, [1, 2, 3, 4], blank=0, vocab_size=6)
    assert all(s is not None and s <= 1e-6 for s in scores)


def test_gop_sf_near_zero_when_phone_is_unambiguous():
    """A phone with no viable competing sound anywhere near its frames (a
    huge, isolated hot spike, cold everywhere else including other vocab
    entries at those frames) should have almost nothing for the wildcard to
    latch onto -- GOP_SF should sit close to its 0 ceiling."""
    blank = 0
    # frames: blank, a(very hot, isolated), blank, b(very hot), blank
    lp = np.full((5, 4), -40.0)
    lp[0, blank] = 0.0
    lp[1, 1] = 0.0
    lp[2, blank] = 0.0
    lp[3, 2] = 0.0
    lp[4, blank] = 0.0
    scores = align_phone.gop_sf(lp, [1, 2], blank=blank, vocab_size=4)
    assert scores[0] == pytest.approx(0.0, abs=0.5)
    assert scores[1] == pytest.approx(0.0, abs=0.5)


def test_gop_sf_strongly_negative_for_confident_substitution():
    """If a different phone (not the canonical one) is confidently produced
    across the frames the canonical phone would need, the wildcard sees
    that alternative as a much better fit -- GOP_SF should be a large
    deficit, not near zero. token 2 (canonical, "b") is never hot anywhere
    in this audio; token 3 dominates its whole slot instead. This is the
    concrete case posterior-deficit GOP (align_words_gop) structurally
    cannot catch the way section 5c/5e's empirical finding shows."""
    blank = 0
    lp = np.full((7, 5), -40.0)
    for t in (0, 1, 2):
        lp[t, 1] = 0.0   # "a" clearly present
    lp[3, blank] = 0.0
    for t in (4, 5, 6):
        lp[t, 3] = 0.0   # something else ("c") clearly present where "b" was expected
    scores = align_phone.gop_sf(lp, [1, 2], blank=blank, vocab_size=5)
    assert scores[0] == pytest.approx(0.0, abs=0.5)   # "a" is unambiguous
    assert scores[1] < -5.0                            # "b" is a confident miss


def test_gop_sf_returns_none_when_audio_too_short():
    lp = _rand_lp(T=1, V=4, seed=5)
    scores = align_phone.gop_sf(lp, [1, 2, 3], blank=0, vocab_size=4)
    assert scores == [None, None, None]


def test_align_words_gop_sf_shape_matches_align_words_gop(monkeypatch):
    """Plumbing test mirroring test_align_words_gop_word_score_is_mean_of_its_phone_scores:
    same word/phone grouping, scored with gop_sf instead of Viterbi posterior-deficit."""
    blank = 0
    monkeypatch.setattr(align_phone, "_TOK2ID", {"a": 1, "b": 2, "c": 3})
    monkeypatch.setattr(align_phone, "_BLANK", blank)
    monkeypatch.setattr(align_phone, "_session", lambda model_id=None, use_int8=True: None)
    monkeypatch.setattr(align_phone, "_phonemize_word",
                         lambda w: {"one": ("a",), "two": ("b", "c")}[w])
    lp = _rand_lp(T=10, V=4, seed=6)
    monkeypatch.setattr(align_phone, "_logprobs", lambda samples, model_id=None, use_int8=True: lp)

    samples = np.zeros(1600, dtype=np.float32)
    result = align_phone.align_words_gop_sf(samples, ["one", "two"], sr=align_phone.SAMPLE_RATE)

    assert [p["phone"] for p in result["phone_gop"][0]] == ["a"]
    assert [p["phone"] for p in result["phone_gop"][1]] == ["b", "c"]


# --- alternate acoustic model backend (PLAN.md section 5i) -----------------

class _FakeTorchModel:
    """Stands in for a transformers Wav2Vec2ForCTC instance: only the
    attributes/calls _session and _logprobs actually touch."""
    def __init__(self, vocab_size):
        self.config = type("Cfg", (), {"vocab_size": vocab_size})()
        self._eval_called = False

    def eval(self):
        self._eval_called = True
        return self


class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_session_selects_onnx_backend_for_default_model_id(monkeypatch):
    """model_id=None (the default) must take the existing ONNX path, not
    the new torch path -- this is the "must not change default behavior"
    guarantee the whole backend refactor depends on."""
    calls = {"onnx": 0, "torch": 0}

    class _FakeOrt:
        class InferenceSession:
            def __init__(self, path):
                calls["onnx"] += 1

    monkeypatch.setattr(align_phone, "_SESSIONS", {})
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", _FakeOrt)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, fname: "/tmp/fake-vocab.json" if fname == "vocab.json" else "/tmp/fake-model.onnx",
    )
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *a, **k: __import__("io").StringIO('{"<pad>": 0, "a": 1}'),
    )
    align_phone._session(None)
    assert calls["onnx"] == 1
    assert align_phone._BACKEND == "onnx"


def test_session_torch_backend_filters_tokens_the_model_cannot_output(monkeypatch):
    """Regression test for a real bug hit building this: the xlsr-53
    checkpoint's tokenizer vocab has one more entry ("|", a word-delimiter
    token) than the model's CTC head actually outputs (393 vocab entries,
    392-class classifier) -- feeding that id into _ctc_viterbi/gop_sf
    indexes the log-prob array out of bounds. _session must drop any
    token whose id is >= the model's own vocab_size."""
    monkeypatch.setattr(align_phone, "_SESSIONS", {})

    fake_vocab = {"<pad>": 0, "a": 1, "b": 2, "|": 3}  # "|" is out of range
    fake_transformers = type("M", (), {
        "Wav2Vec2CTCTokenizer": type("T", (), {
            "from_pretrained": staticmethod(lambda model_id: _FakeTokenizer(fake_vocab))
        }),
        "Wav2Vec2ForCTC": type("F", (), {
            "from_pretrained": staticmethod(lambda model_id: _FakeTorchModel(vocab_size=3))
        }),
    })
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    align_phone._session("some/other-repo")

    assert align_phone._BACKEND == "torch"
    assert "|" not in align_phone._TOK2ID
    assert align_phone._TOK2ID == {"<pad>": 0, "a": 1, "b": 2}
    assert align_phone._BLANK == 0


def test_session_caches_every_model_not_just_the_most_recent(monkeypatch):
    """Regression test for a real bug hit building the model-paired eval
    scripts: a single-slot cache reloads a model's full weights from
    scratch every time the caller alternates between two models within
    one loop (scripts/eval_*_model_paired.py score every utterance with
    BOTH models to compare them) -- observed as a ~3s/utterance rate on
    speechocean762, ~10x the expected combined cost of two already-loaded
    models. `_session` must cache every (model_id, use_int8) it has ever
    loaded, not just swap a single slot, so switching back to a
    previously-used model_id is free."""
    monkeypatch.setattr(align_phone, "_SESSIONS", {})

    calls = []
    fake_transformers = type("M", (), {
        "Wav2Vec2CTCTokenizer": type("T", (), {
            "from_pretrained": staticmethod(lambda model_id: calls.append(model_id) or _FakeTokenizer({"<pad>": 0}))
        }),
        "Wav2Vec2ForCTC": type("F", (), {
            "from_pretrained": staticmethod(lambda model_id: _FakeTorchModel(vocab_size=1))
        }),
    })
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    align_phone._session("repo-a")
    align_phone._session("repo-a")  # same id -> cached, no second load
    align_phone._session("repo-b")  # different id -> must load
    align_phone._session("repo-a")  # switching BACK -> must still be cached, not reloaded
    align_phone._session("repo-b")  # and switching back again -> also still cached

    assert calls == ["repo-a", "repo-b"]  # each repo loaded exactly once, ever

    assert calls == ["repo-a", "repo-b"]


# --- ZIPA backend: char-decomposed IPA vocab (PLAN.md section 5k) ----------

def test_word_phone_spans_onnx_backend_is_one_token_per_phone(monkeypatch):
    """Non-zipa backends: _word_phone_spans must be a thin wrapper around
    _word_phones_and_ids, every span length 1 -- the "must not change
    default behavior" guarantee the whole multi-token-span generalization
    depends on."""
    monkeypatch.setattr(align_phone, "_BACKEND", "onnx")
    monkeypatch.setattr(align_phone, "_TOK2ID", {"k": 1, "æ": 2, "t": 3})
    monkeypatch.setattr(align_phone, "_phonemize_word", lambda w: ("k", "æ", "t"))
    phones, ids, spans = align_phone._word_phone_spans("cat")
    assert phones == ["k", "æ", "t"]
    assert ids == [1, 2, 3]
    assert spans == [(0, 1), (1, 2), (2, 3)]


def test_word_phone_spans_zipa_backend_decomposes_multichar_phone(monkeypatch):
    """zipa backend: a phone espeak emits as one multi-codepoint IPA string
    ("ɑːɹ") must expand into a span covering each of its characters'
    separate token ids, in order -- the char-level vocab ZIPA's tokens.txt
    actually uses (base letter, length mark, rhotic hook each their own
    entry)."""
    monkeypatch.setattr(align_phone, "_BACKEND", "zipa")
    monkeypatch.setattr(align_phone, "_TOK2ID", {"m": 1, "ɑ": 2, "ː": 3, "ɹ": 4, "k": 5})
    monkeypatch.setattr(align_phone, "_phonemize_word", lambda w: ("m", "ɑːɹ", "k"))
    phones, ids, spans = align_phone._word_phone_spans("mark")
    assert phones == ["m", "ɑːɹ", "k"]
    assert ids == [1, 2, 3, 4, 5]
    assert spans == [(0, 1), (1, 4), (4, 5)]


def test_word_phone_spans_zipa_backend_drops_phone_with_no_known_chars(monkeypatch):
    """A phone whose IPA characters are *all* missing from ZIPA's 127-symbol
    vocab is dropped entirely, from both the phone-text and token lists in
    lockstep -- mirroring _word_phones_and_ids's drop-together behavior for
    the other backends, and keeping `phones`/`spans` index-aligned."""
    monkeypatch.setattr(align_phone, "_BACKEND", "zipa")
    monkeypatch.setattr(align_phone, "_TOK2ID", {"k": 1, "æ": 2})
    monkeypatch.setattr(align_phone, "_phonemize_word", lambda w: ("k", "??", "æ"))
    phones, ids, spans = align_phone._word_phone_spans("cat")
    assert phones == ["k", "æ"]
    assert ids == [1, 2]
    assert spans == [(0, 1), (1, 2)]


def test_leading_marker_prefix_only_for_zipa_backend(monkeypatch):
    """The utterance-initial "▁" prefix (see module docstring / PLAN.md
    section 5k) is a zipa-only concept -- a no-op for the other two
    backends, which have no such symbol in their vocab."""
    monkeypatch.setattr(align_phone, "_BACKEND", "zipa")
    monkeypatch.setattr(align_phone, "_TOK2ID", {"▁": 3, "m": 1})
    assert align_phone._leading_marker_prefix() == [3]

    monkeypatch.setattr(align_phone, "_BACKEND", "onnx")
    monkeypatch.setattr(align_phone, "_TOK2ID", {"m": 1})
    assert align_phone._leading_marker_prefix() == []

    monkeypatch.setattr(align_phone, "_BACKEND", "zipa")
    monkeypatch.setattr(align_phone, "_TOK2ID", {"m": 1})  # no "▁" entry
    assert align_phone._leading_marker_prefix() == []


def test_align_words_gop_zipa_multitoken_phone_is_frame_weighted_mean(monkeypatch):
    """End-to-end (stubbed model) check that a multi-token phone's GOP under
    align_words_gop is the frame-weighted mean across its own token/state
    range, not (say) an unweighted mean of per-token means -- the
    generalization align_words_gop needed to support ZIPA's char-decomposed
    phones without changing behavior for the other two backends (see the
    "one token" test above, which confirms this reduces to the old code
    there). Flat sequence (after the leading "▁"): m, a, k -- extended CTC
    states 0..8 = blank,▁,blank,m,blank,a,blank,k,blank. Uses T=10 frames
    (one more than the 9 states) so state 5 ("a") can legally occupy 2
    frames via its self-loop, matching a real multi-frame phone."""
    blank = 0
    monkeypatch.setattr(align_phone, "_BACKEND", "zipa")
    monkeypatch.setattr(align_phone, "_TOK2ID", {"▁": 1, "m": 2, "a": 3, "k": 4})
    monkeypatch.setattr(align_phone, "_BLANK", blank)
    monkeypatch.setattr(align_phone, "_session", lambda model_id=None, use_int8=True: None)
    monkeypatch.setattr(align_phone, "_phonemize_word", lambda w: {"ma": ("m", "ak")}[w])

    V = 5
    lp = np.full((10, V), -30.0)
    # state0 blank, state1 "▁", state2 blank, state3 "m", state4 blank:
    # frames 0-4, each cleanly hot on its own token.
    for t, tok in enumerate([blank, 1, blank, 2, blank]):
        lp[t, tok] = 0.0
    # state5 "a": frames 5-6 (2 frames via self-loop). Frame 5 is
    # deliberately imperfect: some other token (4, "k" -- irrelevant to the
    # true topology at this position, just a distractor) is *this frame's*
    # argmax at 0.0, with "a" present but 5.0 nats behind, so GOP there is
    # a genuine -5.0 deficit (not 0, which is what "a" being its own
    # frame's argmax would give regardless of its absolute log-prob).
    # Frame 6 is clean (no distractor, "a" is the argmax).
    lp[5, 4], lp[5, 3] = 0.0, -5.0
    lp[6, 3] = 0.0
    # state6 blank, state7 "k", state8 blank: frames 7-9, clean.
    for t, tok in enumerate([blank, 4, blank], start=7):
        lp[t, tok] = 0.0
    monkeypatch.setattr(align_phone, "_logprobs", lambda samples, model_id=None, use_int8=True: lp)

    samples = np.zeros(1600, dtype=np.float32)
    result = align_phone.align_words_gop(samples, ["ma"], sr=align_phone.SAMPLE_RATE)

    phone_gop = result["phone_gop"][0]
    assert [p["phone"] for p in phone_gop] == ["m", "ak"]
    assert phone_gop[0]["gop"] == pytest.approx(0.0, abs=1e-6)  # "m": single clean frame
    # "ak" = state5("a", 2 frames: pgop -5.0 and 0.0 -> state mean -2.5) +
    # state7("k", 1 frame: pgop 0.0). Frame-weighted mean over 3 frames:
    # (-2.5*2 + 0.0*1) / 3 = -5/3, NOT a naive mean of (-2.5, 0.0) = -1.25.
    assert phone_gop[1]["gop"] == pytest.approx(-5.0 / 3, abs=1e-6)
    assert phone_gop[1]["n_frames"] == 3


def test_align_words_gop_reports_phone_frame_spans(monkeypatch):
    """`phone_gop` entries carry the [first, last+1) CTC frame range their own
    Viterbi state(s) occupy (used by phone-level DTW, PLAN.md section 5m) --
    checked on a path whose frame allocation is known by construction."""
    blank = 0
    monkeypatch.setattr(align_phone, "_BACKEND", "onnx")
    monkeypatch.setattr(align_phone, "_TOK2ID", {"a": 1, "b": 2, "c": 3})
    monkeypatch.setattr(align_phone, "_BLANK", blank)
    monkeypatch.setattr(align_phone, "_session", lambda model_id=None, use_int8=True: None)
    monkeypatch.setattr(align_phone, "_phonemize_word",
                         lambda w: {"one": ("a",), "two": ("b", "c")}[w])
    # a: frames 0-1, blank: 2, b: 3, blank: 4, c: 5
    lp = _lp_for_path([1, 1, blank, 2, blank, 3], T=6, V=4)
    monkeypatch.setattr(align_phone, "_logprobs", lambda samples, model_id=None, use_int8=True: lp)

    result = align_phone.align_words_gop(np.zeros(1600, dtype=np.float32), ["one", "two"],
                                          sr=align_phone.SAMPLE_RATE)
    assert result["phone_gop"][0][0]["span"] == (0, 2)
    assert [p["span"] for p in result["phone_gop"][1]] == [(3, 4), (5, 6)]
