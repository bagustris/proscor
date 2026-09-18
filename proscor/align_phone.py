"""Phone-level CTC forced-alignment GOP using a wav2vec2 phoneme-CTC ONNX
model (`onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX`, an ONNX export of
`facebook/wav2vec2-lv-60-espeak-cv-ft`, espeak-ng IPA output) — the
literature-comparable headline metric that `proscor/align.py`'s word-level
BPE proxy cannot produce, since that model's vocab is BPE subwords, not
phones (see PLAN.md section 5a).

Heavier than `align.py`'s NeMo CTC model (wav2vec2-large: ~1.2GB fp32 /
~320MB int8, downloaded from the Hub on first use, not shipped in `models/`)
and **not wired into the CLI/web app** — an evaluation-only path used by
`scripts/eval_so762_phone.py`.

Targets are generated with the `phonemizer` package's espeak-ng backend (the
same tool that produced this model's training labels), not a hand-built
ARPABET->IPA table: a first attempt at a static table showed espeak merges
vowel+R into single rhotic tokens (e.g. "ɑːɹ" for the vowel in "mark", not
separate "ɑː"+"ɹ") and sometimes drops the R sound in other contexts
entirely — phonemizing live with espeak reproduces whatever convention the
model actually learned instead of guessing it.
"""
import json
import re
from functools import lru_cache

import numpy as np

from proscor.align import _NEG, _ctc_loglik, _ctc_viterbi

MODEL_REPO = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX"
SAMPLE_RATE = 16000

_SESSION = None
_SESSION_INT8 = None
_TOK2ID = None
_BLANK = None


def available() -> bool:
    """True if the optional phone-level alignment dependencies are installed."""
    try:
        import onnxruntime  # noqa: F401
        import phonemizer  # noqa: F401
    except ImportError:
        return False
    return True


def _session(use_int8: bool = True):
    global _SESSION, _SESSION_INT8, _TOK2ID, _BLANK
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    if _SESSION is not None and _SESSION_INT8 == use_int8:
        return _SESSION
    fname = "onnx/model_int8.onnx" if use_int8 else "onnx/model.onnx"
    path = hf_hub_download(MODEL_REPO, fname)
    _SESSION = ort.InferenceSession(path)
    _SESSION_INT8 = use_int8

    vocab_path = hf_hub_download(MODEL_REPO, "vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        _TOK2ID = json.load(f)
    _BLANK = _TOK2ID["<pad>"]  # HF Wav2Vec2CTCTokenizer convention: pad = CTC blank
    return _SESSION


def _logprobs(samples: np.ndarray, use_int8: bool = True) -> np.ndarray:
    """float32 mono 16 kHz [-1, 1] -> (T, vocab) CTC log-probs. wav2vec2
    takes raw waveform (per-utterance zero-mean/unit-variance normalized,
    matching the HF feature extractor's `do_normalize`) -- no fbank step."""
    sess = _session(use_int8)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    samples = (samples - samples.mean()) / (samples.std() + 1e-7)
    logits = sess.run(None, {"input_values": samples[None, :]})[0][0]
    m = logits.max(-1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))


@lru_cache(maxsize=4096)
def _phonemize_word(word: str) -> tuple:
    """espeak-ng's en-us phonemization is case-sensitive in a way that has
    nothing to do with pronunciation: an all-caps word that also reads as a
    plausible acronym gets spelled out letter-by-letter instead of
    pronounced ("IT" -> "aɪ t iː" = "I-T", vs. "it" -> "ɪ t"; "US" -> "j uː
    ɛ s" = "U-S", vs. "us" -> "ʌ s"). speechocean762's transcripts are all
    caps, so words are lowercased before phonemizing to avoid this -- caught
    by comparing every unique word in the test split against its lowercase
    form (2/1869 differed, 2.14% of word occurrences: "IT" and "US")."""
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    out = phonemize([word.lower()], language="en-us", backend="espeak", strip=True,
                     separator=Separator(phone=" ", word=""),
                     preserve_punctuation=False, with_stress=False)[0]
    return tuple(out.split())


def _word_phones_and_ids(word: str) -> tuple:
    """Espeak-ng IPA phones for `word`, and their ids in this model's vocab.
    Phones espeak emits that aren't in this model's 392-entry vocab (rare)
    are dropped from *both* lists together, so they stay index-aligned."""
    kept = [p for p in _phonemize_word(word) if p in _TOK2ID]
    return kept, [_TOK2ID[p] for p in kept]


def align_words_gop(samples: np.ndarray, words: list, sr: int = SAMPLE_RATE,
                     use_int8: bool = True) -> dict:
    """Force-align a full utterance's `words` against `samples` at phone
    granularity. Returns
    `{"word_gop": [...], "phone_gop": [[...], ...]}` (both parallel to
    `words`): `word_gop[i]` is `{"gop", "n_frames"}` or None (unsegmentable/
    unaligned), mirroring `proscor.align.align_words_gop`'s per-word GOP but
    computed over true phone spans rather than BPE spans; `phone_gop[i]` is
    the per-phone breakdown for word i (`[{"phone", "gop", "n_frames"}, ...]`),
    used for the coverage-limited direct phones-accuracy comparison in
    scripts/eval_so762_phone.py (mismatched phone counts vs. the dataset's
    own ARPABET segmentation -- e.g. from the R-merging above -- mean not
    every word can be compared phone-for-phone)."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.max(initial=0.0) > 1.0 or samples.min(initial=0.0) < -1.0:
        samples = samples / 32768.0
    if sr != SAMPLE_RATE:
        from audiokit import resample

        samples = np.asarray(resample(samples, sr, SAMPLE_RATE), dtype=np.float32)

    _session(use_int8)
    lp = _logprobs(samples, use_int8)

    flat, spans, phone_texts = [], [], []
    for w in words:
        phones, toks = _word_phones_and_ids(w)
        start = len(flat)
        flat.extend(toks)
        spans.append((start, len(flat)))
        phone_texts.append(phones)

    if not flat:
        return {"word_gop": [None] * len(words), "phone_gop": [[] for _ in words]}

    path, loglik = _ctc_viterbi(lp, flat, _BLANK)
    if path is None:
        return {"word_gop": [None] * len(words), "phone_gop": [[] for _ in words]}

    ext_labels = np.array([_BLANK if i % 2 == 0 else flat[i // 2] for i in range(2 * len(flat) + 1)])
    frame_best = lp.max(axis=-1)

    word_gop, phone_gop = [], []
    for (start, end), phones in zip(spans, phone_texts):
        if start == end:
            word_gop.append(None)
            phone_gop.append([])
            continue

        this_word_phones = []
        for k, phone in zip(range(start, end), phones):
            state = 2 * k + 1
            pidx = np.nonzero(path == state)[0]
            if pidx.size == 0:
                this_word_phones.append(None)
                continue
            pgop = float((lp[pidx, ext_labels[state]] - frame_best[pidx]).mean())
            this_word_phones.append({"phone": phone, "gop": pgop, "n_frames": int(pidx.size)})
        phone_gop.append(this_word_phones)

        # Word GOP = mean of its phones' own GOP (phone-state frames only).
        # An earlier version averaged over the word's whole frame *span*
        # (phone states + the blank states between them, mirroring
        # proscor.align.align_words_gop's BPE formula) -- but a word here has
        # many more internal phone/blank transitions than a BPE word has
        # subword/blank transitions, so that diluted the per-phone signal
        # with blank-frame posterior-deficits that aren't informative about
        # pronunciation quality. Measured on a 100-utterance sample: pooled
        # word-level Pearson r rose from 0.03 to 0.22 switching to this
        # formula (still see PLAN.md section 5a for the full result and why
        # it still trails the BPE model's word-level number).
        phone_gops = [p["gop"] for p in this_word_phones if p is not None]
        if phone_gops:
            word_gop.append({"gop": float(np.mean(phone_gops)),
                              "n_frames": sum(p["n_frames"] for p in this_word_phones if p is not None)})
        else:
            word_gop.append(None)

    return {"word_gop": word_gop, "phone_gop": phone_gop}


# --- Segmentation-free GOP (PLAN.md section 5f) ----------------------------
#
# align_words_gop's GOP is "posterior-deficit at the Viterbi-aligned frames":
# force-align, then compare the aligned token's log-prob to the best token's
# log-prob at those same frames. Section 5c/5e found this catches deletions
# well but is nearly blind to substitutions (median GOP for a substituted
# phone is exactly 0.0 in every L1 tested) -- with a peaky CTC model,
# Viterbi lands on the frames where the *wrong* phone the speaker actually
# produced peaks, so "posterior at the aligned frame" is high even though
# the wrong phone was said.
#
# Cao, Fan, Svendsen & Salvi, "Segmentation-free Goodness of Pronunciation"
# (arXiv:2507.16838, IEEE 2025) sidesteps this: instead of scoring the
# aligned frames of the canonical phone, compare two whole-sequence CTC
# likelihoods -- log P(canonical sequence) vs. log P(the same sequence with
# this one phone replaced by "any phone, or nothing"), both marginalized
# over *every* alignment via the CTC forward algorithm (no Viterbi/
# segmentation step at all). GOP_SF(i) = logP(L_C) - logP(L_SDI) is large
# when the canonical phone is clearly the best explanation for that stretch
# of audio, and collapses toward 0 when some other phone (or silence) fits
# just as well -- including a confidently-produced *wrong* phone, which is
# exactly the case posterior-deficit GOP misses.
#
# Implementation, corrected version: an earlier attempt tried to compute
# the substitution term in one forward pass by replacing the target
# position's emission with log-sum-exp over every candidate phone's
# log-prob at each frame (a "wildcard" state sharing one scalar DP with the
# rest of the sequence). That's wrong, not just imprecise: whenever the
# wildcard state is active for more than one frame -- which is the normal
# case, since forward marginalizes over every possible frame-count
# allocation -- a per-frame log-sum-exp lets different frames within the
# SAME span implicitly "vote" for different candidates (mixing candidate
# c's evidence at frame t with candidate c''s at frame t+1), which is not
# what "this whole span is candidate c" means. Caught by brute-force
# comparison in tests/test_align_phone.py: it overcounted probability mass
# by 2x-40x, not a rounding-level gap. logP(any candidate here) genuinely
# requires logsumexp_c logP(sequence with c substituted in) -- summing
# whole-sequence likelihoods per candidate, not per-frame emissions.
#
# _ctc_loglik_wildcard now does exactly that: one _ctc_loglik call per
# candidate (correct by construction -- it's the literal definition, and
# _ctc_loglik is already the well-tested forward algorithm). The cost is
# O(T*S) per candidate, so gop_sf restricts each position's candidate pool
# to a small, data-driven confusion set (the canonical phone plus the
# top-K phones by peak log-prob in a local window around that position)
# rather than the full ~40-392-symbol vocabulary -- benchmarked at ~1
# second/utterance at K=10 on a realistic 40-phone utterance (T=300,
# V=392), which is what makes a full-corpus run tractable. A confidently
# *wrong* phone -- the exact case this method exists to catch -- has high
# local posterior by construction, so it lands in the top-K; this doesn't
# undercut the method's point, it just skips checking phones with no local
# acoustic support to begin with.


def _ctc_loglik_wildcard(lp: np.ndarray, tokens: list, position: int, blank: int,
                          candidate_ids: np.ndarray) -> float:
    """logP(tokens, with tokens[position] replaced by "any phone in
    candidate_ids") = logsumexp over one _ctc_loglik call per candidate.
    Marginalizing a *whole-sequence* likelihood over candidates, not a
    per-frame emission -- see the module comment above for why the faster
    per-frame version is unsound."""
    from scipy.special import logsumexp

    lls = [_ctc_loglik(lp, tokens[:position] + [int(c)] + tokens[position + 1:], blank)
           for c in candidate_ids]
    return float(logsumexp(lls))


def _local_candidates(lp: np.ndarray, tokens: list, position: int, blank: int,
                       top_k: int = 10, pad_frac: float = 0.5) -> np.ndarray:
    """The canonical phone at `position` plus the `top_k` phones with the
    highest peak log-prob in a local window around it -- a bounded,
    data-driven confusion set for `_ctc_loglik_wildcard`'s candidate_ids.
    The window is `tokens`' position mapped proportionally onto `lp`'s T
    frames (tokens are ~evenly spaced in time on average), padded by
    `pad_frac` of one token's average width on each side so a phone whose
    true span drifted from the proportional estimate is still covered."""
    T, N = lp.shape[0], len(tokens)
    width = max(T / max(N, 1), 1.0)
    lo = max(0, int(position * width - pad_frac * width))
    hi = min(T, int((position + 1) * width + pad_frac * width) + 1)
    window = lp[lo:hi]  # (w, V)
    peak = window.max(axis=0)  # (V,) -- this phone's best frame in the window
    peak = peak.copy()
    peak[blank] = -np.inf
    order = np.argsort(peak)[::-1]
    top = [int(c) for c in order[:top_k] if peak[c] > -np.inf]
    return np.array(sorted(set(top) | {tokens[position]}))


def gop_sf(lp: np.ndarray, tokens: list, blank: int, vocab_size: int,
           top_k: int = 10) -> list:
    """Segmentation-free GOP (Cao et al. 2025, GOP-SF-SD variant) for every
    position in `tokens`, forced through `lp`'s CTC posteriors with no
    alignment/segmentation step. Returns one float per token (or None for
    every position if the audio can't even fit the canonical sequence).
    `tokens` should be the *whole utterance's* flat phone-id sequence (not
    one word in isolation), so surrounding-word context is available the
    same way `align_words_gop`'s Viterbi pass already uses it. `vocab_size`
    is accepted for API stability but unused now that the substitution
    candidate pool comes from `_local_candidates` rather than the full
    vocabulary -- see the module comment above."""
    canonical_ll = _ctc_loglik(lp, tokens, blank)
    if canonical_ll <= _NEG / 2:
        return [None] * len(tokens)

    scores = []
    for i in range(len(tokens)):
        candidate_ids = _local_candidates(lp, tokens, i, blank, top_k=top_k)
        sub_ll = _ctc_loglik_wildcard(lp, tokens, i, blank, candidate_ids)
        del_tokens = tokens[:i] + tokens[i + 1:]
        del_ll = _ctc_loglik(lp, del_tokens, blank) if del_tokens else _NEG
        sdi_ll = float(np.logaddexp(sub_ll, del_ll))
        scores.append(canonical_ll - sdi_ll)
    return scores


def align_words_gop_sf(samples: np.ndarray, words: list, sr: int = SAMPLE_RATE,
                        use_int8: bool = True) -> dict:
    """Segmentation-free counterpart to `align_words_gop`: same target-phone
    construction (espeak phonemization per word, flattened to one
    utterance-level sequence), but scored with `gop_sf` instead of
    Viterbi-alignment posterior-deficit. Returns
    `{"word_gop": [...], "phone_gop": [[...], ...]}` in the same shape as
    `align_words_gop`, so it's a drop-in swap in eval scripts."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.max(initial=0.0) > 1.0 or samples.min(initial=0.0) < -1.0:
        samples = samples / 32768.0
    if sr != SAMPLE_RATE:
        from audiokit import resample

        samples = np.asarray(resample(samples, sr, SAMPLE_RATE), dtype=np.float32)

    _session(use_int8)
    lp = _logprobs(samples, use_int8)

    flat, spans, phone_texts = [], [], []
    for w in words:
        phones, toks = _word_phones_and_ids(w)
        start = len(flat)
        flat.extend(toks)
        spans.append((start, len(flat)))
        phone_texts.append(phones)

    if not flat:
        return {"word_gop": [None] * len(words), "phone_gop": [[] for _ in words]}

    scores = gop_sf(lp, flat, _BLANK, lp.shape[1])

    word_gop, phone_gop = [], []
    for (start, end), phones in zip(spans, phone_texts):
        if start == end:
            word_gop.append(None)
            phone_gop.append([])
            continue
        this_word = [{"phone": p, "gop": scores[k]} if scores[k] is not None else None
                     for k, p in zip(range(start, end), phones)]
        phone_gop.append(this_word)
        vals = [p["gop"] for p in this_word if p is not None]
        word_gop.append({"gop": float(np.mean(vals))} if vals else None)

    return {"word_gop": word_gop, "phone_gop": phone_gop}


# --- ARPABET <-> espeak-IPA reconciliation (PLAN.md section 5a item 4) -----
#
# `align_words_gop`'s phone_gop is keyed by *espeak's* phone segmentation,
# which doesn't always have the same phone count as speechocean762's own
# ARPABET segmentation for a word (94.6% of words match; see PLAN.md). Three
# distinct many-to-one patterns cause most mismatches:
#   (a) rhotic-vowel merge: espeak fuses vowel+R into one token
#       ("mark" -> "m ɑːɹ k", ARPABET "M AA0 R K")
#   (b) syllabic-L merge: espeak's vocab has a dedicated "əl" token for a
#       syllabic L that ARPABET spells as two phones, AH0 L
#       ("difficult" -> "...k əl t")
#   (c) diphthong-cluster merge: espeak treats some vowel sequences as one
#       complex-nucleus token ("iə", "aɪə", "aɪɚ") where ARPABET keeps two
#       vowel symbols ("idea" -> "d iə" vs "D IH AH1")
# A fourth cause is *not* reconcilable: genuine dialectal disagreement
# (espeak's en-us applies yod-dropping, "new" -> "n uː", no /j/, where
# CMUdict's canonical entry keeps the historical glide "N Y UW0"). No
# realignment recovers a phone the model was never asked to produce.
#
# `reconcile_phones` aligns the two segmentations with a constrained
# Needleman-Wunsch DP (1:1 matches via an ARPABET/IPA equivalence table,
# plus 2:1/3:1 merges for (a)-(c)) so every ARPABET phone gets a GOP
# estimate where possible, instead of discarding the whole word on a count
# mismatch. A merge duplicates one espeak phone's GOP across the 2-3
# ARPABET phones it represents -- an approximation (one frame region can't
# be split further), not a precise per-phone estimate for those slots.

_ARPABET_EQUIV = {
    "AA": {"ɑː", "ɑ", "ɒ"}, "AE": {"æ"}, "AH": {"ʌ", "ə", "ɐ"},
    "AO": {"ɔː", "ɔ", "oː", "o"}, "AW": {"aʊ"}, "AY": {"aɪ"},
    "B": {"b"}, "CH": {"tʃ"}, "D": {"d", "ɾ"}, "DH": {"ð"}, "EH": {"ɛ"},
    "ER": {"ɜː", "ɚ", "ɝ"}, "EY": {"eɪ"}, "F": {"f"}, "G": {"ɡ"}, "HH": {"h"},
    "IH": {"ɪ", "ᵻ", "i"}, "IY": {"iː", "i"}, "JH": {"dʒ"}, "K": {"k"},
    "L": {"l", "ɫ"}, "M": {"m"}, "N": {"n"}, "NG": {"ŋ"}, "OW": {"oʊ"},
    "OY": {"ɔɪ"}, "P": {"p"}, "R": {"ɹ", "r"}, "S": {"s"}, "SH": {"ʃ"},
    "T": {"t", "ɾ", "ʔ"}, "TH": {"θ"}, "UH": {"ʊ"}, "UW": {"uː", "u"},
    "V": {"v"}, "W": {"w"}, "Y": {"j"}, "Z": {"z"}, "ZH": {"ʒ"},
}
# Symbols that can stand in for a trailing R, or for AH0 acting as a
# rhotic offglide in place of a separate R phone (e.g. "entire" -> "IH0 N
# T AY1 AH0", no R phone at all, matched against espeak's "aɪɚ").
_RHOTIC_TAIL = {"ɹ", "ɚ", "ɜː", "r"}


_DEL_COST = 1.0
_INS_COST = 1.0
_SUB_COST = 0.9  # strictly < _DEL_COST: a mismatched-but-real GOP is more
                 # informative than no GOP at all, so a substitution always
                 # wins over discarding the phone, not just on a tie.
_MERGE_COST = 0.5


def _stress_strip(phone: str) -> str:
    return re.sub(r"\d", "", phone)


def _strip_length(s: str) -> str:
    return s.replace("ː", "")


def _match_cost(arpabet_phone: str, espeak_phone: str) -> float:
    """0.0 if same equivalence class (stress-agnostic), else SUB_COST."""
    equiv = _ARPABET_EQUIV.get(_stress_strip(arpabet_phone), set())
    if espeak_phone in equiv:
        return 0.0
    stripped = _strip_length(espeak_phone)
    if stripped in {_strip_length(e) for e in equiv}:
        return 0.0
    return _SUB_COST


def _merge_compatible(espeak_token: str, arpabet_phones: list) -> bool:
    """True if `espeak_token` plausibly represents the 2-3 `arpabet_phones`
    merged together: its (length-mark-stripped) text starts with a
    representative of the first phone's equivalence class, and either ends
    with a representative of the last phone's, or the last phone is
    R/ER/AH0 (rhoticity) and the token ends in a rhotic-tail symbol. This is
    a heuristic, not a proof -- see the "player" case in
    tests/test_align_phone.py for a known pattern it doesn't catch (the DP's
    del/ins fallback still applies, it just can't do better than the
    status quo for that word)."""
    if len(arpabet_phones) not in (2, 3):
        return False
    first, last = _stress_strip(arpabet_phones[0]), _stress_strip(arpabet_phones[-1])
    tok = _strip_length(espeak_token)

    first_equiv = {_strip_length(e) for e in _ARPABET_EQUIV.get(first, set())}
    first_ok = any(e and tok.startswith(e) for e in first_equiv)

    if last in ("R", "ER") or (last == "AH" and arpabet_phones[-1].endswith("0")):
        last_ok = tok and (tok[-1] in "ɹɚr" or tok.endswith(tuple(_strip_length(s) for s in _RHOTIC_TAIL)))
    else:
        last_equiv = {_strip_length(e) for e in _ARPABET_EQUIV.get(last, set())}
        last_ok = any(e and tok.endswith(e) for e in last_equiv)
    return bool(first_ok and last_ok)


def reconcile_phones(dataset_phones: list, espeak_phones: list, espeak_gops: list) -> tuple:
    """Align `dataset_phones` (ARPABET, the reference) against
    `espeak_phones`/`espeak_gops` (this word's espeak IPA phones and their
    GOP scores from `phone_gop`) via a constrained Needleman-Wunsch DP.
    Returns `(gops, ops)`: `gops` has one GOP-or-None per dataset phone
    (None = a genuine deletion, e.g. yod-dropping -- no evidence exists);
    `ops` is the parallel list of {"match","sub","merge2","merge3","del"}
    for diagnostics (scripts/eval_so762_phone.py reports op-type counts to
    catch over/under-firing of the merge rule)."""
    N, K = len(dataset_phones), len(espeak_phones)
    INF = float("inf")
    cost = [[INF] * (K + 1) for _ in range(N + 1)]
    back = [[None] * (K + 1) for _ in range(N + 1)]
    cost[0][0] = 0.0

    for i in range(N + 1):
        for j in range(K + 1):
            if i == 0 and j == 0:
                continue
            best, best_op = INF, None
            if i > 0 and j > 0:
                c = _match_cost(dataset_phones[i - 1], espeak_phones[j - 1])
                op = "match" if c == 0.0 else "sub"
                if cost[i - 1][j - 1] + c < best:
                    best, best_op = cost[i - 1][j - 1] + c, (1, 1, op)
            if i > 1 and j > 0 and _merge_compatible(espeak_phones[j - 1], dataset_phones[i - 2:i]):
                if cost[i - 2][j - 1] + _MERGE_COST < best:
                    best, best_op = cost[i - 2][j - 1] + _MERGE_COST, (2, 1, "merge2")
            if i > 2 and j > 0 and _merge_compatible(espeak_phones[j - 1], dataset_phones[i - 3:i]):
                if cost[i - 3][j - 1] + _MERGE_COST < best:
                    best, best_op = cost[i - 3][j - 1] + _MERGE_COST, (3, 1, "merge3")
            if i > 0 and cost[i - 1][j] + _DEL_COST < best:
                best, best_op = cost[i - 1][j] + _DEL_COST, (1, 0, "del")
            if j > 0 and cost[i][j - 1] + _INS_COST < best:
                best, best_op = cost[i][j - 1] + _INS_COST, (0, 1, "ins")
            cost[i][j] = best
            back[i][j] = best_op

    gops, ops = [None] * N, [None] * N
    i, j = N, K
    while i > 0 or j > 0:
        di, dj, op = back[i][j]
        if op == "del":
            gops[i - 1], ops[i - 1] = None, "del"
        elif op == "ins":
            pass
        elif op in ("match", "sub"):
            gops[i - 1], ops[i - 1] = espeak_gops[j - 1], op
        elif op in ("merge2", "merge3"):
            g = espeak_gops[j - 1]
            for k in range(i - di, i):
                gops[k], ops[k] = g, op
        i, j = i - di, j - dj
    return gops, ops
