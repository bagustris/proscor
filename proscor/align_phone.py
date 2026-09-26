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

**Alternate acoustic model** (PLAN.md section 5i): `TORCH_MODEL_REPO`
(`facebook/wav2vec2-xlsr-53-espeak-cv-ft`, XLSR-53 cross-lingual
pretraining instead of `MODEL_REPO`'s LibriLight-60k English-only
pretraining) measurably beats the default on speechocean762 phone-level
correlation, for both scoring methods -- a real, unexplored lever pulled
alongside the scoring-formula work in sections 5f-5h. No ONNX export
exists for it, so passing `model_id=TORCH_MODEL_REPO` to `align_words_gop`/
`align_words_gop_sf` loads it via `transformers`+`torch` instead of
`onnxruntime` (optional deps, only needed for this path); the default
`model_id=None` keeps the original ONNX/lv-60 path byte-for-byte
unchanged.

**A third backend, ZIPA** (PLAN.md section 5k): `ZIPA_MODEL_REPO`
(`anyspeech/zipa-large-crctc-ns-800k`, Zhu et al., ACL 2025) is a
Zipformer-CTC phone recognizer trained on IPAPack++ (17k phone-labeled
hours + 11.8k pseudo-labeled hours, 88 languages) -- an order of
magnitude more phone-labeled pretraining data than TORCH_MODEL_REPO's
espeak-cv-ft fine-tune, a direct extension of the "more multilingual
phone-labeled pretraining helps" finding from section 5i. Two structural
differences from the other two backends, both handled below rather than
worked around: (1) its frontend takes 80-dim fbank features (via
`lhotse`, an optional dep only needed for this path), not raw waveform;
(2) its 127-symbol CTC vocabulary is *IPA characters*, not *IPA phones*
-- a phone espeak emits as one multi-codepoint string (e.g. "ɑːɹ") is
1-3 separate tokens in ZIPA's output (base letter + length mark +
rhotic-hook, each its own codepoint and each already a standalone entry
in `tokens.txt`); espeak's own codepoint boundaries line up with ZIPA's
token boundaries directly for most phones, but three common atomic
codepoints (script-g, r-colored schwa, espeak's "schwi") have no token
in ZIPA's 127-symbol vocab at all and need a small substitution table
(`_ZIPA_CHAR_SUBS`) rather than a straight per-character lookup -- see
PLAN.md section 5k for how that gap was found (systematic, not random:
~20% of speechocean762's unique words use one of the three). `_word_phone_spans`
is the one function that knows about this: for the ONNX/torch backends
it's a thin wrapper around `_word_phones_and_ids` where every phone spans
exactly one token (unchanged behavior, unit-tested); for ZIPA it expands
each phone into its constituent character tokens and records the
resulting multi-token span. `align_words_gop`/`align_words_gop_sf` both
consume these spans as "the token range this phone owns" and
average/frame-weight-average over however many tokens that is -- one for
the existing backends (mathematically identical to the old one-token-
per-phone code), 1-3 for ZIPA.
"""
import json
import re
from functools import lru_cache

import numpy as np

from proscor.align import _NEG, _ctc_loglik, _ctc_viterbi

MODEL_REPO = "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX"
TORCH_MODEL_REPO = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
ZIPA_MODEL_REPO = "anyspeech/zipa-large-crctc-ns-800k"
SAMPLE_RATE = 16000

_SESSIONS = {}  # (model_id, use_int8_or_None) -> (session, backend, vocab, blank)
_BACKEND = None       # "onnx" or "torch" -- whichever _SESSIONS entry is currently active
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


def _session(model_id: str = None, use_int8: bool = True):
    """Loads (or reuses) the session for `model_id`, and makes it the
    *active* one -- `_BACKEND`/`_TOK2ID`/`_BLANK` always reflect whichever
    session this call returns, since `_word_phones_and_ids` and friends
    read those as plain globals rather than taking the session as an
    argument. Sessions are cached per (model_id, use_int8) in `_SESSIONS`,
    not just the single most-recently-used one: a caller that alternates
    between two models within one loop (scripts/eval_*_model_paired.py,
    comparing them on the same utterances) must not reload either one's
    weights every call -- an earlier version cached only one slot, so
    alternating between MODEL_REPO and TORCH_MODEL_REPO per utterance
    reloaded xlsr-53's full weights from scratch every single utterance
    (caught by the 3+ second/utterance rate it produced, ~10x the
    expected combined cost of two already-loaded models)."""
    global _BACKEND, _TOK2ID, _BLANK
    from huggingface_hub import hf_hub_download

    model_id = model_id or MODEL_REPO
    if model_id == MODEL_REPO:
        backend = "onnx"
    elif model_id == ZIPA_MODEL_REPO:
        backend = "zipa"
    else:
        backend = "torch"
    key = (model_id, use_int8 if backend in ("onnx", "zipa") else None)

    if key not in _SESSIONS:
        if backend == "onnx":
            import onnxruntime as ort

            fname = "onnx/model_int8.onnx" if use_int8 else "onnx/model.onnx"
            path = hf_hub_download(model_id, fname)
            session = ort.InferenceSession(path)
            vocab_path = hf_hub_download(model_id, "vocab.json")
            with open(vocab_path, encoding="utf-8") as f:
                vocab = json.load(f)
            blank = vocab["<pad>"]
        elif backend == "zipa":
            import onnxruntime as ort

            fname = "model.int8.onnx" if use_int8 else "model.onnx"
            path = hf_hub_download(model_id, fname)
            session = ort.InferenceSession(path)
            tokens_path = hf_hub_download(model_id, "tokens.txt")
            vocab = {}
            with open(tokens_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        vocab[parts[0]] = int(parts[1])
            blank = vocab["<blk>"]
        else:
            from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

            tok = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
            model = Wav2Vec2ForCTC.from_pretrained(model_id)
            model.eval()
            # The tokenizer's vocab can include a "|" word-delimiter entry the
            # CTC head was never trained to emit (seen on the xlsr-53
            # checkpoint: 393 vocab entries, 392-class output) -- keep only
            # ids the model actually outputs, or downstream indexing goes
            # out of bounds.
            vocab = {k: v for k, v in tok.get_vocab().items() if v < model.config.vocab_size}
            session = model
            blank = vocab["<pad>"]
        _SESSIONS[key] = (session, backend, vocab, blank)

    session, backend, vocab, blank = _SESSIONS[key]
    _BACKEND = backend
    _TOK2ID = vocab
    _BLANK = blank  # HF Wav2Vec2CTCTokenizer convention: pad = CTC blank
    return session


def _logprobs(samples: np.ndarray, model_id: str = None, use_int8: bool = True) -> np.ndarray:
    """float32 mono 16 kHz [-1, 1] -> (T, vocab) CTC log-probs. wav2vec2
    (onnx/torch backends) takes raw waveform (per-utterance zero-mean/
    unit-variance normalized, matching the HF feature extractor's
    `do_normalize`) -- no fbank step. ZIPA's Zipformer frontend instead
    takes 80-dim fbank features (`lhotse`'s kaldi-compatible extractor,
    the same one its own inference code uses -- see PLAN.md section 5k),
    computed from the *unnormalized* waveform; wav2vec2's amplitude
    normalization is specific to that model's training convention and
    doesn't apply here."""
    sess = _session(model_id, use_int8)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if _BACKEND == "zipa":
        import torch
        from lhotse.features.kaldi.extractors import Fbank, FbankConfig

        extractor = Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))
        audio_tensor = torch.from_numpy(samples).unsqueeze(0)
        feature = extractor.extract_batch([audio_tensor], sampling_rate=SAMPLE_RATE)[0]
        feature = feature.unsqueeze(0).numpy().astype(np.float32)
        feat_lens = np.array([feature.shape[1]], dtype=np.int64)
        logits = sess.run(None, {"x": feature, "x_lens": feat_lens})[0][0]
    else:
        samples = (samples - samples.mean()) / (samples.std() + 1e-7)
        if _BACKEND == "onnx":
            logits = sess.run(None, {"input_values": samples[None, :]})[0][0]
        else:
            import torch

            with torch.no_grad():
                logits = sess(torch.from_numpy(samples)[None, :]).logits[0].numpy()
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


def _leading_marker_prefix() -> list:
    """ZIPA's CTC output includes a leading "▁" (SentencePiece word-boundary
    symbol, id 3) that the model emits with near-certainty at the very
    start of an utterance, *before* the first real phone -- confirmed by
    inspection (PLAN.md section 5k): the first several frames put >99%
    mass on "▁", not blank or the canonical first phone. It is not
    re-emitted before every word (only utterance-initial), so it isn't
    modeled per-word the way blank is implicit between phones -- it's
    prepended once, here, as an utterance-level prefix `flat` starts from,
    outside every phone's span so it never gets scored as if it were a
    phone. Without this, the canonical sequence's first phone is forced to
    explain frames the model spent on "▁" instead, producing a large
    spurious GOP/GOP-SF penalty on utterance-initial phones only (an
    isolated m=-52.7 GOP-SF outlier vs. the same phone's -0.0 to -0.01 on
    the other two backends is what surfaced this). The other two backends
    have no such symbol in their vocab, so this is a no-op for them."""
    if _BACKEND == "zipa" and "▁" in _TOK2ID:
        return [_TOK2ID["▁"]]
    return []


_ZIPA_CHAR_SUBS = {
    # espeak-ng emits these three atomic codepoints for very common English
    # sounds, none of which exist as their own token in ZIPA's 127-symbol
    # vocabulary (confirmed by scanning every phone speechocean762's word
    # list produces, PLAN.md section 5k): without a substitution these
    # phones are dropped entirely (not approximated), and they are not
    # rare -- "ɚ" alone (the unstressed r-colored vowel: "-er" as in
    # "mother", "computer", "teacher") appears in ~10% of speechocean762's
    # unique words.
    "ɡ": "g",     # IPA "script g" (U+0261) vs. ZIPA's ASCII "g" (U+0067) --
                  # same phone, a Unicode-convention mismatch, not a real
                  # substitution.
    "ɚ": "ə˞",    # r-colored schwa has no Unicode decomposition (confirmed:
                  # unicodedata.normalize("NFD", "ɚ") is a no-op) but ZIPA's
                  # vocab has both the base vowel and the rhotic-hook
                  # diacritic as separate tokens, so this is written out.
    "ɝ": "ɜ˞",    # same pattern, the stressed counterpart (not observed in
                  # speechocean762 but included for robustness/generality).
    "ᵻ": "ɪ",     # espeak's "schwi" (unstressed/reduced vowel between /ɪ/
                  # and /ə/, e.g. "roses", "wanted"); mapped to /ɪ/, the
                  # same approximation `_ARPABET_EQUIV["IH"]` already makes
                  # elsewhere in this file.
}


def _zipa_chars(phone: str):
    """Expand `phone`'s characters through `_ZIPA_CHAR_SUBS`, so a phone
    espeak writes with a codepoint ZIPA's vocab lacks still contributes the
    ZIPA-compatible characters it substitutes to (e.g. "ɚ" -> "ə", "˞"),
    instead of being silently dropped."""
    for ch in phone:
        yield from _ZIPA_CHAR_SUBS.get(ch, ch)


def _word_phone_spans(word: str) -> tuple:
    """(phones, token_ids, phone_spans): `token_ids` is this word's flat
    token sequence and `phone_spans[i]` is the (start, end) range within it
    that `phones[i]` owns. For the onnx/torch backends this is a thin
    wrapper around `_word_phones_and_ids` -- one token per phone, so every
    span has length 1, identical to indexing `token_ids` directly. For the
    zipa backend (PLAN.md section 5k) a phone is 1-3 *characters* in
    ZIPA's IPA vocabulary (e.g. "ɑːɹ" -> "ɑ", "ː", "ɹ", each already a
    standalone token in tokens.txt -- espeak's own codepoint boundaries
    coincide with ZIPA's, no remapping table needed), so a phone's span
    can be longer than 1; a phone whose characters are *all* missing from
    the vocab (essentially never observed -- ZIPA's 127 symbols cover the
    IPA inventory espeak uses) is dropped entirely, mirroring
    `_word_phones_and_ids`'s drop-both-lists-in-lockstep behavior."""
    if _BACKEND != "zipa":
        phones, ids = _word_phones_and_ids(word)
        return phones, ids, [(k, k + 1) for k in range(len(ids))]

    phones, flat, spans = [], [], []
    for p in _phonemize_word(word):
        ids = [_TOK2ID[ch] for ch in _zipa_chars(p) if ch in _TOK2ID]
        if not ids:
            continue
        start = len(flat)
        flat.extend(ids)
        spans.append((start, len(flat)))
        phones.append(p)
    return phones, flat, spans


def align_words_gop(samples: np.ndarray, words: list, sr: int = SAMPLE_RATE,
                     use_int8: bool = True, model_id: str = None) -> dict:
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
    every word can be compared phone-for-phone). `model_id`: None (default)
    uses `MODEL_REPO` (ONNX); pass `TORCH_MODEL_REPO` (or another
    espeak-phone HF repo with no ONNX export) to score with that model
    instead, via `transformers`+`torch` (PLAN.md section 5i)."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.max(initial=0.0) > 1.0 or samples.min(initial=0.0) < -1.0:
        samples = samples / 32768.0
    if sr != SAMPLE_RATE:
        from audiokit import resample

        samples = np.asarray(resample(samples, sr, SAMPLE_RATE), dtype=np.float32)

    _session(model_id, use_int8)
    lp = _logprobs(samples, model_id, use_int8)

    flat, spans, phone_texts, phone_spans = _leading_marker_prefix(), [], [], []
    for w in words:
        phones, toks, local_spans = _word_phone_spans(w)
        offset = len(flat)
        flat.extend(toks)
        spans.append((offset, len(flat)))
        phone_texts.append(phones)
        phone_spans.append([(offset + s, offset + e) for s, e in local_spans])

    if not flat:
        return {"word_gop": [None] * len(words), "phone_gop": [[] for _ in words]}

    path, loglik = _ctc_viterbi(lp, flat, _BLANK)
    if path is None:
        return {"word_gop": [None] * len(words), "phone_gop": [[] for _ in words]}

    ext_labels = np.array([_BLANK if i % 2 == 0 else flat[i // 2] for i in range(2 * len(flat) + 1)])
    frame_best = lp.max(axis=-1)

    word_gop, phone_gop = [], []
    for (start, end), phones, ph_spans in zip(spans, phone_texts, phone_spans):
        if start == end:
            word_gop.append(None)
            phone_gop.append([])
            continue

        this_word_phones = []
        for (pstart, pend), phone in zip(ph_spans, phones):
            # A phone owns 1 token (onnx/torch) or 1-3 tokens (zipa, one
            # per IPA character -- see _word_phone_spans); frame-weighted
            # mean across however many states/tokens it owns. With exactly
            # one token this reduces to the old single-state computation
            # byte-for-byte.
            state_gops = []
            first_frame, last_frame = None, None
            for k in range(pstart, pend):
                state = 2 * k + 1
                pidx = np.nonzero(path == state)[0]
                if pidx.size == 0:
                    continue
                pgop = float((lp[pidx, ext_labels[state]] - frame_best[pidx]).mean())
                state_gops.append((pgop, int(pidx.size)))
                first_frame = int(pidx[0]) if first_frame is None else min(first_frame, int(pidx[0]))
                last_frame = int(pidx[-1]) if last_frame is None else max(last_frame, int(pidx[-1]))
            if not state_gops:
                this_word_phones.append(None)
                continue
            total_frames = sum(n for _, n in state_gops)
            mean_gop = sum(g * n for g, n in state_gops) / total_frames
            # "span" = [first, last+1) CTC frame range the phone's own state(s)
            # occupy on the Viterbi path (20 ms frames); used by phone-level
            # DTW to cut a native template into per-phone regions (PLAN.md 5m).
            this_word_phones.append({"phone": phone, "gop": mean_gop, "n_frames": total_frames,
                                     "span": (first_frame, last_frame + 1)})
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
# Implementation: log P(L_SDI)'s substitution term needs, for one phone
# position, "some candidate phone was produced here" marginalized over the
# whole (~392-symbol) vocabulary. A first attempt tried a single forward
# pass with the target position's emission replaced by log-sum-exp over
# every candidate's log-prob *per frame* -- wrong, not just imprecise:
# CTC's "stay" transition lets a state persist over several frames, and a
# per-frame log-sum-exp lets frame t "vote" for one candidate while frame
# t+1 (still inside the same persistence) votes for a different one, which
# isn't a valid single-candidate path. Caught by brute-force comparison in
# tests/test_align_phone.py: it overcounted probability mass 2x-40x on toy
# examples, not a rounding-level gap. Restricting candidates to a small
# per-position confusion set (rather than the full vocabulary) was
# considered and rejected: it would make GOP-SF a different metric than the
# paper's (which marginalizes over the full vocabulary), undercutting any
# comparison to their published numbers later.
#
# The correct fix keeps full-vocabulary marginalization in one pass:
# candidate identity only matters *while a path is inside the wildcard's
# own self-loop*; entry into and exit from that state can be marginalized
# over candidates safely, because (given the precondition below) every
# candidate shares the same predecessor/successor topology. So the
# self-loop is tracked as `len(candidate_ids)` separate per-candidate
# lanes (no cross-lane mixing), and only merged (logsumexp) at entry/exit
# -- exactly the shared-prefix/shared-suffix factoring a forward-backward
# derivation would give, without needing a separate backward pass. Cost is
# O(T*(S+V)) for one phone position (matching the paper's stated
# complexity), verified against brute-force enumeration over the full
# candidate set (not a restricted one) in tests/test_align_phone.py.


def _ctc_loglik_wildcard(lp: np.ndarray, tokens: list, position: int, blank: int,
                          candidate_ids: np.ndarray) -> float:
    """logP(tokens, with tokens[position] replaced by "any phone in
    candidate_ids"), marginalized over which one -- mathematically
    logsumexp over one _ctc_loglik call per candidate (verified against
    that brute-force definition in tests/test_align_phone.py), computed
    here in one pass via per-candidate self-loop lanes that merge only at
    the wildcard state's entry/exit (see the module comment above for why
    a naive per-frame merge inside the self-loop is wrong).

    **Precondition, load-bearing for correctness:** `candidate_ids` must
    not contain `tokens[position - 1]` or `tokens[position + 1]` (the
    tokens immediately flanking this position), nor `blank`. CTC's skip
    transition (bypassing the blank between two states) is legal only when
    those two states carry different labels; excluding neighbor-colliding
    candidates makes every remaining candidate's identity provably
    different from both neighbors, so entry/exit legality can be computed
    once (from the wildcard's sentinel identity) instead of per-candidate.
    `gop_sf` enforces this."""
    from scipy.special import logsumexp

    ext = []
    for t in tokens:
        ext += [blank, t]
    ext.append(blank)
    ext = np.array(ext)
    ws = 2 * position + 1  # wildcard_state; always odd, so ws == 1 iff position == 0

    ext_topo = ext.copy()
    ext_topo[ws] = -1  # sentinel: != blank, != any real token id
    S, T = len(ext), lp.shape[0]
    can_skip = np.zeros(S, dtype=bool)
    can_skip[2:] = (ext_topo[2:] != blank) & (ext_topo[2:] != ext_topo[:-2])

    ext_idx = ext.copy()
    ext_idx[ws] = 0  # placeholder column; state ws's emission is always overridden below
    emit = lp[:, ext_idx]  # (T, S)

    alpha = np.full(S, _NEG)
    alpha_wc = np.full(len(candidate_ids), _NEG)  # per-candidate lanes, state ws only

    alpha[0] = emit[0, 0]
    if S > 1:
        alpha[1] = emit[0, 1]
    if ws == 1:  # position 0: the path can start directly in the wildcard, no leading blank
        alpha_wc = lp[0, candidate_ids].astype(np.float64)
        alpha[1] = logsumexp(alpha_wc)

    for t in range(1, T):
        stay = alpha
        step = np.concatenate(([_NEG], alpha[:-1]))
        skip = np.where(can_skip, np.concatenate(([_NEG, _NEG], alpha[:-2])), _NEG)
        new_alpha = np.logaddexp(np.logaddexp(stay, step), skip) + emit[t]

        entry = np.logaddexp(step[ws], skip[ws])  # scalar: mass entering the wildcard this frame
        alpha_wc = np.logaddexp(alpha_wc, entry) + lp[t, candidate_ids]  # per-candidate, no mixing
        new_alpha[ws] = logsumexp(alpha_wc)  # merge only here, at exit

        alpha = new_alpha

    return float(np.logaddexp(alpha[-1], alpha[-2] if S > 1 else _NEG))


def gop_sf(lp: np.ndarray, tokens: list, blank: int, vocab_size: int) -> list:
    """Segmentation-free GOP (Cao et al. 2025, GOP-SF-SD variant) for every
    position in `tokens`, forced through `lp`'s CTC posteriors with no
    alignment/segmentation step. Returns one float per token (or None for
    every position if the audio can't even fit the canonical sequence).
    `tokens` should be the *whole utterance's* flat phone-id sequence (not
    one word in isolation), so surrounding-word context is available the
    same way `align_words_gop`'s Viterbi pass already uses it.

    Known narrow edge case, not fixed: at a position whose canonical phone
    equals an immediately adjacent phone (rare -- needs the same phone to
    repeat with no phone between, e.g. across a word boundary like "big
    gate"), the candidate set excludes that value too (see
    `_ctc_loglik_wildcard`'s precondition), so the wildcard can't
    reconstruct the canonical path exactly at that one position and the
    `<= 0` guarantee can (rarely) be violated there. Not observed to matter
    in practice; flagged rather than silently accepted."""
    canonical_ll = _ctc_loglik(lp, tokens, blank)
    if canonical_ll <= _NEG / 2:
        return [None] * len(tokens)

    scores = []
    for i in range(len(tokens)):
        exclude = {blank}
        if i > 0:
            exclude.add(tokens[i - 1])
        if i + 1 < len(tokens):
            exclude.add(tokens[i + 1])
        candidate_ids = np.array([c for c in range(vocab_size) if c not in exclude])
        sub_ll = _ctc_loglik_wildcard(lp, tokens, i, blank, candidate_ids)
        del_tokens = tokens[:i] + tokens[i + 1:]
        del_ll = _ctc_loglik(lp, del_tokens, blank) if del_tokens else _NEG
        sdi_ll = float(np.logaddexp(sub_ll, del_ll))
        scores.append(canonical_ll - sdi_ll)
    return scores


def align_words_gop_sf(samples: np.ndarray, words: list, sr: int = SAMPLE_RATE,
                        use_int8: bool = True, model_id: str = None) -> dict:
    """Segmentation-free counterpart to `align_words_gop`: same target-phone
    construction (espeak phonemization per word, flattened to one
    utterance-level sequence), but scored with `gop_sf` instead of
    Viterbi-alignment posterior-deficit. Returns
    `{"word_gop": [...], "phone_gop": [[...], ...]}` in the same shape as
    `align_words_gop`, so it's a drop-in swap in eval scripts. `model_id`:
    see `align_words_gop`."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.max(initial=0.0) > 1.0 or samples.min(initial=0.0) < -1.0:
        samples = samples / 32768.0
    if sr != SAMPLE_RATE:
        from audiokit import resample

        samples = np.asarray(resample(samples, sr, SAMPLE_RATE), dtype=np.float32)

    _session(model_id, use_int8)
    lp = _logprobs(samples, model_id, use_int8)

    flat, spans, phone_texts, phone_spans = _leading_marker_prefix(), [], [], []
    for w in words:
        phones, toks, local_spans = _word_phone_spans(w)
        offset = len(flat)
        flat.extend(toks)
        spans.append((offset, len(flat)))
        phone_texts.append(phones)
        phone_spans.append([(offset + s, offset + e) for s, e in local_spans])

    if not flat:
        return {"word_gop": [None] * len(words), "phone_gop": [[] for _ in words]}

    scores = gop_sf(lp, flat, _BLANK, lp.shape[1])

    word_gop, phone_gop = [], []
    for (start, end), phones, ph_spans in zip(spans, phone_texts, phone_spans):
        if start == end:
            word_gop.append(None)
            phone_gop.append([])
            continue
        this_word = []
        for (pstart, pend), phone in zip(ph_spans, phones):
            # One score per token (gop_sf is scored on the flat sequence);
            # a phone's GOP is the mean over however many tokens it owns
            # (1 for onnx/torch, 1-3 for zipa -- see _word_phone_spans).
            # With exactly one token this reduces to scores[k] directly.
            vals = [scores[k] for k in range(pstart, pend) if scores[k] is not None]
            this_word.append({"phone": phone, "gop": float(np.mean(vals))} if vals else None)
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
