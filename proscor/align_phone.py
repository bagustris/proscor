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
from functools import lru_cache

import numpy as np

from proscor.align import _ctc_viterbi

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
