"""CTC forced-alignment scoring for single-word prompts.

Free-decoding isolated short words is outside general ASR models' operating
envelope: measured on TTS minimal pairs, free decode gets 36-47% word accuracy
across NeMo CTC / Parakeet TDT / Whisper small.en ("ship" -> "six"). For a
KNOWN target word the right question is not "what word is this?" but "how well
does this audio fit the target vs. its confusable neighbours?" — a
forced-alignment likelihood question. Same clips, same NeMo CTC model: 85%
minimal-pair discrimination vs. 42% free decode.

sherpa-onnx exposes no frame posteriors, so this module runs the model's ONNX
encoder directly (onnxruntime + kaldi_native_fbank; feature pipeline
replicated from the model card's test.py) and computes:

- CTC forced-alignment log-likelihood of the target word. The model dir ships
  no sentencepiece model, so every BPE segmentation of the word is enumerated
  by DP over tokens.txt and the best-scoring one is used.
- The same likelihood for cmudict confusables (phoneme edit distance 1).
- A GOP-style fit: (loglik(target) - unconstrained best path) / frames,
  which is ~0 when the target explains the audio as well as anything can.

Score = 100 * (ALIGN_POSTERIOR_WEIGHT * p(target | candidates)
               + ALIGN_GOP_WEIGHT * exp(gop / ALIGN_GOP_SCALE))   (config.py)
"""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from proscor import config
from proscor.config import ROOT, SAMPLE_RATE

_NEG = -1e30

_SESSION = None
_SESSION_DIR = None
_VOCAB = None  # (tok2id, blank_id)


def available() -> bool:
    """True if the optional alignment dependencies are installed."""
    try:
        import kaldi_native_fbank  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _model_dir(model_dir: str = None) -> Path:
    from proscor.asr import _ensure_asr_model

    model_dir = _ensure_asr_model(model_dir or config.ASR_MODEL_DIR)
    path = Path(model_dir)
    return path if path.is_absolute() else (ROOT / path)


def _session(model_dir: str = None):
    global _SESSION, _SESSION_DIR, _VOCAB
    import onnxruntime as ort

    mdir = _model_dir(model_dir)
    if _SESSION is not None and _SESSION_DIR == mdir:
        return _SESSION
    onnx = mdir / "model.int8.onnx"
    if not onnx.exists():
        onnx = mdir / "model.onnx"
    _SESSION = ort.InferenceSession(str(onnx))
    _SESSION_DIR = mdir

    tok2id = {}
    with open(mdir / "tokens.txt", encoding="utf-8") as f:
        for line in f:
            sym, idx = line.rstrip("\n").split()
            tok2id[sym] = int(idx)
    _VOCAB = (tok2id, len(tok2id) - 1)  # blank is the last token (model test.py)
    return _SESSION


def _logprobs(samples: np.ndarray, model_dir: str = None) -> np.ndarray:
    """float32 mono 16 kHz [-1, 1] -> (T, vocab) CTC log-probs."""
    import kaldi_native_fbank as knf

    sess = _session(model_dir)
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(SAMPLE_RATE, (samples * 32768).tolist())
    fb.input_finished()
    feats = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])
    feats = (feats - feats.mean(0, keepdims=True)) / (feats.std(0, keepdims=True) + 1e-5)
    feats = feats.astype(np.float32)[None].transpose(0, 2, 1)  # (1, 80, T)
    in0, in1 = (i.name for i in sess.get_inputs())
    out = sess.run(None, {in0: feats, in1: np.array([feats.shape[2]], np.int64)})
    return out[0][0]


def _segmentations(word: str, tok2id: dict, blank: int, limit: int = 200) -> list:
    """All splits of '▁'+word into BPE vocab pieces (longest pieces first)."""
    target = "▁" + word.lower()
    results = []

    def rec(pos, toks):
        if len(results) >= limit:
            return
        if pos == len(target):
            results.append(list(toks))
            return
        for end in range(len(target), pos, -1):
            piece = target[pos:end]
            if pos > 0 and "▁" in piece:
                continue
            tid = tok2id.get(piece)
            if tid is not None and tid != blank and piece != "<unk>":
                toks.append(tid)
                rec(end, toks)
                toks.pop()

    rec(0, [])
    return results


def _ctc_loglik(lp: np.ndarray, tokens: list, blank: int) -> float:
    """Forward-algorithm log P(token sequence | audio) under CTC."""
    ext = []
    for t in tokens:
        ext += [blank, t]
    ext.append(blank)
    ext = np.array(ext)
    S, T = len(ext), lp.shape[0]
    # skip transition s-2 -> s allowed only onto a non-blank differing from s-2
    can_skip = np.zeros(S, dtype=bool)
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    alpha = np.full(S, _NEG)
    alpha[0] = lp[0, ext[0]]
    if S > 1:
        alpha[1] = lp[0, ext[1]]
    for t in range(1, T):
        stay = alpha
        step = np.concatenate(([_NEG], alpha[:-1]))
        skip = np.where(can_skip, np.concatenate(([_NEG, _NEG], alpha[:-2])), _NEG)
        alpha = np.logaddexp(np.logaddexp(stay, step), skip) + lp[t, ext]
    return float(np.logaddexp(alpha[-1], alpha[-2] if S > 1 else _NEG))


def _word_loglik(lp: np.ndarray, word: str, model_dir: str = None) -> float:
    tok2id, blank = _VOCAB
    segs = _segmentations(word, tok2id, blank)
    if not segs:
        return _NEG
    return max(_ctc_loglik(lp, s, blank) for s in segs)


def _greedy_text(lp: np.ndarray) -> str:
    tok2id, blank = _VOCAB
    id2tok = {i: t for t, i in tok2id.items()}
    ids = lp.argmax(-1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != blank:
            out.append(id2tok[int(i)])
        prev = i
    return "".join(out).replace("▁", " ").strip()


def _strip_stress(phonemes: list) -> list:
    return [re.sub(r"\d", "", p) for p in phonemes]


@lru_cache(maxsize=256)
def confusables(word: str) -> tuple:
    """cmudict words whose pronunciation is 1 phoneme edit from `word`'s.
    Homophones (distance 0) are excluded — alignment cannot discriminate
    them. Returns () if cmudict data is not available."""
    from rapidfuzz.distance import Levenshtein

    try:
        from nltk.corpus import cmudict

        entries = cmudict.dict()
    except (ImportError, LookupError):
        return ()

    from proscor.g2p import expected_phonemes

    target_ph = _strip_stress(expected_phonemes(word)[0] if expected_phonemes(word) else [])
    if not target_ph:
        return ()

    try:
        from nltk.corpus import words as nltk_words

        real_words = {w.lower() for w in nltk_words.words()}
    except (ImportError, LookupError):
        real_words = None  # optional corpus not downloaded -> skip filtering

    found = []
    for cand, prons in entries.items():
        if cand == word.lower() or not cand.isalpha():
            continue
        if real_words is not None and cand not in real_words:
            continue  # cmudict junk entries ("dru", "leve") make bad competitors
        dist = min(Levenshtein.distance(target_ph, _strip_stress(p), score_cutoff=2) for p in prons)
        if dist == 1:
            found.append(cand)
    # The cap is a safety valve only — cutting real confusables ("true" for
    # "through") creates false accepts, so keep it high and rank by length
    # similarity to the target so junk cmudict entries ("dru", "leve") go last.
    found.sort(key=lambda w: (abs(len(w) - len(word)), w))
    return tuple(found[:config.ALIGN_MAX_CONFUSABLES])


def _map_score(p_target: float, gop: float) -> float:
    """Blend candidate posterior + GOP fit into 0-100 (weights in config)."""
    fit = float(np.exp(gop / config.ALIGN_GOP_SCALE))
    blended = config.ALIGN_POSTERIOR_WEIGHT * p_target + config.ALIGN_GOP_WEIGHT * fit
    return max(0.0, min(100.0, 100.0 * blended))


def score_word(samples: np.ndarray, target_word: str, sr: int = SAMPLE_RATE,
               include_stress: bool = False, model_dir: str = None) -> dict:
    """Score one spoken word against `target_word` -> ScoreReport dict
    (same shape as proscor.score.score's report)."""
    from proscor.g2p import expected_phonemes
    from proscor.score import _clean, _phoneme_edits

    target = _clean(target_word)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.max(initial=0.0) > 1.0 or samples.min(initial=0.0) < -1.0:
        samples = samples / 32768.0
    if sr != SAMPLE_RATE:
        from audiokit import resample

        samples = np.asarray(resample(samples, sr, SAMPLE_RATE), dtype=np.float32)

    _session(model_dir)
    lp = _logprobs(samples, model_dir)
    greedy = _greedy_text(lp)

    expected_ph = expected_phonemes(target_word, include_stress)
    expected_ph = expected_ph[0] if expected_ph else []

    if not greedy:
        return {
            "score": 0.0,
            "words": [{
                "target": target_word, "recognized": None, "correct": False,
                "word_score": 0.0, "phonemes_expected": expected_ph,
                "phonemes_heard": [], "edits": [],
            }],
            "notes": "nothing recognized",
        }

    # Candidate set: the target plus its cmudict phoneme neighbours — the
    # pedagogically meaningful contrasts. The free decode does NOT compete:
    # free-decoding isolated words is the very failure mode this module works
    # around ("ship" -> "six"), so it is reported in the notes instead.
    candidates = [target] + [c for c in confusables(target) if c != target]
    logliks = {c: _word_loglik(lp, c, model_dir) for c in candidates}
    others = {c: v for c, v in logliks.items() if c != target}
    best_other = max(others, key=others.get) if others else None
    # margin > 0: the target explains the audio better than any confusable.
    # The prior gives the learner the benefit of the doubt on near-ties.
    margin = logliks[target] - others[best_other] if best_other else float("inf")
    correct = margin + config.ALIGN_TARGET_PRIOR >= 0
    heard = target if correct else best_other

    gop = (logliks[target] - float(lp.max(-1).sum())) / lp.shape[0]
    fit = float(np.exp(gop / config.ALIGN_GOP_SCALE))
    if best_other is None:
        word_score = max(0.0, min(100.0, 100.0 * fit))
        p_target = 1.0
    else:
        p_target = float(1.0 / (1.0 + np.exp(-(margin + config.ALIGN_TARGET_PRIOR))))
        word_score = _map_score(p_target, gop)
    # GOP veto: winning the confusable race means nothing if the target
    # explains the audio poorly overall (learner said an unrelated word —
    # its confusables are absurd, so the margin is trivially positive).
    if fit < config.ALIGN_MIN_FIT:
        correct = False
        heard = best_other
        word_score = min(word_score, 100.0 * fit)
    heard_ph = expected_phonemes(heard, include_stress) if heard else []
    heard_ph = heard_ph[0] if heard_ph else []
    edits = _phoneme_edits(expected_ph, heard_ph) if not correct else []

    if correct:
        notes = "aligned to target (single-word mode)"
    elif heard:
        notes = f'sounded closer to "{heard}" (single-word mode)'
    else:
        notes = "did not match the target (single-word mode)"
    if greedy and _clean(greedy) != target:
        notes += f'; free decode heard "{greedy}"'

    return {
        "score": round(word_score, 1),
        "words": [{
            "target": target_word,
            "recognized": heard,
            "correct": correct,
            "word_score": round(word_score, 1),
            "phonemes_expected": expected_ph,
            "phonemes_heard": heard_ph,
            "edits": edits,
        }],
        "notes": notes,
        "gop": round(float(gop), 3),
        "posterior": round(p_target, 3),
    }
