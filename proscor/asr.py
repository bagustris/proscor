"""Transcribe recorded audio via sherox (sherpa-onnx) -> text + per-word timing."""
import numpy as np

from proscor import config
from proscor.config import SAMPLE_RATE

_REC = None
_REC_MODEL_DIR = None


def _cfg(model_dir: str, model_type: str):
    from sherox.config import Config

    return Config(
        model_dir=model_dir, model_type=model_type, offline=True,
        sample_rate=SAMPLE_RATE, num_threads=config.ASR_NUM_THREADS,
        word_timestamps=True, language="en",
    )


def _recognizer(model_dir: str = None, model_type: str = None):
    global _REC, _REC_MODEL_DIR
    model_dir = model_dir or config.ASR_MODEL_DIR
    model_type = model_type or config.ASR_MODEL_TYPE
    if _REC is None or _REC_MODEL_DIR != model_dir:
        from sherox.asr_engine import build_offline_recognizer

        _REC = build_offline_recognizer(_cfg(model_dir, model_type))
        _REC_MODEL_DIR = model_dir
    return _REC


def _words_from_tokens(tokens: list, timestamps: list) -> list:
    """Group sub-word tokens (leading-space = new word) into words with
    start/end times. Used because sherpa-onnx's offline `result.words` and
    `result.ys_log_probs` are empty for the NeMo CTC model as of sherpa-onnx
    1.13.2 -- confidence is therefore not available and defaults to 1.0."""
    words = []
    for tok, ts in zip(tokens, timestamps):
        if tok.startswith(" ") or not words:
            words.append({"word": tok.strip(), "start": ts})
        else:
            words[-1]["word"] += tok
    for i, w in enumerate(words):
        w["end"] = words[i + 1]["start"] if i + 1 < len(words) else w["start"] + 0.3
        w["conf"] = 1.0
    return words


def transcribe(samples: np.ndarray, sr: int = SAMPLE_RATE, model_dir: str = None) -> dict:
    """Transcribe one utterance -> {"text": str, "words": [{"word","conf","start","end"}]}."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.max(initial=0.0) > 1.0 or samples.min(initial=0.0) < -1.0:
        samples = samples / 32768.0  # int16 range -> float32 [-1, 1]

    rec = _recognizer(model_dir=model_dir)
    stream = rec.create_stream()
    stream.accept_waveform(sr, samples)
    rec.decode_stream(stream)
    res = stream.result

    raw_words = getattr(res, "words", None) or []
    if raw_words:
        words = [
            {
                "word": w.word,
                "conf": getattr(w, "confidence", getattr(w, "prob", 1.0)),
                "start": float(w.start),
                "end": float(w.end),
            }
            for w in raw_words
        ]
    else:
        words = _words_from_tokens(list(res.tokens or []), list(res.timestamps or []))

    return {"text": res.text.strip(), "words": words}
