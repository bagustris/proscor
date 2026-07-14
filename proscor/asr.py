"""Transcribe recorded audio via sherox (sherpa-onnx) -> text + per-word timing."""
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

from proscor import config
from proscor.config import ROOT, SAMPLE_RATE

_REC = None
_REC_MODEL_DIR = None

# Default ASR model auto-download metadata.
# Kept in sync with scripts/get_asr_model.sh and the k2-fsa/sherpa-onnx release
# (https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models). The README
# and CLAUDE.md promise the ASR model "auto-downloads into models/ on first use";
# _ensure_asr_model() below honours that contract.
_ASR_MODEL_NAME = "sherpa-onnx-nemo-ctc-en-conformer-medium"
_ASR_MODEL_ARCHIVE = f"{_ASR_MODEL_NAME}.tar.bz2"
_ASR_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{_ASR_MODEL_ARCHIVE}"
)


def _cfg(model_dir: str, model_type: str):
    from sherox.config import Config

    return Config(
        model_dir=model_dir, model_type=model_type, offline=True,
        sample_rate=SAMPLE_RATE, num_threads=config.ASR_NUM_THREADS,
        word_timestamps=True, language="en",
    )


def _model_ready(model_path: Path) -> bool:
    """True when *model_path* holds the files sherpa-onnx needs to build the
    recognizer — a tokens file and a model .onnx. The globs match the ones used
    by sherox.asr_engine.build_offline_recognizer for the nemo_ctc path, so a
    "ready" dir here is exactly what build_offline_recognizer will accept."""
    return (
        model_path.is_dir()
        and any(model_path.glob("*tokens.txt"))
        and any(model_path.glob("model*.onnx"))
    )


def _download_default_asr_model(target: Path) -> None:
    """Download + extract the default NeMo CTC English model into target's
    parent (the models/ root). Mirrors sherox's TTS auto-download
    (sherox.tts._ensure_model) and scripts/get_asr_model.sh, reusing sherox's
    resumable downloader and path-traversal-safe tar extraction."""
    from sherox.utils import download_file, safe_tar_members

    models_root = target.parent
    models_root.mkdir(parents=True, exist_ok=True)
    archive = models_root / _ASR_MODEL_ARCHIVE

    print(
        f"[proscor] ASR model not found — downloading {_ASR_MODEL_NAME} "
        f"(~158 MB, one-time)…",
        file=sys.stderr,
    )
    download_file(_ASR_MODEL_URL, archive)

    print("[proscor] Extracting ASR model…", file=sys.stderr)
    try:
        with tarfile.open(archive, "r:bz2") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(models_root, filter="data")
            else:  # pragma: no cover - depends on Python version
                tf.extractall(models_root, members=safe_tar_members(tf, models_root))
    except KeyboardInterrupt:
        # Keep the downloaded archive (158 MB) so a retry doesn't re-fetch it;
        # only the possibly-partial extraction needs to go.
        shutil.rmtree(target, ignore_errors=True)
        raise
    except Exception as exc:
        archive.unlink(missing_ok=True)
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(f"ASR model extraction failed: {exc}") from exc

    archive.unlink(missing_ok=True)
    if not _model_ready(target):
        raise RuntimeError(
            f"ASR model extraction did not produce the expected files in {target}"
        )
    print(f"[proscor] ASR model ready at {target}", file=sys.stderr)


def _ensure_asr_model(model_dir: str) -> str:
    """Return an ASR model directory that exists, auto-downloading the bundled
    default NeMo CTC English model on first use so the CLI "just works" the
    first time (as documented in README.md / CLAUDE.md).

    Only the default model (config.ASR_MODEL_DIR) is auto-fetched — a custom
    --model-dir is the caller's responsibility and is returned unchanged, so a
    missing custom dir still surfaces sherpa-onnx's FileNotFoundError. Relative
    paths are anchored to the repo root so the download lands in models/
    regardless of the cwd the CLI is launched from."""
    if model_dir != config.ASR_MODEL_DIR:
        return model_dir
    path = Path(model_dir)
    target = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if _model_ready(target):
        return str(target)
    _download_default_asr_model(target)
    return str(target)


def _recognizer(model_dir: str = None, model_type: str = None):
    global _REC, _REC_MODEL_DIR
    model_dir = _ensure_asr_model(model_dir or config.ASR_MODEL_DIR)
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
