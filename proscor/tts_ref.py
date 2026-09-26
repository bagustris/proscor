"""Synthetic *reference* voices for template-based scoring (`dtw_ssl`) when
no native recording of a prompt exists (PLAN.md section 5m) -- evaluation-
only, separate from `proscor/tts.py` (the learner-facing "hear the target"
path via sherox), and it does not touch sherox.

A *voice set* is a named list of `(model, speaker_id)` pairs; `synthesize`
runs one of them through `sherpa_onnx`. Sets differ in synthesis quality and
speaker diversity, which is exactly what section 5m compares: the original
Kitten-Nano set (`proscor.tts`, int8, 8 voices) vs. higher-quality models.
Model archives come from the k2-fsa/sherpa-onnx `tts-models` release and are
unpacked under `models/ref_tts/` (gitignored); `ensure_model` downloads on
first use.
"""
import subprocess
from pathlib import Path

import numpy as np

from proscor.config import ROOT

MODEL_DIR = Path(ROOT) / "models" / "ref_tts"
_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{name}.tar.bz2"

# family: "vits" (piper/coqui VITS) or "kokoro"
_MODELS = {
    "kokoro": ("kokoro-int8-en-v0_19", "kokoro"),
    "libritts": ("vits-piper-en_US-libritts_r-medium-int8", "vits"),
    "arctic": ("vits-piper-en_US-arctic-medium-int8", "vits"),
    "lessac": ("vits-piper-en_US-lessac-high-int8", "vits"),
    "ryan": ("vits-piper-en_US-ryan-high-int8", "vits"),
    "joe": ("vits-piper-en_US-joe-medium-int8", "vits"),
    "john": ("vits-piper-en_US-john-medium-int8", "vits"),
    # British-English Piper voices (en_GB phonemization: non-rhotic etc.) -- the
    # controlled accent comparison against "piper4" in PLAN.md section 5m.
    "cori": ("vits-piper-en_GB-cori-high-int8", "vits"),
    "alan": ("vits-piper-en_GB-alan-medium-int8", "vits"),
    "jenny": ("vits-piper-en_GB-jenny_dioco-medium-int8", "vits"),
    "northern": ("vits-piper-en_GB-northern_english_male-medium-int8", "vits"),
}

# Named voice sets. libritts ids are arbitrary but fixed LibriTTS-R speakers
# (spread across the id range); kokoro ids 0-10 are its bundled voices.
VOICE_SETS = {
    "kokoro4": [("kokoro", 0), ("kokoro", 2), ("kokoro", 4), ("kokoro", 6)],
    "libritts4": [("libritts", 10), ("libritts", 200), ("libritts", 400), ("libritts", 700)],
    "arctic4": [("arctic", 0), ("arctic", 4), ("arctic", 8), ("arctic", 12)],
    "piper4": [("lessac", 0), ("ryan", 0), ("joe", 0), ("john", 0)],
    "piper_gb4": [("cori", 0), ("alan", 0), ("jenny", 0), ("northern", 0)],
}

_TTS = {}


def ensure_model(key: str) -> Path:
    name, _family = _MODELS[key]
    d = MODEL_DIR / name
    if not d.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tar = MODEL_DIR / f"{name}.tar.bz2"
        subprocess.run(["curl", "-sL", "-o", str(tar), _URL.format(name=name)], check=True)
        subprocess.run(["tar", "xjf", str(tar), "-C", str(MODEL_DIR)], check=True)
        tar.unlink()
    return d


def _get(key: str, num_threads: int = 2):
    if key not in _TTS:
        import sherpa_onnx

        d = ensure_model(key)
        family = _MODELS[key][1]
        if family == "kokoro":
            mcfg = sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(d / "model.int8.onnx"), voices=str(d / "voices.bin"),
                    tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")),
                num_threads=num_threads)
        else:
            onnx = next(p for p in d.glob("*.onnx"))
            mcfg = sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(onnx), tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")),
                num_threads=num_threads)
        cfg = sherpa_onnx.OfflineTtsConfig(model=mcfg)
        if not cfg.validate():
            raise RuntimeError(f"invalid sherpa-onnx TTS config for {key}")
        _TTS[key] = sherpa_onnx.OfflineTts(cfg)
    return _TTS[key]


def synthesize(voice: tuple, text: str, speed: float = 1.0):
    """`voice` = (model key, speaker id) -> (float32 samples, sample rate)."""
    key, sid = voice
    audio = _get(key).generate(text=text.lower(), sid=sid, speed=speed)
    return np.array(audio.samples, dtype=np.float32), audio.sample_rate
