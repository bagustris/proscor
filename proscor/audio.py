"""Record (CLI) / load WAV (web). Always normalized to mono, 16 kHz, 16-bit."""
import numpy as np
import soundfile as sf

from proscor.config import SAMPLE_RATE


def record(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Record `seconds` of mono audio from the default input device."""
    import sounddevice as sd

    samples = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    return samples.reshape(-1)


def save_wav(path: str, samples: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf.write(path, samples, sr, subtype="PCM_16")


def load_wav(path: str):
    """Load a WAV file, resampled to mono 16 kHz 16-bit if needed."""
    samples, sr = sf.read(path, dtype="int16", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        from audiokit import resample

        samples = resample(samples.astype(np.float32), sr, SAMPLE_RATE).astype(np.int16)
        sr = SAMPLE_RATE
    return samples, sr
