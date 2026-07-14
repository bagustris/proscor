"""Reference pronunciation audio via sherox.tts (hear-the-target)."""
import io
from pathlib import Path

from proscor.config import ROOT, TTS_LANG

_TTS = None
_TTS_LANG = None


def _build_tts(lang: str):
    try:
        from sherox.tts import TtsConfig, build_tts
    except ImportError as exc:
        raise ImportError(
            "sherox is required for TTS. Install the sibling repo: "
            "`pip install -e ../audiokit && pip install -e ../sherox` "
            "(fallback: `pip install git+https://github.com/bagustris/sherox`). "
            "See PLAN.md section 2."
        ) from exc
    cfg = TtsConfig(model_dir="", language=lang, num_threads=1)
    return build_tts(cfg, ROOT)


def _get_tts(lang: str = TTS_LANG):
    global _TTS, _TTS_LANG
    if _TTS is None or _TTS_LANG != lang:
        _TTS = _build_tts(lang)
        _TTS_LANG = lang
    return _TTS


def synthesize(
    text: str, lang: str = TTS_LANG, speed: float = 1.0, speaker_id: int = 0,
    out_path: str = None,
):
    """Synthesize `text` -> (float32 samples, sample_rate)."""
    from sherox.tts import TtsConfig, synthesise_to_file

    tts = _get_tts(lang)
    cfg = TtsConfig(
        model_dir="", language=lang, speaker_id=speaker_id, speed=speed,
        output=out_path or "", play=False, no_save=out_path is None,
        num_threads=1, audio_prompt="", audio_prompt_text="",
    )
    return synthesise_to_file(tts, text, cfg)


def play_reference(text: str, lang: str = TTS_LANG, speed: float = 1.0) -> None:
    """Synthesize `text` and play it through the default output device."""
    import sounddevice as sd

    samples, sr = synthesize(text, lang=lang, speed=speed)
    sd.play(samples, sr)
    sd.wait()


def reference_bytes(text: str, lang: str = TTS_LANG, speed: float = 1.0) -> bytes:
    """Synthesize `text` and return WAV bytes (for the web /api/reference endpoint)."""
    import soundfile as sf

    samples, sr = synthesize(text, lang=lang, speed=speed)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()
