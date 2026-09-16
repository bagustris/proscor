import numpy as np
import pytest

from proscor import align, score as score_module
from proscor.score import score, score_audio, score_gop_lite


def _fake_asr(words):
    return {
        "text": " ".join(words),
        "words": [{"word": w, "conf": 1.0, "start": i * 1.0, "end": i * 1.0 + 0.5} for i, w in enumerate(words)],
    }


def test_all_correct():
    report = score("thought through", _fake_asr(["thought", "through"]))
    assert report["score"] == 100.0
    assert all(w["correct"] for w in report["words"])


def test_one_substitution():
    # "through" (TH R UW) misheard as "true" (T R UW) -> one phoneme sub
    report = score("thought through", _fake_asr(["thought", "true"]))
    assert report["score"] < 100.0
    words = {w["target"]: w for w in report["words"]}
    assert words["thought"]["correct"] is True
    assert words["through"]["correct"] is False
    assert len(words["through"]["edits"]) >= 1


def test_one_deletion():
    report = score("she sells sea shells", _fake_asr(["she", "sells", "shells"]))
    assert report["score"] < 100.0
    words = {w["target"]: w for w in report["words"]}
    assert words["sea"]["recognized"] is None
    assert words["sea"]["correct"] is False


def test_nothing_recognized():
    report = score("hello world", {"text": "", "words": []})
    assert report["score"] == 0.0
    assert report["notes"] == "nothing recognized"


def test_score_gop_lite_maps_gop_to_word_score(monkeypatch):
    def fake_align_words_gop(samples, words, sr=16000, model_dir=None):
        # "hello" fits well (gop=0), "world" doesn't (gop very negative)
        return [{"gop": 0.0, "n_frames": 5}, {"gop": -100.0, "n_frames": 5}]

    monkeypatch.setattr(align, "align_words_gop", fake_align_words_gop)
    report = score_gop_lite("hello world", np.zeros(1600, dtype=np.float32))

    words = {w["target"]: w for w in report["words"]}
    assert words["hello"]["word_score"] == 100.0
    assert words["hello"]["correct"] is True
    assert words["world"]["word_score"] == pytest.approx(0.0, abs=1e-6)
    assert words["world"]["correct"] is False
    # force-aligned to the target either way -- never "heard something else"
    assert words["hello"]["recognized"] == "hello"
    assert words["world"]["recognized"] == "world"
    assert report["score"] == pytest.approx(50.0, abs=0.1)


def test_score_gop_lite_handles_unaligned_word(monkeypatch):
    def fake_align_words_gop(samples, words, sr=16000, model_dir=None):
        return [{"gop": 0.0, "n_frames": 5}, None]

    monkeypatch.setattr(align, "align_words_gop", fake_align_words_gop)
    report = score_gop_lite("hello world", np.zeros(1600, dtype=np.float32))

    words = {w["target"]: w for w in report["words"]}
    assert words["world"]["word_score"] == 0.0
    assert words["world"]["correct"] is False


def test_score_audio_dispatches_to_gop_lite_engine(monkeypatch):
    called = {}

    def fake_score_gop_lite(target_text, samples, sr=16000, include_stress=False, model_dir=None):
        called["used"] = True
        return {"score": 42.0, "words": [], "notes": "gop-lite"}

    monkeypatch.setattr(align, "available", lambda: True)
    monkeypatch.setattr(score_module, "score_gop_lite", fake_score_gop_lite)
    report = score_audio("hello world", np.zeros(1600, dtype=np.float32), engine="gop-lite")

    assert called.get("used") is True
    assert report["score"] == 42.0


def test_score_audio_gop_lite_falls_back_when_align_unavailable(monkeypatch):
    from proscor import asr

    monkeypatch.setattr(align, "available", lambda: False)
    monkeypatch.setattr(asr, "transcribe", lambda *a, **kw: {"text": "", "words": []})
    report = score_audio("hello world", np.zeros(1600, dtype=np.float32), engine="gop-lite")
    # falls back to the intelligibility path (stubbed transcribe -> "nothing recognized")
    assert report["score"] == 0.0
    assert report["notes"] == "nothing recognized"


def test_score_audio_rejects_unknown_engine():
    with pytest.raises(ValueError):
        score_audio("hello", np.zeros(1600, dtype=np.float32), engine="bogus")
