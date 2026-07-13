from proscor.score import score


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
