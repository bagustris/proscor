from proscor.feedback import format_report, phoneme_hint


def test_format_report_all_correct():
    report = {
        "score": 100.0,
        "words": [
            {"target": "thought", "recognized": "thought", "correct": True, "word_score": 100.0,
             "phonemes_expected": ["TH", "AO", "T"], "phonemes_heard": ["TH", "AO", "T"], "edits": []},
        ],
        "notes": "all words correct",
    }
    out = format_report(report)
    assert "Score: 100/100" in out
    assert "OK   thought" in out


def test_format_report_miss_shows_substitution_hint():
    report = {
        "score": 33.0,
        "words": [
            {"target": "through", "recognized": "true", "correct": False, "word_score": 33.0,
             "phonemes_expected": ["TH", "R", "UW"], "phonemes_heard": ["T", "R", "UW"],
             "edits": [{"op": "sub", "at": 0, "expected": "TH", "heard": "T"}]},
        ],
        "notes": "1 of 1 words mispronounced",
    }
    out = format_report(report)
    assert "Score: 33/100" in out
    assert 'MISS through    -> heard "true"' in out
    assert "TH -> T" in out


def test_phoneme_hint():
    assert "think" in phoneme_hint("TH")
    assert phoneme_hint("ZZZ") == "ZZZ"
