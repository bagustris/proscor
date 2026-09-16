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


def test_format_report_low_fit_does_not_claim_something_else_was_heard():
    """GOP-lite (proscor.score.score_gop_lite) force-aligns to the target, so
    recognized == target even when the fit is poor -- format_report must not
    render that as a MISS/"heard X instead" line."""
    report = {
        "score": 41.0,
        "words": [
            {"target": "see", "recognized": "see", "correct": False, "word_score": 41.0,
             "phonemes_expected": ["S", "IY"], "phonemes_heard": ["S", "IY"], "edits": []},
        ],
        "notes": "1 of 1 words below threshold (gop-lite)",
    }
    out = format_report(report)
    assert "LOW  see" in out
    assert "fit 41/100" in out
    assert "heard" not in out.lower()


def test_phoneme_hint():
    assert "think" in phoneme_hint("TH")
    assert phoneme_hint("ZZZ") == "ZZZ"
