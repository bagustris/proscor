"""Align target vs. recognized words, compute phoneme edit distance -> 0-100 score.

This is an ASR-based **intelligibility** score (phoneme edit distance), not
Goodness-of-Pronunciation. See PLAN.md section 5 for the optional GOP track.
"""
import re

from rapidfuzz.distance import Levenshtein

from proscor.config import CONF_WEIGHT, PHONEME_WEIGHT
from proscor.g2p import expected_phonemes


def _clean(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def _phoneme_edits(expected: list, heard: list) -> list:
    """Return a list of {"op", "at", "expected", "heard"} edits (Levenshtein ops)."""
    edits = []
    for op in Levenshtein.opcodes(expected, heard):
        if op.tag == "equal":
            continue
        exp_slice = expected[op.src_start:op.src_end]
        heard_slice = heard[op.dest_start:op.dest_end]
        n = max(len(exp_slice), len(heard_slice))
        for i in range(n):
            edits.append({
                "op": "sub" if exp_slice[i:i + 1] and heard_slice[i:i + 1] else
                      ("del" if exp_slice[i:i + 1] else "ins"),
                "at": op.src_start + i,
                "expected": exp_slice[i] if i < len(exp_slice) else None,
                "heard": heard_slice[i] if i < len(heard_slice) else None,
            })
    return edits


def _word_score(expected_ph: list, heard_ph: list) -> float:
    if not expected_ph and not heard_ph:
        return 100.0
    denom = max(len(expected_ph), len(heard_ph), 1)
    edits = Levenshtein.distance(expected_ph, heard_ph)
    return max(0.0, 1 - edits / denom) * 100


def _align_words(target_words: list, recognized_words: list) -> list:
    """Monotone word-level alignment: for each target word, the best-matching
    recognized word (or None if deleted), via Levenshtein alignment over words."""
    ops = Levenshtein.opcodes(target_words, recognized_words)
    aligned = [None] * len(target_words)
    for op in ops:
        if op.tag in ("equal", "replace"):
            for i in range(op.src_end - op.src_start):
                t_idx = op.src_start + i
                r_idx = op.dest_start + min(i, op.dest_end - op.dest_start - 1)
                if op.dest_end > op.dest_start:
                    aligned[t_idx] = recognized_words[r_idx]
        # "delete": target word has no match -> aligned[t_idx] stays None
        # "insert": extra recognized word, not attached to any target word
    return aligned


def score(target_text: str, asr_result: dict, include_stress: bool = False, has_confidence: bool = True) -> dict:
    """Compute a ScoreReport comparing `target_text` to an ASR result
    (`{"text": str, "words": [{"word", "conf", "start", "end"}]}`)."""
    target_words = target_text.split()
    recognized = asr_result.get("words") or []
    recognized_words = [_clean(w["word"]) for w in recognized if _clean(w["word"])]

    if not recognized_words:
        return {
            "score": 0.0,
            "words": [
                {
                    "target": tw, "recognized": None, "correct": False,
                    "word_score": 0.0,
                    "phonemes_expected": expected_phonemes(tw, include_stress)[0] if tw else [],
                    "phonemes_heard": [], "edits": [],
                }
                for tw in target_words
            ],
            "notes": "nothing recognized",
        }

    target_clean = [_clean(w) for w in target_words]
    aligned = _align_words(target_clean, recognized_words)
    conf_by_word = {_clean(w["word"]): w.get("conf", 1.0) for w in recognized}

    words_report = []
    mispronounced = 0
    for target_word, recognized_word in zip(target_words, aligned):
        expected_ph = expected_phonemes(target_word, include_stress)
        expected_ph = expected_ph[0] if expected_ph else []
        if recognized_word is None:
            heard_ph = []
            correct = False
        else:
            heard_ph_list = expected_phonemes(recognized_word, include_stress)
            heard_ph = heard_ph_list[0] if heard_ph_list else []
            correct = _clean(target_word) == recognized_word

        w_score = _word_score(expected_ph, heard_ph)
        if has_confidence and recognized_word is not None:
            conf = conf_by_word.get(recognized_word, 1.0)
            w_score = PHONEME_WEIGHT * w_score + CONF_WEIGHT * conf * 100

        edits = _phoneme_edits(expected_ph, heard_ph) if not correct else []
        if not correct:
            mispronounced += 1

        words_report.append({
            "target": target_word,
            "recognized": recognized_word,
            "correct": correct,
            "word_score": round(w_score, 1),
            "phonemes_expected": expected_ph,
            "phonemes_heard": heard_ph,
            "edits": edits,
        })

    overall = round(sum(w["word_score"] for w in words_report) / len(words_report), 1) if words_report else 0.0
    notes = "all words correct" if mispronounced == 0 else f"{mispronounced} of {len(words_report)} words mispronounced"

    return {"score": overall, "words": words_report, "notes": notes}
