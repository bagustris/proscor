"""Human-readable feedback from a ScoreReport (proscor/score.py)."""

# Minimal phoneme -> example-word hint table for common ARPABET confusions.
PHONEME_HINTS = {
    "TH": "th as in think",
    "DH": "th as in this",
    "AE": "a as in cat",
    "AA": "a as in father",
    "AO": "o as in thought",
    "IY": "ee as in see",
    "IH": "i as in bit",
    "UW": "oo as in food",
    "UH": "oo as in book",
    "EH": "e as in bed",
    "ER": "er as in bird",
    "SH": "sh as in ship",
    "ZH": "s as in measure",
    "CH": "ch as in chip",
    "JH": "j as in judge",
    "NG": "ng as in sing",
    "R": "r as in red",
    "L": "l as in led",
    "V": "v as in van",
    "W": "w as in win",
    "Y": "y as in yes",
}


def phoneme_hint(phoneme: str) -> str:
    return PHONEME_HINTS.get(phoneme.upper(), phoneme)


def _word_line(w: dict) -> str:
    expected = " ".join(w["phonemes_expected"])
    if w["correct"]:
        return f"  OK   {w['target']:<10} [{expected}]"

    heard_word = w["recognized"] if w["recognized"] else "(nothing)"
    heard = " ".join(w["phonemes_heard"])
    line = f'  MISS {w["target"]:<10} -> heard "{heard_word}"  [expect {expected} | heard {heard}]'
    subs = [e for e in w["edits"] if e["op"] == "sub" and e["expected"] and e["heard"]]
    if subs:
        hints = ", ".join(f'{e["expected"]} -> {e["heard"]}' for e in subs)
        line += f"  ({hints})"
    return line


def format_report(report: dict) -> str:
    """CLI-friendly human-readable rendering of a ScoreReport."""
    lines = [f"Score: {round(report['score'])}/100"]
    lines.extend(_word_line(w) for w in report["words"])
    return "\n".join(lines)
