"""Text -> expected ARPABET phonemes (g2p_en + CMUdict, with lexicon overrides)."""
import re
from pathlib import Path

from proscor.config import DEFAULT_LEXICON_PATH

_G2P = None


def _ensure_nltk_data() -> None:
    """g2p_en needs these NLTK corpora; auto-fetch if missing rather than
    crashing with an opaque LookupError (resource names vary across NLTK
    versions, e.g. `averaged_perceptron_tagger` vs. `..._eng`)."""
    import nltk

    for find_path, download_name in (
        ("corpora/cmudict", "cmudict"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ):
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


def _get_g2p():
    global _G2P
    if _G2P is None:
        _ensure_nltk_data()
        from g2p_en import G2p
        _G2P = G2p()
    return _G2P


def _strip_stress(phoneme: str) -> str:
    return phoneme[:-1] if phoneme and phoneme[-1].isdigit() else phoneme


def _clean_word(word: str) -> str:
    return re.sub(r"[^A-Za-z']", "", word)


def _load_lexicon(path: Path) -> dict:
    lexicon = {}
    if not path or not Path(path).exists():
        return lexicon
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        word, phones = parts
        lexicon[word.upper()] = [p.upper() for p in phones.split()]
    return lexicon


def expected_phonemes(
    text: str,
    include_stress: bool = False,
    lexicon_path: Path = DEFAULT_LEXICON_PATH,
) -> list:
    """Per-word ARPABET phonemes for `text`. Returns a list of phoneme lists,
    one per whitespace-separated word. Stress is stripped by default."""
    lexicon = _load_lexicon(lexicon_path)
    g2p = _get_g2p()
    words_phonemes = []
    for word in text.split():
        clean = _clean_word(word)
        if not clean:
            continue
        key = clean.upper()
        if key in lexicon:
            phones = list(lexicon[key])
        else:
            raw = g2p(clean)
            phones = [p for p in raw if re.match(r"^[A-Z]+[0-9]?$", p)]
        if not include_stress:
            phones = [_strip_stress(p) for p in phones]
        words_phonemes.append(phones)
    return words_phonemes
