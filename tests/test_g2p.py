from pathlib import Path

from proscor.g2p import expected_phonemes


def test_hello():
    assert expected_phonemes("hello", lexicon_path=Path("/nonexistent")) == [
        ["HH", "AH", "L", "OW"]
    ]


def test_lexicon_override_wins(tmp_path):
    lex = tmp_path / "lexicon.txt"
    lex.write_text("HELLO\tH EH L OW\n")
    assert expected_phonemes("hello", lexicon_path=lex) == [["H", "EH", "L", "OW"]]


def test_lexicon_whitespace_and_tab(tmp_path):
    lex = tmp_path / "lexicon.txt"
    lex.write_text("FOO\tF UW\nBAR    B AA R\n# comment\n\n")
    assert expected_phonemes("foo bar", lexicon_path=lex) == [
        ["F", "UW"],
        ["B", "AA", "R"],
    ]
