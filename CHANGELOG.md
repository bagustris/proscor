# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Note: the versions below are not yet git-tagged. Tag them (e.g.
> `git tag v1.0.0`) to make them official.

## [Unreleased]

## [1.0.1] - 2026-07-14

### Added
- `proscor/g2p.py` now auto-fetches the NLTK corpora it needs (`cmudict`,
  `averaged_perceptron_tagger_eng`) on first use instead of crashing with an
  opaque `LookupError` (resource names vary across NLTK versions).

### Changed
- Install instructions for the `sherox` sibling dependency: the fallback is now
  `pip install git+https://github.com/bagustris/sherox` (sherox is **not** on
  PyPI). Updated in `README.md`, `PLAN.md`, and the `tts.py` import error
  message.

### Fixed
- Web score report (`web/static/index.html`) now builds rows with DOM APIs
  (`createElement` / `textContent`) instead of `innerHTML`, preventing HTML
  injection from ASR-recognized text.

### Removed
- Japanese removed from the project scope and multi-language plan. The future
  language roadmap is now Indonesian and Arabic only (see `PLAN.md` section 9).
  Affects `PLAN.md` and the `README.md` "Scope" section.

## [1.0.0] - 2026-07-13

First complete release: an offline, CPU-only English pronunciation scorer with
a CLI and a web app. Show a prompt, read it aloud, get a 0–100 score with
word-by-word phoneme feedback.

### Added
- **G2P** (`proscor/g2p.py`): target text → expected phonemes via `g2p_en` +
  CMUdict, with `data/lexicon.txt` overrides.
- **Reference TTS** (`proscor/tts.py`): synthetic reference pronunciation audio
  (Piper TTS via `sherox`) so learners can hear the target.
- **Audio** (`proscor/audio.py`): record / load WAV, normalized to mono 16 kHz.
- **ASR** (`proscor/asr.py`): offline transcription via `sherox` (sherpa-onnx,
  NeMo CTC Conformer English), with per-word confidence.
- **Scoring** (`proscor/score.py`): align recognized words to target words and
  compare phoneme-by-phoneme (edit distance) → 0–100 per word and overall.
- **Feedback** (`proscor/feedback.py`): human-readable report of mismatches
  with expected vs. heard phonemes and plain-English hints (e.g. `TH as in think`).
- **Prompts** (`proscor/prompts.py`): prompt list loading and selection
  (`data/prompts.txt`).
- **Config** (`proscor/config.py`): single source for paths, ASR model choice,
  and scoring weights.
- **CLI** (`cli.py`): interactive loop — play reference, record, score, retry,
  next prompt.
- **Web app** (`web/server.py` + `web/static/index.html`): FastAPI backend +
  vanilla-JS frontend to play reference, record in-browser, and view the score.
- **Evaluation & QA**: offline unit tests (`tests/`) and an end-to-end
  `scripts/selftest.py` that synthesizes clean + degraded audio and checks
  scores land in expected ranges.
- **Docs**: `README.md` and `PLAN.md` (full design rationale and build history).

[Unreleased]: https://github.com/bagustris/proscor/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/bagustris/proscor/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/bagustris/proscor/releases/tag/v1.0.0
