# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Note: the versions below are not yet git-tagged. Tag them (e.g.
> `git tag v1.0.0`) to make them official.

## [Unreleased]

### Added
- **Sentence-level GOP-lite (word granularity)** (`proscor/align.py`): CTC
  Viterbi forced alignment (`_ctc_viterbi`) and `align_words_gop`, which
  force-aligns a full utterance's canonical words against the audio (greedy
  BPE segmentation per word) and returns a per-word Goodness-of-Pronunciation
  score (mean posterior deficit over the word's aligned frames). Extends the
  existing single-word forced-alignment scoring (`score_word`) to multi-word
  utterances; see `PLAN.md` section 5a.
- **`scripts/eval_so762.py`**: validates GOP-lite against
  [speechocean762](https://huggingface.co/datasets/mispeech/speechocean762)'s
  human per-word/utterance accuracy labels. Full `test` split (2500
  utterances, 15,967 words): word-level Pearson r = 0.47, utterance-level r =
  0.56-0.59 (int8; fp32 near-identical, see `PLAN.md` section 5a). Zero-shot,
  ~75-90% of trained GOPT's PCC (Gong et al., ICASSP 2022) at word/utterance
  level. New optional `requirements-eval.txt` (huggingface_hub, pyarrow,
  scipy) for this script only; new `ALIGN_USE_INT8` config toggle
  (`proscor/config.py`) for int8 vs. fp32 alignment weights.
- Offline unit tests for the new alignment functions in `tests/test_align.py`
  (Viterbi backtracking, greedy BPE segmentation, `align_words_gop` with a
  stubbed ONNX session).
- **`--engine gop-lite`**: `proscor.score.score_gop_lite` + `score_audio(...,
  engine=...)` wire the word-level GOP-lite scorer into the app (falls back
  to the intelligibility engine if the alignment extras aren't installed).
  Exposed as `cli.py --engine {intelligibility,gop-lite}`, an `engine` form
  field on `POST /api/score`, and a scoring-engine dropdown in
  `web/static/index.html`. Because GOP-lite force-aligns to the target
  rather than free-decoding, `feedback.format_report` (CLI) now renders a
  low-scoring GOP-lite word as a `LOW word [phones] fit NN/100` line and the
  web results table shows `(fit NN/100)`, instead of a `MISS ... -> heard
  "X"` line/cell that would misreport what was actually heard.
- **Phone-level GOP-lite** (`proscor/align_phone.py`, evaluation-only, not
  wired into the app): forced alignment against a genuine phoneme-CTC ONNX
  model (`wav2vec2-lv-60-espeak-cv-ft`, espeak-ng IPA output), with targets
  phonemized live via `phonemizer`/espeak-ng rather than a hand-built
  ARPABET→IPA table. `scripts/eval_so762_phone.py`: full `test` split
  reconciled phone-level Pearson r = 0.433 at 99.73% coverage (98.4%/96.2%
  of the classic trained RF/SVR baselines' PCC, zero-shot); confirmed (not
  lower) on the untouched `train` split at the pre-reconciliation
  matched-only metric (r = 0.476). Word/utterance-level PCC
  (0.325 / 0.536 / 0.572) trail the simpler BPE model, so the BPE model
  stays the shipped `--engine gop-lite` default; see `PLAN.md` section 5a
  item 4 for the full comparison, the root-caused aggregation bug fix along
  the way (word GOP as mean of per-phone GOP, not mean over the whole frame
  span including blanks), and the `reconcile_phones` constrained-alignment
  design (1:1 ARPABET/IPA equivalence matches plus 2:1/3:1 merges for three
  known espeak segmentation patterns, plus principled deletions for
  phones with no espeak counterpart at all, e.g. yod-dropping). New
  `requirements-eval.txt` entry: `phonemizer` (needs system espeak-ng).
- **Second-corpus generalization: UME-ERJ** (`scripts/eval_umeerj.py`,
  PLAN.md section 5b): validates both GOP-lite engines against
  [UME-ERJ](https://research.nii.ac.jp/src/en/UME-ERJ.html) (Japanese-L1
  English speakers — a different L1 than speechocean762's Mandarin-L1
  children). No per-phone labels exist in this corpus, so it's a
  word/utterance-level-only check. Segmental correlations hold in the same
  band as speechocean762 (BPE 0.29-0.33, phone model 0.43) across 9,484
  rated items with zero failures — but the word/utterance-level ranking
  between the two engines **flips** relative to speechocean762 (phone model
  wins here; BPE model won there), with two plausible, non-separable
  explanations (acoustic/age domain match vs. L1-specific error profile).
  Rhythm/intonation/stress correlate weakly for both engines, as expected —
  GOP measures phone/word identity fit, not prosody.
- **Disentangling age vs. L1 (`scripts/eval_l2arctic.py`, PLAN.md section
  5c):** validates against L2-ARCTIC's adult Mandarin-L1 speakers (same L1
  as speechocean762, same age category as UME-ERJ) to test which variable
  drove the section 5b ranking flip. Result: a near-tie (BPE -0.107, phone
  -0.119, n=600; per-speaker breakdown shows 3 different winners across 4
  speakers) that rules out "L1 alone" but doesn't confirm "age alone" —
  correlations here are 2-3x weaker than the other two corpora (a coarser
  proxy signal: character-edit-distance between canonical/perceived IPA,
  not an expert score), so the honest conclusion is the test is
  underpowered at this effect size, not that the question is resolved. A
  sharper phone-level follow-up (using the original TAMU/Kaggle
  distribution's per-phone TextGrid annotations) is scoped but not built.

### Fixed
- `proscor/tts.py`: `synthesize()` passed `audio_prompt`/`audio_prompt_text`
  to `sherox.tts.TtsConfig`, which the installed `sherox` version's
  `TtsConfig` no longer accepts (`TypeError: unexpected keyword argument`).
  Broke reference-audio playback everywhere it's used: the CLI's `p`lay
  command, the web app's `/api/reference` endpoint, and
  `scripts/selftest.py`. Dropped the two removed fields.
- `proscor/align_phone.py`: `_phonemize_word` fed words to espeak-ng in
  their original case; speechocean762's all-caps transcripts triggered
  espeak's acronym heuristic on two words ("IT" -> spelled out "I-T", "US"
  -> "U-S", instead of pronounced), corrupting their target phones for
  2.14% of test-split word occurrences. Now lowercases before phonemizing;
  regression test in `tests/test_align_phone.py`. Full re-evaluation after
  the fix: every phone-level GOP-lite number rose slightly (see the
  `scripts/eval_so762_phone.py` entry above).
- `proscor/align.py`: `_ctc_viterbi`'s backtracking loop did
  `s -= back[t, s]`, mixing a plain Python state-index int with a NumPy
  `int8` value; under NumPy 2's stricter type-promotion rules this raised
  `OverflowError: Python integer N out of bounds for int8` once a state
  index exceeded 127 (utterances with more than ~64 tokens — silent
  wraparound instead of a crash under NumPy 1.x). speechocean762's shorter
  utterances never triggered it; UME-ERJ's longer sentences did (0.4% of
  one category). Affects the shipped `--engine gop-lite` path
  (`proscor.align`), not just the eval scripts — `proscor.align_phone`
  reuses the same function. Fixed with an explicit `int(...)` cast;
  regression test in `tests/test_align.py`.

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
