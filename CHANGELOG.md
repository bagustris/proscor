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
  speakers, who are mixed-age 6-43, not children as this entry originally
  said; see the section 5e entry). No per-phone labels exist in this corpus, so it's a
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
  underpowered at this effect size, not that the question is resolved.
- **Phone-level L2-ARCTIC (`scripts/eval_l2arctic_phone.py`, PLAN.md
  section 5c):** the sharper follow-up using the original TAMU/Kaggle
  distribution's expert-aligned per-phone TextGrid annotations (fetched
  per-file via the Kaggle API; whole-dataset downloads hit Google
  Drive/Kaggle rate limits). Phone-level PCC = 0.206 (n=19,636 phones, 4
  Mandarin speakers, zero failures) — lower than speechocean762's
  0.425/0.433, partly explained by a checked asymmetry: GOP-lite catches
  deletions (mean GOP -2.30) much better than substitutions (-1.11, median
  0.0 — same as correct phones), and L2-ARCTIC's Mandarin-L1 errors are
  ~4:1 substitution-heavy (classic L1-transfer, e.g. θ→s, r→l). Caught and
  fixed a real bug first: the error-tag regex didn't match ARPABET stress
  digits, silently mislabeling vowel deletions/substitutions as correct
  (BWC alone: r=0.151 buggy -> 0.233 fixed, a 54% change from one regex
  character class).
- **Full-corpus L2-ARCTIC phone-level (all 24 speakers, 6 L1s,
  `results/l2arctic_phone_full.json`, PLAN.md section 5c):** downloaded
  the remaining 20 speakers (7,198 files total via the Kaggle API,
  weathering a sustained rate-limit spiral with slower per-file pacing;
  zero permanent failures). Refactored `scripts/eval_l2arctic_phone.py`
  to report per-speaker/per-language, not just pooled (regression-tested:
  Mandarin-only subset unchanged at r=0.2061). Pooled PCC = 0.224 (n =
  118,455 phones), but that hides a per-language range of 0.06-0.37:
  Vietnamese 0.371, Arabic 0.219, Mandarin 0.206, Spanish 0.186, Korean
  0.103, Hindi 0.060. **Correction to the first cut of this entry:**
  "ranking languages by the substitution/correct GOP gap reproduces the
  PCC ranking exactly, 6/6" was overclaimed as an independent finding --
  it's largely the point-biserial formula's own numerator (substitutions
  are 80-93% of errors everywhere, so the "gap" is most of what the
  correlation is computed from). What's still real: substitution-error
  mean GOP varies -0.42 (Hindi) to -1.58 (Vietnamese) while mean
  correct-phone GOP is flat (-0.28 to -0.40) across L1s, and since
  substitutions dominate every language's errors, that's *where* the
  per-language PCC range numerically lives -- stated as arithmetic
  location, not a mechanistic discovery. It's also not just an
  error-rate artifact (Arabic has the fewest errors of all 6 languages
  yet the second-highest PCC). This run only used the phone
  model (BPE has no phone output), so it can't repeat the 5b BPE-vs-phone
  ranking-flip test per L1 — but it shows L1-specific error profile is a
  large, real effect independent of age (all 24 L2-ARCTIC speakers are
  adults), so that explanation can't be dismissed as minor next to
  age/domain match.
- **Cluster-bootstrap CIs and paired-difference significance tests for
  every "engine A beats engine B" claim** (`proscor/stats.py`, unit-tested
  in `tests/test_stats.py`; PLAN.md section 5d). Every prior correlation
  was a point estimate at the raw item count, which clusters within
  speakers (naive Fisher-z would be badly wrong) -- added a speaker-level
  percentile bootstrap (`cluster_bootstrap_pearson`), a paired bootstrap
  on the difference for two engines scored on the same items
  (`cluster_bootstrap_paired_diff`, stronger than comparing two marginal
  CIs since correlated engine errors are preserved per resample), and a
  Kruskal-Wallis test on per-speaker r's for "does language have a real
  effect" (`kruskal_by_group`). Reran so762 (`scripts/eval_so762.py`,
  `eval_so762_phone.py`, and new `scripts/eval_so762_paired.py`),
  UME-ERJ, and both L2-ARCTIC scripts to capture speaker IDs and compute
  these. Headline results: the section 5a word-level "BPE beats phone"
  gap is real (paired diff CI 0.102-0.186, excludes 0) but the
  utterance-level gap is NOT (CI -0.043 to 0.090) -- that number
  shouldn't have been read as a real margin. The section 5b UME-ERJ
  ranking flip (phone beats BPE) is real for both segmental categories
  (CIs exclude 0), confirming the central cross-corpus finding isn't
  point-estimate noise. The section 5c Chinese near-tie holds up as a
  genuine tie (paired diff CI includes 0) rather than an underpowered
  guess, though the other 5 languages' CER-proxy comparisons are too
  noisy to read individually (wrong-signed or non-significant marginals
  in several cases). The section 5c phone-level per-language spread is
  real overall (Kruskal-Wallis p=0.0115) but only the extremes (Hindi
  lowest, Vietnamese highest) clearly separate -- the middle four
  languages have heavily overlapping 4-speaker-cluster CIs and shouldn't
  be read as a precise 6-way ranking. Also ran the binarized-label check
  flagged as owed: so762's phone PCC drops from 0.433 (graded label) to
  0.337 (binarized like L2-ARCTIC's), closing about half the apparent gap
  to L2-ARCTIC's 0.224 -- real, but a smaller real gap than it looked.
- **Age ruled out as the driver of the BPE-vs-phone ranking flip**
  (PLAN.md section 5e). Correction first: speechocean762 is mixed-age
  (`age` 6-43; test split 64 speakers under 18, 61 over), not the
  children's corpus sections 5a-5c called it -- those passages are
  amended in place. `scripts/eval_so762_paired.py` now records speaker
  age, splits the paired speaker-cluster comparison at 18, and saves
  per-item arrays next to the summary (`<out>.arrays.json`) so re-cuts
  don't need another inference run. BPE beats the phone model at word
  level in both groups (under-18 diff +0.124, CI 0.063-0.169; 18+ diff
  +0.171, CI 0.112-0.224), the adult margin being the larger -- so the
  phone model's UME-ERJ win is not an adult-acoustics effect. Also
  checked the L2-ARCTIC annotator confound against the README, the TAMU
  docs page and the Interspeech'18 paper: 3 ISU PhD-student annotators,
  no speaker-to-annotator mapping, no double annotation, no agreement
  figure published -- recorded as a limitation. Parser sanity check
  against the README's official counts: 98.4% of substitutions and
  98.7% of deletions recovered.
- **L1 ruled out too: clean word-level labels for L2-ARCTIC**
  (`scripts/eval_l2arctic_word.py`, `results/l2arctic_word.json`,
  PLAN.md section 5e). Word labels derived from the expert per-phone
  TextGrid tags replace section 5c's char-edit-distance proxy; both
  engines score the same 33,980 words from all 24 adult speakers. BPE
  beats the phone model in all six L1s (pooled diff +0.102, CI
  0.080-0.117; five of six per-L1 CIs exclude zero), so 5c's Mandarin
  "tie" was a proxy artifact. Combined with the age split: the phone
  model wins only on UME-ERJ (Japanese adults) -- the flip is neither
  age nor non-Mandarin L1 in general, and is now localized to that
  corpus (Japanese-L1 specifically, or its holistic rating scheme /
  recording conditions -- not separable with the corpora on hand).
- **Segmentation-free GOP** (`proscor/align_phone.py`: `gop_sf`,
  `align_words_gop_sf`; `--engine sf` in `scripts/eval_so762_phone.py`/
  `eval_l2arctic_phone.py`; PLAN.md section 5f). Targets the substitution
  blindness sections 5c/5e found in posterior-deficit GOP (median GOP
  exactly 0.0 for a substituted phone, every L1) by comparing
  whole-sequence CTC likelihoods marginalized over every alignment
  (Cao et al. 2025, arXiv:2507.16838) instead of scoring Viterbi-aligned
  frames. Computing the substitution term efficiently took two wrong
  attempts, both caught by brute-force comparison in
  `tests/test_align_phone.py` before being trusted: a single forward pass
  with a per-frame log-sum-exp over candidates (overcounted probability
  2x-40x -- lets adjacent frames implicitly vote for different
  candidates, not a valid single path) and a local top-K candidate
  restriction (correct but a weaker, non-comparable metric than the
  paper's full-vocabulary version). Landed on tracking each candidate's
  likelihood in an independent "lane" through the wildcard state's
  self-loop, merged only at entry/exit -- O(T*(S+V)) per phone position,
  matching the paper's complexity, full ~392-symbol vocabulary. Measured
  0.79-0.83s/utterance on full-corpus runs (vs. posterior-deficit's single
  Viterbi pass), so evaluation-only, not wired into `--engine gop-lite`.
  **Full-corpus result:** the mechanism works as predicted -- pooled over
  all 24 L2-ARCTIC speakers, substitution and correct phones had the
  exact same median GOP under posterior-deficit (0.0 = 0.0); under GOP-SF
  they separate for the first time (-0.223 vs. -0.061). But the aggregate
  correlation gain is modest and every CI overlaps its posterior-deficit
  counterpart (so762 phone reconciled: 0.441 vs. 0.433; L2-ARCTIC pooled:
  0.233 vs. 0.224) -- consistent with a small real improvement, not
  proof of one; a proper paired significance test (same items, same
  script, mirroring `scripts/eval_so762_paired.py`) wasn't run yet.
- **Paired posterior-vs-SF significance tests, result: GOP-SF wins,
  proven not just observed**
  (`scripts/eval_so762_phone_paired.py`, `eval_l2arctic_phone_paired.py`):
  the test flagged as owed above -- scores every item with both engines
  in one loop, phone-level pairing verified by asserting
  `reconcile_phones` picks identical alignment ops for both engines'
  GOPs against the same phone-identity sequence (it does, since ops
  depend only on identity, never on the GOP values carried alongside).
  **GOP-SF beats posterior-deficit significantly on all four so762
  metrics** (word: diff -0.0135, CI -0.0207 to -0.0066; utterance:
  -0.0246, CI -0.0324 to -0.0169; phone graded: -0.0083, CI -0.0148 to
  -0.0016; phone binary: -0.0161, CI -0.0207 to -0.0117) **and on pooled
  L2-ARCTIC** (-0.0088, CI -0.0140 to -0.0046, n=118,455 phones/24
  speakers). Per-language L2-ARCTIC: significant in Arabic/Hindi/
  Spanish/Vietnamese, not in Korean/Mandarin (both still point the same
  direction, underpowered at 4 clusters). The marginal-CI overlap that
  looked inconclusive was a real limitation of comparing separately-
  bootstrapped CIs, not evidence of no effect -- the paired test controls
  for the two engines scoring the same, correlated audio.
- **Closing the UME-ERJ open question** (PLAN.md section 5g). Confirmed
  UME-ERJ *is* ERJ (its own `doc/introduction.txt` names itself
  "略称：ERJ データベース") -- no new Japanese-L1 corpus needed, only a
  phone-level annotation layer we don't have. Found the companion
  annotation resource that would supply it (Makino & Aoki 2012, "ERJ
  Phonetic Corpus") but it's a real risk, not a solid lead: the paper
  states under 10% of files were completed as of publication, with no
  later release found and speaker coverage of that subset unstated --
  an email to ask status is drafted for the researcher to send, not
  planned around. **New finding along the way**
  (`scripts/check_gop_peakiness.py`): posterior-deficit GOP's zero-
  inflation (the substitution-blindness mechanism from 5c/5e/5f) is far
  less saturated on UME-ERJ audio than on L2-ARCTIC audio, with no
  correctness labels involved at all -- 55% of UME-ERJ phones pinned at
  exactly 0.0 vs. 85% for L2-ARCTIC (matched 150-utterance samples, same
  engine). A third, previously unconsidered hypothesis for the UME-ERJ
  flip: an acoustic/recording-domain effect on the model's own posterior
  peakiness, independent of L1 or UME-ERJ's holistic rating scale.
  **Correction:** the J-AESOP corpus was called a weak secondary lead
  ("no evidence of phone-level error tags") based on search summaries;
  reading the actual 2025 paper (Yazawa, Konishi & Kondo) found real
  word-level substitution/deletion/insertion tags on 180 Japanese
  speakers -- far more speakers than any corpus used in this plan -- plus
  separate holistic 1-10 ratings on the *same* recordings, which could
  test the L1-vs-methodology question on identical audio. Now the lead
  candidate among the external-contact leads (email drafted for the
  corresponding author); audio/TextGrid access is conditional-on-request,
  not yet public (the rating data alone is, but isn't independently
  useful without matching audio).
- **Checked whether GOP-SF should extend to the BPE model -- no, and two
  alternatives tried didn't help either** (`proscor/align.py`:
  `gop_deletion_term`, `align_words_gop_deletion`; PLAN.md section 5h).
  Confirmed BPE has the same substitution/deletion asymmetry the phone
  model has (milder: substitution's separation from correct is ~45% of
  deletion's, vs. ~28% for phones), but full substitution-marginalization
  GOP-SF doesn't port cleanly (BPE pieces are orthographic chunks, not
  phonetic units, and the vocabulary is much larger). Tried a cheap
  deletion-only term instead (no marginalization needed, reuses
  `_ctc_loglik`) -- checked empirically on 600 L2-ARCTIC utterances, it's
  weaker than the existing score alone (r=0.078 vs. 0.146) and
  wrong-signed against the binary label (-0.080), and combining it with
  the existing score makes both worse. Also tried ensembling BPE with the
  phone model's word-level score (free to test, already on disk) -- best
  case found (85/15 BPE-weighted blend) is a noise-level 0.003 gain on
  one label and a real loss on the other; phone's word-level signal is
  too weak to add value by simple combination. Both negative results,
  kept as tested infrastructure, not wired into any shipped path.

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

### Removed
- **Indonesian/Arabic future-language roadmap** (`PLAN.md` section 9, and
  the corresponding "Scope" note in `README.md`). No code existed for
  either language; this was planning-only. Project focus stays entirely
  on English going forward — no other languages are planned.

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
