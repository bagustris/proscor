# PLAN.md: English Pronunciation Scoring (proscor-en)

> Show words -> the user reads them aloud -> the system returns a 0-100 score and feedback.
> Target: **simple**, runs on **any PC** (CPU only, no GPU), usable as a **CLI** and a **web app**.
> Scope: **English only in v1.** Indonesian / Arabic are a future TODO (see section 9).

This is a build spec for an autonomous coding agent (e.g., Claude Code / Sonnet).
Implement it step by step, commit after each step, and keep the code working at every step.

---

## 0. Critique of the previous plan (why we changed course)

The earlier version of this file described **building a custom Grapheme-to-Phoneme
(G2P) seq2seq model with `cmusphinx/g2p-seq2seq`**. That plan had several problems:

1. **Wrong target.** G2P only converts *text -> expected phonemes*. It never records
   audio, never listens to the user, and never scores anything. It is at most one
   small sub-component of a pronunciation scorer. The actual goal (show words ->
   speak -> score) was not addressed.
2. **Dead dependency.** `g2p-seq2seq` requires **TensorFlow >= 1.8** and
   **Tensor2Tensor 1.6.6** (per its README). TF 1.x is end-of-life and will not
   install on modern Python (3.10+). The repo has 43 open issues and no maintenance.
   It is unusable as written.
3. **Malformed command.** Step 1 listed `https://github.com/cmusphinx/g2p-seq2seq`
   as a shell command - it is a URL, not `git clone ...`.
4. **Unnecessary GPU/CUDA** for a "works on any PC" goal. Training a seq2seq from
   scratch is the opposite of simple.
5. **TF vs PyTorch inconsistency** - the body mentions TensorFlow, the deps list
   mentions PyTorch; the chosen tool is TF-only.
6. **No audio, no ASR, no scoring metric** - the entire core of the product was missing.
7. **Empty "CLI Commands Summary"** section - the prompt was unfinished.
8. **`audiokit`** dependency pointed to a personal repo with unclear maintenance.

**Decision:** Drop custom G2P training. Use a pretrained, pip-installable G2P
(`g2p_en`) plus CMUdict, build the missing audio -> recognition -> scoring ->
feedback pipeline with the author's `sherox` toolkit (sherpa-onnx) for **both ASR
and TTS** on CPU, and add **reference pronunciation audio** (let the learner hear
the target) via `sherox.tts`. Everything runs on CPU. **English only for v1**;
Indonesian/Arabic are a future TODO (see section 9).

---

## 1. Architecture overview

```
[Prompt]  ->  [G2P]  ->  [Record]  ->  [ASR]  ->  [Score + feedback]
show words    expected    audio       heard      0-100 + per-word detail
              phonemes    (mic)       text
                  ^                                  |
                  +---------- compare phonemes <------+
```

Pipeline (all CPU, all offline after a one-time model download):

1. **Prompt** - pick/show target words or a sentence.
2. **G2P** - convert target text to expected phonemes (CMUdict via `g2p_en`).
3. **Record** - capture the user reading the prompt.
4. **ASR** - transcribe what was actually said (offline `sherox.asr` / sherpa-onnx,
   default English model **NeMo CTC Conformer medium** (`nemo_ctc`, per-word
   confidence), CPU).
5. **Score** - align recognized words/phonemes to expected ones; compute 0-100.
6. **Feedback** - report which words/phonemes were wrong and how.

> **Scope note (important).** ASR-based scoring measures **intelligibility**
> ("did the recognizer understand the intended word?"), which is a strong, cheap
> proxy for pronunciation quality and is what most learners need. True
> *native-likeness/accent* scoring needs acoustic-model posterior probabilities
> (Goodness-of-Pronunciation, GOP) via forced alignment (e.g., Montreal Forced
> Aligner / Kaldi). That is provided as an **optional advanced track** in section 9,
> not the default.

---

## 2. Tech stack (simple, any PC)

- **Python 3.11-3.12** (CPU only; no CUDA required). 3.11+ is required by `sherox`.
- **G2P:** `g2p_en` (pretrained, NumPy-only inference, CMUdict + neural fallback
  for OOV words, handles homographs and numbers). `pip install g2p_en`.
  - Lighter alternative if `g2p_en` is too heavy or breaks on NumPy 2.x: `pronouncing`
    (pure CMUDict, zero ML, no OOV prediction).
- **ASR:** `sherox.asr` (sherpa-onnx on CPU). Default English offline model =
  **NeMo CTC Conformer medium** (`nemo_ctc`, `models/sherpa-onnx-nemo-ctc-en-conformer-medium`,
  ~158 MB, **auto-downloaded** on first run) - chosen because CTC emits **per-word
  confidence**, which the score blend (Step 5) uses. Offline mode also auto-downloads
  a Silero VAD. Word timestamps via `word_timestamps=True`.
  - Non-CTC fallback (no confidence): Parakeet TDT int8 (`nemo_transducer`,
    `models/parakeet-tdt-0.6b-v2-int8`, also auto-downloads).
  - Other English swaps via `config.py`: NeMo CTC small (lighter), Moonshine tiny
    (`moonshine`), Whisper small.en (`whisper`).
- **TTS / reference audio:** `sherox` (`sherox.tts`, sherpa-onnx on CPU). English
  voice `eng` = `vits-piper-en_US-amy-medium` (22.05 kHz), auto-downloaded on first
  use to `models/`. Used so the learner can **hear the target** before speaking, and
  to synthesize clean reference clips for the self-test (Step 10).
  - `sherox` is a **sibling repo** at `../sherox` (editable: `pip install -e ../sherox`);
    it depends on the sibling `../audiokit` (`pip install -e ../audiokit`). sherox's
    `pyproject.toml` already declares `audiokit = { path = "../audiokit", editable = true }`,
    so installing sherox editable resolves audiokit locally. If a sherox-side change is
    ever needed, edit `../sherox` directly or open an issue: `gh issue create -R bagustris/sherox`.
  - Fallback if the sibling repos are absent: install both from git:
    `pip install git+https://github.com/bagustris/sherox` and
    `pip install git+https://github.com/bagustris/audiokit`.
- **Audio (CLI):** `sounddevice` + `soundfile` (record N seconds to WAV).
- **Audio (web):** browser `MediaRecorder` -> upload WAV to backend.
- **Backend:** `fastapi` + `uvicorn`.
- **Frontend:** single `index.html` with vanilla JS (no build step).
- **Utils:** `rapidfuzz` (word/phoneme edit distance), `pydantic`. (`numpy`,
  `sherpa-onnx`, `sounddevice`, `soundfile` are pulled in by `sherox`.)

`requirements.txt`:
```
g2p_en>=2.1.0
-e ../sherox            # ASR + TTS, sibling repo (pulls sherpa-onnx, audiokit@../audiokit, numpy>=2.2)
soundfile>=0.12.1
fastapi>=0.110
uvicorn[standard]>=0.27
rapidfuzz>=3.6
pydantic>=2.5
```
> `../sherox` and `../audiokit` are sibling repos installed editable (see Step 1).
> All models (NeMo CTC English ASR, Silero VAD, Piper TTS) auto-download into
> `models/` on first run via sherox - **no manual model download step**.
> If `g2p_en` breaks on NumPy 2.x (sherox needs >=2.2), switch G2P to `pronouncing`.

---

## 3. Project structure

```
proscor-en/
  PLAN.md
  requirements.txt
  proscor/
    __init__.py
    config.py       # paths, model dir, sample rate, scoring weights
    g2p.py          # text -> expected phonemes (g2p_en + CMUdict + custom overrides)
    tts.py          # reference pronunciation audio via sherox (hear-the-target)
    audio.py        # record (CLI) / load wav (web); always mono 16kHz 16-bit
    asr.py          # sherox.asr (sherpa-onnx) transcribe -> text + per-word timing
    score.py        # align + phoneme edit distance -> 0-100 + per-word detail
    feedback.py     # human-readable feedback from the score report
    prompts.py      # word/sentence lists + prompt selection
  cli.py            # `python cli.py` interactive loop
  web/
    server.py       # FastAPI app
    static/index.html  # recorder UI (vanilla JS)
  scripts/
    selftest.py       # also warms up (auto-downloads) ASR + VAD + TTS models on first run
  data/
    prompts.txt     # default prompts (one per line; '#' comments)
    lexicon.txt     # OPTIONAL custom overrides: WORD<TAB>P AH L
  models/           # ASR + VAD + TTS auto-download here (gitignored)
  tests/
    test_g2p.py
    test_score.py
    test_feedback.py
```

---

## 4. Step-by-step implementation

### Step 1 - Environment & setup
- Requires the sibling repos checked out next to this one: `../sherox` and
  `../audiokit` (i.e. under the same parent dir, e.g. `~/github/{sherox,audiokit,proscor-en}`).
- Create venv: `python -m venv .venv && source .venv/bin/activate`.
- Install (editable, local): `pip install -e ../audiokit && pip install -e ../sherox`
  (install audiokit first to avoid any resolution hiccup; sherox's pyproject also
  resolves audiokit from `../audiokit`), then `pip install -r requirements.txt`.
  If the siblings are absent, fall back to installing both from git:
  `pip install git+https://github.com/bagustris/sherox` and
  `pip install git+https://github.com/bagustris/audiokit`.
- `python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger_eng')"`.
- **No manual model download.** sherox auto-downloads the ASR model (NeMo CTC
  English medium), Silero VAD (for offline ASR), and the TTS model (Piper
  `en_US-amy-medium`) into `models/` on first use. The first run is slower because
  of this; subsequent runs load from cache.
- Add `models/`, `data/uploads/`, `.venv/` to `.gitignore`.
- **Verify:** `python -c "from g2p_en import G2p; print(G2p()('hello'))"` prints
  phonemes; `python -c "from sherox.asr_engine import build_recognizer; from sherox.config import Config; print('ok')"`
  imports cleanly; `python -c "from sherox.tts import build_tts, synthesise_to_file, TtsConfig; print('ok')"`
  imports cleanly. (ASR + VAD + TTS models download on first synthesis/transcription.)

### Step 2 - G2P module (`proscor/g2p.py`)
- `expected_phonemes(text: str) -> list[list[str]]`: per-word ARPABET phonemes,
  **stress stripped** by default (normalize `AE1` -> `AE`) so scoring is
  stress-agnostic; keep an `include_stress=False` option.
- Cache the `g2p_en.G2p()` instance as a module-level singleton (loads once).
- Optional custom overrides: read `data/lexicon.txt` (format: `WORD<TAB>P AH L`);
  entries here win over the model. This keeps the good "fallback dictionary"
  idea from the old plan.
- **Tests (`tests/test_g2p.py`):** "hello" -> ['HH','AH','L','OW']; override-file
  entry wins; whitespace/space-vs-tab both accepted.

### Step 2b - Reference TTS (`proscor/tts.py`)
- `synthesize(text: str, lang="eng", speed=1.0, speaker_id=0, out_path=None)
   -> tuple[np.ndarray, int]` returning `(float32 samples, sample_rate)`.
- Thin wrapper over `sherox.tts`:
  ```python
  from pathlib import Path
  from sherox.tts import TtsConfig, build_tts, synthesise_to_file
  _ROOT = Path(__file__).resolve().parents[1]   # repo root -> models/ lives here
  def synthesize(text, lang="eng", speed=1.0, speaker_id=0, out_path=None):
      cfg = TtsConfig(model_dir="", language=lang, speaker_id=speaker_id,
                      speed=speed, output=out_path or "", play=False,
                      no_save=out_path is None, num_threads=1,
                      audio_prompt="", audio_prompt_text="")
      tts = build_tts(cfg, _ROOT)
      return synthesise_to_file(tts, text, cfg)   # (samples, sr)
  ```
  - English `lang="eng"` -> Piper VITS `en_US-amy-medium`, 22050 Hz, auto-downloaded
    to `models/vits-piper-en_US-amy-medium/` on first call. CPU only.
- `play_reference(text)`: synthesize then play via `sounddevice` (CLI).
- `reference_bytes(text) -> bytes`: synthesize, write a temp WAV, return bytes for
  the web `/api/reference` endpoint. (22.05 kHz is fine for playback; do NOT feed
  TTS output straight into ASR without resampling to 16 kHz.)
- Cache the built `tts` object (model load is the expensive part).
- If `sherox` import fails, raise a clear error pointing to the install/fallback
  steps in section 2 (or the vendored copy).
- **Verify:** `synthesize("hello world")` returns non-empty samples at 22050 Hz.

### Step 3 - Audio (`proscor/audio.py`)
- `record(seconds: float, sr=16000) -> np.ndarray` using `sounddevice` (mono,
  int16 PCM).
- `save_wav(path, samples, sr)` / `load_wav(path) -> (samples, sr)` via
  `soundfile`. Always normalize to **mono, 16 kHz, 16-bit** (sherpa-onnx's expected
  format; sherox's `read_wav` resamples to 16k automatically).
- For web: the endpoint receives a WAV blob and reuses `load_wav`.
- **Verify:** record 2 s, save, reload, assert shape and sample rate.

### Step 4 - ASR (`proscor/asr.py`)
- `transcribe(samples: np.ndarray, sr: int = 16000) -> dict` returning:
  `{"text": str, "words": [{"word": str, "conf": float, "start": float, "end": float}]}`.
  (Accepts float32 mono samples; `conf` is 1.0 if the model doesn't emit confidence.)
- Build the recognizer via sherox (which handles model auto-download + config
  routing), then decode one utterance with the sherpa-onnx offline API:
  ```python
  from pathlib import Path
  from sherox.asr_engine import build_recognizer
  from sherox.config import Config
  import numpy as np
  _ROOT = Path(__file__).resolve().parents[1]   # models/ lives here

  def _cfg():
      return Config(model_dir="models/sherpa-onnx-nemo-ctc-en-conformer-medium",
                    model_type="nemo_ctc", offline=True,
                    sample_rate=16000, num_threads=1,
                    word_timestamps=True, language="en")

  _REC = None
  def _recognizer():
      global _REC
      if _REC is None:
          _REC = build_recognizer(_cfg())   # auto-downloads NeMo CTC EN on first call
      return _REC

  def transcribe(samples: np.ndarray, sr: int = 16000) -> dict:
      samples = np.ascontiguousarray(samples, dtype=np.float32)
      rec = _recognizer()
      stream = rec.create_stream()
      stream.accept_waveform(sr, samples)
      rec.decode_stream(stream)
      res = stream.result                       # OfflineRecognitionResult
      words = []
      for w in getattr(res, "words", []) or []:
          words.append({"word": w.word, "conf": getattr(w, "confidence",
                          getattr(w, "prob", 1.0)),
                        "start": float(w.start), "end": float(w.end)})
      return {"text": res.text, "words": words}
  ```
  - `word_timestamps=True` populates `res.words` (`.word/.start/.end`). The default
    **NeMo CTC** model also sets per-word `confidence`, which the score blend (Step 5)
    uses. If you switch to a non-CTC model (e.g. Parakeet `nemo_transducer`),
    confidence may be absent and is treated as 1.0 above.
- Cache the recognizer (model load is the expensive part).
- Keep model dir/type configurable via `config.py` (`ASR_MODEL_DIR`, `ASR_MODEL_TYPE`).
- For **offline VAD-segmented** accuracy, sherox wraps a Silero VAD (auto-downloaded).
  For our single-utterance case, feeding the whole clip via `accept_waveform` is
  simplest and accurate enough; if very long prompts are added later, route through
  sherox's `run_offline_vad_streaming` instead.
- **Verify:** record yourself saying "hello world"; confirm text + word timings.

### Step 5 - Scoring (`proscor/score.py`)  <-- the core
> This is an **ASR-based intelligibility score** (phoneme edit distance), **not**
> Goodness-of-Pronunciation. GOP needs acoustic-model posteriors from forced
> alignment; see the optional advanced track in section 5.
Inputs: `target_text` and the ASR result. Algorithm:
1. Tokenize `target_text` -> target words; call `expected_phonemes` per word.
2. Get recognized words from ASR (lowercased, alpha-only). If ASR gave nothing,
   return score 0 with reason `"nothing recognized"`.
3. Align recognized words to target words with a **monotone word-level alignment**
   (Levenshtein/DTW over words via `rapidfuzz`). For each target word find its
   best-matching recognized word (or mark as a deletion).
4. For each target word:
   - **Word correct** if its aligned recognized word matches (case-insensitive).
   - **Phoneme edit distance** between target phones and the recognized word's
     phones (run G2P on the recognized word too).
   - `word_score = max(0, 1 - edits / max(len(target_ph), len(heard_ph))) * 100`.
   - Capture substitution detail, e.g. expected `TH` heard `D`.
5. **Overall score** = mean of per-target word scores (0-100), rounded.
   - **Confidence blend (active by default with the CTC model):**
     `0.8*phoneme_score + 0.2*conf*100` (weights in `config.py`).
   - If a non-CTC ASR model is selected (no confidence), fall back to pure phoneme
     edit distance (the primary, model-agnostic signal).
6. Return a structured `ScoreReport`:
   ```json
   {
     "score": 78.0,
     "words": [
       {"target": "thought", "recognized": "thought", "correct": true,
        "word_score": 100.0,
        "phonemes_expected": ["TH","AO","T"], "phonemes_heard": ["TH","AO","T"],
        "edits": []},
       {"target": "through", "recognized": "true", "correct": false,
        "word_score": 33.0,
        "phonemes_expected": ["TH","R","UW"], "phonemes_heard": ["T","R","UW"],
        "edits": [{"op": "sub", "at": 0, "expected": "TH", "heard": "T"}]}
     ],
     "notes": "2 of 5 words mispronounced"
   }
   ```
- **Tests (`tests/test_score.py`):** feed a *fake* ASR result (no mic needed):
  all correct -> 100; one substitution -> <100 and an edit entry; one deletion;
  nothing recognized -> 0.

### Step 6 - Feedback (`proscor/feedback.py`)
- `format_report(report) -> str`: human-readable, CLI-friendly. Example:
  ```
  Score: 78/100
    OK   thought    [TH AO T]
    MISS through  -> heard "true"  [expect TH R UW | heard T R UW]  (TH -> T)
  ```
- Also expose the structured dict for the web API (already structured in step 5).
- Ship a small phoneme -> example-word hint table so feedback can say
  `` `TH` as in **th**ink ``.

### Step 7 - Prompts (`proscor/prompts.py`)
- Load `data/prompts.txt` (one word/sentence per line; `#` comments).
- `get_prompt(index=None, random=True) -> {"text": str, "id": int, "category": str}`.
- Ship a sensible default list: minimal pairs (ship/sheep, think/tink, bit/beat,
  thought/though) plus a few short sentences.

### Step 8 - CLI (`cli.py`)
Interactive loop:
```
$ python cli.py
proscor> Prompt #3: "She sells sea shells."
proscor> (p)lay reference, then press ENTER to record (3 s)...
[recording...]
proscor> Score: 84/100
    OK   she
    OK   sells
    OK   sea   -> heard "see"   [phonemes match]
    OK   shells
proscor> (n)ext  (r)etry  (q)uit
```
- Flags: `--seconds`, `--prompt-file`, `--include-stress`, `--model-dir`,
  `--tts-lang eng` (reference voice), `--no-tts` (disable reference playback).
- In-loop keys: `p` play reference, `r` retry, `n` next, `q` quit.
- No network needed after the one-time model download.

### Step 9 - Web app (`web/server.py` + `web/static/index.html`)
- FastAPI endpoints:
  - `GET /api/prompt` -> `{"text": "...", "id": 3}`.
  - `GET /api/reference?text=...` -> WAV bytes (sherox-synthesized target audio)
    so the browser can play the correct pronunciation.
  - `POST /api/score` (multipart: `audio` WAV + form field `target_text`)
    -> the `ScoreReport`.
  - `GET /` -> serves `index.html`.
- `index.html` (vanilla JS, no framework):
  - Show prompt; a "Play reference" button (fetches `/api/reference` and plays it)
    and a Record button using `MediaRecorder`, then encode to WAV (16 kHz mono)
    before POSTing.
  - Render the overall score, a per-word table with color coding, and substitution
    hints from the report.
- Run: `uvicorn web.server:app --reload --port 8000`.
- **CORS:** same origin (static served by the same app) -> no CORS config needed.

### Step 10 - Evaluation & QA
- Unit tests run offline: `pytest -q`.
- A `scripts/selftest.py` that feeds WAVs in `tests/audio/` through the pipeline and
  asserts score ranges: clean reference >= 90, intentionally-bad <= 40.
  - Generate the **clean reference** clip with `proscor.tts.synthesize(prompt)`
    (sherox) and resample to 16 kHz for ASR.
  - Generate a **bad** clip by degrading the reference (add noise / shift pitch /
    drop a phoneme via simple DSP) so no mic recording is needed for the test.
- Manual smoke test for both CLI and web.
- Track ASR word accuracy on the prompt set and phoneme-error behavior over time.

---

## 5. Optional advanced track - true pronunciation scoring (GOP)

The default score (Step 5) is **intelligibility** (did ASR understand the word?),
not **native-likeness**. For the latter, add **Goodness-of-Pronunciation**:

- **GOP-lite (still any-PC, CPU):** now that `sherox` (sherpa-onnx) is in the stack,
  load a sherpa-onnx **CTC** English ASR model, force-align the audio to the
  expected phoneme sequence, read per-frame phoneme posteriors, and compute
  `GOP(p) = log P(p | audio) / duration(p)` per phoneme -> 0-100. No Kaldi/MFA
  install needed; reuse the ONNX runtime already present. This is the recommended
  upgrade path.
- **GOP full (heavier):** Montreal Forced Aligner (MFA) or `gentle` (Kaldi) for
  HMM-DNN posteriors + alignment. More accurate accent scoring, but a heavier
  install that fights the "any PC" goal.

Wire both behind `--engine gop` (or `--engine gop-lite`) and keep
`--engine intelligibility` (Step 5) as the default. Blend optionally:
`score = 0.6*intelligibility + 0.4*gop` (weights in `config.py`).

### 5a. Status: word-level GOP-lite implemented and validated; phone-level pending

**Correction to the paragraph above:** the default ASR model
(`sherpa-onnx-nemo-ctc-en-conformer-medium`) is a **BPE-subword CTC model**
(`tokens.txt` is `▁the`, `ing`, `ed`, single letters, ...), not a phoneme
model. There is no per-*phone* posterior to read from it. This was discovered
when implementing the "read per-frame phoneme posteriors" step above, and
changed the plan to a staged rollout:

1. **Word-level GOP-lite (implemented, validated):** `proscor/align.py` gained
   sentence-level CTC **Viterbi** forced alignment (`_ctc_viterbi`, max-path
   with backpointers, alongside the existing forward-only `_ctc_loglik` used
   for single-word scoring) plus `align_words_gop(samples, words)`, which
   force-aligns a full utterance's canonical word sequence (greedy BPE
   segmentation per word — sentencepiece isn't shipped with this model dir,
   see `_greedy_segmentation`) and returns a per-word GOP score:
   `mean over the word's aligned frames of (log P(aligned token) - max_k log P(token k))`
   — how much posterior mass the model placed elsewhere at each frame the
   word occupies.
2. **Validation harness (implemented):** `scripts/eval_so762.py` downloads
   [speechocean762](https://huggingface.co/datasets/mispeech/speechocean762)
   (HF parquet mirror, cached under `~/.cache/huggingface`, not stored in this
   repo), force-aligns each utterance's *dataset-canonical* word text (never
   proscor's G2P, so G2P errors can't contaminate the eval) against the
   dataset audio, and correlates the GOP score against the dataset's human
   per-word `accuracy` label (0-10). Needs `pip install -r requirements-eval.txt`
   (huggingface_hub, pyarrow, scipy — kept out of the core `requirements.txt`
   since the CLI/web app never needs them).
3. **Result (full `test` split, 2500 utterances, 15,967 words):**

   | precision | time (CPU) | word PCC vs. `accuracy` | word ρ | utt PCC vs. `accuracy` | utt PCC vs. `total` |
   |---|---|---|---|---|---|
   | int8 (`model.int8.onnx`, app default) | 243s | 0.471 | 0.392 | 0.556 | 0.589 |
   | fp32 (`model.onnx`) | 114s | 0.467 | 0.395 | 0.544 | 0.579 |

   Quantization makes essentially no difference to correlation quality here
   (and fp32 ran *faster* in this onnxruntime build — not the expected
   direction, not chased further). `ALIGN_USE_INT8` (`config.py`) controls
   which weights `proscor/align.py` loads; `scripts/eval_so762.py --fp32`
   overrides it for eval runs. Mean GOP drops monotonically with human
   accuracy (int8: -0.68 at word-accuracy 10 down to -11.1 at word-accuracy
   0). Align rate 100% (every canonical word in the test split segmented
   under the model's BPE vocab). Raw per-word records + logs for both runs
   are under `results/` (gitignored - regenerate via
   `python scripts/eval_so762.py [--fp32]`, ~2-4 min).

   **Comparison to the literature** (Table 1 of [GOPT (Gong et al., ICASSP
   2022, arXiv:2205.03432)](https://arxiv.org/abs/2205.03432), the standard
   speechocean762 reference, mean over 5 seeds): the original
   speechocean762 paper's RF/SVR baselines only report **phone-level** PCC
   (0.440 / 0.450 — no word/utterance numbers exist to compare against at
   those granularities). GOPT's own LibriSpeech-acoustic-model variant (the
   closest published setup to ours — a public, non-domain-specific ASR
   model) reports phone PCC 0.612, word-accuracy PCC 0.533, utterance-
   accuracy PCC 0.714, utterance-total PCC 0.742. GOPT is a **trained**
   multi-task Transformer on top of Kaldi TDNN-F GOP features; GOP-lite here
   is **zero-shot** — no training data, no pronunciation-scoring-specific
   model, just posterior deficit from a general BPE ASR model. Landing at
   88% of GOPT's word-level PCC (0.471/0.533) and 78-79% of its utterance-
   level PCC (0.556/0.714 accuracy, 0.589/0.742 total) with zero training is
   the systems-paper result: a CPU-only, off-the-shelf-ASR GOP proxy gets
   most of the way to a trained baseline for free.

   **Limitation found during validation - GOP zero-inflation:** the fraction
   of words with `gop == 0.0` (the Viterbi-aligned label *is* the frame's
   argmax, i.e. the CTC path can't fit any better) rises monotonically with
   human accuracy: 0% at word-accuracy 0-2, 39% at accuracy 8, **74% at
   accuracy 10**. The metric saturates on "obviously correct" speech instead
   of distinguishing good from excellent — expected for whole-word BPE units
   (few, coarse-grained decision points per word) and the likely reason
   Spearman ρ trails Pearson r at word level (rank ties on both axes). A
   finer unit (phones - see next item) should shrink this.
4. **Phone-level GOP (implemented, validated) — a genuine phone model, plus
   a caveat about aggregation:** `proscor/align_phone.py` uses
   `onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX` (an ONNX export of
   `facebook/wav2vec2-lv-60-espeak-cv-ft`, ~320MB int8 / ~1.2GB fp32,
   downloaded from the Hub on first use — heavier than the BPE model, an
   evaluation-only path, **not wired into the CLI/web app**). It takes raw
   waveform (no fbank step) and outputs a 392-symbol espeak-ng IPA vocab —
   genuinely phone-level, unlike the BPE model. Targets are generated by
   phonemizing each canonical word live with the `phonemizer` package's
   espeak-ng backend (same tool that produced the model's training labels),
   not a hand-built ARPABET→IPA table: a static-table attempt was checked
   against the model's actual greedy decode output first and found wrong —
   espeak merges vowel+R into single rhotic tokens (e.g. `ɑːɹ` for the vowel
   in "mark") and the table missed that, so live phonemization was used
   instead. Reuses `proscor.align._ctc_viterbi` (generic over any token
   sequence, not BPE-specific) for the forced alignment itself.

   **Result (full `test` split, 2500 utterances, 15,967 words, 47,369
   phones, int8, ~7 min CPU — `scripts/eval_so762_phone.py`,
   `results/so762_test_phone_int8_summary.json`; confirmed on the untouched
   `train` split, see "Train-split confirmation" below):**

   | metric | this (zero-shot, test split) | literature (trained) |
   |---|---|---|
   | **phone-level PCC, reconciled** (n=47,239/47,369 phones, 99.73% coverage) | **0.433** | GOPT-LibriSpeech 0.612; classic RF/SVR baselines (phone-only, no word/utt numbers exist for them; computed on all 47,369 phones) 0.440/0.450 |
   | phone-level PCC, matched-length-only (legacy metric, n=43,888/47,369, 92.65% coverage) | 0.425 | — |
   | word-accuracy PCC | 0.325 | GOPT-LibriSpeech 0.533 |
   | utterance-accuracy PCC | 0.536 | GOPT-LibriSpeech 0.714 |
   | utterance-total PCC | 0.572 | GOPT-LibriSpeech 0.742 |

   (GOPT numbers are Table 1 of [Gong et al., ICASSP 2022, arXiv:2205.03432](https://arxiv.org/abs/2205.03432),
   mean over 5 seeds, LibriSpeech-acoustic-model variant — the closest
   published setup to a general, non-domain-specific ASR model. RF/SVR are
   the original speechocean762 paper's baselines, phone-level only.)

   The reconciled phone-level number (0.433, on 99.73% of phones — see
   "Phone-count reconciliation" below for how the remaining 0.27% is a
   principled exclusion, not a gap) is the literature-comparable headline
   metric this section originally wanted, and it now lands at **98.4% of
   the classic *trained* RF baseline and 96.2% of SVR** (70.8% of GOPT,
   which is a substantially heavier trained model) with **zero training**
   — a legitimate systems-paper result. The unexpected finding: **this
   phone model's word/utterance-level PCC is lower than the simpler BPE
   model's** (item 3 above: word 0.471, utt-accuracy 0.556, utt-total
   0.589) despite finer alignment granularity and a purpose-built phoneme
   vocabulary. Likely causes, not disentangled here: (a) domain mismatch —
   this wav2vec2-large model is trained on adult multilingual CommonVoice
   speech via espeak's *automatically generated* (not hand-verified)
   phoneme labels, while speechocean762 is child L2 English speech, a
   harder acoustic domain than adult native/fluent L2 speech; (b)
   word-level aggregation averages over more, noisier phone-level units
   per word than the BPE model averages over subword units, so per-phone
   noise has more opportunities to accumulate. A first version of
   `align_words_gop`'s word score (mean over the word's whole frame *span*,
   including blank frames between phones — mirroring the BPE model's
   formula) measured far worse (word PCC 0.03 on a 100-utterance sample)
   than the current version (mean of the word's *phone*-only GOP values,
   ignoring inter-phone blank frames, PCC 0.22 on the same sample) — blank
   frames dilute the signal more here because a word has many more
   phone/blank transitions than BPE/blank transitions. **A competitor-set
   contamination hypothesis was checked and ruled out:** the 392-symbol
   vocab has near-duplicate symbols across languages for the same English
   sound (e.g. `uː`/`u`, `iː`/`i`), which could make `max_k lp_t(k)` an
   unfair competitor if the model splits mass across spelling variants of
   the same sound. A manual check of the worst (`gop < -1`) phones on
   human-accuracy-10 words (300 utterances) found the dominant confusions
   were phonetically genuine (vowel reduction to schwa, place/manner shifts
   like `t`→`k`, `ð`→`d`, weak/blank realization of light consonants), not
   cross-lingual spelling duplicates — so this is left as real GOP signal,
   not a vocab artifact to fix by restricting the competitor set.

   Bottom line for `--engine gop-lite`: the BPE model (item 3) remains the
   shipped default — it is both cheaper and empirically better at
   word/utterance granularity, which is what the CLI/web app actually
   surfaces to a learner. The phone model's role is the validated
   literature-comparable phone-level number above, not a better runtime
   engine as-is; wiring it in would need the aggregation question above
   resolved first (e.g. a learned or tuned per-phone weighting instead of a
   flat mean).

   **Bug found and fixed: espeak-ng phonemization is case-sensitive in a way
   unrelated to pronunciation.** speechocean762's transcripts are all-caps;
   feeding `"IT"`/`"US"` to `phonemizer` got them spelled out letter-by-letter
   ("I-T", "U-S" — espeak's acronym heuristic), not pronounced, corrupting
   the target phones for those words. Checked systematically: 2 of 1,869
   unique test-split words differed between upper- and lower-case
   phonemization, covering 2.14% of word occurrences. `_phonemize_word` now
   lowercases before calling espeak (with a regression test in
   `tests/test_align_phone.py`, skipped when espeak-ng isn't installed). Full
   re-run after the fix: every number in the table above rose slightly
   (word-level PCC 0.302→0.325, phone coverage 91.21%→92.65% matched-only)
   — the fix helped, as expected, and the effect size matches the bug's
   small blast radius. The all-caps text is also fed unmodified to
   `align.py`'s BPE path (`eval_so762.py`) and `g2p_en`/CMUdict
   (case-insensitive lookups), neither of which showed any analogous
   sensitivity — this was specific to `phonemizer`'s espeak-ng backend.

   **Phone-count reconciliation (implemented and validated):**
   `proscor.align_phone.reconcile_phones` aligns every word's espeak phones
   against its ARPABET phones with a constrained Needleman-Wunsch DP (1:1
   matches via an ARPABET/IPA equivalence table — the same one an earlier
   abandoned static-mapping attempt produced, repurposed here as a
   *matching* heuristic instead of a *target-generation* one — plus 2:1/3:1
   merges for three known patterns, plus deletions for phones with no
   espeak counterpart at all), instead of dropping the whole word on a
   count mismatch as the original "matched-length-only" metric did. The
   patterns, from a systematic breakdown of the (post-case-fix) 5.41%
   word/7.35% phone mismatch rate — two different denominators, since
   mismatched words run longer on average; do not conflate them:
   (a) **rhotic-vowel merge** (~60% of mismatches) — espeak fuses vowel+R
   into one token ("mark" → `m ɑːɹ k`, "four" → `f oːɹ`); (b)
   **syllabic-L merge** — espeak has a dedicated `əl` vocab token for a
   syllabic L that ARPABET spells as two phones, `AH0 L` ("difficult" →
   `...k əl t`); (c) **diphthong-cluster merge** — espeak treats some vowel
   sequences as one complex-nucleus token (`iə`, `aɪə`, `aɪɚ`) where
   ARPABET keeps two vowel symbols ("idea" → mine `d iə`, dataset
   `D IH AH1`). A fourth, *not* an alignment problem: **genuine dialectal
   disagreement** — espeak's en-us applies yod-dropping ("new" → `n uː`,
   no `j`) where CMUdict's canonical entry keeps the historical glide
   (`N Y UW0`); no realignment recovers a phone the model was never asked
   to produce, so the DP correctly returns `None` (a deletion) for it
   rather than forcing a bad merge (`tests/test_align_phone.py` pins this
   exact case, plus a known-imperfect case, "player", where the 3-merge
   heuristic declines and the DP falls back to a less precise but still
   non-empty substitution/deletion mix — never worse than the pre-
   reconciliation status quo, since dropping the whole word is always the
   DP's worst-case fallback, not an improvement over it).

   Full test-split result: coverage rose from 92.65% (matched-only) to
   **99.73%** (`op_counts`: 43,429 match, 2,639 substitution, 1,150
   `merge2`, 21 `merge3`, 130 deletion — merge-op count is the same order
   of magnitude as the ~864 mismatched words, not the thousands that would
   signal the merge rule over-firing on already-matched words). Reconciled
   phone-level PCC (0.433) is *higher* than the matched-only PCC (0.425),
   not just wider coverage at the same quality — and the **newly-covered
   phones alone** (the 3,353 that only exist because of reconciliation)
   correlate *better* than the pre-existing matched set (PCC 0.501 vs.
   0.425), addressing the zero-inflation concern raised when this was
   scoped: duplicating one espeak phone's GOP across 2-3 ARPABET slots does
   inherit that phone's zero-inflation (many well-pronounced phones score
   exactly 0.0), but words needing reconciliation are modestly less likely
   to be perfectly pronounced than words that didn't (83.1% word-accuracy-10
   among mismatched words vs. 90.1% among matched, measured directly rather
   than assumed), so the newly-added scores skew slightly toward the more
   informative, less-saturated part of the accuracy range rather than
   diluting it.

   **Precision:** fp32 was not run at full scale for this model (it's 4x
   larger than the BPE model; a full run was judged not worth the added
   compute). A 100-utterance pre-aggregation-fix check showed int8 vs. fp32
   within ~0.03 PCC on the matched-phone comparison, consistent with the BPE
   model's full-scale finding (item 3) that precision barely matters here.

   **Train-split confirmation:** the switch from span-based to phone-mean
   word-GOP aggregation (PCC 0.03 -> 0.22) and the competitor-set
   contamination check were the two decisions actually shaped by looking at
   `test`-split behavior — standard practice for debugging, but it meant the
   `test`-split numbers weren't purely held-out for the aggregation
   *choice* (the diagnostic itself changed nothing, just confirmed a null
   result). `train` was never inspected for either of those. The
   case-sensitivity bug fix above was also *diagnosed* on `test` (that's
   where the mismatch investigation happened), but it is a correctness fix
   applied uniformly to the phonemization code, not a parameter tuned to
   the data — it changes "IT" to phonemize correctly regardless of which
   split contains it, so it doesn't reopen the held-out question the same
   way a data-driven choice would. Running `scripts/eval_so762_phone.py
   --split train` (2,500 further utterances, 15,849 words, 47,076 phones)
   confirms the two test-inspected decisions above generalize — if
   anything, the numbers are slightly higher: word-level PCC 0.383,
   utterance-accuracy PCC 0.590, utterance-total PCC 0.594, phone-level
   (matched-only) PCC 0.476 at 43,648/47,076 = 92.72% coverage (43,648 is
   the `n` the run reported for that comparison,
   `results/so762_train_phone_int8_summary.json`; 47,076 is the train
   split's total phone count, not itself in that JSON — this predates the
   script emitting an explicit `n_phones_total`/coverage field, so the
   denominator is cited from the dataset description rather than read
   directly out of the file). Matched-only, not reconciled — the
   reconciliation code postdates this run and wasn't re-run on `train`,
   since the held-out question it would answer, "was the aggregation choice
   overfit to test," was already settled above.
   No sign of overfitting to test-set inspection.
5. **Wired into the app:** `proscor.score.score_gop_lite` force-aligns a
   multi-word target (`align.align_words_gop`) and maps each word's GOP to
   0-100 via the shared `align.gop_to_fit` transform (factored out of
   `_map_score`, same `exp(gop / ALIGN_GOP_SCALE)` formula). `score_audio`
   takes `engine="intelligibility"` (default) or `"gop-lite"`, falling back
   to the intelligibility path if the alignment extras aren't installed.
   Exposed as `cli.py --engine {intelligibility,gop-lite}`, an `engine` form
   field on `POST /api/score` (default `"intelligibility"`, so existing
   callers are unaffected), and a scoring-engine `<select>` in
   `web/static/index.html`. `GOP_LITE_CORRECT_THRESHOLD = 60` (`config.py`)
   decides the "correct" pass/fail line; from the full test-split GOP-lite
   scores, 81.2% of human-accuracy-10 words land >= 60 and 73.8% of
   accuracy-<=5 words land < 60 — a real but noisy classifier (word-level
   Pearson r = 0.47), so both `feedback.format_report` (CLI) and the web
   table also show the continuous `word_score` rather than only pass/fail.

   Because GOP-lite force-aligns *to* the target instead of free-decoding,
   `recognized` is always the target word even when the fit is poor — a
   low-scoring word is always a poor *fit*, never a different word that was
   "heard" instead, and both renderers special-case `recognized == target &&
   !correct` so they don't misreport it as one: the CLI prints `LOW word
   [phones] fit NN/100` instead of a `MISS ... -> heard "X"` line, and the
   web table's "Heard" column prints `(fit NN/100)` instead of the (always
   identical to target) recognized word. Verified end-to-end over real HTTP
   requests to `POST /api/score` for both engines, not just at the
   Python-call level.

### 5b. Second-corpus generalization: UME-ERJ

Everything in 5a was validated on one corpus (speechocean762: Mandarin-L1
children). [UME-ERJ](https://research.nii.ac.jp/src/en/UME-ERJ.html) (NII
Speech Resources Consortium) is a second corpus with a different L1
(Japanese) and, plausibly, a different age range — it's recorded at Japanese
universities (Tohoku, Kyoto, Toyohashi Tech, Tokyo, Tokyo Tech, Iwate,
Waseda, Ritsumeikan, Ryukoku — `workgroup.txt`) and literally named "English
Speech Database Read by Japanese **Students**." Unlike speechocean762
(which has an explicit per-speaker `age` field), no document found in this
corpus states speaker ages numerically, so "adult university students" here
is an inference from the recording-site/corpus-name evidence, not a
verified field — worth being precise about, since the inference does load-
bearing work in the discussion below.

**Access & format:** ships as a local directory tree (Shift-JIS-encoded
metadata, CRLF line endings, `S#_###.wav`/`W#_###.wav` recording filenames
under `wav/JE/<SITE>/<GEN><SPK>/`), not a downloadable HF/parquet mirror
like speechocean762 — `scripts/eval_umeerj.py --data-root <path>` (default
`/data/UME-ERJ`). **Crucially, it has no per-phone accuracy labels**:
ratings are holistic 1-5 scores from up to 5 native-English-teacher raters
(averaged here per item; individual rater identity isn't needed for a
correlation against the mean), per **sentence** (segmental/rhythm/
intonation) or per **word** (segmental/accent). So this validates
word/utterance-level GOP-lite only — it cannot extend the phone-level
reconciled-PCC number from item 4, which needs per-phone ground truth that
doesn't exist here.

**Result** (both engines at their int8 defaults, `scripts/eval_umeerj.py`,
`results/umeerj_summary.json`; zero failures across all 9,484 rated items —
0 missing text, 0 missing audio, 0 scoring exceptions for either engine):

| category | unit | n (rated recordings) | BPE-model PCC | phone-model PCC |
|---|---|---|---|---|
| segmental | sentence | 1,900 | 0.328 | **0.432** |
| segmental | word | 3,784 | 0.294 | **0.428** |
| rhythm | sentence | 950 | 0.055 | 0.156 |
| intonation | sentence | 950 | 0.056 | 0.039 |
| accent (stress) | word | 1,900 | 0.110 | 0.106 |

**Generalization holds:** the segmental correlations (0.29-0.43) land in
the same band as speechocean762's word/utterance numbers (0.33-0.59, item
3-4) on a corpus with a different L1 and recording protocol — that's what
"generalizes" should look like, not a coincidence of one dataset's
particulars.

**The ranking flips, and that's the most interesting result in this
section:** on speechocean762, the BPE model beat the phone model at
word-level (item 3 vs. item 4: word PCC 0.471 vs. 0.325) and
utterance-level (0.556/0.589 vs. 0.536/0.572). On UME-ERJ, the **phone
model wins clearly** at both word- and sentence-segmental (0.43 vs.
0.29-0.33). Two candidate explanations, **not separable with only two
corpora**:
1. **Domain/acoustic match.** The phoneme-CTC model is trained on adult
   multilingual CommonVoice speech. If UME-ERJ's speakers are adult
   university students (the inferred-not-verified claim above) and
   speechocean762's are documented children (ages 5-15), the phone model's
   training distribution is simply closer to UME-ERJ's speakers acoustically.
2. **L1-specific error profile.** Japanese-accented English has a
   well-documented, specific phoneme-substitution profile (/l/~/r/,
   /θ/~/s/, vowel epenthesis, /v/~/b/) that a genuine phoneme-level model
   may be structurally better positioned to catch than a BPE model routing
   everything through orthography; Mandarin-accented child speech has a
   different error profile the BPE model might happen to fit better.

Both are plausible, both are consistent with the data, and this pair of
corpora changes L1 *and* (probably) age/register simultaneously, so neither
can be isolated here. A third corpus that holds one variable fixed while
changing the other (e.g. a documented-adult, non-Japanese-L1 corpus, or a
documented-child Japanese-L1 corpus) would be needed to separate them —
noted as a natural follow-up, not done.

**Negative controls behave as expected:** rhythm and intonation correlate
weakly (0.04-0.16) for both engines — GOP is a posterior-deficit measure of
phone/word *identity* fit, not pitch contour or timing, so it having
little to say about prosody is the metric measuring what it claims to
measure, not a failure. Word accent/stress (0.11 both) is similarly weak —
placement of stress isn't something either engine's GOP formula
represents. (The phone model's small edge on rhythm, 0.156 vs. 0.055, is
plausibly its per-phone segmentation carrying a little durational
information the BPE model's word-span-only granularity doesn't — mentioned
once, not built on; the two intonation numbers and the two accent numbers
are statistically indistinguishable from each other and from zero given
these sample sizes.)

**Bug found and fixed during this run:** `proscor.align._ctc_viterbi`'s
backtracking loop did `s -= back[t, s]`, mixing a plain Python state-index
int with a NumPy `int8` array value; under NumPy 2's stricter
type-promotion rules this raised `OverflowError` once a state index
exceeded 127 (utterances with more than ~64 tokens). speechocean762's
shorter utterances never hit this (zero warnings across every full-scale
run there); UME-ERJ's longer TIMIT-based sentences did (8/1,900 = 0.4% of
the first, since-superseded sentence-segmental run). Fixed with an explicit
`int(...)` cast and a regression test (`tests/test_align.py`); the results
above are from the re-run after the fix, with zero failures. This bug was
latent in the shipped `--engine gop-lite` path too (`proscor.align`, not
just the phone-model eval), not just the eval scripts.

### 5c. Disentangling age/domain-match from L1-specific error profile: L2-ARCTIC

Section 5b found the BPE-vs-phone-model ranking flips between speechocean762
(Mandarin-L1 children, BPE wins) and UME-ERJ (Japanese-L1 adults, phone
wins), with two candidate explanations that two corpora can't separate:
acoustic/age domain match, or an L1-specific phoneme-substitution profile a
phone model is structurally better positioned to catch. **The test:**
[L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) has **adult
Mandarin-L1** speakers — same L1 as speechocean762, same age category as
UME-ERJ — so if adult-Mandarin behaves like child-Mandarin (BPE wins), L1
dominates; if it behaves like adult-Japanese (phone wins), age/domain
dominates.

**Data:** the [KoelLabs/L2Arctic](https://huggingface.co/datasets/KoelLabs/L2Arctic)
HF mirror (gated, requires an approved token) rather than the original
TAMU/Kaggle distribution — a clean parquet of the "scripted" split (3,599
utterances, all 24 speakers, all 6 L1s, audio pre-converted to 16kHz
float32). Trade-off: this mirror gives two **unaligned** IPA strings per
utterance (`g2p` = canonical, `ipa` = expert-verified perceived), not the
original per-phone-aligned canonical/perceived/error-tag triples the raw
TextGrid annotations have (which `proscor.align_phone.reconcile_phones`-style
machinery could exploit for a real phone-level number — see "Not done"
below). `scripts/eval_l2arctic.py` uses a coarser proxy instead: normalized
character-level Levenshtein distance between `g2p` and `ipa` as an
utterance-level pronunciation-error signal (expected to correlate
*negatively* with GOP — higher edit distance should mean worse
pronunciation).

**Ground-truth signal differs across all three corpora — magnitudes below
are not comparable to 5a/5b without accounting for this:**

| corpus | unit | signal | BPE PCC | phone PCC |
|---|---|---|---|---|
| speechocean762 (5a) | word | expert 0-10 score | 0.471 | 0.325 (0.433 reconciled, phone-level) |
| UME-ERJ (5b) | sentence/word | expert 1-5 mean (5 raters) | 0.29-0.33 | **0.43** |
| L2-ARCTIC, Chinese (5c) | sentence | char-level CER proxy | -0.107 | -0.119 |
| L2-ARCTIC, Chinese, phone-level (5c below) | phone | expert binary correct/error | n/a (BPE model has no phone output) | **0.206** |

**Result: a near-tie, not a replication of either prior pattern.**
speechocean762 (n=15,967 words): |BPE| clearly beats |phone| (0.471 vs.
0.325). UME-ERJ (n=1,900-3,784): |phone| clearly beats |BPE| (0.43 vs.
0.29-0.33, a ~1.3-1.5x margin). L2-ARCTIC Chinese (n=600): |phone| =
0.119 vs. |BPE| = 0.107 — a difference within noise at this n, not a clear
win either way. **Per-speaker breakdown** (4 speakers, 150 utterances each,
`results/l2arctic_chinese_summary.json`) confirms this isn't an artifact of
averaging out a real effect: BWC (BPE -0.062, phone -0.186, phone clearly
stronger), TXHC (BPE -0.057, phone -0.106, phone stronger), NCC (BPE
-0.160, phone -0.100, BPE stronger), LXC (BPE -0.090, phone +0.014, phone
*wrong-signed*, effectively zero). Four speakers, three different
winners-or-ties.

**Read the "before caveats" framing first, not last: the correlations here
(|r| ≈ 0.02-0.29 across all six L1s, `results/l2arctic_summary.json`) are
2-3x weaker across the board than either speechocean762 or UME-ERJ.**
Before attributing the Chinese tie to a real phenomenon, the char-level CER
proxy has to be trusted to resolve a difference this size, and the
all-language table below suggests it can't always be trusted to do even
that: Korean shows BPE at **+0.131** (wrong sign, n=600) — either a genuine
anomaly in that subpopulation or the proxy breaking down; there's no way to
tell which from this data. Two known weaknesses in the proxy, checked but
not fully resolved: (a) `ipa` carries stress marks (`ˈ`/`ˌ`) that `g2p`
never has (mean 8.0 stress marks/utterance vs. 0), inflating raw CER by
~1.8x (0.336 mean raw vs. 0.189 stripped) — checked whether this
contaminates the *ranking* rather than just the scale: `corr(raw CER,
n_words) = -0.015` and `corr(stripped CER, n_words) = -0.031` (both
negligible), and raw vs. stripped CER correlate at 0.915, so the length
confound doesn't appear to be driving the result, but this wasn't
re-verified against GOP directly with the stripped variant; (b) `g2p`/`ipa`
have no word-boundary markers, so a character-level edit at a word boundary
can "substitute" a stress mark for a neighboring word's phone — a
structural limitation of this proxy, not fixed here.

**All 6 L1s** (secondary, same weak-signal caveat applies to all of them):
Vietnamese shows the strongest signal for both engines (BPE -0.268, phone
-0.289); Arabic and Hindi are weak/mixed for both (|r| < 0.08); pooled
across all languages phone slightly trails BPE (-0.186 vs. -0.204) — the
opposite of the per-language Chinese lean, another sign of noise rather
than a stable pattern.

**Bottom line:** this test rules out *"L1 alone explains the ranking
flip"* — adult Mandarin-L1 speech does not reproduce speechocean762's
clear BPE-wins pattern, so L1 identity by itself isn't sufficient. It does
**not** confirm *"age/domain-match alone explains it"* either — the
phone model's edge here (0.119 vs. 0.107) is far short of UME-ERJ's
1.3-1.5x margin, and per-speaker results don't even agree on which model
wins. The honest conclusion is that this corpus, with this proxy, cannot
resolve the question at the effect size it would take — not that the
question is resolved in either hypothesis's favor.

**The sharper follow-up (implemented): `scripts/eval_l2arctic_phone.py`.**
The original TAMU/Kaggle L2-ARCTIC distribution's raw `annotation/*.TextGrid`
files give expert-aligned canonical-phone/perceived-phone/error-tag triples
per phone (substitution/deletion/addition, ARPABET) — exactly the
granularity `phones-accuracy` gives on speechocean762. All 1,200 files
needed for the 4 Mandarin speakers (`annotation/` + matching `wav/`, 150
manually-annotated utterances each) were fetched via the Kaggle API
per-file (whole-dataset downloads hit Google Drive/Kaggle-side rate
limits on this corpus; per-file requests didn't). A from-scratch Praat
TextGrid parser groups `phones`-tier intervals into their parent `words`-
tier word by time-overlap, extracts (canonical ARPABET phone, correct/
substitution/deletion) per phone (additions are excluded — an inserted
extra sound has no canonical phone to score), force-aligns each word with
the phoneme-CTC model, and reuses `reconcile_phones` unchanged to map
espeak's phones back onto the canonical ARPABET sequence.

**A real bug, caught before trusting any number:** the first version's
error-tag regex (`[A-Za-z]+` for the canonical-phone group) didn't match
ARPABET's stress digits, so any deletion/substitution of a *vowel*
(`"IH0, sil, d"` etc. — vowels always carry a stress digit) silently fell
through to the "bare correct phone" branch, mislabeling the error as
correct **and** feeding the literal string `"IH0, sil, d"` into
`reconcile_phones` as if it were a phone symbol. Fixed (`[A-Za-z]+\d?`);
BWC alone went from r=0.151 (buggy) to r=0.233 (fixed) on the same 150
utterances — a 54% relative change from one regex character class, exactly
the kind of silent corruption spot-checking individual reconciled examples
(the same technique used to validate `reconcile_phones` in section 5a) is
for.

**Result (all 4 Mandarin speakers, 600 utterances, 19,636 phones scored,
zero failures — `results/l2arctic_phone_all.json`):**

| metric | value |
|---|---|
| phone-level PCC (GOP vs. binary correct/error) | **0.206** (Spearman ρ 0.209) |
| fraction correct | 0.847 |

Lower than speechocean762's 0.425 (matched-only) / 0.433 (reconciled) —
about half. Two differences from speechocean762 make the two numbers not
directly comparable at face value, not just a straight "phone GOP works
worse here": (a) the label here is **binary** (correct/error) vs.
speechocean762's 0-2 graded scale, which caps achievable point-biserial
correlation relative to a graded target; (b) **the error-type mix differs,
and GOP-lite is asymmetric across error types** — checked directly, not
assumed: mean GOP is -0.395 for correct phones, **-1.106 for
substitutions** (n=2,419), **-2.303 for deletions** (n=596) — deletions
produce a ~3x larger deviation from "correct" than substitutions do, and
substitutions still show a median GOP of exactly 0.0 (same as "correct"),
meaning *most* substitutions aren't flagged at all. This is a real,
checked limitation of naive posterior-deficit GOP: a confidently-wrong
substitution (the model clearly recognizes *some* phone, just not the
target one) doesn't create the same posterior collapse a genuine deletion
(nothing resembling the target phone exists in the audio at all) does.
L2-ARCTIC's Mandarin-L1 errors are substitution-heavy (2,419 vs. 596, ~4:1
— consistent with well-documented L1-transfer phone substitutions like
θ→s, r→l, rather than omissions), which plausibly explains a meaningful
part of the gap to speechocean762's number, though speechocean762's own
error-type mix wasn't broken out the same way to confirm this
quantitatively — stated as the leading explanation, not a proven one.

**What this settles and doesn't:** phone-level GOP-lite is real signal on
a second corpus (r=0.206, clearly above zero, n=19,636) but weaker here
than on speechocean762 — and the substitution/deletion breakdown gives a
mechanistic reason for at least part of that gap, rather than leaving it
as an unexplained cross-corpus difference. It doesn't change the section
5c disentangling conclusion (still inconclusive at the utterance level);
a phone-level UME-ERJ-equivalent (Japanese-L1 adults, phone-level ground
truth) would be needed to run the same three-way disentangling comparison
at this sharper granularity, which UME-ERJ's holistic ratings can't
provide.

---

## 6. CLI commands summary  (completing the empty section from the old plan)

```bash
# --- setup (needs sibling repos ../sherox and ../audiokit checked out) ---
python -m venv .venv && source .venv/bin/activate
pip install -e ../audiokit && pip install -e ../sherox
pip install -r requirements.txt
python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger_eng')"
# NeMo CTC English ASR + Silero VAD + Piper TTS all auto-download into models/ on
# first CLI/web run — no manual download step.

# --- verify deps ---
python -c "from g2p_en import G2p; print(G2p()('hello'))"
python -c "from sherox.asr_engine import build_recognizer; from sherox.config import Config; print('ok')"
python -c "from sherox.tts import build_tts, synthesise_to_file, TtsConfig; print('ok')"

# --- run CLI ---
python cli.py
python cli.py --seconds 4 --prompt-file data/prompts.txt
python cli.py --tts-lang eng        # enable "play reference" before recording

# --- run web ---
uvicorn web.server:app --reload --port 8000
# open http://localhost:8000

# --- tests ---
pytest -q
python scripts/selftest.py
```

---

## 7. Limitations & assumptions
- **Intelligibility proxy:** a strong accent that is still intelligible may score
  high; a correctly-pronounced-but-unintelligible-to-ASR word may score low. State
  this to users. See section 5 for accent scoring.
- **Reference audio is synthetic:** the "play reference" voice is a Piper VITS
  model voice, not a human; fine as an imitation target, not a gold standard.
- **Quiet room / single speaker** gives the best ASR results. Background noise
  hurts accuracy; swap to a larger/better model via `config.py` - e.g. Whisper
  small.en (`whisper`), or NeMo CTC En (`nemo_ctc`, which also gives confidence).
- **Stress ignored by default** (normalized) - configurable via `--include-stress`.
- **Mic access** in CLI depends on PortAudio (Linux: `apt install portaudio19-dev`;
  macOS: `brew install portaudio`).
- Numbers/acronyms in prompts are normalized by `g2p_en` before scoring; align the
  *displayed* prompt with the *scored* text accordingly.

---

## 8. Maintenance
- Pin versions in `requirements.txt`; run `pip-audit` periodically.
- Default ASR is NeMo CTC En (per-word confidence, auto-downloaded). Swap via
  `config.py` if accuracy is insufficient: NeMo CTC small (lighter), Parakeet TDT
  int8 (transducer, no confidence), Whisper small.en, Moonshine. Try
  `faster-whisper` only as a last resort.
- Upgrade `sherox`/`audiokit` and try other Piper voices (`--tts-lang`) for
  reference audio; consider the GOP-lite path (section 5) once stable.
- Expand `data/prompts.txt` and `data/lexicon.txt` from real user errors.
- Keep scoring weights and the phoneme-hint table in `config.py` (single source).
- Log scoring results anonymously (target vs recognized, score) to improve prompts.

---

## 9. Future TODO: multi-language (Indonesian, Arabic)

**English is the only target for v1.** This section records the plan for adding
Indonesian (id) and Arabic (ar) later, so today's design doesn't
accidentally block it.

### Single repo, not separate repos (recommendation)

Keep **one repo** (rename `proscor-en` -> `proscor` when multi-language lands).
Reason: the whole pipeline - record -> ASR -> G2P -> score -> feedback -> CLI/web -
is language-agnostic. Only three things vary per language:

1. **G2P backend** (the only genuinely per-language piece):
   - English: `g2p_en` (CMUdict + neural OOV).
   - Indonesian: near one-to-one orthography -> a small rule table / `epitran`-style
     mapping suffices; or a lexicon.
   - Arabic: mostly one-to-one but with diacritic/hamza/sun-letter rules; a rule
     table + lexicon.
   - A common `expected_phonemes(text, lang)` interface with per-lang backends keeps
     `score.py` unchanged. Cross-language phoneme comparison needs a shared inventory
     (e.g. IPA via NRC-ILT `g2p`, or map each language to a common phone set).
2. **Prompt/lexicon data**: `data/prompts.<lang>.txt`, `data/lexicon.<lang>.txt`.
3. **sherox ASR + TTS model selection** (already multilingual via `--lang`/model_dir):
   - ASR: sherox has a multilingual streaming zipformer (ar/en/id/...).
     Indonesian/Arabic can use the multilingual model or a per-language sherpa-onnx
     model.
   - TTS: sherox has `ind` (Piper id_ID) and Arabic via Supertonic-3 (`ara`).

So multi-language is a **registry/config addition**, not a new codebase. Split into
separate repos only if a language needs a fundamentally different scoring algorithm
or a separate release cadence - none of id/ar do.

### Why not now

- **G2P quality is the long pole:** id/ar are easy, but doing it well per language
  is real work; doing it badly hurts scoring. Ship a solid English v1 first, then
  add one language at a time.
- **Cross-language phone-set alignment** (so the edit-distance score is comparable
  across languages) needs a deliberate inventory decision - defer until the 2nd
  language lands.

### TODO list (when we get there)
- [x] Rename `proscor-en` -> `proscor`
- [ ] Add `LANG` config + `--lang` CLI flag. Make default to English
- [ ] `proscor/g2p.py`: pluggable backends per lang (id: rules, ar: rules);
      common phone inventory (IPA or a shared set).
- [ ] `data/prompts.<lang>.txt` + `data/lexicon.<lang>.txt` per language.
- [ ] `config.py`: per-lang ASR model_dir/type + TTS lang (sherox already supports).
- [ ] Per-language tests; per-language feedback phoneme-hint table.
