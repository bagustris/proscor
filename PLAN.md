# PLAN.md: English Pronunciation Scoring (proscor-en)

> Show words -> the user reads them aloud -> the system returns a 0-100 score and feedback.
> Target: **simple**, runs on **any PC** (CPU only, no GPU), usable as a **CLI** and a **web app**.
> Scope: **English only.** No other languages are planned.

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
the target) via `sherox.tts`. Everything runs on CPU. **English only** — no
other languages are planned.

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
> Aligner / Kaldi). That is provided as an **optional advanced track** in section 5,
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
   phoneme labels, while speechocean762 is L2 English speech from a
   mixed-age pool (its `age` field runs 6-43, roughly half the speakers
   under 18 — this was previously described here as "child speech";
   corrected in section 5e), a harder acoustic domain than adult
   native/fluent L2 speech; (b)
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
speakers of mixed age, 6-43 — earlier drafts of this section called it a
children's corpus; see the correction in section 5e).
[UME-ERJ](https://research.nii.ac.jp/src/en/UME-ERJ.html) (NII
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
   speechocean762's include children (its `age` field runs 6-43, about
   half the speakers under 18 — this line originally said "documented
   children (ages 5-15)", which was wrong; see section 5e), the phone
   model's training distribution is closer to UME-ERJ's speakers
   acoustically. That speechocean762 is only *half* children weakens this
   explanation a priori but doesn't kill it.
2. **L1-specific error profile.** Japanese-accented English has a
   well-documented, specific phoneme-substitution profile (/l/~/r/,
   /θ/~/s/, vowel epenthesis, /v/~/b/) that a genuine phoneme-level model
   may be structurally better positioned to catch than a BPE model routing
   everything through orthography; Mandarin-accented (mixed-age) speech has
   a different error profile the BPE model might happen to fit better.

Both are plausible, both are consistent with the data, and this pair of
corpora changes L1 *and* (probably) age/register simultaneously, so neither
can be isolated here. A third corpus that holds one variable fixed while
changing the other (e.g. a documented-adult, non-Japanese-L1 corpus, or a
documented-child Japanese-L1 corpus) would be needed to separate them —
noted as a natural follow-up, not done. (Section 5c does the adult-Mandarin
version with L2-ARCTIC; section 5e does a within-corpus version using
speechocean762's own `age` field, which turned out to span 6-43.)

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
(Mandarin-L1, BPE wins — described as "children" when this section was
written; it's actually mixed-age 6-43, see section 5e) and UME-ERJ
(Japanese-L1 adults, phone wins), with two candidate explanations that two
corpora can't separate: acoustic/age domain match, or an L1-specific
phoneme-substitution profile a phone model is structurally better
positioned to catch. **The test:**
[L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) has **adult
Mandarin-L1** speakers — same L1 as speechocean762, same age category as
UME-ERJ — so if adult-Mandarin behaves like speechocean762's Mandarin
speakers (BPE wins), L1 dominates; if it behaves like adult-Japanese (phone
wins), age/domain dominates.

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

**Full corpus, all 24 speakers, all 6 L1s (implemented):** the Mandarin
result above was the first 1,200 files fetched (4 speakers) while the
remaining 20 speakers' TextGrids/wavs downloaded in the background (all
7,198 files, per-file via the Kaggle API — whole-dataset downloads
continued to 404 on this mirror; per-file requests eventually hit a
sustained low-throughput 429 rate-limit toward the end of the run, worked
around with slower client-side pacing (0.3s -> 1.2s/file) and patience
rather than anything clever — zero permanent failures, every file
eventually succeeded via retry). `scripts/eval_l2arctic_phone.py` was
refactored to track results per-speaker (`by_speaker` dict keyed off a
`SPEAKERS_BY_LANG`/`LANG_BY_SPEAKER` mapping for all 24 speakers x 6 L1s)
instead of one pooled list, enabling a `by_language` breakdown — regression-
tested against the untouched Mandarin-only result first (pooled r=0.2061,
n=19,636, exact match) to confirm the refactor changed no behavior before
trusting the new numbers.

**Result (3,599 utterances, 118,455 phones scored, zero failures —
`results/l2arctic_phone_full.json`):**

| scope | n phones | PCC | fraction correct |
|---|---|---|---|
| pooled (all 24 speakers) | 118,455 | **0.224** | 0.854 |
| Vietnamese | 20,017 | 0.371 | 0.766 |
| Arabic | 19,644 | 0.219 | 0.905 |
| Mandarin | 19,636 | 0.206 | 0.847 |
| Spanish | 19,946 | 0.186 | 0.840 |
| Korean | 19,578 | 0.103 | 0.909 |
| Hindi | 19,634 | 0.060 | 0.864 |

**Yes, the full corpus gives a different result — a 6x spread by L1 (0.06
to 0.37) that the 4-Mandarin-speaker run couldn't show.** The pooled
number (0.224) is close to the Mandarin-only number (0.206) more or less
by coincidence (Mandarin happens to sit near the middle of the range),
which would have been easy to over-read as "representative" without
running the other 20 speakers.

**Correction (caught before this went further): the "substitution gap
ranks languages exactly" framing below was overclaimed — it's largely
restating the point-biserial formula, not an independent finding.**
Point-biserial r is literally `(mean_correct - mean_error) * sqrt(p*q) /
SD`. Substitutions are 80-93% of all errors in every L1 here, so
`mean_error` is dominated by the substitution mean, and a quick check
confirms it: recomputing the formula's numerator directly from the
per-language tag means (weighting substitutions and deletions by count,
not isolating substitutions alone) reproduces the same 6/6 language
ordering as the actual Pearson r. Ranking languages by a quantity that's
algebraically most of the correlation's own numerator is close to
circular, not a mechanistic discovery — a 1/720 permutation p-value would
have been the wrong test to reach for here (the null of "random rank
order" doesn't apply when the two quantities are this entangled).

**What's still true and worth keeping, stated as arithmetic rather than a
"prediction that came true":** every language's substitution-error phones
have a median GOP of exactly 0.0 (statistically indistinguishable from
"correct" at the median), while deletions are always caught (means -1.0
to -3.6). Substitution-error *mean* GOP itself varies a lot by L1 — from
-0.42 (Hindi) to -1.58 (Vietnamese) — while mean correct-phone GOP is
nearly flat across L1s (-0.28 to -0.40). Since substitutions dominate the
error population everywhere, that spread in substitution-mean-GOP is
*where* the L1-driven range in phone-level PCC numerically lives. This
doesn't independently explain *why* Vietnamese/Arabic substitutions
produce a larger mean posterior deficit than Hindi/Korean ones do (a
plausible guess is that their characteristic L1-transfer substitutions —
consonant-cluster and final-consonant phenomena — swap in a phone
acoustically further from the target, but that's not verified against a
phonological-distance metric here) — it just locates the effect instead of
leaving it as an unexplained per-language spread. The plain range across
languages is 0.06-0.37; "6x" is technically correct but oversells a small
absolute range as more dramatic than it is.

This also isn't purely an error-rate artifact: ranking by raw error rate
(`1 - frac_correct`) does **not** reproduce the PCC order (Arabic has the
*fewest* errors of all 6 languages, 9.5%, yet the second-highest PCC).

**What this does and doesn't add to the 5c disentangling question:**
this run only exercises the phone model (the BPE model has no phone-level
output, per the table earlier in this section), so it cannot repeat the
BPE-vs-phone ranking-flip test at phone granularity for the other 5 L1s.
What it does add: L2-ARCTIC holds age fixed (all 24 speakers are adults)
while L1 varies across all 6 languages, and phone-level PCC still swings
6x (0.06-0.37) purely as a function of L1-specific error-type profile.
That's new, direct evidence that L1-specific error profile is a real and
large effect on its own, independent of age — it doesn't resolve which
factor drove the original section 5b BPE-vs-phone ranking flip (that
remains open), but it rules out dismissing "L1-specific error profile" as
a minor or speculative factor next to "age/domain match".

---

### 5d. Statistical rigor pass: cluster-bootstrap CIs and paired-difference tests

Everywhere above, Pearson r is reported as a point estimate at the raw
item count (words/phones/utterances) with no uncertainty interval, and
every "engine A beats engine B" claim (5a: BPE beats phone at word level;
5b: phone beats BPE on UME-ERJ; 5c: a near-tie on L2-ARCTIC) rests on
eyeballing two point estimates. Both are real gaps for a systems paper: a
Fisher-z CI computed on the item count would be wrong (words/phones cluster
within speakers — a child who's a strong/weak speaker produces correlated
scores across all their words), and comparing two independently-computed
marginal CIs for overlap is a weaker test than it looks (two engines
scoring the *same* audio produce correlated errors, so their difference
has less sampling noise than the two marginals combined would suggest).

**Method** (`proscor/stats.py`, unit-tested in `tests/test_stats.py`):
`cluster_bootstrap_pearson` resamples whole speakers (not items) with
replacement, 2000 draws, percentile 95% CI — the item count is reported
alongside the cluster count so a reader can see which one actually governs
the interval width. `cluster_bootstrap_paired_diff` does the same but on
`r(engine_A, label) - r(engine_B, label)` computed on the *same* resampled
clusters each draw, preserving the item-level pairing; "significant" means
the CI excludes zero. `kruskal_by_group` is a Kruskal-Wallis H-test taking
per-*speaker* r values (not per-item scores) as the unit, for "does language
have a real effect on phone-level PCC" (section 5c).

**5a (speechocean762), word-level BPE vs. phone — the flip survives:**
BPE r=0.471 (CI 0.390-0.536) vs. phone r=0.325 (CI 0.264-0.382); paired
diff = 0.146 (CI **0.102-0.186, excludes zero**) — BPE's word-level win is
real, not point-estimate noise. **Utterance-level does NOT survive**: BPE
0.556 (CI 0.484-0.618) vs. phone 0.536 (CI 0.456-0.604), paired diff =
0.021 (CI **-0.043 to 0.090, includes zero**) — the previously-reported
"BPE 0.556 vs phone 0.536" utterance-level margin should not be read as a
real difference; the two engines are statistically indistinguishable at
utterance granularity, only the word-level comparison supports "BPE wins."
(`scripts/eval_so762_paired.py`, `results/so762_paired_test.json`.)

**5a phone-level, binarized-label check (the cheap test flagged as owed in
section 5c):** speechocean762's phone label is graded 0-2
(reconciled PCC 0.433, CI 0.387-0.477); binarizing it the same way
L2-ARCTIC's label is binary (`accuracy == 2` -> correct) drops the
correlation to r=0.337 (CI 0.305-0.370). The like-for-like comparison is
against L2-ARCTIC's *pooled* binary phone-level r=0.224 (CI 0.159-0.280,
not the Mandarin-only 0.206 — so762's 125 speakers are a mixed-ability
pool, not a single-L1 slice, so pooled is the right comparison): original
gap 0.433-0.224=0.209, binarized gap 0.337-0.224=0.113, so label
granularity alone closes about 46% of it (0.096 of 0.209) — real corpus
difficulty accounts for less of the original gap than it looked like, but
a real gap remains (0.337's CI floor, 0.305, still sits above 0.224's CI
ceiling, 0.280 — not overlapping). (`scripts/eval_so762_phone.py`,
`phone_level_reconciled_binarized` in `results/so762_phone_test.json`.)

**5b (UME-ERJ), BPE vs. phone per category — the ranking flip is real for
the categories that matter most:**

| category | BPE r (CI) | phone r (CI) | paired diff (CI) | verdict |
|---|---|---|---|---|
| sentence-segmental | 0.328 (0.284-0.373) | 0.432 (0.384-0.475) | -0.104 (-0.155 to -0.050) | **phone wins, significant** |
| word-segmental | 0.294 (0.262-0.323) | 0.428 (0.393-0.461) | -0.135 (-0.170 to -0.100) | **phone wins, significant** |
| sentence-rhythm | 0.055 (-0.015-0.130) | 0.156 (0.079-0.231) | -0.101 (-0.176 to -0.020) | significant, likely a halo effect (see below) |
| sentence-intonation | 0.056 (-0.013-0.122) | 0.039 (-0.035-0.111) | 0.017 (-0.060-0.088) | no difference |
| word-accent | 0.110 (0.058-0.162) | 0.106 (0.053-0.160) | 0.004 (-0.062-0.068) | no difference |

The two categories that most directly parallel speechocean762's
"pronunciation accuracy" target — sentence- and word-*segmental* — both
show phone beating BPE with the CI clearly excluding zero. That's the
load-bearing result: **the 5a-vs-5b ranking flip is a statistically real
phenomenon, not two noisy point estimates that happened to land on
opposite sides.** Rhythm's significant phone-favoring diff should **not**
be read as a third confirmation of the same pattern: GOP has no rhythm
model, so a plausible explanation is that raters' rhythm scores correlate
with their (unrecorded) segmental impression of the same recording — a
halo effect on the human rating, not a genuine rhythm-sensitivity
difference between engines. Intonation and accent show no significant
difference either way, consistent with GOP measuring phone/word identity
fit rather than prosody.
(`scripts/eval_umeerj.py`, `results/umeerj_summary_v2.json`.)

**5c CER-proxy (L2-ARCTIC), BPE vs. phone — the Chinese near-tie is
confirmed as a real (non-significant) tie, not just an underpowered
guess:** pooled (24-speaker cluster) paired diff = -0.018 (CI -0.050 to
0.022, includes zero); Chinese alone (the original disentangling test) =
0.012 (CI -0.075 to 0.094, includes zero). Both confirm the section 5c
conclusion ("this test can't resolve the question at this effect size")
was the right call, not an artifact of not having run the numbers.
**The other five languages show a messier picture that should NOT be
over-read as five more real per-language effects:** Arabic, Hindi, and
Korean show paired diffs whose CIs exclude zero, but two of the three
have a wrong-signed or non-significant *marginal* correlation for at
least one engine (Korean's BPE r=+0.131 is wrong-signed per the original
5c caveat about this proxy breaking down for some subpopulations; Hindi's
own BPE r's CI already includes zero) — a "significant" paired difference
between two shaky or wrong-signed marginals is not evidence of a real
engine ranking, it's evidence the underlying char-edit-distance proxy is
too noisy to trust language-by-language at n=4 speaker-clusters. Spanish's
paired diff is significant despite *both* marginal CIs including zero, a
reminder that the paired test controls a different source of noise than
the marginals and the two can disagree — take it as a caveat on reading
marginal CIs as sufficient, not as evidence Spanish is special.
(`scripts/eval_l2arctic.py`, `results/l2arctic_summary_v2.json`.)

**5c phone-level (all 24 L2-ARCTIC speakers), the per-language spread —
real overall, but only the extremes separate individually:** pooled
r=0.224 (CI 0.159-0.280, 24 speaker clusters). Per-language, 95% CIs from
a 4-speaker cluster bootstrap:

| language | r | 95% CI |
|---|---|---|
| Vietnamese | 0.371 | 0.264-0.427 |
| Arabic | 0.219 | 0.148-0.270 |
| Mandarin | 0.206 | 0.158-0.236 |
| Spanish | 0.186 | 0.104-0.251 |
| Korean | 0.103 | 0.052-0.148 |
| Hindi | 0.060 | 0.042-0.080 |

**Caveat on the table above:** at n=4 clusters, resampling 4 speakers
with replacement has only 35 distinct multisets, so the bootstrap
percentiles are coarse — the *width* of each CI is informative (it's
honestly telling you 4 speakers isn't much), but the exact bounds aren't
precise to the decimal shown, and a narrower CI (e.g. Mandarin's
0.158-0.236 vs. Spanish's 0.104-0.251) reflects that those 4 speakers
happened to be more homogeneous, not that the estimate is more accurate.
Pairwise CI overlap here is a descriptive summary, not a formal test —
the same limitation already flagged for the engine-comparison CIs above.

The formal inferential statement is the Kruskal-Wallis test on the 24
per-speaker r's grouped by language (the correct unit — speaker, not
phone): significant, H=14.75, **p=0.0115** — language does have a real
effect on phone-level PCC, not just sampling noise dressed up as a
spread. Descriptively, only the extremes clearly separate: Hindi's CI
doesn't overlap Vietnamese's, Arabic's, or Mandarin's; the middle
cluster — Arabic, Mandarin, Spanish, Korean — has heavily overlapping
CIs and shouldn't be read as four distinguishable points on a ranked
list (Vietnamese's CI floor, 0.264, sits just above Arabic's ceiling,
0.270 — at 35 distinct resamples that's noise-level precision, not a
confirmed separation). Read section 5c's per-language table as "one
clear top language, one clear bottom language, and a wide
indistinguishable middle," not as a precise 6-way ranking. One
confound this data can't rule out: L2-ARCTIC's annotations came from
multiple human annotators with no per-speaker annotator ID in the
release, so per-speaker annotator differences (strictness, sub-vs-del
labeling habits) may contribute to the between-speaker r variance —
not separable from L1 here.
(`scripts/eval_l2arctic_phone.py`, `results/l2arctic_phone_full.json`.)

---

### 5e. Closing the open questions: the age split, clean L2-ARCTIC word labels, and the annotator confound

Section 5d left two questions open: (1) is the 5a-vs-5b BPE/phone ranking
flip driven by speaker *age* (acoustic domain match) or by *L1* (error
profile)? and (2) could per-annotator differences be behind L2-ARCTIC's
per-language spread? This section does what could be done about each
without new data.

**A correction first: speechocean762 is not a children's corpus.** Every
earlier section describes it as "Mandarin-L1 children (ages 5-15)." Its
`age` field actually runs **6-43**: in the `test` split, 64 of 125
speakers are under 18 (1,280 utterances) and 61 are 18+ (1,220
utterances); `train` is 58/67. There are no speakers aged 16-18, so 18 is a
clean cut. The earlier characterization has been amended in place in 5a,
5b, and 5c with pointers here rather than silently rewritten. It also
means the cheapest possible age test was sitting in data already on disk.

**(1) Age, tested within one corpus.** `scripts/eval_so762_paired.py` now
records each speaker's age and runs the paired, speaker-cluster BPE-vs-
phone comparison separately for the two age groups — same L1, same
annotators, same 0-10 rating scheme, same recording setup; only age
varies (`results/so762_paired_age_test.json`; per-item arrays saved
alongside so re-cuts don't need another inference run):

| group | n words / speakers | BPE r (CI) | phone r (CI) | paired diff (CI) | verdict |
|---|---|---|---|---|---|
| under 18 | 7,266 / 64 | 0.409 (0.236-0.528) | 0.285 (0.160-0.396) | +0.124 (0.063-0.169) | **BPE wins** |
| 18 and over | 8,701 / 61 | 0.514 (0.426-0.576) | 0.343 (0.278-0.397) | +0.171 (0.112-0.224) | **BPE wins** |
| all (5d, reproduced exactly) | 15,967 / 125 | 0.471 (0.390-0.536) | 0.325 (0.264-0.382) | +0.146 (0.102-0.186) | BPE wins |

BPE beats the phone model in **both** age groups, with the CI excluding
zero both times — and the adult subset's BPE margin is the *larger* one.
**Age is not what drives the 5b flip.** If the phone model's advantage on
UME-ERJ came from its adult-CommonVoice training data matching adult
speakers acoustically, adult Mandarin speakers in speechocean762 should
have moved toward the phone model; they moved the other way. (Utterance
level, secondary: under-18 diff +0.066, CI 0.012-0.112, significant; 18+
diff +0.029, CI -0.057-0.121, not — consistent with 5d's finding that the
utterance-level comparison is the weaker one.)

This also reframes section 5c: L2-ARCTIC's adult Mandarin speakers showed
a "tie" between engines, but speechocean762's adult Mandarin speakers show
BPE winning clearly — same L1, same age category. The difference between
those two results is therefore not age; it's either the char-edit-distance
proxy 5c had to use (already flagged as noisy) or something about the
corpus itself (read CMU ARCTIC prompts, proficiency range, microphone).
Part (2) below separates those with clean labels.

**(2) L1, tested among adults with clean labels.**
`scripts/eval_l2arctic_word.py` replaces 5c's char-edit-distance proxy
with word labels derived from the expert per-phone TextGrid tags (same
parser as `eval_l2arctic_phone.py`): `frac_correct` = share of a word's
canonical phones tagged correct (graded; 85.2% of words are 1.0), and
`any_error` (binary; 39.7% of words have at least one error). Both
engines score the same 33,980 words from all 24 speakers (3,599
utterances, zero failures), paired speaker-cluster bootstrap as in 5d
(`results/l2arctic_word.json`, per-word records in `.arrays.json`).
Word level vs. `frac_correct` (expected sign positive):

| L1 (4 adult speakers each) | n words | BPE r (CI) | phone r (CI) | paired diff (CI) | verdict |
|---|---|---|---|---|---|
| Arabic | 5,661 | 0.142 (0.106-0.174) | 0.074 (0.026-0.104) | +0.068 (0.043-0.095) | BPE wins |
| Hindi | 5,661 | 0.064 (0.037-0.086) | -0.039 (-0.060 to -0.005) | +0.103 (0.085-0.133) | BPE wins; phone wrong-signed |
| Korean | 5,618 | 0.069 (0.007-0.155) | -0.006 (-0.047-0.046) | +0.075 (0.050-0.110) | BPE wins; phone ~zero |
| Mandarin | 5,647 | 0.144 (0.103-0.169) | 0.029 (0.016-0.043) | +0.115 (0.081-0.140) | BPE wins |
| Spanish | 5,702 | 0.092 (0.001-0.157) | 0.034 (-0.053-0.128) | +0.058 (-0.014-0.103) | BPE leads; diff CI includes 0 (binary label: -0.123, CI -0.134 to -0.105, significant) |
| Vietnamese | 5,691 | 0.288 (0.195-0.343) | 0.181 (0.065-0.266) | +0.107 (0.057-0.144) | BPE wins |
| pooled (24 speakers) | 33,980 | 0.169 (0.102-0.223) | 0.067 (0.015-0.119) | +0.102 (0.080-0.117) | **BPE wins** |

Utterance level (mean word GOP vs. mean `frac_correct`, pooled): BPE
0.302 (CI 0.164-0.387) vs. phone 0.160 (0.018-0.265), diff +0.143 (CI
0.104-0.181). Absolute correlations are lower than on speechocean762
because these labels are near-binary (85% of words are exactly 1.0),
which caps point-biserial-style r; the *comparison* between engines is
what this test is for.

**BPE beats the phone model in every one of the six adult L1s** (five
with the CI excluding zero; Spanish's graded-label diff just includes it
but its binary-label diff doesn't), with the same four-speaker caveat as
5d's per-language CIs. Two consequences: (a) 5c's Mandarin "tie" was an
artifact of the char-edit-distance proxy — with clean labels, adult
Mandarin speakers show BPE ahead by +0.115, matching speechocean762's
adult Mandarin subset (+0.171) — so 5c's "this test is underpowered"
conclusion was right for the wrong reason: the *label* was the problem,
not the sample size. (b) The phone model is close to zero or even
wrong-signed at word level for Hindi and Korean, the two L1s where its
*phone*-level PCC was weakest in 5c (0.060 and 0.103): whatever it
catches per phone doesn't survive aggregation to words there.

**Where this leaves the 5b question.** Putting 5a, 5c, and 5e together:
BPE beats the phone model on speechocean762 (Mandarin, children *and*
adults) and on L2-ARCTIC (adults; Arabic, Hindi, Korean, Mandarin,
Spanish, Vietnamese). The phone model wins only on UME-ERJ (Japanese
adults). So the flip is **not age** (5e-1) and **not "non-Mandarin L1"
in general** (5e-2: five non-Mandarin L1s still favor BPE). What's left
is specific to UME-ERJ: either Japanese-L1 specifically (the narrowest
form of 5b's "L1-specific error profile" hypothesis — plausible, since
Japanese-accented English has an unusually systematic substitution
profile, but note the phone model was *bad* at substitutions on
L2-ARCTIC), or properties of the corpus itself (holistic 1-5 ratings by
five native-teacher raters rather than phone-derived labels, recording
setup, proficiency range). Those two can't be separated with the corpora
we have; it would take either a second Japanese-L1 corpus with
pronunciation labels or UME-ERJ-style holistic ratings collected on
L2-ARCTIC. For the paper, the honest statement is: the BPE model is the
better zero-shot GOP engine on every corpus with phone-derived or
per-word expert labels we tested, across seven L1s and both age groups;
UME-ERJ is the one documented exception, and we can localize it to the
corpus, not to age or to L1 in general.

**(3) The annotator confound: checked against every public source, not
resolvable.** The corpus README (fetched from the Kaggle mirror) documents
the tag conventions and, for the *spontaneous* "suitcase" subset only, the
procedure ("two research assistants ... each did half ... then checked the
other half ... all transcriptions were checked by John Levis"). For the
3,599 scripted annotations used here it says nothing about who annotated
which speaker. The Interspeech 2018 paper (Zhao et al., section 3.3) adds
only: "The annotators (N=3) were PhD students in the Applied Linguistics
and Technology program at ISU. They were experienced in transcribing speech
samples of native or non-native English speakers," plus automated
consistency checks with human fix-ups — no speaker-to-annotator
assignment, no double-annotated subset, no inter-annotator agreement. (That
paper covers the initial 10-speaker release; the README credits two more
people with annotation help for the 24-speaker release.) The TAMU docs
page repeats the README. So the confound stays a stated limitation; the
only remaining route is asking the authors for the mapping. Two things
limit its reach: the paper's claim is "L1 has an effect and the extremes
differ," not a precise ranking; and 5c already showed error *rate* doesn't
explain the PCC spread (Arabic: fewest errors, second-highest PCC), which
argues against a pure annotator-strictness story.

**Two sanity checks recorded while reading the README.** (a) It gives the
official error totals for the 3,599 annotated utterances — 14,098
substitutions, 3,420 deletions, 1,092 additions. `eval_l2arctic_phone.py`
scored 13,869 substitutions and 3,375 deletions (98.4% / 98.7%; the
remainder are phones `reconcile_phones` couldn't map or words filtered
for non-alphabetic text), and excludes additions by design — the parser's
coverage matches the corpus's own bookkeeping. (b) The paper's own
baseline (section 5, Kaldi GMM GOP with phone-independent thresholding,
10 speakers) reaches only precision = recall = 0.29 on *substitution*
detection — an independent, peer-reviewed number agreeing with 5c's
finding that substitutions are the hard case on this corpus.

---

### 5f. Segmentation-free GOP: fixing the substitution blindness directly

Sections 5c/5e's substitution/deletion asymmetry (median GOP exactly 0.0
for a substituted phone, in every L1 tested) has a specific mechanism:
`align_words_gop`'s posterior-deficit score compares the Viterbi-aligned
frames' log-prob for the *canonical* phone against the best phone at
those same frames. With a peaky CTC model, Viterbi still has to place the
canonical phone's required state somewhere, and it picks the least-bad
frames available — but if the speaker confidently produced a *different*
phone throughout that whole stretch, "posterior at the aligned frame" can
still land close to 0, because the frames Viterbi picked aren't
necessarily the frames where the wrong phone's dominance is starkest.

**Cao, Fan, Svendsen & Salvi, "Segmentation-free Goodness of Pronunciation"**
(arXiv:2507.16838, IEEE 2025) sidesteps alignment entirely: compare two
*whole-sequence* CTC likelihoods, both marginalized over every possible
alignment via the forward algorithm (no Viterbi step at all) — log
P(canonical phone sequence) vs. log P(the same sequence with one phone
replaced by "any phone, or nothing"). `GOP_SF(i) = logP(L_C) - logP(L_SDI)`
sits at its 0 ceiling when the canonical phone is clearly the only good
explanation for that stretch of audio, and drops toward a large deficit
when some alternative (including a confidently-produced *wrong* phone)
explains it just as well or better — exactly the case posterior-deficit
GOP structurally can't distinguish from "no strong alternative exists."

**Implementation** (`proscor/align_phone.py`: `gop_sf`, `align_words_gop_sf`,
drop-in replacements for `align_words_gop`'s per-phone/per-word GOP,
wired into `scripts/eval_so762_phone.py`/`eval_l2arctic_phone.py` as
`--engine sf`): computing the substitution term efficiently — "some phone
from the ~392-symbol vocabulary was produced here," marginalized without
Viterbi — took two wrong attempts before a correct one, both caught by
brute-force comparison (`tests/test_align_phone.py`) before being
trusted, not after:
1. A single forward pass with the target position's emission replaced by
   log-sum-exp over every candidate's log-prob *per frame*. Wrong, not
   just imprecise: CTC's "stay" transition lets a state persist over
   several frames, and a per-frame log-sum-exp lets frame *t* implicitly
   vote for one candidate while frame *t+1* (still the same persistence)
   votes for a different one — not a valid single-candidate path. This
   overcounted probability mass by 2x-40x on toy examples.
2. Restricting the candidate set to a small local top-K subset (avoiding
   attempt 1's bug by never needing the shared per-frame trick). Correct,
   but a different, weaker metric than the paper's full-vocabulary
   marginalization, and awkward to compare against their published numbers.
3. **The correct fix**, kept: candidate identity only matters *while a
   path is inside the wildcard state's own self-loop*; entry into and
   exit from that state can be safely marginalized over candidates,
   because every candidate shares the same predecessor/successor
   topology (given the precondition that the candidate set excludes the
   phones immediately flanking that position — a CTC skip-transition
   subtlety, see the function's docstring). So the self-loop is tracked
   as one independent running likelihood per candidate (no cross-candidate
   mixing at any single frame), merged via logsumexp only at entry/exit —
   the same shared-prefix/shared-suffix factoring a forward-backward
   derivation would give, in one forward pass. `O(T*(S+V))` per phone
   position, matching the paper's stated complexity; verified against
   brute-force enumeration over the *full* (not restricted) candidate set.

**Cost, measured on the real model (not estimated):** so762 full test
split (2,500 utterances) ran in 2,065s = 0.83s/utterance; L2-ARCTIC full
corpus (3,599 utterances, 24 speakers) ran in 2,835s = 0.79s/utterance.
Both are full background runs, not a projection — roughly 2-3 orders of
magnitude slower than posterior-deficit's single Viterbi pass per
utterance, same order as the earlier synthetic/small-sample estimate, but
easily affordable for a corpus-scale run (still evaluation-only, not
wired into `--engine gop-lite`).

**Result: the mechanism works exactly as predicted, but the aggregate
correlation gain is modest, not transformative.**

The direct test of the hypothesis — does GOP-SF separate substitutions
from correct phones where posterior-deficit couldn't — is unambiguous.
Pooled over all 24 L2-ARCTIC speakers (`results/l2arctic_phone_sf.json`
vs. `l2arctic_phone_full.json`, same phones, same reconciliation):

| tag | posterior-deficit mean / median | GOP-SF mean / median |
|---|---|---|
| correct | -0.359 / **0.0** | -0.571 / -0.061 |
| substitution | -1.062 / **0.0** | -1.413 / -0.223 |
| deletion | -2.832 / -0.662 | -3.121 / -1.586 |

Under posterior-deficit, substitution and correct phones had the *exact
same* median (0.0 = 0.0, not just close) — the literal blind spot section
5c/5e kept finding. Under GOP-SF, they're separated (-0.061 vs. -0.223,
a real if modest gap) for the first time. Deletions also separate more
clearly (median gap from correct widens from 0.662 to 1.525). This
confirms the mechanism: marginalizing over every alignment instead of
scoring one Viterbi path does catch confidently-wrong substitutions that
posterior-deficit structurally missed.

**But this only moved the aggregate correlation a little, with heavily
overlapping CIs (no paired significance test run yet — see below):**

| metric | posterior-deficit r (CI) | GOP-SF r (CI) |
|---|---|---|
| so762 word-level | 0.325 (0.264-0.382) | 0.338 (0.275-0.396) |
| so762 utterance-level | 0.536 | 0.560 |
| so762 phone, matched-only | 0.425 (0.380-0.467) | 0.432 (0.387-0.475) |
| so762 phone, reconciled | 0.433 (0.387-0.477) | 0.441 (0.395-0.486) |
| so762 phone, reconciled binarized | 0.337 (0.305-0.370) | 0.354 (0.320-0.388) |
| L2-ARCTIC phone, pooled (24 speakers) | 0.224 (0.159-0.280) | 0.233 (0.169-0.287) |

Every number moved in the right direction, none by much, and every pair
of CIs overlaps substantially — these should be read as "consistent with
a small real improvement" not "GOP-SF proven better." A proper answer
needs the same paired, speaker-cluster bootstrap used everywhere else in
this plan (section 5d), scoring both engines on the *same* items in one
run (mirroring `scripts/eval_so762_paired.py`) rather than comparing two
separately-bootstrapped marginal CIs — not done here (each full-corpus
run already takes 35-47 minutes single-engine; a paired run scores both
engines per item, roughly doubling that). Noted as the natural next step,
not run yet.

**Why the aggregate gain is modest despite the clean mechanistic fix:**
substitutions are 80-93% of all errors in every L1 (section 5d), but even
under GOP-SF the median substitution deficit (-0.223) is still an order
of magnitude smaller than the median deletion deficit (-1.586) — GOP-SF
fixed the *structural* blindness (substitutions are no longer
indistinguishable from correct at the median) without closing most of
the *magnitude* gap to deletions. A confidently-produced wrong phone
apparently still costs less log-likelihood, on the model's own terms,
than an outright missing one — which may be a genuine acoustic fact
about this model and these substitution errors (many are subtle,
phonetically close swaps, e.g. θ→s) rather than a remaining bug.

---

### 5g. A third hypothesis for the UME-ERJ flip, and where the search for a fourth corpus stands

Section 5e ruled out age (speechocean762's own adult subset still favors
BPE) and ruled out non-Mandarin L1 in general (BPE wins in all six adult
L2-ARCTIC L1s with clean labels). That left the UME-ERJ flip localized to
one corpus, with two candidate explanations that couldn't be separated:
Japanese-L1-specificity, or something about UME-ERJ's own methodology
(holistic 1-5 ratings vs. phone-derived labels). Asked to find a way to
close this, two things came out of it: one clarification that removes a
false lead, and one new, checked, third hypothesis.

**UME-ERJ *is* ERJ — confirmed directly, not inferred.** `/data/UME-ERJ`'s
own `doc/introduction.txt` (Shift-JIS; `iconv -f SHIFT_JIS -t UTF-8`)
states its short name outright: "略称：ERJ データベース" ("Abbreviation:
ERJ database"), by Minematsu et al., 202 speakers (100M/102F) — matching
published descriptions of "ERJ" exactly. This means no *new* Japanese-L1
corpus is needed for the base audio; what's missing is a phone-level
error-tag annotation layer analogous to L2-ARCTIC's, which the corpus we
already have doesn't include (only the holistic `lbl/` ratings used in
5b).

**A companion annotation resource exists but is a real risk, not a
solid lead.** Takehiko Makino and Rika Aoki (Chuo University / U. Tokyo)
built exactly the missing layer — the "ERJ Phonetic Corpus": Praat
TextGrids with manually-transcribed actual phones and a substitution
tier, added to Penn Phonetics Lab forced alignments of 1,902 of ERJ's
sentence files (Makino & Aoki, *Research in Language* 10.1, 2012,
DOI 10.2478/v10015-011-0046-5 — read directly via PDF, not just the
abstract, since the abstract alone doesn't carry the caveat that matters).
That caveat: **"fewer than 10% of the files have been completed and the
corpus-building is still in its initial stage"** as of that 2012 report —
under ~190 of 1,902 sentence files, transcribed by hand, one paper, no
later publication or dataset repository found in searches. Speaker
coverage of that subset isn't stated anywhere found; each ERJ sentence is
read by ~12 different speakers, so <190 files could span anywhere from a
handful of speakers to most of the 202 — unknowable without asking. Given
section 5d/5e's established bar (a 4-speaker/language cluster bootstrap in
L2-ARCTIC was already flagged as producing "necessarily wide" CIs), this
resource could easily be *worse* than what we already ruled insufficient,
not better. Worth one email to ask current status and speaker coverage
(draft below); not worth planning an analysis around before that answer
comes back.

**The cheap, on-disk check the advisor suggested surfaced a genuine third
hypothesis: posterior-deficit GOP behaves structurally differently on
UME-ERJ audio than on L2-ARCTIC audio, independent of L1 or rating
methodology.** Ran the *same* posterior-deficit phone model
(`align_words_gop`) over a matched sample from each corpus (150
utterances each; L2-ARCTIC: 4 speakers/4,835 phones; UME-ERJ: word-
segmental items/492 phones) and compared the raw GOP distribution,
with no correctness filtering (UME-ERJ has no per-phone labels to filter
by — this compares the *engine's own output distribution*, not accuracy):

| corpus | frac. phones exactly 0.0 | mean | median | frac. < -1 | frac. < -3 |
|---|---|---|---|---|---|
| L2-ARCTIC | 0.849 | -0.467 | 0.0 | 0.112 | 0.060 |
| UME-ERJ (word-segmental) | 0.555 | -1.747 | 0.0 | 0.384 | 0.258 |

L2-ARCTIC's distribution is the familiar one from 5c/5e/5f: heavily
saturated at the 0.0 ceiling (85% of phones), the exact peaky-CTC pattern
that causes the substitution blindness. **UME-ERJ's distribution is
markedly less saturated** — only 55% pinned at 0.0, a mean 3.7x more
negative, more than 3x the mass below -1 and -3. That's the phone model
producing far more graded, less-collapsed scores on UME-ERJ audio *before
any question of correctness enters* — exactly the direction that would
make posterior-deficit GOP more informative there than on L2-ARCTIC or
speechocean762, independent of anything about Japanese-L1 error patterns
or UME-ERJ's holistic rating scale. Candidate causes not distinguished
here: recording setup (UME-ERJ: Sennheiser HMD25-1 headset mic,
multi-university academic setup; L2-ARCTIC: Samson C03U desktop mic +
pop filter), sentence material (TIMIT-derived vs. CMU ARCTIC), or the
wav2vec2 phone-CTC model's domain response differing by corpus in a way
that happens to manifest as posterior peakiness. This is a checked
observation, not a fully closed explanation — but it's a concrete,
falsifiable third hypothesis the paper didn't have before, sitting
alongside (not necessarily replacing) L1-specificity.

**Two emails drafted for the researcher to send, not sent from here:**
one to Takehiko Makino asking current ERJ Phonetic Corpus completion
status and speaker coverage; one to Ricardo Gutierrez-Osuna / Guanlong
Zhao (L2-ARCTIC, the annotator-confound question from section 5e) — both
external-contact dependencies with unknown response time, appropriately
outside what this session can resolve on its own.

**Secondary lead, not pursued:** J-AESOP (Asian English Speech cOrpus
Project) includes Japanese speakers with corrected forced alignments and
holistic accentedness/comprehensibility ratings, but no evidence found of
phone-level error tags — would extend the L1 roster (Thai, Indonesian,
Korean, etc. alongside Japanese) under one shared methodology, which could
independently test "is it Japanese specifically" if it ever needs
revisiting, but doesn't obviously improve on what UME-ERJ already gives
for the *immediate* question and wasn't investigated further here.

---

### 5h. Does extending GOP-SF to the BPE model help? Checked, no — two honest negative results

Asked directly: would porting section 5f's segmentation-free fix to the
BPE model likely improve it, and if not, what would? Answer, checked
empirically rather than assumed either way: **full substitution-style
GOP-SF is not well-motivated for BPE, and the two cheaper alternatives
tried both failed to help.**

**Does BPE have the same disease first.** Before building anything,
checked whether BPE's word-level posterior-deficit GOP shows the same
substitution/deletion asymmetry the phone model had at phone level
(400 L2-ARCTIC utterances, words bucketed by whether their phone-level
tags were substitutions only, deletions only, both, or none):

| bucket | n | mean | median | frac exactly 0.0 |
|---|---|---|---|---|
| correct | 2,534 | -0.132 | 0.0 | 0.928 |
| substitution only | 1,159 | -0.320 | 0.0 | 0.810 |
| deletion only | 174 | -0.549 | 0.0 | 0.747 |
| both | 125 | -0.477 | 0.0 | 0.552 |

Yes — same qualitative pattern (median pinned at 0.0 for every bucket,
substitutions separate less from correct than deletions do: mean gap
0.19 vs. 0.42, frac-zero gap 0.12 vs. 0.18), just milder than the phone
model's version (there, substitution's gap was ~28% of deletion's; here
it's ~45% — plausibly because a BPE token spans several phones, so one
wrong phone inside it gets partly diluted rather than fully masking the
whole unit, unlike phone-level scoring where the unit *is* the error).

**Why full GOP-SF is still the wrong port, despite the disease being
present:** the fix works by marginalizing "any phone, or nothing" at a
position — a well-defined question because phones are the actual unit of
mispronunciation. "Any other BPE subword piece, or nothing" at a token
position isn't the same kind of question: BPE pieces are orthographic
chunks (`_segmentations` in `proscor/align.py` shows a single word can
tokenize several different ways with no pronunciation difference implied
at all), so marginalizing over arbitrary alternate pieces doesn't clearly
correspond to "was this mispronounced" the way phone substitution does —
and the vocabulary to marginalize over is far larger than the phone
model's ~392 symbols, so it would cost more for a weaker-motivated signal
on top of a milder underlying problem. Not implemented.

**Alternative 1, cheap and well-motivated on paper: a deletion-only term**
(`proscor.align.gop_deletion_term` / `align_words_gop_deletion`,
unit-tested in `tests/test_align.py`). No marginalization needed at all —
"does the audio fit better if this word's tokens weren't required at
all," reusing the already-tested `_ctc_loglik` with the word's tokens
removed. Well-defined regardless of subword granularity, and the bucket
table above shows deletions carry the real signal for BPE too, same as
for phones. **Tested on 600 L2-ARCTIC utterances (5,679 words) — it does
not help:**

| | vs. frac_correct | vs. any_error (inverted) |
|---|---|---|
| posterior-deficit alone | 0.146 | 0.112 |
| deletion-term alone | 0.078 | **-0.080** |
| sum, z-normalized | 0.137 | 0.020 |

The deletion term alone is weaker than the existing score, wrong-signed
against the binary label, and combining the two (even after normalizing
scale so one doesn't dominate) makes both worse, not better. The
mechanistic read: a mispronounced word usually still has roughly the
right *duration/shape* as a word-sized chunk (some internal phones are
off, but something clearly fills that slot) — "is this word's span
audibly absent altogether" is a cruder, different question than "was it
pronounced correctly," and L2-ARCTIC's per-word labels test the latter.
Kept in the codebase as correct, tested, useful infrastructure (the same
`_ctc_loglik`-reuse pattern section 5f's deletion term uses), not wired
into any shipped or default scoring path, and not claimed as an
improvement.

**Alternative 2: ensemble BPE with the phone model's word-level score**
(motivated by their partially complementary error profiles — that's the
whole reason the ranking flips between corpora). Free to test: both
scores already sit in `results/l2arctic_word.json.arrays.json` from
section 5e, no new inference needed. Best case found, an 85/15
BPE-weighted z-normalized blend: r=0.172 vs. frac_correct (BPE alone:
0.169 — a 0.003 gain, noise) and r=0.143 vs. any_error (BPE alone: 0.149
— a **loss**). Phone's word-level signal (r=0.067 / 0.008 alone,
barely above zero for the binary label) is too weak and noisy to add
value through a simple linear blend at any weighting tried. Not pursued
further.

**Bottom line:** the one improvement this whole investigation actually
found and validated is section 5f's GOP-SF applied to the *phone* model,
where the direct hypothesis test (substitution/correct phones separating
at the median, 0.0=0.0 -> -0.061 vs. -0.223) is unambiguous even though
the aggregate correlation gain is modest and not yet proven significant
(the paired test in progress will settle that). For BPE specifically, two
plausible, cheap, evidence-motivated fixes were tried and both failed —
a real finding (it rules out two ideas a reviewer might otherwise
suggest) even though it isn't the improvement that was asked for. No
further BPE-specific fix is proposed here without a better-motivated idea
than these two.

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

