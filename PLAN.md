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
