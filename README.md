# proscor

English pronunciation scoring, offline and CPU-only. Show a prompt, read it aloud,
get a 0-100 score with word-by-word phoneme feedback — as a CLI or a web app.

```
proscor> Prompt #3: "She sells sea shells."
proscor> (p)lay reference, then press ENTER to record...
[recording...]
Score: 84/100
  OK   she
  OK   sells
  MISS sea    -> heard "see"  [expect S IY | heard S IY]
  OK   shells
```

## How it works

```
Prompt  ->  G2P  ->  Record  ->  ASR  ->  Score + feedback
```

1. **Prompt** — a word or sentence is shown (`data/prompts.txt`).
2. **G2P** — the target text is converted to expected phonemes (`g2p_en` + CMUdict).
3. **Record** — the microphone (CLI) or browser (web) captures the reading.
4. **ASR** — the audio is transcribed offline via [sherox](https://github.com/bagustris/sherox)
   (sherpa-onnx, NeMo CTC Conformer, CPU).
5. **Score** — recognized words are aligned to target words and compared
   phoneme-by-phoneme (edit distance), producing a 0-100 score per word and overall.
6. **Feedback** — mismatches are reported with the expected vs. heard phonemes
   (e.g. `TH -> T`) and a plain-English hint (`TH as in think`).

> [!NOTE]
> This scores **intelligibility** — whether the ASR model understood the intended
> word — not native-likeness/accent. It's a strong, cheap proxy for most learners.
> True accent scoring (Goodness-of-Pronunciation) is documented as an optional,
> not-yet-implemented track in [`PLAN.md`](PLAN.md#5-optional-advanced-track---true-pronunciation-scoring-gop).

A synthetic reference voice (Piper TTS via sherox) can also read the prompt aloud
first, so learners know what they're aiming for.

## Requirements

- Python 3.11+, CPU only (no GPU/CUDA needed).
- Two sibling repos checked out next to this one, providing ASR + TTS:
  [`sherox`](https://github.com/bagustris/sherox) and
  [`audiokit`](https://github.com/bagustris/audiokit).

```
github/
  proscor/
  sherox/
  audiokit/
```

## Setup

Using [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv && source .venv/bin/activate

# install the sibling repos editable, then this project's own deps
uv pip install -e ../audiokit -e ../sherox
uv pip install -r requirements.txt

python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger_eng')"
```

(Plain `python -m venv` + `pip install` works the same way if you'd rather not use uv.)

> [!TIP]
> Don't have the sibling repos? Install both from git as a fallback:
> `pip install git+https://github.com/bagustris/sherox` plus
> `pip install git+https://github.com/bagustris/audiokit`.

> [!TIP]
> On a machine without PyPI access, build the venv with
> `uv venv --system-site-packages .venv` so it can see any of these packages
> already installed system-wide, and install the sibling repos with
> `uv pip install --no-deps --no-build-isolation -e ../audiokit -e ../sherox`
> to skip re-resolving their dependencies from the index.

The ASR model (NeMo CTC English, ~158 MB), a Silero VAD, and the Piper TTS voice
all **auto-download into `models/` on first use** — there's no manual download step.

## Usage

### CLI

```bash
python cli.py
```

Reads a prompt, lets you play a reference pronunciation, records your attempt, and
scores it. Useful flags:

```bash
python cli.py --seconds 4 --prompt-file data/prompts.txt
python cli.py --tts-lang eng      # spoken reference before recording
python cli.py --no-tts            # skip the reference voice
```

In-loop keys: `p` play reference, `r` retry, `n` next prompt, `q` quit.

### Web app

```bash
uvicorn web.server:app --reload --port 8000
```

Open <http://localhost:8000>: play the reference audio, record with the browser,
and see the score plus a per-word breakdown.

## Project layout

```
proscor/
  config.py     # paths, ASR model choice, scoring weights
  g2p.py        # text -> expected phonemes (g2p_en, + data/lexicon.txt overrides)
  tts.py        # reference pronunciation audio (sherox)
  audio.py      # record / load WAV, always mono 16 kHz
  asr.py        # transcribe recorded audio (sherox)
  score.py      # align + phoneme edit distance -> 0-100 score
  feedback.py   # human-readable score report
  prompts.py    # prompt list loading + selection
cli.py          # interactive CLI
web/            # FastAPI app + static frontend
data/           # prompts.txt, optional lexicon.txt overrides
scripts/        # selftest.py, model download fallback
tests/
```

## Testing

```bash
python -m pytest -q          # unit tests, fully offline (fake ASR results, no models)
python scripts/selftest.py   # end-to-end: synthesizes clean + degraded audio,
                              # checks scores land in the expected ranges
```

> [!NOTE]
> Use `python -m pytest`, not a bare `pytest`, so the venv's interpreter (and
> `proscor`'s cwd-based imports) are used regardless of what else is on `PATH`.

## Configuration

Scoring weights, the default ASR model, and paths all live in
[`proscor/config.py`](proscor/config.py). Custom pronunciations for specific words
can be added to `data/lexicon.txt` (`WORD<TAB>P AH L`), which take priority over the
G2P model's output.

## Scope

English only for now. `PLAN.md` outlines a path to add Indonesian and Arabic
without a rewrite — see [section 9](PLAN.md#9-future-todo-multi-language-indonesian-arabic).

## Limitations

- Reference audio is a synthetic TTS voice, not a native speaker recording.
- Background noise hurts ASR accuracy; a quiet room and a single speaker work best.
- Word stress is ignored by default (`--include-stress` to keep it).
- Recording in the CLI requires PortAudio (`apt install portaudio19-dev` on Linux,
  `brew install portaudio` on macOS).

See [`PLAN.md`](PLAN.md) for the full design rationale and build history.
