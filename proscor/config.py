"""Central config: paths, model selection, sample rate, scoring weights."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

SAMPLE_RATE = 16000

# ASR (sherox / sherpa-onnx)
ASR_MODEL_DIR = "models/sherpa-onnx-nemo-ctc-en-conformer-medium"
ASR_MODEL_TYPE = "nemo_ctc"
ASR_NUM_THREADS = 1

# TTS (sherox, reference pronunciation audio)
TTS_LANG = "eng"

# Scoring weights (Step 5): score = PHONEME_WEIGHT*phoneme_score + CONF_WEIGHT*conf*100
PHONEME_WEIGHT = 0.8
CONF_WEIGHT = 0.2

# GOP blend (optional advanced track, section 5): score = INTELLIGIBILITY_WEIGHT*intelligibility + GOP_WEIGHT*gop
INTELLIGIBILITY_WEIGHT = 0.6
GOP_WEIGHT = 0.4

DEFAULT_LEXICON_PATH = DATA_DIR / "lexicon.txt"
DEFAULT_PROMPTS_PATH = DATA_DIR / "prompts.txt"
