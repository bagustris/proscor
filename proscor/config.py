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

# Quantized (model.int8.onnx) vs full-precision (model.onnx) weights for the
# CTC forced-alignment path (proscor/align.py). int8 is faster and is what
# the CLI/web app uses by default; quantization noise measurably lowers GOP
# posterior quality (see scripts/eval_so762.py / PLAN.md section 5a), so
# evaluation runs should override with use_int8=False.
ALIGN_USE_INT8 = True

# Single-word forced-alignment scoring (proscor/align.py):
# score = 100 * (ALIGN_POSTERIOR_WEIGHT * sigmoid(margin + ALIGN_TARGET_PRIOR)
#                + ALIGN_GOP_WEIGHT * exp(gop / ALIGN_GOP_SCALE))
# where margin = loglik(target) - loglik(best confusable).
ALIGN_POSTERIOR_WEIGHT = 0.6
ALIGN_GOP_WEIGHT = 0.4
ALIGN_GOP_SCALE = 2.0
ALIGN_TARGET_PRIOR = 1.0   # benefit of the doubt: the learner is trying to say the target
ALIGN_MIN_FIT = 0.35       # exp(gop/scale) below this vetoes "correct" (unrelated word said)
ALIGN_MAX_CONFUSABLES = 40

# --engine gop-lite (proscor.score.score_gop_lite, PLAN.md section 5a): a
# word's GOP-lite score (100 * align.gop_to_fit(gop)) at or above this counts
# as "correct" for the pass/fail line in feedback.format_report. There's no
# principled threshold here (unlike the intelligibility engine's exact-match
# "correct") -- from scripts/eval_so762.py's full test-split records (int8,
# n=15,967 words): at 60, 81.2% of human-accuracy-10 words score >= 60
# ("correct"), and 73.8% of human-accuracy-<=5 words score < 60 ("low fit").
# Neither is near 100%: GOP-lite is a noisy proxy (word-level Pearson r =
# 0.47 against human accuracy), not a clean classifier -- the continuous
# word_score is the more informative signal, this threshold is only for the
# terse feedback.format_report pass/fail line. Retune if the score
# distribution shifts (re-derive from results/so762_test_int8.json).
GOP_LITE_CORRECT_THRESHOLD = 60.0

DEFAULT_LEXICON_PATH = DATA_DIR / "lexicon.txt"
DEFAULT_PROMPTS_PATH = DATA_DIR / "prompts.txt"
