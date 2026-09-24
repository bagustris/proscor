"""DTW-over-WavLM-embeddings pronunciation scoring: compare a learner's
speech to one or more native reference recordings of the *same text*
(McIntosh, Smit, Saito, Minematsu & Kamper, "Self-supervised Speech
Comparison for L2 Phone, Rhythm, and Intonation Scoring", arXiv:2607.13721)
-- a template-based, fully training-free alternative to posterior-deficit
GOP (`proscor/align.py`/`align_phone.py`). PLAN.md section 5l.

Unlike GOP, this needs no phone inventory, no canonical transcription
pipeline, and no acoustic model trained on phone labels -- just an
off-the-shelf SSL model (WavLM-Large) and at least one native recording
of the target text. Motivation: on the same underlying database family
(UME-ERJ/ERJ), the paper's DTW-SSL phone score (r=0.576, sentence-level
phonetic-accuracy Pearson) beats this project's own posterior-deficit
phone-model result (r=0.432, PLAN.md section 5b) -- see PLAN.md section
5k's research phase for how that gap was established.

**Method** (paper section III.A, "Phone scoring"): extract WavLM-Large's
final-layer frame representations for the learner utterance and for each
available native template, align them via DTW using frame-level cosine
distance, and take the cumulative alignment cost along the optimal path,
normalized by path length. The reported score averages this cost over
every available native template (the paper's main method). Lower cost =
closer to the native template(s) = better predicted pronunciation, so
this is a *distance*, not a *goodness* score -- flip its sign (or use a
negative correlation) when comparing against an accuracy/goodness label.

**"Final layer" is `last_hidden_state`**, WavLM's standard forward-pass
output (which implicitly includes the encoder's own trailing LayerNorm)
-- not `output_hidden_states=True`'s `hidden_states[-1]`, the
pre-LayerNorm hidden state one step earlier, which is numerically
different (confirmed on a smoke test: mean per-frame cosine similarity
0.97, not 1.0, between the two). The paper doesn't specify which
convention "final layer" means; `last_hidden_state` is what any plain
WavLM forward pass returns without extra introspection, so it's the
more defensible reading absent a code release.
"""
import numpy as np

MODEL_REPO = "microsoft/wavlm-large"
SAMPLE_RATE = 16000

_MODEL = None
_FEATURE_EXTRACTOR = None


def available() -> bool:
    """True if the optional DTW-SSL dependencies are installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model():
    global _MODEL, _FEATURE_EXTRACTOR
    if _MODEL is None:
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel

        _FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_REPO)
        _MODEL = WavLMModel.from_pretrained(MODEL_REPO)
        _MODEL.eval()
    return _MODEL, _FEATURE_EXTRACTOR


def embed(samples: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """float32 mono waveform -> (T, 1024) WavLM-Large final-layer frame
    embeddings, L2-normalized per frame (so `dtw_cost`'s cosine-distance
    matrix is a plain dot product). Uses the model's own feature
    extractor for input normalization (`do_normalize=True` for
    wavlm-large -- verified, not assumed; matches wav2vec2's convention
    despite WavLM's different pretraining objective)."""
    import torch

    model, fe = _get_model()
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if sr != SAMPLE_RATE:
        from audiokit import resample

        samples = np.asarray(resample(samples, sr, SAMPLE_RATE), dtype=np.float32)
    inputs = fe(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    emb = out.last_hidden_state[0].numpy()
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    return (emb / np.clip(norms, 1e-8, None)).astype(np.float32)


def dtw_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Standard DTW over two L2-normalized frame sequences `a` (T1,D),
    `b` (T2,D): the per-cell distance matrix is `1 - cosine_similarity`
    (a plain dot product, since rows are already unit norm), aligned by
    the usual 3-step dynamic program (stay/step-down/step-right, every
    frame of both sequences visited at least once -- no skipping). The
    returned score is the cumulative cost along the optimal path,
    normalized by the path's actual length in steps (via backtracking),
    matching the paper's "normalized by path length" -- not `T1+T2`,
    which overcounts whenever the path takes more diagonal steps than
    the longer sequence's own length."""
    T1, T2 = a.shape[0], b.shape[0]
    dist = 1.0 - a @ b.T  # (T1, T2) cosine distance

    acc = np.full((T1, T2), np.inf, dtype=np.float64)
    acc[0, 0] = dist[0, 0]
    for i in range(1, T1):
        acc[i, 0] = acc[i - 1, 0] + dist[i, 0]
    for j in range(1, T2):
        acc[0, j] = acc[0, j - 1] + dist[0, j]
    for i in range(1, T1):
        row_prev = acc[i - 1]
        row = acc[i]
        d_row = dist[i]
        for j in range(1, T2):
            row[j] = d_row[j] + min(row_prev[j], row[j - 1], row_prev[j - 1])

    i, j = T1 - 1, T2 - 1
    length = 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = ((acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1),
                          (acc[i - 1, j - 1], i - 1, j - 1))
            _, i, j = min(candidates, key=lambda c: c[0])
        length += 1

    return float(acc[T1 - 1, T2 - 1] / length)


def embed_wav_cached(wav_path, cache_dir) -> np.ndarray:
    """`embed()` for a wav file on disk, cached to `cache_dir` as one .npy
    per file (keyed by the wav's path relative to nothing in particular --
    just its basename plus a short hash of the full path, to keep
    filenames short while staying collision-safe across corpora that
    reuse basenames). Native reference templates are reused across every
    learner utterance that targets the same text (and across repeated
    eval runs), so this turns ~11 templates x 625 texts (~7k WavLM-Large
    forward passes) into a one-time cost -- PLAN.md section 5l."""
    import hashlib

    from pathlib import Path
    import soundfile as sf

    wav_path = Path(wav_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(str(wav_path).encode()).hexdigest()[:10]
    cache_path = cache_dir / f"{wav_path.stem}_{h}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    samples, sr = sf.read(str(wav_path), dtype="float32")
    emb = embed(samples, sr=sr)
    np.save(cache_path, emb)
    return emb


def score(learner_emb: np.ndarray, template_embs: list) -> dict:
    """DTW cost between `learner_emb` and each of `template_embs`
    (pre-embedded native references of the same text) ->
    `{"mean", "min", "n_templates", "costs"}`. `mean` (average cost over
    every available template) is the paper's main aggregation; `min` is
    kept for the mean-vs-min ablation this project ran on the analogous
    ZIPA multi-token question (PLAN.md section 5k) and repeats here for
    consistency (section 5l)."""
    costs = [dtw_cost(learner_emb, t) for t in template_embs]
    return {"mean": float(np.mean(costs)), "min": float(np.min(costs)),
            "n_templates": len(costs), "costs": costs}
