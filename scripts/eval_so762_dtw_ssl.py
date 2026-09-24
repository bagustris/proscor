#!/usr/bin/env python
"""DTW-SSL vs. xlsr-53 GOP-lite on speechocean762 -- PLAN.md section 5l's
third corpus, with a real methodological difference from the other two
flagged up front, not buried: **speechocean762 has no native-reference
recordings to draw on.** Unlike L2-ARCTIC (CMU ARCTIC's own native
readers, bdl/slt, share its exact prompt set) and UME-ERJ (20 American
reference speakers reading the same prompts under a mapped filename
convention), speechocean762's 2,500 test-split prompts are almost all
unique (2,499/2,500 -- essentially no repeats to exploit either), and
correspond to no other corpus this project has native audio for.

**Adaptation: synthetic references via `sherox`'s 8-speaker Kitten TTS
voice (`lang="eng-kitten"`).** This keeps the method training-free (an
off-the-shelf TTS model, not fine-tuned on anything pronunciation-
related) and gives genuine multi-template averaging (8 distinct
synthetic voices, verified non-identical -- different durations/RMS per
speaker_id on a smoke test), but it is a real departure from the paper's
own design (real native speech, not synthetic) and from what this
project validated on L2-ARCTIC/UME-ERJ. Treat this corpus's numbers as a
"TTS-reference variant" of DTW-SSL, not a third replication of the exact
method -- if it underperforms the other two corpora, synthesis quality
(Kitten TTS Nano is int8-quantized, optimized for speed/size not
naturalness) is a live confound that real-native-template results don't
have to account for.

No cross-item template caching is possible here (unlike the other two
scripts' `embed_wav_cached`) since there's essentially nothing to share
across items -- each utterance's 8 templates are synthesized and embedded
fresh. This makes the per-item cost real (~12s: 8 TTS syntheses + 9
WavLM embeddings + xlsr-53 GOP), budget the run size accordingly.

Usage:
    python scripts/eval_so762_dtw_ssl.py --out results/so762_dtw_ssl.json
    python scripts/eval_so762_dtw_ssl.py --limit 20
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proscor import align_phone, dtw_ssl, stats as gopstats, tts

N_TEMPLATES = 8
TTS_LANG = "eng-kitten"


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "mispeech/speechocean762", f"data/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    return pq.read_table(path).to_pylist()


def synthesize_templates(text: str) -> list:
    """8 distinct Kitten-TTS voices reading `text` -> [(T,1024) WavLM
    embedding, ...]. Not cached (see module docstring: speechocean762's
    prompts are essentially all unique, nothing to reuse)."""
    embs = []
    for speaker_id in range(N_TEMPLATES):
        samples, sr = tts.synthesize(text, lang=TTS_LANG, speaker_id=speaker_id)
        embs.append(dtw_ssl.embed(samples, sr=sr))
    return embs


def run(rows: list, progress_every: int = 20) -> dict:
    out = {k: [] for k in ("dtw_mean", "dtw_min", "xlsr_post", "xlsr_sf", "human", "speaker")}
    n_failed = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]

        try:
            learner_emb = dtw_ssl.embed(samples, sr=sr)
            template_embs = synthesize_templates(row["text"])
            dtw = dtw_ssl.score(learner_emb, template_embs)

            xlsr_post = align_phone.align_words_gop(samples, words, sr=sr,
                                                      model_id=align_phone.TORCH_MODEL_REPO)
            xlsr_sf = align_phone.align_words_gop_sf(samples, words, sr=sr,
                                                       model_id=align_phone.TORCH_MODEL_REPO)
            post_vals = [r["gop"] for r in xlsr_post["word_gop"] if r is not None]
            sf_vals = [r["gop"] for r in xlsr_sf["word_gop"] if r is not None]
            if not post_vals or not sf_vals:
                n_failed += 1
                continue
        except Exception as e:
            print(f"  [warn] utt {i} failed: {e}", file=sys.stderr)
            n_failed += 1
            continue

        out["dtw_mean"].append(-dtw["mean"])
        out["dtw_min"].append(-dtw["min"])
        out["xlsr_post"].append(float(np.mean(post_vals)))
        out["xlsr_sf"].append(float(np.mean(sf_vals)))
        out["human"].append(row["accuracy"])
        out["speaker"].append(row["speaker"])

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1):.1f}s/utt)", file=sys.stderr)

    return {**out, "n_utterances_total": len(rows), "n_failed": n_failed, "elapsed_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align_phone.available() or not dtw_ssl.available():
        print("proscor.align_phone and proscor.dtw_ssl optional deps must both be installed.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading speechocean762 [{args.split}] ...", file=sys.stderr)
    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Evaluating {len(rows)} utterances (DTW-SSL via {N_TEMPLATES}-voice TTS reference + xlsr-53) ...",
          file=sys.stderr)

    r = run(rows)

    def corr(key):
        return gopstats.cluster_bootstrap_pearson(r[key], r["human"], r["speaker"])

    def diff(k1, k2):
        return gopstats.cluster_bootstrap_paired_diff(r[k1], r[k2], r["human"], r["speaker"])

    summary = {
        "n_utterances_total": r["n_utterances_total"],
        "n_utterances_scored": len(r["human"]),
        "n_failed": r["n_failed"],
        "elapsed_s": round(r["elapsed_s"], 1),
        "note": "TTS-reference variant (see module docstring) -- not a like-for-like replication "
                "of the real-native-template method validated on L2-ARCTIC/UME-ERJ. "
                "dtw scores are -dtw_cost (higher = better). "
                "diff = r(dtw,human) - r(xlsr,human); positive+significant means DTW-SSL wins",
        "dtw_mean_vs_human": corr("dtw_mean"),
        "dtw_min_vs_human": corr("dtw_min"),
        "xlsr_post_vs_human": corr("xlsr_post"),
        "xlsr_sf_vs_human": corr("xlsr_sf"),
        "dtw_mean_vs_xlsr_post_paired_diff": diff("dtw_mean", "xlsr_post"),
        "dtw_mean_vs_xlsr_sf_paired_diff": diff("dtw_mean", "xlsr_sf"),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        with open(args.out + ".arrays.json", "w") as f:
            json.dump(r, f)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
