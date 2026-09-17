#!/usr/bin/env python
"""Validate GOP-lite (proscor/align.py, PLAN.md section 5) against
speechocean762's human pronunciation-accuracy labels.

Stage 1 of the GOP-lite validation: the wired ASR model
(sherpa-onnx-nemo-ctc-en-conformer-medium) is a BPE-subword CTC model, not a
phoneme model (see PLAN.md section 5 and proscor/align.py's docstring), so it
cannot produce per-*phone* posteriors. This script validates at the
per-*word* level instead, using speechocean762's own canonical word text as
the forced-alignment target (never proscor's G2P, so G2P errors don't
contaminate the eval) and correlating `align.align_words_gop`'s GOP score
against the dataset's human word `accuracy` label (0-10). A phone-level model
is a follow-up (PLAN.md section 5 lists it as the literature-comparable
headline number).

Dataset: https://huggingface.co/datasets/mispeech/speechocean762 (a parquet
mirror of the OpenSLR speechocean762 corpus; downloaded via huggingface_hub
and cached in ~/.cache/huggingface, not stored in this repo).

Usage:
    pip install -r requirements-eval.txt           # huggingface_hub, pyarrow, scipy
    python scripts/eval_so762.py                   # full test split (~2.5k utterances)
    python scripts/eval_so762.py --limit 100       # quick smoke run
    python scripts/eval_so762.py --split train --out /tmp/so762_train.json
"""
import argparse
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proscor import align, stats as gopstats


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "mispeech/speechocean762", f"data/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    return pq.read_table(path).to_pylist()


def run(rows: list, model_dir: str = None, use_int8: bool = None, progress_every: int = 200) -> dict:
    word_gop, word_acc, word_text, word_speaker = [], [], [], []
    utt_gop_mean, utt_acc, utt_total, utt_speaker = [], [], [], []
    n_words_total = 0
    n_words_aligned = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        n_words_total += len(words)
        speaker = row["speaker"]
        try:
            results = align.align_words_gop(samples, words, sr=sr, model_dir=model_dir, use_int8=use_int8)
        except Exception as e:  # keep a bad utterance from killing a multi-minute run
            print(f"  [warn] utt {i} ({row['text']!r}) failed: {e}", file=sys.stderr)
            continue

        this_utt_gops = []
        for w, r in zip(row["words"], results):
            if r is None:
                continue
            n_words_aligned += 1
            word_gop.append(r["gop"])
            word_acc.append(w["accuracy"])
            word_text.append(w["text"])
            word_speaker.append(speaker)
            this_utt_gops.append(r["gop"])

        if this_utt_gops:
            utt_gop_mean.append(float(np.mean(this_utt_gops)))
            utt_acc.append(row["accuracy"])
            utt_total.append(row["total"])
            utt_speaker.append(speaker)

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "word_gop": word_gop, "word_acc": word_acc, "word_text": word_text, "word_speaker": word_speaker,
        "utt_gop_mean": utt_gop_mean, "utt_acc": utt_acc, "utt_total": utt_total, "utt_speaker": utt_speaker,
        "n_utterances": len(rows), "n_words_total": n_words_total,
        "n_words_aligned": n_words_aligned, "elapsed_s": time.time() - t0,
    }


def correlations(x: list, y: list) -> dict:
    from scipy.stats import pearsonr, spearmanr

    if len(x) < 2:
        return {"pearson_r": None, "spearman_rho": None, "n": len(x)}
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    return {"pearson_r": round(float(pr), 4), "spearman_rho": round(float(sr), 4), "n": len(x)}


def bucket_by_accuracy(gop: list, acc: list) -> dict:
    buckets = defaultdict(list)
    for g, a in zip(gop, acc):
        buckets[a].append(g)
    return {a: {"n": len(v), "mean_gop": round(float(np.mean(v)), 3),
                "std_gop": round(float(np.std(v)), 3),
                "frac_gop_zero": round(sum(1 for x in v if x == 0.0) / len(v), 3)}
            for a, v in sorted(buckets.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N utterances")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--fp32", action="store_true",
                     help="use model.onnx (full precision) instead of the default model.int8.onnx; "
                          "quantization noise measurably lowers GOP posterior quality")
    ap.add_argument("--out", default=None, help="write full results (incl. per-word records) as JSON")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed when --limit is set (0 = no shuffle, take the first N)")
    args = ap.parse_args()

    if not align.available():
        print("proscor.align optional deps (onnxruntime, kaldi_native_fbank) are not installed.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading speechocean762 [{args.split}] ...", file=sys.stderr)
    rows = load_split(args.split)
    if args.limit:
        if args.seed:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(rows), size=min(args.limit, len(rows)), replace=False)
            rows = [rows[i] for i in idx]
        else:
            rows = rows[: args.limit]
    print(f"Evaluating {len(rows)} utterances ...", file=sys.stderr)

    results = run(rows, model_dir=args.model_dir, use_int8=not args.fp32)

    word_corr = correlations(results["word_gop"], results["word_acc"])
    utt_corr_acc = correlations(results["utt_gop_mean"], results["utt_acc"])
    utt_corr_total = correlations(results["utt_gop_mean"], results["utt_total"])
    buckets = bucket_by_accuracy(results["word_gop"], results["word_acc"])

    # Speaker-cluster bootstrap CIs (words cluster within a speaker) --
    # see PLAN.md section 5c and proscor/stats.py.
    word_corr_ci = gopstats.cluster_bootstrap_pearson(
        results["word_gop"], results["word_acc"], results["word_speaker"]) if results["word_gop"] else None
    utt_corr_acc_ci = gopstats.cluster_bootstrap_pearson(
        results["utt_gop_mean"], results["utt_acc"], results["utt_speaker"]) if results["utt_gop_mean"] else None

    summary = {
        "split": args.split,
        "precision": "fp32" if args.fp32 else "int8",
        "n_utterances": results["n_utterances"],
        "n_words_total": results["n_words_total"],
        "n_words_aligned": results["n_words_aligned"],
        "align_rate": round(results["n_words_aligned"] / max(1, results["n_words_total"]), 4),
        "elapsed_s": round(results["elapsed_s"], 1),
        "word_level_vs_word_accuracy": word_corr,
        "word_level_vs_word_accuracy_speaker_cluster_ci": word_corr_ci,
        "utterance_level_vs_accuracy": utt_corr_acc,
        "utterance_level_vs_accuracy_speaker_cluster_ci": utt_corr_acc_ci,
        "utterance_level_vs_total": utt_corr_total,
        "mean_gop_by_word_accuracy": buckets,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({**summary, "records": [
                {"text": t, "gop": g, "accuracy": a}
                for t, g, a in zip(results["word_text"], results["word_gop"], results["word_acc"])
            ]}, f, indent=2)
        print(f"Full records written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
