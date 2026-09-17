#!/usr/bin/env python
"""Paired BPE-vs-phone-model comparison on speechocean762, word level --
the significance test the headline PLAN.md section 5a numbers (BPE r=0.471
vs. phone r=0.325, computed by two separate scripts on two separate runs)
never had. Comparing two independently-bootstrapped marginal CIs is a weak
test when both engines score the same audio (their errors are correlated),
so this runs both engines in the SAME loop over the SAME utterances and
computes a paired, speaker-cluster bootstrap CI on the difference
r_bpe - r_phone (proscor/stats.py).

Only words both engines managed to align are compared (a small subset of
each engine's own align-rate denominator; see n_words_both in the output).

Usage:
    python scripts/eval_so762_paired.py                # full test split
    python scripts/eval_so762_paired.py --limit 200    # quick smoke run
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

from proscor import align, align_phone, stats as gopstats


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "mispeech/speechocean762", f"data/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    return pq.read_table(path).to_pylist()


def run(rows: list, progress_every: int = 200) -> dict:
    bpe_gop, phone_gop, word_acc, word_speaker = [], [], [], []
    utt_bpe_mean, utt_phone_mean, utt_acc, utt_total, utt_speaker = [], [], [], [], []
    n_words_total = n_words_both = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        speaker = row["speaker"]
        n_words_total += len(words)
        try:
            bpe_result = align.align_words_gop(samples, words, sr=sr)
        except Exception as e:
            print(f"  [warn] utt {i} bpe failed: {e}", file=sys.stderr)
            bpe_result = [None] * len(words)
        try:
            phone_result = align_phone.align_words_gop(samples, words, sr=sr)["word_gop"]
        except Exception as e:
            print(f"  [warn] utt {i} phone failed: {e}", file=sys.stderr)
            phone_result = [None] * len(words)

        this_bpe, this_phone = [], []
        for w, br, pr in zip(row["words"], bpe_result, phone_result):
            if br is None or pr is None:
                continue
            n_words_both += 1
            bpe_gop.append(br["gop"])
            phone_gop.append(pr["gop"])
            word_acc.append(w["accuracy"])
            word_speaker.append(speaker)
            this_bpe.append(br["gop"])
            this_phone.append(pr["gop"])

        if this_bpe:
            utt_bpe_mean.append(float(np.mean(this_bpe)))
            utt_phone_mean.append(float(np.mean(this_phone)))
            utt_acc.append(row["accuracy"])
            utt_total.append(row["total"])
            utt_speaker.append(speaker)

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "bpe_gop": bpe_gop, "phone_gop": phone_gop, "word_acc": word_acc, "word_speaker": word_speaker,
        "utt_bpe_mean": utt_bpe_mean, "utt_phone_mean": utt_phone_mean,
        "utt_acc": utt_acc, "utt_total": utt_total, "utt_speaker": utt_speaker,
        "n_utterances": len(rows), "n_words_total": n_words_total, "n_words_both": n_words_both,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align.available() or not align_phone.available():
        print("proscor.align and proscor.align_phone optional deps must both be installed.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading speechocean762 [{args.split}] ...", file=sys.stderr)
    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Evaluating {len(rows)} utterances (both engines) ...", file=sys.stderr)

    results = run(rows)

    word_diff = gopstats.cluster_bootstrap_paired_diff(
        results["bpe_gop"], results["phone_gop"], results["word_acc"], results["word_speaker"]
    ) if results["bpe_gop"] else None
    utt_diff = gopstats.cluster_bootstrap_paired_diff(
        results["utt_bpe_mean"], results["utt_phone_mean"], results["utt_acc"], results["utt_speaker"]
    ) if results["utt_bpe_mean"] else None

    summary = {
        "split": args.split,
        "n_utterances": results["n_utterances"],
        "n_words_total": results["n_words_total"],
        "n_words_both": results["n_words_both"],
        "elapsed_s": round(results["elapsed_s"], 1),
        "word_level_bpe_vs_phone_paired_diff": word_diff,
        "utterance_level_bpe_vs_phone_paired_diff": utt_diff,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
