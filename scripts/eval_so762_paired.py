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
    bpe_gop, phone_gop, word_acc, word_speaker, word_age = [], [], [], [], []
    utt_bpe_mean, utt_phone_mean, utt_acc, utt_total, utt_speaker, utt_age = [], [], [], [], [], []
    n_words_total = n_words_both = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        speaker = row["speaker"]
        age = row["age"]
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
            word_age.append(age)
            this_bpe.append(br["gop"])
            this_phone.append(pr["gop"])

        if this_bpe:
            utt_bpe_mean.append(float(np.mean(this_bpe)))
            utt_phone_mean.append(float(np.mean(this_phone)))
            utt_acc.append(row["accuracy"])
            utt_total.append(row["total"])
            utt_speaker.append(speaker)
            utt_age.append(age)

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "bpe_gop": bpe_gop, "phone_gop": phone_gop, "word_acc": word_acc,
        "word_speaker": word_speaker, "word_age": word_age,
        "utt_bpe_mean": utt_bpe_mean, "utt_phone_mean": utt_phone_mean,
        "utt_acc": utt_acc, "utt_total": utt_total, "utt_speaker": utt_speaker, "utt_age": utt_age,
        "n_utterances": len(rows), "n_words_total": n_words_total, "n_words_both": n_words_both,
        "elapsed_s": time.time() - t0,
    }


def _subset(results: dict, prefix: str, keep) -> tuple:
    """Filter the per-item lists for one level (word/utt) by a predicate on
    age; returns (x1, x2, y, clusters) ready for the paired bootstrap."""
    if prefix == "word":
        keys = ("bpe_gop", "phone_gop", "word_acc", "word_speaker", "word_age")
    else:
        keys = ("utt_bpe_mean", "utt_phone_mean", "utt_acc", "utt_speaker", "utt_age")
    cols = [results[k] for k in keys]
    picked = [(a, b, c, d) for a, b, c, d, age in zip(*cols) if keep(age)]
    if not picked:
        return None
    return tuple(zip(*picked))


def group_summary(results: dict, keep) -> dict:
    """Paired BPE-vs-phone comparison (plus each engine's own speaker-cluster
    CI) restricted to items whose speaker age satisfies `keep`."""
    out = {}
    for level in ("word", "utt"):
        sub = _subset(results, level, keep)
        if sub is None:
            out[level] = None
            continue
        x1, x2, y, clusters = sub
        out[level] = {
            "n_items": len(y),
            "n_speakers": len(set(clusters)),
            "bpe_speaker_cluster_ci": gopstats.cluster_bootstrap_pearson(x1, y, clusters),
            "phone_speaker_cluster_ci": gopstats.cluster_bootstrap_pearson(x2, y, clusters),
            "bpe_vs_phone_paired_diff": gopstats.cluster_bootstrap_paired_diff(x1, x2, y, clusters),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--age-threshold", type=int, default=18,
                     help="speakers with age < threshold are the 'younger' group (default 18; "
                          "speechocean762 has no speakers aged 16-18, so 18 splits cleanly)")
    ap.add_argument("--out", default=None,
                     help="summary JSON; per-item arrays (gop, label, speaker, age) are also "
                          "written next to it as <out>.arrays.json so age splits can be "
                          "re-analyzed without re-running inference")
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
    thr = args.age_threshold

    # The age split is the cheapest available test of the section 5b/5c
    # "age vs. L1" question: same corpus, same L1, same annotators, same
    # rating scheme -- only speaker age varies (PLAN.md section 5e).
    summary = {
        "split": args.split,
        "n_utterances": results["n_utterances"],
        "n_words_total": results["n_words_total"],
        "n_words_both": results["n_words_both"],
        "age_threshold": thr,
        "elapsed_s": round(results["elapsed_s"], 1),
        "all": group_summary(results, lambda a: True),
        f"age_under_{thr}": group_summary(results, lambda a: a < thr),
        f"age_{thr}_plus": group_summary(results, lambda a: a >= thr),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        arrays_path = args.out + ".arrays.json"
        with open(arrays_path, "w") as f:
            json.dump({k: results[k] for k in (
                "bpe_gop", "phone_gop", "word_acc", "word_speaker", "word_age",
                "utt_bpe_mean", "utt_phone_mean", "utt_acc", "utt_total", "utt_speaker", "utt_age",
            )}, f)
        print(f"Summary written to {args.out}; per-item arrays to {arrays_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
