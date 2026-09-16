#!/usr/bin/env python
"""Validate phone-level GOP-lite (proscor/align_phone.py, PLAN.md section 5a
item 4) against speechocean762's human pronunciation-accuracy labels.

This is the phone-granularity counterpart to scripts/eval_so762.py (which
validates the BPE-subword NeMo model at word granularity, since that model
has no phone-level output). align_phone.py uses a genuine phoneme-CTC model
(wav2vec2-lv-60-espeak-cv-ft, ONNX) and phonemizes each canonical word live
with espeak-ng (the same tool that produced the model's training labels) to
get the target phone sequence.

Three comparisons are reported:
1. **Word-level** (comparable to scripts/eval_so762.py's BPE-model 0.471 and
   to GOPT's 0.533, Table 1 of arXiv:2205.03432): mean phone GOP per word
   vs. the dataset's per-word `accuracy` (0-10). Finer-grained than the BPE
   version's word aggregation, but empirically *lower* correlation than it
   -- see PLAN.md section 5a item 4 for the full comparison and discussion.
2. **Utterance-level**: mean word GOP vs. sentence `accuracy`/`total`.
3. **Direct per-phone** (the literature-comparable headline number, but
   coverage-limited): espeak's phones for a word don't always line up 1:1
   with the dataset's own ARPABET phone segmentation (e.g. espeak merges
   vowel+R into one rhotic token: "ɑːɹ" for the vowel in "mark" vs. the
   dataset's separate AA/R) -- see proscor/align_phone.py's docstring. This
   comparison only uses words where the phone COUNTS happen to match,
   zipping the two sequences by position without checking phone identity.
   The excluded words are not a random sample (disproportionately
   vowel+R words), so this is an *approximation with an unknown bias
   direction*, not a lower or upper bound -- not the rigorous phone-aligned
   metric a DTW reconciliation would give (a documented follow-up, see
   PLAN.md section 5a).

Usage:
    pip install -r requirements-eval.txt   # + phonemizer (needs system espeak-ng)
    python scripts/eval_so762_phone.py                # full test split
    python scripts/eval_so762_phone.py --limit 100    # quick smoke run
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

from proscor import align_phone


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "mispeech/speechocean762", f"data/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    return pq.read_table(path).to_pylist()


def run(rows: list, use_int8: bool = True, progress_every: int = 200) -> dict:
    word_gop, word_acc = [], []
    utt_gop_mean, utt_acc, utt_total = [], [], []
    phone_gop_matched, phone_acc_matched = [], []
    n_words_total = n_words_aligned = 0
    n_words_len_matched = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        n_words_total += len(words)
        try:
            result = align_phone.align_words_gop(samples, words, sr=sr, use_int8=use_int8)
        except Exception as e:
            print(f"  [warn] utt {i} ({row['text']!r}) failed: {e}", file=sys.stderr)
            continue

        this_utt_gops = []
        for w, wg, pg in zip(row["words"], result["word_gop"], result["phone_gop"]):
            if wg is None:
                continue
            n_words_aligned += 1
            word_gop.append(wg["gop"])
            word_acc.append(w["accuracy"])
            this_utt_gops.append(wg["gop"])

            ph_acc = w["phones-accuracy"]
            if len(pg) == len(ph_acc) and len(pg) > 0:
                n_words_len_matched += 1
                for p, a in zip(pg, ph_acc):
                    if p is not None:
                        phone_gop_matched.append(p["gop"])
                        phone_acc_matched.append(a)

        if this_utt_gops:
            utt_gop_mean.append(float(np.mean(this_utt_gops)))
            utt_acc.append(row["accuracy"])
            utt_total.append(row["total"])

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "word_gop": word_gop, "word_acc": word_acc,
        "utt_gop_mean": utt_gop_mean, "utt_acc": utt_acc, "utt_total": utt_total,
        "phone_gop_matched": phone_gop_matched, "phone_acc_matched": phone_acc_matched,
        "n_utterances": len(rows), "n_words_total": n_words_total,
        "n_words_aligned": n_words_aligned, "n_words_len_matched": n_words_len_matched,
        "elapsed_s": time.time() - t0,
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
                "frac_gop_zero": round(sum(1 for x in v if x == 0.0) / len(v), 3)}
            for a, v in sorted(buckets.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not align_phone.available():
        print("proscor.align_phone optional deps (onnxruntime, phonemizer + espeak-ng) "
              "are not installed.", file=sys.stderr)
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
    print(f"Evaluating {len(rows)} utterances (phone-level model, "
          f"{'int8' if not args.fp32 else 'fp32'}) ...", file=sys.stderr)

    results = run(rows, use_int8=not args.fp32)

    word_corr = correlations(results["word_gop"], results["word_acc"])
    utt_corr_acc = correlations(results["utt_gop_mean"], results["utt_acc"])
    utt_corr_total = correlations(results["utt_gop_mean"], results["utt_total"])
    phone_corr = correlations(results["phone_gop_matched"], results["phone_acc_matched"])
    buckets = bucket_by_accuracy(results["word_gop"], results["word_acc"])

    summary = {
        "split": args.split,
        "precision": "fp32" if args.fp32 else "int8",
        "n_utterances": results["n_utterances"],
        "n_words_total": results["n_words_total"],
        "n_words_aligned": results["n_words_aligned"],
        "align_rate": round(results["n_words_aligned"] / max(1, results["n_words_total"]), 4),
        "n_words_len_matched": results["n_words_len_matched"],
        "phone_count_match_rate": round(results["n_words_len_matched"] / max(1, results["n_words_aligned"]), 4),
        "elapsed_s": round(results["elapsed_s"], 1),
        "word_level_vs_word_accuracy": word_corr,
        "utterance_level_vs_accuracy": utt_corr_acc,
        "utterance_level_vs_total": utt_corr_total,
        "phone_level_vs_phones_accuracy_MATCHED_LENGTH_ONLY": phone_corr,
        "mean_gop_by_word_accuracy": buckets,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
