#!/usr/bin/env python
"""Validate phone-level GOP-lite (proscor/align_phone.py, PLAN.md section 5a
item 4) against speechocean762's human pronunciation-accuracy labels.

This is the phone-granularity counterpart to scripts/eval_so762.py (which
validates the BPE-subword NeMo model at word granularity, since that model
has no phone-level output). align_phone.py uses a genuine phoneme-CTC model
(wav2vec2-lv-60-espeak-cv-ft, ONNX) and phonemizes each canonical word live
with espeak-ng (the same tool that produced the model's training labels) to
get the target phone sequence.

Comparisons reported:
1. **Word-level** (comparable to scripts/eval_so762.py's BPE-model 0.471 and
   to GOPT's 0.533, Table 1 of arXiv:2205.03432): mean phone GOP per word
   vs. the dataset's per-word `accuracy` (0-10). Finer-grained than the BPE
   version's word aggregation, but empirically *lower* correlation than it
   -- see PLAN.md section 5a item 4 for the full comparison and discussion.
2. **Utterance-level**: mean word GOP vs. sentence `accuracy`/`total`.
3. **Direct per-phone, matched-length-only** (legacy metric, kept for
   comparison): espeak's phones for a word don't always line up 1:1 with
   the dataset's own ARPABET phone segmentation (e.g. espeak merges
   vowel+R into one rhotic token: "ɑːɹ" for the vowel in "mark" vs. the
   dataset's separate AA/R) -- see proscor/align_phone.py's docstring. This
   comparison only uses words where the phone COUNTS happen to match,
   zipping the two sequences by position without checking phone identity.
4. **Direct per-phone, reconciled** (the headline number):
   `align_phone.reconcile_phones` aligns every word's espeak phones against
   its ARPABET phones with a constrained DP (1:1 matches plus 2:1/3:1
   merges for known patterns), so words with a phone-count mismatch
   contribute real (if occasionally approximate) GOP estimates instead of
   being dropped entirely. Reported alongside a **reconciled-only** row
   (just the phones a length-mismatched word contributed, i.e. genuinely
   new coverage) so a reader can see whether the newly-recovered phones
   correlate as well as the already-matched ones. `op_counts` in the output
   tallies how the DP resolved every dataset phone (match/sub/merge2/
   merge3/del) as a sanity check against over-firing.

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

from proscor import align_phone, stats as gopstats


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "mispeech/speechocean762", f"data/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    return pq.read_table(path).to_pylist()


def run(rows: list, use_int8: bool = True, progress_every: int = 200, engine: str = "posterior",
        model_id: str = None) -> dict:
    align_fn = align_phone.align_words_gop if engine == "posterior" else align_phone.align_words_gop_sf
    word_gop, word_acc, word_speaker = [], [], []
    utt_gop_mean, utt_acc, utt_total, utt_speaker = [], [], [], []
    phone_gop_matched, phone_acc_matched, phone_speaker_matched = [], [], []
    phone_gop_reconciled, phone_acc_reconciled, phone_speaker_reconciled = [], [], []
    phone_gop_newly_covered, phone_acc_newly_covered = [], []
    op_counts = defaultdict(int)
    n_words_total = n_words_aligned = 0
    n_words_len_matched = 0
    n_phones_total = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        n_words_total += len(words)
        speaker = row["speaker"]
        try:
            result = align_fn(samples, words, sr=sr, use_int8=use_int8, model_id=model_id)
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
            word_speaker.append(speaker)
            this_utt_gops.append(wg["gop"])

            dataset_phones = w["phones"]
            ph_acc = w["phones-accuracy"]
            n_phones_total += len(dataset_phones)
            length_matched = len(pg) == len(ph_acc) and len(pg) > 0
            if length_matched:
                n_words_len_matched += 1
                for p, a in zip(pg, ph_acc):
                    if p is not None:
                        phone_gop_matched.append(p["gop"])
                        phone_acc_matched.append(a)
                        phone_speaker_matched.append(speaker)

            espeak_phones = [p["phone"] for p in pg if p is not None]
            espeak_gops = [p["gop"] for p in pg if p is not None]
            if espeak_phones and dataset_phones:
                rec_gops, rec_ops = align_phone.reconcile_phones(dataset_phones, espeak_phones, espeak_gops)
                for g, a, op in zip(rec_gops, ph_acc, rec_ops):
                    op_counts[op] += 1
                    if g is None:
                        continue
                    phone_gop_reconciled.append(g)
                    phone_acc_reconciled.append(a)
                    phone_speaker_reconciled.append(speaker)
                    if not length_matched:
                        phone_gop_newly_covered.append(g)
                        phone_acc_newly_covered.append(a)

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
        "word_gop": word_gop, "word_acc": word_acc, "word_speaker": word_speaker,
        "utt_gop_mean": utt_gop_mean, "utt_acc": utt_acc, "utt_total": utt_total, "utt_speaker": utt_speaker,
        "phone_gop_matched": phone_gop_matched, "phone_acc_matched": phone_acc_matched,
        "phone_speaker_matched": phone_speaker_matched,
        "phone_gop_reconciled": phone_gop_reconciled, "phone_acc_reconciled": phone_acc_reconciled,
        "phone_speaker_reconciled": phone_speaker_reconciled,
        "phone_gop_newly_covered": phone_gop_newly_covered, "phone_acc_newly_covered": phone_acc_newly_covered,
        "op_counts": dict(op_counts),
        "n_utterances": len(rows), "n_words_total": n_words_total,
        "n_words_aligned": n_words_aligned, "n_words_len_matched": n_words_len_matched,
        "n_phones_total": n_phones_total,
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
    ap.add_argument("--engine", default="posterior", choices=["posterior", "sf"],
                     help="posterior = Viterbi posterior-deficit (align_words_gop, default); "
                          "sf = segmentation-free (align_words_gop_sf, PLAN.md section 5f)")
    ap.add_argument("--model-id", default=None,
                     help="acoustic model repo; default None uses align_phone.MODEL_REPO (ONNX/lv-60). "
                          "Pass align_phone.TORCH_MODEL_REPO (xlsr-53) for PLAN.md section 5i "
                          "(needs torch+transformers, requirements-eval.txt)")
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

    results = run(rows, use_int8=not args.fp32, engine=args.engine, model_id=args.model_id)

    word_corr = correlations(results["word_gop"], results["word_acc"])
    utt_corr_acc = correlations(results["utt_gop_mean"], results["utt_acc"])
    utt_corr_total = correlations(results["utt_gop_mean"], results["utt_total"])
    phone_corr_matched = correlations(results["phone_gop_matched"], results["phone_acc_matched"])
    phone_corr_reconciled = correlations(results["phone_gop_reconciled"], results["phone_acc_reconciled"])
    phone_corr_newly_covered = correlations(results["phone_gop_newly_covered"], results["phone_acc_newly_covered"])
    buckets = bucket_by_accuracy(results["word_gop"], results["word_acc"])

    # Speaker-cluster bootstrap CIs -- words/phones cluster within a speaker
    # (a child's ability level correlates across all their words/phones), so
    # a naive per-item Fisher-z CI is not the right uncertainty to report;
    # see PLAN.md section 5c and proscor/stats.py for why.
    word_corr_ci = gopstats.cluster_bootstrap_pearson(
        results["word_gop"], results["word_acc"], results["word_speaker"]) if results["word_gop"] else None
    utt_corr_acc_ci = gopstats.cluster_bootstrap_pearson(
        results["utt_gop_mean"], results["utt_acc"], results["utt_speaker"]) if results["utt_gop_mean"] else None
    phone_corr_matched_ci = gopstats.cluster_bootstrap_pearson(
        results["phone_gop_matched"], results["phone_acc_matched"], results["phone_speaker_matched"]
    ) if results["phone_gop_matched"] else None
    phone_corr_reconciled_ci = gopstats.cluster_bootstrap_pearson(
        results["phone_gop_reconciled"], results["phone_acc_reconciled"], results["phone_speaker_reconciled"]
    ) if results["phone_gop_reconciled"] else None

    # Like-for-like check against L2-ARCTIC's binary correct/error label
    # (speechocean762's phones-accuracy is graded 0-2): binarizing here lets
    # a reader tell how much of the so762-vs-L2-ARCTIC phone-level PCC gap
    # is a label-granularity artifact rather than a real corpus difference.
    binary_acc_reconciled = [1 if a >= 2 else 0 for a in results["phone_acc_reconciled"]]
    phone_corr_reconciled_binarized = correlations(results["phone_gop_reconciled"], binary_acc_reconciled)
    phone_corr_reconciled_binarized_ci = gopstats.cluster_bootstrap_pearson(
        results["phone_gop_reconciled"], binary_acc_reconciled, results["phone_speaker_reconciled"]
    ) if results["phone_gop_reconciled"] else None

    # Two different denominators on purpose -- do not conflate them (see
    # PLAN.md section 5a item 4, "phone coverage" vs. "word match rate" were
    # mixed up twice before this explicit split was added).
    word_match_rate = round(results["n_words_len_matched"] / max(1, results["n_words_aligned"]), 4)
    phone_coverage_matched = round(len(results["phone_gop_matched"]) / max(1, results["n_phones_total"]), 4)
    phone_coverage_reconciled = round(len(results["phone_gop_reconciled"]) / max(1, results["n_phones_total"]), 4)

    summary = {
        "split": args.split,
        "engine": args.engine,
        "model_id": args.model_id or align_phone.MODEL_REPO,
        "precision": "fp32" if args.fp32 else "int8",
        "n_utterances": results["n_utterances"],
        "n_words_total": results["n_words_total"],
        "n_words_aligned": results["n_words_aligned"],
        "align_rate": round(results["n_words_aligned"] / max(1, results["n_words_total"]), 4),
        "n_words_len_matched": results["n_words_len_matched"],
        "word_match_rate": word_match_rate,
        "n_phones_total": results["n_phones_total"],
        "phone_coverage_matched_only": phone_coverage_matched,
        "phone_coverage_reconciled": phone_coverage_reconciled,
        "op_counts": results["op_counts"],
        "elapsed_s": round(results["elapsed_s"], 1),
        "word_level_vs_word_accuracy": word_corr,
        "word_level_vs_word_accuracy_speaker_cluster_ci": word_corr_ci,
        "utterance_level_vs_accuracy": utt_corr_acc,
        "utterance_level_vs_accuracy_speaker_cluster_ci": utt_corr_acc_ci,
        "utterance_level_vs_total": utt_corr_total,
        "phone_level_matched_only": phone_corr_matched,
        "phone_level_matched_only_speaker_cluster_ci": phone_corr_matched_ci,
        "phone_level_reconciled": phone_corr_reconciled,
        "phone_level_reconciled_speaker_cluster_ci": phone_corr_reconciled_ci,
        "phone_level_reconciled_binarized": phone_corr_reconciled_binarized,
        "phone_level_reconciled_binarized_speaker_cluster_ci": phone_corr_reconciled_binarized_ci,
        "phone_level_newly_covered_only": phone_corr_newly_covered,
        "mean_gop_by_word_accuracy": buckets,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
