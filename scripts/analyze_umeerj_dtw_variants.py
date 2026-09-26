#!/usr/bin/env python
"""Offline analysis of `results/umeerj_dtw_variants.jsonl`
(scripts/eval_umeerj_dtw_variants.py), PLAN.md section 5m: text-difficulty
normalization and layer choice for DTW-SSL on UME-ERJ's phonetic sentences.

Scores per item and layer (all "higher = better", i.e. negated costs):
  raw     = -mean(learner costs)                     (section 5l's method)
  sub     = -(mean(learner costs) - nn_mean)          (native-native offset removed)
  z       = -(mean(learner costs) - nn_mean) / nn_std (offset and spread removed)
  ratio   = -mean(learner costs) / nn_mean

Layer choice is *never* made on the items it is reported on: speaker sites
are split into two disjoint halves (seeded), the best layer for a given
scoring variant is picked on half A and reported on half B, and vice versa;
the reported number pools both held-out halves. The full-data layer curve is
printed for inspection only (not used for any reported selection).

Usage:
    python scripts/analyze_umeerj_dtw_variants.py --data results/umeerj_dtw_variants.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proscor import stats as gs

VARIANTS = ("raw", "sub", "z", "ratio")


def score(rec, layer, variant):
    d = rec["layers"][layer]
    c = float(np.mean(d["learner_costs"]))
    if variant == "raw":
        return -c
    if variant == "sub":
        return -(c - d["nn_mean"])
    if variant == "z":
        return -(c - d["nn_mean"]) / max(d["nn_std"], 1e-6)
    if variant == "ratio":
        return -c / d["nn_mean"]
    raise ValueError(variant)


def pearson(recs, layer, variant):
    x = [score(r, layer, variant) for r in recs]
    y = [r["human"] for r in recs]
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/umeerj_dtw_variants.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = [r for r in (json.loads(l) for l in open(args.data)) if not r.get("skipped")]
    layers = list(recs[0]["layers"])
    print(f"{len(recs)} items, {len({r['speaker'] for r in recs})} learner speakers, layers {layers}")

    print("\nfull-data layer curve (inspection only; NOT used for any selection):")
    print(f"{'layer':>6s} " + " ".join(f"{v:>7s}" for v in VARIANTS))
    for L in layers:
        print(f"{L:>6s} " + " ".join(f"{pearson(recs, L, v):7.3f}" for v in VARIANTS))

    print("\nfinal-layer ('last') variants with bootstrap CIs (no layer selection involved):")
    y = [r["human"] for r in recs]
    c = [r["speaker"] for r in recs]
    base = [score(r, "last", "raw") for r in recs]
    for v in VARIANTS:
        x = [score(r, "last", v) for r in recs]
        e = gs.cluster_bootstrap_pearson(x, y, c)
        line = f"  {v:6s} r={e['r']:.3f} [{e['ci_lo']:.3f},{e['ci_hi']:.3f}]"
        if v != "raw":
            p = gs.cluster_bootstrap_paired_diff(x, base, y, c)
            line += f"   vs raw: {p['diff']:+.3f} sig={p['significant']}"
        print(line)

    # speaker-disjoint 2-fold layer selection
    speakers = sorted({r["speaker"] for r in recs})
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    half = {s: i % 2 for i, s in enumerate(speakers)}
    folds = [[r for r in recs if half[r["speaker"]] == k] for k in (0, 1)]
    print(f"\nheld-out layer selection (fold sizes {len(folds[0])}/{len(folds[1])}):")
    for v in VARIANTS:
        xs, ys, cs, chosen = [], [], [], []
        for k in (0, 1):
            sel, test = folds[k], folds[1 - k]
            best = max(layers, key=lambda L: pearson(sel, L, v))
            chosen.append(best)
            xs += [score(r, best, v) for r in test]
            ys += [r["human"] for r in test]
            cs += [r["speaker"] for r in test]
        e = gs.cluster_bootstrap_pearson(xs, ys, cs)
        # same held-out items scored with the fixed final layer, raw: the honest comparator
        xb, _, _ = [], None, None
        xb = []
        for k in (0, 1):
            xb += [score(r, "last", "raw") for r in folds[1 - k]]
        p = gs.cluster_bootstrap_paired_diff(xs, xb, ys, cs)
        print(f"  {v:6s} layers chosen {chosen}: held-out r={e['r']:.3f} [{e['ci_lo']:.3f},{e['ci_hi']:.3f}]"
              f"  vs final-layer raw: {p['diff']:+.3f} sig={p['significant']}")


if __name__ == "__main__":
    main()
