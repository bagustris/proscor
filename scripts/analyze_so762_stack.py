#!/usr/bin/env python
"""Stacked zero-shot pipeline on the full speechocean762 collection, with
held-out configuration selection (PLAN.md section 5m).

The single-lever probes in scripts/analyze_so762_full.py each looked at the
test labels. Stacking them and reporting the best combination on the same
labels would be selection on the test set. Instead: build a small grid of
label-free configurations, split the 125 speakers into two disjoint halves
(seeded), pick each level's best configuration on one half, report it on the
other, and pool both held-out halves. The full-data numbers for the
pre-declared "all-in" configuration are printed for inspection only.

Grid (every axis is label-free):
  calibration : none | native per-phone shift | native per-phone z  (native stats
                from results/native_gop.jsonl -- UME-ERJ American speakers)
  models      : xlsr-53 GOP-SF | unweighted mean of xlsr-53 + lv-60 GOP-SF
                (calibration, when on, is applied per model before averaging)
  word pooling: mean of phones | mean of the 2 lowest phones
  utt pooling : mean of word scores | mean over all phones

Then, on top of each level's selected utterance configuration, an unweighted
z-score fusion with DTW-SSL (mean cost vs. 4 synthetic-voice templates,
`dtw_mean` in the collected records) is evaluated at utterance level.

Usage:
    python scripts/analyze_so762_stack.py
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import stats as gs
from analyze_so762_full import (clean, evaluate, load_jsonl, native_stats, scorer_calibrated,
                                scorer_plain)

LEVELS = ("phone", "word", "utt_acc", "utt_tot")


def make_scorer(models, calib, nstats):
    if calib == "none":
        parts = [scorer_plain(m) for m in models]
    else:
        parts = [scorer_calibrated(m, nstats[m], calib) for m in models]

    def f(w):
        outs = [p(w) for p in parts]
        if any(not o for o in outs):
            return None
        phones = [p for p, _ in outs[0]]
        if any([p for p, _ in o] != phones for o in outs):
            return None
        return [(p, float(np.mean([o[i][1] for o in outs]))) for i, p in enumerate(phones)]
    return f


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/so762_full.jsonl")
    ap.add_argument("--native", default="results/native_gop.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = load_jsonl(args.data)
    nat = load_jsonl(args.native)
    nstats = {m: native_stats(nat, m) for m in ("xlsr_sf", "lv60_sf")}
    speakers = sorted({r["speaker"] for r in recs})
    random.Random(args.seed).shuffle(speakers)
    fold_of = {s: i % 2 for i, s in enumerate(speakers)}
    print(f"{len(recs)} utterances / {len(speakers)} speakers; folds {sum(v == 0 for v in fold_of.values())}/"
          f"{sum(v == 1 for v in fold_of.values())} speakers")

    grid = list(itertools.product(("none", "shift", "z"), (("xlsr_sf",), ("xlsr_sf", "lv60_sf")),
                                  ("mean", "mean_low2"), ("mean_word", "mean_phone")))
    evals = {}
    for cfg in grid:
        calib, models, wh, uh = cfg
        evals[cfg] = evaluate(recs, make_scorer(models, calib, nstats), word_how=wh, utt_how=uh)
    base_cfg = ("none", ("xlsr_sf",), "mean", "mean_word")
    allin_cfg = ("z", ("xlsr_sf", "lv60_sf"), "mean_low2", "mean_phone")

    def sub(triple, fold):
        s, y, c = triple
        idx = [i for i, sp in enumerate(c) if fold_of[sp] == fold]
        return [s[i] for i in idx], [y[i] for i in idx], [c[i] for i in idx]

    def rr(t):
        return float(np.corrcoef(t[0], t[1])[0, 1])

    results = {}
    print("\nheld-out selection (config chosen on one speaker-half, reported on the other):")
    for lvl in LEVELS:
        pooled = {"sel": ([], [], []), "base": ([], [], [])}
        chosen = []
        for k in (0, 1):
            best = max(grid, key=lambda cfg: rr(sub(evals[cfg][lvl], k)))
            chosen.append(best)
            for name, cfg in (("sel", best), ("base", base_cfg)):
                s, y, c = sub(evals[cfg][lvl], 1 - k)
                for dst, src in zip(pooled[name], (s, y, c)):
                    dst.extend(src)
        e = gs.cluster_bootstrap_pearson(*pooled["sel"])
        b = gs.cluster_bootstrap_pearson(*pooled["base"])
        # paired: identical held-out item order only if the two configs yield the same items
        same = len(pooled["sel"][0]) == len(pooled["base"][0])
        d = (gs.cluster_bootstrap_paired_diff(pooled["sel"][0], pooled["base"][0], pooled["sel"][1], pooled["sel"][2])
             if same else None)
        results[lvl] = {"selected": e, "baseline": b, "paired": d, "chosen": [str(c) for c in chosen]}
        line = (f"  {lvl:8s} baseline {b['r']:.3f} -> selected {e['r']:.3f} [{e['ci_lo']:.3f},{e['ci_hi']:.3f}]")
        line += (f"  diff {d['diff']:+.3f} sig={d['significant']}" if d else "  (item sets differ: unpaired)")
        print(line)
        for c in chosen:
            print(f"      chose {c}")

    print("\nfull-data numbers for the pre-declared all-in config (inspection only, not held-out):")
    for lvl in LEVELS:
        b, a = evals[base_cfg][lvl], evals[allin_cfg][lvl]
        print(f"  {lvl:8s} baseline {rr(b):.3f} -> all-in {rr(a):.3f}   (n {len(b[0])} vs {len(a[0])})")

    # utterance-level fusion with DTW-SSL on the all-in config
    print("\nutterance-level fusion with DTW-SSL (4 TTS templates), all-in GOP config, full data:")
    by_idx = {r["idx"]: r for r in recs}
    for lvl, key in (("utt_acc", "accuracy"), ("utt_tot", "total")):
        # rebuild per-utterance vectors aligned to recs order used by evaluate()
        us, ys, cs, dt = [], [], [], []
        scorer = make_scorer(allin_cfg[1], allin_cfg[0], nstats)
        for r in recs:
            ws = [w for w in (scorer(w) for w in r["words"]) if w]
            if not ws:
                continue
            us.append(float(np.mean([s for w in ws for _, s in w])))
            ys.append(r[key]); cs.append(r["speaker"]); dt.append(-r["dtw_mean"])
        fused = list((z(us) + z(dt)) / 2)
        for name, x in (("GOP all-in", us), ("DTW-SSL(TTS)", dt), ("fused", fused)):
            e = gs.cluster_bootstrap_pearson(x, ys, cs)
            print(f"  {lvl:8s} {name:13s} r={e['r']:.3f} [{e['ci_lo']:.3f},{e['ci_hi']:.3f}]")
        p = gs.cluster_bootstrap_paired_diff(fused, us, ys, cs)
        print(f"  {lvl:8s} fused - GOP all-in: {p['diff']:+.3f} sig={p['significant']}")
        results[lvl + "_fusion"] = {"fused": gs.cluster_bootstrap_pearson(fused, ys, cs),
                                    "gop_allin": gs.cluster_bootstrap_pearson(us, ys, cs),
                                    "dtw": gs.cluster_bootstrap_pearson(dt, ys, cs), "paired": p}
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
