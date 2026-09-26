#!/usr/bin/env python
"""Fully held-out utterance-level fusion on speechocean762 (PLAN.md section
5m). scripts/analyze_so762_stack.py's fused numbers used a post-hoc "all-in"
GOP configuration and z-scored on the evaluated set's own statistics; this
removes both: for each of two speaker-disjoint halves, the GOP configuration
(same grid as analyze_so762_stack.py) is chosen on half A, the z-score
means/stds of the GOP score and of -DTW cost are taken from half A only,
and the fused score is evaluated on half B (and vice versa). Pooled
held-out correlations are reported against utterance accuracy/total, with
paired differences vs. the GOP-only held-out score and vs. the plain
baseline (xlsr-53 GOP-SF, mean of word means). Nothing uses labels except
the configuration choice, and that only ever sees the other half.

Usage:
    python scripts/analyze_so762_fusion_heldout.py
"""
import itertools
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import stats as gs
from analyze_so762_full import evaluate, load_jsonl, native_stats, scorer_plain
from analyze_so762_stack import make_scorer


def utt_vectors(recs, scorer, utt_how):
    xs, dt, acc, tot, spk = [], [], [], [], []
    for r in recs:
        ws = [w for w in (scorer(w) for w in r["words"]) if w]
        if not ws:
            continue
        if utt_how == "mean_phone":
            xs.append(float(np.mean([s for w in ws for _, s in w])))
        else:
            xs.append(float(np.mean([np.mean([s for _, s in w]) for w in ws])))
        dt.append(-r["dtw_mean"]); acc.append(r["accuracy"]); tot.append(r["total"]); spk.append(r["speaker"])
    return np.array(xs), np.array(dt), np.array(acc), np.array(tot), np.array(spk)


def main():
    recs = load_jsonl("results/so762_full.jsonl")
    nat = load_jsonl("results/native_gop.jsonl")
    nstats = {m: native_stats(nat, m) for m in ("xlsr_sf", "lv60_sf")}
    speakers = sorted({r["speaker"] for r in recs})
    random.Random(0).shuffle(speakers)
    fold = {s: i % 2 for i, s in enumerate(speakers)}
    grid = list(itertools.product(("none", "shift", "z"), (("xlsr_sf",), ("xlsr_sf", "lv60_sf")),
                                  ("mean",), ("mean_word", "mean_phone")))
    base_vecs = utt_vectors(recs, scorer_plain("xlsr_sf"), "mean_word")
    cfg_vecs = {cfg: utt_vectors(recs, make_scorer(cfg[1], cfg[0], nstats), cfg[3]) for cfg in grid}
    spk = base_vecs[4]
    z = lambda v, ref: (v - ref.mean()) / (ref.std() + 1e-12)

    for lname, li in (("utt_acc", 2), ("utt_tot", 3)):
        pooled = {k: ([], [], []) for k in ("base", "gop", "dtw", "fused", "fused_base")}
        chosen = []
        for k in (0, 1):
            sel, tst = fold_mask(spk, fold, k), fold_mask(spk, fold, 1 - k)
            y = base_vecs[li]
            best = max(grid, key=lambda c: np.corrcoef(cfg_vecs[c][0][sel], y[sel])[0, 1])
            chosen.append(best)
            g = cfg_vecs[best][0]
            d = base_vecs[1]
            fused = (z(g[tst], g[sel]) + z(d[tst], d[sel])) / 2
            fused_base = (z(base_vecs[0][tst], base_vecs[0][sel]) + z(d[tst], d[sel])) / 2
            for name, v in (("base", base_vecs[0][tst]), ("gop", g[tst]), ("dtw", d[tst]),
                            ("fused", fused), ("fused_base", fused_base)):
                pooled[name][0].extend(v); pooled[name][1].extend(y[tst]); pooled[name][2].extend(spk[tst])
        print(f"\n{lname}: configs chosen {chosen}")
        for name in ("base", "gop", "dtw", "fused_base", "fused"):
            e = gs.cluster_bootstrap_pearson(*pooled[name])
            print(f"  {name:11s} r={e['r']:.3f} [{e['ci_lo']:.3f},{e['ci_hi']:.3f}]")
        for a, b in (("fused", "gop"), ("fused", "base"), ("dtw", "base")):
            p = gs.cluster_bootstrap_paired_diff(pooled[a][0], pooled[b][0], pooled[a][1], pooled[a][2])
            print(f"  {a} - {b}: {p['diff']:+.3f} [{p['diff_ci_lo']:+.3f},{p['diff_ci_hi']:+.3f}] sig={p['significant']}")


def fold_mask(spk, fold, k):
    return np.array([fold[s] == k for s in spk])


if __name__ == "__main__":
    main()
