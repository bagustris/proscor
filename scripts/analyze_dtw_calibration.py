#!/usr/bin/env python
"""Does calibrating the per-phone DTW cost with native-speech statistics help?
(PLAN.md section 5m.) speechocean762, all 2,500 utterances, kitten4 templates:
per-phone DTW cost (final WavLM layer) is used (a) as-is, (b) minus the
native mean for that phone, (c) z-scored by the native mean/std, where the
statistics come from scripts/collect_native_dtw.py run with the *same*
references (`results/native_dtw_kitten4.jsonl`: native UME-ERJ speakers scored
against the same 4 Kitten voices), so the calibration removes exactly the
per-phone offset that synthetic-reference scoring adds. Everything else is
fixed and pre-declared (GOP = xlsr-53 GOP-SF with native shift calibration;
fusion = unweighted mean of z-scores; word pooling = mean of 2 lowest phones;
utterance = mean over phones); only the three DTW variants are compared, so
nothing is picked from a grid. Speaker-cluster bootstrap; paired differences
of each calibrated variant vs. the uncalibrated one, and of each fused score
vs. GOP-only.

Usage:
    python scripts/analyze_dtw_calibration.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import stats as gs
import analyze_so762_phone_fusion as F


def main():
    ph, woff, wutt, wacc, iidx, iy, ispk, uacc, utot, uspk = F.build(
        "results/so762_phone_dtw.jsonl", dtw_native="results/native_dtw_kitten4.jsonl")
    nw = len(wacc)
    w_spk = np.array([uspk[u] for u in wutt])
    utts = sorted(set(wutt.tolist())); upos = {u: i for i, u in enumerate(utts)}
    uow = np.array([upos[u] for u in wutt]); wlen = np.diff(woff)
    u_spk = np.array([uspk[u] for u in utts])
    ya = np.array([uacc[u] for u in utts]); yt = np.array([utot[u] for u in utts])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    g = ph["gop"][("shift", ("xlsr_sf",))]

    def low2(sc):
        return np.array([np.sort(sc[woff[i]:woff[i + 1]])[:2].mean() for i in range(nw)])

    def umean(sc):
        return (np.bincount(uow, weights=np.add.reduceat(sc, woff[:-1]), minlength=len(utts))
                / np.bincount(uow, weights=wlen, minlength=len(utts)))

    def levels(sc):
        return {"phone": (sc[iidx], iy, ispk), "word": (low2(sc), wacc, w_spk),
                "utt_acc": (umean(sc), ya, u_spk), "utt_tot": (umean(sc), yt, u_spk)}

    variants = {"dtw none": ph["dtw"]["last"], "dtw shift": ph["dtwc"][("shift", "last")],
                "dtw z": ph["dtwc"][("z", "last")]}
    scores = {"gop only": z(g)}
    for name, d in variants.items():
        scores[name] = z(d)
        scores["fused " + name.split()[1]] = (z(g) + z(d)) / 2
    L = {k: levels(v) for k, v in scores.items()}
    print(f"{'score':14s} " + " ".join(f"{l:>17s}" for l in ("phone", "word", "utt_acc", "utt_tot")))
    for k in scores:
        cells = []
        for lvl in ("phone", "word", "utt_acc", "utt_tot"):
            e = gs.cluster_bootstrap_pearson(*L[k][lvl])
            cells.append(f"{e['r']:.3f} [{e['ci_lo']:.2f},{e['ci_hi']:.2f}]")
        print(f"{k:14s} " + " ".join(f"{c:>17s}" for c in cells))
    print("\npaired differences:")
    for a, b in (("dtw shift", "dtw none"), ("dtw z", "dtw none"), ("fused shift", "fused none"),
                 ("fused z", "fused none"), ("fused shift", "gop only"), ("fused z", "gop only"),
                 ("fused none", "gop only")):
        row = []
        for lvl in ("phone", "word", "utt_acc", "utt_tot"):
            A, B = L[a][lvl], L[b][lvl]
            p = gs.cluster_bootstrap_paired_diff(list(A[0]), list(B[0]), list(A[1]), list(A[2]))
            row.append(f"{p['diff']:+.3f}{'*' if p['significant'] else ' '}")
        print(f"  {a:12s} - {b:11s} " + "  ".join(f"{lvl}:{r}" for lvl, r in zip(("phone", "word", "utt_acc", "utt_tot"), row)))


if __name__ == "__main__":
    main()
