#!/usr/bin/env python
"""Compare synthetic reference voice sets on the so762 pilot subset (PLAN.md
section 5m): the same utterances (6 per speaker, speaker-half 0), scored with
DTW-SSL against each voice set, everything else held fixed.

Fixed, pre-declared scoring (no per-set tuning): GOP = xlsr-53 GOP-SF with
native per-phone shift calibration; DTW = -cost at the final WavLM layer;
fused = unweighted mean of the two z-scores (z-stats over the pilot items).
Reported per voice set: DTW-only utterance correlation (whole-path cost) with
accuracy and total, DTW-only phone/word, and fused phone/word/utterance. The
kitten4 row comes from the full collection restricted to the same
utterances (identical procedure). Selection metric = mean of the four numbers
that matter for the decision: DTW-only utt-acc, DTW-only utt-tot, fused
phone, fused word. Selection happens on half 0 only; half 1 is used later to
confirm the chosen set (`--confirm`).

Usage:
    python scripts/analyze_tts_pilot.py results/so762_phone_dtw.jsonl results/pilot_kokoro4.jsonl ...
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import stats as gs
import analyze_so762_phone_fusion as F
from analyze_so762_full import load_jsonl


def evaluate(path, idx_filter=None):
    dtw_recs = {r["idx"]: r for r in load_jsonl(path)}
    if idx_filter is not None:
        keep = set(idx_filter)
    else:
        keep = set(dtw_recs)
    # write a filtered temp view for build()
    import json, tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for i, r in dtw_recs.items():
        if i in keep:
            tmp.write(json.dumps(r) + "\n")
    tmp.close()
    ph, woff, wutt, wacc, iidx, iy, ispk, uacc, utot, uspk = F.build(tmp.name)
    nw = len(wacc)
    g = ph["gop"][("shift", ("xlsr_sf",))]
    d = ph["dtw"]["last"]
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    fused = (z(g) + z(d)) / 2
    w_spk = np.array([uspk[u] for u in wutt])
    utts = sorted(set(wutt.tolist()))
    upos = {u: i for i, u in enumerate(utts)}
    uow = np.array([upos[u] for u in wutt])
    wlen = np.diff(woff)

    def word_low2(sc):
        return np.array([np.sort(sc[woff[i]:woff[i + 1]])[:2].mean() for i in range(nw)])

    def utt_mean(sc):
        return (np.bincount(uow, weights=np.add.reduceat(sc, woff[:-1]), minlength=len(utts))
                / np.bincount(uow, weights=wlen, minlength=len(utts)))

    u_spk = np.array([uspk[u] for u in utts])
    res = {"n_utt": len(utts), "n_words": nw, "n_phone_items": len(iy), "n_spk": len(set(u_spk))}
    # whole-path utterance DTW cost (last layer): from file if present, else from full collection
    if all("utt_cost" in dtw_recs[u] for u in utts):
        wp = np.array([-dtw_recs[u]["utt_cost"]["last"] for u in utts])
    else:
        full = {r["idx"]: r for r in load_jsonl("results/so762_full.jsonl")}
        wp = np.array([-full[u]["dtw_mean"] for u in utts])
    ya = np.array([uacc[u] for u in utts]); yt = np.array([utot[u] for u in utts])
    R = lambda x, y, c: gs.cluster_bootstrap_pearson(list(x), list(y), list(c))["r"]
    res["dtw_utt_acc"], res["dtw_utt_tot"] = R(wp, ya, u_spk), R(wp, yt, u_spk)
    res["dtw_phone"] = R(d[iidx], iy, ispk)
    res["dtw_word"] = R(word_low2(d), wacc, w_spk)
    res["fused_phone"] = R(fused[iidx], iy, ispk)
    res["fused_word"] = R(word_low2(fused), wacc, w_spk)
    res["fused_utt_acc"], res["fused_utt_tot"] = R(utt_mean(fused), ya, u_spk), R(utt_mean(fused), yt, u_spk)
    res["gop_phone"] = R(g[iidx], iy, ispk)
    res["select"] = np.mean([res["dtw_utt_acc"], res["dtw_utt_tot"], res["fused_phone"], res["fused_word"]])
    return res


def main():
    paths = sys.argv[1:]
    # restrict every set to the utterances all sets share
    idx_sets = [set(r["idx"] for r in load_jsonl(p)) for p in paths]
    common = set.intersection(*idx_sets)
    print(f"{len(common)} utterances common to all {len(paths)} sets")
    cols = ("n_spk", "dtw_utt_acc", "dtw_utt_tot", "dtw_phone", "dtw_word", "fused_phone", "fused_word",
            "fused_utt_acc", "fused_utt_tot", "gop_phone", "select")
    print(f"{'set':22s} " + " ".join(f"{c[-11:]:>11s}" for c in cols))
    for p in paths:
        r = evaluate(p, common)
        print(f"{Path(p).stem:22s} " + " ".join(f"{r[c]:11.3f}" if c != "n_spk" else f"{r[c]:11d}" for c in cols))


if __name__ == "__main__":
    main()
