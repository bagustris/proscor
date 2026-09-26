#!/usr/bin/env python
"""Phone/word/utterance-level fusion of calibrated GOP and per-phone
DTW-SSL on speechocean762, fully held-out (PLAN.md section 5m).

Inputs: results/so762_full.jsonl (GOP variants + labels + whole-utterance
DTW), results/so762_phone_dtw.jsonl (per-phone DTW cost, 4 WavLM layers,
4 synthetic templates), results/native_gop.jsonl (native per-phone GOP
statistics for label-free calibration).

Everything is defined over the flat vector of espeak phones of the words
that survive one common filter (learner-side xlsr-53/lv-60 phone lists and
the template-side DTW phone list must agree, and every phone must have a DTW
cost), so all scores are on identical items. Phone-level evaluation
reconciles to ARPABET once (`reconcile_phones`'s alignment depends only on
phone identity, so it is computed once and reused for every score vector);
word scores pool phones (mean, or mean of the two lowest); utterance scores
average all phones.

Held-out protocol: speakers are split into two disjoint halves (seeded).
For each half k the configuration -- GOP calibration/ensemble, WavLM layer,
and (per level) word pooling -- is chosen on half k by maximizing that
level's correlation of the *fused* score, the z-score means/stds are taken
from half k only, and the score is evaluated on the other half; pooled
held-out correlations are reported with speaker-cluster CIs and paired
differences. Also reported: GOP-only and DTW-only under the same protocol,
and the plain baseline (xlsr-53 GOP-SF, uncalibrated).

Usage:
    python scripts/analyze_so762_phone_fusion.py
"""
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import align_phone, stats as gs
from analyze_so762_full import clean, load_jsonl, native_stats

LAYERS = ("last", "15", "18", "21")
CALIBS = ("none", "shift", "z")
MODELSETS = (("xlsr_sf",), ("xlsr_sf", "lv60_sf"))


def calibrate(phones, gops, stats, mode):
    if mode == "none":
        return np.array(gops)
    st, glob = stats
    out = []
    for p, g in zip(phones, gops):
        m, s, _v = st.get(p) or glob
        out.append(g - m if mode == "shift" else (g - m) / s)
    return np.array(out)


def native_dtw_stats(path, layer, min_n=20):
    """Per-espeak-phone mean/std of the *negated* DTW cost of native speech
    (scripts/collect_native_dtw.py output) at `layer`, with a global
    fallback for rare phones -- the DTW analogue of `native_stats`."""
    from collections import defaultdict
    per = defaultdict(list)
    for r in load_jsonl(path):
        for w in r["words"]:
            for p, c in zip(w["phones"], w["cost"][layer]):
                if p is not None and c is not None:
                    per[p].append(-c)
    allv = np.array([v for vs in per.values() for v in vs])
    glob = (allv.mean(), allv.std() + 1e-6)
    return {p: (np.mean(v), np.std(v) + 1e-6) for p, v in per.items() if len(v) >= min_n}, glob


def build(dtw_path="results/so762_phone_dtw.jsonl", dtw_native=None):
    full = {r["idx"]: r for r in load_jsonl("results/so762_full.jsonl")}
    dtw = {r["idx"]: r for r in load_jsonl(dtw_path)}
    full = {i: r for i, r in full.items() if i in dtw}   # subset runs (pilots) score only their utterances
    nat = load_jsonl("results/native_gop.jsonl")
    nstats = {m: native_stats(nat, m) for m in ("xlsr_sf", "lv60_sf")}

    ph = {"gop": {(c, m): [] for c in CALIBS for m in MODELSETS}, "dtw": {L: [] for L in LAYERS}}
    dstats = {L: native_dtw_stats(dtw_native, L) for L in LAYERS} if dtw_native else None
    if dstats:
        ph["dtwc"] = {(mode, L): [] for mode in ("shift", "z") for L in LAYERS}
    word_off, word_utt, word_acc = [], [], []       # per kept word: start offset, utterance id, accuracy
    item_idx, item_y, item_spk = [], [], []          # reconciled phone items -> espeak index
    utt_acc, utt_tot, utt_spk = {}, {}, {}
    n = 0
    skipped = {"missing_dtw": 0, "mismatch": 0}
    for idx, r in full.items():
        d = dtw.get(idx)
        if d is None:
            skipped["missing_dtw"] += 1
            continue
        for w, dw in zip(r["words"], d["words"]):
            sf, lv = clean(w, "xlsr_sf"), clean(w, "lv60_sf")
            phones = [p for p, _ in sf]
            dp = dw["phones"]
            costs = {L: dw["cost"][L] for L in LAYERS}
            if (not phones or [p for p, _ in lv] != phones or [p for p in dp if p is not None] != phones
                    or any(c is None for L in LAYERS for c, p in zip(costs[L], dp) if p is not None)):
                skipped["mismatch"] += 1
                continue
            for c in CALIBS:
                for m in MODELSETS:
                    parts = [calibrate(phones, [g for _, g in (sf if v == "xlsr_sf" else lv)], nstats[v], c) for v in m]
                    ph["gop"][(c, m)].extend(np.mean(parts, axis=0))
            for L in LAYERS:
                vals = -np.array([c for c, p in zip(costs[L], dp) if p is not None])
                ph["dtw"][L].extend(vals)
                if dstats:
                    st, glob = dstats[L]
                    ms = [st.get(p, glob) for p in phones]
                    ph["dtwc"][("shift", L)].extend(vals - np.array([m for m, _ in ms]))
                    ph["dtwc"][("z", L)].extend((vals - np.array([m for m, _ in ms])) / np.array([sd for _, sd in ms]))
            if w["arpa"]:
                mapped, _ = align_phone.reconcile_phones(w["arpa"], phones, [float(i) for i in range(len(phones))])
                for g, a in zip(mapped, w["phone_acc"]):
                    if g is not None:
                        item_idx.append(n + int(g)); item_y.append(a); item_spk.append(r["speaker"])
            word_off.append(n); word_utt.append(idx); word_acc.append(w["accuracy"])
            n += len(phones)
        utt_acc[idx], utt_tot[idx], utt_spk[idx] = r["accuracy"], r["total"], r["speaker"]
    word_off.append(n)
    for k in ph["gop"]:
        ph["gop"][k] = np.array(ph["gop"][k])
    for k in ph["dtw"]:
        ph["dtw"][k] = np.array(ph["dtw"][k])
    for k in ph.get("dtwc", {}):
        ph["dtwc"][k] = np.array(ph["dtwc"][k])
    print(f"{n} espeak phones, {len(word_acc)} words, {len(item_y)} reconciled phone items; skipped {skipped}")
    return ph, np.array(word_off), np.array(word_utt), np.array(word_acc), np.array(item_idx), \
        np.array(item_y), np.array(item_spk), utt_acc, utt_tot, utt_spk


def main():
    ph, woff, wutt, wacc, iidx, iy, ispk, uacc, utot, uspk = build()
    nw = len(wacc)
    w_of_phone = np.repeat(np.arange(nw), np.diff(woff))
    w_spk = np.array([uspk[u] for u in wutt])
    utts = sorted(set(wutt.tolist()))
    u_pos = {u: i for i, u in enumerate(utts)}
    utt_of_word = np.array([u_pos[u] for u in wutt])
    u_spk = np.array([uspk[u] for u in utts]); u_y = {"utt_acc": np.array([uacc[u] for u in utts]),
                                                     "utt_tot": np.array([utot[u] for u in utts])}
    speakers = sorted(set(u_spk))
    random.Random(0).shuffle(speakers)
    fold = {s: i % 2 for i, s in enumerate(speakers)}
    f_phone_item = np.array([fold[s] for s in ispk]); f_word = np.array([fold[s] for s in w_spk])
    f_utt = np.array([fold[s] for s in u_spk])

    def pool(score, how):
        """word scores from an espeak-phone score vector."""
        if how == "mean":
            return np.add.reduceat(score, woff[:-1]) / np.diff(woff)
        return np.array([np.sort(score[woff[i]:woff[i + 1]])[:2].mean() for i in range(nw)])

    def utt_scores(score):
        num = np.bincount(utt_of_word, weights=np.add.reduceat(score, woff[:-1]), minlength=len(utts))
        den = np.bincount(utt_of_word, weights=np.diff(woff), minlength=len(utts))
        return num / den

    def stats_on(vec, mask_fn):
        return vec.mean(), vec.std() + 1e-12

    def phone_mask(k):
        # espeak phones belonging to fold-k speakers
        return np.array([fold[s] == k for s in w_spk])[w_of_phone]

    def fused_score(gcfg, L, method, ref_mask):
        g, d = ph["gop"][gcfg], ph["dtw"][L]
        gm, gs_ = g[ref_mask].mean(), g[ref_mask].std() + 1e-12
        dm, ds = d[ref_mask].mean(), d[ref_mask].std() + 1e-12
        zg, zd = (g - gm) / gs_, (d - dm) / ds
        return {"gop": zg, "dtw": zd, "fused": (zg + zd) / 2}[method]

    levels = ("phone", "word", "utt_acc", "utt_tot")

    def level_r(score, lvl, fold_k, how="mean"):
        if lvl == "phone":
            m = f_phone_item == fold_k
            return np.corrcoef(score[iidx[m]], iy[m])[0, 1]
        if lvl == "word":
            m = f_word == fold_k
            return np.corrcoef(pool(score, how)[m], wacc[m])[0, 1]
        m = f_utt == fold_k
        return np.corrcoef(utt_scores(score)[m], u_y[lvl][m])[0, 1]

    def level_items(score, lvl, fold_k, how="mean"):
        if lvl == "phone":
            m = f_phone_item == fold_k
            return score[iidx[m]], iy[m], ispk[m]
        if lvl == "word":
            m = f_word == fold_k
            return pool(score, how)[m], wacc[m], w_spk[m]
        m = f_utt == fold_k
        return utt_scores(score)[m], u_y[lvl][m], u_spk[m]

    base = ph["gop"][("none", ("xlsr_sf",))]
    grid = list(itertools.product(list(ph["gop"]), LAYERS))
    results = {}
    print("\nheld-out (config chosen on one speaker-half incl. z-stats; reported on the other):")
    for lvl in levels:
        hows = ("mean", "mean_low2") if lvl == "word" else ("mean",)
        pooled = {k: ([], [], []) for k in ("base", "gop", "dtw", "fused")}
        chosen = {}
        for k in (0, 1):
            ref = phone_mask(k)
            for method in ("gop", "dtw", "fused"):
                best, best_r = None, -9
                for gcfg, L in grid:
                    if method == "dtw" and gcfg != grid[0][0]:
                        continue  # DTW-only doesn't depend on the GOP config
                    if method == "gop" and L != LAYERS[0]:
                        continue  # GOP-only doesn't depend on the layer
                    for how in hows:
                        r_ = level_r(fused_score(gcfg, L, method, ref), lvl, k, how)
                        if r_ > best_r:
                            best, best_r = (gcfg, L, how), r_
                chosen.setdefault(method, []).append((str(best[0][0]) + "/" + "+".join(best[0][1]), best[1], best[2]))
                gcfg, L, how = best
                s = fused_score(gcfg, L, method, ref)
                items = level_items(s, lvl, 1 - k, how)
                for dst, src in zip(pooled[method], items):
                    dst.extend(src)
            items = level_items(base, lvl, 1 - k, "mean")
            for dst, src in zip(pooled["base"], items):
                dst.extend(src)
        res = {}
        for name in ("base", "gop", "dtw", "fused"):
            res[name] = gs.cluster_bootstrap_pearson(*pooled[name])
        print(f"\n{lvl}: " + "  ".join(f"{n} {res[n]['r']:.3f}" for n in ("base", "gop", "dtw", "fused")))
        print(f"   fused CI [{res['fused']['ci_lo']:.3f},{res['fused']['ci_hi']:.3f}]   chosen(fused)={chosen['fused']}")
        for a, b in (("fused", "base"), ("fused", "gop"), ("fused", "dtw")):
            if len(pooled[a][0]) == len(pooled[b][0]):
                p = gs.cluster_bootstrap_paired_diff(pooled[a][0], pooled[b][0], pooled[a][1], pooled[a][2])
                print(f"   {a} - {b}: {p['diff']:+.3f} [{p['diff_ci_lo']:+.3f},{p['diff_ci_hi']:+.3f}] sig={p['significant']}")
        results[lvl] = {n: res[n] for n in res} | {"chosen": {k: [str(c) for c in v] for k, v in chosen.items()}}
    json.dump(results, open("results/so762_phone_fusion.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
