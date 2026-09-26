#!/usr/bin/env python
"""Multi-granularity context for the phone score (PLAN.md section 5m).
Trained GOP models (GOPT and relatives) predict each phone's accuracy with
access to its word and utterance -- a speaker who is doing badly overall
makes any given phone more likely to be wrong. This tests a label-free,
unweighted-family analogue on the fused zero-shot phone score s:

    phone' = z(s) + lam * z(word mean of s) + mu * z(utterance mean of s)
    word'  = z(word mean of s) + mu * z(utterance mean of s)

with lam, mu in a small grid {0, .25, .5, 1}. Held-out exactly as in
scripts/analyze_so762_phone_fusion.py (speaker-disjoint halves; the
(lam, mu, calibration, layer) choice and all z-stats come from one half,
scores are evaluated on the other). No labels enter except that choice.

Usage:
    python scripts/analyze_so762_context.py
"""
import itertools
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import stats as gs
import analyze_so762_phone_fusion as F

GRID = (0.0, 0.25, 0.5, 1.0)


def main():
    ph, woff, wutt, wacc, iidx, iy, ispk, uacc, utot, uspk = F.build()
    nw = len(wacc)
    w_of_phone = np.repeat(np.arange(nw), np.diff(woff))
    w_spk = np.array([uspk[u] for u in wutt])
    utts = sorted(set(wutt.tolist()))
    upos = {u: i for i, u in enumerate(utts)}
    utt_of_word = np.array([upos[u] for u in wutt])
    utt_of_phone = utt_of_word[w_of_phone]
    u_spk = np.array([uspk[u] for u in utts])
    speakers = sorted(set(u_spk)); random.Random(0).shuffle(speakers)
    fold = {s: i % 2 for i, s in enumerate(speakers)}
    f_ph = np.array([fold[s] for s in w_spk])[w_of_phone]
    f_item = np.array([fold[s] for s in ispk])
    f_word = np.array([fold[s] for s in w_spk])
    wlen = np.diff(woff)

    def word_mean(s):
        return np.add.reduceat(s, woff[:-1]) / wlen

    def utt_mean(s):
        return (np.bincount(utt_of_word, weights=np.add.reduceat(s, woff[:-1]), minlength=len(utts))
                / np.bincount(utt_of_word, weights=wlen, minlength=len(utts)))

    def zed(v, ref):
        return (v - v[ref].mean()) / (v[ref].std() + 1e-12)

    def build_score(gcfg, L, ref_ph, ref_w, ref_u):
        g, d = ph["gop"][gcfg], ph["dtw"][L]
        s = (zed(g, ref_ph) + zed(d, ref_ph)) / 2
        return s

    def phone_ctx(s, lam, mu, ref_ph, ref_w, ref_u):
        wm, um = word_mean(s), utt_mean(s)
        return zed(s, ref_ph) + lam * zed(wm, ref_w)[w_of_phone] + mu * zed(um, ref_u)[utt_of_phone]

    def word_ctx(s, mu, ref_w, ref_u):
        wm, um = word_mean(s), utt_mean(s)
        return zed(wm, ref_w) + mu * zed(um, ref_u)[utt_of_word]

    grid = list(itertools.product(list(ph["gop"]), F.LAYERS))
    for lvl in ("phone", "word"):
        pooled = {k: ([], [], []) for k in ("base", "fused", "ctx")}
        for k in (0, 1):
            ref_ph = f_ph == k
            ref_w = f_word == k
            ref_u = np.array([fold[s] == k for s in u_spk])
            sel_item, tst_item = f_item == k, f_item != k
            sel_word, tst_word = f_word == k, f_word != k
            best, best_r = None, -9
            for gcfg, L in grid:
                s = build_score(gcfg, L, ref_ph, ref_w, ref_u)
                for lam, mu in (itertools.product(GRID, GRID) if lvl == "phone" else [(0, m) for m in GRID]):
                    if lvl == "phone":
                        v = phone_ctx(s, lam, mu, ref_ph, ref_w, ref_u)
                        r = np.corrcoef(v[iidx[sel_item]], iy[sel_item])[0, 1]
                    else:
                        v = word_ctx(s, mu, ref_w, ref_u)
                        r = np.corrcoef(v[sel_word], wacc[sel_word])[0, 1]
                    if r > best_r:
                        best, best_r = (gcfg, L, lam, mu), r
            gcfg, L, lam, mu = best
            s = build_score(gcfg, L, ref_ph, ref_w, ref_u)
            print(f"  {lvl} fold {k}: chose {gcfg[0]}/{'+'.join(gcfg[1])} layer {L} lam={lam} mu={mu}")
            base = ph["gop"][("none", ("xlsr_sf",))]
            if lvl == "phone":
                cands = {"base": base, "fused": s, "ctx": phone_ctx(s, lam, mu, ref_ph, ref_w, ref_u)}
                for name, v in cands.items():
                    for dst, src in zip(pooled[name], (v[iidx[tst_item]], iy[tst_item], ispk[tst_item])):
                        dst.extend(src)
            else:
                cands = {"base": word_mean(base), "fused": word_mean(s), "ctx": word_ctx(s, mu, ref_w, ref_u)}
                for name, v in cands.items():
                    for dst, src in zip(pooled[name], (v[tst_word], wacc[tst_word], w_spk[tst_word])):
                        dst.extend(src)
        print(f"{lvl}:", "  ".join(f"{n} {gs.cluster_bootstrap_pearson(*pooled[n])['r']:.3f}" for n in pooled))
        e = gs.cluster_bootstrap_pearson(*pooled["ctx"])
        print(f"   ctx CI [{e['ci_lo']:.3f},{e['ci_hi']:.3f}]")
        for a, b in (("ctx", "fused"), ("ctx", "base")):
            p = gs.cluster_bootstrap_paired_diff(pooled[a][0], pooled[b][0], pooled[a][1], pooled[a][2])
            print(f"   {a} - {b}: {p['diff']:+.3f} [{p['diff_ci_lo']:+.3f},{p['diff_ci_hi']:+.3f}] sig={p['significant']}")


if __name__ == "__main__":
    main()
