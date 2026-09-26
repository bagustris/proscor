#!/usr/bin/env python
"""Offline zero-shot-improvement probes on the collected speechocean762 data
(PLAN.md section 5m). Reads `results/so762_full.jsonl`
(scripts/collect_so762_full.py) and, for calibration,
`results/native_gop.jsonl` (scripts/collect_native_gop.py); no model is run.

Every scorer is defined at the *phone* level -- a function from a word
record to (espeak phones, per-phone score) -- and word and utterance scores
are aggregated from those, so all three levels are consistent and the
baseline (xlsr-53 GOP-SF, plain means) can be checked against PLAN.md
section 5i's full-corpus numbers before trusting any variant.

Labels/levels (same as section 5a): phone = reconciled ARPABET phone vs.
`phones-accuracy` (graded 0-2); word = mean phone score vs. word
`accuracy`; utterance = mean of word scores vs. utterance `accuracy` and
`total`. Pearson r with speaker-cluster bootstrap CIs; paired differences
vs. the baseline use `proscor.stats.cluster_bootstrap_paired_diff`.

Label discipline: nothing here fits to so762 labels. Native calibration
uses native audio only; fusion/ensembles are unweighted; z-scoring uses
score statistics only. Any scalar hyper-parameter chosen by looking at
labels is flagged where used.

Usage:
    python scripts/analyze_so762_full.py --data results/so762_full.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proscor import align_phone, stats as gs


def load_jsonl(path):
    return [r for r in (json.loads(l) for l in open(path)) if not r.get("failed") and not r.get("skipped")]


def clean(word, variant):
    """[(phone, gop)] with None entries dropped (same filter the eval
    scripts use before reconcile_phones)."""
    d = word[variant]
    return [(p, g) for p, g in zip(d["phones"], d["gops"]) if p is not None and g is not None]


# ---- phone-level scorers: word record -> [(phone, score)] -----------------

def scorer_plain(variant):
    return lambda w: clean(w, variant)


def scorer_ensemble(v1, v2):
    """Unweighted mean of two variants' per-phone GOP, only where both
    models saw the same phone sequence (same espeak call, so nearly always)."""
    def f(w):
        a, b = clean(w, v1), clean(w, v2)
        if [p for p, _ in a] != [p for p, _ in b]:
            return None
        return [(p, (x + y) / 2) for (p, x), (_, y) in zip(a, b)]
    return f


def native_stats(native_recs, variant, min_n=20):
    per = defaultdict(list)
    for r in native_recs:
        for w in r["words"]:
            for p, g in zip(w[variant]["phones"], w[variant]["gops"]):
                if p is not None and g is not None:
                    per[p].append(g)
    allg = np.array([g for v in per.values() for g in v])
    st = {}
    for p, v in per.items():
        v = np.sort(np.array(v))
        st[p] = (v.mean(), v.std() + 1e-6, v) if len(v) >= min_n else None
    return st, (allg.mean(), allg.std() + 1e-6, np.sort(allg))


def scorer_calibrated(variant, stats, mode):
    st, glob = stats

    def f(w):
        out = []
        for p, g in clean(w, variant):
            m, s, v = st.get(p) or glob
            if mode == "shift":
                out.append((p, g - m))
            elif mode == "z":
                out.append((p, (g - m) / s))
            elif mode == "pct":  # fraction of native instances of this phone scoring <= g
                out.append((p, float(np.searchsorted(v, g, side="right")) / len(v)))
        return out
    return f


# ---- aggregation ---------------------------------------------------------

def word_agg(scores, how):
    v = np.array([s for _, s in scores])
    if how == "mean":
        return float(v.mean())
    if how == "min":
        return float(v.min())
    if how == "mean_low2":
        return float(np.sort(v)[:2].mean())
    raise ValueError(how)


def evaluate(recs, scorer, word_how="mean", utt_how="mean_word"):
    ph_s, ph_y, ph_c = [], [], []
    w_s, w_y, w_c = [], [], []
    u_s, u_acc, u_tot, u_c = [], [], [], []
    for r in recs:
        word_scores, all_ph = [], []
        for w in r["words"]:
            sc = scorer(w)
            if not sc:
                continue
            ws = word_agg(sc, word_how)
            word_scores.append(ws)
            all_ph.extend(s for _, s in sc)
            w_s.append(ws); w_y.append(w["accuracy"]); w_c.append(r["speaker"])
            if w["arpa"]:
                gops, _ops = align_phone.reconcile_phones(w["arpa"], [p for p, _ in sc], [s for _, s in sc])
                for g, a in zip(gops, w["phone_acc"]):
                    if g is not None:
                        ph_s.append(g); ph_y.append(a); ph_c.append(r["speaker"])
        if not word_scores:
            continue
        if utt_how == "mean_word":
            u = float(np.mean(word_scores))
        elif utt_how == "mean_phone":
            u = float(np.mean(all_ph))
        elif utt_how == "min_word":
            u = float(np.min(word_scores))
        elif utt_how == "q25_word":
            u = float(np.quantile(word_scores, 0.25))
        u_s.append(u); u_acc.append(r["accuracy"]); u_tot.append(r["total"]); u_c.append(r["speaker"])
    return {"phone": (ph_s, ph_y, ph_c), "word": (w_s, w_y, w_c),
            "utt_acc": (u_s, u_acc, u_c), "utt_tot": (u_s, u_tot, u_c)}


def r_of(triple):
    s, y, c = triple
    return gs.cluster_bootstrap_pearson(s, y, c)


def fmt(e):
    return f"{e['r']:.3f} [{e['ci_lo']:.3f},{e['ci_hi']:.3f}]"


def paired(a, b):
    """diff = r(a) - r(b) on identical item lists (a, b are evaluate()
    triples that must have been built from the same items)."""
    (sa, y, c), (sb, y2, _c2) = a, b
    if len(sa) != len(sb) or list(y) != list(y2):
        return None
    return gs.cluster_bootstrap_paired_diff(sa, sb, y, c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/so762_full.jsonl")
    ap.add_argument("--native", default="results/native_gop.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = load_jsonl(args.data)
    print(f"{len(recs)} utterances, {len({r['speaker'] for r in recs})} speakers")

    base = evaluate(recs, scorer_plain("xlsr_sf"))
    variants = {"baseline xlsr_sf": base}
    for v in ("xlsr_post", "lv60_sf", "lv60_post"):
        variants[v] = evaluate(recs, scorer_plain(v))
    variants["ens xlsr_sf+lv60_sf"] = evaluate(recs, scorer_ensemble("xlsr_sf", "lv60_sf"))
    variants["ens xlsr_post+xlsr_sf"] = evaluate(recs, scorer_ensemble("xlsr_post", "xlsr_sf"))
    for how in ("min", "mean_low2"):
        variants[f"word agg={how}"] = evaluate(recs, scorer_plain("xlsr_sf"), word_how=how)
    for how in ("mean_phone", "min_word", "q25_word"):
        variants[f"utt agg={how}"] = evaluate(recs, scorer_plain("xlsr_sf"), utt_how=how)

    if Path(args.native).exists():
        nat = load_jsonl(args.native)
        print(f"native calibration utterances: {len(nat)}")
        for var in ("xlsr_sf", "xlsr_post"):
            stats = native_stats(nat, var)
            for mode in ("shift", "z", "pct"):
                variants[f"calib {var} {mode}"] = evaluate(recs, scorer_calibrated(var, stats, mode))

    summary = {}
    for name, ev in variants.items():
        row = {lvl: r_of(ev[lvl]) for lvl in ("phone", "word", "utt_acc", "utt_tot")}
        row["n"] = {lvl: len(ev[lvl][0]) for lvl in row}
        summary[name] = row
        print(f"{name:26s} phone {fmt(row['phone'])}  word {fmt(row['word'])}  "
              f"utt-acc {fmt(row['utt_acc'])}  utt-tot {fmt(row['utt_tot'])}")
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
