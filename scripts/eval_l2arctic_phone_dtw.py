#!/usr/bin/env python
"""Phone-level DTW-SSL on L2-ARCTIC (PLAN.md section 5m): does cutting the
DTW alignment into per-phone regions give a phone score that (a) is
informative on its own and (b) adds to xlsr-53 GOP at the *phone* level --
the granularity where this project's zero-shot numbers trail trained GOP
the most (so762 phone 0.472 vs. GOPT 0.612)?

Method per learner utterance and native template (CMU ARCTIC bdl/slt, the
same prompt): (1) CTC-force-align the *template* audio against the prompt
words (clean audio, so boundaries are reliable) to get each espeak phone's
frame span; (2) DTW the learner's WavLM embeddings against the template's
over the whole utterance; (3) each template phone's cost = mean cosine
distance over the DTW-path cells inside its span (`dtw_ssl.phone_costs`);
(4) average over templates. The learner-side comparison score is xlsr-53
GOP-SF on the same phones. Phones are reconciled to L2-ARCTIC's canonical
ARPABET phones exactly as scripts/eval_l2arctic_phone.py does, and scored
against its binary expert correct/error tags (point-biserial r, speaker-
cluster bootstrap). Words where the learner-side and template-side espeak
phone sequences differ are skipped and counted.

Fusion is an unweighted mean of the two phone scores after z-scoring each
over the items being evaluated (label-free; states the score statistics
used, not labels).

Usage:
    OMP_NUM_THREADS=4 python scripts/eval_l2arctic_phone_dtw.py --per-speaker 40 --out results/l2arctic_phone_dtw.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import align_phone, dtw_ssl, stats as gs
from eval_l2arctic_phone import LANG_BY_SPEAKER, SPEAKERS_BY_LANG, _WORD_RE, parse_textgrid, words_and_phones
from analyze_so762_full import load_jsonl, native_stats
from analyze_so762_phone_fusion import native_dtw_stats

NATIVES = ("bdl", "slt")
_TEMPLATE_CACHE = {}


def embed_layer(path, layer):
    key = (str(path), layer)
    if key not in _TEMPLATE_CACHE:
        s, sr = sf.read(str(path), dtype="float32")
        _TEMPLATE_CACHE[key] = dtw_ssl.embed_layers(s, [layer], sr=sr)[layer]
    return _TEMPLATE_CACHE[key]


def template_spans(path, texts):
    """[[(phone, (start,end)) per phone] per word] from a CTC forced alignment of the template (lv-60)."""
    key = ("spans", str(path), tuple(texts))
    if key not in _TEMPLATE_CACHE:
        s, sr = sf.read(str(path), dtype="float32")
        res = align_phone.align_words_gop(s, texts, sr=sr)
        _TEMPLATE_CACHE[key] = [[(p["phone"], p["span"]) if p else (None, None) for p in pg]
                                for pg in res["phone_gop"]]
    return _TEMPLATE_CACHE[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/L2-ARCTIC/all")
    ap.add_argument("--native-root", default="/data/CMU-ARCTIC-native")
    ap.add_argument("--speakers", nargs="+", default=[s for spks in SPEAKERS_BY_LANG.values() for s in spks[:2]])
    ap.add_argument("--per-speaker", type=int, default=40)
    ap.add_argument("--layer", default="last", help="WavLM layer: int or 'last'")
    ap.add_argument("--native-gop", default="results/native_gop.jsonl",
                    help="native per-phone GOP stats for label-free calibration (PLAN.md 5m); '' to skip")
    ap.add_argument("--native-dtw", default="results/native_dtw_native.jsonl",
                    help="native-vs-native per-phone DTW stats for calibrating the DTW cost; '' to skip")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    nstats = native_stats(load_jsonl(args.native_gop), "xlsr_sf") if args.native_gop else None
    layer = args.layer if args.layer == "last" else int(args.layer)
    dstats = native_dtw_stats(args.native_dtw, str(layer)) if args.native_dtw else None

    import torch
    torch.set_num_threads(4)

    root, nroot = Path(args.data_root), Path(args.native_root)
    spk_dirs = {p.name: p for p in root.glob("*/*") if (p / "annotation").is_dir()}
    rows = {k: [] for k in ("gop_sf", "gop_post", "dtw", "correct", "speaker", "gop_shift", "gop_z", "dtw_shift", "dtw_z")}
    n_words = n_skip_mismatch = n_skip_other = 0
    t0 = time.time()

    for spk in args.speakers:
        anns = sorted((spk_dirs[spk] / "annotation").glob("*.TextGrid"))[: args.per_speaker]
        for ann in anns:
            wav = ann.parents[1] / "wav" / (ann.stem + ".wav")
            words = [w for w in words_and_phones(parse_textgrid(ann))
                     if _WORD_RE.fullmatch(w["text"].lower()) and w["phones"]]
            if not wav.exists() or not words:
                continue
            texts = [w["text"].lower() for w in words]
            s, sr = sf.read(str(wav), dtype="float32")
            learner = dtw_ssl.embed_layers(s, [layer], sr=sr)[layer]
            g_sf = align_phone.align_words_gop_sf(s, texts, sr=sr, model_id=align_phone.TORCH_MODEL_REPO)
            g_po = align_phone.align_words_gop(s, texts, sr=sr, model_id=align_phone.TORCH_MODEL_REPO)

            per_template = []  # [template][word] -> [(phone, cost)]
            for nat in NATIVES:
                npath = nroot / nat / (ann.stem + ".wav")
                if not npath.exists():
                    continue
                spans = template_spans(npath, texts)
                flat = [sp for w in spans for (_p, sp) in w if sp is not None]
                costs = dtw_ssl.phone_costs(learner, embed_layer(npath, layer), flat)
                it = iter(costs)
                per_template.append([[(p, next(it) if sp is not None else None) for (p, sp) in w] for w in spans])
            if not per_template:
                continue

            for j, w in enumerate(words):
                n_words += 1
                sf_ph = [(p["phone"], p["gop"]) for p in g_sf["phone_gop"][j] if p]
                po_ph = [(p["phone"], p["gop"]) for p in g_po["phone_gop"][j] if p]
                dtw_by_t = [[(p, c) for (p, c) in t[j] if p is not None and c is not None] for t in per_template]
                phones = [p for p, _ in sf_ph]
                if not sf_ph or [p for p, _ in po_ph] != phones or any([p for p, _ in d] != phones for d in dtw_by_t):
                    n_skip_mismatch += 1
                    continue
                dtw_cost = [float(np.mean([d[k][1] for d in dtw_by_t])) for k in range(len(phones))]
                canon = [p[0] for p in w["phones"]]
                tags = [p[1] for p in w["phones"]]
                r_sf, _ = align_phone.reconcile_phones(canon, phones, [g for _, g in sf_ph])
                r_po, _ = align_phone.reconcile_phones(canon, phones, [g for _, g in po_ph])
                r_dt, _ = align_phone.reconcile_phones(canon, phones, [-c for c in dtw_cost])
                shifts, zs = [], []
                for ph_, g_ in sf_ph:
                    m_, sd_, _v = (nstats[0].get(ph_) or nstats[1]) if nstats else (0.0, 1.0, None)
                    shifts.append(g_ - m_); zs.append((g_ - m_) / sd_)
                dsh, dz = [], []
                for ph_, c_ in zip(phones, dtw_cost):
                    m_, sd_ = ((dstats[0].get(ph_) or dstats[1]) if dstats else (0.0, 1.0))
                    dsh.append(-c_ - m_); dz.append((-c_ - m_) / sd_)
                r_dsh, _ = align_phone.reconcile_phones(canon, phones, dsh)
                r_dz, _ = align_phone.reconcile_phones(canon, phones, dz)
                r_sh, _ = align_phone.reconcile_phones(canon, phones, shifts)
                r_z, _ = align_phone.reconcile_phones(canon, phones, zs)
                for a, b, c_, tag, sh_, zz_, ds_, dz_ in zip(r_sf, r_po, r_dt, tags, r_sh, r_z, r_dsh, r_dz):
                    if a is None or b is None or c_ is None:
                        continue
                    rows["gop_shift"].append(sh_); rows["gop_z"].append(zz_)
                    rows["dtw_shift"].append(ds_); rows["dtw_z"].append(dz_)
                    rows["gop_sf"].append(a); rows["gop_post"].append(b); rows["dtw"].append(c_)
                    rows["correct"].append(1 if tag == "correct" else 0); rows["speaker"].append(spk)
        print(f"  {spk}: {len(rows['correct'])} phones so far ({time.time()-t0:.0f}s)", file=sys.stderr)

    z = lambda v: (np.asarray(v) - np.mean(v)) / np.std(v)
    rows["fused"] = list((z(rows["gop_sf"]) + z(rows["dtw"])) / 2)
    rows["fused_shift"] = list((z(rows["gop_shift"]) + z(rows["dtw"])) / 2)
    rows["fused_z"] = list((z(rows["gop_z"]) + z(rows["dtw"])) / 2)
    rows["fused_cal"] = list((z(rows["gop_z"]) + z(rows["dtw_z"])) / 2)      # both calibrated (z)
    rows["fused_cal_shift"] = list((z(rows["gop_shift"]) + z(rows["dtw_shift"])) / 2)
    y, c = rows["correct"], rows["speaker"]
    res = {"n_phones": len(y), "n_words": n_words, "n_words_skipped_mismatch": n_skip_mismatch,
           "n_speakers": len(set(c)), "layer": args.layer, "elapsed_s": round(time.time() - t0, 1)}
    for k in ("gop_post", "gop_sf", "gop_shift", "gop_z", "dtw", "dtw_shift", "dtw_z", "fused", "fused_shift",
              "fused_z", "fused_cal", "fused_cal_shift"):
        res[k] = gs.cluster_bootstrap_pearson(rows[k], y, c)
    for a, b in (("dtw", "gop_sf"), ("fused", "gop_sf"), ("fused", "dtw"), ("gop_shift", "gop_sf"),
                 ("gop_z", "gop_sf"), ("fused_shift", "fused"), ("fused_shift", "gop_sf"),
                 ("dtw_shift", "dtw"), ("dtw_z", "dtw"), ("fused_cal", "fused_z"), ("fused_cal", "gop_z"),
                 ("fused_cal", "gop_sf")):
        res[f"{a}_minus_{b}"] = gs.cluster_bootstrap_paired_diff(rows[a], rows[b], y, c)
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        json.dump(rows, open(args.out + ".arrays.json", "w"))


if __name__ == "__main__":
    main()
