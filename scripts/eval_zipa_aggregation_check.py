#!/usr/bin/env python
"""One-off check (not a permanent eval script) for PLAN.md section 5k's
dilution hypothesis: for ZIPA's multi-token phones (1-3 IPA-character
tokens per phone -- see `_word_phone_spans`), does averaging GOP-SF's
per-token scores dilute the signal compared to taking the *worst*
(min, i.e. most negative) token in the span? Computes both aggregations
from the same GOP-SF run (no extra inference) and reports each against
phone accuracy.

Usage:
    python scripts/eval_zipa_aggregation_check.py --limit 250
"""
import argparse
import io
import sys
import time
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


def zipa_gop_sf_both_aggregations(samples, words, sr):
    """Mirrors align_phone.align_words_gop_sf's body, but returns phone_gop
    computed both ways (mean, the shipped behavior; min, the dilution
    check) from a single GOP-SF scoring pass."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if sr != align_phone.SAMPLE_RATE:
        from audiokit import resample
        samples = np.asarray(resample(samples, sr, align_phone.SAMPLE_RATE), dtype=np.float32)

    align_phone._session(align_phone.ZIPA_MODEL_REPO, True)
    lp = align_phone._logprobs(samples, align_phone.ZIPA_MODEL_REPO, True)

    flat, spans, phone_texts, phone_spans = align_phone._leading_marker_prefix(), [], [], []
    for w in words:
        phones, toks, local_spans = align_phone._word_phone_spans(w)
        offset = len(flat)
        flat.extend(toks)
        spans.append((offset, len(flat)))
        phone_texts.append(phones)
        phone_spans.append([(offset + s, offset + e) for s, e in local_spans])

    if not flat:
        return [[] for _ in words], [[] for _ in words]

    scores = align_phone.gop_sf(lp, flat, align_phone._BLANK, lp.shape[1])

    phone_gop_mean, phone_gop_min = [], []
    for (start, end), phones, ph_spans in zip(spans, phone_texts, phone_spans):
        if start == end:
            phone_gop_mean.append([]); phone_gop_min.append([])
            continue
        this_mean, this_min = [], []
        for (pstart, pend), phone in zip(ph_spans, phones):
            vals = [scores[k] for k in range(pstart, pend) if scores[k] is not None]
            this_mean.append({"phone": phone, "gop": float(np.mean(vals))} if vals else None)
            this_min.append({"phone": phone, "gop": float(np.min(vals))} if vals else None)
        phone_gop_mean.append(this_mean)
        phone_gop_min.append(this_min)

    return phone_gop_mean, phone_gop_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=250)
    args = ap.parse_args()

    rows = load_split(args.split)[: args.limit]
    print(f"Evaluating {len(rows)} utterances, ZIPA GOP-SF mean- vs min-aggregation ...", file=sys.stderr)

    ph = {"mean": [], "min": [], "acc_binary": [], "acc_graded": [], "speaker": []}
    n_skipped = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        speaker = row["speaker"]
        try:
            pg_mean, pg_min = zipa_gop_sf_both_aggregations(samples, words, sr)
        except Exception as e:
            print(f"  [warn] utt {i} failed: {e}", file=sys.stderr)
            continue

        for j, w in enumerate(row["words"]):
            dataset_phones = w["phones"]
            ph_acc_word = w["phones-accuracy"]
            if not dataset_phones:
                continue
            phones_m = [(p["phone"], p["gop"]) for p in pg_mean[j] if p is not None]
            phones_n = [(p["phone"], p["gop"]) for p in pg_min[j] if p is not None]
            if not phones_m or [p for p, _ in phones_m] != [p for p, _ in phones_n]:
                n_skipped += 1
                continue
            espeak_phones = [p for p, _ in phones_m]
            rec_m, _ = align_phone.reconcile_phones(dataset_phones, espeak_phones, [g for _, g in phones_m])
            rec_n, _ = align_phone.reconcile_phones(dataset_phones, espeak_phones, [g for _, g in phones_n])
            for gm, gn, a in zip(rec_m, rec_n, ph_acc_word):
                if gm is None or gn is None:
                    continue
                ph["mean"].append(gm); ph["min"].append(gn)
                ph["acc_binary"].append(1 if a >= 2 else 0)
                ph["acc_graded"].append(a)
                ph["speaker"].append(speaker)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    r_mean = gopstats.cluster_bootstrap_pearson(ph["mean"], ph["acc_graded"], ph["speaker"])
    r_min = gopstats.cluster_bootstrap_pearson(ph["min"], ph["acc_graded"], ph["speaker"])
    diff = gopstats.cluster_bootstrap_paired_diff(ph["min"], ph["mean"], ph["acc_graded"], ph["speaker"])
    print(f"n_phones={len(ph['mean'])} n_skipped={n_skipped}")
    print(f"mean-aggregation r={r_mean['r']:.4f} (ci {r_mean['ci_lo']:.4f},{r_mean['ci_hi']:.4f})")
    print(f"min-aggregation  r={r_min['r']:.4f} (ci {r_min['ci_lo']:.4f},{r_min['ci_hi']:.4f})")
    print(f"diff (min-mean)={diff['diff']:.4f} significant={diff['significant']}")


if __name__ == "__main__":
    main()
