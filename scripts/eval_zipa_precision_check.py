#!/usr/bin/env python
"""One-off check (not a permanent eval script) for PLAN.md section 5k:
does ZIPA's int8 quantization (the default `use_int8=True` every other
ZIPA eval in this project uses) explain part of the phone-level gap
against xlsr-53 (which is always loaded fp32 -- the torch backend ignores
`use_int8`)? Scores the same utterances with ZIPA under both precisions
and reports each against phone accuracy, no xlsr-53 involved (faster: one
model, not two).

Usage:
    python scripts/eval_zipa_precision_check.py --limit 250
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=250)
    args = ap.parse_args()

    rows = load_split(args.split)[: args.limit]
    print(f"Evaluating {len(rows)} utterances, ZIPA int8 vs fp32 ...", file=sys.stderr)

    ph = {"int8": [], "fp32": [], "acc_binary": [], "acc_graded": [], "speaker": []}
    n_skipped = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        speaker = row["speaker"]
        try:
            res_int8 = align_phone.align_words_gop_sf(samples, words, sr=sr,
                                                        model_id=align_phone.ZIPA_MODEL_REPO, use_int8=True)
            res_fp32 = align_phone.align_words_gop_sf(samples, words, sr=sr,
                                                        model_id=align_phone.ZIPA_MODEL_REPO, use_int8=False)
        except Exception as e:
            print(f"  [warn] utt {i} failed: {e}", file=sys.stderr)
            continue

        for j, w in enumerate(row["words"]):
            dataset_phones = w["phones"]
            ph_acc_word = w["phones-accuracy"]
            if not dataset_phones:
                continue
            pg_i, pg_f = res_int8["phone_gop"][j], res_fp32["phone_gop"][j]
            phones_i = [(p["phone"], p["gop"]) for p in pg_i if p is not None]
            phones_f = [(p["phone"], p["gop"]) for p in pg_f if p is not None]
            if not phones_i or not phones_f or [p for p, _ in phones_i] != [p for p, _ in phones_f]:
                n_skipped += 1
                continue
            espeak_phones = [p for p, _ in phones_i]
            rec_i, ops_i = align_phone.reconcile_phones(dataset_phones, espeak_phones, [g for _, g in phones_i])
            rec_f, ops_f = align_phone.reconcile_phones(dataset_phones, espeak_phones, [g for _, g in phones_f])
            for gi, gf, a in zip(rec_i, rec_f, ph_acc_word):
                if gi is None or gf is None:
                    continue
                ph["int8"].append(gi); ph["fp32"].append(gf)
                ph["acc_binary"].append(1 if a >= 2 else 0)
                ph["acc_graded"].append(a)
                ph["speaker"].append(speaker)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    r_int8 = gopstats.cluster_bootstrap_pearson(ph["int8"], ph["acc_graded"], ph["speaker"])
    r_fp32 = gopstats.cluster_bootstrap_pearson(ph["fp32"], ph["acc_graded"], ph["speaker"])
    diff = gopstats.cluster_bootstrap_paired_diff(ph["fp32"], ph["int8"], ph["acc_graded"], ph["speaker"])
    print(f"n_phones={len(ph['int8'])} n_skipped={n_skipped}")
    print(f"int8 r={r_int8['r']:.4f} (ci {r_int8['ci_lo']:.4f},{r_int8['ci_hi']:.4f})")
    print(f"fp32 r={r_fp32['r']:.4f} (ci {r_fp32['ci_lo']:.4f},{r_fp32['ci_hi']:.4f})")
    print(f"diff (fp32-int8)={diff['diff']:.4f} significant={diff['significant']}")


if __name__ == "__main__":
    main()
