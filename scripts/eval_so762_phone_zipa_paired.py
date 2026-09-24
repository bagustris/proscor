#!/usr/bin/env python
"""Paired ZIPA-vs-xlsr-53 acoustic-model significance test on
speechocean762 phone-level GOP (PLAN.md section 5k) -- the direct test of
whether ZIPA's IPAPack++ pretraining (17k phone-labeled + 11.8k
pseudo-labeled hours, 88 languages -- an order of magnitude more
phone-labeled data than xlsr-53-espeak-cv-ft) improves on xlsr-53
(section 5i's winner) the same way xlsr-53 improved on lv-60. Mirrors
scripts/eval_so762_model_paired.py's design exactly, just with a third
backend and both scoring formulas (posterior-deficit and GOP-SF) scored
in the same loop so one run answers both "does ZIPA help posterior-deficit"
and "does ZIPA help GOP-SF" with paired, speaker-cluster-bootstrapped
significance.

Phone-level pairing: same reconcile_phones-identity-invariance property as
eval_so762_model_paired.py -- both models phonemize with the same espeak
call, so their filtered phone-identity lists for a word normally match;
they can disagree when a phone's IPA characters are in one model's vocab
but not the other's (ZIPA's 127-symbol set vs. the other models' 392),
in which case the word is skipped from the phone-level comparison for
that scoring formula and counted separately.

Usage:
    python scripts/eval_so762_phone_zipa_paired.py --out results/so762_phone_zipa_paired.json
    python scripts/eval_so762_phone_zipa_paired.py --limit 150
"""
import argparse
import io
import json
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


def run(rows: list, progress_every: int = 50) -> dict:
    word = {k: [] for k in ("xlsr_post", "zipa_post", "xlsr_sf", "zipa_sf", "acc", "speaker")}
    ph = {k: [] for k in ("xlsr_post", "zipa_post", "xlsr_sf", "zipa_sf",
                           "acc_binary", "acc_graded", "speaker")}
    n_skipped = {"post": 0, "sf": 0}
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        speaker = row["speaker"]
        try:
            xlsr_post = align_phone.align_words_gop(samples, words, sr=sr, model_id=align_phone.TORCH_MODEL_REPO)
            zipa_post = align_phone.align_words_gop(samples, words, sr=sr, model_id=align_phone.ZIPA_MODEL_REPO)
            xlsr_sf = align_phone.align_words_gop_sf(samples, words, sr=sr, model_id=align_phone.TORCH_MODEL_REPO)
            zipa_sf = align_phone.align_words_gop_sf(samples, words, sr=sr, model_id=align_phone.ZIPA_MODEL_REPO)
        except Exception as e:
            print(f"  [warn] utt {i} failed: {e}", file=sys.stderr)
            continue

        for j, w in enumerate(row["words"]):
            wg_xp, wg_zp = xlsr_post["word_gop"][j], zipa_post["word_gop"][j]
            wg_xs, wg_zs = xlsr_sf["word_gop"][j], zipa_sf["word_gop"][j]
            if None not in (wg_xp, wg_zp, wg_xs, wg_zs):
                word["xlsr_post"].append(wg_xp["gop"]); word["zipa_post"].append(wg_zp["gop"])
                word["xlsr_sf"].append(wg_xs["gop"]); word["zipa_sf"].append(wg_zs["gop"])
                word["acc"].append(w["accuracy"]); word["speaker"].append(speaker)

            dataset_phones = w["phones"]
            ph_acc_word = w["phones-accuracy"]
            if not dataset_phones:
                continue

            for label, pg_x, pg_z, d_x, d_z in (
                ("post", xlsr_post["phone_gop"][j], zipa_post["phone_gop"][j], "xlsr_post", "zipa_post"),
                ("sf", xlsr_sf["phone_gop"][j], zipa_sf["phone_gop"][j], "xlsr_sf", "zipa_sf"),
            ):
                phones_x = [(p["phone"], p["gop"]) for p in pg_x if p is not None]
                phones_z = [(p["phone"], p["gop"]) for p in pg_z if p is not None]
                if not phones_x or not phones_z:
                    continue
                if [p for p, _ in phones_x] != [p for p, _ in phones_z]:
                    n_skipped[label] += 1
                    continue
                espeak_phones = [p for p, _ in phones_x]
                rec_x, ops_x = align_phone.reconcile_phones(dataset_phones, espeak_phones, [g for _, g in phones_x])
                rec_z, ops_z = align_phone.reconcile_phones(dataset_phones, espeak_phones, [g for _, g in phones_z])
                assert ops_x == ops_z
                for gx, gz, a in zip(rec_x, rec_z, ph_acc_word):
                    if gx is None or gz is None:
                        continue
                    ph[d_x].append(gx); ph[d_z].append(gz)
                    if label == "post":
                        ph["acc_binary"].append(1 if a >= 2 else 0)
                        ph["acc_graded"].append(a)
                        ph["speaker"].append(speaker)

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "word": word, "ph": ph, "n_skipped": n_skipped,
        "n_utterances": len(rows), "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align_phone.available():
        print("proscor.align_phone optional deps must be installed.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading speechocean762 [{args.split}] ...", file=sys.stderr)
    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Evaluating {len(rows)} utterances (xlsr-53 + ZIPA, post + sf) ...", file=sys.stderr)

    r = run(rows)
    word, ph = r["word"], r["ph"]

    def corr(x, y, c):
        return gopstats.cluster_bootstrap_pearson(x, y, c) if x else None

    def diff(x1, x2, y, c):
        return gopstats.cluster_bootstrap_paired_diff(x1, x2, y, c) if x1 else None

    summary = {
        "split": args.split,
        "n_utterances": r["n_utterances"],
        "n_words_both": len(word["acc"]),
        "n_phones_both": len(ph["acc_binary"]),
        "n_words_phone_skipped_mismatch": r["n_skipped"],
        "elapsed_s": round(r["elapsed_s"], 1),
        "note": "diff = r(zipa, label) - r(xlsr53, label); positive+significant means ZIPA wins",
        "word_level": {
            "post": {"xlsr_r": corr(word["xlsr_post"], word["acc"], word["speaker"]),
                     "zipa_r": corr(word["zipa_post"], word["acc"], word["speaker"]),
                     "paired_diff": diff(word["zipa_post"], word["xlsr_post"], word["acc"], word["speaker"])},
            "sf": {"xlsr_r": corr(word["xlsr_sf"], word["acc"], word["speaker"]),
                   "zipa_r": corr(word["zipa_sf"], word["acc"], word["speaker"]),
                   "paired_diff": diff(word["zipa_sf"], word["xlsr_sf"], word["acc"], word["speaker"])},
        },
        "phone_level_graded": {
            "post": {"xlsr_r": corr(ph["xlsr_post"], ph["acc_graded"], ph["speaker"]),
                     "zipa_r": corr(ph["zipa_post"], ph["acc_graded"], ph["speaker"]),
                     "paired_diff": diff(ph["zipa_post"], ph["xlsr_post"], ph["acc_graded"], ph["speaker"])},
            "sf": {"xlsr_r": corr(ph["xlsr_sf"], ph["acc_graded"], ph["speaker"]),
                   "zipa_r": corr(ph["zipa_sf"], ph["acc_graded"], ph["speaker"]),
                   "paired_diff": diff(ph["zipa_sf"], ph["xlsr_sf"], ph["acc_graded"], ph["speaker"])},
        },
        "phone_level_binary": {
            "post": {"xlsr_r": corr(ph["xlsr_post"], ph["acc_binary"], ph["speaker"]),
                     "zipa_r": corr(ph["zipa_post"], ph["acc_binary"], ph["speaker"]),
                     "paired_diff": diff(ph["zipa_post"], ph["xlsr_post"], ph["acc_binary"], ph["speaker"])},
            "sf": {"xlsr_r": corr(ph["xlsr_sf"], ph["acc_binary"], ph["speaker"]),
                   "zipa_r": corr(ph["zipa_sf"], ph["acc_binary"], ph["speaker"]),
                   "paired_diff": diff(ph["zipa_sf"], ph["xlsr_sf"], ph["acc_binary"], ph["speaker"])},
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
