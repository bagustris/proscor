#!/usr/bin/env python
"""Paired lv-60-vs-xlsr-53 acoustic-model significance test on
speechocean762 phone-level GOP (PLAN.md section 5i) -- the test flagged as
owed after the full-corpus xlsr-53 marginal results showed a large,
consistent gain but hadn't been proven with the same paired,
speaker-cluster bootstrap used everywhere else in this plan (section 5d/5f).
Scores every utterance with posterior-deficit GOP under BOTH acoustic
models (`align_phone.MODEL_REPO`, ONNX/lv-60; `align_phone.TORCH_MODEL_REPO`,
xlsr-53) in the same loop, then runs the paired bootstrap on the difference.

Phone-level pairing uses the same reconcile_phones-identity-invariance
property as scripts/eval_so762_phone_paired.py: its chosen alignment ops
depend only on phone identity, never on the GOP values passed alongside
them, so reconciling each model's own GOPs against an identical
espeak-phone identity list gives positionally-aligned output for free.
The two models' vocabularies aren't identical, though (each has its own
392-symbol set from its own fine-tuning), so a phone espeak emits for a
word can be in one model's vocab and not the other's -- when that
happens the two models' filtered phone lists for that word disagree, and
the word is skipped from the phone-level comparison (word-level is
unaffected, since it only needs each model's own word_gop). Counted
separately, not silently dropped.

Usage:
    python scripts/eval_so762_model_paired.py --out results/so762_model_paired.json
    python scripts/eval_so762_model_paired.py --limit 100
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


def run(rows: list, progress_every: int = 200) -> dict:
    word_lv60, word_xlsr, word_acc, word_speaker = [], [], [], []
    ph_lv60, ph_xlsr, ph_acc_binary, ph_acc_graded, ph_speaker = [], [], [], [], []
    n_words_phone_skipped_mismatch = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        speaker = row["speaker"]
        try:
            res_lv60 = align_phone.align_words_gop(samples, words, sr=sr, model_id=align_phone.MODEL_REPO)
            res_xlsr = align_phone.align_words_gop(samples, words, sr=sr, model_id=align_phone.TORCH_MODEL_REPO)
        except Exception as e:
            print(f"  [warn] utt {i} failed: {e}", file=sys.stderr)
            continue

        for w, wg_l, wg_x, pg_l, pg_x in zip(
            row["words"], res_lv60["word_gop"], res_xlsr["word_gop"],
            res_lv60["phone_gop"], res_xlsr["phone_gop"],
        ):
            if wg_l is None or wg_x is None:
                continue
            word_lv60.append(wg_l["gop"])
            word_xlsr.append(wg_x["gop"])
            word_acc.append(w["accuracy"])
            word_speaker.append(speaker)

            phones_l = [(p["phone"], p["gop"]) for p in pg_l if p is not None]
            phones_x = [(p["phone"], p["gop"]) for p in pg_x if p is not None]
            dataset_phones = w["phones"]
            ph_acc_word = w["phones-accuracy"]
            if not dataset_phones:
                continue
            if [ph for ph, _ in phones_l] != [ph for ph, _ in phones_x]:
                n_words_phone_skipped_mismatch += 1
                continue
            if not phones_l:
                continue
            espeak_phones = [ph for ph, _ in phones_l]
            gops_lv60 = [g for _, g in phones_l]
            gops_xlsr = [g for _, g in phones_x]
            rec_lv60, ops_l = align_phone.reconcile_phones(dataset_phones, espeak_phones, gops_lv60)
            rec_xlsr, ops_x = align_phone.reconcile_phones(dataset_phones, espeak_phones, gops_xlsr)
            assert ops_l == ops_x  # same identities in -> same DP choice, always
            for gl, gx, a in zip(rec_lv60, rec_xlsr, ph_acc_word):
                if gl is None or gx is None:
                    continue
                ph_lv60.append(gl)
                ph_xlsr.append(gx)
                ph_acc_binary.append(1 if a >= 2 else 0)
                ph_acc_graded.append(a)
                ph_speaker.append(speaker)

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "word_lv60": word_lv60, "word_xlsr": word_xlsr, "word_acc": word_acc, "word_speaker": word_speaker,
        "ph_lv60": ph_lv60, "ph_xlsr": ph_xlsr, "ph_acc_binary": ph_acc_binary,
        "ph_acc_graded": ph_acc_graded, "ph_speaker": ph_speaker,
        "n_words_phone_skipped_mismatch": n_words_phone_skipped_mismatch,
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
    print(f"Evaluating {len(rows)} utterances (both models) ...", file=sys.stderr)

    r = run(rows)

    word_diff = gopstats.cluster_bootstrap_paired_diff(r["word_lv60"], r["word_xlsr"], r["word_acc"], r["word_speaker"])
    phone_graded_diff = gopstats.cluster_bootstrap_paired_diff(
        r["ph_lv60"], r["ph_xlsr"], r["ph_acc_graded"], r["ph_speaker"])
    phone_binary_diff = gopstats.cluster_bootstrap_paired_diff(
        r["ph_lv60"], r["ph_xlsr"], r["ph_acc_binary"], r["ph_speaker"])

    summary = {
        "split": args.split,
        "n_utterances": r["n_utterances"],
        "n_words_both": len(r["word_lv60"]),
        "n_phones_both": len(r["ph_lv60"]),
        "n_words_phone_skipped_mismatch": r["n_words_phone_skipped_mismatch"],
        "elapsed_s": round(r["elapsed_s"], 1),
        "note": "diff = r(lv60, label) - r(xlsr53, label); negative diff / significant=True means xlsr-53 wins",
        "word_level_lv60_vs_xlsr53_paired_diff": word_diff,
        "phone_level_graded_lv60_vs_xlsr53_paired_diff": phone_graded_diff,
        "phone_level_binary_lv60_vs_xlsr53_paired_diff": phone_binary_diff,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
