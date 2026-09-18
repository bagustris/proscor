#!/usr/bin/env python
"""Paired posterior-deficit-vs-segmentation-free significance test on
speechocean762 phone-level GOP (PLAN.md section 5f) -- the test flagged as
owed after the full-corpus `--engine sf` run in eval_so762_phone.py showed
small numeric gains with heavily overlapping marginal CIs, which isn't
enough to call GOP-SF better. Scores every utterance with BOTH
`align_phone.align_words_gop` (posterior) and `align_words_gop_sf` (sf) in
the same loop, then runs `proscor.stats.cluster_bootstrap_paired_diff` on
the difference -- the same paired-on-the-same-items approach used
throughout section 5d/5e for BPE-vs-phone comparisons.

Phone-level pairing detail: reconcile_phones's chosen alignment ops depend
only on phone *identity* (dataset ARPABET phones vs. espeak IPA phones),
never on the GOP values passed alongside them -- so as long as both
engines' filtered (non-None) espeak-phone identity lists for a word are
IDENTICAL, reconciling each engine's own GOPs against that same identity
list is guaranteed to produce positionally-aligned outputs. When the two
engines disagree on which phones aligned at all (rare -- align_words_gop
occasionally drops an individual phone when Viterbi assigns it zero
frames; align_words_gop_sf essentially never does), the word is skipped
from the phone-level comparison rather than force-paired; count reported
as `n_words_phone_skipped_mismatch`.

Usage:
    python scripts/eval_so762_phone_paired.py --out results/so762_phone_paired.json
    python scripts/eval_so762_phone_paired.py --limit 100
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
    word_post, word_sf, word_acc, word_speaker = [], [], [], []
    utt_post, utt_sf, utt_acc, utt_total, utt_speaker = [], [], [], [], []
    ph_post, ph_sf, ph_acc_binary, ph_acc_graded, ph_speaker = [], [], [], [], []
    n_words_phone_skipped_mismatch = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = [w["text"] for w in row["words"]]
        speaker = row["speaker"]
        try:
            res_post = align_phone.align_words_gop(samples, words, sr=sr)
            res_sf = align_phone.align_words_gop_sf(samples, words, sr=sr)
        except Exception as e:
            print(f"  [warn] utt {i} failed: {e}", file=sys.stderr)
            continue

        this_utt_post, this_utt_sf = [], []
        for w, wg_p, wg_s, pg_p, pg_s in zip(
            row["words"], res_post["word_gop"], res_sf["word_gop"],
            res_post["phone_gop"], res_sf["phone_gop"],
        ):
            if wg_p is None or wg_s is None:
                continue
            word_post.append(wg_p["gop"])
            word_sf.append(wg_s["gop"])
            word_acc.append(w["accuracy"])
            word_speaker.append(speaker)
            this_utt_post.append(wg_p["gop"])
            this_utt_sf.append(wg_s["gop"])

            phones_p = [(p["phone"], p["gop"]) for p in pg_p if p is not None]
            phones_s = [(p["phone"], p["gop"]) for p in pg_s if p is not None]
            dataset_phones = w["phones"]
            ph_acc_word = w["phones-accuracy"]
            if not dataset_phones:
                continue
            if [ph for ph, _ in phones_p] != [ph for ph, _ in phones_s]:
                n_words_phone_skipped_mismatch += 1
                continue
            if not phones_p:
                continue
            espeak_phones = [ph for ph, _ in phones_p]
            gops_post = [g for _, g in phones_p]
            gops_sf = [g for _, g in phones_s]
            rec_post, ops_post = align_phone.reconcile_phones(dataset_phones, espeak_phones, gops_post)
            rec_sf, ops_sf = align_phone.reconcile_phones(dataset_phones, espeak_phones, gops_sf)
            assert ops_post == ops_sf  # same identities in -> same DP choice, always
            for gp, gs, a in zip(rec_post, rec_sf, ph_acc_word):
                if gp is None or gs is None:
                    continue
                ph_post.append(gp)
                ph_sf.append(gs)
                ph_acc_binary.append(1 if a >= 2 else 0)
                ph_acc_graded.append(a)
                ph_speaker.append(speaker)

        if this_utt_post:
            utt_post.append(float(np.mean(this_utt_post)))
            utt_sf.append(float(np.mean(this_utt_sf)))
            utt_acc.append(row["accuracy"])
            utt_total.append(row["total"])
            utt_speaker.append(speaker)

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "word_post": word_post, "word_sf": word_sf, "word_acc": word_acc, "word_speaker": word_speaker,
        "utt_post": utt_post, "utt_sf": utt_sf, "utt_acc": utt_acc, "utt_total": utt_total,
        "utt_speaker": utt_speaker,
        "ph_post": ph_post, "ph_sf": ph_sf, "ph_acc_binary": ph_acc_binary,
        "ph_acc_graded": ph_acc_graded, "ph_speaker": ph_speaker,
        "n_words_phone_skipped_mismatch": n_words_phone_skipped_mismatch,
        "n_utterances": len(rows), "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None,
                     help="summary JSON; per-item arrays also written to <out>.arrays.json")
    args = ap.parse_args()

    if not align_phone.available():
        print("proscor.align_phone optional deps must be installed.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading speechocean762 [{args.split}] ...", file=sys.stderr)
    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Evaluating {len(rows)} utterances (both engines) ...", file=sys.stderr)

    r = run(rows)

    word_diff = gopstats.cluster_bootstrap_paired_diff(r["word_post"], r["word_sf"], r["word_acc"], r["word_speaker"])
    utt_diff = gopstats.cluster_bootstrap_paired_diff(r["utt_post"], r["utt_sf"], r["utt_acc"], r["utt_speaker"])
    phone_graded_diff = gopstats.cluster_bootstrap_paired_diff(
        r["ph_post"], r["ph_sf"], r["ph_acc_graded"], r["ph_speaker"])
    phone_binary_diff = gopstats.cluster_bootstrap_paired_diff(
        r["ph_post"], r["ph_sf"], r["ph_acc_binary"], r["ph_speaker"])

    summary = {
        "split": args.split,
        "n_utterances": r["n_utterances"],
        "n_words_both": len(r["word_post"]),
        "n_phones_both": len(r["ph_post"]),
        "n_words_phone_skipped_mismatch": r["n_words_phone_skipped_mismatch"],
        "elapsed_s": round(r["elapsed_s"], 1),
        "note": "diff = r(posterior, label) - r(sf, label); negative diff / significant=True means SF wins",
        "word_level_posterior_vs_sf_paired_diff": word_diff,
        "utterance_level_posterior_vs_sf_paired_diff": utt_diff,
        "phone_level_graded_posterior_vs_sf_paired_diff": phone_graded_diff,
        "phone_level_binary_posterior_vs_sf_paired_diff": phone_binary_diff,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        with open(args.out + ".arrays.json", "w") as f:
            json.dump(r, f)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
