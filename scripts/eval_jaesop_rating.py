#!/usr/bin/env python
"""Holistic-rating BPE-vs-phone-model comparison on J-AESOP (PLAN.md
section 5j) -- the other half of the ideal disentangling test alongside
scripts/eval_jaesop_word.py's clean-label comparison. Same 180 Japanese-L1
speakers, same Task 6_01 "North Wind and the Sun" recordings, but scored
against `Documents/Rating6_01.csv`'s holistic 1-10 segmental/prosody/
fluency/nativelikeness ratings instead of the phone-error tags -- the same
category structure UME-ERJ uses (section 5b), on the *same audio* the
clean-label test uses, so a difference in which engine wins between the
two scripts isolates methodology (clean tags vs. holistic ratings) from
everything else (speakers, L1, recording conditions), which no other
comparison in this plan can do.

**Granularity, a deliberate simplification:** ratings are recorded per
*section* (the recording is split into 3 parts, matching paragraph
groupings not explicitly re-derivable from the TextGrid without guessing
at pause-based boundaries) and per rater (up to 16 raters/section). Rather
than risk an incorrect section-boundary guess corrupting the result, this
script averages ratings across all sections and raters into ONE holistic
score per speaker, and correlates it against ONE whole-passage GOP mean
per speaker (mirroring UME-ERJ's *sentence*-level granularity, not
word-level) -- a real loss of resolution versus a correctly section-
aligned version, but a safe one: wrong on granularity is a weaker test,
wrong on boundaries would be a wrong answer.

Usage:
    python scripts/eval_jaesop_rating.py --data-root /data/J-AESOP --out results/jaesop_rating.json
    python scripts/eval_jaesop_rating.py --limit 10
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import align, align_phone, stats as gopstats
from eval_jaesop_word import CANONICAL_TEXT

CATEGORIES = ("Segment", "Prosody", "Fluency", "Nativelikeness")


def load_ratings(data_root: Path) -> dict:
    """-> {speaker: {category: mean_rating}} averaged across all sections
    and all raters for that speaker's Task 6_01 recording."""
    path = data_root / "Documents" / "Rating6_01.csv"
    per_speaker = defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row["Speaker"].startswith("J_"):
                continue
            for cat in CATEGORIES:
                per_speaker[row["Speaker"]][cat].append(float(row[cat]))
    return {spk: {cat: float(np.mean(vals)) for cat, vals in cats.items()}
            for spk, cats in per_speaker.items()}


def run(data_root: Path, ratings: dict, limit: int = None, progress_every: int = 20) -> dict:
    audio_dir = data_root / "Audio"
    speakers = sorted(ratings)
    if limit:
        speakers = speakers[:limit]

    recs = []
    n_done = n_failed = 0
    t0 = time.time()

    for i, speaker in enumerate(speakers):
        wav_path = audio_dir / f"{speaker}_6_01.wav"
        if not wav_path.exists():
            n_failed += 1
            continue
        samples, sr = sf.read(str(wav_path), dtype="float32")
        try:
            bpe_result = align.align_words_gop(samples, [w.lower() for w in CANONICAL_TEXT], sr=sr)
            phone_result = align_phone.align_words_gop(samples, CANONICAL_TEXT, sr=sr)["word_gop"]
        except Exception as e:
            print(f"  [warn] {speaker} failed: {e}", file=sys.stderr)
            n_failed += 1
            continue

        bpe_vals = [r["gop"] for r in bpe_result if r is not None]
        phone_vals = [r["gop"] for r in phone_result if r is not None]
        if not bpe_vals or not phone_vals:
            n_failed += 1
            continue
        recs.append({
            "speaker": speaker,
            "bpe_gop": float(np.mean(bpe_vals)), "phone_gop": float(np.mean(phone_vals)),
            **ratings[speaker],
        })
        n_done += 1

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(speakers)} speakers ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1):.1f}s/speaker)", file=sys.stderr)

    return {"records": recs, "n_done": n_done, "n_failed": n_failed, "elapsed_s": time.time() - t0}


def correlations_ci(recs: list, gop_key: str, cat: str) -> dict:
    x = [r[gop_key] for r in recs]
    y = [r[cat] for r in recs]
    c = [r["speaker"] for r in recs]
    return gopstats.cluster_bootstrap_pearson(x, y, c)


def paired_diff(recs: list, cat: str) -> dict:
    x1 = [r["bpe_gop"] for r in recs]
    x2 = [r["phone_gop"] for r in recs]
    y = [r[cat] for r in recs]
    c = [r["speaker"] for r in recs]
    return gopstats.cluster_bootstrap_paired_diff(x1, x2, y, c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/J-AESOP")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align.available() or not align_phone.available():
        print("proscor.align and proscor.align_phone optional deps must both be installed.", file=sys.stderr)
        sys.exit(1)

    data_root = Path(args.data_root)
    ratings = load_ratings(data_root)
    print(f"Loaded ratings for {len(ratings)} speakers", file=sys.stderr)

    results = run(data_root, ratings, limit=args.limit)
    recs = results["records"]

    summary = {
        "n_speakers_done": results["n_done"],
        "n_speakers_failed": results["n_failed"],
        "elapsed_s": round(results["elapsed_s"], 1),
        "note": "one holistic score per speaker (mean over all sections/raters), "
                "one GOP mean per speaker (mean over the whole passage) -- see module docstring",
        "by_category": {
            cat: {
                "bpe_vs_rating": correlations_ci(recs, "bpe_gop", cat),
                "phone_vs_rating": correlations_ci(recs, "phone_gop", cat),
                "bpe_vs_phone_paired_diff": paired_diff(recs, cat),
            }
            for cat in CATEGORIES
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        with open(args.out + ".arrays.json", "w") as f:
            json.dump(recs, f)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
