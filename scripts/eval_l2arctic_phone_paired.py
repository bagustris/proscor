#!/usr/bin/env python
"""Paired posterior-deficit-vs-segmentation-free significance test on
L2-ARCTIC phone-level GOP (PLAN.md section 5f) -- the L2-ARCTIC counterpart
to scripts/eval_so762_phone_paired.py; see that script's docstring for the
pairing method (reconcile_phones's chosen ops depend only on phone
identity, never GOP values, so reconciling each engine's own GOPs against
an identical espeak-phone identity list gives positionally-aligned output;
words where the two engines disagree on which phones aligned at all are
skipped from the phone-level comparison, counted separately).

Usage:
    python scripts/eval_l2arctic_phone_paired.py --data-root /data/L2-ARCTIC/all --out results/l2arctic_phone_paired.json
    python scripts/eval_l2arctic_phone_paired.py --speakers BWC --limit 20
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

from proscor import align_phone, stats as gopstats
from eval_l2arctic_phone import LANG_BY_SPEAKER, _WORD_RE, parse_textgrid, words_and_phones


def run(speaker_dirs: list, limit: int = None, progress_every: int = 200) -> dict:
    ph_post, ph_sf, ph_correct, ph_speaker, ph_lang = [], [], [], [], []
    n_utt = n_utt_failed = n_words_phone_skipped_mismatch = 0
    t0 = time.time()

    utt_files = []
    for spk_dir in speaker_dirs:
        for f in sorted((spk_dir / "annotation").glob("*.TextGrid")):
            utt_files.append((spk_dir.name, f))
    if limit:
        utt_files = utt_files[:limit]

    for i, (speaker, ann_path) in enumerate(utt_files):
        wav_path = ann_path.parents[1] / "wav" / (ann_path.stem + ".wav")
        if not wav_path.exists():
            continue
        words = words_and_phones(parse_textgrid(ann_path))
        words = [w for w in words if _WORD_RE.fullmatch(w["text"].lower()) and w["phones"]]
        if not words:
            continue
        texts = [w["text"].lower() for w in words]
        samples, sr = sf.read(str(wav_path), dtype="float32")

        try:
            res_post = align_phone.align_words_gop(samples, texts, sr=sr)
            res_sf = align_phone.align_words_gop_sf(samples, texts, sr=sr)
        except Exception as e:
            print(f"  [warn] {speaker}/{ann_path.name} failed: {e}", file=sys.stderr)
            n_utt_failed += 1
            continue

        lang = LANG_BY_SPEAKER.get(speaker, "unknown")
        for w, pg_p, pg_s in zip(words, res_post["phone_gop"], res_sf["phone_gop"]):
            phones_p = [(p["phone"], p["gop"]) for p in pg_p if p is not None]
            phones_s = [(p["phone"], p["gop"]) for p in pg_s if p is not None]
            canon = [p[0] for p in w["phones"]]
            tags = [p[1] for p in w["phones"]]
            if not canon:
                continue
            if [ph for ph, _ in phones_p] != [ph for ph, _ in phones_s]:
                n_words_phone_skipped_mismatch += 1
                continue
            if not phones_p:
                continue
            espeak_phones = [ph for ph, _ in phones_p]
            gops_post = [g for _, g in phones_p]
            gops_sf = [g for _, g in phones_s]
            rec_post, ops_post = align_phone.reconcile_phones(canon, espeak_phones, gops_post)
            rec_sf, ops_sf = align_phone.reconcile_phones(canon, espeak_phones, gops_sf)
            assert ops_post == ops_sf
            for gp, gs, tag in zip(rec_post, rec_sf, tags):
                if gp is None or gs is None:
                    continue
                ph_post.append(gp)
                ph_sf.append(gs)
                ph_correct.append(1 if tag == "correct" else 0)
                ph_speaker.append(speaker)
                ph_lang.append(lang)

        n_utt += 1
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(utt_files)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "ph_post": ph_post, "ph_sf": ph_sf, "ph_correct": ph_correct,
        "ph_speaker": ph_speaker, "ph_lang": ph_lang,
        "n_words_phone_skipped_mismatch": n_words_phone_skipped_mismatch,
        "n_utterances": n_utt, "n_utterances_failed": n_utt_failed,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/L2-ARCTIC/all")
    ap.add_argument("--speakers", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align_phone.available():
        print("proscor.align_phone optional deps must be installed.", file=sys.stderr)
        sys.exit(1)

    root = Path(args.data_root)
    speaker_dirs = sorted(p for p in root.glob("*/*") if (p / "annotation").is_dir())
    if args.speakers:
        speaker_dirs = [p for p in speaker_dirs if p.name in args.speakers]
    print(f"Speakers: {[p.name for p in speaker_dirs]}", file=sys.stderr)

    r = run(speaker_dirs, limit=args.limit)

    pooled_diff = gopstats.cluster_bootstrap_paired_diff(r["ph_post"], r["ph_sf"], r["ph_correct"], r["ph_speaker"])

    by_lang = {}
    langs = sorted(set(r["ph_lang"]))
    for lang in langs:
        idx = [j for j, lg in enumerate(r["ph_lang"]) if lg == lang]
        if len(idx) < 2:
            continue
        by_lang[lang] = gopstats.cluster_bootstrap_paired_diff(
            [r["ph_post"][j] for j in idx], [r["ph_sf"][j] for j in idx],
            [r["ph_correct"][j] for j in idx], [r["ph_speaker"][j] for j in idx],
        )

    summary = {
        "speakers": [p.name for p in speaker_dirs],
        "n_utterances": r["n_utterances"],
        "n_utterances_failed": r["n_utterances_failed"],
        "n_phones_both": len(r["ph_post"]),
        "n_words_phone_skipped_mismatch": r["n_words_phone_skipped_mismatch"],
        "elapsed_s": round(r["elapsed_s"], 1),
        "note": "diff = r(posterior, correct) - r(sf, correct); negative diff / significant=True means SF wins",
        "pooled_posterior_vs_sf_paired_diff": pooled_diff,
        "by_language_posterior_vs_sf_paired_diff": by_lang,
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
