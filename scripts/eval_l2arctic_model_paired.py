#!/usr/bin/env python
"""Paired lv-60-vs-xlsr-53 acoustic-model significance test on L2-ARCTIC
phone-level GOP (PLAN.md section 5i) -- the L2-ARCTIC counterpart to
scripts/eval_so762_model_paired.py; see that script's docstring for the
pairing method.

Usage:
    python scripts/eval_l2arctic_model_paired.py --data-root /data/L2-ARCTIC/all --out results/l2arctic_model_paired.json
    python scripts/eval_l2arctic_model_paired.py --speakers BWC --limit 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import align_phone, stats as gopstats
from eval_l2arctic_phone import LANG_BY_SPEAKER, _WORD_RE, parse_textgrid, words_and_phones


def run(speaker_dirs: list, limit: int = None, progress_every: int = 200) -> dict:
    ph_lv60, ph_xlsr, ph_correct, ph_speaker, ph_lang = [], [], [], [], []
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
            res_lv60 = align_phone.align_words_gop(samples, texts, sr=sr, model_id=align_phone.MODEL_REPO)
            res_xlsr = align_phone.align_words_gop(samples, texts, sr=sr, model_id=align_phone.TORCH_MODEL_REPO)
        except Exception as e:
            print(f"  [warn] {speaker}/{ann_path.name} failed: {e}", file=sys.stderr)
            n_utt_failed += 1
            continue

        lang = LANG_BY_SPEAKER.get(speaker, "unknown")
        for w, pg_l, pg_x in zip(words, res_lv60["phone_gop"], res_xlsr["phone_gop"]):
            phones_l = [(p["phone"], p["gop"]) for p in pg_l if p is not None]
            phones_x = [(p["phone"], p["gop"]) for p in pg_x if p is not None]
            canon = [p[0] for p in w["phones"]]
            tags = [p[1] for p in w["phones"]]
            if not canon:
                continue
            if [ph for ph, _ in phones_l] != [ph for ph, _ in phones_x]:
                n_words_phone_skipped_mismatch += 1
                continue
            if not phones_l:
                continue
            espeak_phones = [ph for ph, _ in phones_l]
            gops_lv60 = [g for _, g in phones_l]
            gops_xlsr = [g for _, g in phones_x]
            rec_lv60, ops_l = align_phone.reconcile_phones(canon, espeak_phones, gops_lv60)
            rec_xlsr, ops_x = align_phone.reconcile_phones(canon, espeak_phones, gops_xlsr)
            assert ops_l == ops_x
            for gl, gx, tag in zip(rec_lv60, rec_xlsr, tags):
                if gl is None or gx is None:
                    continue
                ph_lv60.append(gl)
                ph_xlsr.append(gx)
                ph_correct.append(1 if tag == "correct" else 0)
                ph_speaker.append(speaker)
                ph_lang.append(lang)

        n_utt += 1
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(utt_files)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "ph_lv60": ph_lv60, "ph_xlsr": ph_xlsr, "ph_correct": ph_correct,
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

    pooled_diff = gopstats.cluster_bootstrap_paired_diff(r["ph_lv60"], r["ph_xlsr"], r["ph_correct"], r["ph_speaker"])

    by_lang = {}
    for lang in sorted(set(r["ph_lang"])):
        idx = [j for j, lg in enumerate(r["ph_lang"]) if lg == lang]
        if len(idx) < 2:
            continue
        by_lang[lang] = gopstats.cluster_bootstrap_paired_diff(
            [r["ph_lv60"][j] for j in idx], [r["ph_xlsr"][j] for j in idx],
            [r["ph_correct"][j] for j in idx], [r["ph_speaker"][j] for j in idx],
        )

    summary = {
        "speakers": [p.name for p in speaker_dirs],
        "n_utterances": r["n_utterances"],
        "n_utterances_failed": r["n_utterances_failed"],
        "n_phones_both": len(r["ph_lv60"]),
        "n_words_phone_skipped_mismatch": r["n_words_phone_skipped_mismatch"],
        "elapsed_s": round(r["elapsed_s"], 1),
        "note": "diff = r(lv60, correct) - r(xlsr53, correct); negative diff / significant=True means xlsr-53 wins",
        "pooled_lv60_vs_xlsr53_paired_diff": pooled_diff,
        "by_language_lv60_vs_xlsr53_paired_diff": by_lang,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
