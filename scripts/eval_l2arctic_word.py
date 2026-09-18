#!/usr/bin/env python
"""Word-level BPE-vs-phone-model comparison on L2-ARCTIC with CLEAN labels
(PLAN.md section 5e) -- the replacement for scripts/eval_l2arctic.py's
character-edit-distance proxy, which turned out too noisy to compare the
two engines language-by-language (section 5d).

Label: from the expert per-phone TextGrid annotations (same parser as
scripts/eval_l2arctic_phone.py), each word gets
  * `frac_correct` = fraction of its canonical phones tagged correct
    (graded, the analogue of speechocean762's 0-10 word accuracy), and
  * `any_error`    = 1 if any phone was substituted/deleted (binary).
Both engines score the SAME words, so the paired speaker-cluster bootstrap
(proscor/stats.py) can test "which engine wins" per L1 -- the question
section 5c could not answer cleanly: does L1 alone, with age held fixed
(all 24 speakers are adults), flip the BPE-vs-phone ranking?

Usage:
    python scripts/eval_l2arctic_word.py --data-root /data/L2-ARCTIC/all --out results/l2arctic_word.json
    python scripts/eval_l2arctic_word.py --speakers BWC --limit 20
"""
import argparse
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
from eval_l2arctic_phone import LANG_BY_SPEAKER, _WORD_RE, parse_textgrid, words_and_phones


def run(speaker_dirs: list, limit: int = None, progress_every: int = 200) -> dict:
    recs = []  # one dict per word both engines scored
    utts = []  # one dict per utterance
    n_utt = n_utt_failed = 0
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
            bpe_result = align.align_words_gop(samples, texts, sr=sr)
        except Exception as e:
            print(f"  [warn] bpe failed on {speaker}/{ann_path.name}: {e}", file=sys.stderr)
            bpe_result = [None] * len(texts)
        try:
            phone_result = align_phone.align_words_gop(samples, texts, sr=sr)["word_gop"]
        except Exception as e:
            print(f"  [warn] phone failed on {speaker}/{ann_path.name}: {e}", file=sys.stderr)
            phone_result = [None] * len(texts)

        this = []
        for w, br, pr in zip(words, bpe_result, phone_result):
            if br is None or pr is None:
                continue
            n_err = sum(1 for _p, tag in w["phones"] if tag != "correct")
            rec = {
                "speaker": speaker, "language": LANG_BY_SPEAKER.get(speaker, "unknown"),
                "bpe_gop": br["gop"], "phone_gop": pr["gop"],
                "frac_correct": 1.0 - n_err / len(w["phones"]),
                "any_error": 1 if n_err else 0,
            }
            recs.append(rec)
            this.append(rec)
        if not this:
            n_utt_failed += 1
            continue
        utts.append({
            "speaker": speaker, "language": LANG_BY_SPEAKER.get(speaker, "unknown"),
            "bpe_gop": float(np.mean([r["bpe_gop"] for r in this])),
            "phone_gop": float(np.mean([r["phone_gop"] for r in this])),
            "frac_correct": float(np.mean([r["frac_correct"] for r in this])),
        })
        n_utt += 1

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(utt_files)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {"words": recs, "utts": utts, "n_utterances": n_utt, "n_utterances_failed": n_utt_failed,
            "elapsed_s": time.time() - t0}


def compare(recs: list, label: str) -> dict:
    if len(recs) < 2:
        return None
    x1 = [r["bpe_gop"] for r in recs]
    x2 = [r["phone_gop"] for r in recs]
    y = [r[label] for r in recs]
    c = [r["speaker"] for r in recs]
    return {
        "n_items": len(recs), "n_speakers": len(set(c)),
        "bpe_speaker_cluster_ci": gopstats.cluster_bootstrap_pearson(x1, y, c),
        "phone_speaker_cluster_ci": gopstats.cluster_bootstrap_pearson(x2, y, c),
        "bpe_vs_phone_paired_diff": gopstats.cluster_bootstrap_paired_diff(x1, x2, y, c),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/L2-ARCTIC/all")
    ap.add_argument("--speakers", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="summary JSON; per-word/utterance records go to <out>.arrays.json")
    args = ap.parse_args()

    if not align.available() or not align_phone.available():
        print("proscor.align and proscor.align_phone optional deps must both be installed.", file=sys.stderr)
        sys.exit(1)

    root = Path(args.data_root)
    speaker_dirs = sorted(p for p in root.glob("*/*") if (p / "annotation").is_dir())
    if args.speakers:
        speaker_dirs = [p for p in speaker_dirs if p.name in args.speakers]
    print(f"Speakers: {[p.name for p in speaker_dirs]}", file=sys.stderr)

    results = run(speaker_dirs, limit=args.limit)
    words, utts = results["words"], results["utts"]

    by_lang_w = defaultdict(list)
    by_lang_u = defaultdict(list)
    for r in words:
        by_lang_w[r["language"]].append(r)
    for u in utts:
        by_lang_u[u["language"]].append(u)

    summary = {
        "speakers": [p.name for p in speaker_dirs],
        "n_utterances": results["n_utterances"],
        "n_utterances_failed": results["n_utterances_failed"],
        "n_words_both": len(words),
        "word_frac_correct_mean": round(float(np.mean([r["frac_correct"] for r in words])), 4) if words else None,
        "word_any_error_rate": round(float(np.mean([r["any_error"] for r in words])), 4) if words else None,
        "elapsed_s": round(results["elapsed_s"], 1),
        "note": "labels come from the expert per-phone annotations; expected sign of r is POSITIVE "
                "for frac_correct (higher GOP = more correct phones) and NEGATIVE for any_error",
        "pooled": {
            "word_vs_frac_correct": compare(words, "frac_correct"),
            "word_vs_any_error": compare(words, "any_error"),
            "utt_vs_frac_correct": compare(utts, "frac_correct"),
        },
        "by_language": {
            lang: {
                "word_vs_frac_correct": compare(by_lang_w[lang], "frac_correct"),
                "word_vs_any_error": compare(by_lang_w[lang], "any_error"),
                "utt_vs_frac_correct": compare(by_lang_u[lang], "frac_correct"),
            }
            for lang in sorted(by_lang_w)
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        with open(args.out + ".arrays.json", "w") as f:
            json.dump({"words": words, "utts": utts}, f)
        print(f"Summary written to {args.out}; records to {args.out}.arrays.json", file=sys.stderr)


if __name__ == "__main__":
    main()
