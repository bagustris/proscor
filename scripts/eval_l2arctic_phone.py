#!/usr/bin/env python
"""Phone-level GOP-lite validation for L2-ARCTIC's adult Mandarin-L1
speakers (PLAN.md section 5c "sharper follow-up") -- the rigorous
counterpart to scripts/eval_l2arctic.py's utterance-level char-error-rate
proxy. Uses the original TAMU/Kaggle L2-ARCTIC distribution's raw
`annotation/*.TextGrid` files (not the KoelLabs HF mirror, which only gives
two unaligned IPA strings), which have expert-aligned per-phone
canonical/perceived/error-tag triples -- the same granularity as
speechocean762's `phones-accuracy`.

Pipeline per utterance:
1. Parse the TextGrid's "words" and "phones" interval tiers.
2. For each phone interval: bare ARPABET text (no comma) = correctly
   pronounced; "CPL, PPL, s" = substitution; "CPL, sil, d" = deletion (both
   errors); "sil, PPL, a" = addition (an extra phone with no canonical
   counterpart -- excluded, there's nothing to score); "sp"/"sil" bare
   intervals are pause markers, not phones -- excluded.
3. Group phone intervals into their parent word by time-overlap with the
   words tier (both tiers come from the same forced alignment, so
   boundaries coincide).
4. For each word, force-align it with the phoneme-CTC model
   (proscor.align_phone.align_words_gop, same as scripts/eval_so762_phone.py
   and eval_umeerj.py) to get espeak-phone-level GOP scores, then reconcile
   those against this word's canonical ARPABET phones with
   proscor.align_phone.reconcile_phones (the same constrained-alignment
   machinery PLAN.md section 5a item 4 validated on speechocean762) to get
   one GOP estimate per canonical phone.
5. Pool (GOP, is_correct) pairs across all utterances/speakers and compute
   the point-biserial correlation (Pearson correlation against a 0/1
   label) -- directly comparable to speechocean762's phone-level PCC
   (0.425 matched-only / 0.433 reconciled), since both are "GOP vs. a
   phone-level correctness signal" at the same granularity (this one is
   binary rather than 0-2, and expert-annotated either way).

Usage:
    pip install -r requirements-eval.txt
    python scripts/eval_l2arctic_phone.py --data-root /data/L2-ARCTIC/mandarin
    python scripts/eval_l2arctic_phone.py --speakers BWC --limit 20
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proscor import align_phone

_WORD_RE = re.compile(r"[a-z']+")
_ERROR_RE = re.compile(r"^([A-Za-z]+\d?|sil)\s*,\s*(\S+)\s*,\s*([sda])$")


def parse_textgrid(path: Path) -> dict:
    """Minimal Praat TextGrid parser -> {tier_name: [(xmin, xmax, text), ...]}."""
    text = path.read_text(encoding="utf-8")
    tiers = {}
    for block in re.split(r"\n\s*item \[\d+\]:\n", text)[1:]:
        name = re.search(r'name = "([^"]*)"', block).group(1)
        intervals = [
            (float(m.group(1)), float(m.group(2)), m.group(3))
            for m in re.finditer(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', block)
        ]
        tiers[name] = intervals
    return tiers


def parse_phone_interval(text: str):
    """One phones-tier interval -> (canonical_phone, tag) or None to skip (a
    pause marker, or an addition with no canonical counterpart). tag is
    "correct", "s" (substitution) or "d" (deletion)."""
    text = text.strip()
    if text in ("sp", "sil", "", "spn"):
        return None
    m = _ERROR_RE.match(text)
    if not m:
        return (text, "correct")  # bare ARPABET symbol: correctly pronounced
    cpl, _ppl, tag = m.groups()
    if tag == "a" or cpl == "sil":
        return None  # addition: no canonical phone to score
    return (cpl, tag)  # "s" or "d"


def words_and_phones(tiers: dict) -> list:
    """-> [{"text": str, "phones": [(canonical_phone, tag), ...]}, ...]
    in time order, grouping phones-tier intervals into their parent word by
    midpoint containment."""
    words = [(xmin, xmax, t) for xmin, xmax, t in tiers["words"] if t.strip()]
    phones = tiers["phones"]

    result = [{"text": t, "start": xmin, "end": xmax, "phones": []} for xmin, xmax, t in words]
    wi = 0
    for xmin, xmax, text in phones:
        parsed = parse_phone_interval(text)
        if parsed is None:
            continue
        mid = (xmin + xmax) / 2
        while wi < len(result) - 1 and mid >= result[wi]["end"]:
            wi += 1
        if result and result[wi]["start"] <= mid < result[wi]["end"] + 1e-6:
            result[wi]["phones"].append(parsed)
    return result


def run(speaker_dirs: list, limit: int = None, progress_every: int = 50) -> dict:
    gop_all, correct_all = [], []
    gop_by_tag = defaultdict(list)  # "correct" / "s" / "d" -> [gop, ...]
    n_utt = n_utt_failed = n_words_total = n_words_aligned = 0
    t0 = time.time()

    utt_files = []
    for spk_dir in speaker_dirs:
        ann_dir = spk_dir / "annotation"
        for f in sorted(ann_dir.glob("*.TextGrid")):
            utt_files.append((spk_dir.name, f))
    if limit:
        utt_files = utt_files[:limit]

    for i, (speaker, ann_path) in enumerate(utt_files):
        wav_path = ann_path.parents[1] / "wav" / (ann_path.stem + ".wav")
        if not wav_path.exists():
            continue
        tiers = parse_textgrid(ann_path)
        words = words_and_phones(tiers)
        words = [w for w in words if _WORD_RE.fullmatch(w["text"].lower()) and w["phones"]]
        if not words:
            continue
        n_words_total += len(words)

        samples, sr = sf.read(str(wav_path), dtype="float32")
        try:
            result = align_phone.align_words_gop(samples, [w["text"].lower() for w in words], sr=sr)
        except Exception as e:
            print(f"  [warn] {speaker}/{ann_path.name} failed: {e}", file=sys.stderr)
            n_utt_failed += 1
            continue

        for w, pg in zip(words, result["phone_gop"]):
            espeak_phones = [p["phone"] for p in pg if p is not None]
            espeak_gops = [p["gop"] for p in pg if p is not None]
            canon = [p[0] for p in w["phones"]]
            tags = [p[1] for p in w["phones"]]
            if not espeak_phones or not canon:
                continue
            rec_gops, _ops = align_phone.reconcile_phones(canon, espeak_phones, espeak_gops)
            for g, tag in zip(rec_gops, tags):
                if g is None:
                    continue
                n_words_aligned += 1
                gop_all.append(g)
                correct_all.append(1 if tag == "correct" else 0)
                gop_by_tag[tag].append(g)

        n_utt += 1
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(utt_files)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "gop": gop_all, "correct": correct_all, "gop_by_tag": dict(gop_by_tag),
        "n_utterances": n_utt, "n_utterances_failed": n_utt_failed,
        "n_words_total": n_words_total, "n_phones_scored": len(gop_all),
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/L2-ARCTIC/mandarin")
    ap.add_argument("--speakers", nargs="+", default=None, help="default: all speakers found under --data-root")
    ap.add_argument("--limit", type=int, default=None, help="limit utterances (across all speakers combined)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align_phone.available():
        print("proscor.align_phone optional deps must be installed.", file=sys.stderr)
        sys.exit(1)

    root = Path(args.data_root)
    all_speaker_dirs = sorted(p for p in root.glob("*/*") if (p / "annotation").is_dir())
    if args.speakers:
        all_speaker_dirs = [p for p in all_speaker_dirs if p.name in args.speakers]
    print(f"Speakers: {[p.name for p in all_speaker_dirs]}", file=sys.stderr)

    results = run(all_speaker_dirs, limit=args.limit)

    from scipy.stats import pearsonr, spearmanr

    gop, correct = results["gop"], results["correct"]
    if len(gop) >= 2:
        pr, _ = pearsonr(gop, correct)
        sr_, _ = spearmanr(gop, correct)
    else:
        pr = sr_ = None

    summary = {
        "speakers": [p.name for p in all_speaker_dirs],
        "n_utterances": results["n_utterances"],
        "n_utterances_failed": results["n_utterances_failed"],
        "n_words_total": results["n_words_total"],
        "n_phones_scored": results["n_phones_scored"],
        "elapsed_s": round(results["elapsed_s"], 1),
        "frac_correct": round(float(np.mean(correct)), 4) if correct else None,
        "phone_level_gop_vs_correct": {
            "pearson_r": round(float(pr), 4) if pr is not None else None,
            "spearman_rho": round(float(sr_), 4) if sr_ is not None else None,
            "n": len(gop),
        },
        "gop_by_error_tag": {
            tag: {"n": len(vals), "mean_gop": round(float(np.mean(vals)), 3),
                  "median_gop": round(float(np.median(vals)), 3)}
            for tag, vals in sorted(results["gop_by_tag"].items())
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
