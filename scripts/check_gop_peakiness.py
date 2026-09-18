#!/usr/bin/env python
"""Does posterior-deficit GOP's zero-inflation (PLAN.md section 5c/5e/5f's
"substitution blindness") look different on UME-ERJ audio than on
L2-ARCTIC audio? A third hypothesis for the section 5b/5e UME-ERJ flip,
alongside L1-specificity and UME-ERJ's holistic rating methodology (PLAN.md
section 5g): if UME-ERJ audio produces a markedly less saturated GOP
distribution *before any question of correctness enters*, the phone
model's advantage there could be partly an acoustic-environment artifact
(recording setup, sentence material, model domain response) rather than
something about Japanese-L1 or the rating scale.

Collects every individual phone GOP (posterior-deficit engine,
`align_words_gop`; no correctness filtering -- UME-ERJ has no per-phone
labels to filter by, so this compares the engine's raw output
distribution, not accuracy) from a matched-size sample of each corpus.

Usage:
    python scripts/check_gop_peakiness.py
    python scripts/check_gop_peakiness.py --n-utterances 300 --out results/gop_peakiness.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import align_phone
from eval_l2arctic_phone import parse_textgrid, words_and_phones, _WORD_RE

_WORD_RE_ERJ = re.compile(r"[A-Za-z']+")


def l2arctic_sample(root: Path, n_utt: int) -> tuple:
    speaker_dirs = sorted(p for p in root.glob("*/*") if (p / "annotation").is_dir())[:4]
    gops, n_done = [], 0
    for spk_dir in speaker_dirs:
        for ann_path in sorted((spk_dir / "annotation").glob("*.TextGrid")):
            if n_done >= n_utt:
                break
            wav_path = ann_path.parents[1] / "wav" / (ann_path.stem + ".wav")
            if not wav_path.exists():
                continue
            words = words_and_phones(parse_textgrid(ann_path))
            words = [w for w in words if _WORD_RE.fullmatch(w["text"].lower()) and w["phones"]]
            if not words:
                continue
            samples, sr = sf.read(str(wav_path), dtype="float32")
            try:
                res = align_phone.align_words_gop(samples, [w["text"].lower() for w in words], sr=sr)
            except Exception:
                continue
            for pg in res["phone_gop"]:
                for p in pg:
                    if p is not None:
                        gops.append(p["gop"])
            n_done += 1
        if n_done >= n_utt:
            break
    return gops, n_done


def _load_tab(path: Path) -> dict:
    mapping = {}
    with open(path, encoding="shift_jis") as f:
        for line in f:
            parts = line.rstrip().split(None, 2)
            if len(parts) == 3:
                rec, _sel, text = parts
                mapping[rec] = text
    return mapping


def umeerj_sample(root: Path, n_utt: int) -> tuple:
    tab_dir = root / "doc" / "JEcontent" / "tab"
    text_map = _load_tab(tab_dir / "word2.tab")
    text_map.update(_load_tab(tab_dir / "word1.tab"))

    records = []
    for gender in ("male", "female"):
        path = root / "lbl" / f"{gender}-word" / "segmental" / "scores.lst"
        with open(path, encoding="shift_jis") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                records.append(line.split()[0])

    gops, n_done = [], 0
    for rel_path in records:
        if n_done >= n_utt:
            break
        text = text_map.get(Path(rel_path).name)
        if text is None:
            continue
        words = _WORD_RE_ERJ.findall(text)
        if not words:
            continue
        wav_path = root / "wav" / "JE" / rel_path
        if not wav_path.exists():
            continue
        samples, sr = sf.read(str(wav_path), dtype="float32")
        try:
            res = align_phone.align_words_gop(samples, [w.lower() for w in words], sr=sr)
        except Exception:
            continue
        for pg in res["phone_gop"]:
            for p in pg:
                if p is not None:
                    gops.append(p["gop"])
        n_done += 1
    return gops, n_done


def summarize(gops: list, n_utt: int) -> dict:
    arr = np.array(gops)
    return {
        "n_utterances": n_utt, "n_phones": len(arr),
        "frac_exactly_zero": round(float(np.mean(arr == 0.0)), 4),
        "mean": round(float(arr.mean()), 4), "median": round(float(np.median(arr)), 4),
        "std": round(float(arr.std()), 4),
        "frac_below_neg1": round(float(np.mean(arr < -1.0)), 4),
        "frac_below_neg3": round(float(np.mean(arr < -3.0)), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l2arctic-root", default="/data/L2-ARCTIC/all")
    ap.add_argument("--umeerj-root", default="/data/UME-ERJ")
    ap.add_argument("--n-utterances", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align_phone.available():
        print("proscor.align_phone optional deps must be installed.", file=sys.stderr)
        sys.exit(1)

    l2_gops, l2_n = l2arctic_sample(Path(args.l2arctic_root), args.n_utterances)
    erj_gops, erj_n = umeerj_sample(Path(args.umeerj_root), args.n_utterances)

    summary = {"l2arctic": summarize(l2_gops, l2_n), "umeerj_word_segmental": summarize(erj_gops, erj_n)}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
