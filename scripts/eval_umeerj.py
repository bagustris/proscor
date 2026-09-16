#!/usr/bin/env python
"""Validate GOP-lite (both engines: proscor/align.py's BPE model and
proscor/align_phone.py's phoneme model) against UME-ERJ, a second corpus
with a different L1 (Japanese, vs. speechocean762's Mandarin) and likely a
different age range (university students, vs. speechocean762's children) --
a generalization check for the word/utterance-level numbers in PLAN.md
section 5a.

UME-ERJ (NII Speech Resources Consortium, https://research.nii.ac.jp/src/en/UME-ERJ.html)
ships as a local directory tree (Shift-JIS-encoded metadata), not a
downloadable HF/parquet mirror -- pass its root via --data-root (default
/data/UME-ERJ). It has **no per-phone accuracy labels**: ratings are
holistic 1-5 scores from up to 5 native-English-teacher raters, per
utterance (segmental/rhythm/intonation for sentences) or per word
(segmental/accent), averaged here across whichever raters scored that item
(rater identity/position isn't preserved in the source files, and isn't
needed for a correlation against the mean). This script therefore validates
word- and utterance-level GOP-lite only; it cannot replicate the
phone-level reconciled-PCC headline number scripts/eval_so762_phone.py
reports on speechocean762, since there's nothing to reconcile against.

Directory layout assumed (see /data/UME-ERJ/doc/JElabel and
doc/JEcontent for the source documentation):
  wav/JE/<SITE>/<GEN><SPK>/{S#_###,W#_###}.wav          -- audio, 16kHz/16bit/mono
  doc/JEcontent/tab/{sentence,word}{1,2}.tab             -- recording filename -> text
  lbl/{male,female}-{sentence,word}/<category>/scores.lst -- ratings (one row per
      rated recording: "<rel_wav_path>  <score> <score> ...", variable rater count)

Five category files (male+female pooled) are each correlated separately
against both engines' mean GOP: sentence-segmental, sentence-rhythm,
sentence-intonation, word-segmental, word-accent.

Usage:
    pip install -r requirements-eval.txt
    python scripts/eval_umeerj.py                       # all 5 categories, both engines
    python scripts/eval_umeerj.py --categories sentence-segmental word-segmental
    python scripts/eval_umeerj.py --limit 100            # quick smoke run
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proscor import align, align_phone

CATEGORIES = {
    "sentence-segmental": ("sentence", "segmental"),
    "sentence-rhythm": ("sentence", "rhythm"),
    "sentence-intonation": ("sentence", "intonation"),
    "word-segmental": ("word", "segmental"),
    "word-accent": ("word", "accent"),
}


def _load_tab(path: Path) -> dict:
    """[recording_filename] -> text, from a Shift-JIS, CRLF-terminated,
    whitespace-separated (not literal-tab-separated, despite the extension)
    3-column file: recording filename, "selection" filename (unused here),
    text (itself whitespace-containing for sentences)."""
    mapping = {}
    with open(path, encoding="shift_jis") as f:
        for line in f:
            parts = line.rstrip().split(None, 2)
            if len(parts) == 3:
                rec, _sel, text = parts
                mapping[rec] = text
    return mapping


def load_text_mapping(data_root: Path, unit: str) -> dict:
    """Merge tab{1,2}.tab (near-duplicates with minor coverage differences;
    see PLAN.md section 5a for how this was discovered) into one filename ->
    text dict, preferring tab1's entry on overlap."""
    tab_dir = data_root / "doc" / "JEcontent" / "tab"
    mapping = _load_tab(tab_dir / f"{unit}2.tab")
    mapping.update(_load_tab(tab_dir / f"{unit}1.tab"))
    return mapping


def load_scores(data_root: Path, unit: str, category: str) -> list:
    """Combine male-<unit>/<category>/scores.lst and
    female-<unit>/<category>/scores.lst -> [(rel_wav_path, mean_score, n_raters), ...]."""
    records = []
    for gender in ("male", "female"):
        path = data_root / "lbl" / f"{gender}-{unit}" / category / "scores.lst"
        with open(path, encoding="shift_jis") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                rel_path, scores = parts[0], [float(s) for s in parts[1:]]
                if scores:
                    records.append((rel_path, float(np.mean(scores)), len(scores)))
    return records


_WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> list:
    return _WORD_RE.findall(text)


def run(data_root: Path, unit: str, category: str, limit: int = None,
        progress_every: int = 200) -> dict:
    text_map = load_text_mapping(data_root, unit)
    records = load_scores(data_root, unit, category)
    if limit:
        records = records[:limit]

    bpe_gop, phone_gop, human_score, n_raters_list = [], [], [], []
    n_total = len(records)
    n_no_text = n_no_audio = n_bpe_failed = n_phone_failed = 0
    t0 = time.time()

    for i, (rel_path, score, n_raters) in enumerate(records):
        text = text_map.get(Path(rel_path).name)
        if text is None:
            n_no_text += 1
            continue
        words = tokenize(text)
        if not words:
            n_no_text += 1
            continue

        wav_path = data_root / "wav" / "JE" / rel_path
        if not wav_path.exists():
            n_no_audio += 1
            continue
        samples, sr = sf.read(str(wav_path), dtype="float32")

        try:
            bpe_result = align.align_words_gop(samples, [w.lower() for w in words], sr=sr)
            bpe_vals = [r["gop"] for r in bpe_result if r is not None]
            bpe_gop_i = float(np.mean(bpe_vals)) if bpe_vals else None
        except Exception as e:
            print(f"  [warn] bpe failed on {rel_path!r}: {e}", file=sys.stderr)
            bpe_gop_i = None
        if bpe_gop_i is None:
            n_bpe_failed += 1

        try:
            phone_result = align_phone.align_words_gop(samples, words, sr=sr)
            phone_vals = [r["gop"] for r in phone_result["word_gop"] if r is not None]
            phone_gop_i = float(np.mean(phone_vals)) if phone_vals else None
        except Exception as e:
            print(f"  [warn] phone failed on {rel_path!r}: {e}", file=sys.stderr)
            phone_gop_i = None
        if phone_gop_i is None:
            n_phone_failed += 1

        if bpe_gop_i is None and phone_gop_i is None:
            continue
        bpe_gop.append(bpe_gop_i)
        phone_gop.append(phone_gop_i)
        human_score.append(score)
        n_raters_list.append(n_raters)

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{n_total} items ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/item)", file=sys.stderr)

    return {
        "bpe_gop": bpe_gop, "phone_gop": phone_gop, "human_score": human_score,
        "n_raters": n_raters_list, "n_total": n_total,
        "n_no_text": n_no_text, "n_no_audio": n_no_audio,
        "n_bpe_failed": n_bpe_failed, "n_phone_failed": n_phone_failed,
        "elapsed_s": time.time() - t0,
    }


def correlations(x: list, y: list) -> dict:
    from scipy.stats import pearsonr, spearmanr

    pairs = [(a, b) for a, b in zip(x, y) if a is not None]
    if len(pairs) < 2:
        return {"pearson_r": None, "spearman_rho": None, "n": len(pairs)}
    xs, ys = zip(*pairs)
    pr, _ = pearsonr(xs, ys)
    sr, _ = spearmanr(xs, ys)
    return {"pearson_r": round(float(pr), 4), "spearman_rho": round(float(sr), 4), "n": len(pairs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/UME-ERJ")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N items per category")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align.available() or not align_phone.available():
        print("proscor.align and proscor.align_phone optional deps must both be installed "
              "(onnxruntime, kaldi_native_fbank, phonemizer + espeak-ng).", file=sys.stderr)
        sys.exit(1)

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"UME-ERJ not found at {data_root} (pass --data-root)", file=sys.stderr)
        sys.exit(1)

    summary = {}
    for cat in args.categories:
        unit, subcat = CATEGORIES[cat]
        print(f"=== {cat} ===", file=sys.stderr)
        results = run(data_root, unit, subcat, limit=args.limit)
        summary[cat] = {
            "n_total": results["n_total"],
            "n_scored": len(results["human_score"]),
            "n_no_text": results["n_no_text"],
            "n_no_audio": results["n_no_audio"],
            "n_bpe_failed": results["n_bpe_failed"],
            "n_phone_failed": results["n_phone_failed"],
            "elapsed_s": round(results["elapsed_s"], 1),
            "bpe_vs_human": correlations(results["bpe_gop"], results["human_score"]),
            "phone_vs_human": correlations(results["phone_gop"], results["human_score"]),
        }
        print(json.dumps(summary[cat], indent=2), file=sys.stderr)

    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
