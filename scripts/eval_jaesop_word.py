#!/usr/bin/env python
"""Word-level BPE-vs-phone-model comparison on the J-AESOP corpus (PLAN.md
section 5j) -- the direct test PLAN.md section 5g identified as the way to
close the open UME-ERJ question: does the phone model beat BPE for
Japanese-L1 speakers specifically, or was section 5b/5e's UME-ERJ result
about that corpus's holistic-rating methodology (or its recording
acoustics, section 5g) rather than Japanese-L1 pronunciation error
patterns? J-AESOP gives 180 Japanese-L1 speakers reading the same fixed
"North Wind and the Sun" passage (Task 6_01), with real word-level error
tags (substitution/deviant-pronunciation/deletion, not holistic ratings)
from 7 trained phoneticians -- the same clean-label methodology already
validated on L2-ARCTIC's six L1s in section 5e/5h, now on Japanese, with
far more speakers (180, not 4/L1) than any single-L1 test in this plan.

**Label derivation** (validated against the corpus before trusting it --
see PLAN.md section 5j): the "Word" tier's plain intervals are correct
productions of the next canonical word; "(SBT)CANON->PERCEIVED" and
"(DVN)CANON" tag a canonical word as mispronounced; "(RPT)"/"(DSF)"/"(INS)"
are repair attempts, disfluencies, and insertions with no *resolved*
canonical-word production (skipped, not counted as either correct or
error) -- the repeated/disfluent word's real outcome is whatever
production follows it in the stream; the "Misc" tier's "(DLT)CANON"
points mark a canonical word omitted entirely (no Word-tier interval at
all). These are greedily matched, in time order, against a hardcoded
copy of the fixed canonical passage text (`CANONICAL_TEXT`, transcribed
from the corpus's own tag-free files and cross-checked against the
published passage) -- 99.6% of canonical positions across all 180 files
match cleanly; the ~0.4% that don't are concentrated in a handful of
files with genuine word-order self-corrections in natural speech (e.g.
"when traveler... a traveler"), not a parser bug -- those positions get
no label (`None`) and are excluded, not guessed at.

Every speaker reads the *same* fixed passage, so (unlike L2-ARCTIC) there
is no per-speaker target-word construction: `align_words_gop`/
`align_words_gop_sf` are called once per speaker against the shared
`CANONICAL_TEXT`, and only the audio and the derived labels vary.

Usage:
    python scripts/eval_jaesop_word.py --data-root /data/J-AESOP --out results/jaesop_word.json
    python scripts/eval_jaesop_word.py --limit 10
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

from proscor import align, align_phone, stats as gopstats

CANONICAL_TEXT = (
    "THE NORTH WIND AND THE SUN WERE DISPUTING WHICH WAS THE STRONGER "
    "WHEN A TRAVELER CAME ALONG WRAPPED IN A WARM CLOAK THEY AGREED THAT "
    "THE ONE WHO FIRST SUCCEEDED IN MAKING THE TRAVELER TAKE HIS CLOAK "
    "OFF SHOULD BE CONSIDERED STRONGER THAN THE OTHER THEN THE NORTH "
    "WIND BLEW AS HARD AS HE COULD BUT THE MORE HE BLEW THE MORE "
    "CLOSELY DID THE TRAVELER FOLD HIS CLOAK AROUND HIM AND AT LAST THE "
    "NORTH WIND GAVE UP THE ATTEMPT THEN THE SUN SHONE OUT WARMLY AND "
    "IMMEDIATELY THE TRAVELER TOOK OFF HIS CLOAK AND SO THE NORTH WIND "
    "WAS OBLIGED TO CONFESS THAT THE SUN WAS THE STRONGER OF THE TWO"
).split()

_TAG_RE = re.compile(r"^\((SBT|DVN|RPT|INS|DSF|FLR)\)(.*)$")
_INSTANCE_SUFFIX_RE = re.compile(r"\(\d+\)$")


def _strip_instance(word: str) -> str:
    return _INSTANCE_SUFFIX_RE.sub("", word)


def parse_textgrid_tiers(path: Path) -> dict:
    """-> {tier_name: [(xmin, xmax, text), ...] for interval tiers,
    or [(time, mark), ...] for the point tier ("Misc")}."""
    text = path.read_text(encoding="utf-8")
    tiers = {}
    for block in re.split(r"item \[\d+\]:\n", text)[1:]:
        m = re.search(r'name = "([^"]*)"', block)
        if not m:
            continue
        name = m.group(1)
        if "points:" in block:
            items = [(float(n), mark) for n, mark in
                      re.findall(r'number = ([\d.]+)\s+mark = "([^"]*)"', block)]
        else:
            items = [(float(a), float(b), t) for a, b, t in
                      re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', block)]
        tiers[name] = items
    return tiers


def word_events(tiers: dict) -> list:
    """-> [(canonical_word, "correct"|"error"), ...] in time order, for
    every Word-tier production or Misc-tier deletion that resolves to a
    definite canonical-word outcome (RPT/DSF/INS repair attempts are
    dropped, not labeled)."""
    word_ev = [(xmin, t) for xmin, xmax, t in tiers.get("Word", []) if t.strip() and t.strip() != "-"]
    dlt_ev = [(n, mark) for n, mark in tiers.get("Misc", []) if mark.startswith("(DLT)")]
    events = sorted([(t, "word", v) for t, v in word_ev] + [(t, "dlt", v) for t, v in dlt_ev],
                     key=lambda e: e[0])

    out = []
    for _, kind, raw in events:
        if kind == "dlt":
            out.append((_strip_instance(raw[len("(DLT)"):]), "error"))
            continue
        m = _TAG_RE.match(raw)
        if m is None:
            out.append((_strip_instance(raw), "correct"))
            continue
        tag, rest = m.groups()
        if tag == "SBT":
            out.append((_strip_instance(rest.split("->")[0]), "error"))
        elif tag == "DVN":
            out.append((_strip_instance(rest), "error"))
        # RPT, INS, DSF, FLR: no resolved canonical-word outcome -- skip
    return out


def align_to_canonical(events: list, canonical: list, lookahead: int = 6) -> tuple:
    """Greedy positional match of `events` against `canonical` (both word
    sequences); returns (labels, n_unmatched) where labels[i] is
    "correct"/"error"/None (canonical position i never resolved to a
    match) -- see the module docstring for the ~0.4% unmatched rate and
    why (natural-speech word-order self-corrections, not a parser bug)."""
    labels = [None] * len(canonical)
    ci = 0
    n_unmatched = 0
    for word, label in events:
        matched = False
        for k in range(lookahead):
            idx = ci + k
            if idx >= len(canonical):
                break
            if canonical[idx] == word:
                labels[idx] = label
                ci = idx + 1
                matched = True
                break
        if not matched:
            n_unmatched += 1
    return labels, n_unmatched


def run(data_root: Path, limit: int = None, progress_every: int = 20) -> dict:
    ann_dir = data_root / "Annotations"
    audio_dir = data_root / "Audio"
    files = sorted(ann_dir.glob("J_*_6_01.TextGrid"))
    if limit:
        files = files[:limit]

    recs = []  # per (speaker, canonical-position) dict
    n_speakers_done = n_speakers_failed = 0
    total_unmatched = 0
    t0 = time.time()

    for i, ann_path in enumerate(files):
        speaker = ann_path.stem.rsplit("_6_01", 1)[0]
        wav_path = audio_dir / (ann_path.stem + ".wav")
        if not wav_path.exists():
            n_speakers_failed += 1
            continue
        tiers = parse_textgrid_tiers(ann_path)
        events = word_events(tiers)
        labels, n_unmatched = align_to_canonical(events, CANONICAL_TEXT)
        total_unmatched += n_unmatched

        samples, sr = sf.read(str(wav_path), dtype="float32")
        try:
            bpe_result = align.align_words_gop(samples, [w.lower() for w in CANONICAL_TEXT], sr=sr)
            phone_result = align_phone.align_words_gop(samples, CANONICAL_TEXT, sr=sr)["word_gop"]
        except Exception as e:
            print(f"  [warn] {speaker} failed: {e}", file=sys.stderr)
            n_speakers_failed += 1
            continue

        for word, label, br, pr in zip(CANONICAL_TEXT, labels, bpe_result, phone_result):
            if label is None or br is None or pr is None:
                continue
            recs.append({
                "speaker": speaker, "word": word, "label": label,
                "bpe_gop": br["gop"], "phone_gop": pr["gop"],
            })
        n_speakers_done += 1

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(files)} speakers ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1):.1f}s/speaker)", file=sys.stderr)

    return {
        "records": recs, "n_speakers_done": n_speakers_done, "n_speakers_failed": n_speakers_failed,
        "total_unmatched_events": total_unmatched, "elapsed_s": time.time() - t0,
    }


def compare(recs: list, engine_key: str) -> dict:
    if len(recs) < 2:
        return None
    x = [r[engine_key] for r in recs]
    y = [1 if r["label"] == "correct" else 0 for r in recs]
    c = [r["speaker"] for r in recs]
    return gopstats.cluster_bootstrap_pearson(x, y, c)


def paired_diff(recs: list) -> dict:
    if len(recs) < 2:
        return None
    x1 = [r["bpe_gop"] for r in recs]
    x2 = [r["phone_gop"] for r in recs]
    y = [1 if r["label"] == "correct" else 0 for r in recs]
    c = [r["speaker"] for r in recs]
    return gopstats.cluster_bootstrap_paired_diff(x1, x2, y, c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/J-AESOP")
    ap.add_argument("--limit", type=int, default=None, help="limit number of speakers (for smoke tests)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align.available() or not align_phone.available():
        print("proscor.align and proscor.align_phone optional deps must both be installed.", file=sys.stderr)
        sys.exit(1)

    results = run(Path(args.data_root), limit=args.limit)
    recs = results["records"]
    n_correct = sum(1 for r in recs if r["label"] == "correct")

    summary = {
        "n_speakers_done": results["n_speakers_done"],
        "n_speakers_failed": results["n_speakers_failed"],
        "n_canonical_positions": len(CANONICAL_TEXT),
        "n_words_scored": len(recs),
        "frac_correct": round(n_correct / len(recs), 4) if recs else None,
        "total_unmatched_events": results["total_unmatched_events"],
        "elapsed_s": round(results["elapsed_s"], 1),
        "bpe_vs_correct": compare(recs, "bpe_gop"),
        "phone_vs_correct": compare(recs, "phone_gop"),
        "bpe_vs_phone_paired_diff": paired_diff(recs),
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
