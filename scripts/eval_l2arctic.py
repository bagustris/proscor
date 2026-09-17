#!/usr/bin/env python
"""Disentangle the acoustic/age-domain-match hypothesis from the
L1-specific-error-profile hypothesis for the BPE-vs-phone-model ranking
flip found between speechocean762 (Mandarin-L1 *children*, BPE model wins
at word/utterance level, PLAN.md section 5a) and UME-ERJ (Japanese-L1
*adults*, phone model wins, section 5b): L2-ARCTIC has **adult Mandarin-L1**
speakers, holding L1 fixed relative to speechocean762 and age fixed
relative to UME-ERJ.

Uses the KoelLabs/L2Arctic HF mirror (https://huggingface.co/datasets/KoelLabs/L2Arctic,
gated -- run `huggingface-cli login` with a token approved for it), not the
original TAMU/Kaggle distribution: it's a clean, pre-processed parquet of
the "scripted" (3,599 utterances, all 24 speakers, 6 L1s) and "spontaneous"
splits, audio already 16kHz float32. Trade-off versus the raw TextGrid
annotations (which give an expert-aligned canonical-phone / perceived-phone
/ error-tag triple per phone): this mirror only gives two *unaligned* IPA
strings per utterance -- `g2p` (canonical) and `ipa` (perceived, expert-
verified). Re-deriving a phone-level alignment between them (mirroring
proscor.align_phone.reconcile_phones) is a documented follow-up, not done
here. Instead, this script uses a coarser but still corpus-derived signal:
normalized character-level Levenshtein distance between `g2p` and `ipa` as
an utterance-level pronunciation-error proxy (higher = more deviation from
canonical = expected to correlate *negatively* with GOP) -- the same
granularity UME-ERJ's holistic ratings gave, not a phone-level metric.

Usage:
    pip install -r requirements-eval.txt
    huggingface-cli login   # token must be approved for KoelLabs/L2Arctic
    python scripts/eval_l2arctic.py                          # all languages
    python scripts/eval_l2arctic.py --languages Chinese       # the core disentangling test
    python scripts/eval_l2arctic.py --limit 100
"""
import argparse
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proscor import align, align_phone, stats as gopstats

_WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> list:
    return _WORD_RE.findall(text)


def load_split(split: str) -> list:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download("KoelLabs/L2Arctic", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(path).to_pylist()


def char_error_rate(canonical: str, perceived: str) -> float:
    from rapidfuzz.distance import Levenshtein

    denom = max(len(canonical), len(perceived), 1)
    return Levenshtein.distance(canonical, perceived) / denom


def run(rows: list, progress_every: int = 200) -> dict:
    records = []  # per-row dicts
    t0 = time.time()
    n_bpe_failed = n_phone_failed = 0

    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        words = tokenize(row["text"])
        if not words:
            continue
        cer = char_error_rate(row["g2p"], row["ipa"])

        try:
            bpe_result = align.align_words_gop(samples, [w.lower() for w in words], sr=sr)
            bpe_vals = [r["gop"] for r in bpe_result if r is not None]
            bpe_gop = float(np.mean(bpe_vals)) if bpe_vals else None
        except Exception as e:
            print(f"  [warn] bpe failed on speaker {row['speaker_code']}: {e}", file=sys.stderr)
            bpe_gop = None
        if bpe_gop is None:
            n_bpe_failed += 1

        try:
            phone_result = align_phone.align_words_gop(samples, words, sr=sr)
            phone_vals = [r["gop"] for r in phone_result["word_gop"] if r is not None]
            phone_gop = float(np.mean(phone_vals)) if phone_vals else None
        except Exception as e:
            print(f"  [warn] phone failed on speaker {row['speaker_code']}: {e}", file=sys.stderr)
            phone_gop = None
        if phone_gop is None:
            n_phone_failed += 1

        records.append({
            "speaker": row["speaker_code"], "language": row["speaker_native_language"],
            "cer": cer, "bpe_gop": bpe_gop, "phone_gop": phone_gop,
        })

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {"records": records, "n_bpe_failed": n_bpe_failed, "n_phone_failed": n_phone_failed,
            "elapsed_s": time.time() - t0}


def correlations(gop: list, cer: list) -> dict:
    from scipy.stats import pearsonr, spearmanr

    pairs = [(g, c) for g, c in zip(gop, cer) if g is not None]
    if len(pairs) < 2:
        return {"pearson_r": None, "spearman_rho": None, "n": len(pairs)}
    gs, cs = zip(*pairs)
    pr, _ = pearsonr(gs, cs)
    sr, _ = spearmanr(gs, cs)
    return {"pearson_r": round(float(pr), 4), "spearman_rho": round(float(sr), 4), "n": len(pairs)}


def correlations_ci(gop: list, cer: list, speakers: list) -> dict:
    triples = [(g, c, s) for g, c, s in zip(gop, cer, speakers) if g is not None]
    if len(triples) < 2:
        return None
    gs, cs, ss = zip(*triples)
    return gopstats.cluster_bootstrap_pearson(gs, cs, ss)


def paired_diff_ci(bpe: list, phone: list, cer: list, speakers: list) -> dict:
    """Restrict to utterances where BOTH engines produced a score, then run
    the paired speaker-cluster bootstrap on r(bpe,cer) - r(phone,cer) -- the
    real test for the 5c "near-tie" (PLAN.md section 5c), since comparing
    two independently-bootstrapped marginal CIs would be a weaker test."""
    quads = [(b, p, c, s) for b, p, c, s in zip(bpe, phone, cer, speakers) if b is not None and p is not None]
    if len(quads) < 2:
        return None
    bs, ps, cs, ss = zip(*quads)
    return gopstats.cluster_bootstrap_paired_diff(bs, ps, cs, ss)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="scripted", choices=["scripted", "spontaneous"])
    ap.add_argument("--languages", nargs="+", default=None,
                     help="filter to these speaker_native_language values (default: all 6)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align.available() or not align_phone.available():
        print("proscor.align and proscor.align_phone optional deps must both be installed.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading L2-ARCTIC [{args.split}] ...", file=sys.stderr)
    rows = load_split(args.split)
    if args.languages:
        rows = [r for r in rows if r["speaker_native_language"] in args.languages]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Evaluating {len(rows)} utterances ...", file=sys.stderr)

    results = run(rows)
    records = results["records"]

    by_lang = defaultdict(lambda: {"bpe": [], "phone": [], "cer": [], "speaker": []})
    by_speaker = defaultdict(lambda: {"bpe": [], "phone": [], "cer": [], "language": None})
    for r in records:
        by_lang[r["language"]]["bpe"].append(r["bpe_gop"])
        by_lang[r["language"]]["phone"].append(r["phone_gop"])
        by_lang[r["language"]]["cer"].append(r["cer"])
        by_lang[r["language"]]["speaker"].append(r["speaker"])
        by_lang["ALL"]["bpe"].append(r["bpe_gop"])
        by_lang["ALL"]["phone"].append(r["phone_gop"])
        by_lang["ALL"]["cer"].append(r["cer"])
        by_lang["ALL"]["speaker"].append(r["speaker"])
        by_speaker[r["speaker"]]["bpe"].append(r["bpe_gop"])
        by_speaker[r["speaker"]]["phone"].append(r["phone_gop"])
        by_speaker[r["speaker"]]["cer"].append(r["cer"])
        by_speaker[r["speaker"]]["language"] = r["language"]

    summary = {
        "split": args.split,
        "n_utterances": len(rows),
        "n_scored": len(records),
        "n_bpe_failed": results["n_bpe_failed"],
        "n_phone_failed": results["n_phone_failed"],
        "elapsed_s": round(results["elapsed_s"], 1),
        "note": "correlations are GOP vs. char-error-rate (canonical/perceived IPA edit distance); "
                "expected sign is NEGATIVE (higher error rate = worse pronunciation = lower GOP)",
        "by_language": {
            lang: {
                "n": len(d["cer"]),
                "bpe_vs_cer": correlations(d["bpe"], d["cer"]),
                "bpe_vs_cer_speaker_cluster_ci": correlations_ci(d["bpe"], d["cer"], d["speaker"]),
                "phone_vs_cer": correlations(d["phone"], d["cer"]),
                "phone_vs_cer_speaker_cluster_ci": correlations_ci(d["phone"], d["cer"], d["speaker"]),
                "bpe_vs_phone_paired_diff": paired_diff_ci(d["bpe"], d["phone"], d["cer"], d["speaker"]),
            }
            for lang, d in sorted(by_lang.items())
        },
        "by_speaker": {
            spk: {
                "language": d["language"],
                "n": len(d["cer"]),
                "bpe_vs_cer": correlations(d["bpe"], d["cer"]),
                "phone_vs_cer": correlations(d["phone"], d["cer"]),
            }
            for spk, d in sorted(by_speaker.items())
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
