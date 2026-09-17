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

from proscor import align_phone, stats as gopstats

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


def run(speaker_dirs: list, limit: int = None, progress_every: int = 200) -> dict:
    by_speaker = defaultdict(lambda: {"gop": [], "correct": [], "gop_by_tag": defaultdict(list)})
    n_utt = n_utt_failed = n_words_total = 0
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

        spk_data = by_speaker[speaker]
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
                spk_data["gop"].append(g)
                spk_data["correct"].append(1 if tag == "correct" else 0)
                spk_data["gop_by_tag"][tag].append(g)

        n_utt += 1
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(utt_files)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1) * 1000:.0f}ms/utt)", file=sys.stderr)

    return {
        "by_speaker": {spk: {"gop": d["gop"], "correct": d["correct"], "gop_by_tag": dict(d["gop_by_tag"])}
                       for spk, d in by_speaker.items()},
        "n_utterances": n_utt, "n_utterances_failed": n_utt_failed,
        "n_words_total": n_words_total,
        "n_phones_scored": sum(len(d["gop"]) for d in by_speaker.values()),
        "elapsed_s": time.time() - t0,
    }


SPEAKERS_BY_LANG = {
    "Arabic": ["ABA", "YBAA", "SKA", "ZHAA"],
    "Mandarin": ["BWC", "TXHC", "LXC", "NCC"],
    "Hindi": ["ASI", "RRBI", "SVBI", "TNI"],
    "Korean": ["HKK", "YKWK", "HJK", "YDCK"],
    "Spanish": ["EBVS", "ERMS", "MBMPS", "NJS"],
    "Vietnamese": ["HQTV", "TLV", "PNV", "THV"],
}
LANG_BY_SPEAKER = {spk: lang for lang, spks in SPEAKERS_BY_LANG.items() for spk in spks}


def correlations(gop: list, correct: list) -> dict:
    from scipy.stats import pearsonr, spearmanr

    if len(gop) < 2:
        return {"pearson_r": None, "spearman_rho": None, "n": len(gop)}
    pr, _ = pearsonr(gop, correct)
    sr_, _ = spearmanr(gop, correct)
    return {"pearson_r": round(float(pr), 4), "spearman_rho": round(float(sr_), 4), "n": len(gop)}


def tag_breakdown(gop_by_tag: dict) -> dict:
    return {
        tag: {"n": len(vals), "mean_gop": round(float(np.mean(vals)), 3),
              "median_gop": round(float(np.median(vals)), 3)}
        for tag, vals in sorted(gop_by_tag.items()) if vals
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
    by_speaker = results["by_speaker"]

    by_lang = defaultdict(lambda: {"gop": [], "correct": [], "speaker": [], "gop_by_tag": defaultdict(list)})
    gop_pooled, correct_pooled, speaker_pooled = [], [], []
    gop_by_tag_pooled = defaultdict(list)
    for spk, d in by_speaker.items():
        lang = LANG_BY_SPEAKER.get(spk, "unknown")
        by_lang[lang]["gop"].extend(d["gop"])
        by_lang[lang]["correct"].extend(d["correct"])
        by_lang[lang]["speaker"].extend([spk] * len(d["gop"]))
        for tag, vals in d["gop_by_tag"].items():
            by_lang[lang]["gop_by_tag"][tag].extend(vals)
            gop_by_tag_pooled[tag].extend(vals)
        gop_pooled.extend(d["gop"])
        correct_pooled.extend(d["correct"])
        speaker_pooled.extend([spk] * len(d["gop"]))

    # Phones cluster within speakers (a speaker who mispronounces a sound
    # produces many correlated error phones), so the naive Fisher-z CI on
    # n=118,455 phones would be nonsense (~+-0.006). The true unit of
    # independence is the speaker (24 total, 4 per language) -- see
    # PLAN.md section 5c for why a speaker-level cluster bootstrap replaces
    # the point estimates below, and proscor/stats.py for the method.
    speaker_r = {
        spk: correlations(d["gop"], d["correct"])["pearson_r"]
        for spk, d in by_speaker.items()
    }
    kw = gopstats.kruskal_by_group(
        [speaker_r[spk] for spk in speaker_r if speaker_r[spk] is not None],
        [LANG_BY_SPEAKER.get(spk, "unknown") for spk in speaker_r if speaker_r[spk] is not None],
    )

    summary = {
        "speakers": [p.name for p in all_speaker_dirs],
        "n_utterances": results["n_utterances"],
        "n_utterances_failed": results["n_utterances_failed"],
        "n_words_total": results["n_words_total"],
        "n_phones_scored": results["n_phones_scored"],
        "elapsed_s": round(results["elapsed_s"], 1),
        "pooled": {
            "frac_correct": round(float(np.mean(correct_pooled)), 4) if correct_pooled else None,
            "phone_level_gop_vs_correct": correlations(gop_pooled, correct_pooled),
            "phone_level_gop_vs_correct_speaker_cluster_ci": (
                gopstats.cluster_bootstrap_pearson(gop_pooled, correct_pooled, speaker_pooled)
                if gop_pooled else None
            ),
            "gop_by_error_tag": tag_breakdown(gop_by_tag_pooled),
        },
        "by_language": {
            lang: {
                "n_speakers": len(SPEAKERS_BY_LANG.get(lang, [])),
                "frac_correct": round(float(np.mean(d["correct"])), 4) if d["correct"] else None,
                "phone_level_gop_vs_correct": correlations(d["gop"], d["correct"]),
                "phone_level_gop_vs_correct_speaker_cluster_ci": (
                    gopstats.cluster_bootstrap_pearson(d["gop"], d["correct"], d["speaker"])
                    if d["gop"] else None
                ),
                "gop_by_error_tag": tag_breakdown(d["gop_by_tag"]),
            }
            for lang, d in sorted(by_lang.items())
        },
        "by_speaker": {
            spk: {
                "language": LANG_BY_SPEAKER.get(spk, "unknown"),
                "frac_correct": round(float(np.mean(d["correct"])), 4) if d["correct"] else None,
                "phone_level_gop_vs_correct": correlations(d["gop"], d["correct"]),
            }
            for spk, d in sorted(by_speaker.items())
        },
        "language_effect_kruskal_wallis": {
            "note": "H-test on the 24 per-speaker point-biserial r's (4 per "
                    "language), i.e. speaker is the unit -- not a test on "
                    "pooled phones.",
            **kw,
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
