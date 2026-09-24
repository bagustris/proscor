#!/usr/bin/env python
"""DTW-SSL vs. xlsr-53 GOP-lite on L2-ARCTIC -- the second-corpus
confirmation of scripts/eval_umeerj_dtw_ssl.py's result (PLAN.md section
5l). L2-ARCTIC speakers read the CMU ARCTIC "arctic_aXXXX" prompt set,
which is *also* the prompt set of CMU ARCTIC's own native speakers (bdl,
US male; slt, US female -- downloaded separately from festvox.org, not
part of L2-ARCTIC itself), so every L2-ARCTIC utterance has a same-text
native reference by construction: no tab-file/prompt-matching needed here,
unlike UME-ERJ, since the filenames are already identical
(`arctic_a0001.wav` etc. on both sides).

**Label granularity mismatch, handled explicitly:** L2-ARCTIC's expert
annotations are per-*phone* (correct/substitution/deletion, via
`scripts/eval_l2arctic_phone.py`'s TextGrid parser), but DTW-SSL produces
one score per *utterance* (a single DTW alignment against the whole
native template, not phone-segmentable without further work -- see
PLAN.md section 5l for why that's future work, not done here). This
script therefore uses the utterance-level fraction of phones tagged
"correct" as the accuracy label -- coarser than the phone-level PCC
`eval_l2arctic_phone.py` reports, but the right granularity to compare
DTW-SSL and xlsr-53 GOP-lite against each other on equal footing (xlsr-53
is *also* averaged to one utterance-level GOP here, even though it could
be scored at phone level -- apples to apples is the point of this
script, not xlsr-53's best possible number).

Only 2 native templates are available per item here (bdl + slt), far
fewer than UME-ERJ's ~11 -- noted, not hidden: the paper's own template-
count ablation found diminishing returns above ~5, so 2 is a real but
modest disadvantage for DTW-SSL's averaging, not disqualifying.

Usage:
    python scripts/eval_l2arctic_dtw_ssl.py --out results/l2arctic_dtw_ssl.json
    python scripts/eval_l2arctic_dtw_ssl.py --speakers BWC --limit 20
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

from proscor import align_phone, dtw_ssl, stats as gopstats
from eval_l2arctic_phone import _WORD_RE, parse_textgrid, words_and_phones

NATIVE_ROOT_DEFAULT = "/data/CMU-ARCTIC-native"
NATIVE_SPEAKERS = ("bdl", "slt")


def run(speaker_dirs: list, native_root: Path, cache_dir: Path, limit: int = None,
        progress_every: int = 20) -> dict:
    native_dirs = [native_root / spk for spk in NATIVE_SPEAKERS]

    utt_files = []
    for spk_dir in speaker_dirs:
        for f in sorted((spk_dir / "annotation").glob("*.TextGrid")):
            utt_files.append((spk_dir.name, f))
    if limit:
        utt_files = utt_files[:limit]

    out = {k: [] for k in ("dtw_mean", "dtw_min", "xlsr_post", "xlsr_sf", "frac_correct", "speaker", "n_templates")}
    n_no_template = n_failed = 0
    t0 = time.time()

    for i, (speaker, ann_path) in enumerate(utt_files):
        wav_path = ann_path.parents[1] / "wav" / (ann_path.stem + ".wav")
        if not wav_path.exists():
            continue
        words = words_and_phones(parse_textgrid(ann_path))
        words = [w for w in words if _WORD_RE.fullmatch(w["text"].lower()) and w["phones"]]
        if not words:
            continue

        template_paths = [d / (ann_path.stem + ".wav") for d in native_dirs]
        template_paths = [p for p in template_paths if p.exists()]
        if len(template_paths) < 1:
            n_no_template += 1
            continue

        all_tags = [tag for w in words for _, tag in w["phones"]]
        frac_correct = sum(1 for t in all_tags if t == "correct") / len(all_tags)
        texts = [w["text"].lower() for w in words]

        try:
            learner_samples, sr = sf.read(str(wav_path), dtype="float32")
            learner_emb = dtw_ssl.embed(learner_samples, sr=sr)
            template_embs = [dtw_ssl.embed_wav_cached(p, cache_dir) for p in template_paths]
            dtw = dtw_ssl.score(learner_emb, template_embs)

            xlsr_post = align_phone.align_words_gop(learner_samples, texts, sr=sr,
                                                      model_id=align_phone.TORCH_MODEL_REPO)
            xlsr_sf = align_phone.align_words_gop_sf(learner_samples, texts, sr=sr,
                                                       model_id=align_phone.TORCH_MODEL_REPO)
            post_vals = [r["gop"] for r in xlsr_post["word_gop"] if r is not None]
            sf_vals = [r["gop"] for r in xlsr_sf["word_gop"] if r is not None]
            if not post_vals or not sf_vals:
                n_failed += 1
                continue
        except Exception as e:
            print(f"  [warn] {speaker}/{ann_path.name} failed: {e}", file=sys.stderr)
            n_failed += 1
            continue

        out["dtw_mean"].append(-dtw["mean"])
        out["dtw_min"].append(-dtw["min"])
        out["xlsr_post"].append(float(np.mean(post_vals)))
        out["xlsr_sf"].append(float(np.mean(sf_vals)))
        out["frac_correct"].append(frac_correct)
        out["speaker"].append(speaker)
        out["n_templates"].append(len(template_paths))

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(utt_files)} utterances ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1):.1f}s/utt)", file=sys.stderr)

    return {**out, "n_utterances_total": len(utt_files), "n_no_template": n_no_template,
            "n_failed": n_failed, "elapsed_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/L2-ARCTIC/all")
    ap.add_argument("--native-root", default=NATIVE_ROOT_DEFAULT)
    ap.add_argument("--cache-dir", default="results/dtw_ssl_template_cache")
    ap.add_argument("--speakers", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align_phone.available() or not dtw_ssl.available():
        print("proscor.align_phone and proscor.dtw_ssl optional deps must both be installed.",
              file=sys.stderr)
        sys.exit(1)

    root = Path(args.data_root)
    speaker_dirs = sorted(p for p in root.glob("*/*") if (p / "annotation").is_dir())
    if args.speakers:
        speaker_dirs = [p for p in speaker_dirs if p.name in args.speakers]
    print(f"Speakers: {[p.name for p in speaker_dirs]}", file=sys.stderr)

    r = run(speaker_dirs, Path(args.native_root), Path(args.cache_dir), limit=args.limit)

    def corr(key):
        return gopstats.cluster_bootstrap_pearson(r[key], r["frac_correct"], r["speaker"])

    def diff(k1, k2):
        return gopstats.cluster_bootstrap_paired_diff(r[k1], r[k2], r["frac_correct"], r["speaker"])

    summary = {
        "speakers": [p.name for p in speaker_dirs],
        "n_utterances_total": r["n_utterances_total"],
        "n_utterances_scored": len(r["frac_correct"]),
        "n_no_template": r["n_no_template"],
        "n_failed": r["n_failed"],
        "elapsed_s": round(r["elapsed_s"], 1),
        "note": "dtw scores are -dtw_cost (higher = better). label = utterance-level "
                "fraction of phones tagged correct. diff = r(dtw,label) - r(xlsr,label); "
                "positive+significant means DTW-SSL wins",
        "dtw_mean_vs_label": corr("dtw_mean"),
        "dtw_min_vs_label": corr("dtw_min"),
        "xlsr_post_vs_label": corr("xlsr_post"),
        "xlsr_sf_vs_label": corr("xlsr_sf"),
        "dtw_mean_vs_xlsr_post_paired_diff": diff("dtw_mean", "xlsr_post"),
        "dtw_mean_vs_xlsr_sf_paired_diff": diff("dtw_mean", "xlsr_sf"),
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
