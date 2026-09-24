#!/usr/bin/env python
"""DTW-SSL (McIntosh/Smit/Saito/Minematsu/Kamper, arXiv:2607.13721) vs.
xlsr-53 GOP-lite on UME-ERJ's "Phones (sentence)" subset -- PLAN.md
section 5l, the first like-for-like test of `proscor/dtw_ssl.py` against
this project's own posterior-deficit/GOP-SF phone scoring, on the exact
task the paper reports r=0.576 for (vs. this project's own broader-subset
0.432 from section 5b -- recomputed here on the *matched* item set, not
reused from that older, differently-scoped number; see PLAN.md section 5k
for why that matters).

**Item selection:** UME-ERJ's `lbl/{male,female}-sentence/segmental/scores.lst`
rates a mix of materials (phonetic-balance sentences, difficult-phoneme
sentences, sentences actually used in phoneme training, *and* prosody
("S_PR_*") sentences also segmentally rated). `doc/JEcontent/tab/sentence{1,2}.tab`
maps each learner recording filename to BOTH its text and the "selection
filename" used for the corresponding native (AE) recordings; filtering
to `S_PH_*`-prefixed selection filenames isolates the phonetic-content
task specifically (B/D/E = TIMIT-phoneme-balanced/Japanese-difficult/
actually-used-in-training, 460+32+100=592 texts here -- close to but not
identical to the paper's 625, plausibly a corpus-version/subset
difference, not a methodology error).

**Native templates:** each `S_PH_*` selection filename recurs identically
across multiple `wav/AE/<SPK>/` directories (typically ~11 of UME-ERJ's
20 American reference speakers per item) -- every matching file found is
used as a template, DTW-SSL's mean-cost aggregation averaging over all of
them (the paper's main method).

**Pairing discipline:** both DTW-SSL and xlsr-53 GOP-lite (posterior-deficit
and GOP-SF) are scored on the exact same filtered item set in the same
loop, so the paired significance test compares them on identical items,
not a stale number from a different item set (PLAN.md section 5k's
lesson). Speaker-cluster bootstrap resamples *learner* speakers only --
native templates are shared across every learner of a text and must never
be treated as an independent cluster.

DTW-SSL's score is a *distance* (lower = closer to native = better),
opposite in sign convention to GOP (higher = better) -- reported here as
`-dtw_cost` so its correlation sign is directly comparable to GOP's.

Usage:
    python scripts/eval_umeerj_dtw_ssl.py --out results/umeerj_dtw_ssl.json
    python scripts/eval_umeerj_dtw_ssl.py --limit 30
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

from proscor import align_phone, dtw_ssl, stats as gopstats

UME_ROOT_DEFAULT = "/data/UME-ERJ"
_AE_PREFIX_RE = re.compile(r"_\d+\.wav$")


def load_je_to_ae(data_root: Path) -> dict:
    """[learner recording filename] -> (AE selection filename, text),
    tab1 overriding tab2 on overlap (near-duplicate tables; see
    scripts/eval_umeerj.py's identical merge for why)."""
    mapping = {}
    tab_dir = data_root / "doc" / "JEcontent" / "tab"
    for n in (2, 1):
        path = tab_dir / f"sentence{n}.tab"
        with open(path, encoding="shift_jis") as f:
            for line in f:
                parts = re.split(r" {2,}", line.strip())
                if len(parts) >= 3:
                    mapping[parts[0]] = (parts[1], parts[2])
    return mapping


def load_scores(data_root: Path) -> list:
    """Combine male/female sentence-segmental scores.lst ->
    [(rel_wav_path, mean_score, n_raters), ...]. Mirrors
    scripts/eval_umeerj.py's load_scores for the sentence/segmental
    category specifically."""
    records = []
    for gender in ("male", "female"):
        path = data_root / "lbl" / f"{gender}-sentence" / "segmental" / "scores.lst"
        with open(path, encoding="shift_jis") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                rel_path, scores = parts[0], [float(s) for s in parts[1:]]
                if scores:
                    records.append((rel_path, float(np.mean(scores)), len(scores)))
    return records


def ae_template_paths(data_root: Path, ae_filename: str, ae_index: dict) -> list:
    return [data_root / "wav" / "AE" / spk / ae_filename
            for spk in ae_index.get(ae_filename, [])]


def build_ae_index(data_root: Path) -> dict:
    """[AE filename] -> [speaker dirs that have it] -- avoids re-globbing
    20 speaker directories per item."""
    index = {}
    ae_root = data_root / "wav" / "AE"
    for spk_dir in ae_root.iterdir():
        if not spk_dir.is_dir():
            continue
        for f in spk_dir.glob("*.wav"):
            index.setdefault(f.name, []).append(spk_dir.name)
    return index


_WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> list:
    return _WORD_RE.findall(text)


def run(data_root: Path, cache_dir: Path, limit: int = None, min_templates: int = 3,
        progress_every: int = 20) -> dict:
    je_to_ae = load_je_to_ae(data_root)
    ae_index = build_ae_index(data_root)
    records = load_scores(data_root)

    # filter to the S_PH_* phonetic-content task (see module docstring)
    items = []
    for rel_path, score, n_raters in records:
        mapped = je_to_ae.get(Path(rel_path).name)
        if mapped is None:
            continue
        ae_filename, text = mapped
        if not ae_filename.startswith("S_PH_"):
            continue
        items.append((rel_path, score, ae_filename, text))
    print(f"  {len(items)}/{len(records)} sentence-segmental items are S_PH_* "
          f"(phonetic-content task)", file=sys.stderr)
    if limit:
        items = items[:limit]

    out = {k: [] for k in ("dtw_mean", "dtw_min", "xlsr_post", "xlsr_sf", "human", "speaker", "n_templates")}
    n_no_audio = n_too_few_templates = n_failed = 0
    t0 = time.time()

    for i, (rel_path, human_score, ae_filename, text) in enumerate(items):
        learner_wav = data_root / "wav" / "JE" / rel_path
        if not learner_wav.exists():
            n_no_audio += 1
            continue
        template_paths = [p for p in ae_template_paths(data_root, ae_filename, ae_index) if p.exists()]
        if len(template_paths) < min_templates:
            n_too_few_templates += 1
            continue

        words = tokenize(text)
        try:
            learner_samples, sr = sf.read(str(learner_wav), dtype="float32")
            learner_emb = dtw_ssl.embed(learner_samples, sr=sr)
            template_embs = [dtw_ssl.embed_wav_cached(p, cache_dir) for p in template_paths]
            dtw = dtw_ssl.score(learner_emb, template_embs)

            xlsr_post = align_phone.align_words_gop(learner_samples, words, sr=sr,
                                                      model_id=align_phone.TORCH_MODEL_REPO)
            xlsr_sf = align_phone.align_words_gop_sf(learner_samples, words, sr=sr,
                                                       model_id=align_phone.TORCH_MODEL_REPO)
            post_vals = [r["gop"] for r in xlsr_post["word_gop"] if r is not None]
            sf_vals = [r["gop"] for r in xlsr_sf["word_gop"] if r is not None]
            if not post_vals or not sf_vals:
                n_failed += 1
                continue
        except Exception as e:
            print(f"  [warn] {rel_path} failed: {e}", file=sys.stderr)
            n_failed += 1
            continue

        out["dtw_mean"].append(-dtw["mean"])  # negate: distance -> goodness-comparable sign
        out["dtw_min"].append(-dtw["min"])
        out["xlsr_post"].append(float(np.mean(post_vals)))
        out["xlsr_sf"].append(float(np.mean(sf_vals)))
        out["human"].append(human_score)
        out["speaker"].append(str(Path(rel_path).parent))  # learner site/speaker, e.g. "TOK/F01"
        out["n_templates"].append(len(template_paths))

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(items)} items ({elapsed:.0f}s, "
                  f"{elapsed / (i + 1):.1f}s/item)", file=sys.stderr)

    return {
        **out, "n_items_total": len(items), "n_no_audio": n_no_audio,
        "n_too_few_templates": n_too_few_templates, "n_failed": n_failed,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=UME_ROOT_DEFAULT)
    ap.add_argument("--cache-dir", default="results/dtw_ssl_template_cache")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-templates", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not align_phone.available() or not dtw_ssl.available():
        print("proscor.align_phone and proscor.dtw_ssl optional deps must both be installed.",
              file=sys.stderr)
        sys.exit(1)

    data_root = Path(args.data_root)
    r = run(data_root, Path(args.cache_dir), limit=args.limit, min_templates=args.min_templates)

    def corr(x, key):
        return gopstats.cluster_bootstrap_pearson(r[x], r["human"], r["speaker"])

    def diff(x1, x2):
        return gopstats.cluster_bootstrap_paired_diff(r[x1], r[x2], r["human"], r["speaker"])

    summary = {
        "n_items_total": r["n_items_total"],
        "n_items_scored": len(r["human"]),
        "n_no_audio": r["n_no_audio"],
        "n_too_few_templates": r["n_too_few_templates"],
        "n_failed": r["n_failed"],
        "elapsed_s": round(r["elapsed_s"], 1),
        "note": "dtw scores are -dtw_cost (higher = better, comparable sign to GOP). "
                "diff = r(dtw, human) - r(xlsr, human); positive+significant means DTW-SSL wins",
        "dtw_mean_vs_human": corr("dtw_mean", "human"),
        "dtw_min_vs_human": corr("dtw_min", "human"),
        "xlsr_post_vs_human": corr("xlsr_post", "human"),
        "xlsr_sf_vs_human": corr("xlsr_sf", "human"),
        "dtw_mean_vs_xlsr_post_paired_diff": diff("dtw_mean", "xlsr_post"),
        "dtw_mean_vs_xlsr_sf_paired_diff": diff("dtw_mean", "xlsr_sf"),
        "dtw_mean_vs_dtw_min_paired_diff": diff("dtw_mean", "dtw_min"),
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
