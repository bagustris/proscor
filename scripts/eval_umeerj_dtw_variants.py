#!/usr/bin/env python
"""Collect per-layer DTW-SSL costs and native-native cost statistics on
UME-ERJ's phonetic-sentence task (PLAN.md section 5m), for two
label-free improvements to `proscor/dtw_ssl.py`'s raw cost:

1. **Text-difficulty normalization.** The raw learner-vs-native cost
   carries a per-sentence offset (some sentences are just harder to align
   even native-to-native: the section 5l smoke test saw native-native
   cost vary 0.27-0.28 across sentences). With ~11 native templates per
   text there are ~55 native-native pairs per text -- a label-free,
   native-only estimate of that offset and its spread.
2. **Layer choice.** WavLM's final layer is the source paper's choice;
   layer-wise analyses generally put phonetic content in mid-upper
   layers. Sweeping layers *and picking the best on the same items being
   reported* would be tuning on the test set, so this script only
   *collects* the per-layer numbers; the offline analysis
   (`scripts/analyze_umeerj_dtw_variants.py`) selects the layer on one
   speaker-disjoint half and reports it on the other.

Same item selection and order as scripts/eval_umeerj_dtw_ssl.py (first N
S_PH_* items of sentence-segmental), so numbers line up with section 5l.
One JSONL line per item.

Usage:
    OMP_NUM_THREADS=4 python scripts/eval_umeerj_dtw_variants.py --limit 300 --out results/umeerj_dtw_variants.jsonl
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import dtw_ssl
from eval_umeerj_dtw_ssl import build_ae_index, load_je_to_ae, load_scores

LAYERS = (6, 9, 12, 15, 18, 21, 24, "last")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/UME-ERJ")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(12)

    root = Path(args.data_root)
    je_to_ae, ae_index, records = load_je_to_ae(root), build_ae_index(root), load_scores(root)
    items = []
    for rel, score, _n in records:
        m = je_to_ae.get(Path(rel).name)
        if m and m[0].startswith("S_PH_"):
            items.append((rel, score, m[0]))
    items = items[: args.limit]

    out = Path(args.out)
    done = sum(1 for _ in open(out)) if out.exists() else 0
    print(f"{len(items)} items, {done} done", file=sys.stderr)
    t0 = time.time()
    with open(out, "a") as fh:
        for i in range(done, len(items)):
            rel, human, ae_name = items[i]
            wav = root / "wav" / "JE" / rel
            tpaths = [root / "wav" / "AE" / spk / ae_name for spk in ae_index.get(ae_name, [])]
            tpaths = [p for p in tpaths if p.exists()]
            if not wav.exists() or len(tpaths) < 3:
                fh.write(json.dumps({"idx": i, "skipped": True}) + "\n"); fh.flush(); continue

            try:
                s, sr = sf.read(str(wav), dtype="float32")
                learner = dtw_ssl.embed_layers(s, LAYERS, sr=sr)
                temps = []
                for p in tpaths:
                    ts, tsr = sf.read(str(p), dtype="float32")
                    temps.append(dtw_ssl.embed_layers(ts, LAYERS, sr=tsr))
            except Exception as e:  # e.g. a zero-length/too-short recording
                print(f"  [skip] item {i} ({rel}): {e}", file=sys.stderr)
                fh.write(json.dumps({"idx": i, "skipped": True, "error": str(e)[:120]}) + "\n"); fh.flush(); continue

            rec = {"idx": i, "speaker": str(Path(rel).parent), "human": human,
                   "ae": ae_name, "n_templates": len(temps), "layers": {}}
            for L in LAYERS:
                lc = [dtw_ssl.dtw_cost(learner[L], t[L]) for t in temps]
                nn = [dtw_ssl.dtw_cost(temps[a][L], temps[b][L])
                      for a, b in itertools.combinations(range(len(temps)), 2)]
                rec["layers"][str(L)] = {"learner_costs": lc, "nn_mean": float(np.mean(nn)),
                                         "nn_std": float(np.std(nn))}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t0; n = i + 1 - done
                print(f"  {i+1}/{len(items)} ({el:.0f}s, {el/n:.1f}s/item, ETA {(len(items)-i-1)*el/n/60:.0f}min)",
                      file=sys.stderr)


if __name__ == "__main__":
    main()
