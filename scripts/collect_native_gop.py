#!/usr/bin/env python
"""Per-phone GOP distributions on *native* speech (PLAN.md section 5m),
the raw material for label-free per-phone calibration: instead of one
global scale for "how much posterior deficit is bad", score each learner
phone against how that same phone behaves when native speakers say it.
Uses only audio + transcript of native speakers (UME-ERJ's 20 American
reference speakers, S_PH_* sentences) -- no pronunciation labels anywhere,
so the resulting method stays zero-shot in the sense this project uses.

One JSONL line per native utterance: espeak phones and per-phone GOP for
the same four variants scripts/collect_so762_full.py records (xlsr-53 and
lv-60, posterior-deficit and GOP-SF), so calibration statistics can be
built for any of them offline. Sampling is seeded and spread across
speakers and sentences.

Usage:
    OMP_NUM_THREADS=4 python scripts/collect_native_gop.py --n 1200 --out results/native_gop.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from proscor import align_phone
from eval_umeerj_dtw_ssl import build_ae_index, load_je_to_ae

_WORD_RE = re.compile(r"[A-Za-z']+")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/UME-ERJ")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(4)

    root = Path(args.data_root)
    ae_text = {ae: text for _je, (ae, text) in load_je_to_ae(root).items() if ae.startswith("S_PH_")}
    pairs = [(spk, ae) for ae, spks in build_ae_index(root).items() if ae in ae_text for spk in spks]
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n]

    out = Path(args.out)
    done = sum(1 for _ in open(out)) if out.exists() else 0
    print(f"{len(pairs)} native utterances, {done} done", file=sys.stderr)
    t0 = time.time()
    with open(out, "a") as fh:
        for i in range(done, len(pairs)):
            spk, ae = pairs[i]
            words = _WORD_RE.findall(ae_text[ae])
            samples, sr = sf.read(str(root / "wav" / "AE" / spk / ae), dtype="float32")
            try:
                variants = {
                    "xlsr_post": align_phone.align_words_gop(samples, words, sr=sr, model_id=align_phone.TORCH_MODEL_REPO),
                    "xlsr_sf": align_phone.align_words_gop_sf(samples, words, sr=sr, model_id=align_phone.TORCH_MODEL_REPO),
                    "lv60_post": align_phone.align_words_gop(samples, words, sr=sr),
                    "lv60_sf": align_phone.align_words_gop_sf(samples, words, sr=sr),
                }
            except Exception as e:
                print(f"  [warn] {spk}/{ae}: {e}", file=sys.stderr)
                fh.write(json.dumps({"idx": i, "failed": True}) + "\n"); fh.flush(); continue
            rec = {"idx": i, "speaker": spk, "ae": ae, "text": ae_text[ae], "words": []}
            for j, w in enumerate(words):
                rec["words"].append({"text": w, **{
                    name: {"phones": [p["phone"] if p else None for p in res["phone_gop"][j]],
                           "gops": [p["gop"] if p else None for p in res["phone_gop"][j]]}
                    for name, res in variants.items()}})
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0; n = i + 1 - done
                print(f"  {i+1}/{len(pairs)} ({el:.0f}s, {el/n:.1f}s/utt, ETA {(len(pairs)-i-1)*el/n/60:.0f}min)",
                      file=sys.stderr)


if __name__ == "__main__":
    main()
