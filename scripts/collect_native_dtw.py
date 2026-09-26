#!/usr/bin/env python
"""Per-phone DTW-SSL costs of *native* speech, for label-free calibration of
the DTW cost (PLAN.md section 5m) -- the DTW counterpart of
scripts/collect_native_gop.py, over the same sampled UME-ERJ American-speaker
utterances (same seed/order). Per-phone DTW cost varies systematically with
phone type (and, for synthetic references, with which phones the TTS renders
poorly); scoring a learner phone against what *native* speech costs for that
phone removes that offset the way native GOP statistics do for GOP.

Two reference modes, because the statistics must mirror how learners are
scored later:
  --refs native   the utterance's native speaker is scored against up to 4
                  *other* American speakers reading the same prompt (the real-
                  native-template setting, e.g. L2-ARCTIC's bdl/slt);
  --refs <set>    scored against a synthetic voice set from
                  `collect_so762_phone_dtw.voice_fn` (the so762 setting).
Output format matches scripts/collect_so762_phone_dtw.py's (per word: espeak
phones + per-layer per-phone cost lists), one JSONL line per utterance.

Usage:
    OMP_NUM_THREADS=8 python scripts/collect_native_dtw.py --refs native --n 1200 --out results/native_dtw_native.jsonl
    OMP_NUM_THREADS=8 python scripts/collect_native_dtw.py --refs kokoro4 --n 1200 --out results/native_dtw_kokoro4.jsonl
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

from collect_so762_phone_dtw import score_utterance, voice_fn
from eval_umeerj_dtw_ssl import build_ae_index, load_je_to_ae

_WORD_RE = re.compile(r"[A-Za-z']+")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data/UME-ERJ")
    ap.add_argument("--refs", required=True, help="'native' or a voice-set name")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 8)))

    root = Path(args.data_root)
    ae_text = {ae: text for _je, (ae, text) in load_je_to_ae(root).items() if ae.startswith("S_PH_")}
    ae_index = build_ae_index(root)
    pairs = [(spk, ae) for ae, spks in ae_index.items() if ae in ae_text for spk in spks]
    random.Random(args.seed).shuffle(pairs)   # identical order to collect_native_gop.py
    pairs = pairs[: args.n]
    voices = None if args.refs == "native" else voice_fn(args.refs)[: args.n_templates]

    out = Path(args.out)
    done = sum(1 for _ in open(out)) if out.exists() else 0
    print(f"{len(pairs)} native utterances, refs={args.refs}, {done} done", file=sys.stderr)
    t0 = time.time()
    with open(out, "a") as fh:
        for i in range(done, len(pairs)):
            spk, ae = pairs[i]
            words = _WORD_RE.findall(ae_text[ae])
            try:
                s, sr = sf.read(str(root / "wav" / "AE" / spk / ae), dtype="float32")
                if voices is None:
                    others = sorted(o for o in ae_index[ae] if o != spk)[: args.n_templates]
                    refs = [sf.read(str(root / "wav" / "AE" / o / ae), dtype="float32") for o in others]
                else:
                    refs = [v(ae_text[ae]) for v in voices]
                res = score_utterance(s, sr, words, refs)
            except Exception as e:
                print(f"  [warn] {spk}/{ae}: {e}", file=sys.stderr)
                res = None
            if res is None:
                fh.write(json.dumps({"idx": i, "failed": True}) + "\n")
            else:
                fh.write(json.dumps({"idx": i, "speaker": spk, "ae": ae, "words": res[0], "utt_cost": res[1]}) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0; n = i + 1 - done
                print(f"  {i+1}/{len(pairs)} ({el:.0f}s, {el/n:.1f}s/utt, ETA {(len(pairs)-i-1)*el/n/3600:.1f}h)", file=sys.stderr)


if __name__ == "__main__":
    main()
