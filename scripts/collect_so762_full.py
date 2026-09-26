#!/usr/bin/env python
"""Resumable full-corpus data collector for speechocean762 (PLAN.md section
5m): one JSONL line per utterance holding everything the zero-shot-
improvement probes need, so aggregation / calibration / ensembling /
fusion experiments can be re-analysed offline without re-running any model.

Per utterance it stores: the labels (accuracy, total, fluency, prosodic,
completeness), per-word records (text, word accuracy, dataset ARPABET
phones and per-phone accuracy, and for each of four GOP variants --
xlsr-53 and lv-60, each posterior-deficit and GOP-SF -- the espeak phones
and their per-phone GOP), and the DTW-SSL cost against N synthetic-voice
templates (sherox Kitten TTS; see scripts/eval_so762_dtw_ssl.py for why
so762 needs synthetic references and what that costs in validity).

Resumable: re-running with the same --out appends after however many
lines already exist (utterance order is deterministic).

Usage:
    OMP_NUM_THREADS=12 python scripts/collect_so762_full.py --out results/so762_full.jsonl
    python scripts/collect_so762_full.py --limit 5 --out /tmp/x.jsonl
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proscor import align_phone, dtw_ssl, tts

TTS_LANG = "eng-kitten"
SPEAKER_IDS = (0, 2, 4, 6, 1, 3, 5, 7)


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "mispeech/speechocean762", f"data/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    return pq.read_table(path).to_pylist()


def phone_lists(result: dict, j: int) -> dict:
    pg = result["phone_gop"][j]
    return {"phones": [p["phone"] if p else None for p in pg],
            "gops": [p["gop"] if p else None for p in pg]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(12)

    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(out)) if out.exists() else 0
    print(f"{len(rows)} utterances, {done} already done, {args.n_templates} templates", file=sys.stderr)

    t0 = time.time()
    with open(out, "a") as fh:
        for i in range(done, len(rows)):
            row = rows[i]
            samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            words = [w["text"] for w in row["words"]]
            try:
                learner = dtw_ssl.embed(samples, sr=sr)
                templates = []
                for sid in SPEAKER_IDS[: args.n_templates]:
                    ts, tsr = tts.synthesize(row["text"], lang=TTS_LANG, speaker_id=sid)
                    templates.append(dtw_ssl.embed(ts, sr=tsr))
                dtw = dtw_ssl.score(learner, templates)

                variants = {
                    "xlsr_post": align_phone.align_words_gop(samples, words, sr=sr, model_id=align_phone.TORCH_MODEL_REPO),
                    "xlsr_sf": align_phone.align_words_gop_sf(samples, words, sr=sr, model_id=align_phone.TORCH_MODEL_REPO),
                    "lv60_post": align_phone.align_words_gop(samples, words, sr=sr),
                    "lv60_sf": align_phone.align_words_gop_sf(samples, words, sr=sr),
                }
            except Exception as e:
                print(f"  [warn] utt {i} failed: {e}", file=sys.stderr)
                fh.write(json.dumps({"idx": i, "failed": True}) + "\n")
                fh.flush()
                continue

            rec_words = []
            for j, w in enumerate(row["words"]):
                rec_words.append({
                    "text": w["text"], "accuracy": w["accuracy"],
                    "arpa": w["phones"], "phone_acc": w["phones-accuracy"],
                    **{name: phone_lists(res, j) for name, res in variants.items()},
                })
            rec = {"idx": i, "speaker": row["speaker"], "text": row["text"],
                   "accuracy": row["accuracy"], "total": row["total"],
                   "fluency": row["fluency"], "prosodic": row["prosodic"],
                   "completeness": row["completeness"],
                   "dtw_mean": dtw["mean"], "dtw_min": dtw["min"], "dtw_costs": dtw["costs"],
                   "dur_s": len(samples) / sr, "words": rec_words}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                n = i + 1 - done
                print(f"  {i + 1}/{len(rows)} ({el:.0f}s, {el / n:.1f}s/utt, "
                      f"ETA {(len(rows) - i - 1) * el / n / 3600:.1f}h)", file=sys.stderr)


if __name__ == "__main__":
    main()
