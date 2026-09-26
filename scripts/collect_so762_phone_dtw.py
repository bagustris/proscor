#!/usr/bin/env python
"""Per-phone DTW-SSL costs for speechocean762 (PLAN.md section 5m), for
phone/word-level fusion with the GOP data in results/so762_full.jsonl.

Per utterance: synthesize N templates (sherox Kitten TTS voices, as in
scripts/collect_so762_full.py), CTC-force-align each template against the
prompt (lv-60) to get its espeak-phone frame spans, DTW the learner's WavLM
embeddings against each template, and record per-phone mean path cost
(`dtw_ssl.phone_costs`) averaged over templates -- for several WavLM layers
from the same forward passes (the layer question is answered offline, with
held-out selection, not by re-running this). One JSONL line per utterance:
per word, the template-side espeak phone list and per-layer cost list
(None where a phone has no path cell). Resumable.

`--voices` picks the reference voice set: "kitten4" (default; sherox's
Kitten Nano voices 0/2/4/6, what the original collection used) or any
`proscor.tts_ref.VOICE_SETS` name (PLAN.md section 5m compares them).
`--pilot-fold K --per-speaker N` restricts to the first N utterances of every
speaker in speaker-half K (same seeded split as the analysis scripts) for
cheap voice-set comparisons. Also records `utt_cost`: the whole-path DTW cost
per layer averaged over templates.

Usage:
    OMP_NUM_THREADS=12 python scripts/collect_so762_phone_dtw.py --out results/so762_phone_dtw.jsonl
    python scripts/collect_so762_phone_dtw.py --voices kokoro4 --pilot-fold 0 --per-speaker 6 --out results/pilot_kokoro4.jsonl
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

from proscor import align_phone, dtw_ssl, tts, tts_ref

LAYERS = ("last", 15, 18, 21)
SPEAKER_IDS = (0, 2, 4, 6, 1, 3, 5, 7)


def voice_fn(name):
    """name -> synthesize(text) -> (samples, sr) for each voice in the set."""
    if name == "kitten4":
        return [lambda t, sid=sid: tts.synthesize(t, lang="eng-kitten", speaker_id=sid) for sid in SPEAKER_IDS[:4]]
    return [lambda t, v=v: tts_ref.synthesize(v, t) for v in tts_ref.VOICE_SETS[name]]


def pilot_indices(rows, fold_k, per_speaker):
    """First `per_speaker` utterances of every speaker in speaker-half
    `fold_k`, *interleaved* (each speaker's 1st, then each's 2nd, ...) so a
    run cut short still covers every speaker (so762 rows are ordered by
    speaker)."""
    import random
    speakers = sorted({r["speaker"] for r in rows})
    random.Random(0).shuffle(speakers)
    fold = {s: i % 2 for i, s in enumerate(speakers)}
    by_spk = {}
    for i, r in enumerate(rows):
        if fold[r["speaker"]] == fold_k:
            by_spk.setdefault(r["speaker"], []).append(i)
    order = sorted(by_spk)
    return [by_spk[sp][j] for j in range(per_speaker) for sp in order if j < len(by_spk[sp])]


def load_split(split):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download("mispeech/speechocean762", f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(path).to_pylist()


def score_utterance(samples, sr, words, ref_audios):
    """Per-phone and whole-path DTW cost of one learner utterance against
    `ref_audios` = [(samples, sr), ...] (synthetic or native references of
    the same text). Each reference is CTC-force-aligned (lv-60) to get its
    espeak-phone frame spans. Returns `(words_out, utt_cost)` or None when
    the references' phone lists disagree. `words_out[j]` =
    `{"phones": [...], "cost": {layer: [per-phone mean cost or None]}}`."""
    learner = dtw_ssl.embed_layers(samples, LAYERS, sr=sr)
    per_layer = {L: [] for L in LAYERS}   # [reference] -> [word] -> [cost per phone]
    utt_costs = {L: [] for L in LAYERS}
    phones_ref = None
    for ts, tsr in ref_audios:
        res = align_phone.align_words_gop(ts, words, sr=tsr)
        spans = [[(p["phone"], p["span"]) if p else (None, None) for p in pg] for pg in res["phone_gop"]]
        plist = [[p for p, _ in w] for w in spans]
        if phones_ref is None:
            phones_ref = plist
        elif plist != phones_ref:
            return None
        flat = [sp for w in spans for _p, sp in w if sp is not None]
        temb = dtw_ssl.embed_layers(ts, LAYERS, sr=tsr)
        for L in LAYERS:
            utt_costs[L].append(dtw_ssl.dtw_cost(learner[L], temb[L]))
            costs = dtw_ssl.phone_costs(learner[L], temb[L], flat)
            it = iter(costs)
            per_layer[L].append([[next(it) if sp is not None else None for (_p, sp) in w] for w in spans])
    words_out = []
    for j in range(len(words)):
        entry = {"phones": phones_ref[j], "cost": {}}
        for L in LAYERS:
            cols = []
            for k in range(len(phones_ref[j])):
                vals = [t[j][k] for t in per_layer[L] if t[j][k] is not None]
                cols.append(float(np.mean(vals)) if vals else None)
            entry["cost"][str(L)] = cols
        words_out.append(entry)
    return words_out, {str(L): float(np.mean(v)) for L, v in utt_costs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--voices", default="kitten4")
    ap.add_argument("--pilot-fold", type=int, default=None)
    ap.add_argument("--per-speaker", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 12)))

    rows = load_split(args.split)
    all_rows = rows
    indices = list(range(len(rows)))
    if args.limit:
        indices = indices[: args.limit]
    if args.pilot_fold is not None:
        indices = pilot_indices(all_rows, args.pilot_fold, args.per_speaker)
    voices = voice_fn(args.voices)[: args.n_templates]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(out)) if out.exists() else 0
    print(f"{len(indices)} utterances ({args.voices}), {done} done", file=sys.stderr)
    t0 = time.time()
    with open(out, "a") as fh:
        for pos in range(done, len(indices)):
            i = indices[pos]
            row = all_rows[i]
            try:
                s, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                words = [w["text"] for w in row["words"]]
                res = score_utterance(s, sr, words, [synth(row["text"]) for synth in voices])
                if res is None:
                    fh.write(json.dumps({"idx": i, "failed": True, "why": "template phone lists differ"}) + "\n")
                    fh.flush()
                    continue
                words_out, utt_cost = res
                fh.write(json.dumps({"idx": i, "speaker": row["speaker"], "words": words_out,
                                     "utt_cost": utt_cost}) + "\n")
            except Exception as e:
                print(f"  [warn] utt {i}: {e}", file=sys.stderr)
                fh.write(json.dumps({"idx": i, "failed": True, "why": str(e)[:100]}) + "\n")
            fh.flush()
            if (pos + 1) % 25 == 0:
                el = time.time() - t0; n = pos + 1 - done
                print(f"  {pos+1}/{len(indices)} ({el:.0f}s, {el/n:.1f}s/utt, ETA {(len(indices)-pos-1)*el/n/3600:.1f}h)", file=sys.stderr)


if __name__ == "__main__":
    main()
