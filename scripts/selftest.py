#!/usr/bin/env python
"""Self-test: synthesize clean + degraded reference clips, assert score ranges.

Also warms up (auto-downloads) the Silero VAD + Piper TTS models on first run.
No mic recording needed: the "bad" clip is the clean TTS reference degraded
with noise and dropped chunks (simple DSP), not a real recording.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audiokit import resample

from proscor.asr import transcribe
from proscor.score import score as score_report
from proscor.tts import synthesize

TEST_PROMPTS = ["thought through", "she sells sea shells"]


def degrade(samples: np.ndarray) -> np.ndarray:
    """Add heavy noise + drop random chunks to simulate bad pronunciation."""
    rng = np.random.default_rng(0)
    peak = np.abs(samples).max() or 1.0
    noisy = samples + rng.normal(0, 0.2 * peak, size=samples.shape).astype(np.float32)
    n = len(noisy)
    chunk = max(1, n // 10)
    for _ in range(4):
        start = int(rng.integers(0, max(1, n - chunk)))
        noisy[start:start + chunk] = 0.0
    return noisy.astype(np.float32)


def run_case(text: str) -> tuple:
    samples, sr = synthesize(text)
    samples16 = resample(samples, sr, 16000)

    clean_report = score_report(text, transcribe(samples16, 16000))
    bad_report = score_report(text, transcribe(degrade(samples16), 16000))
    return clean_report["score"], bad_report["score"]


def main():
    failures = []
    for text in TEST_PROMPTS:
        clean_score, bad_score = run_case(text)
        print(f'"{text}": clean={clean_score} bad={bad_score}')
        if clean_score < 90:
            failures.append(f'"{text}" clean score {clean_score} < 90')
        if bad_score > 40:
            failures.append(f'"{text}" bad score {bad_score} > 40')

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nAll self-test score ranges OK.")


if __name__ == "__main__":
    main()
