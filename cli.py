#!/usr/bin/env python
"""Interactive CLI: show a prompt, record, score, give feedback."""
import argparse
import sys

from proscor import audio, prompts, score as scorer, feedback as fb
from proscor.config import DEFAULT_PROMPTS_PATH, TTS_LANG


def parse_args():
    p = argparse.ArgumentParser(description="proscor-en: English pronunciation scoring CLI")
    p.add_argument("--seconds", type=float, default=3.0, help="recording duration")
    p.add_argument("--prompt-file", default=str(DEFAULT_PROMPTS_PATH))
    p.add_argument("--include-stress", action="store_true")
    p.add_argument("--model-dir", default=None, help="override ASR model dir")
    p.add_argument("--tts-lang", default=TTS_LANG, help="reference voice language")
    p.add_argument("--no-tts", action="store_true", help="disable reference playback")
    return p.parse_args()


def _score_and_report(target_text: str, seconds: float, include_stress: bool, model_dir: str):
    from proscor.asr import transcribe

    print(f"[recording {seconds:.0f}s...]")
    samples = audio.record(seconds)
    result = transcribe(samples, model_dir=model_dir)
    report = scorer.score(target_text, result, include_stress=include_stress)
    print(fb.format_report(report))


def main():
    args = parse_args()
    try:
        prompt_list = prompts.load_prompts(args.prompt_file)
    except FileNotFoundError:
        print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
        sys.exit(1)

    index = 0
    while True:
        try:
            prompt = prompt_list[index % len(prompt_list)]
            print(f'\nproscor> Prompt #{prompt["id"]}: "{prompt["text"]}"')
            if not args.no_tts:
                print("proscor> (p)lay reference, then press ENTER to record...")
            else:
                print("proscor> Press ENTER to record...")

            while True:
                key = input("proscor> ").strip().lower()
                if key == "p" and not args.no_tts:
                    from proscor.tts import play_reference

                    play_reference(prompt["text"], lang=args.tts_lang)
                    continue
                break

            _score_and_report(prompt["text"], args.seconds, args.include_stress, args.model_dir)

            action = input("proscor> (n)ext  (r)etry  (q)uit  ").strip().lower()
            if action == "q":
                break
            elif action == "r":
                continue
            else:
                index += 1
        except (KeyboardInterrupt, EOFError):
            # Ctrl-C (KeyboardInterrupt) and Ctrl-D (EOFError) exit the loop
            # cleanly, just like pressing (q)uit — no traceback, exit code 0.
            # The `print()` moves past the ^C / ^D the terminal echoed.
            print()
            break


if __name__ == "__main__":
    main()
