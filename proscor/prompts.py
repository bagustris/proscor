"""Prompt word/sentence list + selection."""
import random as _random
from pathlib import Path

from proscor.config import DEFAULT_PROMPTS_PATH

_CACHE = {}


def load_prompts(path: Path = DEFAULT_PROMPTS_PATH) -> list:
    path = Path(path)
    key = str(path)
    if key in _CACHE:
        return _CACHE[key]

    prompts = []
    category = "general"
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comment = line.lstrip("#").strip()
            if comment.lower().startswith("category:"):
                category = comment.split(":", 1)[1].strip()
            continue
        prompts.append({"id": len(prompts), "text": line, "category": category})

    _CACHE[key] = prompts
    return prompts


def get_prompt(index: int = None, random: bool = True, path: Path = DEFAULT_PROMPTS_PATH) -> dict:
    prompts = load_prompts(path)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    if index is not None:
        return prompts[index % len(prompts)]
    if random:
        return _random.choice(prompts)
    return prompts[0]
