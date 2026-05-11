from datetime import datetime, timezone
import json
from pathlib import Path

import anthropic

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def generate_prototypes(word: str, n: int = 4, model: str = "claude-sonnet-4-6") -> list[str]:
    prompt = (
        f'Generate {n} words or short phrases that represent the most prototypical examples of the concept "{word}".\n\n'
        f'These should be the clearest, most canonical instances — the first things that come to mind when someone thinks of a "{word}".\n\n'
        "Requirements:\n"
        "- Return ONLY the prototype words/phrases, one per line\n"
        "- No numbering, no explanations, no extra commentary\n"
        "- Prefer concrete, specific examples over abstract ones"
    )
    message = _get_client().messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = [s.strip() for s in message.content[0].text.strip().splitlines() if s.strip()]
    return lines[:n]


def save_prototypes(word: str, prototypes: list[str], model: str, output_dir: Path) -> Path:
    word_dir = output_dir / word
    word_dir.mkdir(parents=True, exist_ok=True)
    out_path = word_dir / "metadata.json"
    data = {
        "word": word,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prototypes": prototypes,
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return out_path
