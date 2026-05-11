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


def generate_usages(word: str, n: int = 10, model: str = "claude-sonnet-4-6") -> list[str]:
    prompt = (
        f'Generate {n} diverse example sentences that use the word "{word}" naturally.\n\n'
        "Requirements:\n"
        "- Vary register (formal, informal, technical, casual)\n"
        "- Vary tense and sentence structure\n"
        "- Each sentence should feel authentic, not forced\n"
        "- Return ONLY the sentences, one per line, no numbering or extra commentary"
    )
    message = _get_client().messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = [s.strip() for s in message.content[0].text.strip().splitlines() if s.strip()]
    return lines[:n]


def save_usages(word: str, usages: list[str], model: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{word}.json"
    data = {
        "word": word,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "usages": usages,
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return out_path
