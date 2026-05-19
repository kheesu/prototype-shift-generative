from datetime import datetime, timezone
import json
from pathlib import Path

from openai import OpenAI

_client: OpenAI | None = None


def _get_client(base_url: str | None = None) -> OpenAI:
    global _client
    if _client is None:
        if base_url:
            _client = OpenAI(base_url=base_url, api_key="vllm")
        else:
            _client = OpenAI()
    return _client


def generate_prototypes(word: str, n: int = 4, model: str = "gpt-4o", base_url: str | None = None) -> list[str]:
    prompt = (
        f'Generate {n} words or short phrases that represent the most prototypical examples of the concept "{word}".\n\n'
        f'These should be the clearest, most canonical instances — the first things that come to mind when someone thinks of a "{word}".\n\n'
        "Requirements:\n"
        "- Return ONLY the prototype words/phrases, one per line\n"
        "- No numbering, no explanations, no extra commentary\n"
        "- Prefer concrete, specific examples over abstract ones"
    )
    response = _get_client(base_url).chat.completions.create(
        model=model,
        max_completion_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = [s.strip() for s in response.choices[0].message.content.strip().splitlines() if s.strip()]
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
