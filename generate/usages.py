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


def generate_usages(word: str, n: int = 10, model: str = "gpt-4o", base_url: str | None = None) -> list[str]:
    prompt = (
        f'Generate {n} diverse example sentences that use the word "{word}" naturally.\n\n'
        "Requirements:\n"
        "- Vary register (formal, informal, technical, casual)\n"
        "- Vary tense and sentence structure\n"
        "- Each sentence should feel authentic, not forced\n"
        "- Return ONLY the sentences, one per line, no numbering or extra commentary"
    )
    response = _get_client(base_url).chat.completions.create(
        model=model,
        max_completion_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    lines = [s.strip() for s in response.choices[0].message.content.strip().splitlines() if s.strip()]
    return lines[:n]


def generate_usages_persona(
    word: str,
    time_period: str,
    n: int = 10,
    model: str = "gpt-4o",
    base_url: str | None = None,
) -> list[str]:
    system = (
        f"You are a person living in {time_period}. "
        "Think and respond from the perspective of someone from that era — "
        "use vocabulary, concepts, and cultural references authentic to that time period."
    )
    prompt = (
        f'Generate {n} diverse example sentences that use the word "{word}" naturally, '
        f"as someone from {time_period} might speak or write.\n\n"
        "Requirements:\n"
        "- Vary register (formal, informal, technical, casual) within what was natural for the era\n"
        "- Vary tense and sentence structure\n"
        "- Each sentence should feel authentic to the time period, not forced\n"
        "- Return ONLY the sentences, one per line, no numbering or extra commentary"
    )
    response = _get_client(base_url).chat.completions.create(
        model=model,
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    lines = [s.strip() for s in response.choices[0].message.content.strip().splitlines() if s.strip()]
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
