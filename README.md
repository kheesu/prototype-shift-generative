# genexp

Generates prototype exemplars and example usages for target words using an LLM.

- **Prototypes**: canonical, most-representative instances of a concept (e.g. for "bird" → robin, sparrow, eagle)
- **Usages**: diverse example sentences using the word naturally, varying register and structure

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
uv run genexp -t <word> [<word> ...]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-t`, `--targets` | required | One or more target words |
| `--n` | 10 (usages), 4 (prototypes) | Number of outputs per word |
| `--tm` | `gpt-4o` | OpenAI model override |
| `--usages-only` | — | Skip prototype generation |
| `--prototypes-only` | — | Skip usage generation |

**Examples:**

```bash
# Generate both for a single word
uv run genexp -t bird

# Generate usages only for multiple words
uv run genexp -t chair table lamp --usages-only

# Use a different model, more outputs
uv run genexp -t vehicle --tm gpt-4o-mini --n 8
```

## Output

Results are written to `outputs/`:

```
outputs/
  usages/
    bird.json
  prototypes/
    bird/
      metadata.json
```

Each file contains the word, model used, generation timestamp, and results.
