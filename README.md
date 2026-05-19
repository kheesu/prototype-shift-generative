# genexp

Generates prototype exemplars and example usages for target words using an LLM.

- **Prototypes**: canonical, most-representative instances of a concept (e.g. for "bird" → robin, sparrow, eagle)
- **Usages**: diverse example sentences using the word naturally, varying register and structure

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Set your OpenAI API key either by exporting it or putting it in a `.env` file:

```bash
export OPENAI_API_KEY=sk-...
# or
echo "OPENAI_API_KEY=sk-..." > .env
```

## Usage

```bash
uv run genexp [-t <word> ...]
```

If `-t` is omitted, all words from `datasets/categories.csv` are used.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-t`, `--targets` | all words in `datasets/categories.csv` | One or more target words |
| `--n` | 10 (usages), 4 (prototypes) | Number of outputs per word |
| `--tm` | `gpt-4o` | Text model override |
| `--time-period` | — | Persona era for generation (e.g. `"1920s America"`) |
| `--vllm-url` | — | Base URL of a vLLM OpenAI-compatible server |
| `--output-dir` | `outputs/` | Root directory for results |
| `--usages-only` | — | Skip prototype generation |
| `--prototypes-only` | — | Skip usage generation |

**Examples:**

```bash
# Run over all words in datasets/categories.csv
uv run genexp

# Generate both for a single word
uv run genexp -t bird

# Generate usages only for multiple words
uv run genexp -t chair table lamp --usages-only

# Use a different model, more outputs
uv run genexp -t vehicle --tm gpt-4o-mini --n 8

# Generate with a historical persona
uv run genexp -t phone --time-period "1920s America"

# Use a local vLLM server
uv run genexp -t bird --tm meta-llama/Llama-3-8b-instruct --vllm-url http://localhost:8000/v1
```

## Running experiments

`run_experiments.sh` runs the full word set across a baseline (no persona) and every decade from the 1900s through the 2000s, writing each run to its own subdirectory:

```bash
./run_experiments.sh
```

The model can be overridden via the `MODEL` env var:

```bash
MODEL=gpt-4o-mini ./run_experiments.sh
```

Output is organised by era under `outputs/`:

```
outputs/
  baseline/
    usages/   prototypes/
  1900s/
    usages/   prototypes/
  1910s/
    usages/   prototypes/
  ...
  2000s/
    usages/   prototypes/
```

## Output

Each `usages/<word>.json` and `prototypes/<word>/metadata.json` file contains the word, model used, generation timestamp, and results.
