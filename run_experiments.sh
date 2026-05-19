#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt-4o}"

echo "=== baseline ==="
uv run genexp --tm "$MODEL" --output-dir outputs/baseline

for decade in $(seq 1900 10 2000); do
    era="${decade}s"
    echo "=== ${era} ==="
    uv run genexp --tm "$MODEL" --time-period "$era" --output-dir "outputs/${era}"
done
