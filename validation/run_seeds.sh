#!/bin/bash
# Seed-robustness runner for Attempt B (VCDR multi-task).
# Runs seeds sequentially (single MPS device) into results/seed_runs/, never
# touching the canonical results/generalize_attemptB.json.
set -u
cd "$(dirname "$0")/.." || exit 1
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())" 2>/dev/null)
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p results/seed_runs logs

# Wait for any already-running seed0 process to finish (avoid MPS contention).
while pgrep -f "train_generalize_vcdr" >/dev/null 2>&1; do sleep 5; done

for s in "$@"; do
  echo "=== seed $s starting ==="
  .venv/bin/python validation/train_generalize_vcdr.py --seed "$s" \
      --out "results/seed_runs/attemptB_seed${s}.json" > "logs/seed${s}.log" 2>&1
  echo "=== seed $s done -> results/seed_runs/attemptB_seed${s}.json ==="
done
echo "ALL_SEEDS_DONE"
