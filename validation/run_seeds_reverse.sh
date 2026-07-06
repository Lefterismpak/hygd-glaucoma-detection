#!/bin/bash
# Seed-robustness runner for the REVERSE symmetric-validation direction.
set -u
cd "$(dirname "$0")/.." || exit 1
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())" 2>/dev/null)
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p results/seed_runs_reverse logs
while pgrep -f "train_generalize_vcdr" >/dev/null 2>&1; do sleep 5; done
for s in "$@"; do
  echo "=== reverse seed $s starting ==="
  .venv/bin/python validation/train_generalize_vcdr_reverse.py --seed "$s" \
      --out "results/seed_runs_reverse/reverse_seed${s}.json" > "logs/reverse_seed${s}.log" 2>&1
  echo "=== reverse seed $s done ==="
done
echo "ALL_REVERSE_DONE"
