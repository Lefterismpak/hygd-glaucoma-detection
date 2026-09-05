#!/bin/bash
# Seed-robustness runner for Attempt B (VCDR multi-task).
# Runs seeds sequentially (single MPS device) into results/patient_aware/, never
# touching the canonical results/generalize_attemptB.json.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())" 2>/dev/null)
export PYTORCH_ENABLE_MPS_FALLBACK=1
: "${HYGD_RIMONE_SUBJECT_MAP:?set the private RIM-ONE stem-to-subject mapping path}"
: "${HYGD_RIMONE_SUBJECT_MAP_SHA256:?set the expected lowercase SHA-256}"
: "${HYGD_ACKNOWLEDGE_RIMONE_MIXED_USE_NEEDS_PROOF:?set to 1 for an authorized local research run}"
if [ "$HYGD_ACKNOWLEDGE_RIMONE_MIXED_USE_NEEDS_PROOF" != "1" ]; then
  echo "HYGD_ACKNOWLEDGE_RIMONE_MIXED_USE_NEEDS_PROOF must equal 1" >&2
  exit 2
fi
if [ "$#" -eq 0 ]; then
  echo "provide one or more distinct seeds from 0..4" >&2
  exit 2
fi
seen_seeds=" "
for s in "$@"; do
  case "$s" in
    0|1|2|3|4) ;;
    *) echo "invalid seed '$s'; allowed values are 0..4" >&2; exit 2 ;;
  esac
  case "$seen_seeds" in
    *" $s "*) echo "duplicate seed '$s'" >&2; exit 2 ;;
  esac
  seen_seeds="$seen_seeds$s "
done

# Wait for any already-running seed0 process to finish (avoid MPS contention).
while pgrep -f "train_generalize_vcdr" >/dev/null 2>&1; do sleep 5; done

for s in "$@"; do
  echo "=== seed $s starting ==="
  .venv/bin/python validation/train_generalize_vcdr.py --seed "$s" \
      --rimone-subject-map "$HYGD_RIMONE_SUBJECT_MAP" \
      --rimone-subject-map-sha256 "$HYGD_RIMONE_SUBJECT_MAP_SHA256" \
      --acknowledge-rimone-mixed-use-needs-proof \
      --out "results/patient_aware/attemptB_seed${s}.json"
  echo "=== seed $s done -> results/patient_aware/attemptB_seed${s}.json ==="
done
echo "ALL_SEEDS_DONE"
