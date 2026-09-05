#!/bin/bash
# Seed-robustness runner for the REVERSE symmetric-validation direction.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())" 2>/dev/null)
export PYTORCH_ENABLE_MPS_FALLBACK=1
: "${HYGD_RIMONE_SUBJECT_MAP:?set the private RIM-ONE stem-to-subject mapping path}"
: "${HYGD_RIMONE_SUBJECT_MAP_SHA256:?set the expected lowercase SHA-256}"
: "${HYGD_ACKNOWLEDGE_EXTERNAL_DATA_LICENSE_NEEDS_PROOF:?set to 1 for an authorized local research run}"
if [ "$HYGD_ACKNOWLEDGE_EXTERNAL_DATA_LICENSE_NEEDS_PROOF" != "1" ]; then
  echo "HYGD_ACKNOWLEDGE_EXTERNAL_DATA_LICENSE_NEEDS_PROOF must equal 1" >&2
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
while pgrep -f "train_generalize_vcdr" >/dev/null 2>&1; do sleep 5; done
for s in "$@"; do
  echo "=== reverse seed $s starting ==="
  .venv/bin/python validation/train_generalize_vcdr_reverse.py --seed "$s" \
      --rimone-subject-map "$HYGD_RIMONE_SUBJECT_MAP" \
      --rimone-subject-map-sha256 "$HYGD_RIMONE_SUBJECT_MAP_SHA256" \
      --acknowledge-external-data-license-needs-proof \
      --out "results/patient_aware/reverse_seed${s}.json"
  echo "=== reverse seed $s done ==="
done
echo "ALL_REVERSE_DONE"
