"""Build a unified PAPILA label CSV for external validation.

PAPILA layout: FundusImages/RET{ID}{OD|OS}.jpg, and per-eye clinical data in
ClinicalData/patient_data_{od,os}.xlsx with a Diagnosis column coded
0 = healthy, 1 = glaucoma, 2 = suspect. Per PROTOCOL.md the primary analysis is
binary (glaucoma vs healthy) and EXCLUDES suspects. patient_id groups both eyes.

Output columns: image_path, patient_id, eye, label (1=glaucoma). Suspects dropped.
The CSV contains row-level identifiers/paths and stays private alongside images.
Unknown labels, ambiguous identifiers and missing included images are errors;
they must not silently alter the cohort or be mapped to healthy.
"""

from pathlib import Path
import argparse
import io
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from validation.evaluation_utils import assert_fresh_output_bundle, atomic_write_text, read_verified_file_bytes, resolve_private_output_path

PAPILA = Path(__file__).resolve().parents[1] / "validation/data/papila/PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f"
IMAGES = PAPILA / "FundusImages"
OUT = Path(__file__).resolve().parents[1] / "validation/data/papila_labels.csv"


def parse_eye(xlsx, eye):
    df = pd.read_excel(xlsx, engine="openpyxl")
    return parse_eye_frame(df, eye, IMAGES)


def parse_eye_frame(df, eye, images):
    if eye not in {"OD", "OS"} or "Diagnosis" not in df or df.empty:
        raise ValueError("Expected an OD/OS clinical table with Diagnosis")
    id_col, dx_col = df.columns[0], "Diagnosis"
    header_rows = set()
    if len(df) >= 2:
        first, second = df.iloc[0], df.iloc[1]
        diagnoses = [first[dx_col], second[dx_col]]
        if (pd.isna(first[id_col]) and str(second[id_col]).strip() == "ID"
                and all(pd.isna(value) or str(value).strip() == "Diagnosis" for value in diagnoses)
                and any(str(value).strip() == "Diagnosis" for value in diagnoses)):
            header_rows = {0, 1}
    rows, seen = [], set()
    for position, (_, r) in enumerate(df.iterrows()):
        # The released workbook has a staggered two-row ID/Diagnosis header.
        # Recognize the pair, not arbitrary missing clinical observations.
        if position in header_rows:
            continue
        raw = str(r[id_col]).strip()
        m = re.fullmatch(r"#?([0-9]+)", raw)
        if not m:
            raise ValueError("Malformed PAPILA subject identifier")
        pid = m.group(1).zfill(3)
        if pid in seen:
            raise ValueError("Duplicate PAPILA subject within one eye table")
        seen.add(pid)
        try:
            dx = float(r[dx_col])
        except (ValueError, TypeError):
            raise ValueError("PAPILA diagnosis must be 0, 1 or 2") from None
        if isinstance(r[dx_col], (bool, np.bool_)) or not np.isfinite(dx) or dx not in {0, 1, 2}:
            raise ValueError("PAPILA diagnosis must be exactly 0, 1 or 2")
        if dx == 2:  # suspect -> excluded (primary analysis)
            continue
        img = Path(images) / f"RET{pid}{eye}.jpg"
        if not img.is_file() or img.is_symlink():
            raise ValueError("An included PAPILA image is missing or linked")
        rows.append({"image_path": str(img), "patient_id": pid, "eye": eye, "label": int(dx == 1)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PAPILA)
    parser.add_argument("--out", default="validation/data/papila_labels.csv")
    args = parser.parse_args()
    out = resolve_private_output_path(ROOT, args.out, "validation/data")
    assert_fresh_output_bundle([out])
    rows = []
    for eye in ("OD", "OS"):
        path = args.dataset_root / f"ClinicalData/patient_data_{eye.lower()}.xlsx"
        content, _ = read_verified_file_bytes(path)
        frame = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        rows.extend(parse_eye_frame(frame, eye, args.dataset_root / "FundusImages"))
    df = pd.DataFrame(rows)
    if df.empty or df.duplicated(["patient_id", "eye"]).any():
        raise ValueError("PAPILA output must contain unique, nonempty patient/eye rows")
    atomic_write_text(out, df.to_csv(index=False))
    print(f"{len(df)} eyes / {df['patient_id'].nunique()} patients | "
          f"glaucoma {int(df.label.sum())} ({df.label.mean():.1%}), healthy {int((df.label==0).sum())}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
