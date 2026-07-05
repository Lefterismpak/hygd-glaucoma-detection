"""Build a unified PAPILA label CSV for external validation.

PAPILA layout: FundusImages/RET{ID}{OD|OS}.jpg, and per-eye clinical data in
ClinicalData/patient_data_{od,os}.xlsx with a Diagnosis column coded
0 = healthy, 1 = glaucoma, 2 = suspect. Per PROTOCOL.md the primary analysis is
binary (glaucoma vs healthy) and EXCLUDES suspects. patient_id groups both eyes.

Output columns: image_path, patient_id, eye, label (1=glaucoma). Suspects dropped.
The CSV is committable; the images are NOT (git-ignored, GPL-3.0+ / not redistributed).
"""

from pathlib import Path
import re
import pandas as pd

PAPILA = Path(__file__).resolve().parents[1] / "validation/data/papila/PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f"
IMAGES = PAPILA / "FundusImages"
OUT = Path(__file__).resolve().parents[1] / "validation/data/papila_labels.csv"


def parse_eye(xlsx, eye):
    df = pd.read_excel(xlsx, engine="openpyxl")
    id_col, dx_col = df.columns[0], "Diagnosis"
    rows = []
    for _, r in df.iterrows():
        raw = str(r[id_col]).strip()
        m = re.match(r"#?(\d+)", raw)
        if not m:
            continue
        pid = m.group(1).zfill(3)
        try:
            dx = int(float(r[dx_col]))
        except (ValueError, TypeError):
            continue
        if dx == 2:  # suspect -> excluded (primary analysis)
            continue
        img = IMAGES / f"RET{pid}{eye}.jpg"
        if not img.exists():
            continue
        rows.append({"image_path": str(img), "patient_id": pid, "eye": eye, "label": int(dx == 1)})
    return rows


def main():
    rows = parse_eye(PAPILA / "ClinicalData/patient_data_od.xlsx", "OD")
    rows += parse_eye(PAPILA / "ClinicalData/patient_data_os.xlsx", "OS")
    df = pd.DataFrame(rows).drop_duplicates("image_path").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} eyes / {df['patient_id'].nunique()} patients | "
          f"glaucoma {int(df.label.sum())} ({df.label.mean():.1%}), healthy {int((df.label==0).sum())}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
