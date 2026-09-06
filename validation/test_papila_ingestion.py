"""Strict clinical label parsing, including the two documented header rows."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class PapilaIngestionTests(unittest.TestCase):
    def run_parser(self, frame, files=("RET002OD.jpg", "RET004OD.jpg")):
        from validation import make_papila_labels as parser
        with tempfile.TemporaryDirectory() as directory:
            image_root = Path(directory)
            for name in files:
                (image_root / name).write_bytes(b"not decoded by label parser")
            with patch.object(parser, "IMAGES", image_root), patch.object(parser.pd, "read_excel", return_value=frame):
                return parser.parse_eye(image_root / "clinical.xlsx", "OD")

    def test_documented_headers_and_suspects_are_explicitly_excluded(self):
        frame = pd.DataFrame({"Unnamed: 0": [None, "ID", "#002", "#004", "#005"],
                              "Diagnosis": ["Diagnosis", None, 0, 1, 2]})
        rows = self.run_parser(frame)
        self.assertEqual([(r["patient_id"], r["eye"], r["label"]) for r in rows], [("002", "OD", 0), ("004", "OD", 1)])

    def test_unknown_fractional_missing_and_nonfinite_diagnoses_fail(self):
        for bad in [3, -1, 1.8, float("nan"), float("inf"), "not-a-diagnosis"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.run_parser(pd.DataFrame({"ID": ["#002"], "Diagnosis": [bad]}))

    def test_malformed_identifiers_cannot_be_truncated_to_an_existing_patient(self):
        for bad in ["#002junk", "2.5", "002/other", None, ""]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.run_parser(pd.DataFrame({"ID": [bad], "Diagnosis": [1]}))

    def test_missing_included_image_and_duplicate_patient_rows_fail(self):
        for frame in [pd.DataFrame({"ID": ["#003"], "Diagnosis": [1]}),
                      pd.DataFrame({"ID": ["#002", "#002"], "Diagnosis": [0, 1]})]:
            with self.assertRaises(ValueError):
                self.run_parser(frame)

    def test_missing_clinical_row_cannot_masquerade_as_a_header(self):
        frame = pd.DataFrame({"ID": [None, "#002"], "Diagnosis": [None, 1]})
        with self.assertRaises(ValueError):
            self.run_parser(frame)


if __name__ == "__main__":
    unittest.main()
