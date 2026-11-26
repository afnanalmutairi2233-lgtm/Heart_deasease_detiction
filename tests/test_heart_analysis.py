import unittest
import os
from pathlib import Path

import pandas as pd

from heart_analysis import BINARY_FEATURES, REQUIRED_COLUMNS, data_quality_report, load_dataset, train_models, validate_dataset


def sample_frame(rows=80):
    data = {column: [i % 2 for i in range(rows)] for column in BINARY_FEATURES}
    data["Gender"] = [i % 2 for i in range(rows)]
    data["Age"] = [18 + i % 73 for i in range(rows)]
    data["Heart_Risk"] = [i % 2 for i in range(rows)]
    return pd.DataFrame(data, columns=REQUIRED_COLUMNS)


class HeartAnalysisTests(unittest.TestCase):
    def test_valid_complete_dataset(self):
        self.assertTrue(validate_dataset(sample_frame()))

    def test_empty_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contains no rows"):
            validate_dataset(sample_frame().iloc[:0])

    def test_default_dataset_path_is_relative_to_module_not_current_directory(self):
        original_directory = os.getcwd()
        try:
            os.chdir(Path(__file__).resolve().parent)
            loaded = load_dataset()
        finally:
            os.chdir(original_directory)
        self.assertEqual(loaded.shape, (70_000, 19))

    def test_missing_column_is_rejected(self):
        frame = sample_frame().drop(columns=["Age"])
        with self.assertRaisesRegex(ValueError, "Missing required columns: Age"):
            validate_dataset(frame)

    def test_missing_cell_is_rejected_without_row_drop(self):
        frame = sample_frame()
        frame.loc[3, "Age"] = None
        with self.assertRaisesRegex(ValueError, "1 missing required cells"):
            validate_dataset(frame)
        self.assertEqual(len(frame), 80)

    def test_invalid_binary_code_is_rejected(self):
        frame = sample_frame()
        frame.loc[0, "Smoking"] = 2
        with self.assertRaisesRegex(ValueError, "Smoking must contain only 0 and 1"):
            validate_dataset(frame)

    def test_nonfinite_values_and_negative_age_are_rejected(self):
        frame = sample_frame()
        frame["Age"] = frame["Age"].astype(float)
        frame.loc[0, "Age"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite values"):
            validate_dataset(frame)
        frame = sample_frame()
        frame.loc[0, "Age"] = -1
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            validate_dataset(frame)

    def test_duplicate_rows_are_reported_and_source_frame_is_retained(self):
        frame = sample_frame()
        frame = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
        report = data_quality_report(frame)
        self.assertEqual(report["duplicate_rows"], 2)
        self.assertEqual(report["rows"], 82)
        self.assertEqual(len(frame), 82)

    def test_model_evaluation_excludes_exact_duplicates_before_split(self):
        frame = sample_frame(100)
        frame = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
        result = train_models(frame)
        self.assertEqual(result["rows_removed_as_exact_duplicates_for_evaluation"], 2)
        self.assertEqual(len(result["y_test"]), 20)
        self.assertEqual(set(result["results"]), {"Random Forest", "Linear SVM"})
        for model_result in result["results"].values():
            self.assertEqual(set(model_result["class_metrics"]), {0, 1})
            for metrics in model_result["class_metrics"].values():
                self.assertEqual(set(metrics), {"Precision", "Recall", "F1-score", "Support"})


if __name__ == "__main__":
    unittest.main()
