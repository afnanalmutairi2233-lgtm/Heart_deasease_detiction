"""Data validation and non-clinical model evaluation for the heart dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


BINARY_FEATURES = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations", "Dizziness",
    "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea", "High_BP",
    "High_Cholesterol", "Diabetes", "Smoking", "Obesity", "Sedentary_Lifestyle",
    "Family_History", "Chronic_Stress",
]
FEATURES = BINARY_FEATURES + ["Gender", "Age"]
TARGET = "Heart_Risk"
REQUIRED_COLUMNS = FEATURES + [TARGET]


def load_dataset(path=None):
    """Load the bundled CSV relative to this module and validate its schema."""
    source = Path(path) if path is not None else Path(__file__).resolve().with_name(
        "project_dataset_heart_diesese.csv"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source}")
    for separator in (";", ","):
        try:
            frame = pd.read_csv(source, sep=separator)
        except (OSError, pd.errors.ParserError, UnicodeError):
            continue
        if set(REQUIRED_COLUMNS).issubset(frame.columns):
            validate_dataset(frame)
            return frame[REQUIRED_COLUMNS].copy()
    raise ValueError("CSV does not contain all required columns: " + ", ".join(REQUIRED_COLUMNS))


def validate_dataset(frame):
    """Raise clear errors for missing columns, nulls, nonnumeric, or invalid codes."""
    if frame.empty:
        raise ValueError("The dataset contains no rows.")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))
    missing_cells = int(frame[REQUIRED_COLUMNS].isna().sum().sum())
    if missing_cells:
        raise ValueError(f"Dataset contains {missing_cells} missing required cells; no rows were dropped.")
    numeric = frame[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_cells = int(numeric.isna().sum().sum())
    if invalid_cells:
        raise ValueError(f"Dataset contains {invalid_cells} non-numeric values in required fields.")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Required numeric fields must contain finite values.")
    for column in BINARY_FEATURES + ["Gender", TARGET]:
        invalid = ~numeric[column].isin([0, 1])
        if bool(invalid.any()):
            raise ValueError(f"Column {column} must contain only 0 and 1.")
    if bool((numeric["Age"] < 0).any()):
        raise ValueError("Age values cannot be negative.")
    return True


def data_quality_report(frame):
    """Report source row counts without changing or dropping any rows."""
    validate_dataset(frame)
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": frame[TARGET].value_counts().sort_index().to_dict(),
    }


def train_models(frame, random_state=42, max_rows=20_000):
    """Evaluate two classifiers on a stratified holdout after exact-row deduplication.

    Duplicate observations are reported separately by data_quality_report. The
    model evaluation uses unique complete rows to avoid copies leaking across splits.
    All learned preprocessing is fitted on the training fold by sklearn Pipelines.
    """
    validate_dataset(frame)
    model_frame = frame[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="raise")
    model_frame = model_frame.drop_duplicates().reset_index(drop=True)
    X = model_frame[FEATURES]
    y = model_frame[TARGET]
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Model evaluation requires at least two observations in each target class.")
    if max_rows is not None and len(y) > max_rows:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows, random_state=random_state, stratify=y
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    estimators = {
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=1),
        "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(random_state=random_state)),
    }
    results = {}
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )
        results[name] = {
            "model": estimator,
            "predictions": predictions,
            "class_metrics": {
                label: {
                    "Precision": report[str(label)]["precision"],
                    "Recall": report[str(label)]["recall"],
                    "F1-score": report[str(label)]["f1-score"],
                    "Support": int(report[str(label)]["support"]),
                }
                for label in (0, 1)
            },
            "metrics": {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1-score": f1_score(y_test, predictions, zero_division=0),
            },
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
            "feature_importances": (
                pd.Series(estimator.feature_importances_, index=FEATURES).sort_values(ascending=False)
                if name == "Random Forest" else None
            ),
        }
    return {
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
        "model_rows": int(len(model_frame)),
        "sample_rows": int(len(X)),
        "rows_removed_as_exact_duplicates_for_evaluation": int(len(frame) - len(model_frame)),
    }
