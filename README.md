# Heart Dataset: Descriptive Analysis

## Project Overview

This project explores a bundled tabular dataset containing 70,000 records, 18 input fields, and a binary `Heart_Risk` label. It includes exploratory summaries, a Streamlit dashboard, and an illustrative comparison of two classifiers that reproduce the supplied labels.

This is a descriptive data analysis project, not a validated medical prediction or diagnostic system.

## Problem Statement

The project examines how the recorded fields vary with the `Heart_Risk` label and demonstrates a basic, stratified model evaluation. The repository does not document the label definition or how it was assigned, so model scores describe agreement with labels in this file only.

## Dataset Description

- File: `project_dataset_heart_diesese.csv` (semicolon-delimited)
- Shape: 70,000 rows and 19 columns
- Fields: 16 binary symptom or risk-factor codes, `Gender`, `Age`, and `Heart_Risk`
- Missing required cells: 0 in the bundled file
- Exact duplicate rows: 6,245 in the bundled file
- Supplied target counts: 35,000 records for each value (0 and 1)

The repository does not provide reliable dataset provenance, a sampling description, a definition for the target-generation process, or an independent reference outcome. Field descriptions in the dashboard are limited to the column names and observed numeric coding; code meanings and clinical units are not independently documented.

## Technologies & Tools

- Python
- pandas for tabular data work, with NumPy declared as a package dependency
- scikit-learn for validation, holdout splitting, preprocessing, and classification
- Streamlit for the dashboard
- Plotly for interactive charts
- Matplotlib for notebook charts
- `unittest` for automated checks

The package list is in `requirements.txt`.

## Methodology

The data loader resolves the CSV path relative to the application and validates the required columns, missing values, numeric values, and binary codes. Invalid input raises a clear error; rows with missing values are not silently discarded. Exact duplicates are counted. They remain in the source file and descriptive views; exact copies are excluded in memory from model evaluation to reduce train/test overlap.

The model evaluation draws a reproducible stratified sample of up to 20,000 unique records, then uses an 80/20 stratified random split with a fixed seed. Standardization for the Linear SVM is fitted inside a scikit-learn pipeline using training data only. The Random Forest is trained without scaling. Accuracy, precision, recall, F1-score, and a confusion matrix are reported.

## Exploratory Data Analysis

The notebook and dashboard summarize the supplied target distribution, age-group label rates, numeric correlations, and prevalence of recorded binary fields. These are associations within the supplied file and do not establish causation.

## Model Development

The notebook evaluates a Random Forest and a Linear SVM as classifiers of the supplied `Heart_Risk` labels. The previous notebook fit a separate scaler to the test data and separately fit PCA on the train and test data. Its saved model scores and PCA outputs are superseded by the corrected pipeline in `heart_analysis.py` and `Project_Afnan_Alshalahi.ipynb`.

Corrected scores are generated when the notebook is run. They should be interpreted only as held-out agreement with the provided labels, not as evidence of clinical accuracy.

## Evaluation Results

Corrected results on the bundled dataset use 20,000 unique records selected by stratified sampling (seed 42), followed by a stratified 80/20 split (16,000 train / 4,000 test):

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.9903 | 0.9914 | 0.9889 | 0.9901 |
| Linear SVM | 0.9920 | 0.9914 | 0.9924 | 0.9919 |

The Precision, Recall, and F1 columns above refer to label 1. Per-class metrics on the same held-out sample are:

| Model | Label | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0 | 0.9892 | 0.9916 | 0.9904 | 2,025 |
| Random Forest | 1 | 0.9914 | 0.9889 | 0.9901 | 1,975 |
| Linear SVM | 0 | 0.9926 | 0.9916 | 0.9921 | 2,025 |
| Linear SVM | 1 | 0.9914 | 0.9924 | 0.9919 | 1,975 |

These scores measure agreement with the supplied labels only. Their unusually high values are not evidence of clinical accuracy; the label-generation process is undocumented, and age and several recorded fields are strongly associated with the label. No independent validation set or clinical ground truth is provided.

## Dashboard Features

- Filters by recorded gender code, age, and supplied label
- Dataset overview and downloadable filtered records
- Descriptive label, age, and feature summaries
- Data-quality counts
- On-demand classifier evaluation and confusion matrix
- Persistent notice describing the target and clinical limitations

## Installation and Usage

Run these commands from the project directory:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app1.py
```

To open the analysis notebook, start Jupyter from the project directory and open `Project_Afnan_Alshalahi.ipynb`. The notebook imports `heart_analysis.py` and reads the bundled CSV by a path relative to that module.

Run automated checks with:

```bash
python -m unittest discover -s tests -v
```

## Limitations

- The dataset source, sampling method, and label-generation process are not documented in the repository.
- The target is strongly associated with age and several input fields; this may reflect how the labels were constructed, but the available files do not establish the cause.
- A random holdout from the same file is not independent external validation.
- The classifiers are not clinically validated and must not guide diagnosis, treatment, or care decisions.
- Exact duplicates are present and are excluded only from the model-evaluation frame; descriptive summaries retain the supplied records.

## Future Improvements

- Document the data source, collection process, field definitions, and target labeling before making predictive claims.
- Validate any future model against an independently collected and appropriately reviewed dataset.
- Add temporal or group-based evaluation if the data collection structure becomes available.
- Compare calibrated models and class-specific errors only after the target and intended use are defined.
