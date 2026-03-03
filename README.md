# Phishing Website Detection (Machine Learning)

This project implements an end-to-end machine learning pipeline to detect phishing websites using a Kaggle phishing dataset.

The pipeline includes:

- **Data loading and preprocessing**
- **Feature scaling** (with `StandardScaler`, fit only on the training set)
- **Training multiple classifiers**:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest (optional but included)
- **Model evaluation** with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrices
- **Metric comparison table** saved to `results/metrics.csv`

## Project Structure

```text
phishing-detection/
├── data/
│   └── phishing.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── main.py
├── results/
│   └── metrics.csv  # generated after running the pipeline
├── requirements.txt
└── README.md
```

## Dataset

The project expects a CSV file (e.g., `data/phishing.csv`, `data/dataset.csv`) containing:

- Multiple feature columns (numerical and/or categorical)
- One binary target/label column (commonly named `class` or `Result`)

The included Kaggle-style datasets typically have:

- An ID column like `Index` or `index` (dropped automatically if present)
- Multiple numerical feature columns with values often in \{-1, 0, 1\}
- A binary target column with values in \{-1, 1\}

> **Note**: The pipeline can infer the label column (preferring `class`, `label`, `target`, `result`). If inference is ambiguous, pass `--label-col`.

## Installation

1. Create and activate a virtual environment (recommended).

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

From the project root:

```bash
python -m src.main --data-path data/phishing.csv --output-metrics results/metrics.csv
```

If your label column is different (e.g., `Result`):

```bash
python -m src.main --data-path data/dataset.csv --label-col Result --output-metrics results/metrics_dataset.csv
```

To change which numeric value is treated as the **positive** class for Precision/Recall/F1:

```bash
python -m src.main --data-path data/dataset.csv --positive-label -1
```

By default the script **saves** confusion matrices to `results/confusion_matrices.png` **and opens the plot window after a successful run**. To disable the popup (useful in headless environments):

```bash
python -m src.main --data-path data/phishing.csv --no-show-plots
```

This will:

- Load and preprocess the data
- Train all configured models
- Evaluate them on the test set
- Print a comparison table to the console
- Save the comparison table to `results/metrics.csv`
- Generate confusion matrix plots for each model

## Notebook

The `notebooks/analysis.ipynb` notebook can be used for exploratory data analysis (EDA) and for experimenting with the preprocessing and models interactively.

Launch Jupyter from the project root:

```bash
jupyter notebook
```

Then open `notebooks/analysis.ipynb`.

## Reproducibility

- A fixed `random_state` is used wherever applicable (train-test split, tree-based models, SVM where supported).
- The scaler is **fit only on the training data** and then applied to the test data to avoid data leakage.

## License

This project is intended for educational and academic use (e.g., assignments, reports, and demonstrations).

