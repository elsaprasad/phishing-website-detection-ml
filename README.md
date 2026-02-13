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

The project expects a CSV file at `data/phishing.csv`. The sample dataset used here has:

- An `Index` column (row identifier)
- Multiple numerical feature columns with values in \{-1, 0, 1\}
- A binary target column named `class` with values \{-1, 1\}, where one of the values represents phishing and the other legitimate websites.

> **Note**: No feature or label names are hard-coded beyond using `class` as the target label and dropping the `Index` column. All remaining columns are treated as numerical features.

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

