"""
Preprocessing utilities for the phishing website detection project.

This module is responsible for:
- Loading the CSV dataset
- Handling missing values
- Splitting into train and test sets
- Encoding categorical features (if present)
- Scaling numerical features with StandardScaler (fit on train, transform on test)

All functions are written to avoid data leakage and to keep the code
modular and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE: int = 42


@dataclass
class PreprocessedData:
    """
    Container for preprocessed phishing dataset.

    Attributes
    ----------
    X_train : np.ndarray
        Scaled training features.
    X_test : np.ndarray
        Scaled test features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    feature_names : List[str]
        Names of the feature columns used in X.
    transformer : ColumnTransformer
        Fitted preprocessing transformer (fit on training features only).
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    transformer: ColumnTransformer


def load_dataset(csv_path: Path | str) -> pd.DataFrame:
    """
    Load the phishing dataset from a CSV file.

    Parameters
    ----------
    csv_path : Path or str
        Path to the CSV file containing the phishing dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    return df


def infer_label_column(df: pd.DataFrame) -> str:
    """
    Infer the label column name for a binary classification dataset.

    Heuristics:
    - Prefer common target names (case-insensitive): class, label, target, result
    - Otherwise, if exactly one non-ID column is binary-valued, use it

    If inference is ambiguous, raise a ValueError with guidance.
    """
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (features + label).")

    lowered = {c.lower(): c for c in df.columns}
    preferred = ["class", "label", "target", "result"]
    preferred_hits = [lowered[p] for p in preferred if p in lowered]
    if len(preferred_hits) == 1:
        return preferred_hits[0]
    if len(preferred_hits) > 1:
        raise ValueError(
            "Multiple possible label columns found: "
            f"{preferred_hits}. Please pass label_col explicitly."
        )

    id_like = {"index", "id", "rowid"}
    candidate_cols = [c for c in df.columns if c.lower() not in id_like]
    binary_cols = [
        c for c in candidate_cols if df[c].nunique(dropna=True) == 2
    ]
    if len(binary_cols) == 1:
        return binary_cols[0]

    raise ValueError(
        "Could not infer a unique binary label column. "
        "Please pass label_col explicitly. "
        f"Binary-like columns found: {binary_cols}"
    )


def preprocess_data(
    df: pd.DataFrame,
    label_col: str | None = None,
    drop_cols: Tuple[str, ...] = ("Index", "index", "ID", "id"),
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> PreprocessedData:
    """
    Preprocess the phishing dataset.

    Steps:
    - Drop non-feature identifier columns (e.g., 'Index')
    - Separate features (X) and labels (y)
    - Handle missing values
    - Split into train and test sets
    - Scale features with StandardScaler (fit on train, transform on test)

    Notes on scaling:
    -----------------
    Many machine learning algorithms (such as SVM and KNN) are sensitive
    to the scale of the input features. Standardizing features to have
    zero mean and unit variance ensures that:
    - No single feature dominates solely because of its scale
    - Distance-based methods (KNN, SVM with RBF kernel) behave correctly
    - Optimization converges more reliably

    Importantly, the scaler is **fit on the training data only** to
    prevent information from the test set leaking into the training
    process (data leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    label_col : str or None, default None
        Name of the target column. If None, the function attempts to infer it
        (e.g., 'class' or 'Result').
    drop_cols : Tuple[str, ...], default ("Index",)
        Columns to drop before modeling (e.g., identifiers).
    test_size : float, default 0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default RANDOM_STATE
        Random state for reproducibility.

    Returns
    -------
    PreprocessedData
        Dataclass containing the train/test splits and scaler.
    """
    df = df.copy()

    # Drop identifier or index-like columns if they exist.
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # determine which column contains the target labels
    if label_col is None:
        label_col = infer_label_column(df)

    # perform a case-insensitive lookup so that users can specify
    # --label-col CLASS or ``class`` and still match a column named
    # ``Class`` or ``CLASS`` in the CSV.
    if label_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == label_col.lower()]
        if len(matches) == 1:
            label_col = matches[0]  # use the actual column name
        else:
            available = ", ".join(df.columns)
            raise ValueError(
                f"Label column '{label_col}' not found in dataset. "
                f"Available columns: {available}"
            )

    # Separate features and labels.
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Drop rows with missing label (cannot be used for supervised learning).
    if y.isnull().any():
        keep = ~y.isnull()
        X = X.loc[keep].copy()
        y = y.loc[keep].copy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Build preprocessing:
    # - Numerical: impute median + StandardScaler
    # - Categorical: impute most-frequent + OneHotEncode
    numeric_cols: Sequence[str] = X_train_df.select_dtypes(include=[np.number]).columns
    categorical_cols: Sequence[str] = [
        c for c in X_train_df.columns if c not in set(numeric_cols)
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # Scaling is required for distance / margin based models (KNN, SVM)
            # so that features contribute proportionally and optimization is stable.
            # Fit on train only, then transform train/test to prevent leakage.
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_cols)),
            ("cat", categorical_pipeline, list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train = transformer.fit_transform(X_train_df)
    X_test = transformer.transform(X_test_df)

    feature_names = list(transformer.get_feature_names_out())

    return PreprocessedData(
        X_train=np.asarray(X_train),
        X_test=np.asarray(X_test),
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        feature_names=feature_names,
        transformer=transformer,
    )

