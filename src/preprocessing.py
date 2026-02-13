"""
Preprocessing utilities for the phishing website detection project.

This module is responsible for:
- Loading the CSV dataset
- Handling missing values
- Splitting into train and test sets
- Scaling numerical features with StandardScaler (fit on train, transform on test)

All functions are written to avoid data leakage and to keep the code
modular and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    scaler : StandardScaler
        Fitted scaler instance (fit on training features only).
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler


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


def preprocess_data(
    df: pd.DataFrame,
    label_col: str = "class",
    drop_cols: Tuple[str, ...] = ("Index",),
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
    label_col : str, default "class"
        Name of the target column.
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

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    # Separate features and labels.
    X = df.drop(columns=[label_col])
    y = df[label_col]

    feature_names = X.columns.tolist()

    # Handle missing values: simple strategy -> fill with column median.
    # For numerical-only datasets, median is robust and works well.
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median(numeric_only=True))

    # Convert to numpy arrays.
    X_values = X.values
    y_values = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_values,
        y_values,
        test_size=test_size,
        random_state=random_state,
        stratify=y_values,
    )

    # Feature scaling: fit StandardScaler on training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return PreprocessedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )

