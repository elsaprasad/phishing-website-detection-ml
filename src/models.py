"""
Model definitions and training utilities for the phishing website detection project.

This module defines several baseline classifiers and helper functions
to train them on the preprocessed data.
"""

from __future__ import annotations

from typing import Dict

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_models(random_state: int = 42) -> Dict[str, object]:
    """
    Create and return a dictionary of classification models to evaluate.

    Parameters
    ----------
    random_state : int, default 42
        Random state for reproducibility where applicable.

    Returns
    -------
    Dict[str, object]
        Mapping from model name to an instantiated (unfitted) estimator.
    """
    models = {
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=random_state,
        ),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    return models


def train_model(model, X_train, y_train):
    """
    Fit a given model on the training data.

    Parameters
    ----------
    model : object
        Any scikit-learn compatible estimator implementing `fit`.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.

    Returns
    -------
    object
        The fitted model (same instance).
    """
    model.fit(X_train, y_train)
    return model

