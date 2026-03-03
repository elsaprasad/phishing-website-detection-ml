"""
Evaluation utilities for the phishing website detection project.

This module provides functions to:
- Compute standard classification metrics
- Aggregate results into a comparison table
- Plot confusion matrices for multiple models
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ModelResult:
    """
    Container for evaluation results of a single model.

    Attributes
    ----------
    name : str
        Name of the model.
    accuracy : float
        Accuracy score on the test set.
    precision : float
        Precision score on the test set.
    recall : float
        Recall score on the test set.
    f1 : float
        F1-score on the test set.
    confusion : np.ndarray
        Confusion matrix (2x2 for binary classification).
    """

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray


def evaluate_model(
    name: str,
    model,
    X_test,
    y_test,
    positive_label: int | float = 1,
) -> ModelResult:
    """
    Evaluate a fitted model on the test set.

    Parameters
    ----------
    name : str
        Name of the model.
    model : object
        Fitted scikit-learn estimator with `predict` method.
    X_test : array-like
        Test features.
    y_test : array-like
        True labels for the test set.
    positive_label : int or float, default 1
        The label value considered the positive class.

    Returns
    -------
    ModelResult
        Dataclass with scalar metrics and confusion matrix.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=positive_label)
    rec = recall_score(y_test, y_pred, pos_label=positive_label)
    f1 = f1_score(y_test, y_pred, pos_label=positive_label)
    cm = confusion_matrix(y_test, y_pred)

    return ModelResult(
        name=name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion=cm,
    )


def results_to_dataframe(results: List[ModelResult]) -> pd.DataFrame:
    """
    Convert a list of ModelResult objects into a Pandas DataFrame.

    Parameters
    ----------
    results : List[ModelResult]
        Evaluation results for multiple models.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per model and metric columns.
    """
    records = [
        {
            "Model": r.name,
            "Accuracy": r.accuracy,
            "Precision": r.precision,
            "Recall": r.recall,
            "F1-score": r.f1,
        }
        for r in results
    ]
    df_results = pd.DataFrame.from_records(records)
    df_results = df_results.sort_values(by="F1-score", ascending=False).reset_index(
        drop=True
    )
    return df_results


def plot_confusion_matrices(
    results: List[ModelResult],
    class_labels: List[str] | None = None,
    figsize: tuple[int, int] = (14, 8),
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot confusion matrices for multiple model results.

    Parameters
    ----------
    results : List[ModelResult]
        List of model evaluation results.
    class_labels : List[str] or None, default None
        Optional custom labels for the classes (e.g., ["Legitimate", "Phishing"]).
    figsize : tuple[int, int], default (14, 8)
        Size of the matplotlib figure.
    save_path : str | Path | None, default None
        If provided, save the generated figure to this path.
    show : bool, default True
        If True, display the figure interactively. If False, the figure is closed
        after saving (useful for non-interactive/scripted runs).
    """
    n_models = len(results)
    if n_models == 0:
        return

    cols = min(3, n_models)
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax, res in zip(axes, results):
        # Build keyword arguments for tick labels only when labels are provided.
        # Passing xticklabels=None / yticklabels=None explicitly can cause
        # issues in some seaborn versions, so we omit them to use defaults.
        heatmap_kwargs = {}
        if class_labels is not None:
            heatmap_kwargs["xticklabels"] = class_labels
            heatmap_kwargs["yticklabels"] = class_labels

        sns.heatmap(
            res.confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            **heatmap_kwargs,
        )
        ax.set_title(res.name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    # Hide any unused subplots.
    for ax in axes[len(results) :]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

