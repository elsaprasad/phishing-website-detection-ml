"""
Main execution script for the phishing website detection project.

This script:
- Loads the phishing dataset from a CSV file
- Preprocesses the data (handling missing values, scaling features)
- Trains several classification models
- Evaluates each model on the test set
- Prints a comparison table of metrics
- Saves the metrics to a CSV file
- Plots confusion matrices for all models

Usage (from project root):
    python -m src.main --data-path data/phishing.csv --output-metrics results/metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .evaluation import ModelResult, evaluate_model, plot_confusion_matrices, results_to_dataframe
from .models import get_models, train_model
from .preprocessing import PreprocessedData, load_dataset, preprocess_data, RANDOM_STATE


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the main script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Phishing website detection using classical machine learning models."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/phishing.csv",
        help="Path to the phishing CSV dataset.",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="results/metrics.csv",
        help="Path to save the evaluation metrics as a CSV file.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Name of the target label column. If omitted, it will be inferred (e.g., 'class' or 'Result').",
    )
    parser.add_argument(
        "--positive-label",
        type=float,
        default=1,
        help="Which label value should be treated as the positive class for precision/recall/F1.",
    )
    parser.add_argument(
        "--confusion-plot-path",
        type=str,
        default="results/confusion_matrices.png",
        help="Path to save the combined confusion matrix figure.",
    )
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="If set, do not display confusion matrix plots (only save to file).",
    )
    return parser.parse_args()


def run_pipeline(
    data_path: Path,
    output_metrics_path: Path,
    label_col: str | None,
    positive_label: float,
    show_plots: bool,
    confusion_plot_path: Path,
) -> pd.DataFrame:
    """
    Run the full ML pipeline: load, preprocess, train, evaluate, and save metrics.

    Parameters
    ----------
    data_path : Path
        Path to the phishing dataset CSV.
    output_metrics_path : Path
        Path where the metrics CSV will be saved.

    Returns
    -------
    pd.DataFrame
        DataFrame containing evaluation metrics for all models.
    """
    # 1. Load data.
    print(f"[INFO] Loading dataset from: {data_path}")
    df = load_dataset(data_path)

    # 2. Preprocess data.
    print("[INFO] Preprocessing data (handling missing values, splitting, scaling)...")
    preprocessed: PreprocessedData = preprocess_data(df, label_col=label_col)

    # 3. Define models.
    print("[INFO] Initializing models...")
    models = get_models(random_state=RANDOM_STATE)

    results: List[ModelResult] = []

    # 4. Train and evaluate each model.
    for name, model in models.items():
        print(f"[INFO] Training model: {name}")
        trained_model = train_model(model, preprocessed.X_train, preprocessed.y_train)

        print(f"[INFO] Evaluating model: {name}")
        result = evaluate_model(
            name=name,
            model=trained_model,
            X_test=preprocessed.X_test,
            y_test=preprocessed.y_test,
            positive_label=positive_label,
        )
        results.append(result)

    # 5. Create comparison table.
    metrics_df = results_to_dataframe(results)

    print("\n==== Model Performance Comparison ====\n")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # Ensure output directory exists.
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Save metrics to CSV.
    metrics_df.to_csv(output_metrics_path, index=False)
    print(f"[INFO] Metrics saved to: {output_metrics_path}")

    # 7. Plot confusion matrices.
    print("[INFO] Plotting confusion matrices...")
    # We do not hard-code semantic class labels here because the mapping
    # between numeric labels (e.g., -1 and 1) and semantic meaning
    # (phishing vs. legitimate) may vary by dataset. The default behavior
    # is to use the label values as ticks.
    plot_confusion_matrices(
        results,
        save_path=confusion_plot_path,
        show=show_plots,
    )

    return metrics_df


def main() -> None:
    """Entry point for the script."""
    args = parse_args()
    data_path = Path(args.data_path)
    output_metrics = Path(args.output_metrics)
    run_pipeline(
        data_path=data_path,
        output_metrics_path=output_metrics,
        label_col=args.label_col,
        positive_label=args.positive_label,
        show_plots=not args.no_show_plots,
        confusion_plot_path=Path(args.confusion_plot_path),
    )


if __name__ == "__main__":
    main()

