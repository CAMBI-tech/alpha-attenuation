import csv
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
from alpha_experiment import load_data
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from loguru import logger
from rich.console import Console
from rich.table import Table
from sklearn.metrics import balanced_accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


def reorder(data):
    return data.transpose(1, 0, 2)


def main(input_path, output_path):
    # extract relevant session information from parameters file
    data, labels, _ = load_data(input_path, trial_length=0.5, pre_stim_offset=0)

    model = make_pipeline(FunctionTransformer(reorder), PcaRdaKdeModel(k_folds=10))

    n_folds = 10
    np.random.seed(1)
    results = cross_validate(
        model,
        data,
        labels,
        cv=n_folds,
        n_jobs=-1,
        return_train_score=True,
        scoring={
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
        },
    )

    report = {
        "Avg fit time": results["fit_time"].mean(),
        "Std fit time": results["fit_time"].std(),
        "Avg score time": results["score_time"].mean(),
        "Std score time": results["score_time"].std(),
        "Avg train roc auc": results["train_roc_auc"].mean(),
        "Std train roc auc": results["train_roc_auc"].std(),
        "Avg test roc auc": results["test_roc_auc"].mean(),
        "Std test roc auc": results["test_roc_auc"].std(),
        "Avg train balanced accuracy": results["train_balanced_accuracy"].mean(),
        "Std train balanced accuracy": results["train_balanced_accuracy"].std(),
        "Avg test balanced accuracy": results["test_balanced_accuracy"].mean(),
        "Std test balanced accuracy": results["test_balanced_accuracy"].std(),
    }
    report = {k: str(round(v, 3)) for k, v in report.items()}
    report["Model Name"] = "PCA/RDA/KDE"

    table = Table(title=f"Alpha Classifier Comparison ({n_folds}-fold cross validation)")
    colors = cycle(["green", "blue"])
    for col_name, color in zip(report.keys(), colors):
        table.add_column(col_name, style=color, no_wrap=True)
    table.add_row(*report.values())

    with open(output_path / f"results.{n_folds=}.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(report.keys()))
        writer.writeheader()
        writer.writerow(report)

    console = Console(record=True, width=500)
    console.print(table)
    console.save_html(output_path / f"results.{n_folds=}.html")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.exists():
        raise ValueError("data path does not exist")
    args.output.mkdir(exist_ok=True, parents=True)

    logger.info(f"Input data folder: {str(args.input)}")
    with logger.catch(onerror=lambda _: sys.exit(1)):
        main(args.input, args.output)
