import csv
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pywt
from bcipy.helpers.load import load_json_parameters, load_raw_data
from bcipy.helpers.triggers import trigger_decoder
from bcipy.signal.model.pca_rda_kde import PcaRdaKdeModel
from bcipy.signal.process import get_default_transform
from loguru import logger
from preprocessing import AlphaTransformer
from pyriemann.classification import TSclassifier
from pyriemann.estimation import Covariances
from rich.console import Console
from rich.table import Table
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.utils._testing import ignore_warnings


def cwt(data: np.ndarray, freq: int, fs: int) -> np.ndarray:
    """
    Transform data into frequency domain using Continuous Wavelet Transform (CWT).
    Keeps only a single wavelet scale, specified by `freq`.

    Args:
        data (np.ndarray): shape (trials, channels, time)
        freq (int): frequency of wavelet to keep
        fs (int): sampling rate of data (Hz)

    Returns:
        np.ndarray: frequency domain data, shape (trials, wavelet_scales*channels, time)
                    Note that here we only use 1 wavelet_scale.
    """
    wavelet = "cmor1.5-1.0"
    scales = pywt.central_frequency(wavelet) * fs / np.array(freq)
    all_coeffs = []
    for trial in data:
        coeffs, _ = pywt.cwt(trial, scales, wavelet)  # shape == (scales, channels, time)
        all_coeffs.append(coeffs)

    final_data = np.stack(all_coeffs)
    if np.any(np.iscomplex(final_data)):
        logger.info("Converting complex to real")
        final_data = np.abs(final_data) ** 2

    # have shape == (trials, freqs, channels, time)
    # want shape == (trials, freqs*channels, time)
    return final_data.reshape(final_data.shape[0], -1, final_data.shape[-1])


def load_data(data_folder: Path, trial_length=2.5, pre_stim_offset=-1.25):
    """Loads raw data, and performs preprocessing by notch filtering, bandpass filtering, and downsampling.

    Args:
        data_folder (Path): path to raw data in BciPy format
        trial_length (float): length of each trial in seconds
        pre_stim_offset (float): window of time before stimulus onset to include in analysis

    Returns:
        np.ndarray: data, shape (trials, channels, time)
        np.ndarray: labels, shape (trials,)
        int: sampling rate (Hz)
    """
    parameters = load_json_parameters(data_folder / "parameters.json", value_cast=True)

    # extract relevant session information from parameters file
    triggers_file = parameters.get("trigger_file_name", "triggers.txt")
    raw_data_file = parameters.get("raw_data_name", "raw_data.csv")

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")  # default 2
    notch_filter = parameters.get("notch_filter_frequency")  # default 60
    hp_filter = parameters.get("filter_high")  # default 45
    lp_filter = parameters.get("filter_low")  # default 2
    filter_order = parameters.get("filter_order")  # default 2

    # get offset and k folds
    static_offset = parameters.get("static_trigger_offset")  # default 0.1

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    fs = raw_data.sample_rate

    logger.info(f"Channels read from csv: {channels}")
    logger.info(f"Device type: {type_amp}")

    default_transform = get_default_transform(
        sample_rate_hz=fs,
        notch_freq_hz=notch_filter,
        bandpass_low=lp_filter,
        bandpass_high=hp_filter,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )
    data, fs = default_transform(raw_data.by_channel(), fs)

    # Process triggers.txt
    _, t_t_i, t_i, offset = trigger_decoder(mode="calibration", trigger_path=f"{data_folder}/{triggers_file}")

    offset = offset + static_offset + pre_stim_offset

    # Channel map can be checked from raw_data.csv file.
    # The timestamp column is already excluded.
    # channel_names = ["P4", "Fz", "Pz", "F7", "PO8", "PO7", "Oz"]
    channel_map = [0, 0, 1, 0, 1, 1, 1, 0]
    data, labels = PcaRdaKdeModel.reshaper(
        trial_labels=t_t_i,
        timing_info=t_i,
        eeg_data=data,
        fs=fs,
        trials_per_inquiry=parameters.get("stim_length"),
        offset=offset,
        channel_map=channel_map,
        trial_length=trial_length,
    )
    return data.transpose([1, 0, 2]), labels, fs


def fit(data: np.ndarray, labels: np.ndarray, n_folds: int, flatten_data: bool, clf: Any) -> Dict[str, str]:
    """Perform K-fold cross-validation to evaluate model performance.

    Args:
        data (np.ndarray): Pre-processed EEG data
        labels (np.ndarray):
        n_folds (int):
        flatten_data (bool): whether data must be flattened for the specified model
        clf (Any): model to evaluate

    Returns:
        Dict[str, str]: mean and standard deviation for each metric being tracked.
    """
    np.random.seed(1)

    if flatten_data:
        data = flatten(data)

    results = cross_validate(
        clf,
        data,
        labels,
        cv=n_folds,
        n_jobs=-1,
        return_train_score=True,
        scoring=["balanced_accuracy", "roc_auc"],
    )

    report = {
        "avg_fit_time": results["fit_time"].mean(),
        "std_fit_time": results["fit_time"].std(),
        "avg_score_time": results["score_time"].mean(),
        "std_score_time": results["score_time"].std(),
        "avg_train_roc_auc": results["train_roc_auc"].mean(),
        "std_train_roc_auc": results["train_roc_auc"].std(),
        "avg_test_roc_auc": results["test_roc_auc"].mean(),
        "std_test_roc_auc": results["test_roc_auc"].std(),
        "avg_train_balanced_accuracy": results["train_balanced_accuracy"].mean(),
        "std_train_balanced_accuracy": results["train_balanced_accuracy"].std(),
        "avg_test_balanced_accuracy": results["test_balanced_accuracy"].mean(),
        "std_test_balanced_accuracy": results["test_balanced_accuracy"].std(),
    }
    report = {k: str(round(v, 3)) for k, v in report.items()}
    return report


def make_plots(data: np.ndarray, labels: np.ndarray, outpath: Path, vlines: Optional[List[int]] = None) -> None:
    """Make plots for inspecting intermediate data
    Produces grand average across trials and channels for target and non-target responses."""
    targets_data = data[labels == 1, :, :]
    nontargets_data = data[labels == 0, :, :]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(targets_data.mean(axis=(0, 1)), c="r", label="target")  # average across channels and trials
    ax.plot(nontargets_data.mean(axis=(0, 1)), c="b", label="non-target")
    ax.axvline(int(targets_data.shape[-1] / 2), c="k")
    targets_sem = targets_data.std(axis=(0, 1)) / np.sqrt(targets_data.shape[0])
    nontargets_sem = targets_data.std(axis=(0, 1)) / np.sqrt(nontargets_data.shape[0])

    # error bars
    ax.fill_between(
        np.arange(targets_data.shape[-1]),
        targets_data.mean(axis=(0, 1)) + targets_sem,
        targets_data.mean(axis=(0, 1)) - targets_sem,
        color="r",
        alpha=0.5,
    )
    ax.fill_between(
        np.arange(nontargets_data.shape[-1]),
        nontargets_data.mean(axis=(0, 1)) + nontargets_sem,
        nontargets_data.mean(axis=(0, 1)) - nontargets_sem,
        color="b",
        alpha=0.5,
    )
    if vlines:
        for v in vlines:
            ax.axvline(v, linestyle="--", alpha=0.3)

    ax.set_title("Average +/- SEM")
    plt.legend()
    fig.savefig(outpath, bbox_inches="tight", dpi=300)


def flatten(data):
    return data.reshape(data.shape[0], -1)


@ignore_warnings(category=ConvergenceWarning)
def main(input_path: Path, output_path: Path, hparam_tuning: bool, z_score_per_trial: bool):
    data, labels, fs = load_data(input_path)

    # CWT preprocess
    make_plots(data, labels, output_path / "0.raw_data.png")
    data = cwt(data, args.freq, fs)
    make_plots(data, labels, output_path / "1.cwt_data.png")
    logger.info(data.shape)

    baseline_duration_s = 0.5
    response_duration_s = 0.5
    default_baseline_start_s = 0.65
    default_response_start_s = 1.25 + 0.3
    if hparam_tuning:
        preprocessing_pipeline = Pipeline(
            steps=[
                (
                    "alpha",
                    AlphaTransformer(
                        baseline_start_s=default_baseline_start_s,
                        baseline_duration_s=baseline_duration_s,
                        response_start_s=default_response_start_s,
                        response_duration_s=response_duration_s,
                        sample_rate_hz=fs,
                    ),
                ),
                ("flatten", FunctionTransformer(flatten)),
                ("logistic", LogisticRegression(class_weight="balanced")),
            ]
        )
        parameters_to_tune = {
            # baseline can be anywhere in [200ms, stim - 100ms]
            "alpha__baseline_start_s": np.linspace(0.2, (2.5 / 2) - baseline_duration_s - 0.1, 10),
            # response can be anywhere in [stim + 150ms, end - 200ms]
            "alpha__response_start_s": np.linspace((2.5 / 2) + 0.15, 2.5 - 0.2 - response_duration_s, 10),
        }

        logger.warning("WARNING - leaking test data")
        cv = GridSearchCV(
            estimator=preprocessing_pipeline, param_grid=parameters_to_tune, scoring="balanced_accuracy", n_jobs=-1
        )
        cv.fit(data, labels)
        baseline_start_s = cv.best_params_["alpha__baseline_start_s"]
        response_start_s = cv.best_params_["alpha__response_start_s"]
    else:
        baseline_start_s = default_baseline_start_s
        response_start_s = default_response_start_s

    with open(output_path / "window_params.txt", "w") as fh:
        print(f"{baseline_start_s=:.2f}s", file=fh)
        print(f"{response_start_s=:.2f}s", file=fh)
        print(f"{z_score_per_trial=:b}", file=fh)

    z_scorer = AlphaTransformer(
        baseline_start_s=baseline_start_s,
        baseline_duration_s=baseline_duration_s,
        response_start_s=response_start_s,
        response_duration_s=response_duration_s,
        sample_rate_hz=fs,
        z_score_per_trial=z_score_per_trial,
    )

    # The copy we care about for modeling
    logger.info(f"{data.min()=}, {data.mean()=}, {data.max()=}")
    z_transformed_target_window = z_scorer.transform(data)
    z_transformed_entire_data = z_scorer.transform(data, do_slice=False)
    # Copy of entire window for plotting
    logger.info(
        f"{z_transformed_target_window.min()=}, "
        + f"{z_transformed_target_window.mean()=}, "
        + f"{z_transformed_target_window.max()=}"
    )

    make_plots(z_transformed_target_window, labels, output_path / "2.z_target_window.png")
    vlines = [
        int(baseline_start_s * fs),
        int((baseline_start_s + baseline_duration_s) * fs),
        int(response_start_s * fs),
        int((response_start_s + response_duration_s) * fs),
    ]
    make_plots(z_transformed_entire_data, labels, output_path / "3.z_entire_data.png", vlines=vlines)

    ts_logr = make_pipeline(Covariances(), TSclassifier(clf=LogisticRegression(class_weight="balanced")))

    reports = []
    lr_kw = {"max_iter": 200, "solver": "liblinear"}
    for model_name, flatten_data, clf in [
        ("Uniform random", True, DummyClassifier(strategy="uniform")),
        ("LogisticRegression, balanced", True, LogisticRegression(class_weight="balanced", **lr_kw)),
        ("Multi-layer Perceptron", True, MLPClassifier()),
        ("Support Vector Classifier, balanced", True, SVC(probability=True, class_weight="balanced")),
        ("Tangent Space, Logistic Regression, balanced", False, ts_logr),
    ]:
        n_folds = 10
        logger.info(f"Run model class: {model_name}")
        report = fit(z_transformed_target_window, labels, n_folds, flatten_data, clf)
        report["name"] = model_name
        reports.append(report)

    table = Table(title=f"Alpha Classifier Comparison ({n_folds}-fold cross validation)")
    colors = cycle(["green", "blue"])

    col_names = [
        ("Model Name", "name"),
        ("Avg fit time", "avg_fit_time"),
        ("Std fit time", "std_fit_time"),
        ("Avg score time", "avg_score_time"),
        ("Std score time", "std_score_time"),
        ("Avg train roc auc", "avg_train_roc_auc"),
        ("Std train roc auc", "std_train_roc_auc"),
        ("Avg test roc auc", "avg_test_roc_auc"),
        ("Std test roc auc", "std_test_roc_auc"),
        ("Avg train balanced accuracy", "avg_train_balanced_accuracy"),
        ("Std train balanced accuracy", "std_train_balanced_accuracy"),
        ("Avg test balanced accuracy", "avg_test_balanced_accuracy"),
        ("Std test balanced accuracy", "std_test_balanced_accuracy"),
    ]

    for col_name, color in zip(col_names, colors):
        table.add_column(col_name[0], style=color, no_wrap=True)

    with open(output_path / f"results.{n_folds=}.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[c[1] for c in col_names])
        writer.writeheader()
        for report in reports:
            table.add_row(*[report[c[1]] for c in col_names])
            writer.writerow(report)

    console = Console(record=True, width=500)
    console.print(table)
    console.save_html(output_path / f"results.{n_folds=}.html")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, help="Path to data folder", required=True)
    p.add_argument("--output", type=Path, help="Path to save outputs", required=True)
    p.add_argument("--freq", type=float, help="Frequency to keep after CWT", default=10)
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--z_score_per_trial", action="store_true", default=False, help="baseline per channel, or per channel per trial"
    )
    group.add_argument("--hparam_tuning", action="store_true", default=False)
    args = p.parse_args()

    if not args.input.exists():
        raise ValueError("data path does not exist")

    args.output.mkdir(exist_ok=True, parents=True)

    logger.info(f"Input data folder: {str(args.input)}")
    logger.info(f"Selected freq: {str(args.freq)}")
    with logger.catch(onerror=lambda _: sys.exit(1)):
        main(args.input, args.output, args.hparam_tuning, args.z_score_per_trial)
