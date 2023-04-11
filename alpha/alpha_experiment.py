import csv
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pywt
from bcipy.helpers.load import load_json_parameters, load_raw_data, load_experimental_data
from bcipy.helpers.triggers import trigger_decoder, TriggerType

from bcipy.signal.process import get_default_transform, filter_inquiries
from loguru import logger
from preprocessing import AlphaTransformer
from base_model import BasePcaRdaKdeModel
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

from bcipy.helpers.acquisition import analysis_channels
from bcipy.config import DEFAULT_PARAMETERS_PATH, TRIGGER_FILENAME, RAW_DATA_FILENAME, STATIC_AUDIO_PATH


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


def load_data(data_folder: Path, trial_length=None, pre_stim=0.0, alpha=False):
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
    # Load parameters
    parameters = load_json_parameters(Path(data_folder, "parameters.json"), value_cast=True)
    poststim_length = trial_length if trial_length is not None else parameters.get("trial_length")
    pre_stim = pre_stim if pre_stim > 0.0 else parameters.get("prestim_length")

    trials_per_inquiry = parameters.get("stim_length")
    # The task buffer length defines the min time between two inquiries
    # We use half of that time here to buffer during transforms
    buffer = int(parameters.get("task_buffer_length") / 2)
    raw_data_file = f"{RAW_DATA_FILENAME}.csv"

    # get signal filtering information
    downsample_rate = parameters.get("down_sampling_rate")
    notch_filter = parameters.get("notch_filter_frequency")
    filter_high = parameters.get("filter_high")
    filter_low = parameters.get("filter_low")
    filter_order = parameters.get("filter_order")
    static_offset = parameters.get("static_trigger_offset")

    logger.info(
        f"\nData processing settings: \n"
        f"Filter: [{filter_low}-{filter_high}], Order: {filter_order},"
        f" Notch: {notch_filter}, Downsample: {downsample_rate} \n"
        f"Poststimulus: {poststim_length}s, Prestimulus: {pre_stim}s, Buffer: {buffer}s \n"
        f"Static offset: {static_offset}"
    )

    # Load raw data
    raw_data = load_raw_data(Path(data_folder, raw_data_file))
    channels = raw_data.channels
    type_amp = raw_data.daq_type
    sample_rate = raw_data.sample_rate

    # setup filtering
    default_transform = get_default_transform(
        sample_rate_hz=sample_rate,
        notch_freq_hz=notch_filter,
        bandpass_low=filter_low,
        bandpass_high=filter_high,
        bandpass_order=filter_order,
        downsample_factor=downsample_rate,
    )

    logger.info(f"Channels read from csv: {channels}")
    logger.info(f"Device type: {type_amp}, fs={sample_rate}")

    k_folds = parameters.get("k_folds")
    model = BasePcaRdaKdeModel(k_folds=k_folds)

    # Process triggers.txt files
    trigger_targetness, trigger_timing, trigger_symbols = trigger_decoder(
        offset=static_offset,
        trigger_path=f"{data_folder}/{TRIGGER_FILENAME}",
        exclusion=[TriggerType.PREVIEW, TriggerType.EVENT, TriggerType.FIXATION],
    )
    # Channel map can be checked from raw_data.csv file or the devices.json located in the acquisition module
    # The timestamp column [0] is already excluded.
    channel_map = analysis_channels(channels, type_amp)
    data, fs = raw_data.by_channel()

    inquiries, inquiry_labels, inquiry_timing = model.reshaper(
        trial_targetness_label=trigger_targetness,
        timing_info=trigger_timing,
        eeg_data=data,
        sample_rate=sample_rate,
        trials_per_inquiry=trials_per_inquiry,
        channel_map=channel_map,
        poststimulus_length=poststim_length,
        prestimulus_length=buffer + pre_stim,
        transformation_buffer=buffer + poststim_length,
    )

    inquiries, fs = filter_inquiries(inquiries, default_transform, sample_rate)
    trial_duration_samples = int(poststim_length * fs)
    if alpha:
        pre_stim_duration_samples = int(pre_stim * fs)
    else:
        pre_stim_duration_samples = 0
    data = model.reshaper.extract_trials(
        inquiries, trial_duration_samples, inquiry_timing, downsample_rate, prestimulus_samples=pre_stim_duration_samples)

    # define the training classes using integers, where 0=nontargets/1=targets
    labels = inquiry_labels.flatten()
    # breakpoint()
    data = np.transpose(data, (1, 0, 2))
    return data, labels, fs


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
def main(input_path: Path, freq: float, hparam_tuning: bool, z_score_per_trial: bool, output_path: Optional[Path] = None):
    data, labels, fs = load_data(input_path, trial_length=1.25, pre_stim=1.25, alpha=True)

    # set output path to input path if not specified
    output_path = output_path or input_path

    # CWT preprocess
    make_plots(data, labels, output_path / "0.raw_data.png")
    data = cwt(data, freq, fs)
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

    table = Table(title=f"Alpha Classifier Comparisons ({n_folds}-fold cross validation)")
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

    with open(output_path / f"ALPHAresults.{n_folds=}.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[c[1] for c in col_names])
        writer.writeheader()
        for report in reports:
            table.add_row(*[report[c[1]] for c in col_names])
            writer.writerow(report)

    console = Console(record=True, width=500)
    console.print(table)
    console.save_html(output_path / f"ALPHAresults.{n_folds=}.html")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    # trial length in seconds for alpha band: 1.25s before and 1.25s after response; z-scored per trial is False and hparam tuning is True/False (make sure both work)
    p.add_argument("--input", type=Path, help="Path to data folder", required=False, default=None)
    p.add_argument("--output", type=Path, help="Path to save outputs", required=False, default=None)
    p.add_argument("--freq", type=float, help="Frequency to keep after CWT", required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--z_score_per_trial", action="store_true", default=False, help="baseline per channel, or per channel per trial"
    )
    group.add_argument("--hparam_tuning", action="store_true", default=False)
    args = p.parse_args()

    if args.input is None:
        folder_path = Path(load_experimental_data())
    else:
        folder_path = args.input


    if not folder_path.exists():
        raise ValueError("data path does not exist")

    logger.info(f"Input data folder: {str(args.input)}")
    logger.info(f"Selected freq: {str(args.freq)}")
    with logger.catch(onerror=lambda _: sys.exit(1)):
        main(folder_path, args.freq, args.hparam_tuning, args.z_score_per_trial, output_path=args.output)
