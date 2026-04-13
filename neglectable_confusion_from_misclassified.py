import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecm_neglectable_analysis import analyze_misclassified_samples, load_frequency_grid


LABEL_NAMES = np.array(["C1", "C2", "C3", "C4", "C5", "C6"])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze misclassified_EIS.csv, fit true/predicted ECM models, "
            "judge neglectable misclassifications, and save 6x6 confusion matrices."
        )
    )
    parser.add_argument(
        "--misclassified-csv",
        default="misclassified_EIS.csv",
        help="Input CSV exported by Classification_ECM.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <misclassified-csv folder>/neglectable_confusion_from_csv.",
    )
    parser.add_argument(
        "--rmse-threshold",
        type=float,
        default=1e-3,
        help="RMSE threshold for treating true/predicted reconstructed EIS as neglectable.",
    )
    parser.add_argument(
        "--fit-trials",
        type=int,
        default=1,
        help="Number of fitting trials per ECM model.",
    )
    parser.add_argument(
        "--fit-method",
        choices=["LSQ", "LBFGS", "Powell"],
        default="LSQ",
        help="Optimizer for ECM fitting.",
    )
    parser.add_argument(
        "--freq-file",
        default="angular_freq.csv",
        help="Angular-frequency CSV. A generated log grid is used if this file is missing/mismatched.",
    )
    parser.add_argument(
        "--freq-min-hz",
        type=float,
        default=0.1,
        help="Fallback minimum frequency when --freq-file is unavailable.",
    )
    parser.add_argument(
        "--freq-max-hz",
        type=float,
        default=10000.0,
        help="Fallback maximum frequency when --freq-file is unavailable.",
    )
    parser.add_argument(
        "--save-reconstruction-plots",
        action="store_true",
        help="Also save one reconstructed EIS plot per misclassified sample.",
    )
    return parser.parse_args()


def collect_point_columns(df):
    pattern = re.compile(r"^(imag|phase|mag)_pt_(\d+)$")
    columns_by_signal = {"imag": {}, "phase": {}, "mag": {}}

    for column in df.columns:
        match = pattern.match(column)
        if match:
            signal_name, point_text = match.groups()
            columns_by_signal[signal_name][int(point_text)] = column

    common_points = set(columns_by_signal["imag"])
    common_points &= set(columns_by_signal["phase"])
    common_points &= set(columns_by_signal["mag"])
    point_numbers = sorted(common_points)

    if not point_numbers:
        raise ValueError(
            "No complete imag_pt_XX / phase_pt_XX / mag_pt_XX columns found in the CSV."
        )

    missing = []
    for point_number in range(point_numbers[0], point_numbers[-1] + 1):
        for signal_name in ("imag", "phase", "mag"):
            if point_number not in columns_by_signal[signal_name]:
                missing.append(f"{signal_name}_pt_{point_number:02d}")
    if missing:
        raise ValueError("Missing EIS point columns: " + ", ".join(missing[:20]))

    return point_numbers, columns_by_signal


def load_original_signals(df):
    point_numbers, columns_by_signal = collect_point_columns(df)
    original_signals = np.empty((len(df), len(point_numbers), 3), dtype=float)

    for point_idx, point_number in enumerate(point_numbers):
        original_signals[:, point_idx, 0] = df[columns_by_signal["imag"][point_number]].astype(float)
        original_signals[:, point_idx, 1] = df[columns_by_signal["phase"][point_number]].astype(float)
        original_signals[:, point_idx, 2] = df[columns_by_signal["mag"][point_number]].astype(float)

    return original_signals


def validate_input_columns(df):
    required_columns = [
        "test_index",
        "true_label_index",
        "true_label_name",
        "predicted_label_index",
        "predicted_label_name",
        "predicted_probability_of_true_label",
        "predicted_probability_of_predicted_label",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))


def build_confusion_matrices(summary_df):
    misclassified_matrix = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=int)
    neglectable_matrix = np.zeros_like(misclassified_matrix)

    for _, row in summary_df.iterrows():
        true_idx = int(row["true_label_index"])
        predicted_idx = int(row["predicted_label_index"])
        if true_idx < 0 or true_idx >= len(LABEL_NAMES):
            raise ValueError(f"Invalid true_label_index: {true_idx}")
        if predicted_idx < 0 or predicted_idx >= len(LABEL_NAMES):
            raise ValueError(f"Invalid predicted_label_index: {predicted_idx}")

        misclassified_matrix[true_idx, predicted_idx] += 1
        is_neglectable = bool(row.get("is_neglectable_misclassification", False))
        if is_neglectable:
            neglectable_matrix[true_idx, predicted_idx] += 1

    return misclassified_matrix, neglectable_matrix


def save_matrix_csv(matrix, save_path):
    matrix_df = pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES)
    matrix_df.index.name = "true_model"
    matrix_df.columns.name = "predicted_model"
    matrix_df.to_csv(save_path)


def save_combined_matrix_csv(misclassified_matrix, neglectable_matrix, save_path):
    combined = np.empty(misclassified_matrix.shape, dtype=object)
    for true_idx in range(misclassified_matrix.shape[0]):
        for predicted_idx in range(misclassified_matrix.shape[1]):
            combined[true_idx, predicted_idx] = (
                f"{misclassified_matrix[true_idx, predicted_idx]} / "
                f"{neglectable_matrix[true_idx, predicted_idx]}"
            )

    combined_df = pd.DataFrame(combined, index=LABEL_NAMES, columns=LABEL_NAMES)
    combined_df.index.name = "true_model"
    combined_df.columns.name = "predicted_model"
    combined_df.to_csv(save_path)


def save_confusion_plot(misclassified_matrix, neglectable_matrix, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    image = ax.imshow(misclassified_matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Misclassification count")

    ax.set_xticks(np.arange(len(LABEL_NAMES)))
    ax.set_yticks(np.arange(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted ECM model")
    ax.set_ylabel("True ECM model")
    ax.set_title(title)

    threshold = misclassified_matrix.max() / 2 if misclassified_matrix.size else 0
    for true_idx in range(misclassified_matrix.shape[0]):
        for predicted_idx in range(misclassified_matrix.shape[1]):
            count = int(misclassified_matrix[true_idx, predicted_idx])
            neglectable_count = int(neglectable_matrix[true_idx, predicted_idx])
            annotation = f"{count}\nNeg:{neglectable_count}"
            ax.text(
                predicted_idx,
                true_idx,
                annotation,
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    args = parse_args()
    misclassified_csv = Path(args.misclassified_csv)
    output_dir = Path(args.output_dir) if args.output_dir else misclassified_csv.parent / "neglectable_confusion_from_csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    misclassified_df = pd.read_csv(misclassified_csv)
    validate_input_columns(misclassified_df)
    original_signals = load_original_signals(misclassified_df)

    angular_freq, freq_hz = load_frequency_grid(
        original_signals.shape[1],
        freq_file=args.freq_file,
        freq_min_hz=args.freq_min_hz,
        freq_max_hz=args.freq_max_hz,
    )

    summary_df = analyze_misclassified_samples(
        misclassified_df=misclassified_df,
        original_signals=original_signals,
        angular_freq=angular_freq,
        freq_hz=freq_hz,
        output_dir=output_dir,
        rmse_threshold=args.rmse_threshold,
        trial_num=args.fit_trials,
        method=args.fit_method,
        save_plots=args.save_reconstruction_plots,
    )

    misclassified_matrix, neglectable_matrix = build_confusion_matrices(summary_df)
    total_misclassified = int(misclassified_matrix.sum())
    total_neglectable = int(neglectable_matrix.sum())

    save_matrix_csv(misclassified_matrix, output_dir / "misclassification_matrix.csv")
    save_matrix_csv(neglectable_matrix, output_dir / "neglectable_misclassification_matrix.csv")
    save_combined_matrix_csv(
        misclassified_matrix,
        neglectable_matrix,
        output_dir / "misclassification_and_neglectable_matrix.csv",
    )
    save_confusion_plot(
        misclassified_matrix,
        neglectable_matrix,
        output_dir / "misclassification_with_neglectable_matrix.png",
        (
            "Misclassification count with neglectable count\n"
            f"Total={total_misclassified}, Neglectable={total_neglectable}, "
            f"RMSE threshold={args.rmse_threshold}"
        ),
    )

    metrics_df = pd.DataFrame(
        [
            {
                "misclassified_csv": str(misclassified_csv),
                "total_misclassified": total_misclassified,
                "total_neglectable_misclassification": total_neglectable,
                "rmse_threshold": float(args.rmse_threshold),
                "fit_trials": int(args.fit_trials),
                "fit_method": args.fit_method,
                "output_dir": str(output_dir),
            }
        ]
    )
    metrics_df.to_csv(output_dir / "neglectable_confusion_metrics.csv", index=False)

    print("Saved neglectable analysis summary:", output_dir / "neglectable_misclassification_summary.csv")
    print("Saved misclassification matrix:", output_dir / "misclassification_matrix.csv")
    print("Saved neglectable matrix:", output_dir / "neglectable_misclassification_matrix.csv")
    print("Saved combined matrix:", output_dir / "misclassification_and_neglectable_matrix.csv")
    print("Saved matrix plot:", output_dir / "misclassification_with_neglectable_matrix.png")
    print(f"Total misclassified: {total_misclassified}")
    print(f"Total neglectable misclassified: {total_neglectable}")


if __name__ == "__main__":
    main()
