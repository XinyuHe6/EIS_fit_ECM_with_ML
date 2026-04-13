from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize


# This module packages the ECM fitting/reconstruction path used in
# /home/henry/ECM-battery-research for the classification workflow.
ML_LABEL_TO_ECM = {
    "C1": "v3CM10",
    "C2": "v3CM9",
    "C3": "v3CM8",
    "C4": "v3CM4",
    "C5": "v3CM5",
    "C6": "v3CM6",
}


PARAMS_NAMES = {
    "v3CM4": ["L", "R0", "R1", "C1", "n1", "Aw"],
    "v3CM5": ["L", "R0", "R1", "R2", "C1", "C2", "n1", "n2", "Aw"],
    "v3CM6": ["L", "R0", "R1", "R2", "R3", "C1", "C2", "C3", "n1", "n2", "n3", "Aw"],
    "v3CM8": ["L", "R0", "R1", "R2", "R3", "C1", "C2", "C3", "n1", "n2", "n3", "Aw"],
    "v3CM9": ["L", "R0", "R1", "R2", "C1", "C2", "n1", "n2", "Aw"],
    "v3CM10": ["L", "R0", "R1", "C1", "n1", "Aw"],
}


INITIAL_GUESS = {
    "v3CM4": [1e-5, 0.005, 0.01, 0.05, 0.9, 0.001],
    "v3CM5": [1e-5, 0.005, 0.01, 0.01, 0.05, 0.05, 0.9, 0.9, 0.001],
    "v3CM6": [1e-5, 0.005, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.9, 0.9, 0.9, 0.001],
    "v3CM8": [1e-5, 0.005, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.9, 0.9, 0.9, 0.001],
    "v3CM9": [1e-5, 0.005, 0.01, 0.01, 0.05, 0.05, 0.9, 0.9, 0.001],
    "v3CM10": [1e-5, 0.005, 0.01, 0.05, 0.9, 0.001],
}


EPS = 1e-9
BOUNDS = {
    "v3CM4": [(EPS, 1), (EPS, 100), (EPS, 100), (EPS, 100), (0.8, 1), (EPS, 100)],
    "v3CM5": [(EPS, 1), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (0.8, 1), (0.8, 1), (EPS, 100)],
    "v3CM6": [(EPS, 1), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (0.8, 1), (0.8, 1), (0.8, 1), (EPS, 100)],
    "v3CM8": [(EPS, 1), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (0.8, 1), (0.8, 1), (0.8, 1), (EPS, 100)],
    "v3CM9": [(EPS, 1), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (EPS, 100), (0.8, 1), (0.8, 1), (EPS, 100)],
    "v3CM10": [(EPS, 1), (EPS, 100), (EPS, 100), (EPS, 100), (0.8, 1), (EPS, 100)],
}


ECM_NUM_RCS = {
    "v3CM4": 1,
    "v3CM5": 2,
    "v3CM6": 3,
    "v3CM8": 3,
    "v3CM9": 2,
    "v3CM10": 1,
}


def compute_v3CM4_impedance(params, angular_freq):
    L, R0, R1, C1, n1, Aw = params
    jw = 1j * angular_freq
    z_l = jw * L
    z_cpe1 = 1 / (C1 * (jw ** n1))
    z_w = (Aw * np.sqrt(2)) / np.sqrt(jw)
    z_rcaw1 = 1 / (1 / (R1 + z_w) + 1 / z_cpe1)
    return z_l + R0 + z_rcaw1


def compute_v3CM5_impedance(params, angular_freq):
    L, R0, R1, R2, C1, C2, n1, n2, Aw = params
    jw = 1j * angular_freq
    z_l = jw * L
    z_cpe1 = 1 / (C1 * (jw ** n1))
    z_cpe2 = 1 / (C2 * (jw ** n2))
    z_w = (Aw * np.sqrt(2)) / np.sqrt(jw)
    z_rc1 = 1 / (1 / R1 + 1 / z_cpe1)
    z_rcaw2 = 1 / (1 / (R2 + z_w) + 1 / z_cpe2)
    return z_l + R0 + z_rc1 + z_rcaw2


def compute_v3CM6_impedance(params, angular_freq):
    L, R0, R1, R2, R3, C1, C2, C3, n1, n2, n3, Aw = params
    jw = 1j * angular_freq
    z_l = jw * L
    z_cpe1 = 1 / (C1 * (jw ** n1))
    z_cpe2 = 1 / (C2 * (jw ** n2))
    z_cpe3 = 1 / (C3 * (jw ** n3))
    z_w = (Aw * np.sqrt(2)) / np.sqrt(jw)
    z_rc1 = 1 / (1 / R1 + 1 / z_cpe1)
    z_rc2 = 1 / (1 / R2 + 1 / z_cpe2)
    z_rcaw3 = 1 / (1 / (R3 + z_w) + 1 / z_cpe3)
    return z_l + R0 + z_rc1 + z_rc2 + z_rcaw3


def compute_v3CM8_impedance(params, angular_freq):
    L, R0, R1, R2, R3, C1, C2, C3, n1, n2, n3, Aw = params
    jw = 1j * angular_freq
    z_l = jw * L
    z_cpe1 = 1 / (C1 * (jw ** n1))
    z_cpe2 = 1 / (C2 * (jw ** n2))
    z_cpe3 = 1 / (C3 * (jw ** n3))
    z_w = (Aw * np.sqrt(2)) / np.sqrt(jw)
    z_rc1 = 1 / (1 / R1 + 1 / z_cpe1)
    z_rc2 = 1 / (1 / R2 + 1 / z_cpe2)
    z_rc3 = 1 / (1 / R3 + 1 / z_cpe3)
    return z_l + R0 + z_rc1 + z_rc2 + z_rc3 + z_w


def compute_v3CM9_impedance(params, angular_freq):
    L, R0, R1, R2, C1, C2, n1, n2, Aw = params
    jw = 1j * angular_freq
    z_l = jw * L
    z_cpe1 = 1 / (C1 * (jw ** n1))
    z_cpe2 = 1 / (C2 * (jw ** n2))
    z_w = (Aw * np.sqrt(2)) / np.sqrt(jw)
    z_rc1 = 1 / (1 / R1 + 1 / z_cpe1)
    z_rc2 = 1 / (1 / R2 + 1 / z_cpe2)
    return z_l + R0 + z_rc1 + z_rc2 + z_w


def compute_v3CM10_impedance(params, angular_freq):
    L, R0, R1, C1, n1, Aw = params
    jw = 1j * angular_freq
    z_l = jw * L
    z_cpe1 = 1 / (C1 * (jw ** n1))
    z_w = (Aw * np.sqrt(2)) / np.sqrt(jw)
    z_rc1 = 1 / (1 / R1 + 1 / z_cpe1)
    return z_l + R0 + z_rc1 + z_w


ECM_IMPEDANCE_MAP = {
    "v3CM4": compute_v3CM4_impedance,
    "v3CM5": compute_v3CM5_impedance,
    "v3CM6": compute_v3CM6_impedance,
    "v3CM8": compute_v3CM8_impedance,
    "v3CM9": compute_v3CM9_impedance,
    "v3CM10": compute_v3CM10_impedance,
}


def load_frequency_grid(num_points, freq_file="angular_freq.csv", freq_min_hz=0.1, freq_max_hz=10000.0):
    freq_path = Path(freq_file) if freq_file else None
    if freq_path and freq_path.exists():
        angular_freq = np.loadtxt(freq_path, delimiter=",", dtype=float).reshape(-1)
        if len(angular_freq) == num_points:
            return angular_freq, angular_freq / (2 * np.pi)
        print(
            f"[WARN] {freq_path} has {len(angular_freq)} points, "
            f"but EIS samples have {num_points}; using generated log grid."
        )

    freq_hz = np.logspace(np.log10(freq_min_hz), np.log10(freq_max_hz), num_points, endpoint=True)
    angular_freq = 2 * np.pi * freq_hz
    return angular_freq, freq_hz


def reconstruct_impedance_from_signal(original_signal):
    phase_deg = original_signal[:, 1].astype(float)
    magnitude = original_signal[:, 2].astype(float)
    return magnitude * np.exp(1j * np.deg2rad(phase_deg))


def complex_rmse(z_a, z_b):
    return float(np.sqrt(np.mean(np.abs(z_a - z_b) ** 2)))


def difference_metrics(z_a, z_b):
    abs_diff = np.abs(z_a - z_b)
    return {
        "rmse_complex": float(np.sqrt(np.mean(abs_diff ** 2))),
        "mae_complex": float(np.mean(abs_diff)),
        "max_abs_complex": float(np.max(abs_diff)),
        "rmse_real": float(np.sqrt(np.mean((z_a.real - z_b.real) ** 2))),
        "rmse_imag": float(np.sqrt(np.mean((z_a.imag - z_b.imag) ** 2))),
        "mape_magnitude": float(
            np.mean(np.abs(np.abs(z_a) - np.abs(z_b)) / np.maximum(np.abs(z_a), 1e-12))
        ),
    }


def compute_time_constant(R, C, n):
    try:
        if R <= 0 or C <= 0 or n <= 0:
            return np.inf
        return (R * C) ** (1 / n)
    except (FloatingPointError, ZeroDivisionError):
        return np.inf


def sort_by_tau(params, ecm_name):
    params = [float(v) for v in params]

    if ecm_name in ("v3CM4", "v3CM10"):
        return params

    if ecm_name in ("v3CM5", "v3CM9"):
        L, R0, R1, R2, C1, C2, n1, n2, Aw = params
        tau1 = compute_time_constant(R1, C1, n1)
        tau2 = compute_time_constant(R2, C2, n2)
        if tau1 > tau2:
            R1, R2 = R2, R1
            C1, C2 = C2, C1
            n1, n2 = n2, n1
        return [L, R0, R1, R2, C1, C2, n1, n2, Aw]

    if ecm_name in ("v3CM6", "v3CM8"):
        L, R0, R1, R2, R3, C1, C2, C3, n1, n2, n3, Aw = params
        rc_list = [
            (compute_time_constant(R1, C1, n1), R1, C1, n1),
            (compute_time_constant(R2, C2, n2), R2, C2, n2),
            (compute_time_constant(R3, C3, n3), R3, C3, n3),
        ]
        rc_list_sorted = sorted(rc_list, key=lambda item: item[0])
        r_sorted = [item[1] for item in rc_list_sorted]
        c_sorted = [item[2] for item in rc_list_sorted]
        n_sorted = [item[3] for item in rc_list_sorted]
        return [
            L,
            R0,
            r_sorted[0],
            r_sorted[1],
            r_sorted[2],
            c_sorted[0],
            c_sorted[1],
            c_sorted[2],
            n_sorted[0],
            n_sorted[1],
            n_sorted[2],
            Aw,
        ]

    return params


def cost_rmse_abs(params, z_exp, angular_freq, impedance_func):
    z_model = impedance_func(params, angular_freq)
    return complex_rmse(z_model, z_exp)


def perturb_initial_guess(base_guess, param_names, rng, eps=EPS):
    scale_map = {"l": 0.5, "r": 0.5, "c": 0.5, "n": 0.1, "a": 0.1}
    perturbed = []
    for value, name in zip(base_guess, param_names):
        scale = scale_map.get(name[:1].lower(), 0.2)
        factor = 1 + rng.uniform(-scale, scale)
        perturbed.append(max(eps, value * factor))
    return perturbed


def clip_to_bounds(values, bounds):
    clipped = []
    for value, (lower, upper) in zip(values, bounds):
        clipped.append(float(np.clip(value, lower, upper)))
    return clipped


def lsq_ecm_estimation(z_exp, angular_freq, ecm_name, impedance_func, initial_guess, bounds):
    def residual_function(params, angular_freq, z_exp):
        z_pred = impedance_func(params, angular_freq)
        residual = z_exp - z_pred
        return np.hstack((residual.real, residual.imag))

    lb, ub = zip(*bounds)
    result = least_squares(
        residual_function,
        initial_guess,
        args=(angular_freq, z_exp),
        bounds=(lb, ub),
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=5000,
    )

    params = sort_by_tau(result.x, ecm_name)
    z_fit = impedance_func(params, angular_freq)
    rmse = complex_rmse(z_fit, z_exp)
    return params, rmse, result


def minimize_ecm_estimation(z_exp, angular_freq, ecm_name, impedance_func, initial_guess, bounds, method):
    raw_cost = partial(cost_rmse_abs, z_exp=z_exp, angular_freq=angular_freq, impedance_func=impedance_func)
    if method == "Powell":
        result = minimize(
            raw_cost,
            initial_guess,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 1000, "xtol": 1e-8, "ftol": 1e-8, "disp": False},
        )
    else:
        result = minimize(
            raw_cost,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-8, "eps": 1e-8, "disp": False},
        )

    params = sort_by_tau(result.x, ecm_name)
    z_fit = impedance_func(params, angular_freq)
    rmse = complex_rmse(z_fit, z_exp)
    return params, rmse, result


def fit_ecm(z_exp, angular_freq, ecm_name, trial_num=3, method="LSQ", seed=1):
    if ecm_name not in ECM_IMPEDANCE_MAP:
        raise ValueError(f"Unsupported ECM model: {ecm_name}")

    impedance_func = ECM_IMPEDANCE_MAP[ecm_name]
    bounds = BOUNDS[ecm_name]
    base_guess = INITIAL_GUESS[ecm_name]
    param_names = PARAMS_NAMES[ecm_name]
    rng = np.random.default_rng(seed)

    best_attempt = None
    trial_records = []
    for trial_id in range(1, trial_num + 1):
        if trial_id == 1:
            initial_guess = list(base_guess)
        else:
            initial_guess = perturb_initial_guess(base_guess, param_names, rng)
        initial_guess = clip_to_bounds(initial_guess, bounds)

        try:
            if method == "LSQ":
                params, rmse, result = lsq_ecm_estimation(
                    z_exp,
                    angular_freq,
                    ecm_name,
                    impedance_func,
                    initial_guess,
                    bounds,
                )
            else:
                params, rmse, result = minimize_ecm_estimation(
                    z_exp,
                    angular_freq,
                    ecm_name,
                    impedance_func,
                    initial_guess,
                    bounds,
                    method,
                )
            success = bool(result.success)
            message = str(result.message)
        except Exception as exc:
            params = None
            rmse = np.inf
            success = False
            message = str(exc)

        attempt = {
            "trial_id": trial_id,
            "success": success,
            "fit_rmse": float(rmse),
            "params": params,
            "message": message,
            "initial_guess": initial_guess,
        }
        trial_records.append(attempt)

        if params is None:
            continue
        if best_attempt is None:
            best_attempt = attempt
            continue
        if success and not best_attempt["success"]:
            best_attempt = attempt
            continue
        if success == best_attempt["success"] and rmse < best_attempt["fit_rmse"]:
            best_attempt = attempt

    if best_attempt is None or best_attempt["params"] is None:
        raise RuntimeError(f"All fitting attempts failed for {ecm_name}")

    best_params = np.asarray(best_attempt["params"], dtype=float)
    z_fit = impedance_func(best_params, angular_freq)
    return {
        "ecm_name": ecm_name,
        "success": bool(best_attempt["success"]),
        "best_trial_id": int(best_attempt["trial_id"]),
        "message": best_attempt["message"],
        "fit_rmse": float(best_attempt["fit_rmse"]),
        "params": best_params,
        "z_fit": z_fit,
        "trial_records": trial_records,
    }


def expand_params(ecm_name, params):
    expanded = {}
    for param_name, param_value in zip(PARAMS_NAMES[ecm_name], params):
        expanded[param_name] = float(param_value)

    for rc_idx in range(1, ECM_NUM_RCS[ecm_name] + 1):
        r_key = f"R{rc_idx}"
        c_key = f"C{rc_idx}"
        n_key = f"n{rc_idx}"
        if r_key in expanded and c_key in expanded and n_key in expanded:
            tau = compute_time_constant(expanded[r_key], expanded[c_key], expanded[n_key])
            expanded[f"tau{rc_idx}"] = float(tau)
            if np.isfinite(tau) and tau > 0:
                expanded[f"freq{rc_idx}"] = float(1.0 / (2.0 * np.pi * tau))
            else:
                expanded[f"freq{rc_idx}"] = np.nan
    return expanded


def build_detail_dataframe(freq_hz, angular_freq, original_signal, z_meas, z_true_fit, z_predicted_fit):
    return pd.DataFrame(
        {
            "freq_hz": freq_hz,
            "angular_freq": angular_freq,
            "raw_signal_imag": original_signal[:, 0],
            "raw_signal_phase_deg": original_signal[:, 1],
            "raw_signal_magnitude": original_signal[:, 2],
            "measured_real": z_meas.real,
            "measured_imag": z_meas.imag,
            "measured_neg_imag": -z_meas.imag,
            "true_ecm_fit_real": z_true_fit.real,
            "true_ecm_fit_imag": z_true_fit.imag,
            "true_ecm_fit_neg_imag": -z_true_fit.imag,
            "predicted_ecm_fit_real": z_predicted_fit.real,
            "predicted_ecm_fit_imag": z_predicted_fit.imag,
            "predicted_ecm_fit_neg_imag": -z_predicted_fit.imag,
            "abs_diff_measured_vs_true_fit": np.abs(z_meas - z_true_fit),
            "abs_diff_measured_vs_predicted_fit": np.abs(z_meas - z_predicted_fit),
            "abs_diff_true_fit_vs_predicted_fit": np.abs(z_true_fit - z_predicted_fit),
        }
    )


def save_reconstruction_plot(plot_path, freq_hz, z_meas, z_true_fit, z_predicted_fit, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    axes[0].plot(z_meas.real, -z_meas.imag, "o-", color="black", label="Measured EIS")
    axes[0].plot(z_true_fit.real, -z_true_fit.imag, "s--", color="tab:green", label="True ECM fit")
    axes[0].plot(
        z_predicted_fit.real,
        -z_predicted_fit.imag,
        "d--",
        color="tab:red",
        label="Predicted ECM fit",
    )
    axes[0].set_xlabel("Re(Z) [Ohm]")
    axes[0].set_ylabel("-Im(Z) [Ohm]")
    axes[0].set_title("Nyquist")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="best")

    axes[1].plot(freq_hz, np.abs(z_true_fit - z_predicted_fit), "^-", color="tab:blue")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("|True ECM fit - predicted ECM fit| [Ohm]")
    axes[1].set_title("Reconstruction Difference")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def analyze_misclassified_samples(
    misclassified_df,
    original_signals,
    angular_freq,
    freq_hz,
    output_dir,
    rmse_threshold=1e-3,
    trial_num=3,
    method="LSQ",
    save_plots=False,
):
    output_dir = Path(output_dir)
    details_dir = output_dir / "reconstructed_eis"
    plots_dir = output_dir / "plots"
    details_dir.mkdir(parents=True, exist_ok=True)
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for row_pos, (_, row) in enumerate(misclassified_df.iterrows()):
        test_index = int(row["test_index"])
        true_label = row["true_label_name"]
        predicted_label = row["predicted_label_name"]
        true_ecm = ML_LABEL_TO_ECM[true_label]
        predicted_ecm = ML_LABEL_TO_ECM[predicted_label]
        sample_stub = f"test_{test_index:05d}__true_{true_label}__pred_{predicted_label}"

        base_summary = {
            "test_index": test_index,
            "true_label_index": int(row["true_label_index"]),
            "true_label_name": true_label,
            "predicted_label_index": int(row["predicted_label_index"]),
            "predicted_label_name": predicted_label,
            "true_ecm_name": true_ecm,
            "predicted_ecm_name": predicted_ecm,
            "predicted_probability_of_true_label": float(row["predicted_probability_of_true_label"]),
            "predicted_probability_of_predicted_label": float(row["predicted_probability_of_predicted_label"]),
            "neglectable_rmse_threshold": float(rmse_threshold),
            "fit_trial_num": int(trial_num),
            "fit_method": method,
        }

        original_signal = original_signals[row_pos]
        z_meas = reconstruct_impedance_from_signal(original_signal)

        try:
            true_fit = fit_ecm(z_meas, angular_freq, true_ecm, trial_num=trial_num, method=method)
            predicted_fit = fit_ecm(
                z_meas,
                angular_freq,
                predicted_ecm,
                trial_num=trial_num,
                method=method,
            )

            metrics_measured_true = difference_metrics(z_meas, true_fit["z_fit"])
            metrics_measured_predicted = difference_metrics(z_meas, predicted_fit["z_fit"])
            metrics_true_predicted = difference_metrics(true_fit["z_fit"], predicted_fit["z_fit"])
            rmse_between_reconstructions = metrics_true_predicted["rmse_complex"]
            is_neglectable = rmse_between_reconstructions <= rmse_threshold

            detail_path = details_dir / f"{sample_stub}.csv"
            detail_df = build_detail_dataframe(
                freq_hz,
                angular_freq,
                original_signal,
                z_meas,
                true_fit["z_fit"],
                predicted_fit["z_fit"],
            )
            detail_df.to_csv(detail_path, index=False)

            plot_path = ""
            if save_plots:
                plot_path = str(plots_dir / f"{sample_stub}.png")
                save_reconstruction_plot(
                    plot_path,
                    freq_hz,
                    z_meas,
                    true_fit["z_fit"],
                    predicted_fit["z_fit"],
                    (
                        f"{sample_stub}\n"
                        f"RMSE(true ECM reconstruction, predicted ECM reconstruction)="
                        f"{rmse_between_reconstructions:.4e}"
                    ),
                )

            summary = {
                **base_summary,
                "fit_error": "",
                "true_fit_success": bool(true_fit["success"]),
                "predicted_fit_success": bool(predicted_fit["success"]),
                "true_fit_best_trial_id": int(true_fit["best_trial_id"]),
                "predicted_fit_best_trial_id": int(predicted_fit["best_trial_id"]),
                "true_fit_message": true_fit["message"],
                "predicted_fit_message": predicted_fit["message"],
                "rmse_measured_vs_true_fit": metrics_measured_true["rmse_complex"],
                "rmse_measured_vs_predicted_fit": metrics_measured_predicted["rmse_complex"],
                "rmse_true_fit_vs_predicted_fit": rmse_between_reconstructions,
                "mae_true_fit_vs_predicted_fit": metrics_true_predicted["mae_complex"],
                "max_abs_true_fit_vs_predicted_fit": metrics_true_predicted["max_abs_complex"],
                "mape_mag_measured_vs_true_fit": metrics_measured_true["mape_magnitude"],
                "mape_mag_measured_vs_predicted_fit": metrics_measured_predicted["mape_magnitude"],
                "is_neglectable_misclassification": bool(is_neglectable),
                "detail_csv": str(detail_path),
                "plot_png": plot_path,
            }

            for key, value in expand_params(true_ecm, true_fit["params"]).items():
                summary[f"true_fit_{key}"] = value
            for key, value in expand_params(predicted_ecm, predicted_fit["params"]).items():
                summary[f"predicted_fit_{key}"] = value
        except Exception as exc:
            summary = {
                **base_summary,
                "fit_error": str(exc),
                "true_fit_success": False,
                "predicted_fit_success": False,
                "rmse_measured_vs_true_fit": np.nan,
                "rmse_measured_vs_predicted_fit": np.nan,
                "rmse_true_fit_vs_predicted_fit": np.nan,
                "mae_true_fit_vs_predicted_fit": np.nan,
                "max_abs_true_fit_vs_predicted_fit": np.nan,
                "mape_mag_measured_vs_true_fit": np.nan,
                "mape_mag_measured_vs_predicted_fit": np.nan,
                "is_neglectable_misclassification": False,
                "detail_csv": "",
                "plot_png": "",
            }
            print(f"[WARN] ECM fitting failed for {sample_stub}: {exc}")

        summary_rows.append(summary)
        if (row_pos + 1) % 25 == 0 or row_pos + 1 == len(misclassified_df):
            print(f"Processed ECM neglectable analysis: {row_pos + 1}/{len(misclassified_df)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "neglectable_misclassification_summary.csv", index=False)
    return summary_df
