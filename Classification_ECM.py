#Tensorflow Version 2.7 is needed
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.io
from tensorflow import keras
import tensorflow as tf
import datetime
from numpy import unique
from numpy import argmax
from pandas import read_csv
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ecm_neglectable_analysis import analyze_misclassified_samples, load_frequency_grid

DEFAULT_DROP_RATE = 0.35
DEFAULT_CONV_FILTERS = (32, 64, 128)
DEFAULT_DENSE_UNITS = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 200
DEFAULT_NEGLECTABLE_RMSE_THRESHOLDS = (1e-3, 1e-2)
DEFAULT_INPUT_SIGNAL_LAYOUT = "frequency_imag_real"
LABEL_NAMES = np.array(["C1", "C2", "C3", "C4", "C5", "C6"])
FEATURE_NAMES_BY_LAYOUT = {
    "frequency_imag_real": ("freq", "imag", "real"),
    "imag_phase_mag": ("imag", "phase", "mag"),
}


def str_to_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def unique_float_sequence(values):
    ordered_values = []
    seen = set()
    for value in values:
        normalized_value = float(value)
        normalized_key = np.format_float_positional(normalized_value, trim="-")
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        ordered_values.append(normalized_value)
    return tuple(ordered_values)


def parse_thresholds_env(var_name):
    raw_value = os.getenv(var_name, "").strip()
    if raw_value == "":
        return None
    return unique_float_sequence(
        float(item.strip())
        for item in raw_value.split(",")
        if item.strip() != ""
    )


def resolve_neglectable_rmse_thresholds(args):
    if args.neglectable_rmse_thresholds:
        return unique_float_sequence(args.neglectable_rmse_thresholds)

    env_thresholds = parse_thresholds_env("NEGLECTABLE_RMSE_THRESHOLDS")
    if env_thresholds:
        return env_thresholds

    if args.neglectable_rmse_threshold != DEFAULT_NEGLECTABLE_RMSE_THRESHOLDS[0]:
        return unique_float_sequence([args.neglectable_rmse_threshold])

    return DEFAULT_NEGLECTABLE_RMSE_THRESHOLDS


def parse_int_sequence_env(var_name):
    raw_value = os.getenv(var_name, "").strip()
    if raw_value == "":
        return None
    return tuple(
        int(item.strip())
        for item in raw_value.split(",")
        if item.strip() != ""
    )


def resolve_conv_filters(args):
    if args.conv_filters:
        return tuple(int(filters) for filters in args.conv_filters)

    env_filters = parse_int_sequence_env("CNN_CONV_FILTERS")
    if env_filters:
        return env_filters

    return DEFAULT_CONV_FILTERS


def parse_args():
    parser = argparse.ArgumentParser(description="Train ECM classifier and evaluate neglectable misclassifications.")
    parser.add_argument(
        "--neglectable-rmse-threshold",
        type=float,
        default=float(os.getenv("NEGLECTABLE_RMSE_THRESHOLD", "1e-3")),
        help="Pointwise relative RMSE threshold for counting different ECM reconstructions as neglectable.",
    )
    parser.add_argument(
        "--neglectable-fit-trials",
        type=int,
        default=int(os.getenv("NEGLECTABLE_FIT_TRIALS", "3")),
        help="Number of ECM fitting trials per true/predicted ECM model.",
    )
    parser.add_argument(
        "--neglectable-fit-method",
        type=str,
        default=os.getenv("NEGLECTABLE_FIT_METHOD", "LSQ"),
        choices=["LSQ", "LBFGS", "Powell"],
        help="ECM fitting optimizer used for reconstructing EIS.",
    )
    parser.add_argument(
        "--skip-neglectable-analysis",
        action="store_true",
        default=str_to_bool(os.getenv("SKIP_NEGLECTABLE_ANALYSIS", "0")),
        help="Skip ECM fitting/reconstruction and keep raw classification metrics.",
    )
    parser.add_argument(
        "--neglectable-freq-file",
        type=str,
        default=os.getenv("NEGLECTABLE_FREQ_FILE", "angular_freq.csv"),
        help="Angular-frequency CSV used for ECM reconstruction.",
    )
    parser.add_argument(
        "--neglectable-freq-min-hz",
        type=float,
        default=float(os.getenv("NEGLECTABLE_FREQ_MIN_HZ", "0.1")),
        help="Fallback minimum frequency if the frequency file is unavailable.",
    )
    parser.add_argument(
        "--neglectable-freq-max-hz",
        type=float,
        default=float(os.getenv("NEGLECTABLE_FREQ_MAX_HZ", "10000.0")),
        help="Fallback maximum frequency if the frequency file is unavailable.",
    )
    parser.add_argument(
        "--save-neglectable-plots",
        action="store_true",
        default=str_to_bool(os.getenv("SAVE_NEGLECTABLE_PLOTS", "0")),
        help="Save per-sample reconstructed EIS plots in addition to CSV outputs.",
    )
    parser.add_argument(
        "--neglectable-rmse-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Pointwise relative RMSE thresholds for running multiple neglectable-misclassification analyses. Defaults to 1e-3 and 1e-2.",
    )
    parser.add_argument(
        "--input-signal-layout",
        type=str,
        default=os.getenv("EIS_INPUT_SIGNAL_LAYOUT", DEFAULT_INPUT_SIGNAL_LAYOUT),
        choices=sorted(FEATURE_NAMES_BY_LAYOUT),
        help="Layout of the 3 raw EIS input channels.",
    )
    parser.add_argument(
        "--drop-rate",
        type=float,
        default=float(os.getenv("CNN_DROP_RATE", str(DEFAULT_DROP_RATE))),
        help="Dropout rate for the 2D-CNN classifier head.",
    )
    parser.add_argument(
        "--conv-filters",
        type=int,
        nargs="+",
        default=None,
        help="Conv2D filters for each CNN block, for example: --conv-filters 32 64 128",
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=int(os.getenv("CNN_DENSE_UNITS", str(DEFAULT_DENSE_UNITS))),
        help="Dense units after the convolution blocks.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("CNN_LEARNING_RATE", str(DEFAULT_LEARNING_RATE))),
        help="Adam learning rate for the 2D-CNN classifier.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("CNN_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("CNN_EPOCHS", str(DEFAULT_EPOCHS))),
        help="Maximum number of training epochs.",
    )
    args, _ = parser.parse_known_args()
    return args


args = parse_args()
drop_rate = float(args.drop_rate)
conv_filters = resolve_conv_filters(args)
dense_units = int(args.dense_units)
learning_rate = float(args.learning_rate)
batch_size = int(args.batch_size)
epochs = int(args.epochs)
neglectable_rmse_thresholds = resolve_neglectable_rmse_thresholds(args)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("2D-CNN drop rate:", drop_rate)
print("2D-CNN conv filters:", conv_filters)
print("2D-CNN dense units:", dense_units)
print("2D-CNN learning rate:", learning_rate)
print("2D-CNN batch size:", batch_size)
print("2D-CNN epochs:", epochs)
print("Neglectable pointwise relative RMSE thresholds:", neglectable_rmse_thresholds)
print("Input signal layout:", args.input_signal_layout)


##### Load EIS data-set #####

filename="xy_data_16k_6circuit_v2.mat"

x=scipy.io.loadmat(filename)["x_data"]
y=scipy.io.loadmat(filename)["y_data"]
y=np.squeeze(y)
x=np.swapaxes(x, 1, 2).astype(np.float32)
y=tf.keras.utils.to_categorical(y)

x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)
if x_train_raw.shape[2] == 3:
    feature_names = FEATURE_NAMES_BY_LAYOUT[args.input_signal_layout]
else:
    feature_names = tuple(f"feature_{idx + 1:02d}" for idx in range(x_train_raw.shape[2]))

feature_mean = x_train_raw.mean(axis=(0, 1), keepdims=True)
feature_std = x_train_raw.std(axis=(0, 1), keepdims=True)
feature_std = np.where(feature_std < 1e-12, 1.0, feature_std)

x_train = ((x_train_raw - feature_mean) / feature_std).astype(np.float32)[..., np.newaxis]
x_test = ((x_test_raw - feature_mean) / feature_std).astype(np.float32)[..., np.newaxis]

print("Using x_data as-is; no layout detection or feature conversion is applied.")
print("2D-CNN input shape:", x_train.shape[1:])
print("Raw sample shape before channel expansion:", x_train_raw.shape[1:])

##### Model #####
# 2D-CNN classifier
# default drop rate 0.35
# default conv filters (32, 64, 128)
# default dense units 128
# default learning rate 1e-3
# default batch size 256
# default epochs 200

def make_drop_rate_tag(dropout_rate):
    drop_text = np.format_float_positional(float(dropout_rate), trim="-")
    return drop_text.replace(".", "")


def make_conv_filters_tag(filters):
    return "x".join(str(filters_per_block) for filters_per_block in filters)


def make_learning_rate_tag(value):
    learning_rate_text = np.format_float_positional(float(value), trim="-")
    return learning_rate_text.replace(".", "p")


Experiment_name=(
    "lab6basicECM_2DCNN_Classification_"
    + f"f{make_conv_filters_tag(conv_filters)}_"
    + f"d{dense_units}_"
    + f"drop{make_drop_rate_tag(drop_rate)}_"
    + f"lr{make_learning_rate_tag(learning_rate)}_batch"
)
fn_tmp=filename.split("xy_data_",1)[1].split(".",1)[0]
Experiment_path="EIS_"+fn_tmp+"_model_"+Experiment_name


#build model
initializer = tf.keras.initializers.HeNormal()

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)
    features = input_layer
    block_dropout_rate = min(drop_rate * 0.5, 0.2)

    for block_idx, filters in enumerate(conv_filters):
        primary_kernel = (5, 3) if block_idx == 0 else (3, 1)
        features = keras.layers.Conv2D(
            filters,
            primary_kernel,
            padding="same",
            use_bias=False,
            kernel_initializer=initializer,
        )(features)
        features = keras.layers.BatchNormalization()(features)
        features = keras.layers.Activation("relu")(features)

        features = keras.layers.Conv2D(
            filters,
            (3, 1),
            padding="same",
            use_bias=False,
            kernel_initializer=initializer,
        )(features)
        features = keras.layers.BatchNormalization()(features)
        features = keras.layers.Activation("relu")(features)
        features = keras.layers.MaxPooling2D(pool_size=(2, 1))(features)

        if block_idx < len(conv_filters) - 1:
            features = keras.layers.Dropout(block_dropout_rate)(features)

    features = keras.layers.GlobalAveragePooling2D()(features)
    features = keras.layers.Dense(
        dense_units,
        use_bias=False,
        kernel_initializer=initializer,
    )(features)
    features = keras.layers.BatchNormalization()(features)
    features = keras.layers.Activation("relu")(features)
    features = keras.layers.Dropout(drop_rate)(features)

    output_layer = keras.layers.Dense(len(LABEL_NAMES), activation="softmax")(features)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(input_shape=x_train.shape[1:])
#Model Summarize
model.summary()
#keras.utils.plot_model(model, show_shapes=True)

##### Training #####
Experiment_path=Experiment_path+"_"+str(batch_size) 
print(Experiment_path)
os.makedirs(Experiment_path, exist_ok=True)

feature_stats_df = pd.DataFrame(
    {
        "feature": feature_names,
        "train_mean": feature_mean.reshape(-1),
        "train_std": feature_std.reshape(-1),
    }
)
feature_stats_df.to_csv(Experiment_path+"/"+"input_feature_stats.csv", index=False)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%y_%m_%d") + "/" \
                      + Experiment_path.split("model_",1)[1]  \
                      +"_"+ filename.split("_",-1)[2] \
                      + datetime.datetime.now().strftime("_%m%d%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                          log_dir=log_dir,
                                          histogram_freq=0,
                                          profile_batch=0)

modelpath= Experiment_path \
           + "/" + "model_{epoch:02d}_{val_loss:.2f}_{val_accuracy:.2f}.h5"

callbacks =[
            keras.callbacks.ModelCheckpoint(
                modelpath, save_best_only=True, 
                monitor="val_loss",mode="min"
                ),
            
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, verbose=0,
                mode='min', min_lr=0.000001
                ),
            
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=30,
                restore_best_weights=True,
                verbose=1,
            ),
            
            #TqdmCallback(verbose=0),
            tensorboard_callback,         
           ]

model.compile(
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss="categorical_crossentropy",
              metrics=["accuracy"],
             )

history = model.fit(
          x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_data=(x_test,y_test),
          verbose=2,
                   )

df_temp = pd.DataFrame(list(zip(history.history["accuracy"],history.history["val_accuracy"],history.history["loss"],history.history["val_loss"])),
                            columns = ["accuracy","val_accuracy","loss","val_loss"])
print(Experiment_path)

##### Evaluation #####

model_to_load = model

def save_accuracy_plot(history, save_path, accuracy_with_neglectable=None):
    fig = plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    if accuracy_with_neglectable is not None:
        plt.axhline(
            y=accuracy_with_neglectable,
            linestyle="--",
            color="tab:green",
            label="validation + neglectable misclassification",
        )
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close(fig)


def save_loss_plot(history, save_path):
    fig = plt.figure()
    plt.plot(history.history["loss"][1:-1])
    plt.plot(history.history["val_loss"][1:-1])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(save_path)
    plt.close(fig)


def save_confusion_matrix(confusion_matrix_data, save_path, title, label_names):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_data,
        display_labels=label_names,
    )
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, ax = plt.subplots(figsize=(600 * px, 600 * px), dpi=300)
    disp.plot(cmap="summer", ax=ax)
    plt.title(title)
    plt.savefig(save_path)
    plt.close(fig)


def make_adjusted_confusion_matrix(confusion_matrix_data, neglectable_confusion_matrix):
    adjusted_matrix = confusion_matrix_data.copy()
    for true_idx in range(adjusted_matrix.shape[0]):
        for predicted_idx in range(adjusted_matrix.shape[1]):
            if true_idx == predicted_idx:
                continue
            neglectable_count = min(
                int(neglectable_confusion_matrix[true_idx, predicted_idx]),
                int(adjusted_matrix[true_idx, predicted_idx]),
            )
            adjusted_matrix[true_idx, predicted_idx] -= neglectable_count
            adjusted_matrix[true_idx, true_idx] += neglectable_count
    return adjusted_matrix


def save_confusion_matrix_with_neglectable(
    confusion_matrix_data,
    neglectable_confusion_matrix,
    save_path,
    title,
    label_names,
):
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, ax = plt.subplots(figsize=(700 * px, 700 * px), dpi=300)
    image = ax.imshow(confusion_matrix_data, interpolation="nearest", cmap="summer")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(label_names))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = confusion_matrix_data.max() / 2.0 if confusion_matrix_data.size else 0
    for true_idx in range(confusion_matrix_data.shape[0]):
        for predicted_idx in range(confusion_matrix_data.shape[1]):
            count = int(confusion_matrix_data[true_idx, predicted_idx])
            neglectable_count = int(neglectable_confusion_matrix[true_idx, predicted_idx])
            annotation = str(count)
            if true_idx != predicted_idx and neglectable_count > 0:
                annotation = f"{count}\nNeg:{neglectable_count}"
            ax.text(
                predicted_idx,
                true_idx,
                annotation,
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
                fontsize=6,
            )

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def format_rmse_threshold(rmse_threshold):
    return np.format_float_positional(float(rmse_threshold), trim="-")


def make_rmse_threshold_tag(rmse_threshold):
    threshold_text = format_rmse_threshold(rmse_threshold)
    return f"rmse_{threshold_text.replace('.', 'p').replace('-', 'm')}"


def build_neglectable_confusion_matrix(summary_df, reference_shape):
    neglectable_confusion_matrix = np.zeros(reference_shape, dtype=int)
    neglectable_count = 0

    if "is_neglectable_misclassification" not in summary_df.columns:
        return neglectable_confusion_matrix, neglectable_count

    neglectable_rows = summary_df[
        summary_df["is_neglectable_misclassification"].fillna(False).astype(bool)
    ]
    neglectable_count = int(len(neglectable_rows))
    for _, row in neglectable_rows.iterrows():
        true_idx = int(row["true_label_index"])
        predicted_idx = int(row["predicted_label_index"])
        neglectable_confusion_matrix[true_idx, predicted_idx] += 1

    return neglectable_confusion_matrix, neglectable_count


save_loss_plot(history, Experiment_path+"/"+"loss.png")

#predict
predict_model = model

x_t= x_test
y_t= y_test
x_t_raw = x_test_raw
# x_t= x_train
# y_t= y_train

m_ev=predict_model.evaluate(x_t,y_t)
y_pred=predict_model.predict(x_t)
label_names = LABEL_NAMES
y_pred_class = np.argmax(y_pred, axis=1).astype(int)
y_test_class = np.argmax(y_t, axis=1).astype(int)
test_list2=y_pred_class
test_list1=y_test_class

cm=confusion_matrix(test_list1,test_list2, labels=np.arange(len(label_names)))
raw_accuracy = accuracy_score(test_list1, test_list2)
raw_loss = float(m_ev[0])
raw_title = "Accuracy :"+str(raw_accuracy*100)+"%"+"\n"+"Loss :"+str(raw_loss)
save_confusion_matrix(cm, Experiment_path+"/"+"CMatrix.png", raw_title, label_names)

# Export all misclassified test-set EIS samples with label/probability details.
misclassified_mask = y_test_class != y_pred_class
misclassified_indices = np.where(misclassified_mask)[0]
misclassified_df = pd.DataFrame()
misclassified_original_signal = np.empty((0, x_t_raw.shape[1], min(3, x_t_raw.shape[2])), dtype=np.float32)

if len(misclassified_indices) > 0:
    misclassified_x_raw = x_t_raw[misclassified_indices]
    misclassified_true = y_t[misclassified_indices]
    misclassified_pred = y_pred[misclassified_indices]
    misclassified_true_class = y_test_class[misclassified_indices].astype(int)
    misclassified_pred_class = y_pred_class[misclassified_indices].astype(int)

    misclassified_df = pd.DataFrame({
        "test_index": misclassified_indices,
        "true_label_index": misclassified_true_class,
        "true_label_name": label_names[misclassified_true_class],
        "predicted_label_index": misclassified_pred_class,
        "predicted_label_name": label_names[misclassified_pred_class],
        "predicted_probability_of_true_label": misclassified_pred[
            np.arange(len(misclassified_indices)),
            misclassified_true_class,
        ],
        "predicted_probability_of_predicted_label": misclassified_pred[
            np.arange(len(misclassified_indices)),
            misclassified_pred_class,
        ],
    })

    for class_idx, class_name in enumerate(label_names):
        misclassified_df[f"true_onehot_{class_name}"] = misclassified_true[:, class_idx]
        misclassified_df[f"pred_softmax_{class_name}"] = misclassified_pred[:, class_idx]

    misclassified_original_signal = misclassified_x_raw[:, :, :3].astype(np.float32)
    for point_idx in range(misclassified_x_raw.shape[1]):
        point_num = point_idx + 1
        for feature_idx, feature_name in enumerate(feature_names):
            misclassified_df[f"{feature_name}_pt_{point_num:02d}"] = misclassified_x_raw[
                :, point_idx, feature_idx
            ]

    misclassified_df.to_csv(Experiment_path+"/"+"misclassified_EIS.csv", index=False)
    print("Saved misclassified EIS samples:", len(misclassified_indices))
    print("Misclassified EIS file:", Experiment_path+"/"+"misclassified_EIS.csv")
else:
    print("No misclassified EIS samples found in the evaluation split.")

save_accuracy_plot(history, Experiment_path+"/"+"accuracy.png")
df_temp.to_csv(Experiment_path+"/"+"trainig_curve.csv")

raw_metrics_df = pd.DataFrame([{
    "model_type": "2D-CNN",
    "input_layout": f"{x_train_raw.shape[1]}x{x_train_raw.shape[2]}x1 raw x_data as-is",
    "input_features": ",".join(feature_names),
    "conv_filters": "x".join(str(filters) for filters in conv_filters),
    "dense_units": int(dense_units),
    "drop_rate": float(drop_rate),
    "learning_rate": float(learning_rate),
    "batch_size": int(batch_size),
    "epochs": int(epochs),
    "loss": raw_loss,
    "accuracy": raw_accuracy,
    "misclassified_count": int(len(misclassified_indices)),
}])
raw_metrics_df.to_csv(Experiment_path+"/"+"classification_metrics.csv", index=False)

threshold_metrics_rows = []
frequency_grid_error = ""
angular_freq = None
freq_hz = None

if not args.skip_neglectable_analysis and len(misclassified_indices) > 0:
    if x_t_raw.shape[2] < 3:
        frequency_grid_error = "Neglectable analysis requires at least 3 raw features per point."
        print("[WARN]", frequency_grid_error)
    else:
        print(
            "[INFO] Neglectable analysis will use the first 3 raw input features "
            f"as {FEATURE_NAMES_BY_LAYOUT[args.input_signal_layout]}."
        )
        try:
            angular_freq, freq_hz = load_frequency_grid(
                x_t_raw.shape[1],
                freq_file=args.neglectable_freq_file,
                freq_min_hz=args.neglectable_freq_min_hz,
                freq_max_hz=args.neglectable_freq_max_hz,
            )
        except Exception as exc:
            frequency_grid_error = str(exc)
            print("[WARN] Failed to load neglectable-analysis frequency grid:", frequency_grid_error)

if args.skip_neglectable_analysis:
    print("Skipped neglectable misclassification analysis.")
else:
    for rmse_threshold in neglectable_rmse_thresholds:
        threshold_text = format_rmse_threshold(rmse_threshold)
        threshold_tag = make_rmse_threshold_tag(rmse_threshold)
        threshold_dir = os.path.join(Experiment_path, f"neglectable_{threshold_tag}")
        os.makedirs(threshold_dir, exist_ok=True)

        threshold_summary_df = pd.DataFrame()
        threshold_summary_file = os.path.join(
            threshold_dir,
            f"neglectable_misclassification_summary_{threshold_tag}.csv",
        )
        threshold_confusion_matrix = np.zeros_like(cm, dtype=int)
        threshold_count = 0
        threshold_error = frequency_grid_error
        threshold_accuracy_with_neglectable = raw_accuracy
        threshold_accuracy_plot = os.path.join(threshold_dir, f"accuracy_{threshold_tag}.png")
        threshold_adjusted_cm_path = ""
        threshold_neglectable_cm_path = ""

        if threshold_error == "" and len(misclassified_indices) > 0:
            try:
                threshold_summary_df = analyze_misclassified_samples(
                    misclassified_df=misclassified_df,
                    original_signals=misclassified_original_signal,
                    angular_freq=angular_freq,
                    freq_hz=freq_hz,
                    output_dir=threshold_dir,
                    rmse_threshold=rmse_threshold,
                    trial_num=args.neglectable_fit_trials,
                    method=args.neglectable_fit_method,
                    save_plots=args.save_neglectable_plots,
                    signal_layout=args.input_signal_layout,
                )
                threshold_summary_df.to_csv(threshold_summary_file, index=False)
                threshold_confusion_matrix, threshold_count = build_neglectable_confusion_matrix(
                    threshold_summary_df,
                    cm.shape,
                )
                print(
                    "Neglectable misclassifications "
                    + f"(pointwise relative RMSE threshold {threshold_text}): {threshold_count}"
                )
                print("Neglectable summary file:", threshold_summary_file)
            except Exception as exc:
                threshold_error = str(exc)
                print(
                    "[WARN] Neglectable misclassification analysis failed "
                    + f"for pointwise relative RMSE threshold {threshold_text}: {threshold_error}"
                )
        else:
            threshold_summary_df.to_csv(threshold_summary_file, index=False)

        if not os.path.exists(threshold_summary_file):
            threshold_summary_df.to_csv(threshold_summary_file, index=False)

        if threshold_error == "":
            threshold_accuracy_with_neglectable = (
                int(np.trace(cm)) + threshold_count
            ) / len(test_list1)
            adjusted_cm = make_adjusted_confusion_matrix(cm, threshold_confusion_matrix)
            adjusted_title = (
                "Accuracy :"+str(raw_accuracy*100)+"%"
                +"\n"+"Accuracy + neglectable :"+str(threshold_accuracy_with_neglectable*100)+"%"
                +"\n"+"Neglectable relative RMSE threshold :"+threshold_text
            )
            threshold_neglectable_cm_path = os.path.join(
                threshold_dir,
                f"CMatrix_with_neglectable_{threshold_tag}.png",
            )
            threshold_adjusted_cm_path = os.path.join(
                threshold_dir,
                f"CMatrix_neglectable_adjusted_{threshold_tag}.png",
            )
            save_confusion_matrix_with_neglectable(
                cm,
                threshold_confusion_matrix,
                threshold_neglectable_cm_path,
                adjusted_title,
                label_names,
            )
            save_confusion_matrix(
                adjusted_cm,
                threshold_adjusted_cm_path,
                adjusted_title,
                label_names,
            )
            save_accuracy_plot(
                history,
                threshold_accuracy_plot,
                threshold_accuracy_with_neglectable,
            )
        else:
            save_accuracy_plot(history, threshold_accuracy_plot)

        threshold_training_curve = df_temp.copy()
        threshold_training_curve["val_accuracy_with_neglectable"] = (
            threshold_accuracy_with_neglectable if threshold_error == "" else np.nan
        )
        threshold_training_curve["neglectable_misclassification_count"] = threshold_count
        threshold_training_curve["neglectable_rmse_threshold"] = rmse_threshold
        threshold_training_curve["neglectable_rmse_threshold_type"] = "pointwise_relative"
        threshold_training_curve["neglectable_analysis_error"] = threshold_error
        threshold_training_curve_path = os.path.join(
            threshold_dir,
            f"trainig_curve_{threshold_tag}.csv",
        )
        threshold_training_curve.to_csv(threshold_training_curve_path)

        threshold_metrics_row = {
            "loss": raw_loss,
            "accuracy": raw_accuracy,
            "misclassified_count": int(len(misclassified_indices)),
            "neglectable_misclassification_count": int(threshold_count),
            "accuracy_with_neglectable": (
                threshold_accuracy_with_neglectable if threshold_error == "" else np.nan
            ),
            "neglectable_rmse_threshold": rmse_threshold,
            "neglectable_rmse_threshold_type": "pointwise_relative",
            "neglectable_fit_trials": args.neglectable_fit_trials,
            "neglectable_fit_method": args.neglectable_fit_method,
            "neglectable_analysis_skipped": False,
            "neglectable_analysis_error": threshold_error,
            "output_dir": threshold_dir,
            "summary_csv": threshold_summary_file,
            "training_curve_csv": threshold_training_curve_path,
            "accuracy_plot_png": threshold_accuracy_plot,
            "confusion_matrix_with_neglectable_png": threshold_neglectable_cm_path,
            "adjusted_confusion_matrix_png": threshold_adjusted_cm_path,
        }
        threshold_metrics_rows.append(threshold_metrics_row)

        threshold_metrics_df = pd.DataFrame([threshold_metrics_row])
        threshold_metrics_df.to_csv(
            os.path.join(
                threshold_dir,
                f"classification_metrics_with_neglectable_{threshold_tag}.csv",
            ),
            index=False,
        )

if threshold_metrics_rows:
    pd.DataFrame(threshold_metrics_rows).to_csv(
        Experiment_path+"/"+"classification_metrics_with_neglectable.csv",
        index=False,
    )

c1,c2,c3,c4,c5,c6=0,0,0,0,0,0
for idx in range(len(test_list1)):
    if test_list1[idx]==0:c1=c1+1
    if test_list1[idx]==1:c2=c2+1    
    if test_list1[idx]==2:c3=c3+1  
    if test_list1[idx]==3:c4=c4+1
    if test_list1[idx]==4:c5=c5+1 
    if test_list1[idx]==5:c6=c6+1   
print(c1,c2,c3,c4,c5,c6)
