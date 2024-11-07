import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from cnn_models.flatten_model import flatten_conv_model
from cnn_models.multi_branch_binary_model import multi_branch_binary_model
from cnn_models.multi_branch_model import multi_branch_model
from cnn_models.simple_model import simple_model
from dataset_configuration import DatasetConfiguration
from step_05_model_visualization import summary_plot
from step_05_generator_all_vs_all import BatchNormAugmentationSequence
from step_05_generator_one_vs_all import OneVsAllAugmentationSequence

C_LOGGER_NAME = "training_model"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True

MODELS_LIST = {
    "simple_model": simple_model,
    "flatten_conv_model": flatten_conv_model,
    "multi_branch_model": multi_branch_model,
    "multi_branch_binary_model": multi_branch_binary_model,
}


def read_data(paths):
    logger.info("Loading data...")

    # Loading array with profiles
    profiles = np.load(paths.profiles_npy)
    logger.info("%s", paths.profiles_npy)

    labels = np.load(paths.labels_npy)
    logger.info("%s", paths.labels_npy)

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    logger.info("Loading data... Done!")

    return profiles, labels, profiles_df


def train_model(
    paths,
    epochs,
    patience,
    batch_size=32,
    add_noise=False,
    noise_ratio=0.2,
    add_artifacts=False,
    artifacts_ratio=0.2,
    learning_rate=0.001,
):
    model_name = paths.model_name
    model_method = MODELS_LIST[model_name]

    x_profiles, y_labels, df = read_data(paths)
    kflods_list = [col for col in df.columns if col.startswith("set_")]

    if len(kflods_list) == 0:
        raise ValueError(f"set_list = 0. No dataset available!")

    df_path = os.path.join(paths.output, paths.model_info)
    model_info = {"set": [], "path": []}
    for i_model, dset in enumerate(kflods_list):

        logger.info("Dataset... %s", dset)

        filename = os.path.join(paths.output, f"{model_name}_{dset}")
        if os.path.exists(filename):
            logger.warning("Model... %s EXIST!", filename)
            continue

        model_info["set"].append(dset)
        model_info["path"].append(filename)

        logger.info("Model... %s", filename)
        if paths.one_vs_all:
            fold_df = df[["idx_in_model", "index_in_npy_array", "prob", dset]]
        else:
            fold_df = df[["idx_in_model", "index_in_npy_array", dset]]
        train_df = fold_df[fold_df[dset] == "train"]
        train_df["new_index"] = np.arange(train_df.shape[0], dtype=int)
        valid_df = fold_df[fold_df[dset] == "valid"]
        valid_df["new_index"] = np.arange(valid_df.shape[0], dtype=int)

        train_idx = train_df.index_in_npy_array.astype(int)
        valid_idx = valid_df.index_in_npy_array.astype(int)

        x_train = x_profiles[train_idx]
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        y_train = y_labels[train_idx]

        x_valid = x_profiles[valid_idx]
        x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))
        y_valid = y_labels[valid_idx]

        if paths.one_vs_all:
            sequence = OneVsAllAugmentationSequence
        else:
            sequence = BatchNormAugmentationSequence

        model = model_method(y_train.shape[1], x_train.shape[1], learning_rate)
        model.summary(line_length=None, positions=None, print_fn=None)

        model_history = model.fit(
            sequence(
                x_train,
                y_train,
                batch_size,
                train_df,
                add_noise,
                noise_ratio,
                add_artifacts,
                artifacts_ratio,
            ),
            epochs=epochs,
            validation_data=sequence(
                x_valid,
                y_valid,
                batch_size,
                valid_df,
                add_noise,
                noise_ratio,
                add_artifacts,
                artifacts_ratio,
            ),
            callbacks=[EarlyStopping(monitor="val_accuracy", patience=patience)],
        )

        plot_path = os.path.join(paths.output, f"{model_name}_{dset}_summary.png")
        summary_plot(model_history, plot_path, i_model)

        model.save(filename)

        models_df = pd.DataFrame(model_info)
        logger.info("Info about models saved in... %s", df_path)
        models_df.to_csv(df_path)
        time.sleep(10)
        del model_history
        time.sleep(15)


def parse_args():
    """
    Provides command-line interface.
    """
    parser = argparse.ArgumentParser(
        description=train_model.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config-fname",
        required=True,
        dest="config_fname",
        type=str,
        metavar="FILENAME",
        help="Path to file with configuration",
    )

    parser.add_argument(
        "-t",
        "--model-name",
        required=True,
        dest="model_name",
        type=str,
        metavar="FILENAME",
        help="Model name",
    )

    parser.add_argument(
        "-x",
        "--profiles",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-y",
        "--labels",
        required=True,
        dest="labels_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),
    #
    parser.add_argument(
        "-d",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-n",
        "--one-vs-all",
        required=False,
        action="store_true",
        dest="one_vs_all",
        help="Path to output directory",
    )

    parser.add_argument(
        "-m",
        "--model-info",
        required=True,
        dest="model_info",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    # tf.debugging.set_log_device_placement(True)

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = (
        0.75  # 0.33  # 0.6 sometimes works better for folks
    )
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    input_options = parse_args()
    config = DatasetConfiguration(input_options.config_fname)
    np.random.seed(config("random_seed"))

    logger.info(
        "!!!! Num GPUs Available: %s", len(tf.config.list_physical_devices("GPU"))
    )
    logger.info(
        "!!!!! Num CPUs Available: %s", len(tf.config.list_physical_devices("CPU"))
    )

    train_model(
        input_options,
        batch_size=config("batch_size"),
        epochs=config("number_of_epochs"),
        patience=config("patience"),
        add_noise=config("add_noise"),
        noise_ratio=config("noise_ratio"),
        add_artifacts=config("add_artifacts"),
        artifacts_ratio=config("artifacts_ratio"),
        learning_rate=config("learning_rate"),
    )
