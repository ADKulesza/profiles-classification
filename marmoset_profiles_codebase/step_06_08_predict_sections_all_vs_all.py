import argparse
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from read_json import read_json

C_LOGGER_NAME = "get_holdout"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    # Loading array with profiles
    profiles = np.load(paths.profiles_npy)
    logger.info("%s", paths.profiles_npy)

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    labels_processed = pd.read_csv(paths.labels_processed)
    logger.info("%s", paths.labels_processed)

    label_names = pd.read_csv(paths.label_names)
    logger.info("%s", paths.label_names)

    area_order = read_json(paths.area_order)
    logger.info("%s", paths.area_order)

    model = load_model(paths.model_path)
    logger.info("%s", paths.model_path)

    logger.info("Loading data... Done!")

    return profiles, profiles_df, labels_processed, label_names, area_order, model


def process(paths):
    (profiles, profiles_df,
     label_processed, label_names,
     area_order,
     model) = read_data(paths)

    x_test = profiles.reshape((profiles.shape[0], profiles.shape[1], 1))

    logger.info("\nCLASSIFING...\n")
    predictions = model.predict(x_test)
    logger.info("\nCLASSIFING... Done!\n")

    pred_path = os.path.join(paths.output, f"pred_y.npy")
    np.save(pred_path, predictions)
    logger.info("Predictions have saved in: %s", pred_path)

    del model

    predicted_classes = np.argmax(predictions, axis=1)
    predicted_confidence = np.max(predictions, axis=1)

    profiles_df[f"pred_y"] = predicted_classes
    profiles_df[f"pred_confidence"] = predicted_confidence

    profiles_df = profiles_df.loc[:, ~profiles_df.columns.str.contains("^Unnamed")]

    label_processed = pd.merge(label_processed, label_names, on="area_id", how="left")

    profiles_df["pred_area"] = np.nan
    profiles_df["pred_area_id"] = np.nan
    profiles_df["pred_color_r"] = np.nan
    profiles_df["pred_color_g"] = np.nan
    profiles_df["pred_color_b"] = np.nan
    for index, row in label_processed.iterrows():
        _area = row.area
        _area_id = row.area_id
        _area_idx = row.idx_in_model
        _r = row.color_r
        _g = row.color_g
        _b = row.color_b

        profiles_df.loc[profiles_df.pred_y == _area_idx, "pred_area"] = _area
        profiles_df.loc[profiles_df.pred_y == _area_idx, "pred_area_id"] = _area_id
        profiles_df.loc[profiles_df.pred_y == _area_idx, "pred_color_r"] = _r
        profiles_df.loc[profiles_df.pred_y == _area_idx, "pred_color_g"] = _g
        profiles_df.loc[profiles_df.pred_y == _area_idx, "pred_color_b"] = _b

    profiles_df["region"] = np.nan
    area_regions = area_order[2]
    for region, area_list in area_regions.items():
        profiles_df.loc[profiles_df.pred_area.isin(area_list), "pred_region"] = region

    df_path = os.path.join(paths.output, "results.csv")

    profiles_df.to_csv(df_path)
    logger.info("Dataframe has saved in: %s", pred_path)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-x",
        "--profiles",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-c",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-b",
        "--labels-processed",
        required=True,
        dest="labels_processed",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with",
    )

    parser.add_argument(
        "-n",
        "--label-names",
        required=True,
        dest="label_names",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with",
    )

    parser.add_argument(
        "-r",
        "--area-order",
        required=True,
        dest="area_order",
        type=str,
        metavar="FILENAME",
        help="Path to file with",
    )

    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        dest="model_path",
        type=str,
        metavar="FILENAME",
        help="Path to ",
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
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = (
        0.33  # 0.33  # 0.6 sometimes works better for folks
    )
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    input_options = parse_args()
    process(input_options)
