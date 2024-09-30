import argparse
import logging

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

C_LOGGER_NAME = "get_labels"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    logger.info("%s", paths.profiles_csv)
    prfoiles_df = pd.read_csv(paths.profiles_csv)

    # Loading .csv file with label assignment
    logger.info("%s", paths.labels_processed)
    labels_df = pd.read_csv(paths.labels_processed)
    labels_df.label = labels_df.area_id

    logger.info("Loading data... Done!")

    return prfoiles_df, labels_df


def save_data(raw_labels_array, one_hot_array, df, labels_df, paths):
    logger.info("Saving data...")
    np.save(paths.raw_labels, raw_labels_array)
    logger.info("%s", paths.raw_labels)

    np.save(paths.one_hot, one_hot_array)
    logger.info("%s", paths.one_hot)

    df.to_csv(paths.output_info)
    logger.info("%s", paths.output_info)

    labels_df.to_csv(paths.output_labels_processed)
    logger.info("%s", paths.output_labels_processed)

    logger.info("Saving data... Done!")


def check_one_hot(one_hot, model_labels_array):
    if np.any(model_labels_array != np.where(one_hot == 1)[1]):
        raise ValueError(f"Wrong data encoding!")


def process(paths):
    df, labels_df = read_data(paths)
    df = df[df.accept]

    labels_df["idx_in_model"] = np.arange(labels_df.shape[0], dtype=np.uint)
    label_order = dict(zip(labels_df["label"], labels_df["idx_in_model"]))

    raw_labels_array = np.array(df.label)
    model_labels_array = np.vectorize(lambda x: label_order.get(x))(raw_labels_array)
    df["idx_in_model"] = model_labels_array

    one_hot_array = to_categorical(model_labels_array, dtype=np.uint)
    check_one_hot(one_hot_array, model_labels_array)

    save_data(raw_labels_array, one_hot_array, df, labels_df, paths)


def parse_args(doc_source):
    """ """
    parser = argparse.ArgumentParser(
        description=doc_source.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-b",
        "--labels-processed",
        required=True,
        dest="labels_processed",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with ",
    )

    parser.add_argument(
        "-f",
        "--profiles-info-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to csv file with info about accepted profiles",
    )

    parser.add_argument(
        "-i",
        "--output-profiles-info-csv",
        required=True,
        dest="output_info",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-r",
        "--output-raw-labels",
        required=True,
        dest="raw_labels",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-y",
        "--output-one-hot",
        required=True,
        dest="one_hot",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-l",
        "--output-labels-processed",
        required=True,
        dest="output_labels_processed",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with ",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args(process)
    process(input_options)
