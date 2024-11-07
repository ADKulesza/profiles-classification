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

    logger.info("Loading data... Done!")

    return prfoiles_df


def save_data(raw_labels_array, one_hot_array, df, paths):
    logger.info("Saving data...")
    np.save(paths.raw_labels, raw_labels_array)
    logger.info("%s", paths.raw_labels)

    np.save(paths.one_hot, one_hot_array)
    logger.info("%s", paths.one_hot)

    logger.info("Saving data... Done!")


def check_one_hot(one_hot, model_labels_array):
    if np.any(model_labels_array != np.where(one_hot == 1)[1]):
        raise ValueError(f"Wrong data encoding!")


def process(paths):
    df = read_data(paths)

    raw_labels_array = np.array(df.label)

    one_hot_array = to_categorical(raw_labels_array, dtype=np.uint)
    check_one_hot(one_hot_array, raw_labels_array)
    save_data(raw_labels_array, one_hot_array, df, paths)


def parse_args(doc_source):
    """ """
    parser = argparse.ArgumentParser(
        description=doc_source.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args(process)
    process(input_options)
