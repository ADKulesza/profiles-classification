import argparse
import logging
import os

import numpy as np
import pandas as pd

from scipy.stats import scoreatpercentile


C_LOGGER_NAME = "get_holdout"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)

AREAS_TYPES = [
    "Kon",
    "Eu3",
    "Eu2",
    "Eu1",
    "Dys",
    "Agr"
]


def read_data(paths):
    logger.info("Loading data...")

    # Loading array with profiles
    pred_y = np.load(paths.pred_y)
    logger.info("%s", paths.pred_y)

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    logger.info("Loading data... Done!")

    return pred_y, profiles_df


def process(paths):
    pred_y, profiles_df = read_data(paths)

    for area_type in AREAS_TYPES:
        _df = profiles_df[profiles_df.type == area_type]
        type_idx = _df.index_in_npy_array.array
        classes_in_type = pd.unique(_df.idx_in_model)

        _pred = pred_y[:,classes_in_type]
        _pred = _pred[type_idx,:]

        val_median = np.median(_pred)
        val_q1 = scoreatpercentile(_pred, 25)
        val_q3 = scoreatpercentile(_pred, 75)

        logger.info("AREA %s", area_type)
        logger.info("AAA %s", val_median)
        logger.info("q1 %s", val_q1)
        logger.info("q3 %s", val_q3)

def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--validation-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-y",
        "--pred-y",
        required=True,
        dest="pred_y",
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
    input_options = parse_args()
    process(input_options)
