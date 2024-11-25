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
    confusion_matrix = np.load(paths.cmat)
    logger.info("%s", paths.cmat)

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    logger.info("Loading data... Done!")

    return confusion_matrix, profiles_df


def process(paths):
    cmat, df = read_data(paths)

    df['correct_classification'] = df['area'] == df['pred_area']
    df['error_within_same_type'] = (~df['correct_classification']) & (df['type_id'] == df['pred_type_id'])
    within_same_type = df[df['error_within_same_type']].shape[0]

    df['error_across_types'] = (~df['correct_classification']) & (df['type_id'] != df['pred_type_id'])
    across_types = df[df["error_across_types"]].shape[0]

    logger.info("Errors within same type: %s", within_same_type)
    logger.info("Errors across types: %s", across_types)

    total_errors = df[~df['correct_classification']].shape[0]

    # Count the total number of profiles per type_id
    type_distribution = df['type_id'].value_counts()
    logger.info("type_distribution: %s", type_distribution)

    # Profiles number
    total_instances = df.shape[0]
    logger.info("total_instances: %s", total_instances)

    type_proportions = type_distribution / total_instances
    logger.info("type_proportions: %s", type_proportions)

    error_proportions = type_proportions * total_errors
    logger.info("error_proportions: %s", error_proportions)



    # df_dict = {
    #     "area_type_1": [],
    #     "area_type_2": [],
    #     "InType_errors_ratio": [],
    #     "NoType_errors_ratio": [],
    # }
    # _temp = {}
    #
    # all_clasess = pd.unique(profiles_df.idx_in_model)
    # for area_type_1 in AREAS_TYPES:
    #     for area_type_2 in AREAS_TYPES:
    #         # if area_type_1 == area_type_2:
    #         #     continue
    #
    #         _df_1 = profiles_df[profiles_df.type == area_type_1]
    #         _df_2 = profiles_df[profiles_df.type == area_type_2]
    #         classes_in_type_1 = pd.unique(_df_1.idx_in_model)
    #         classes_in_type_2 = pd.unique(_df_2.idx_in_model)
    #
    #         df_dict["area_type_1"].append(area_type_1)
    #         df_dict["area_type_2"].append(area_type_2)
    #
    #         # InType
    #         _cmat = cmat.copy()
    #         # in_type_correct_sum = np.trace(_cmat)
    #         np.fill_diagonal(_cmat, 0)
    #         _cmat = _cmat[classes_in_type_1, :]
    #         _cmat = _cmat[:, classes_in_type_1]
    #
    #         in_type_errors_sum = np.sum(_cmat)
    #
    #         _cmat = cmat.copy()
    #         np.fill_diagonal(_cmat, 0)
    #         in_type_whole_errors_sum = np.sum(_cmat[classes_in_type_1, :])
    #
    #         in_type_errors_ratio = in_type_errors_sum / in_type_whole_errors_sum
    #
    #         # NoType
    #         _cmat = cmat.copy()
    #         # no_type_correct_sum = np.trace(_cmat)
    #         np.fill_diagonal(_cmat, 0)
    #         np.fill_diagonal(_cmat, 0)
    #         _cmat = _cmat[classes_in_type_1, :]
    #         _cmat = _cmat[:, classes_in_type_2]
    #
    #         no_type_errors_sum = np.sum(_cmat)
    #
    #         _cmat = cmat.copy()
    #         np.fill_diagonal(_cmat, 0)
    #         no_type_whole_errors_sum = np.sum(_cmat[classes_in_type_2, :])
    #
    #         no_type_errors_ratio = no_type_errors_sum / no_type_whole_errors_sum
    #
    #
    #         # logger.info("InType errors sum: %s", in_type_errors_sum)
    #         # logger.info("InType errors: %s", in_type_errors_sum)
    #         logger.info("InType errors percentage: %s", in_type_errors_ratio)
    #
    #         # logger.info("NoType errors sum: %s", no_type_errors_sum)
    #         # logger.info("NoType errors: %s", no_type_errors_sum)
    #         logger.info("NoType errors percentage: %s", no_type_errors_ratio)
    #
    #         df_dict["InType_errors_ratio"].append(in_type_errors_ratio)
    #
    #         df_dict["NoType_errors_ratio"].append(no_type_errors_ratio)
    #
    # df = pd.DataFrame(df_dict)
    #
    # df_path = os.path.join(paths.output, "errors.csv")
    # df.to_csv(df_path)


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
        "--cmat",
        required=True,
        dest="cmat",
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
