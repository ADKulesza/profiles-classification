import argparse
import logging

import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration

C_LOGGER_NAME = "confidence_trail"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    """
    Read and return data.

    """
    profiles_df = pd.read_csv(paths.profiles_csv)
    prof_arr = np.load(paths.profiles_npy)

    return profiles_df, prof_arr


def confidence_trail(config, paths):
    """

    Parameters
    ----------
    config : instance of DatasetConfiguration class
    paths : argument parser
     requires the following attributes:
      - .config_fname
      - .profiles_csv - path to csv file with whole information about profiles
      - .profiles_npy - intensity profiles in npy array

    """
    profiles_df, prof_arr = read_data(paths)

    min_confidence = config("min_confidence_level")
    max_confidence = config("max_confidence_level")
    profiles_df["accept"] = (profiles_df.confidence > min_confidence) & (
        profiles_df.confidence <= max_confidence
    )

    if config("do_exclude_zero"):
        profiles_df.loc[profiles_df.area_id == 0, "accept"] = False

    profiles_df.to_csv(paths.split_profiles_csv)

    acc_prof_df = profiles_df[profiles_df.accept]

    acc_idx = acc_prof_df.index_in_npy_array
    prof_arr = prof_arr[acc_idx]
    acc_prof_df.loc[:, "index_in_npy_array"] = np.arange(acc_prof_df.shape[0])
    acc_prof_df.loc[:, "npy_path"] = paths.accept_profiles_csv

    acc_prof_df.to_csv(paths.accept_profiles_csv)

    np.save(paths.accept_profiles, prof_arr)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=confidence_trail.__doc__,
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
        "-p",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-x",
        "--profiles-npy",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-s",
        "--split-profiles-csv",
        required=True,
        dest="split_profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-o",
        "--accept-profiles-csv",
        required=True,
        dest="accept_profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-u",
        "--output-accept-profile-npy",
        required=True,
        dest="accept_profiles",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)
    confidence_trail(data_settings, input_options)
