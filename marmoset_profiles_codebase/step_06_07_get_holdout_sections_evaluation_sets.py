import argparse
import logging

import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration
from norm import NormProfiles


C_LOGGER_NAME = "get_sections"
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

    # Loading .csv file with label assignment
    areas_df = pd.read_csv(paths.areas_def)
    logger.info("%s", paths.areas_def)

    logger.info("Loading data... Done!")

    return profiles, profiles_df, areas_df


def process(config, paths):
    profiles, profiles_df, areas_df = read_data(paths)

    # adding information in profiles_df
    areas_df.loc[:, 'label'] = areas_df.index
    profiles_df = pd.merge(profiles_df, areas_df, on="area_id")

    if not paths.one_vs_all:
        profiles_df["idx_in_model"] = profiles_df.label

    profiles_df.loc[:, "npy_path"] = paths.output_profiles

    profiles_df = profiles_df.loc[:, ~profiles_df.columns.str.contains("^Unnamed")]
    logger.info("DataFrame: %s", paths.output_df)
    profiles_df.to_csv(paths.output_df)

    # norm profiles
    profiles = profiles[profiles_df.index_in_npy_array]
    norm_prof = NormProfiles(config, profiles)
    norm_x = norm_prof.norm_profiles

    np.save(paths.output_profiles, norm_x)

    np.save(paths.output_profiles, norm_x)
    logger.info("true_x: %s", paths.output_profiles)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
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
        "-d",
        "--areas-def",
        required=True,
        dest="areas_def",
        type=str,
        metavar="FILENAME",
        help="Path to csv file with info about area_id",
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
        "-s",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-y",
        "--output-y",
        required=True,
        dest="output_y",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    parser.add_argument(
        "-o",
        "--output-profiles",
        required=True,
        dest="output_profiles",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    parser.add_argument(
        "-f",
        "--output-df",
        required=True,
        dest="output_df",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    parser.add_argument(
        "-n",
        "--one-vs-all",
        required=False,
        action="store_true",
        dest="one_vs_all",
        help="Path to output directory",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)
    process(data_settings, input_options)
