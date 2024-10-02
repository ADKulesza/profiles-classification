import argparse
import logging

import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration
from norm import NormProfiles
from read_json import read_json
from sort_map import get_sort_map

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

    labels_df = pd.read_csv(paths.labels_processed)
    logger.info("%s", paths.labels_processed)

    label_names = pd.read_csv(paths.label_names)
    logger.info("%s", paths.label_names)

    area_order = read_json(paths.area_order)
    logger.info("%s", paths.area_order)

    logger.info("Loading data... Done!")

    return profiles, profiles_df, labels_df, label_names, area_order


def process(config, paths):
    (profiles, profiles_df,
     labels_df, label_names, area_order) = read_data(paths)

    sort_map = get_sort_map(area_order)

    label_names["area_order"] = label_names["area"].map(
        sort_map["index"]
    )

    # adding information in profiles_df
    profiles_df.loc[:, "npy_path"] = paths.output_profiles
    profiles_df = profiles_df.loc[:, ~profiles_df.columns.str.contains("^Unnamed")]

    profiles_df = pd.merge(profiles_df, labels_df[["area_id", "idx_in_model"]], how="left", on="area_id")

    profiles_df = pd.merge(profiles_df, label_names, how="left", on="area_id")

    profiles_df["region"] = np.nan
    area_regions = area_order[2]
    for region, area_list in area_regions.items():
        profiles_df.loc[profiles_df.area.isin(area_list), "region"] = region

    logger.info("DataFrame: %s", paths.output_df)
    profiles_df.to_csv(paths.output_df)

    # y
    true_y = profiles_df.idx_in_model.array

    logger.info("true_y: %s", paths.output_y)
    np.save(paths.output_y, true_y)

    # norm profiles
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
        help="Path to csv file with",
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
        "-o",
        "--output-profiles",
        required=True,
        dest="output_profiles",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    parser.add_argument(
        "-d",
        "--output-df",
        required=True,
        dest="output_df",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)
    process(data_settings, input_options)
