import argparse
import logging

import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration
from step_04_01_areas_approaches import get_approach

from data_checker import check_data

C_LOGGER_NAME = "create_datasets"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    # Loading array with profiles
    acc_profile_array = np.load(paths.profiles_npy)
    logger.info("%s", paths.profiles_npy)

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    # Loading .csv file with label assignment
    areas_df = pd.read_csv(paths.areas_def)
    logger.info("%s", paths.areas_def)

    logger.info("Loading data... Done!")

    return acc_profile_array, profiles_df, areas_df


def pick_areas_dataset(config, paths):
    """
    Creates dataset according to the area picking rules
    in dataset_settings.json
    """
    profiles, profiles_df, areas_df = read_data(paths)
    check_data(profiles, profiles_df, logger)

    area_process_approach = get_approach(config)
    areas_handler = area_process_approach(config, profiles_df, areas_df)

    processed_df = areas_handler.process()

    label_ctr = processed_df.label.value_counts()

    for _label in label_ctr[label_ctr > config("max_label_amount")].index:
        _df = processed_df[(processed_df.label == _label)]

        num2drop = _df.shape[0] - config("max_label_amount")
        _drop_df = _df.sample(n=num2drop)

        processed_df = processed_df[~processed_df.index.isin(_drop_df.index)]

    pick_idx = processed_df.index_in_npy_array
    out_profiles = profiles[pick_idx]

    check_data(out_profiles, processed_df, logger)

    processed_df = processed_df.astype({"index_in_npy_array": "int"})
    processed_df.loc[:, "index_in_npy_array"] = np.arange(
        out_profiles.shape[0], dtype=int
    )

    processed_df.loc[:, "npy_path"] = paths.output_profiles
    processed_df = processed_df.drop(columns=["accept", "holdout_section"])
    processed_df = processed_df.loc[:, ~processed_df.columns.str.contains("^Unnamed")]

    logger.info("Saving profiles...")
    np.save(paths.output_profiles, out_profiles)
    logger.info("Done! Result saved to... %s!", paths.output_profiles)

    logger.info("Saving dataframe...")
    processed_df.to_csv(paths.dataset_csv)
    logger.info("Done! Result saved to... %s!", paths.dataset_csv)


def parse_args(doc_source):
    """ """
    parser = argparse.ArgumentParser(
        description=doc_source.__doc__,
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
    ),

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
        "-p",
        "--input-profiles-npy",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to npy file with accepted profiles",
    )

    parser.add_argument(
        "-f",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to csv file with info about accepted profiles",
    ),

    parser.add_argument(
        "-x",
        "--output-profiles",
        required=True,
        dest="output_profiles",
        type=str,
        metavar="FILENAME",
        help="Path to output npy file with profiles dataset",
    ),

    parser.add_argument(
        "-a",
        "--output-csv",
        required=True,
        dest="dataset_csv",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with labels dataset",
    ),

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args(pick_areas_dataset)
    data_settings = DatasetConfiguration(input_options.config_fname)
    pick_areas_dataset(data_settings, input_options)
