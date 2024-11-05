import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from dataset_configuration import DatasetConfiguration
from data_checker import check_data

C_LOGGER_NAME = "get_ev_set"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True


def read_data(paths):
    logger.info("Loading data...")

    # Loading array with profiles
    all_profiles = np.load(paths.all_profiles_npy)
    logger.info("%s", paths.all_profiles_npy)

    # Loading .csv file with information about profiles
    all_profiles_df = pd.read_csv(paths.all_profiles_csv)
    logger.info("%s", paths.all_profiles_csv)

    labels_id = all_profiles_df.area_id.array

    logger.info("Loading data... Done!")

    return all_profiles, all_profiles_df, labels_id


def process(config, paths):
    all_profiles, all_profiles_df, labels = read_data(paths)

    ss_split = StratifiedShuffleSplit(
        n_splits=config("holdout_fold"),
        test_size=config("test_size"),
        random_state=config("random_seed"),
    )

    holdout_sections = config("holdout_sections")

    all_profiles_df['holdout_section'] = all_profiles_df.section.isin(holdout_sections)

    train_profiles_df = all_profiles_df[(~all_profiles_df["holdout_section"]) & (all_profiles_df["accept"])]

    to_split_idx = train_profiles_df.index_in_npy_array.array
    train_profiles = all_profiles[to_split_idx]
    train_labels = labels[to_split_idx]

    train_profiles_df["index_in_npy_array"] = np.arange(train_profiles_df.shape[0])

    set_split = ss_split.split(train_profiles, train_labels)

    logger.info("Splitting data...")
    logger.info("Number of holdout datasets: %s", config("holdout_fold"))

    for k, (train_idx, test_idx) in enumerate(set_split):
        train_profiles_df[f"holdout_{k}"] = False
        holdout_df = train_profiles_df.index_in_npy_array.isin(test_idx)
        train_profiles_df.loc[holdout_df, f"holdout_{k}"] = True

    logger.info("Splitting data... Done!")

    # ---
    check_data(train_profiles, train_profiles_df, logger)

    train_profiles_df.loc[:, "npy_path"] = paths.to_train_npy
    train_profiles_df = train_profiles_df.loc[:, ~train_profiles_df.columns.str.contains('^Unnamed')]

    # ---

    sections = config("holdout_sections")
    section_ho_df = all_profiles_df.loc[all_profiles_df.section.isin(sections)]

    section_ho_profiles = all_profiles[section_ho_df.index_in_npy_array]

    logger.info("%s", paths.sections_npy)

    section_ho_df.loc[:, "index_in_npy_array"] = np.arange(section_ho_profiles.shape[0])
    section_ho_df.loc[:, "npy_path"] = paths.sections_npy
    section_ho_df = section_ho_df.loc[
                    :, ~section_ho_df.columns.str.contains("^Unnamed")
                    ]

    check_data(section_ho_profiles, section_ho_df, logger)
    # ----

    # Saving datasets
    logger.info("Saving data...")

    # Saving train-fold datasets
    np.save(paths.to_train_npy, train_profiles)
    logger.info("%s", paths.to_train_npy)

    train_profiles_df.to_csv(paths.to_train_csv)
    logger.info("%s", paths.to_train_csv)

    # Saving sections holdout datasets
    np.save(paths.sections_npy, section_ho_profiles)
    logger.info("%s", paths.sections_npy)

    section_ho_df.to_csv(paths.sections_csv)
    logger.info("%s", paths.sections_csv)

    logger.info("Saving data... Done!")


def parse_args():
    """
    Provides command-line interface.
    """
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
        "-y",
        "--all-profiles",
        required=True,
        dest="all_profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-f",
        "--all-profiles-csv",
        required=True,
        dest="all_profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-t",
        "--output-to-train-profiles",
        required=True,
        dest="to_train_npy",
        type=str,
        metavar="FILENAME",
        help="Path to output",
    )

    parser.add_argument(
        "-u",
        "--output-to-train-profiles-csv",
        required=True,
        dest="to_train_csv",
        type=str,
        metavar="FILENAME",
        help="Path to output",
    )

    parser.add_argument(
        "-s",
        "--output-holdout-sections-profiles",
        required=True,
        dest="sections_npy",
        type=str,
        metavar="FILENAME",
        help="Path to output",
    )

    parser.add_argument(
        "-e",
        "--output-holdout-sections-profiles-csv",
        required=True,
        dest="sections_csv",
        type=str,
        metavar="FILENAME",
        help="Path to output",
    )

    parser.add_argument(
        "-o",
        "--output-csv",
        required=True,
        dest="output_csv",
        type=str,
        metavar="FILENAME",
        help="Path to output",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)
    process(data_settings, input_options)
