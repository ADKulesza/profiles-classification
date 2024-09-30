import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from dataset_configuration import DatasetConfiguration

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
    profiles = np.load(paths.profiles_npy)
    logger.info("%s", paths.profiles_npy)

    # Loading array with profiles
    all_profiles = np.load(paths.all_profiles_npy)
    logger.info("%s", paths.all_profiles_npy)

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    # Loading .csv file with information about profiles
    all_profiles_df = pd.read_csv(paths.all_profiles_csv)
    logger.info("%s", paths.all_profiles_csv)

    labels_id = profiles_df.area_id.array

    logger.info("Loading data... Done!")

    return profiles, all_profiles, profiles_df, all_profiles_df, labels_id


def extract_section_holdout(config, df):
    df = df.copy()
    sections = config("holdout_sections")
    df[f"section_holdout"] = False
    df.loc[df.section.isin(sections), f"section_holdout"] = True
    df.loc[df.section.isin(sections), f"accept"] = False

    to_split_idx = df.loc[~df.section_holdout].index_in_npy_array

    return df, to_split_idx.array


def check_data(profiles, df):
    if profiles.shape[0] != df.shape[0]:
        logger.error("Mismatch shape!")


def process(config, paths):
    profiles, all_profiles, profiles_df, all_profiles_df, labels = read_data(paths)

    holdout_split = StratifiedKFold(
        n_splits=config("holdout_fold"),
        shuffle=True,
        random_state=config("random_seed"),
    )

    split_df, to_split_idx = extract_section_holdout(config, profiles_df)

    to_split_profiles = profiles[to_split_idx]
    to_split_labels = labels[to_split_idx]

    profiles_df = profiles_df[split_df.accept]
    profiles_df["index_in_npy_array"] = np.arange(to_split_profiles.shape[0])

    set_split = holdout_split.split(to_split_profiles, to_split_labels)

    logger.info("Splitting data...")
    logger.info("Number of holdout datasets: %s", config("holdout_fold"))

    for k, (train_idx, holdout_idx) in enumerate(set_split):
        profiles_df[f"holdout_{k}"] = False
        holdout_df = profiles_df.index_in_npy_array.isin(holdout_idx)
        profiles_df.loc[holdout_df, f"holdout_{k}"] = True

    logger.info("Splitting data... Done!")

    logger.info("Saving data...")

    # Saving datasets
    np.save(paths.to_train_npy, to_split_profiles)
    logger.info("%s", paths.to_train_npy)

    profiles_df.loc[:, "npy_path"] = paths.to_train_npy
    profiles_df = profiles_df.loc[:, ~profiles_df.columns.str.contains('^Unnamed')]
    profiles_df.to_csv(paths.to_train_csv)
    logger.info("%s", paths.to_train_csv)
    check_data(to_split_profiles, profiles_df)

    # Saving sections holdout datasets
    sections = config("holdout_sections")
    section_ho_df = all_profiles_df.loc[all_profiles_df.section.isin(sections)]

    section_ho_profiles = all_profiles[section_ho_df.index_in_npy_array]
    np.save(paths.sections_npy, section_ho_profiles)
    logger.info("%s", paths.sections_npy)

    section_ho_df.loc[:, "index_in_npy_array"] = np.arange(section_ho_profiles.shape[0])
    section_ho_df.loc[:, "npy_path"] = paths.sections_npy
    section_ho_df = section_ho_df.loc[
        :, ~section_ho_df.columns.str.contains("^Unnamed")
    ]
    section_ho_df.to_csv(paths.sections_csv)
    logger.info("%s", paths.sections_csv)
    check_data(section_ho_profiles, section_ho_df)

    # Saving dataframe with modification info
    split_df.to_csv(paths.output_csv)
    logger.info("%s", paths.output_csv)

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
        "-x",
        "--profiles",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

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
        "-d",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
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
