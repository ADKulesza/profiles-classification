import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from dataset_configuration import DatasetConfiguration

C_LOGGER_NAME = "split_datasets"
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

    labels = np.load(paths.labels_npy)
    logger.info("%s", paths.labels_npy)

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    logger.info("Loading data... Done!")

    return profiles, labels, profiles_df


def process(config, paths):
    splitter = StratifiedShuffleSplit(
        n_splits=config("valid_fold"),
        test_size=config("valid_size"),
        random_state=config("random_seed"),
    )

    x_profiles, y_labels, df = read_data(paths)

    new_df = df.copy()

    ho_folds = [col for col in df.columns if col.startswith("holdout")]
    n_sets = len(ho_folds)

    logger.info("Cross shuffle-shuffle...")
    for i, dset in enumerate(ho_folds):
        logger.info("Dataset... %s/%s", i + 1, n_sets)
        _df = df[~df[dset]]

        _x_idx = _df.index_in_npy_array.array
        _x = x_profiles[_x_idx]
        _y = _df.label

        # split dataset
        for i_v, (train_idx, valid_idx) in enumerate(splitter.split(_x, _y)):

            # get indices in original order
            true_valid_idx = _x_idx[valid_idx]

            # get boolen df if the profile is in valid set
            df_valid = df.index_in_npy_array.isin(true_valid_idx)

            # preparing df with subdivision into datasets
            new_df[f"set_{i}{i_v}"] = "train"
            new_df.loc[df[dset], f"set_{i}{i_v}"] = "test"
            new_df.loc[df_valid, f"set_{i}{i_v}"] = "valid"
    logger.info("Cross shuffle-shuffle... Done!")

    new_df = new_df.loc[:, ~new_df.columns.str.contains("^Unnamed")]

    logger.info("Saving data...")
    new_df.to_csv(paths.split_profiles_csv)
    logger.info("%s", paths.split_profiles_csv)
    logger.info("Saving data... Done!")


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
    ),

    parser.add_argument(
        "-y",
        "--labels",
        required=True,
        dest="labels_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-d",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-t",
        "--output-split-profiles-csv",
        required=True,
        dest="split_profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)
    process(data_settings, input_options)
