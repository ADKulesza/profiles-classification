import argparse
import logging

import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration
from step_03_areas_approaches import get_approach
from step_03_areas_balancer import balance_dataset

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
    acc_profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)
    acc_profiles_df["label"] = acc_profiles_df.area_id

    # Loading .csv file with label assignment
    labels_df = pd.read_csv(paths.labels_idx)
    logger.info("%s", paths.labels_idx)
    labels_df.label = labels_df.area_id

    logger.info("Loading data... Done!")

    return acc_profile_array, acc_profiles_df, labels_df


def check_data(prof_arr, df):
    if prof_arr.shape[0] == df.shape[0]:
        logger.info("Data sizes are matching! :)")
    else:
        raise ValueError("Mismatch data shape!", prof_arr.shape[0], df.shape[0])

    if not isinstance(prof_arr, (np.ndarray, np.generic)):
        raise ValueError(f"Wrong data type! {type(prof_arr)}")


def process(config, paths):
    profiles, acc_df, labels_df = read_data(paths)
    check_data(profiles, acc_df)

    area_process_approach = get_approach(config)
    areas_handler = area_process_approach(config, acc_df)

    # Labels processed csv file
    area_id_list = areas_handler.areas_to_process
    if config("other_to_zero"):
        _df = pd.DataFrame({"area_id": [0], "label": [0]})
        labels_df = pd.concat([_df, labels_df], ignore_index=True)

    labels_df = labels_df[labels_df.area_id.isin(area_id_list)]
    labels_df.to_csv(paths.labels_processed)

    profiles_df = areas_handler.process()

    profiles_df = balance_dataset(
        config,
        profiles_df,
        areas_handler.areas_to_process,
        paths.graph,
        paths.label_weights,
    )

    pick_idx = profiles_df[profiles_df.accept].index_in_npy_array
    out_profiles = profiles[pick_idx]

    check_data(out_profiles, profiles_df[profiles_df.accept])

    profiles_df = profiles_df.astype({"index_in_npy_array": "int"})
    profiles_df.loc[profiles_df.accept, "index_in_npy_array"] = np.arange(
        out_profiles.shape[0], dtype=int
    )

    profiles_df = profiles_df[profiles_df.accept]
    profiles_df = profiles_df.loc[:, ~profiles_df.columns.str.contains("^Unnamed")]
    profiles_df.loc[:, "npy_path"] = paths.output_profiles

    logger.info("Saving profiles...")
    np.save(paths.output_profiles, out_profiles)
    logger.info("Done! Result saved to... %s!", paths.output_profiles)

    logger.info("Saving dataframe...")
    profiles_df.to_csv(paths.dataset_csv)
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
        "--labels",
        required=True,
        dest="labels_idx",
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
        "--acc-profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to csv file with info about accepted profiles",
    ),

    parser.add_argument(
        "-g",
        "--graph-path",
        required=True,
        dest="graph",
        type=str,
        metavar="FILENAME",
        help="Path to json file with graph of connected labels",
    ),

    parser.add_argument(
        "-b",
        "--labels-processed",
        required=True,
        dest="labels_processed",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with ",
    )

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

    parser.add_argument(
        "-w",
        "--output-label-weights",
        required=True,
        dest="label_weights",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args(process)
    data_settings = DatasetConfiguration(input_options.config_fname)
    process(data_settings, input_options)
