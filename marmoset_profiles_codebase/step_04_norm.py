import argparse
import logging

import numpy as np

from dataset_configuration import DatasetConfiguration
from norm import NormProfiles

C_LOGGER_NAME = "norm"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    # Loading array with profiles
    profiles = np.load(paths.input_profiles)
    logger.info("%s", paths.input_profiles)

    logger.info("Loading data... Done!")

    return profiles


def process(config, paths):
    """ """
    logger.info("Norm profiles...")
    profiles = read_data(paths)

    norm_prof = NormProfiles(config, profiles)
    norm_prof_arr = norm_prof.norm_profiles

    np.save(paths.norm_profiles, norm_prof_arr)

    logger.info("Norm profiles... Done!")


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
        "-l",
        "--input-profiles",
        required=True,
        dest="input_profiles",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-p",
        "--output-norm-profiles",
        required=True,
        dest="norm_profiles",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)
    process(data_settings, input_options)
