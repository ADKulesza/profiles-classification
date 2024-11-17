import argparse
import logging

import pandas as pd


C_LOGGER_NAME = "label_prob"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    logger.info("Loading data... Done!")

    return profiles_df


def process(paths):
    logger.info("Get probabilities... ")
    profiles_df = read_data(paths)

    n_profiles = profiles_df.shape[0]
    area_distribution = profiles_df['area'].value_counts()

    area_proportions = area_distribution / n_profiles
    area_proportions = area_proportions.to_frame(name="prob")
    logger.info("area_proportions: %s", area_proportions)

    profiles_df = profiles_df.drop(['prob'], axis=1, errors='ignore')
    profiles_df = pd.merge(profiles_df, area_proportions, on="area")

    profiles_df = profiles_df.loc[:, ~profiles_df.columns.str.contains("^Unnamed")]
    profiles_df.to_csv(paths.profiles_csv)
    logger.info("Get probabilities... Done!")
    logger.info("Results saved in... %s", paths.profiles_csv)


def parse_args(doc_source):
    """ """
    parser = argparse.ArgumentParser(
        description=doc_source.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with labels dataset",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args(process)
    process(input_options)
