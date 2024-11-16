import argparse
import logging
import os

import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration


C_LOGGER_NAME = "cam_boot"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)

N_BLOCKS = 32
K_BOOTSTRAP = 10000

COLUMNS_DF = ["left_whisker_h0", "q1_h0", "median_h0", "q3_h0", "right_whisker_ho"]


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    heatmap = np.load(paths.heatmap)
    logger.info("%s", paths.heatmap)

    logger.info("Loading data... Done!")

    return profiles_df, heatmap


def process(config, paths):
    profiles_df, heatmaps = read_data(paths)

    area_list = pd.unique(profiles_df.area)

    area_profiles = {}
    real_heatmap_dict = {}
    bootstrap_dict = {}

    for _area in area_list:
        _df = profiles_df[profiles_df.area == _area]
        array_idx = np.array(_df.index_in_npy_array)

        n_profiles = array_idx.shape[0]

        logger.info("Area: %s Profiles %s", _area, n_profiles)

        area_profiles[_area] = array_idx
        bootstrap_dict[_area] = np.zeros((K_BOOTSTRAP, 5, N_BLOCKS))

        _heatmaps = heatmaps[array_idx, :]
        _heatmaps = np.reshape(_heatmaps, (-1, N_BLOCKS), order="F")

        #  Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR
        # IQR = Q3 - Q1
        real_heatmap_stat = np.zeros((5, N_BLOCKS))
        real_heatmap_stat[1] = np.percentile(_heatmaps, 25, axis=0)
        real_heatmap_stat[2] = np.median(_heatmaps, axis=0)
        real_heatmap_stat[3] = np.percentile(_heatmaps, 75, axis=0)
        IQR = real_heatmap_stat[3] - real_heatmap_stat[1]
        real_heatmap_stat[0] = real_heatmap_stat[1] - 1.5 * IQR
        real_heatmap_stat[4] = real_heatmap_stat[3] + 1.5 * IQR

        real_heatmap_dict[_area] = real_heatmap_stat

        area_dir = os.path.join(paths.output, f"{_area}")
        if os.path.exists(area_dir):
            continue
        else:
            os.mkdir(area_dir)

        area_hm_path = os.path.join(area_dir, "real_heatmap.npy")
        np.save(area_hm_path, real_heatmap_stat)


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
        "-s",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-m",
        "--heatmap",
        required=True,
        dest="heatmap",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        dest="output",
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
