import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration
from read_json import read_json

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

    profile_len = config("profile_length")
    area_list = pd.unique(profiles_df.area)

    area_profiles = {}
    real_heatmap_dict = {}
    bootstrap_dict = {}
    out_df_dict = {
        "area": [],
        "left_whisker_h0": [],
        "q1_h0": [],
        "median_h0": [],
        "q3_h0": [],
        "right_whisker_h0": []
    }

    for _area in area_list:
        _df = profiles_df[profiles_df.area == _area]
        array_idx = np.array(_df.index_in_npy_array)

        n_profiles = array_idx.shape[0]

        logger.info("Area: %s2 Profiles %s", _area, n_profiles)

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

    shuffle_heatmaps = heatmaps.copy()
    for i in range(K_BOOTSTRAP):
        logger.info("%s/%s", i, K_BOOTSTRAP)
        random_idx = np.random.rand(heatmaps.shape[0]).argsort(axis=0)
        shuffle_heatmaps = shuffle_heatmaps.take(random_idx, axis=0)

        for _area, _area_idx in area_profiles.items():
            _heatmaps = heatmaps[_area_idx, :]
            _heatmaps = np.reshape(_heatmaps, (-1, N_BLOCKS), order="F")
            _heatmap_bootstrap = bootstrap_dict[_area][i]

            #  Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR
            # IQR = Q3 - Q1
            _heatmap_bootstrap[1] = np.percentile(_heatmaps, 25, axis=0)
            _heatmap_bootstrap[2] = np.median(_heatmaps, axis=0)
            _heatmap_bootstrap[3] = np.percentile(_heatmaps, 75, axis=0)
            IQR = _heatmap_bootstrap[3] - _heatmap_bootstrap[1]
            _heatmap_bootstrap[0] = _heatmap_bootstrap[1] - 1.5 * IQR
            _heatmap_bootstrap[4] = _heatmap_bootstrap[3] + 1.5 * IQR

    for _area, _area_idx in area_profiles.items():
        _heatmap_straps = bootstrap_dict[_area]
        _real_heatmap_area = real_heatmap_dict[_area]
        # out_df_dict["area"].append(_area)
        strap_heatmap_stat = np.zeros((5, N_BLOCKS))
        for i_stat in range(5):
            stat_diff = np.abs(_heatmap_straps[:, i_stat] - _real_heatmap_area[i_stat])
            stat_count = np.sum(stat_diff > _real_heatmap_area[i_stat], axis=0) / K_BOOTSTRAP
            # out_df_dict[COLUMNS_DF[i_stat]].append(stat_count)
            strap_heatmap_stat[i_stat] = stat_count

        area_dir = os.path.join(paths.output, f"{_area}")

        area_hm_path = os.path.join(area_dir, "heatmap_stats.npy")
        np.save(area_hm_path, strap_heatmap_stat)

    # out_df = pd.DataFrame(out_df_dict)


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
