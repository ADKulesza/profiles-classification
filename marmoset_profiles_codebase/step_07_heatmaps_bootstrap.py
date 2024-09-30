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


def read_data(paths):
    logger.info("Loading data...")

    label_names = pd.read_csv(paths.label_names)
    logger.info("%s", paths.label_names)

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    heatmaps_list = glob.glob(paths.output + "heatmap_*.npy")
    logger.info("%s", heatmaps_list)

    logger.info("Loading data... Done!")

    return label_names, profiles_df, heatmaps_list


# TODO !
def process(config, paths):
    label_names, profiles_df, hm_list = read_data(paths)

    profiles_df = pd.merge(profiles_df, label_names, on="area_id")
    profile_len = config("profile_length")
    area_id_list = np.array(profiles_df.area_id.unique())

    # TODO zmienne
    n_blocks = 32
    k_bootstrap = 10000

    for hm_path in hm_list:
        logger.info("Loading heatmaps...")
        logger.info("%s", hm_path)
        heatmaps = np.load(hm_path)

        heatmaps_dict = {}
        all_values = np.array([])
        area_profiles_number = {}
        medians_dict = {}

        out_df_dict = {
            "area_id": [],
            "median_diff_stat": [],
            "q1_diff_stat": [],
            "q3_diff_stat": [],
            "idx": [],
        }

        for _area_id in area_id_list:

            _df = profiles_df[profiles_df.area_id == _area_id]
            array_idx = np.array(_df.index_in_npy_array)
            no_profiles = array_idx.shape[0]

            logger.info("Area: %s2 Profiles %s", _area_id, no_profiles)

            area_profiles_number[_area_id] = no_profiles

            _heatmaps = np.zeros((no_profiles, profile_len))
            _heatmaps[:] = heatmaps[array_idx, :]
            _heatmaps = np.reshape(_heatmaps, (-1, n_blocks), order="F")

            if all_values.shape[0] == 0:
                all_values = _heatmaps
            else:
                all_values = np.concatenate((all_values, _heatmaps), axis=0)

            medians_dict[_area_id] = np.zeros((k_bootstrap, 3, n_blocks))
            # median = np.median(x_arr, axis=0)
            # q1 = np.percentile(x_arr, 25, axis=0)
            # q3 = np.percentile(x_arr, 75, axis=0)
            heatmaps_dict[_area_id] = np.array(
                [
                    np.median(_heatmaps, axis=0),
                    np.percentile(_heatmaps, 25, axis=0),
                    np.percentile(_heatmaps, 75, axis=0),
                ]
            )

        baseline = np.zeros((area_id_list.shape[0], n_blocks))

        for i in range(k_bootstrap):
            logger.info("%s/%s", i, k_bootstrap)
            last_idx = 0
            np.random.shuffle(all_values)
            for j, _area_id in enumerate(area_id_list):
                n_prof = area_profiles_number[_area_id]
                _x_arr = all_values[last_idx : last_idx + n_prof]
                medians_dict[_area_id][i] = np.array(
                    [
                        np.median(_x_arr, axis=0),
                        np.percentile(_x_arr, 25, axis=0),
                        np.percentile(_x_arr, 75, axis=0),
                    ]
                )

                last_idx += n_prof
                baseline[j] += np.mean(_x_arr, axis=0)

        baseline /= k_bootstrap
        for i, _area_id in enumerate(area_id_list):
            m_diff = np.abs(medians_dict[_area_id][:, 0] - heatmaps_dict[_area_id][0])
            m_diff = np.sum(m_diff > heatmaps_dict[_area_id][0], axis=0) / k_bootstrap
            q1_diff = np.abs(medians_dict[_area_id][:, 1] - heatmaps_dict[_area_id][1])
            q1_diff = np.sum(q1_diff > heatmaps_dict[_area_id][0], axis=0) / k_bootstrap
            q3_diff = np.abs(medians_dict[_area_id][:, 2] - heatmaps_dict[_area_id][2])
            q3_diff = np.sum(q3_diff > heatmaps_dict[_area_id][0], axis=0) / k_bootstrap
            # logger.info("%s %s %s", m_diff.shape, q1_diff.shape, q3_diff.shape)
            out_df_dict["area_id"].append(_area_id)
            out_df_dict["median_diff_stat"].append(m_diff)
            out_df_dict["q1_diff_stat"].append(q1_diff)
            out_df_dict["q3_diff_stat"].append(q3_diff)
            out_df_dict["idx"].append(i)

        out_df = pd.DataFrame(out_df_dict)
        out_df_path = os.path.join(paths.output, f"cam_stat_{paths.holdout_id}.csv")
        out_df.to_csv(out_df_path)

        npy_path = os.path.join(paths.output, f"cam_baseline_{paths.holdout_id}.npy")
        np.save(npy_path, baseline)
    #
    # # area_name = _df.area.values[0]
    # # fname_hm = os.path.join(paths.output_dir, area_name + ".png")
    #
    # # all_values = np.vstack(heatmaps_dict.values())
    logger.info(heatmaps_dict)


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
        "-l",
        "--label-names",
        required=True,
        dest="label_names",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with",
    )

    parser.add_argument(
        "-i",
        "--holdout-id",
        required=True,
        dest="holdout_id",
        type=str,
        metavar="FILENAME",
        help="Path to  directory",
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
