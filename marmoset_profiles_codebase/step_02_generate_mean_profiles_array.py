import argparse
import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration

C_LOGGER_NAME = "mean_profiles_array"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def read_data(paths):
    logger.info("Loading data...")

    profiles = np.load(paths.profiles_npy)
    logger.info("%s", paths.profiles_npy)

    segmentation = np.load(paths.labels_npy)
    logger.info("%s", paths.labels_npy)

    profile_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    logger.info("Loading data... Done!")

    return profiles, segmentation, profile_df


def moving_average(profiles, seg, mean_range):
    prof_means = np.zeros(profiles.shape)
    seg_means = np.zeros(seg.shape)

    for i in range(mean_range, profiles.shape[0] - mean_range, 1):
        prof_means[i, :] = np.mean(profiles[i - mean_range : i + mean_range, :], axis=0)
        seg_means[i] = np.mean(seg[i - mean_range : i + mean_range, :], axis=0)

    for i in range(mean_range):
        # Few first profiles with seg
        profiles_to_avg = prof_means[-2 * mean_range + i + 1 :, :]
        profiles_to_avg = np.concatenate((profiles_to_avg, prof_means[: i + 1, :]))
        prof_means[i, :] = np.mean(profiles_to_avg, axis=0)

        seg_to_avg = seg[: 2 * mean_range - i - 1, :]
        seg_to_avg = np.concatenate((seg_to_avg, seg[: i + 1, :]))
        seg_means[i, :] = np.mean(seg_to_avg, axis=0)

        # Few last profiles with seg
        profiles_to_avg = prof_means[: 2 * mean_range - i - 1, :]
        profiles_to_avg = np.concatenate((profiles_to_avg, prof_means[-i - 1 :, :]))
        prof_means[-i - 1, :] = np.mean(profiles_to_avg, axis=0)

        seg_to_avg = seg[: 2 * mean_range - i - 1, :]
        seg_to_avg = np.concatenate((seg_to_avg, seg[-i - 1 :, :]))
        seg_means[-i - 1, :] = np.mean(seg_to_avg, axis=0)

    seg_means[:, 0] = np.round(seg_means[:, 0])

    return prof_means, seg_means


def process_all_case(config, paths):
    AVG_LEN = config("data_mean")
    profiles, segmentation, profile_df = read_data(paths)

    area_id_list = pd.unique(profile_df.area_id)
    case_list = pd.unique(profile_df.case)

    mean_profiles = np.zeros(profiles.shape)
    mean_segmentation = np.zeros(segmentation.shape)
    mean_profile_df = deepcopy(profile_df)

    for case in case_list:
        logger.info("Case... %s", case)
        case_df = profile_df[profile_df.case == case]
        section_list = pd.unique(case_df.section)
        for section in section_list:
            logger.info("Section... %s", section)
            section_df = case_df[case_df.section == section]

            section_df.sort_values(by=["profile_id"], inplace=True)
            indices = section_df.index_in_npy_array
            section_profiles = profiles[indices]
            section_seg = segmentation[indices]

            x, y = moving_average(section_profiles, section_seg, AVG_LEN)
            mean_profiles[indices, :] = x
            mean_segmentation[indices, :] = y

            mean_profile_df.loc[section_df.index, "index_in_npy_array"] = indices
            mean_profile_df.loc[section_df.index, "area_id"] = y[:, 0]
            mean_profile_df.loc[section_df.index, "confidence"] = y[:, 1]

    mean_profile_df["npy_path"] = paths.output_profiles_npy

    # "Round" bad label to real label
    bad_label = np.setdiff1d(mean_segmentation[:, 0], area_id_list).astype(np.uint8)
    bad_label_idx = np.where(np.isin(mean_segmentation[:, 0], bad_label))[0]

    for i in bad_label_idx:
        false_l = mean_segmentation[i, 0]
        true_idx = np.argmin(np.abs(area_id_list - false_l))
        true_l = area_id_list[true_idx].astype(np.uint8)
        mean_segmentation[i, 0] = true_l
        mean_profile_df.loc[mean_profile_df.index_in_npy_array == i, "area_id"] = true_l

    logger.info("AAAAAAAA %s", np.sort(pd.unique(mean_profile_df.area_id)))
    logger.info("BBBBBB %s", mean_profile_df.columns)
    logger.info("CCCCCCC %s", np.unique(mean_segmentation[:, 0]))

    mean_profile_df = mean_profile_df.loc[
        :, ~mean_profile_df.columns.str.contains("^Unnamed")
    ]
    mean_profile_df.reset_index(inplace=True)

    logger.info("Saving profiles...")

    mean_profile_df.to_csv(paths.output_profiles_csv)
    logger.info("Done! Result saved to... %s!", paths.output_profiles_csv)

    np.save(paths.output_profiles_npy, mean_profiles)
    logger.info("Done! Result saved to... %s!", paths.output_profiles_npy)

    np.save(paths.output_labels_npy, mean_segmentation)
    logger.info("Done! Result saved to... %s!", paths.output_labels_npy)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process_all_case.__doc__,
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
        "-x",
        "--input-profiles-npy",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-y",
        "--input-labels-npy",
        required=True,
        dest="labels_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-p",
        "--input-profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-l",
        "--output-profiles-npy",
        required=True,
        dest="output_profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-m",
        "--output-labels-npy",
        required=True,
        dest="output_labels_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-n",
        "--output-profiles-csv",
        required=True,
        dest="output_profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)
    process_all_case(data_settings, input_options)
