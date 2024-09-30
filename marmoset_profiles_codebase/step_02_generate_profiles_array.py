import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration

C_LOGGER_NAME = "generate_profiles_array"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)


def load_profiles(section_path):
    """
    Load input label and profile arrays.
    """

    seg_path = glob.glob(section_path + "/*_segmentation.npy")[0]
    profiles_path = glob.glob(section_path + "/*_norm_profiles.npy")[0]
    return np.load(seg_path), np.load(profiles_path)


class ProfilesInfo:
    def __init__(self):
        self._last_index_in_array = 0

        self._profiles_info = {
            "case": [],
            "section": [],
            "profile_id": [],
            "area_id": [],
            "confidence": [],
            "npy_path": [],
            "index_in_npy_array": [],
        }

    def update_dict(self, case, section, seg_array, npy_path):
        n = seg_array.shape[0]
        if n == 0:
            return 0
        idx_array = np.arange(self._last_index_in_array, self._last_index_in_array + n)
        self._profiles_info["case"].extend([case] * n)
        self._profiles_info["section"].extend([section] * n)
        self._profiles_info["profile_id"].extend(np.arange(n, dtype=int))
        self._profiles_info["area_id"].extend(seg_array[:, 0].astype(np.uint8))
        self._profiles_info["confidence"].extend(np.round(seg_array[:, 1], 4))
        self._profiles_info["npy_path"].extend([npy_path] * n)
        self._profiles_info["index_in_npy_array"].extend(idx_array)

        self._last_index_in_array = idx_array[-1] + 1

    def save(self, paths):
        df = pd.DataFrame(data=self._profiles_info)
        if os.path.exists(paths.profiles_csv):
            in_df = pd.read_csv(paths.profiles_csv)
            df.loc[:, "index_in_npy_array"] = (
                df.index_in_npy_array + in_df.index_in_npy_array.iloc[-1] + 1
            )
            df = pd.concat((in_df, df))
        df.to_csv(paths.profiles_csv)
        logger.info("Dataframe with profile info saved to... %s", paths.profiles_csv)


def process_all_case(config, paths):
    output_profiles = np.array([])

    profiles_dict = ProfilesInfo()

    case = paths.case

    logger.info("Processing case...%s", case)

    # try - expect ???
    case_path = os.path.join(paths.profile_storage, case)
    if os.path.exists(case_path) is False:
        raise OSError("Path does not exist", case_path)

    sections = config.sections(case_path)

    for section, section_path in zip(*sections):
        logger.info("Section...%s", section)

        seg, profiles = load_profiles(section_path)

        profiles_dict.update_dict(case, section, seg, paths.profiles_npy)

        if output_profiles.size == 0:
            output_profiles = profiles
        else:
            output_profiles = np.concatenate((output_profiles, profiles), axis=0)

    logger.info("Saving profiles...")
    if os.path.exists(paths.profiles_npy):
        in_profiles = np.load(paths.profiles_npy)
        output_profiles = np.concatenate((in_profiles, output_profiles))
    np.save(paths.profiles_npy, output_profiles)
    logger.info("Done! Result saved to... %s", paths.profiles_npy)

    profiles_dict.save(paths)


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
        "-e",
        "--case",
        required=True,
        dest="case",
        type=str,
        metavar="FILENAME",
        help="",
    ),

    parser.add_argument(
        "-s",
        "--profile-storage",
        required=True,
        dest="profile_storage",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-x",
        "--output-profiles-npy",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-p",
        "--output-profiles-csv",
        required=True,
        dest="profiles_csv",
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
