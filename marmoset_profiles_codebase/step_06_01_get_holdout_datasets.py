import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from read_json import read_json
from sort_map import get_sort_map

C_LOGGER_NAME = "get_holdout"
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

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    labels_id = profiles_df.area_id.array

    label_names = pd.read_csv(paths.label_names)
    logger.info("%s", paths.label_names)

    area_order = read_json(paths.area_order)
    logger.info("%s", paths.area_order)

    logger.info("Loading data... Done!")

    return profiles, profiles_df, labels_id, label_names, area_order


class CreateHoldoutSets:
    """ """

    def __init__(self, paths):
        self._paths = paths
        (
            self._profiles,
            self._profiles_df,
            self._labels,
            self._label_names,
            self._area_order,
        ) = read_data(paths)

        sort_map = get_sort_map(self._area_order)

        self._label_names["area_order"] = self._label_names["area"].map(
            sort_map["index"]
        )

    def run(self):

        holdout_list = [
            col for col in self._profiles_df.columns if col.startswith("holdout_")
        ]
        set_col_list = [
            col for col in self._profiles_df.columns if col.startswith("set_")
        ]

        holdout_dict = {}

        for holdout_name in holdout_list:
            _ho_df = self._profiles_df[self._profiles_df[holdout_name]]

            _drop_col_list = holdout_list.copy()
            _drop_col_list.remove(holdout_name)

            _ho_df = _ho_df.drop(_drop_col_list, axis=1)

            _do_holdout_set = _ho_df.loc[:, set_col_list].eq("test").any()
            _drop_col_list = [k for k, v in _do_holdout_set.items() if v is False]

            _ho_df = _ho_df.drop(_drop_col_list, axis=1)

            holdout_set_col_list = [
                col for col in _ho_df.columns if col.startswith("set_")
            ]

            holdout_dict[holdout_name] = holdout_dict.get(holdout_name, [])
            holdout_dict[holdout_name].extend(holdout_set_col_list)

            logger.info("%s", holdout_dict)

            holdout_x = self._profiles[_ho_df.index_in_npy_array]
            holdout_y = self._labels[_ho_df.index_in_npy_array]

            dataset_path = os.path.join(self._paths.output, f"{holdout_name}")
            x_path = os.path.join(dataset_path, "x_norm.npy")
            y_path = os.path.join(dataset_path, "y_true.npy")

            if not os.path.exists(dataset_path):
                os.mkdir(dataset_path)

            logger.info("\n%s paths:", holdout_name)
            logger.info("x: %s", x_path)
            np.save(x_path, holdout_x)
            logger.info("y: %s", y_path)
            np.save(y_path, holdout_y)

            _ho_df.loc[:, "index_in_npy_array"] = np.arange(holdout_x.shape[0])
            _ho_df.loc[:, "npy_path"] = x_path

            _ho_df = _ho_df.loc[:, ~_ho_df.columns.str.contains("^Unnamed")]

            _ho_df = pd.merge(_ho_df, self._label_names, how="left", on="area_id")

            _ho_df["region"] = np.nan
            area_regions = self._area_order[2]
            for region, area_list in area_regions.items():
                _ho_df.loc[_ho_df.area.isin(area_list), "region"] = region

            _ho_df_path = os.path.join(dataset_path, "holdout_info.csv")
            logger.info("DataFrame: %s", _ho_df_path)
            _ho_df.to_csv(_ho_df_path)

        # Serializing json
        json_object = json.dumps(holdout_dict, indent=4)

        # Writing to sample.json
        with open(self._paths.models_order, "w") as outfile:
            outfile.write(json_object)
        logger.info("Holdout datasets: %s", self._paths.models_order)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=CreateHoldoutSets.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-x",
        "--profiles",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-c",
        "--split-profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-n",
        "--label-names",
        required=True,
        dest="label_names",
        type=str,
        metavar="FILENAME",
        help="Path to csv file with",
    )

    parser.add_argument(
        "-r",
        "--area-order",
        required=True,
        dest="area_order",
        type=str,
        metavar="FILENAME",
        help="Path to file with",
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

    parser.add_argument(
        "-j",
        "--output-models-order",
        required=True,
        dest="models_order",
        type=str,
        metavar="FILENAME",
        help="Path to output",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    input_options = parse_args()
    ho_sets = CreateHoldoutSets(input_options)
    ho_sets.run()
