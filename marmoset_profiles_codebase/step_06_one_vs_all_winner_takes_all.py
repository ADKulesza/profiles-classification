import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from plot_methods.plot_logger import get_logger

from read_json import read_json

C_LOGGER_NAME = "mean_metrics"

PREDICTIONS_FNAME = "pred_y.npy"


def read_data(paths):
    logger.info("Loading data...")

    profiles_df = pd.read_csv(paths.profiles_csv)
    logger.info("%s", paths.profiles_csv)

    set_id = paths.holdout_set[-1]
    eval_paths = glob.glob(paths.evaluation_dir + "/[0-9]*/" + paths.holdout_set + f"/*_set_{set_id}[0-9]" + "/pred_y.npy")


    for _idx, y_pred_path in enumerate(eval_paths):
        _pred = np.load(y_pred_path)
        logger.info("DUPA %s", _pred[2])

    # model_order = read_json(paths.models_order)
    # logger.info("%s", paths.models_order)
    #
    # for model_set in model_order[paths.holdout_set]:
    #
    #
    #     _RE_DIR = re.compile(f"{paths.evaluation_dir}"
    #                          + r"[\\/][0-9]*[\\/]"
    #                          + f"{paths.holdout_set}"
    #                          + r"[\\/].*"
    #                          + f"{model_set}"
    #                          + r"$"
    #                          )
    #     for root, dirs, files in os.walk(paths.evaluation_dir):
    #         if not _RE_DIR.match(root):
    #             continue
    #
    #         _path = os.path.join(root, PREDICTIONS_FNAME)
    #         logger.info("PATH:  %s", _path)
    #
    #         pred_y = np.load(_path)
    #
    #         argmax_y = np.argmax(pred_y, axis=1)
    #         max_y = np.max(pred_y, axis=1)
    #         max_y[np.where(argmax_y == 0)[0]] = -max_y[np.where(argmax_y == 0)[0]]
    #
    #         label_id = re.search(r"[\\/][0-9]*[\\/]", root)[0]
    #         logger.info("LABEL ID %s", label_id)
    #         profiles_df[f"pred_confidence_{label_id}"] = max_y
    #
    #
    # logger.info("Loading data... Done!")

    return profiles_df


def clean_df(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.sort_values(by="dataset")
    df = df.set_index("dataset")

    return df


def get_mean_of_area_df(area_df):
    gb = area_df.groupby("area")
    df_mean = gb.mean()
    df_mean = df_mean.add_suffix("_mean", axis=1)

    df_std = gb.std()
    df_std = df_std.add_suffix("_std", axis=1)

    df = pd.concat((df_mean, df_std), axis=1)

    return df


def process(paths):
    profiles_df = read_data(paths)

    # output_df = clean_df(output_df)
    # output_df.loc["mean"] = output_df.mean()
    # output_df.loc["std"] = output_df.std()
    #
    # output_area_df = get_mean_of_area_df(area_df)
    # output_area_df = output_area_df.loc[
    #                  :, ~output_area_df.columns.str.contains("^Unnamed")
    #                  ]
    #
    # output_df.to_csv(paths.output)
    # logger.info("Results saved to... %s", paths.output)
    #
    # output_area_df.to_csv(paths.area_output)
    # logger.info("Results saved to... %s", paths.area_output)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-e",
        "--evaluation-dir",
        required=True,
        dest="evaluation_dir",
        type=str,
        metavar="FILENAME",
        help="Path to output npy file with",
    )

    parser.add_argument(
        "-c",
        "--profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-i",
        "--holdout-id",
        required=True,
        dest="holdout_set",
        type=str,
        metavar="FILENAME",
        help="Path to output npy file with",
    )

    parser.add_argument(
        "-j",
        "--models-order",
        required=True,
        dest="models_order",
        type=str,
        metavar="FILENAME",
        help="Path to output",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()

    process(input_options)
