import argparse
import os
import re

import pandas as pd

from plot_methods.plot_logger import get_logger

C_LOGGER_NAME = "mean_metrics"

RE_DIR = re.compile(r".*holdout_\d*[\\/].*_set_\d*[\\/]metrics$")
RE_AREA_MET = re.compile(".*area_metrics.csv$")
METRICS_FNAME = "macro_metrics.csv"


def read_data(paths):
    logger.info("Loading data...")

    output_df = pd.DataFrame()
    area_df = pd.DataFrame()

    for root, dirs, files in os.walk(paths.evaluation_dir):

        if not RE_DIR.match(root):
            continue

        _path = os.path.join(root, METRICS_FNAME)
        _df = pd.read_csv(_path)
        logger.info("%s", _path)

        if output_df.shape[0] == 0:
            output_df = _df
        else:
            output_df = pd.concat((output_df, _df), axis=0)

        if paths.binary:
            continue

        for _f in files:

            if not RE_AREA_MET.match(_f):
                continue

            _path = os.path.join(root, _f)
            _df = pd.read_csv(_path)
            logger.info("%s", _path)

            if area_df.shape[0] == 0:
                area_df = _df
            else:
                area_df = pd.concat((area_df, _df), axis=0)

    logger.info("Loading data... Done!")

    return output_df, area_df


def clean_df(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.sort_values(by="dataset")
    df = df.set_index("dataset")

    return df


def get_mean_of_area_df(area_df):
    gb = area_df[["area", "accuracy", "recall", "precision", "f1"]].groupby("area")
    df_median = gb.median(numeric_only=True)
    df_median = df_median.add_suffix("_median", axis=1)

    df_q1 = gb.quantile(0.25, numeric_only=True)
    df_q1 = df_q1.add_suffix("_q1", axis=1)

    df_q3 = gb.quantile(0.75, numeric_only=True)
    df_q3 = df_q3.add_suffix("_q3", axis=1)

    gb_info = area_df[["area", "area_order", "area_id", "idx_in_model", "region",
                       "color_r", "color_g", "color_b"]].groupby("area")
    df_info = gb_info.apply(lambda x: x.drop_duplicates()).reset_index(drop=True)
    df_info.set_index('area', inplace=True)
    df = pd.concat((df_info, df_median, df_q1, df_q3), axis=1)

    return df


def process(paths):
    output_df, area_df = read_data(paths)

    output_df = clean_df(output_df)
    output_df.loc["mean"] = output_df.mean()
    output_df.loc["std"] = output_df.std()

    output_df.to_csv(paths.output)
    logger.info("Results saved to... %s", paths.output)

    if paths.binary:
        return 0

    output_area_df = get_mean_of_area_df(area_df)
    output_area_df = output_area_df.loc[
                     :, ~output_area_df.columns.str.contains("^Unnamed")
                     ]

    output_area_df.to_csv(paths.area_output)
    logger.info("Results saved to... %s", paths.area_output)


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
        "-o",
        "--output",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-a",
        "--area-output",
        required=True,
        dest="area_output",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-v",
        "--one-vs-all",
        required=False,
        action="store_true",
        dest="binary",
        help="Path to output directory",
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()

    process(input_options)
