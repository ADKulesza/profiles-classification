import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_methods.plot_formatting import AxesFormattingConfusionMatrix
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties
from read_json import read_json
from sort_map import get_sort_map
from step_08_plot_confusion_matrix import plot_confmat

C_LOGGER_NAME = "mean_cmat"

RE_DIR = re.compile(".*holdout_\d*$")
RE_CMAT = re.compile("^cmat_\d*.npy$")


def read_data(paths):
    logger.info("Loading data...")

    confmat = np.array([])
    cmat_ctr = 0
    for root, dirs, files in os.walk(paths.evaluation_dir):
        if not RE_DIR.match(root):
            continue

        cmat_fname = [s for s in files if RE_CMAT.match(s)][0]

        _path = os.path.join(root, cmat_fname)
        logger.info("%s", _path)
        if confmat.shape[0] == 0:
            confmat = np.load(_path)
        else:
            confmat += np.load(_path)

        cmat_ctr += 1

    confmat = confmat / cmat_ctr

    labels_df = pd.read_csv(paths.labels_processed)
    logger.info("%s", paths.labels_processed)

    label_names = pd.read_csv(paths.label_names)
    logger.info("%s", paths.label_names)

    logger.info("Loading data... Done!")

    return confmat, labels_df, label_names


def process(paths, order):
    confmat, labels_df, label_names = read_data(paths)

    # for i in range(confmat.shape[0]):
    #     _cmat = np.zeros((2,2))
    #     _cmat

    sort_map = get_sort_map(order)

    labels_df = labels_df.sort_values(by=["idx_in_model"])
    _df = pd.merge(labels_df, label_names, how="left", on="area_id")
    _df["area_order"] = _df["area"].map(sort_map["index"])

    _df = _df.sort_values("area_order")
    _df.dropna(inplace=True)

    x_labels = np.array(_df.dropna().area)

    plt_prop = PlotProperties()

    plot_confmat(
        confmat, paths.confmat_plot, x_labels, paths.show_values, plt_prop, logger
    )


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--label-names",
        required=True,
        dest="label_names",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with",
    )

    parser.add_argument(
        "-b",
        "--labels-processed",
        required=True,
        dest="labels_processed",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with",
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
        "--output-confmat-plot",
        required=True,
        dest="confmat_plot",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument(
        "-s",
        "--svg",
        required=False,
        action="svg",
        dest="do_svg",
        help="Do svg plot?",
    )
    #
    parser.add_argument("--show-values", action="store_true", dest="show_values")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()

    order_fname = "areas_order.json"  # TODO
    area_order = read_json(order_fname)
    process(input_options, area_order)
