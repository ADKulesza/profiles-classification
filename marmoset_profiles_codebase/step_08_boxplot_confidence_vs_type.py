import argparse
import os

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from plot_methods.plot_formatting import AxesFormatting
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties
from read_json import read_json

C_DPI = 300

C_FIGSIZE = (9, 5)
MARKERSIZE = 3
C_STD_COLOR = '#cccccc'

X_AXIS = {
    'min': 0,
    'max': 5,
    'step': 1
}

Y_AXIS = {
    'min': 0,
    'max': 1,
    'step': 0.2
}

LINEWIDTH = 1
MEDIANLINE_PROPS = dict(linestyle='-', linewidth=LINEWIDTH, color='k')
BOX_PROPS = dict(linestyle='-', linewidth=LINEWIDTH, color='k')
FLIER_PROPS = dict(marker='o', markeredgecolor='none',
                   markersize=MARKERSIZE,
                   markerfacecolor='k')

C_LOGGER_NAME = "metrics_vs"


def confidence_boxplot(data, metric, output_dir, do_svg=False):
    plt_prop = PlotProperties()
    fig, ax = plt.subplots(figsize=plt_prop.cm2inch(C_FIGSIZE))

    ax.boxplot(data, showfliers=False, widths=0.75,
               flierprops=FLIER_PROPS,
               boxprops=BOX_PROPS, medianprops=MEDIANLINE_PROPS)

    # axes_formatter_right = AxesFormatting(ax)
    # axes_formatter_right.format_axes(X_AXIS, Y_AXIS,)

    ax.set_xlabel(
        "Typ obszaru korowego", labelpad=plt_prop.label_pad, fontproperties=plt_prop.font
    )

    ax.set_ylabel(
        "Pewność modelu", labelpad=plt_prop.label_pad, fontproperties=plt_prop.font
    )

    # ax.set_ylim(0.5, 1)
    # ax["right"].set_xlabel(f"{metric} score", fontproperties=plt_prop.font, labelpad=15, x=0.6)

    prop = dict(left=0.15, right=0.985, top=0.98, bottom=0.2, wspace=0.05)
    plt.subplots_adjust(**prop)
    #
    plt.savefig(os.path.join(output_dir, f"{metric}_vs_type.png"), dpi=C_DPI)

    if do_svg:
        plt.savefig(os.path.join(output_dir, f"{metric}_vs_type.svg"), dpi=C_DPI)


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.validation_csv)
    logger.info("%s", paths.validation_csv)

    label_names = pd.read_csv(paths.label_names)
    logger.info("%s", paths.label_names)

    logger.info("Loading data... Done!")

    return profiles_df, label_names


def process(paths):
    profiles_df, label_names = read_data(paths)
    # profiles_df = profiles_df[profiles_df.idx_in_model == profiles_df.pred_y]

    # labels_y = profiles_df.area.array

    confidence_values = profiles_df["pred_confidence"].array

    type_values = profiles_df["type_id"].array

    boxplot_data = [confidence_values[type_values == i] for i in range(6)]

    confidence_boxplot(boxplot_data, "all_pred_confidence",
                       paths.output_dir, paths.do_svg)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--validation-csv",
        required=True,
        dest="validation_csv",
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
        "-d",
        "--output-dir",
        required=True,
        dest="output_dir",
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

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()

    process(input_options)
