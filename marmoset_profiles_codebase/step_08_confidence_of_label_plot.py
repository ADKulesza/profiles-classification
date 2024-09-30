import argparse
import logging
import os

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties

C_LOGGER_NAME = "label_conf"

# Plot properties
C_DPI = 300

C_DEFAULT_FONT_SIZE = 8

C_FIGSIZE = (2, 4)

LINEWIDTH = 1
BARWIDTH = 1.3
Y_TICKS_LEN = 2
Y_TICKS_PAD = 0.75

LABELPAD = 2
MARKERSIZE = 2

MEDIANLINE_PROPS = dict(linestyle="-", linewidth=LINEWIDTH, color="k")
BOX_PROPS = dict(linestyle="-", linewidth=LINEWIDTH, color="k")
FLIER_PROPS = dict(
    marker="o", markeredgecolor="none", markersize=MARKERSIZE, markerfacecolor="k"
)

Y_AXIS = {"min": 0, "max": 1, "step": 0.2}
X_MAJORS = np.arange(1, 3)
X_LABELS = ["C", "I"]


def axes_formatting(ax, plt_prop):
    """
    Details for the axis
    """
    y_majors = np.arange(Y_AXIS["min"], Y_AXIS["max"] + Y_AXIS["step"], Y_AXIS["step"])

    # Distance between ticks and label of ticks
    ax.tick_params(
        axis="y",
        which="major",
        pad=Y_TICKS_PAD,
        length=Y_TICKS_LEN,
        width=1,
        left="off",
        labelleft="off",
        direction="in",
    )

    ax.tick_params(
        axis="y",
        which="minor",
        pad=Y_TICKS_PAD,
        length=Y_TICKS_LEN / 1.5,
        width=1,
        left="off",
        labelleft="off",
        direction="in",
    )

    ax.tick_params(axis="x", which="both", pad=0.5, length=0)

    ax.xaxis.labelpad = LABELPAD
    ax.yaxis.labelpad = LABELPAD

    # Make rightline invisible
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Limits and ticks for y-axis
    ax.set_ylim(Y_AXIS["min"], Y_AXIS["max"] + 0.05)
    ax.spines["left"].set_bounds(Y_AXIS["min"], Y_AXIS["max"] + 0.001)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    last_tick = ax.yaxis.get_minor_ticks()[-1]
    last_tick.tick1line.set_markersize(0)
    last_tick.tick2line.set_markersize(0)

    labels = map(lambda x: "{:.1f}".format(x) if x != 0 else "0", y_majors)
    ax.set_yticks(y_majors)
    ax.set_yticklabels(labels, fontproperties=plt_prop.font)

    # Limits and ticks for x-axis
    ax.set_xlim(X_MAJORS[0] - BARWIDTH / 1.8, X_MAJORS[-1] + BARWIDTH / 1.7)
    ax.set_xticks(X_MAJORS)
    ax.set_xticklabels(X_LABELS)

    # Format labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(plt_prop.font)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_fontproperties(plt_prop.font)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)


def box_plot(ax, data_plot, plt_prop):
    ax.boxplot(
        data_plot,
        showfliers=True,
        widths=0.75,
        flierprops=FLIER_PROPS,
        boxprops=BOX_PROPS,
        medianprops=MEDIANLINE_PROPS,
    )

    axes_formatting(ax, plt_prop)

    # ax.text(2, 1.145, area_name, fontproperties=plt_prop.font, ha="center")

    # axes_formatting(ax, sbplt)


def read_data(paths):
    logger.info("Loading data...")
    profiles_df = pd.read_csv(paths.validation_csv)
    logger.info("%s", paths.validation_csv)

    areas_info = pd.read_csv(paths.areas_info)
    logger.info("%s", paths.areas_info)

    logger.info("Loading data... Done!")

    return profiles_df, areas_info


def process(paths):
    df, areas_info = read_data(paths)

    test_pred_list = [col for col in df.columns if col.startswith("pred_class_set_")]
    test_conf_list = [
        col for col in df.columns if col.startswith("pred_confidence_set_")
    ]
    for test_pred, test_conf in zip(test_pred_list, test_conf_list):
        correct_df = df[df.idx_in_model == df[test_pred]]
        incorrect_df = df[df.idx_in_model != df[test_pred]]

        logger.info(
            "Plotting boxplots correct vs incorrect classification per label..."
        )

        for area, a_id in zip(areas_info.area, areas_info.area_id):
            _correct = correct_df[correct_df.area_id == a_id]
            _incorrect = incorrect_df[incorrect_df.area_id == a_id]

            logger.info("Area... %s", area)
            logger.info("Area ID... %s", a_id)

            plt_prop = PlotProperties()
            fig, ax = plt.subplots(figsize=plt_prop.cm2inch(C_FIGSIZE))

            box_plot(
                ax, [_correct[test_conf].values, _incorrect[test_conf].values], plt_prop
            )
            prop = dict(left=0.25, right=0.99, top=0.93, bottom=0.1)
            plt.subplots_adjust(**prop)

            set_idx = test_pred[-6:]
            path_fig = os.path.join(paths.output, f"{set_idx}_{area}_[{a_id}].png")
            plt.savefig(path_fig, dpi=C_DPI)
            logger.info("Plot has been saved in... %s\n", path_fig)
            plt.close()
    logger.info("All Done!")
    logger.info("Results saved to... %s", paths.output)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-a",
        "--validation-csv",
        required=True,
        dest="validation_csv",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with labels dataset",
    )

    parser.add_argument(
        "-n",
        "--areas-info",
        required=True,
        dest="areas_info",
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
        help="Path to output ",
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()
    # config = DatasetConfiguration(input_options.config_fname)

    process(input_options)
