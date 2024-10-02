import argparse
import os

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator

from plot_methods.plot_formatting import AxesFormattingVerticalBarhPlot
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties
from read_json import read_json
from sort_map import get_sort_map

# TODO
C_DPI = 300

C_DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
C_DEFAULT_FONT_SIZE = 8
C_DEFAULT_FONT_PROP = font_manager.FontProperties(
    family="Arial", size=C_DEFAULT_FONT_SIZE
)
C_LABEL_FONT_SIZE = 6
C_LABEL_FONT_PROP = font_manager.FontProperties(family="Arial", size=C_LABEL_FONT_SIZE)

LINEWIDTH = 1

BARWIDTH = 0.6
LABELPAD = 1
TICKS_LEN = 2
TICKS_PAD = 0.5

REL_X_AXIS = {"min": 0.0, "max": 1, "step": 0.25}  # values for ticks
TRICK_X_AXIS = {"min": 0, "max": 0.4, "step": 0.1}

# Whisker lengths
WIDTH_B = [2.4, 1.65, 2.9, 3.2, 2.9, 2.4, 2.45, 4.4, 4.4, 4.2, 3.5, 5.7, 5.2]

LEFT_PLOT_IDX = 63  # max index of area in left plot
RIGHT_UP_PLOT_IDX = 116  # max index of area in upper right plot
AREA_N = 116


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(9), cm2inch(15))

C_LOGGER_NAME = "metrics"


def set_grid(ax, left_y_range, up_right_y_range):
    for x in range(25, 125, 25):
        ax["left"].plot(
            [x / 100, x / 100],
            [0.5, left_y_range[-1] + 0.3],
            color="gray",
            linewidth=0.25 * LINEWIDTH,
            ls="--",
        )
        ax["right"].plot(
            [x / 100, x / 100],
            [0.5, up_right_y_range[-1] + 0.3],
            color="gray",
            linewidth=0.25 * LINEWIDTH,
            ls="--",
        )


def barh_plot(density_list, colors, order, metric, labels_y, grid, output_dir):
    gs_kw = dict(width_ratios=[1, 1])
    fig, ax = plt.subplot_mosaic(
        [["left", "right"]],
        gridspec_kw=gs_kw,
        figsize=C_FIGSIZE,
        constrained_layout=False,
    )

    left_y_range = np.arange(1, LEFT_PLOT_IDX + 1)
    ax["left"].barh(
        left_y_range,
        np.array(density_list[AREA_N - LEFT_PLOT_IDX:]) - REL_X_AXIS["min"],
        color=colors[AREA_N - LEFT_PLOT_IDX:],
        linewidth=LINEWIDTH,
        height=BARWIDTH,
    )
    ax["left"].set_zorder(100)
    ax["left"].set_facecolor("none")

    up_right_y_range = np.arange(1, RIGHT_UP_PLOT_IDX - LEFT_PLOT_IDX + 1)
    ax["right"].barh(
        up_right_y_range,
        np.array(density_list[-RIGHT_UP_PLOT_IDX:-LEFT_PLOT_IDX]) - REL_X_AXIS["min"],
        color=colors[-RIGHT_UP_PLOT_IDX:-LEFT_PLOT_IDX],
        linewidth=LINEWIDTH,
        height=BARWIDTH,
    )

    if grid is True:
        set_grid(ax, left_y_range, up_right_y_range)

    text_posy = LEFT_PLOT_IDX + 1
    text_posy_1 = RIGHT_UP_PLOT_IDX - LEFT_PLOT_IDX + 1
    a_i = 0
    for key, items in order[2].items():

        # whisker length
        widthB = WIDTH_B[a_i]

        # left plot annotations
        if a_i < 8:
            posx = text_posy - len(items) / 2 - 0.5
            text_posy -= len(items)
            ax_name = "left"

        # right plot annotations
        else:
            posx = text_posy_1 - len(items) / 2 - 0.5
            text_posy_1 -= len(items)
            ax_name = "right"

        ax[ax_name].annotate(
            f"{key[:2]}",
            fontproperties=C_DEFAULT_FONT_PROP,
            xy=(-0.21, posx),
            xycoords="data",
            xytext=(-0.235, posx),
            textcoords="data",
            verticalalignment="center",
            horizontalalignment="right",
            arrowprops=dict(
                arrowstyle=f"-[, widthB={widthB},lengthB=0.1, angleB=0",
                connectionstyle="arc3, rad=0.0",
                shrinkA=1,
                shrinkB=3,
            ),
        )

        a_i += 1

    ax["right"].set_xlabel(f"{metric} score", fontproperties=C_DEFAULT_FONT_PROP, x=0.6)

    y_left = {"min": 1, "max": LEFT_PLOT_IDX, "step": 1}

    x_majors = np.arange(
        REL_X_AXIS["min"],
        REL_X_AXIS["max"] + REL_X_AXIS["step"] / 2,
        REL_X_AXIS["step"],
    )
    labels_x = map(lambda x: "0" if x == 0 else "{:.1f}".format(x), x_majors)

    axes_formatter_left = AxesFormattingVerticalBarhPlot(ax["left"])
    axes_formatter_left.format_axes(
        TRICK_X_AXIS, y_left, labels_x, labels_y[AREA_N - LEFT_PLOT_IDX:]
    )

    y_right = {"min": 1, "max": 53, "step": 1}
    axes_formatter_right = AxesFormattingVerticalBarhPlot(ax["right"])
    axes_formatter_right.format_axes(
        TRICK_X_AXIS, y_right, labels_x, labels_y[-RIGHT_UP_PLOT_IDX:-LEFT_PLOT_IDX]
    )

    # x_axis, y_axis, y_labels y_t = np.arange(1, len(labelsy) + 1)

    # axes_formatting(ax['left'], labels_y[AREA_N - LEFT_PLOT_IDX:], 0)
    # axes_formatting(ax['right'], labels_y[-RIGHT_UP_PLOT_IDX:-LEFT_PLOT_IDX], 1)
    prop = dict(left=0.1, right=0.97, top=0.97, bottom=0.05, wspace=0.2, hspace=0.3)
    plt.subplots_adjust(**prop)

    plt.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=C_DPI)
    plt.savefig(os.path.join(output_dir, f"{metric}.svg"), dpi=C_DPI)


def read_data(paths):
    logger.info("Loading data...")
    metrics_df = pd.read_csv(paths.area_metrics)
    logger.info("%s", paths.area_metrics)

    logger.info("Loading data... Done!")

    return metrics_df


def process(paths, order, grid=False):
    metrics_df = read_data(paths)

    labels_y = metrics_df.area.array
    labels_y = np.flip(labels_y)

    gb = metrics_df.groupby("region", sort=False)

    metrics_list = ["accuracy", "recall", "precision", "f1"]

    plt_prop = PlotProperties()
    for metric in metrics_list:
        m_values = []
        colors = []
        for k, gp in gb:

            for den, r, g, b in zip(
                    list(gp[metric]),
                    list(gp["color_r"]),
                    list(gp["color_g"]),
                    list(gp["color_b"]),
            ):
                m_values.append(den)
                colors.append([r / 255, g / 255, b / 255])

        barh_plot(m_values, colors, order, metric, labels_y, grid, paths.output_dir)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--area-metrics-csv",
        required=True,
        dest="area_metrics",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with",
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

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()

    order_fname = "areas_order.json"
    area_order = read_json(order_fname)
    process(input_options, area_order)
