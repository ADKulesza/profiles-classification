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

TRICK_X_AXIS = {"min": 0.5, "max": 1, "step": 0.1}
STEP_NUM = (TRICK_X_AXIS["max"] - TRICK_X_AXIS["min"]) / TRICK_X_AXIS["step"]

REL_X_AXIS = {"min": 0.0, "max": 1}  # values for ticks
REL_STEP = (REL_X_AXIS["max"] - REL_X_AXIS["min"]) / STEP_NUM
REL_X_AXIS["step"] = REL_STEP

# Whisker lengths
WIDTH_B = [2.4, 1.65, 2.9, 3.2, 2.9, 2.4, 2.45, 3.8, 3.8, 3.3, 2.8, 4.9, 4.4]

LEFT_PLOT_IDX = 63  # max index of area in left plot
AREA_N = 116  # max index of area in upper right plot
RIGHT_PLOT_IDX = 53


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

    y_range = np.arange(1, LEFT_PLOT_IDX + 1)
    ax["left"].barh(
        y_range,
        np.array(density_list[AREA_N - LEFT_PLOT_IDX:]),
        color=colors[AREA_N - LEFT_PLOT_IDX:],
        linewidth=LINEWIDTH,
        height=BARWIDTH,
    )
    ax["left"].set_zorder(100)
    ax["left"].set_facecolor("none")

    barh_values = np.array(density_list[:RIGHT_PLOT_IDX])

    barh_values = np.concatenate((np.zeros(LEFT_PLOT_IDX - RIGHT_PLOT_IDX), barh_values))
    barh_colors = colors[-AREA_N:-LEFT_PLOT_IDX]

    for _ in range(LEFT_PLOT_IDX - RIGHT_PLOT_IDX):
        barh_colors.insert(0, [1, 1, 1])

    ax["right"].barh(
        y_range,
        barh_values,
        color=barh_colors,
        linewidth=LINEWIDTH,
        height=BARWIDTH,
    )

    if grid is True:
        set_grid(ax, y_range, y_range)

    text_posy = LEFT_PLOT_IDX + 1
    text_posy_1 = AREA_N - RIGHT_PLOT_IDX + 1
    a_i = 0
    for key, items in order[2].items():

        # whisker length
        widthB = WIDTH_B[a_i]

        # left plot annotations
        if a_i < 8:
            posy = text_posy - len(items) / 2 - 0.5
            text_posy -= len(items)
            ax_name = "left"

        # right plot annotations
        else:
            posy = text_posy_1 - len(items) / 2 - 0.5
            text_posy_1 -= len(items)
            ax_name = "right"

        ax[ax_name].annotate(
            f"{key[:2]}",
            fontproperties=C_DEFAULT_FONT_PROP,
            xy=(-0.5, posy),
            xycoords="data",
            xytext=(-0.55, posy),
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
        TRICK_X_AXIS["min"],
        TRICK_X_AXIS["max"] + TRICK_X_AXIS["step"] / 2,
        TRICK_X_AXIS["step"],
    )
    labels_x = map(lambda x: "0" if x == 0 else "{:.1f}".format(x), x_majors)

    span_limits = {
        "bottom": 0.25,
        "left_span": [0.5, LEFT_PLOT_IDX + REL_X_AXIS["step"]]
    }
    axes_formatter_left = AxesFormattingVerticalBarhPlot(ax["left"])
    axes_formatter_left.format_axes(
        REL_X_AXIS, y_left, labels_x, labels_y[AREA_N - LEFT_PLOT_IDX:],
        span_limits
    )

    right_y_labels = labels_y[-AREA_N:-LEFT_PLOT_IDX]
    right_y_labels = np.concatenate((np.full(LEFT_PLOT_IDX - RIGHT_PLOT_IDX, ""), right_y_labels))

    y_right = {"min": 1, "max": LEFT_PLOT_IDX, "step": 1}
    span_limits = {
        "bottom": LEFT_PLOT_IDX - RIGHT_PLOT_IDX,
        "left_span": [LEFT_PLOT_IDX - RIGHT_PLOT_IDX + 0.25, LEFT_PLOT_IDX + REL_X_AXIS["step"]]
    }

    axes_formatter_right = AxesFormattingVerticalBarhPlot(ax["right"])
    axes_formatter_right.format_axes(
        REL_X_AXIS, y_right, labels_x, right_y_labels,
        span_limits
    )

    prop = dict(left=0.05, right=0.98, top=0.97, bottom=0.05, wspace=0.05)
    plt.subplots_adjust(**prop)

    plt.savefig(os.path.join(output_dir, f"{metric}_zoom.png"), dpi=C_DPI)
    plt.savefig(os.path.join(output_dir, f"{metric}_zoom.svg"), dpi=C_DPI)


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
