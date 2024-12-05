import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_methods.plot_formatting import AxesFormattingVerticalBarhPlot
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties
from read_json import read_json

C_DPI = 300

BARWIDTH = 0.6

TRICK_X_AXIS = {"min": 0.6, "max": 1, "step": 0.2}
STEP_NUM = (TRICK_X_AXIS["max"] - TRICK_X_AXIS["min"]) / TRICK_X_AXIS["step"]

REL_X_AXIS = {"min": 0.0, "max": 1}  # values for ticks
REL_STEP = (REL_X_AXIS["max"] - REL_X_AXIS["min"]) / STEP_NUM
REL_X_AXIS["step"] = REL_STEP

# Whisker lengths
WIDTH_B = [2.4, 1.65, 2.9, 3.2, 2.9, 2.4, 2.45, 4.4, 3.8, 3.2, 2.9, 4.9, 4.4]

LEFT_PLOT_IDX = 62  # max index of area in left plot
AREA_N = 115  # max index of area in upper right plot
RIGHT_PLOT_IDX = 53


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(9), cm2inch(15))

C_LOGGER_NAME = "metrics"


def barh_plot(metric_list, colors, order, metric,
              labels_y, plt_prop, output_dir, do_svg=False):
    gs_kw = dict(width_ratios=[1, 1])
    fig, ax = plt.subplot_mosaic(
        [["left", "right"]],
        gridspec_kw=gs_kw,
        figsize=C_FIGSIZE,
        constrained_layout=False,
    )

    metric_medians = metric_list[0]
    metric_q1 = metric_list[1]
    metric_q3 = metric_list[2]
    y_range = np.arange(1, LEFT_PLOT_IDX + 1)

    logger.info("SMALLEST Y: %s", min(metric_q1))

    ax["left"].barh(
        y_range,
        np.array(metric_medians[AREA_N - LEFT_PLOT_IDX:]) - REL_X_AXIS["min"],
        color=colors[AREA_N - LEFT_PLOT_IDX:],
        linewidth=plt_prop.line_width,
        height=BARWIDTH,
    )

    ax["left"].barh(
        y_range,
        np.array([TRICK_X_AXIS["min"]] * len(y_range)),
        color=[[1, 1, 1]] * len(y_range),
        linewidth=plt_prop.line_width,
        height=BARWIDTH,
    )

    # ax["left"].errorbar(
    #     x=np.array(metric_medians[AREA_N - LEFT_PLOT_IDX:]) - REL_X_AXIS["min"],
    #     y=y_range,
    #     xerr=[
    #         np.array(metric_medians[AREA_N - LEFT_PLOT_IDX:]) - np.array(metric_q1[AREA_N - LEFT_PLOT_IDX:]),
    #         np.array(metric_q3[AREA_N - LEFT_PLOT_IDX:]) - np.array(metric_medians[AREA_N - LEFT_PLOT_IDX:])
    #     ],
    #     fmt='none',
    #     ecolor='black',
    #     elinewidth=0.6,
    #     capsize=3
    # )

    ax["left"].set_zorder(100)
    ax["left"].set_facecolor("none")

    ax["right"].set_facecolor("none")

    barh_values = np.array(metric_medians[:RIGHT_PLOT_IDX])

    barh_values = np.concatenate((-np.ones(LEFT_PLOT_IDX - RIGHT_PLOT_IDX), barh_values))
    barh_colors = colors[-AREA_N:-LEFT_PLOT_IDX]

    q1_values = np.array(metric_q1[:RIGHT_PLOT_IDX])
    q1_values = np.concatenate((-np.ones(LEFT_PLOT_IDX - RIGHT_PLOT_IDX), q1_values))

    q3_values = np.array(metric_q3[:RIGHT_PLOT_IDX])
    q3_values = np.concatenate((-np.ones(LEFT_PLOT_IDX - RIGHT_PLOT_IDX), q3_values))

    for _ in range(LEFT_PLOT_IDX - RIGHT_PLOT_IDX):
        barh_colors.insert(0, [1, 1, 1])

    ax["right"].barh(
        y_range,
        barh_values,
        color=barh_colors,
        linewidth=plt_prop.line_width,
        height=BARWIDTH,
    )

    ax["right"].barh(
        y_range,
        np.array([TRICK_X_AXIS["min"]] * len(y_range)),
        color=[[1, 1, 1]] * len(y_range),
        linewidth=plt_prop.line_width,
        height=BARWIDTH,
    )

    # ax["right"].errorbar(
    #     x=barh_values,
    #     y=y_range,
    #     xerr=[
    #         barh_values - q1_values,
    #         q3_values - barh_values
    #     ],
    #     fmt='none',
    #     ecolor='black',
    #     elinewidth=0.6,
    #     capsize=3
    # )

    text_posy = LEFT_PLOT_IDX + 1
    text_posy_1 = AREA_N - RIGHT_PLOT_IDX + 1
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
            fontproperties=plt_prop.font,
            xy=(TRICK_X_AXIS["min"] - 0.2, posx),
            xycoords="data",
            xytext=(TRICK_X_AXIS["min"] - 0.25, posx),
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

    y_left = {"min": 1, "max": LEFT_PLOT_IDX, "step": 1}

    axes_formatter_left = AxesFormattingVerticalBarhPlot(ax["left"])
    axes_formatter_left.format_axes(
        TRICK_X_AXIS, y_left, labels_x, labels_y[AREA_N - LEFT_PLOT_IDX:],
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
        TRICK_X_AXIS, y_right, labels_x, right_y_labels,
        span_limits
    )

    ax["right"].set_xlabel(f"{metric} score", fontproperties=plt_prop.font, labelpad=15, x=0.6)

    prop = dict(left=-TRICK_X_AXIS["min"]+0.1, right=0.97, top=0.97, bottom=0.05, wspace=-TRICK_X_AXIS["min"] + 0.1)
    plt.subplots_adjust(**prop)

    plt.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=C_DPI)

    if do_svg:
        plt.savefig(os.path.join(output_dir, f"{metric}.svg"), dpi=C_DPI)


def read_data(paths):
    logger.info("Loading data...")
    metrics_df = pd.read_csv(paths.area_metrics)
    logger.info("%s", paths.area_metrics)

    logger.info("Loading data... Done!")

    return metrics_df


def process(paths, order):
    metrics_df = read_data(paths)
    metrics_df = metrics_df.sort_values(by="label")

    labels_y = metrics_df.area.array
    labels_y = np.flip(labels_y)

    gb = metrics_df.groupby("region", sort=False)

    metrics_list = ["accuracy", "recall", "precision", "f1"]

    plt_prop = PlotProperties()
    for metric in metrics_list:
        m_values = [[], [], []]
        colors = []
        for k, gp in gb:

            for met, _q1, _q3, r, g, b in zip(
                    list(gp[f"{metric}_median"]),
                    list(gp[f"{metric}_q1"]),
                    list(gp[f"{metric}_q3"]),
                    list(gp["color_r"]),
                    list(gp["color_g"]),
                    list(gp["color_b"]),
            ):
                m_values[0].append(met)
                m_values[1].append(_q1)
                m_values[2].append(_q3)
                colors.append([r / 255, g / 255, b / 255])

        barh_plot(m_values, colors,
                  order, metric, labels_y,
                  plt_prop,
                  paths.output_dir, paths.do_svg)


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

    parser.add_argument(
        "-s",
        "--svg",
        required=False,
        action="store_true",
        dest="do_svg",
        help="Do svg plot?",
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()

    order_fname = "areas_order.json"
    area_order = read_json(order_fname)
    process(input_options, area_order)
