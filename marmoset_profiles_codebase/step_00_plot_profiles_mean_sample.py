import argparse
import os
from math import ceil, floor

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import scoreatpercentile

from plot_methods.plot_logger import get_logger

C_LOGGER_NAME = "profile_plot"

C_EXPORT_DPI = 300

C_DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
C_DEFAULT_FONT_SIZE = 8
C_DEFAULT_FONT_PROP = font_manager.FontProperties(
    fname=C_DEFAULT_FONT_PATH, size=C_DEFAULT_FONT_SIZE
)

LABELPAD = 1
TICKS_LEN = 2.5
TICKS_PAD = 0.5

Y_AXIS = {"min": 0, "max": 1, "step": 1}

C_YLIM = [-0.02, 1.05]
C_THICKNESS_DOMAIN = np.linspace(0.0, 1.0, 128)

# And some other settings:
C_PROFILE_LINECOLOR = "#555555"
C_PROFILE_MEDIAN_LINECOLOR = "#000000"
C_PROFILE_MEDIAN_LW = 1.0
C_PROFILE_LW = 0.5


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(2), cm2inch(4))


def rescale_plot(plot_val, new_max, new_min):
    """
    Returns rescaled values of the plot
    Input: plot_val - array with plot values,
    new_max - a maximum value in the new range of the plot
    new_min - a minimum value in the new range of the plot
    Output: rescaled_values - rescaled array with plot values
    """

    min_value = np.min(plot_val)
    max_value = np.max(plot_val)

    old_range = max_value - min_value
    new_range = new_max - new_min

    rescaled_values = (plot_val - min_value) / old_range * new_range

    return rescaled_values


def axes_formatting(ax, x_axis):
    """ """
    y_majors = np.arange(Y_AXIS["min"], Y_AXIS["max"] + Y_AXIS["step"], Y_AXIS["step"])

    x_majors = np.arange(x_axis["min"], x_axis["max"] + x_axis["step"], x_axis["step"])

    ax.tick_params(
        axis="x",
        which="both",
        pad=TICKS_PAD,
        size=TICKS_LEN,
        bottom=True,
        top=False,
        labeltop=False,
        labelbottom=True,
        labelsize=C_DEFAULT_FONT_SIZE,
    )

    ax.tick_params(
        axis="y",
        pad=TICKS_PAD,
        size=TICKS_LEN,
        which="major",
        right=False,
        labelright=False,
        left=True,
        labelleft=True,
        labelsize=C_DEFAULT_FONT_SIZE,
    )

    ax.tick_params(
        axis="x",
        pad=TICKS_PAD,
        size=TICKS_LEN / 2,
        which="minor",
        labelsize=C_DEFAULT_FONT_SIZE,
    )

    ax.tick_params(
        axis="y",
        pad=TICKS_PAD,
        size=TICKS_LEN / 2,
        which="minor",
        left=True,
        labelleft=True,
        labelsize=C_DEFAULT_FONT_SIZE,
    )

    ax.xaxis.labelpad = LABELPAD
    ax.yaxis.labelpad = LABELPAD

    # Make invisible left and top axis:
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_bounds(x_axis["min"], x_majors[-1])
    ax.spines["bottom"].set_position(("data", 1.05))
    ax.spines["right"].set_bounds(0, 1)
    ax.spines["right"].set_position(("data", x_axis["min"] - 5))

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.set_yticks(y_majors)
    ax.set_yticklabels([0, 1])
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_ylim(C_YLIM)
    ax.invert_yaxis()

    ax.set_xticks(x_majors)

    ax.set_xlim(x_axis["min"], x_axis["max"])
    ax.xaxis.set_minor_locator(AutoMinorLocator((x_majors[-1] - x_majors[0]) // 10))

    for label in ax.get_xticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)

    for label in ax.get_yticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)

    ax.invert_xaxis()


def plot_profile(profiles_arr, fig_path):
    fig, ax = plt.subplots(1, 1, figsize=C_FIGSIZE)

    profiles_stat = scoreatpercentile(profiles_arr, per=[5, 25, 50, 75, 95], axis=0)

    pcdx, pcd, profile_median, pcu, pcux = profiles_stat
    x_axis = {
        "min": floor(np.min(profile_median) / 10) * 10,
        "max": ceil(np.max(profile_median) / 10) * 10,
        "step": 20,
    }
    x_axis["step"] = x_axis["max"] - x_axis["min"]

    ax.fill_betweenx(
        C_THICKNESS_DOMAIN,
        pcux,
        pcdx,
        facecolor="#eeeeee",
        edgecolor=None,
        linewidth=None,
    )

    ax.fill_betweenx(
        C_THICKNESS_DOMAIN,
        pcu,
        pcd,
        facecolor="#cccccc",
        edgecolor=None,
        linewidth=None,
    )

    ax.plot(
        profile_median,
        C_THICKNESS_DOMAIN,
        linestyle="-",
        linewidth=C_PROFILE_MEDIAN_LW,
        color=C_PROFILE_MEDIAN_LINECOLOR,
    )

    # for series in [pcux, pcdx, pcu, pcd]:
    #     ax.plot(series, C_THICKNESS_DOMAIN,
    #             linestyle="-",
    #             linewidth=C_PROFILE_LW,
    #             color=C_PROFILE_LINECOLOR)

    axes_formatting(ax, x_axis)

    # ax.set_xlabel("intensity",
    #               fontproperties=C_DEFAULT_FONT_PROP)
    #
    # ax.set_ylabel("normalized thickness",
    #               fontproperties=C_DEFAULT_FONT_PROP)

    ax.set_xlabel("intensywność", fontproperties=C_DEFAULT_FONT_PROP)

    ax.set_ylabel("znormalizowana długość", fontproperties=C_DEFAULT_FONT_PROP)

    prop = dict(left=0.1, right=0.7, top=0.97, bottom=0.15)
    plt.subplots_adjust(**prop)
    plt.savefig(fig_path + ".png", dpi=C_EXPORT_DPI)
    plt.savefig(fig_path + ".svg", dpi=C_EXPORT_DPI)
    logger.info("Figure saved to... %s", fig_path)


def process(paths):
    profiles_df = pd.read_csv(paths.all_profiles_csv)

    section = paths.section

    profiles_df = profiles_df[profiles_df.section == section]
    profiles_range = np.arange(paths.start_id, paths.end_id)

    profiles_to_plot = profiles_df[profiles_df.profile_id.isin(profiles_range)]
    profiles = np.load(paths.profiles_npy)
    profiles = profiles[profiles_to_plot.index_in_npy_array]
    for case in pd.unique(profiles_df.case):
        os.makedirs(os.path.join(paths.output_dir, case), exist_ok=True)
        _df = profiles_to_plot.iloc[0]

        fig_name = "{}_sec-{}_area_{}_{}-{}".format(
            case, _df.section, _df.area_id, paths.start_id, paths.end_id
        )
        # logger.info("Processing %s/%s", i_plot, n_plots)
        fig_fname = os.path.join(paths.output_dir, case, fig_name)
        plot_profile(profiles, fig_fname)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--input-profiles-npy",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to npy file with accepted profiles",
    )

    parser.add_argument(
        "-n",
        "--all-profiles-csv",
        required=True,
        dest="all_profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to npy file with accepted profiles",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        dest="output_dir",
        type=str,
        metavar="FILENAME",
        help="Path to output directory",
    )

    parser.add_argument(
        "-s",
        "--section",
        required=True,
        default=None,
        dest="section",
        type=int,
        metavar="DECIMAL",
        help="Section number",
    )

    parser.add_argument(
        "-r",
        "--start-profile-id",
        required=False,
        default=1,
        dest="start_id",
        type=int,
        metavar="DECIMAL",
        help="Start profile id",
    )

    parser.add_argument(
        "-t",
        "--end-profile-id",
        required=False,
        default=1,
        dest="end_id",
        type=int,
        metavar="DECIMAL",
        help="End profile id",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()
    process(input_options)
