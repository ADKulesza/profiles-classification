import argparse
import logging
import os

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration

C_LOGGER_NAME = "CAM_plot"
logging.basicConfig(
    level=getattr(logging, "DEBUG"),
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(C_LOGGER_NAME)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True
logging.getLogger("matplotlib").disabled = True
logging.getLogger("matplotlib.colorbar").disabled = True
logging.getLogger("PIL").setLevel(logging.WARNING)

# Plot properties
C_DPI = 300

C_DEFAULT_FONT_PATH = "Arial"
C_DEFAULT_FONT_SIZE = 8
C_DEFAULT_FONT_PROP = font_manager.FontProperties(
    family=C_DEFAULT_FONT_PATH, size=C_DEFAULT_FONT_SIZE
)

C_TITLE_FONT_PROP = font_manager.FontProperties(fname=C_DEFAULT_FONT_PATH, size=6)

C_LINEWIDTH = 1
C_BARWIDTH = 0.2

LABELPAD = 1
TICKS_LEN = 2.5
TICKS_PAD = 0.5

# And some other settings:
C_PROFILE_LINECOLOR = "#555555"
C_PROFILE_MEDIAN_LINECOLOR = "#000000"
C_PROFILE_MEDIAN_LW = 1.0
C_PROFILE_LW = 0.5

Y_AXIS = {"min": 0, "max": 1, "step": 0.2}


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(5), cm2inch(4))


def axes_formatting(ax, x_axis):
    """ """
    y_majors = np.arange(Y_AXIS["min"], Y_AXIS["max"] + Y_AXIS["step"], Y_AXIS["step"])

    x_majors = np.arange(x_axis["min"], x_axis["max"] + x_axis["step"], x_axis["step"])
    x_majors[1:] = x_majors[1:] - 1
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
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_bounds(x_axis["min"], x_majors[-1])
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["left"].set_position(("data", x_axis["min"]))

    ax.set_yticks(y_majors[1:])
    y_labels = map(lambda x: "{:.1f}".format(x), y_majors[1:])
    ax.set_yticklabels(y_labels)
    ax.set_ylim(Y_AXIS["min"], Y_AXIS["max"])
    # ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_xticks(x_majors)
    # x_labels = map(lambda x: "{:.0f}".format(x), x_majors)
    x_labels = np.arange(0, 128 + 16, 32)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(x_axis["min"], x_axis["max"])
    # ax.xaxis.set_minor_locator(AutoMinorLocator(1))

    for label in ax.get_xticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)

    for label in ax.get_yticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)


def get_stat(x_arr):
    median = np.median(x_arr, axis=0)
    q1 = np.percentile(x_arr, 25, axis=0)
    q3 = np.percentile(x_arr, 75, axis=0)
    # iqr = q3 - q1

    stat = np.zeros((3, x_arr.shape[1]))
    # stat[0] = q1 - 1.5 * iqr
    stat[0] = q1
    stat[1] = median
    stat[2] = q3
    # stat[4] = q3 + 1.5 * iqr

    return stat


def get_plot(stat, output_path, area_name):
    fig, ax = plt.subplots(figsize=C_FIGSIZE)

    # profile_corridor = [stat[1] - stat[0], stat[1] + stat[2]]
    x_axis = {"min": 0, "max": stat.shape[1] - 1, "step": 8}
    domain = np.arange(stat.shape[1])

    ax.fill_between(
        domain,
        stat[0],
        stat[2],
        facecolor="#eeeeee",
        edgecolor=None,
        linewidth=None,
        label="IQR",
    )

    for series in [stat[0], stat[2]]:
        ax.plot(
            domain,
            series,
            linestyle="-",
            linewidth=C_PROFILE_LW,
            color=C_PROFILE_LINECOLOR,
        )

    ax.plot(
        domain,
        stat[1],
        linestyle="-",
        linewidth=C_PROFILE_MEDIAN_LW,
        color=C_PROFILE_MEDIAN_LINECOLOR,
        label="median",
    )

    axes_formatting(ax, x_axis)

    ax.legend(
        prop=C_DEFAULT_FONT_PROP,
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(1.02, 1.05),
        labelspacing=0.2,
        handlelength=1.25,
        handleheight=0.5,
        handletextpad=0.25,
    )

    plt.title(area_name, fontproperties=C_DEFAULT_FONT_PROP)

    ax.set_ylabel("Normalized Grad-CAM++", fontproperties=C_DEFAULT_FONT_PROP)

    ax.set_xlabel("Profile length", fontproperties=C_DEFAULT_FONT_PROP)

    prop = dict(left=0.18, right=0.95, top=0.92, bottom=0.18)

    plt.subplots_adjust(**prop)
    plt.savefig(output_path, dpi=C_DPI)
    plt.savefig(output_path.replace(".png", ".svg"), dpi=C_DPI)
    plt.close()


def process(paths):
    profiles_df = pd.read_csv(paths.output_csv)

    heatmaps = np.load(paths.heatmaps)
    logger.info(heatmaps)
    area_id_list = np.array(profiles_df.area_id.unique())

    n_blocks = 32

    for _area_id in area_id_list:

        _df = profiles_df[profiles_df.area_id == _area_id]
        array_idx = np.array(_df.index_in_npy_array)
        if array_idx.shape[0] == 0:
            continue
        _heatmaps = np.zeros((array_idx.shape[0], 128))
        _heatmaps[:] = heatmaps[array_idx, :]

        # _heatmaps = np.delete(_heatmaps, _heatmaps == 0, axis=0)
        corrupted_idx = np.where(~_heatmaps.any(axis=1))[0]
        if corrupted_idx.shape[0] > 0:
            _heatmaps = np.delete(_heatmaps, corrupted_idx, axis=0)
        if _heatmaps.shape[0] == 0:
            continue
        _heatmaps = np.reshape(_heatmaps, (-1, n_blocks), order="F")
        logger.info("dupa: %s", _heatmaps.shape)
        stat_hm = get_stat(_heatmaps)
        area_name = _df.area.values[0]
        logger.info("%s...", area_name)

        fname_hm = os.path.join(paths.output_dir, area_name + ".png")
        get_plot(stat_hm, fname_hm, area_name)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config-fname",
        required=True,
        dest="config_fname",
        type=str,
        metavar="FILENAME",
        help="Path to file with configuration",
    )

    parser.add_argument(
        "-l",
        "--label-names",
        required=True,
        dest="label_names",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with",
    )

    parser.add_argument(
        "-o",
        "--output-csv",
        required=True,
        dest="output_csv",
        type=str,
        metavar="FILENAME",
        help="Path to output csv file with",
    )

    parser.add_argument(
        "-i",
        "--heatmaps",
        required=True,
        dest="heatmaps",
        type=str,
        metavar="FILENAME",
        help="Path to ",
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
    plt.rcParams["axes.titley"] = 0.95
    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)

    process(input_options)
