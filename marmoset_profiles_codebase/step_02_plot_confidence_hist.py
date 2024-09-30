import argparse
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration
from plot_methods.plot_formatting import (
    AxesFormattingProfileSampleHorizontal, LogHistAxesFormatting)
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties

C_COLORS = {"bar": "k", "edge": "white"}

Y_LABEL_HIST = "Liczność profili korowych"
X_LABEL_HIST = "Przedziały ufności przypisania etykiety\ndla danego profilu korowego"

C_THICKNESS_DOMAIN = np.linspace(0.0, 1.0, 128)
Y_AXIS = {"min": 0, "max": 1, "step": 1}

X_LABEL_PROF = "Intensywność\nwokseli"
Y_LABEL_PROF = "Znormalizowana długość"


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(10), cm2inch(5))


def get_hist_plot(df, prof_arr, case_path):
    plt_prop = PlotProperties()
    df = df[df.area_id != 0]

    # Example profile indices (to be adjusted as needed)
    prof_idx = [140314]
    plot_prof = prof_arr[prof_idx, :].reshape((1, 128))
    conf_arr = df["confidence"].array

    fig, axs = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [4, 1]}, figsize=C_FIGSIZE
    )

    counts, bins, p = axs[0].hist(
        conf_arr,
        bins=np.arange(0, 1 + 0.05, 0.05),
        edgecolor=C_COLORS["bar"],
        color=C_COLORS["edge"],
        linewidth=plt_prop.line_width,
        log=True,
        cumulative=True,
    )

    axs[0].axvline(0.67, color="silver", linewidth=plt_prop.line_width)

    x_axis = {"min": 0, "max": 1, "step": 0.2}
    y_axis = {"min": 1, "max": math.ceil(counts[-1] / 10) * 10, "step": 10}
    axes_formatter = LogHistAxesFormatting(axs[0])
    axes_formatter.format_axes(x_axis, y_axis)

    axs[0].set_xlabel(X_LABEL_HIST, fontproperties=plt_prop.font)
    axs[0].set_ylabel(Y_LABEL_HIST, fontproperties=plt_prop.font)

    # profile plot

    x_axis = {"min": 140, "max": 240, "step": 100}

    profile = plot_prof[0, :]

    axs[1].plot(
        profile,
        C_THICKNESS_DOMAIN,
        linestyle="-",
        linewidth=plt_prop.line_width,
        color="blue",
    )

    axes_formatter = AxesFormattingProfileSampleHorizontal(axs[1])
    axes_formatter.format_axes(x_axis, Y_AXIS)

    _row = df[df.index_in_npy_array == prof_idx[0]]
    print("confidence", _row.confidence.iloc[0], _row.area_id.iloc[0])

    conf_idx = int(_row.confidence * 128)

    axs[1].plot(
        profile[conf_idx:],
        C_THICKNESS_DOMAIN[conf_idx:],
        linestyle="-",
        linewidth=plt_prop.line_width,
        color="red",
    )
    axs[1].text(
        np.min(profile) * 0.9,
        -0.1,
        str(_row.confidence.iloc[0]),
        ha="right",
        va="top",
        fontproperties=plt_prop.font,
    )

    axs[1].patch.set_alpha(0)

    axs[-1].set_xlabel(X_LABEL_PROF, fontproperties=plt_prop.font)

    axs[1].set_ylabel(Y_LABEL_PROF, fontproperties=plt_prop.font)

    plt.subplots_adjust(
        left=0.1, right=0.95, top=0.85, bottom=0.23, wspace=-0.1, hspace=0.1
    )

    path_fig = os.path.join(case_path, f"confidence_distribution.png")
    plt.savefig(path_fig, dpi=plt_prop.dpi)
    plt.savefig(path_fig[:-3] + ".svg", dpi=plt_prop.dpi)
    plt.close()


def process(config, paths):
    df = pd.read_csv(paths.profiles_csv)
    prof_arr = np.load(paths.profiles_npy)
    get_hist_plot(df, prof_arr, paths.output)


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
    ),

    parser.add_argument(
        "-p",
        "--input-profiles-csv",
        required=True,
        dest="profiles_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-n",
        "--input-profiles-npy",
        required=True,
        dest="profiles_npy",
        type=str,
        metavar="FILENAME",
        help="Path to npy file with accepted profiles",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    logger = get_logger("step_02_plots")

    input_options = parse_args()
    data_settings = DatasetConfiguration(input_options.config_fname)

    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Arial"
    mpl.rcParams["svg.fonttype"] = "none"

    process(data_settings, input_options)
