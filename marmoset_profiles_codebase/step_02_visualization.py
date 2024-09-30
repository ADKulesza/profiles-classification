import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile

from dataset_configuration import DatasetConfiguration
from plot_methods.plot_formatting import AxesFormatting, AxesFormattingBarPlot
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties

C_DIR_PROFILES_IN_CASE = "profiles_distribution_in_case"
C_DIR_LABELS_IN_CASE = "labels_number_in_case"

C_BARWIDTH = 0.2

C_COLORS = {"all": "#8b8b8b", "accepted": "k", "edge": "white"}  # darker grey  # black

Y_LABEL = "Number of profiles"


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(9), cm2inch(5))


def make_dir(output_dir, case, suffix=""):
    path = os.path.join(output_dir, f"{case}_{suffix}")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def find_axis_step(value):
    thresholds = np.array([1, 40, 100, 500, 1000, 3000, 5000, 10000, 20000, 50000])

    _th = (value - thresholds[:-1]) // thresholds[:-1]
    _th = _th[_th > 0]
    return thresholds[len(_th)]


def plot_bar(ax, x, x_labels, all_profiles, accepted_profiles, xlabel, max_bar_in_plot):
    plt_prop = PlotProperties()
    bar_container_all = ax.bar(
        x - C_BARWIDTH / 2,
        all_profiles,
        edgecolor=C_COLORS["all"],
        color=C_COLORS["edge"],
        linewidth=plt_prop.line_width,
        width=C_BARWIDTH,
        label="all profiles",
    )

    ax.bar(
        x + C_BARWIDTH / 2,
        accepted_profiles,
        edgecolor=C_COLORS["accepted"],
        color=C_COLORS["edge"],
        linewidth=plt_prop.line_width,
        width=C_BARWIDTH,
        label="accepted profiles",
    )

    ax.set_ylabel(Y_LABEL, labelpad=plt_prop.label_pad, fontproperties=plt_prop.font)
    ax.set_xlabel(xlabel, labelpad=plt_prop.label_pad, fontproperties=plt_prop.font)

    ax.xaxis.set_label_coords(0.5 * len(x) / 30 + 0.015, -0.16)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.55, 0.6, 0.5, 0.5),
        prop=plt_prop.font,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.2,
    )

    max_val = max(bar_container_all.datavalues)
    step = find_axis_step(max_val)
    y_axis = {"min": 0, "max": math.ceil(max_val / step) * step, "step": step}

    axes_formatter = AxesFormattingBarPlot(ax)
    axes_formatter.format_axes(y_axis, x, x_labels, max_bar_in_plot)


def plot_distribution_per_section(df, case, case_path, max_bar_in_plot):
    plt_prop = PlotProperties()
    _path = make_dir(case_path, case, C_DIR_PROFILES_IN_CASE)

    x_all_labels = np.sort(np.array((df.section.unique()), dtype=int))
    n_plots = math.ceil(x_all_labels.shape[0] / max_bar_in_plot)

    plot_df = df.groupby(["section", "accept"]).size()
    acc_df = plot_df.iloc[plot_df.index.get_level_values("accept")]
    rec_df = plot_df.iloc[~plot_df.index.get_level_values("accept")]
    for i in range(n_plots):
        logger.info("Profile distrib per section %s/%s", i, n_plots)
        fig, ax = plt.subplots(figsize=C_FIGSIZE)
        # = plot_df.iloc[i*max_bar_in_plot*2:(i+1)*max_bar_in_plot*2]
        x_labels = x_all_labels[i * max_bar_in_plot : (i + 1) * max_bar_in_plot]

        _acc_df = acc_df.iloc[acc_df.index.get_level_values("section").isin(x_labels)]
        _rec_df = rec_df.iloc[rec_df.index.get_level_values("section").isin(x_labels)]
        if _acc_df.shape[0] == _rec_df.shape[0]:
            _acc = _acc_df.values
            _rec = _rec_df.values
        elif _acc_df.shape[0] > _rec_df.shape[0]:
            _acc = _acc_df.values
            _rec = np.zeros(_acc_df.shape[0], dtype=int)
            _sec = _rec_df.index.get_level_values("section")
            _idx = _sec % np.min(_acc_df.index.get_level_values("section"))
            _rec[_idx] = _rec_df.values
        else:
            _rec = rec_df.values
            _acc = np.zeros(_rec_df.shape[0], dtype=int)
            _sec = _acc_df.index.get_level_values("section")
            _idx = _sec % np.min(_rec_df.index.get_level_values("section"))
            _acc[_idx] = _acc_df.values

        _all = _rec + _acc

        x = np.arange(1, _acc.shape[0] + 1)

        plot_bar(ax, x, x_labels, _all, _acc, "section", max_bar_in_plot)

        title = f"case: {case}, section: {x_labels[0]}-{x_labels[-1]}"
        ax.set_title(title, fontproperties=plt_prop.font)

        plt.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.18)

        path_fig = os.path.join(_path, f"{case}_{x_labels[0]}-{x_labels[-1]}.png")
        plt.savefig(path_fig, dpi=plt_prop.dpi)
        plt.close()


def plot_label_distribution(df, case, case_path, max_bar_in_plot):
    plt_prop = PlotProperties()
    _path = make_dir(case_path, case, C_DIR_LABELS_IN_CASE)

    df = df[df.area_id != 0]
    x_all_labels = np.sort(np.array((df.area_id.unique()), dtype=int))
    n_plots = math.ceil(x_all_labels.shape[0] / max_bar_in_plot)

    plot_df = df.groupby(["area_id", "accept"]).size()

    acc_df = plot_df.iloc[plot_df.index.get_level_values("accept")]
    rec_df = plot_df.iloc[~plot_df.index.get_level_values("accept")]

    for i in range(n_plots):
        logger.info("Area distrib  %s/%s", i + 1, n_plots)

        x_labels = x_all_labels[i * max_bar_in_plot : (i + 1) * max_bar_in_plot]

        _acc_df = acc_df.iloc[acc_df.index.get_level_values("area_id").isin(x_labels)]
        _rec_df = rec_df.iloc[rec_df.index.get_level_values("area_id").isin(x_labels)]
        if _acc_df.shape[0] == _rec_df.shape[0]:
            _acc = _acc_df.values
            _rec = _rec_df.values

        else:
            label_order = dict(zip(x_labels, np.arange(x_labels.shape[0], dtype=int)))

            _acc = np.zeros(max_bar_in_plot, dtype=int)
            _rec = np.zeros(max_bar_in_plot, dtype=int)

            _acc_areas = _acc_df.index.get_level_values("area_id")
            _rec_areas = _rec_df.index.get_level_values("area_id")

            _acc_idx = np.vectorize(lambda x: label_order.get(x))(_acc_areas)
            _acc[_acc_idx] = _acc_df.values

            _rec_idx = np.vectorize(lambda x: label_order.get(x))(_rec_areas)
            _rec[_rec_idx] = _rec_df.values

        _all = _rec + _acc

        x = np.arange(1, _acc.shape[0] + 1)

        fig, ax = plt.subplots(figsize=C_FIGSIZE)

        plot_bar(ax, x, x_labels, _all, _acc, "area_id", max_bar_in_plot)
        title = f"case: {case}, area_id: {x_labels[0]}-{x_labels[-1]}"
        ax.set_title(title, fontproperties=plt_prop.font)

        plt.subplots_adjust(left=0.15, right=0.98, top=0.9, bottom=0.18)

        path_fig = os.path.join(_path, f"{case}_{x_labels[0]}-{x_labels[-1]}.png")
        plt.savefig(path_fig, dpi=plt_prop.dpi)
        plt.close()


def get_hist_plot(df, case_path):
    plt_prop = PlotProperties()
    plot_df = df.groupby(["area_id", "accept"]).size()

    acc_df = plot_df.iloc[plot_df.index.get_level_values("accept")]
    hist_val = acc_df.values
    val_median = np.median(hist_val)
    val_q1 = scoreatpercentile(hist_val, 25)
    val_q3 = scoreatpercentile(hist_val, 75)
    fig, ax = plt.subplots(figsize=C_FIGSIZE)

    counts, bins, patches = ax.hist(
        hist_val,
        bins=60,
        edgecolor=C_COLORS["accepted"],
        color=C_COLORS["edge"],
        linewidth=plt_prop.line_width,
    )

    ax.axvline(val_q1, color="silver", linewidth=plt_prop.line_width)
    ax.axvline(val_median, color="gray", linewidth=plt_prop.line_width)
    ax.axvline(val_q3, color="dimgray", linewidth=plt_prop.line_width)

    x_axis = {"min": 0, "max": math.ceil(bins[-1] / 50000) * 50000, "step": 50000}
    y_axis = {"min": 0, "max": math.ceil(counts[0] / 10) * 10, "step": 10}
    axes_formatter = AxesFormatting(ax)
    axes_formatter.format_axes(x_axis, y_axis)

    axes_formatter = AxesFormatting(ax)
    axes_formatter.format_axes(x_axis, y_axis)

    ax.set_title(
        "q1 = {:.0f}, median = {:.0f}, q3 = {:.0f}".format(val_q1, val_median, val_q3),
        fontproperties=plt_prop.font,
    )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)

    path_fig = os.path.join(case_path, f"area_distribution.png")
    plt.savefig(path_fig, dpi=plt_prop.dpi)
    plt.close()


def process(config, paths):
    df = pd.read_csv(paths.profiles_csv)
    for case in config("case_list"):
        logger.info("Case... %s", case)
        df_case = df[df.case == case]
        plot_distribution_per_section(
            df_case, case, paths.output, config("max_bar_in_plot")
        )

        plot_label_distribution(df, case, paths.output, config("max_bar_in_plot"))

    get_hist_plot(df, paths.output)


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
    process(data_settings, input_options)
