import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_configuration import DatasetConfiguration
from plot_methods.plot_formatting import AxesFormattingBarPlot
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties

C_LOGGER_NAME = "visualization"

C_BARWIDTH = 0.2

C_COLORS = {"all": "#8b8b8b", "accepted": "k", "edge": "white"}  # darker grey  # black

Y_LABEL = "Number of profiles"

C_DIR_LABELS_IN_SECTION = "labels_distribution_in_section"
C_DIR_PROFILES_IN_CASE = "profiles_distribution_in_case"
C_DIR_LABELS_IN_CASE = "labels_number_in_case"


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(9), cm2inch(5))


def make_dir(output_dir, case, suffix=""):
    path = os.path.join(output_dir, f"{case}_{suffix}")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def axes_imshow_formatting(ax, y_labels, sections):
    plt_prop = PlotProperties()
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_xticks(np.arange(len(sections))[::5])
    ax.set_xticklabels(sections[::5])
    plt.xticks(rotation=90)

    ax.tick_params(
        axis="y",
        which="major",
        pad=plt_prop.pad,
        length=plt_prop.ticks_len,
        width=1,
        left="off",
        labelleft="off",
        direction="in",
    )

    ax.tick_params(
        axis="x",
        which="major",
        pad=plt_prop.pad,
        length=plt_prop.ticks_len,
        width=1,
        left="off",
        labelleft="off",
        direction="in",
    )

    for label in ax.get_yticklabels():
        label.set_fontproperties(plt_prop.font)

    for label in ax.get_xticklabels():
        label.set_fontproperties(plt_prop.font)


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
    y_axis = {"min": 0, "max": max_val, "step": step}

    axes_formatter = AxesFormattingBarPlot(ax)
    axes_formatter.format(y_axis, x, x_labels, max_bar_in_plot)


def find_axis_step(value):
    thresholds = np.array([1, 40, 100, 500, 1000, 3000, 5000, 10000, 20000, 50000])

    _th = (value - thresholds[:-1]) // thresholds[:-1]
    _th = _th[_th > 0]
    return thresholds[len(_th)]


def plot_profiles_distribution_in_case(case, df, output_dir, max_bar_in_plot):
    plt_prop = PlotProperties()
    case_path = make_dir(output_dir, case, C_DIR_PROFILES_IN_CASE)

    x = np.arange(1, max_bar_in_plot + 1)

    x_all_sections = np.sort(np.array((df.section.unique())))

    df_all = df.groupby(["section", "accept"], dropna=False)
    df_all = df_all.size().unstack(fill_value=0).stack()
    y_all_sections = df_all.sum(level=0)

    df_all = df_all.reset_index()
    df_all.rename(columns={0: "counts"}, inplace=True)

    acc_all_sections = df_all[df_all.accept].counts

    n_plots = math.ceil(x_all_sections.shape[0] / max_bar_in_plot)

    for i in range(n_plots):

        if i < n_plots - 1:
            x_labels = list(
                x_all_sections[i * max_bar_in_plot : (i + 1) * max_bar_in_plot]
            )

            y_all = list(
                y_all_sections[i * max_bar_in_plot : (i + 1) * max_bar_in_plot]
            )
            y_acc = list(
                acc_all_sections[i * max_bar_in_plot : (i + 1) * max_bar_in_plot]
            )
        else:
            # last sections
            x_labels = list(x_all_sections[i * max_bar_in_plot :])
            x = np.arange(1, len(x_labels) + 1)
            y_all = list(y_all_sections[i * max_bar_in_plot :])
            y_acc = list(acc_all_sections[i * max_bar_in_plot :])

        logger.info("[Case: %s]; Sections...%s-%s", case, x_labels[0], x_labels[-1])

        fig, ax = plt.subplots(figsize=C_FIGSIZE)

        plot_bar(ax, x, x_labels, y_all, y_acc, "Section", max_bar_in_plot)

        title = f"case: {case}, section: {x_labels[0]}-{x_labels[-1]}"
        ax.set_title(title, fontproperties=plt_prop.font)

        plt.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.18)

        path_fig = os.path.join(case_path, f"{case}_{x_labels[0]}-{x_labels[-1]}.png")
        plt.savefig(path_fig, dpi=plt_prop.dpi)
        plt.close()


def plot_labels_in_case(case, df, output_dir, max_bar_in_plot):
    plt_prop = PlotProperties()
    case_path = make_dir(output_dir, case, C_DIR_LABELS_IN_CASE)

    x_all_labels = np.sort(np.array((df.label.unique()), dtype=int))
    n_plots = math.ceil(x_all_labels.shape[0] / max_bar_in_plot)
    if x_all_labels[0] == 0:
        n_plots += 1

    df = df[df.area_id != 0]
    # TODO FIX!
    df_all = df.groupby(["label", "accept"], dropna=True).size()
    df_all = df_all.unstack(fill_value=0).stack()
    y_all_labels = np.array(df_all.sum(level=0))

    df_all = df_all.reset_index()
    df_all.rename(columns={0: "counts"}, inplace=True)

    acc_all_labels = np.array(df_all[df_all.accept].counts)

    for i in range(n_plots):
        if i == 0 and x_all_labels[0] == 0:
            x_labels = [x_all_labels[0]]
            x_all_labels = x_all_labels[1:]

            y_all = [y_all_labels[0]]
            y_all_labels = y_all_labels[1:]

            y_acc = [acc_all_labels[0]]
            acc_all_labels = acc_all_labels[1:]
            x = np.arange(1, 2, dtype=int)

        elif i == n_plots - 1:
            # last sections
            x_labels = list(x_all_labels[i * max_bar_in_plot :])
            if len(x_labels) == 0:
                break
            x = np.arange(1, len(x_labels) + 1)
            y_all = list(y_all_labels[(i - 1) * max_bar_in_plot :])
            y_acc = list(acc_all_labels[(i - 1) * max_bar_in_plot :])

        else:
            x_labels = list(
                x_all_labels[(i - 1) * max_bar_in_plot : i * max_bar_in_plot]
            )
            y_all = list(y_all_labels[(i - 1) * max_bar_in_plot : i * max_bar_in_plot])
            y_acc = list(
                acc_all_labels[(i - 1) * max_bar_in_plot : i * max_bar_in_plot]
            )
            x = np.arange(1, max_bar_in_plot + 1, dtype=int)

        fig, ax = plt.subplots(figsize=C_FIGSIZE)

        plot_bar(ax, x, x_labels, y_all, y_acc, "Label", max_bar_in_plot)

        if len(x_labels) == 0:
            x_labels.append(0)
        title = f"case: {case}, labels: {x_labels[0]}-{x_labels[-1]}"
        ax.set_title(title, fontproperties=plt_prop.font)

        logger.info("[Case: %s]; Sections...%s-%s", case, x_labels[0], x_labels[-1])
        path_fig = os.path.join(case_path, f"{case}_{x_labels[0]}-{x_labels[-1]}.png")

        plt.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.18)
        plt.savefig(path_fig, dpi=plt_prop.dpi)
        plt.close()


# def imshow_distribution_labels_in_section(case, df_case, y_labels, output_dir):
#     _df_label = df_case.groupby(["section", "area_id"]).size()
#     _df_label = _df_label.reset_index()
#     logger.info("AAAA %s", _df_label)
#     sections = np.array(df_case["section"].unique())
#     labels = dict(zip(y_labels, np.arange(len(y_labels), dtype=int)))
#     distrib_arr = np.zeros((len(labels), sections.shape[0]))
#     for i_ar, i_sec in enumerate(sections):
#         _df = _df_label[_df_label.section == i_sec]
#         _l_keys = list(_df.area_id)
#         _labels = [labels[v] for v in _l_keys]
#         _labels = np.array(_labels)
#         distrib_arr[_labels, i_ar] = 1
#
#     fig_size = (cm2inch(distrib_arr.shape[1] // 7), cm2inch(distrib_arr.shape[0] // 5))
#
#     fig, ax = plt.subplots(figsize=fig_size)
#     ax.imshow(distrib_arr, cmap=plt.get_cmap('Greys'))
#     axes_imshow_formatting(ax, y_labels, sections)
#
#     plt.tight_layout()
#     path_fig = os.path.join(output_dir, f"{case}.png")
#
#     plt.savefig(path_fig, dpi=300)
#     plt.close()


def plot_zeroed_labels_distribution(df, output_dir, max_bar_in_plot):
    plt_prop = PlotProperties()
    df_mask = (df.label == 0) & df.accept
    df_diff = df[df_mask]

    x_all_labels = np.sort(df_diff.area_id.unique())
    y_all_labels = df_diff.groupby(["area_id"]).size().values
    if x_all_labels[0] == 0:
        x_all_labels = x_all_labels[1:]
        y_all_labels = y_all_labels[1:]
    n_plots = math.ceil(x_all_labels.shape[0] / max_bar_in_plot)

    x = np.arange(1, max_bar_in_plot + 1)

    for i in range(n_plots):
        if i < n_plots - 1:
            x_labels = list(
                x_all_labels[i * max_bar_in_plot : (i + 1) * max_bar_in_plot]
            )
            y = list(y_all_labels[i * max_bar_in_plot : (i + 1) * max_bar_in_plot])
        else:
            x_labels = list(x_all_labels[i * max_bar_in_plot :])
            x = np.arange(1, len(x_labels) + 1)

            y = list(y_all_labels[i * max_bar_in_plot :])

        logger.info("Zeroed out labels...%s-%s", x_labels[0], x_labels[-1])

        fig, ax = plt.subplots(figsize=C_FIGSIZE)

        bar_container_all = ax.bar(
            x - C_BARWIDTH / 2,
            y,
            edgecolor=C_COLORS["accepted"],
            color=C_COLORS["edge"],
            linewidth=plt_prop.line_width,
            width=C_BARWIDTH * 2,
        )

        ax.set_ylabel(Y_LABEL, labelpad=plt_prop.pad, fontproperties=plt_prop.font)
        ax.set_xlabel("Label", labelpad=plt_prop.pad, fontproperties=plt_prop.font)
        ax.xaxis.set_label_coords(0.5 * len(x) / 30 + 0.015, -0.16)

        max_val = max(bar_container_all.datavalues)
        step = find_axis_step(max_val)
        y_axis = {"min": 0, "max": max_val, "step": step}

        axes_formatter = AxesFormattingBarPlot(ax)
        axes_formatter.format_axes(y_axis, x, x_labels, max_bar_in_plot)

        path_fig = os.path.join(
            output_dir, f"zeroed_labels_{x_labels[0]}-{x_labels[-1]}.png"
        )

        plt.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.18)
        plt.savefig(path_fig, dpi=300)
        plt.close()


def find_labels(labels_df_path, labels_to_process):
    df_labels = pd.read_csv(labels_df_path)
    labels_to_process.sort()
    labels = list(df_labels.label.astype(int))
    if len(labels_to_process) == 0:
        return labels, labels
    else:
        train_labels = [0]
        train_labels.extend(labels_to_process)
        return labels, train_labels


def process_visualization(paths):
    output_dir = paths.visualization

    config = DatasetConfiguration(paths.config_fname)

    df = pd.read_csv(paths.visualization_csv)

    max_bar_in_plot = config("max_bar_in_plot")

    if len(config("specific_area_id_list")) != 0:
        logger.info("Zeroed out labels distribution...")
        plot_zeroed_labels_distribution(df, output_dir, max_bar_in_plot)
        logger.info("Zeroed out labels distribution... Done!")

    labels, train_labels = find_labels(
        paths.labels_idx, config("specific_area_id_list")
    )

    for case in config("case_list"):
        logger.info("Case... %s", case)
        df_case = df[df.case == case]

        logger.info("Profiles distribution in case...")
        plot_profiles_distribution_in_case(case, df_case, output_dir, max_bar_in_plot)
        logger.info("Profiles distribution in case... Done!")

        # logger.info('[Case: %s]; Labels distribution in case...', case)
        # plot_labels_in_case(case, df_case, output_dir, max_bar_in_plot)
        # logger.info('[Case: %s]; Profiles distribution in case... Done!', case)

        # TODO move to different place
        # imshow_distribution_labels_in_section(case, df_case, labels, output_dir)

    logger.info("Done! All plots saved to: %s", output_dir)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process_visualization.__doc__,
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
        "-v",
        "--input-visualization-csv",
        required=True,
        dest="visualization_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-d",
        "--labels-to-idx",
        required=True,
        dest="labels_idx",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    parser.add_argument(
        "-o",
        "--output-visualizastion",
        required=True,
        dest="visualization",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    ),

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)

    input_options = parse_args()
    process_visualization(input_options)
