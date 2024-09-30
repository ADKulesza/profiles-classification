import argparse
import os
from math import ceil, floor

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

from plot_methods.plot_formatting import AxesFormattingProfileSample
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties

C_LOGGER_NAME = "profile_plot"

Y_AXIS = {"min": 0, "max": 1, "step": 1}

C_YLIM = [-0.02, 1.05]
C_THICKNESS_DOMAIN = np.linspace(0.0, 1.0, 128)

# And some other settings:
C_PROFILE_LINECOLOR = "#555555"
C_PROFILE_MEDIAN_LINECOLOR = "#000000"


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(2.5), cm2inch(4.5))


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


def plot_profile(profile, fig_path):
    plt_prop = PlotProperties()
    fig, ax = plt.subplots(1, 1, figsize=C_FIGSIZE)

    x_axis = {
        "min": floor(np.min(profile) / 10) * 10,
        "max": ceil(np.max(profile) / 10) * 10,
        "step": 20,
    }
    x_axis["step"] = x_axis["max"] - x_axis["min"]

    ax.plot(
        profile,
        C_THICKNESS_DOMAIN,
        linestyle="-",
        linewidth=plt_prop.line_width,
        color=C_PROFILE_MEDIAN_LINECOLOR,
    )

    # axes_formatting(ax, x_axis)
    axes_formatter = AxesFormattingProfileSample(ax)
    axes_formatter.format_axes(x_axis, Y_AXIS)

    ax.set_xlabel("intensity", fontproperties=plt_prop.font)

    ax.set_ylabel("normalized thickness", fontproperties=plt_prop.font)

    prop = dict(left=0.27, right=0.9, top=0.97, bottom=0.12)
    plt.subplots_adjust(**prop)
    plt.savefig(fig_path, dpi=plt_prop.dpi)
    plt.close()


def process(paths):
    profiles_df = pd.read_csv(paths.acc_profiles_csv)

    labels_list = pd.unique(profiles_df.area_id)
    case_list = pd.unique(profiles_df.case)
    for case in case_list:
        logger.info("Processing case... %s", case)
        case_df = profiles_df[profiles_df.case == case]
        case_path = os.path.join(paths.output_dir, case)
        os.makedirs(case_path, exist_ok=True)
        for label in labels_list:
            logger.info("Processing area... %s", label)
            _label_df = case_df[case_df.area_id == label]

            if paths.n_plots > profiles_df.shape[0]:
                n_plots = profiles_df.shape[0]
            else:
                n_plots = paths.n_plots

            profiles_to_plot = _label_df.sample(n=n_plots)
            prof_arr = np.load(paths.profiles_npy)

            for i_plot, row in enumerate(profiles_to_plot.iterrows()):
                idx_row, _df = row
                profile = prof_arr[_df.index_in_npy_array]
                p_id = str(_df.profile_id)
                p_id = p_id.zfill(6)
                fig_name = "{}_area-{}_sec-{}_id-{}".format(
                    _df.case, label, _df.section, p_id
                )

                fig_fname = os.path.join(case_path, fig_name + ".png")
                plot_profile(profile, fig_fname)
        logger.info("Figures saved to... %s", case_path)


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
        "--acc-profiles-csv",
        required=True,
        dest="acc_profiles_csv",
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
        "-m",
        "--plots-number",
        required=False,
        default=1,
        dest="n_plots",
        type=int,
        metavar="DECIMAL",
        help="Number of profile plots",
    )

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()
    process(input_options)
