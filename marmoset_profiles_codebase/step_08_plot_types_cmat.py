import argparse
import glob
import re
import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_methods.plot_formatting import AxesFormattingRegConfusionMatrix
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties
from plot_methods.fix_svg_files import fix_svg

C_LOGGER_NAME = "cmat"

AREAS_TYPES = [
    "Kon",
    "Eu3",
    "Eu2",
    "Eu1",
    "Dys",
    "Agr"
]

C_FIGSIZE = (13, 13)
C_FIGSIZE_X = (4, 4)


def get_filename(path):
    # Regex dla wyciągnięcia nazwy pliku (znaki po ostatnim slashu lub backslashu)
    match = re.search(r'[^\\/]+$', path)
    if match:
        return match.group(0)
    return None


def read_data(paths):
    logger.info("Loading data...")
    cmat_paths = glob.glob(f"{paths.input_dir}/*.npy")
    confmat_dict = dict()
    for _path in cmat_paths:
        area_type_name = get_filename(_path)[:-9]
        confmat_dict[area_type_name] = np.load(_path)
        logger.info("%s", _path)

    label_names = pd.read_csv(paths.label_names)
    logger.info("%s", paths.label_names)

    logger.info("Loading data... Done!")

    return confmat_dict, label_names


def plot_confmat(confmat, matrix_len, plot_path, xy_labels, plt_prop, x_cmat=False):
    logger.info("Plotting... ")
    if x_cmat:
        fig, ax = plt.subplots(figsize=plt_prop.cm2inch(C_FIGSIZE_X))
        prop = dict(
            left=0.17, right=0.99, top=0.92, bottom=0.03)
    else:
        fig, ax = plt.subplots(figsize=plt_prop.cm2inch(C_FIGSIZE))
        prop = dict(
            left=0.12, right=0.998, top=0.95, bottom=0.013)

    ax.matshow(np.log(confmat + 1), cmap="Oranges", alpha=0.6)

    xy = {"min": 0, "max": confmat.shape[0], "step": 1}

    axes_formatter = AxesFormattingRegConfusionMatrix(ax)
    axes_formatter.format_axes(xy, xy, xy_labels, matrix_len, small_font=False)

    ax.set_ylabel(
        "Prawdziwa klasa", labelpad=plt_prop.label_pad, fontproperties=plt_prop.font
    )

    ax.set_xlabel(
        "Przewidywana klasa", labelpad=0, fontproperties=plt_prop.font
    )
    ax.xaxis.set_label_coords(0.5, 0.0)

    plt.subplots_adjust(**prop)

    plt.savefig(plot_path + ".png", dpi=300)
    plt.savefig(plot_path + ".svg", dpi=300)

    logger.info("Plotting... Done! Results saved to: %s", plot_path)


def process(paths):
    confmats, labels_names = read_data(paths)
    plt_prop = PlotProperties()

    matrix_len = max(confmat.shape[0] for confmat in confmats.values())

    for area_type in AREAS_TYPES:
        confmat = confmats[area_type]

        _labels = labels_names[labels_names.type == area_type]
        _labels = _labels.sort_values("order")

        _labels = list(_labels.area)

        logger.info("%s", _labels)

        plot_fname = os.path.join(paths.output, area_type)

        plot_confmat(
            confmat,
            matrix_len,
            plot_fname,
            _labels,
            plt_prop
        )

    confmat = confmats["typesxtypes"]

    plot_fname = os.path.join(paths.output, "typesxtypes")
    plot_confmat(
        confmat,
        6,
        plot_fname,
        AREAS_TYPES,
        plt_prop,
        x_cmat=True
    )


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--input-directory",
        required=True,
        dest="input_dir",
        type=str,
        metavar="FILENAME",
        help="Path to output npy file with",
    )

    parser.add_argument(
        "-n",
        "--label-names",
        required=True,
        dest="label_names",
        type=str,
        metavar="FILENAME",
        help="Path to csv file with",
    )

    parser.add_argument(
        "-o",
        "--output-confmat-plot",
        required=True,
        dest="output",
        type=str,
        metavar="FILENAME",
        help="Path to "
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    mpl.rcParams['xtick.minor.visible'] = False
    mpl.rcParams['ytick.minor.visible'] = False
    mpl.rcParams['svg.fonttype'] = 'none'

    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()
    process(input_options)
    fix_svg(input_options.output)
