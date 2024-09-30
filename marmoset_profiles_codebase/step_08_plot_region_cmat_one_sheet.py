import argparse
import glob
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_methods.plot_formatting import AxesFormattingRegConfusionMatrix
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties
from read_json import read_json

C_LOGGER_NAME = "cmat"


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
        region_name = get_filename(_path)[:-4]
        confmat_dict[region_name] = np.load(_path)
        logger.info("%s", _path)

    area_order = read_json(paths.area_order)
    logger.info("%s", paths.area_order)

    logger.info("Loading data... Done!")

    return confmat_dict, area_order


def plot_confmat(confmat_dict, plot_path, regions, plt_prop):
    logger.info("Plotting... ")

    fig, axs = plt.subplots(7, 2, figsize=plt_prop.cm2inch((16.8, 67.2)))

    matrix_len = max(confmat.shape[0] for confmat in confmat_dict.values())

    for i, _reg in zip(range(13), regions.keys()):
        ax = axs[i // 2, i % 2]
        _reg_name = _reg.replace(" ", "_")
        confmat = confmat_dict[_reg_name]
        # confmat = np.pad(confmat, pad_width=((0, matrix_len - confmat.shape[0]),
        #                                      (0, matrix_len - confmat.shape[0])),
        #                  mode='constant', constant_values=1)
        #
        # confmat = np.ma.masked_where(confmat == 1, confmat)

        xy = {"min": 0, "max": len(regions[_reg]), "step": 1}

        axes_formatter = AxesFormattingRegConfusionMatrix(ax)

        axes_formatter.format_axes(xy, xy, regions[_reg], matrix_len, small_font=False)

        if i % 2 == 0:
            ax.set_ylabel(
                "Prawdziwa klasa", labelpad=plt_prop.label_pad, fontproperties=plt_prop.font
            )

        ax.matshow(np.log(confmat + 1), cmap="Oranges", alpha=0.6)

    confmat = confmat_dict["regionxregion_cmat"]


    xy = {"min": 0, "max": len(regions[_reg]), "step": 1}
    axs[6, 1].matshow(np.log(confmat + 1), cmap="Oranges", alpha=0.6)
    axes_formatter = AxesFormattingRegConfusionMatrix(axs[6, 1])
    regions_list = [_key[:5] for _key in regions.keys()]
    axes_formatter.format_axes(xy, xy,  regions_list, matrix_len, small_font=False)

    for i in range(2):
        axs[6, i].set_xlabel(
            "Przewidywana klasa", labelpad=plt_prop.label_pad, fontproperties=plt_prop.font
        )
        axs[6, i].xaxis.set_label_coords(0.5, 0.0)

    prop = dict(
        left=0.1, right=1.001, top=0.966, bottom=0.0175, wspace=0.05, hspace=0.01
    )
    # prop = dict(
    #     left=0.05, right=0.99, top=0.966, bottom=0.0175, wspace=0.1, hspace=0.1
    # )
    plt.subplots_adjust(**prop)
    # fig.tight_layout()
    plt.savefig(plot_path + ".png", dpi=300)
    plt.savefig(plot_path + ".svg", dpi=300)

    # dir_path, fig_pref = os.path.split(plot_path)
    logger.info("Plotting... Done! Results saved to: %s", plot_path)

    # logger.info("Fixing .svg files in: %s", dir_path)
    # # fix_svg(dir_path)
    # logger.info("Fixing... Done!")


def process(paths):
    confmats, area_order = read_data(paths)

    regions = area_order[2]
    logger.info("X LABELS: %s", regions)

    plt_prop = PlotProperties()

    plot_confmat(
        confmats,
        paths.confmat_plot,
        regions,
        plt_prop,
    )


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-r",
        "--area-order",
        required=True,
        dest="area_order",
        type=str,
        metavar="FILENAME",
        help="Path to file with",
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
        "-o",
        "--output-confmat-plot",
        required=True,
        dest="confmat_plot",
        type=str,
        metavar="FILENAME",
        help="Path to "
    )

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    mpl.rcParams['xtick.minor.visible'] = False
    mpl.rcParams['ytick.minor.visible'] = False

    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()
    process(input_options)
