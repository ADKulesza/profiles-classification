import argparse
import glob
import re
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_methods.plot_formatting import AxesFormattingRegConfusionMatrix
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties
from read_json import read_json
from plot_methods.fix_svg_files import fix_svg

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


def plot_confmat(confmat, matrix_len, plot_path, xy_labels, plt_prop):
    logger.info("Plotting... ")
    fig, ax = plt.subplots(figsize=plt_prop.cm2inch((6, 6)))

    ax.matshow(np.log(confmat + 1), cmap="Oranges", alpha=0.6)

    xy = {"min": 0, "max": confmat.shape[0], "step": 1}

    axes_formatter = AxesFormattingRegConfusionMatrix(ax)
    axes_formatter.format_axes(xy, xy, xy_labels, matrix_len, small_font=False)

    # ax.set_ylabel(
    #     "Prawdziwa klasa", labelpad=plt_prop.label_pad, fontproperties=plt_prop.font
    # )
    #
    # ax.set_xlabel(
    #     "Przewidywana klasa", labelpad=plt_prop.label_pad, fontproperties=plt_prop.font
    # )
    # ax.xaxis.set_label_coords(0.5, 0.0)
    title = get_filename(plot_path)
    title = title.replace("_", " ")
    ax.set_title(title, fontproperties=plt_prop.font)

    plt.savefig(plot_path + ".png", dpi=300)
    plt.savefig(plot_path + ".svg", dpi=300)

    logger.info("Plotting... Done! Results saved to: %s", plot_path)


def process(paths):
    confmats, area_order = read_data(paths)

    regions = area_order[2]
    logger.info("X LABELS: %s", regions)

    plt_prop = PlotProperties()

    matrix_len = max(confmat.shape[0] for confmat in confmats.values())

    for i, _reg in zip(range(13), regions.keys()):
        _reg_name = _reg.replace(" ", "_")
        confmat = confmats[_reg_name]

        plot_fname = os.path.join(paths.output, _reg_name)
        plot_confmat(
            confmat,
            matrix_len,
            plot_fname,
            regions[_reg],
            plt_prop,
        )

    confmat = confmats["regionxregion_cmat"]
    regions_list = [_key[:2] for _key in regions.keys()]
    plot_fname = os.path.join(paths.output, "regionxregion_cmat")
    plot_confmat(
        confmat,
        matrix_len,
        plot_fname,
        regions_list,
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
