import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_methods.plot_formatting import AxesFormattingBinaryConfusionMatrix
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties

C_LOGGER_NAME = "cmat"


def read_data(paths):
    logger.info("Loading data...")
    confmat = np.load(paths.confmat)
    logger.info("%s", paths.confmat)

    logger.info("Loading data... Done!")

    return confmat


def plot_confmat(confmat, plot_path, xy_labels, figsize, show_values, plt_prop):
    logger.info("Plotting... ")
    xy = {"min": 0, "max": xy_labels.shape[0], "step": 1}
    fig, ax = plt.subplots(figsize=plt_prop.cm2inch((figsize, figsize)))
    ax.matshow(np.log(confmat + 1), cmap="Oranges", alpha=0.6)

    if show_values:
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center", fontsize=4)

    axes_formatter = AxesFormattingBinaryConfusionMatrix(ax)
    axes_formatter.format_axes(xy, xy, xy_labels)

    ax.set_xlabel("Predicted class", fontproperties=plt_prop.font)
    ax.xaxis.set_label_coords(0.35, 0.28)
    ax.set_ylabel("Real class", fontproperties=plt_prop.font)
    ax.yaxis.set_label_coords(-0.05, 0.65)

    prop = dict(left=0.15, right=1.25, top=0.94, bottom=-0.25)
    plt.subplots_adjust(**prop)

    plt.savefig(plot_path + ".png", dpi=300)
    plt.savefig(plot_path + ".svg", dpi=300)

    logger.info("Plotting... Done! Results saved to: %s", plot_path)


def process(paths):
    confmat = read_data(paths)

    xy_labels = np.array(["1", "0"])

    plt_prop = PlotProperties()

    plot_confmat(
        confmat,
        paths.confmat_plot,
        xy_labels,
        paths.figsize,
        paths.show_values,
        plt_prop,
    )


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--confmat",
        required=True,
        dest="confmat",
        type=str,
        metavar="FILENAME",
        help="Path to output npy file with",
    )

    parser.add_argument(
        "-f",
        "--figsize",
        required=True,
        dest="figsize",
        type=float,
        metavar="DECIMAL",
        help="Size of plot in cm",
    )

    parser.add_argument(
        "-o",
        "--output-confmat-plot",
        required=True,
        dest="confmat_plot",
        type=str,
        metavar="FILENAME",
        help="Path to ",
    )

    parser.add_argument("--show-values", action="store_true", dest="show_values")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()

    process(input_options)
