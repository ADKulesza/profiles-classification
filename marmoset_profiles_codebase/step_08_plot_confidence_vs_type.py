import argparse
import os

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from plot_methods.plot_formatting import AxesFormattingVerticalBarhPlot
from plot_methods.plot_logger import get_logger
from plot_methods.plot_properties import PlotProperties
from read_json import read_json

C_DPI = 300

C_FIGSIZE = (15, 5)
MARKERSIZE = 3
C_STD_COLOR = '#cccccc'

X_AXIS = {
    'min': 0,
    'max': 5,
    'step': 1
}

C_LOGGER_NAME = "metrics_vs"


def func_b(x, a, b):
    return a * x + b


def func(x, a):
    return a * x


def metrics_plot(xdata, ydata, metric, output_dir):
    plt_prop = PlotProperties()
    fig, ax = plt.subplots(figsize=plt_prop.cm2inch(C_FIGSIZE))

    # Dopasowanie a*x
    popt, pcov = curve_fit(func, xdata, ydata)
    perr = np.sqrt(np.diag(pcov))

    # Dopasowanie a*x + b
    popt_b, pcov_b = curve_fit(func_b, xdata, ydata)
    perr_b = np.sqrt(np.diag(pcov_b))

    x = np.arange(X_AXIS["min"], X_AXIS["max"] + X_AXIS["step"] / 2, 1 / 10)
    y = func_b(x, *popt_b)

    ax.plot(x, y,
            linewidth=plt_prop.line_width, color='k',
            zorder=100)

    ax.fill_between(x,
                    func_b(x, popt_b[0] + perr_b[0], popt_b[1] + perr_b[1]),
                    func_b(x, popt_b[0] - perr_b[0], popt_b[1] - perr_b[1]),
                    facecolor=C_STD_COLOR,
                    alpha=1,
                    edgecolor=None,
                    linewidth=None,
                    zorder=-10)

    ax.plot(xdata, ydata,
            marker='o',
            fillstyle='full',
            mfc='white',
            markeredgewidth=0.5,
            color='k',
            linestyle='None',
            markersize=MARKERSIZE)


    text = [
        "y=ax; a={:.2f}$\pm${:.2f}".format(popt[0], perr[0]),
        "y=ax+b;",
        "a={:.2f}$\pm${:.2f};".format(popt_b[0], perr_b[0]),
        "b={:.2f}$\pm${:.2f}".format(popt_b[1], perr_b[1])
    ]

    text_prop = dict(fontproperties=plt_prop.font, ha="left", va="top")
    ax.text(.1, .99, text[0], **text_prop)
    ax.text(.1, .95, text[1], **text_prop)
    ax.text(.1, .91, text[2], **text_prop)
    ax.text(.1, .87, text[3], **text_prop)

    plt.show()

    #
    # plt.savefig(os.path.join(output_dir, f"{metric}_vs_type.png"), dpi=C_DPI)
    # plt.savefig(os.path.join(output_dir, f"{metric}_vs_type.svg"), dpi=C_DPI)


def read_data(paths):
    logger.info("Loading data...")

    # Loading .csv file with information about profiles
    profiles_df = pd.read_csv(paths.validation_csv)
    logger.info("%s", paths.validation_csv)

    label_names = pd.read_csv(paths.label_names)
    logger.info("%s", paths.label_names)

    logger.info("Loading data... Done!")

    return profiles_df, label_names


def process(paths):
    profiles_df, label_names = read_data(paths)
    profiles_df = profiles_df[profiles_df.idx_in_model != profiles_df.pred_y]

    confidence_values = profiles_df["pred_confidence"].array

    type_values = profiles_df["type_id"].array

    metrics_plot(type_values, confidence_values, "pred_confidence", paths.output_dir)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description=process.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--validation-csv",
        required=True,
        dest="validation_csv",
        type=str,
        metavar="FILENAME",
        help="Path to ",
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
    logger = get_logger(C_LOGGER_NAME)
    input_options = parse_args()

    process(input_options)
