import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_curve)

from plot_methods.plot_logger import get_logger

C_LOGGER_NAME = "Visualization of model"

# C_DEFAULT_FONT_PATH = '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf'
C_DEFAULT_FONT_SIZE = 8
C_DEFAULT_FONT_PROP = font_manager.FontProperties(
    family="Arial", size=C_DEFAULT_FONT_SIZE
)

C_LINEWIDTH = 1
C_BARWIDTH = 0.2

LINEWIDTH = 1
PAD = 1
LABELPAD = 1
TICKS_LEN = 2

C_COLORS = {"all": "#8b8b8b", "accepted": "k", "edge": "white"}  # darker grey  # black

Y_LABEL = "Number of profiles"

C_DIR_LABELS_IN_SECTION = "labels_distribution_in_section"
C_DIR_PROFILES_IN_CASE = "profiles_distribution_in_case"
C_DIR_LABELS_IN_CASE = "labels_number_in_case"


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(16), cm2inch(10))


def axes_formatting(ax):
    """
    Details for the axis
    """
    # # Make right and top line invisible
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Format labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)


def summary_plot(model_history, plot_path, i_plot):
    logger = get_logger(C_LOGGER_NAME)

    logger.info("Plotting... ")
    fig, axs = plt.subplots(2, 2, figsize=C_FIGSIZE)

    _plot_acc(axs[0, 0], model_history)
    try:
        _plot_recall(axs[1, 0], model_history, i_plot)
        _plot_precision(axs[0, 1], model_history, i_plot)
    except:
        pass
    # _plot_roc(axs[1, 0], y_test, y_predict)
    # _plot_confmat(axs[0, 1], y_test, y_predict)
    # _plot_pr(axs[1, 1], y_test, y_predict)

    # for ax in axs.flatten():
    #     axes_formatting(ax)

    prop = dict(left=0.1, right=0.96, top=0.9, bottom=0.1, wspace=0.25, hspace=0.38)
    plt.subplots_adjust(**prop)
    plt.savefig(plot_path)
    logger.info("Plots saved to... %s", plot_path)


def _plot_recall(ax, model_history, i_plot):
    if i_plot == 0:
        metric_suff = ""
    else:
        metric_suff = f"_{i_plot}"
    ax.plot(
        model_history.history["recall" + metric_suff],
        linestyle="--",
        color=(0.6, 0.6, 0.6),
    )
    ax.plot(model_history.history["val_recall" + metric_suff], color="black")
    ax.set_ylabel("Recall", labelpad=LABELPAD, fontproperties=C_DEFAULT_FONT_PROP)
    ax.set_xlabel("Epochs", labelpad=LABELPAD, fontproperties=C_DEFAULT_FONT_PROP)
    ax.set_title(f"recall", fontproperties=C_DEFAULT_FONT_PROP)
    plt.legend(["train", "test"], loc="upper left")

    acc = max(model_history.history["recall" + metric_suff])
    val_acc = max(model_history.history["val_recall" + metric_suff])
    ax.annotate(
        "recall={:.3f}\nval_recall={:.3f}".format(acc, val_acc),
        xy=(0.95, 0.2),
        xycoords="axes fraction",
        ha="right",
        fontproperties=C_DEFAULT_FONT_PROP,
    )


def _plot_precision(ax, model_history, i_plot):
    if i_plot == 0:
        metric_suff = ""
    else:
        metric_suff = f"_{i_plot}"

    ax.plot(
        model_history.history["precision" + metric_suff],
        linestyle="--",
        color=(0.6, 0.6, 0.6),
    )
    ax.plot(model_history.history["val_precision" + metric_suff], color="black")
    ax.set_ylabel("Precision", labelpad=LABELPAD, fontproperties=C_DEFAULT_FONT_PROP)
    ax.set_xlabel("Epochs", labelpad=LABELPAD, fontproperties=C_DEFAULT_FONT_PROP)
    ax.set_title(f"precision", fontproperties=C_DEFAULT_FONT_PROP)
    plt.legend(["train", "test"], loc="upper left")

    acc = max(model_history.history["precision" + metric_suff])
    val_acc = max(model_history.history["val_precision" + metric_suff])
    ax.annotate(
        "precision={:.3f}\nval_recall={:.3f}".format(acc, val_acc),
        xy=(0.95, 0.2),
        xycoords="axes fraction",
        ha="right",
        fontproperties=C_DEFAULT_FONT_PROP,
    )


def _plot_acc(ax, model_history):
    ax.plot(model_history.history["accuracy"], linestyle="--", color=(0.6, 0.6, 0.6))
    ax.plot(model_history.history["val_accuracy"], color="black")
    ax.set_ylabel("Accuracy", labelpad=LABELPAD, fontproperties=C_DEFAULT_FONT_PROP)
    ax.set_xlabel("Epochs", labelpad=LABELPAD, fontproperties=C_DEFAULT_FONT_PROP)
    ax.set_title(f"ACC", fontproperties=C_DEFAULT_FONT_PROP)
    plt.legend(["train", "test"], loc="upper left")

    acc = max(model_history.history["accuracy"])
    val_acc = max(model_history.history["val_accuracy"])
    ax.annotate(
        "acc={:.3f}\nval_acc={:.3f}".format(acc, val_acc),
        xy=(0.95, 0.2),
        xycoords="axes fraction",
        ha="right",
        fontproperties=C_DEFAULT_FONT_PROP,
    )
