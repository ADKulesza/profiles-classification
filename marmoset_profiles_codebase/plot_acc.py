import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_methods.fix_svg_files import fix_svg

C_EXPORT_DPI = 300
C_DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
C_DEFAULT_FONT_SIZE = 8
C_DEFAULT_FONT_PROP = font_manager.FontProperties(
    fname=C_DEFAULT_FONT_PATH, size=C_DEFAULT_FONT_SIZE
)

C_LINEWIDTH = 1
C_BARWIDTH = 0.2

LINEWIDTH = 1
PAD = 1
LABELPAD = 1
TICKS_LEN = 2
MARKERSIZE = 3

C_COLORS = {
    "grey": "#8b8b8b",  # grey
    "dark_gr": "#5a5a3f",
    "bl": "k",  # black
    "white": "white",
}


def cm2inch(x):
    return x * 0.39


C_FIGSIZE = (cm2inch(23), cm2inch(5))


def axes_formatting(ax, xy_axis, x_labels):
    """
    Details for the axis
    """
    y_axis = {"min": 0, "max": 1, "step": 0.2}
    y_majors = np.arange(y_axis["min"], y_axis["max"] + y_axis["step"], y_axis["step"])

    x_majors = np.arange(xy_axis["min"], xy_axis["max"], xy_axis["step"])
    #
    # # Distance between ticks and label of ticks
    ax.tick_params(
        axis="y",
        which="major",
        pad=PAD,
        length=TICKS_LEN,
        width=1,
        left="off",
        labelleft="off",
        direction="in",
    )

    ax.tick_params(
        axis="x",
        which="major",
        pad=PAD,
        length=0,
        width=1,
        left="off",
        labelleft="off",
        direction="in",
    )
    #
    #
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Limits and ticks for y-axis
    ax.set_ylim(y_axis["max"], y_axis["max"])
    ax.spines["left"].set_bounds((y_axis["min"], y_axis["max"]))
    ax.spines["left"].set_position(("data", -1))
    #
    labels = map(lambda x: "{:.1f}".format(x), y_majors)
    ax.set_yticks(y_majors)
    ax.set_yticklabels(labels)

    #
    # # Limits and ticks for x-axis
    ax.set_xlim(-1, xy_axis["max"] + C_BARWIDTH / 2)
    ax.spines["bottom"].set_bounds((-0.6, len(x_majors) - 0.5))
    #
    ax.set_xticks(x_majors)
    ax.set_xticklabels(x_labels, fontsize=6)

    plt.xticks(rotation=90)
    #

    # Format labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(C_DEFAULT_FONT_PROP)
        label.set_fontsize(C_DEFAULT_FONT_SIZE)
    for label in ax.get_xticklabels():
        label.set_fontproperties(
            font_manager.FontProperties(fname=C_DEFAULT_FONT_PATH, size=6)
        )
        label.set_fontsize(5)


def plot_acc(paths_label_names, df):
    x_label = np.array(df.label)
    y_acc = np.array(df.acc)
    x = np.arange(y_acc.shape[0])
    label_names = pd.read_csv(paths_label_names)
    labels_names_list = []
    colors = []
    for l in x_label:
        labels_names_list.append(
            np.array(label_names[label_names.area_id == l].area)[0]
        )

    for r, g, b in zip(label_names.color_r, label_names.color_g, label_names.color_b):
        colors.append([r / 255, g / 255, b / 255])

    fig, ax = plt.subplots(figsize=C_FIGSIZE)

    box_props = dict(linestyle="-", linewidth=LINEWIDTH, color=colors)
    ax.bar(x, y_acc, **box_props)

    ax.set_ylabel("Accuracy", labelpad=LABELPAD, fontproperties=C_DEFAULT_FONT_PROP)
    ax.set_xlabel("Area", labelpad=LABELPAD, fontproperties=C_DEFAULT_FONT_PROP)

    x_axis = {"min": 0, "max": x.shape[0], "step": 1}

    axes_formatting(ax, x_axis, labels_names_list)

    prop = dict(
        left=0.035, right=0.995, top=0.985, bottom=0.24, wspace=0.18, hspace=0.3
    )
    plt.subplots_adjust(**prop)
    plt.savefig(
        "/home/akulesza/Public/2021-12-29-profiles-pipeline/one_vs_all_different_datasets/one_vs_all_acc.png",
        dpi=C_EXPORT_DPI,
    )
    plt.savefig(
        "/home/akulesza/Public/2021-12-29-profiles-pipeline/one_vs_all_different_datasets/one_vs_all_acc.svg",
        dpi=C_EXPORT_DPI,
    )
    fix_svg(
        "/home/akulesza/Public/2021-12-29-profiles-pipeline/one_vs_all_different_datasets/"
    )


if __name__ == "__main__":
    l_path = "/home/akulesza/Public/2021-12-29-profiles-pipeline/label_names.csv"
    acc_path = "/home/akulesza/Public/2021-12-29-profiles-pipeline/streamlines/one_vs_all_acc.csv"
    df_acc = pd.read_csv(acc_path)
    df_acc = df_acc.sort_values(by="label")
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Arial"
    mpl.rcParams["svg.fonttype"] = "none"
    plot_acc(l_path, df_acc)
