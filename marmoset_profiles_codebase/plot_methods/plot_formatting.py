import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator

from plot_methods.plot_properties import PlotProperties

PLOT_PROP = PlotProperties()


class AxesFormatting:
    PLOT_PROP = PlotProperties()

    def __init__(self, ax):
        self._ax = ax

    def format_axes(self, x_axis, y_axis, *args, **kwargs):
        # Ticks format
        self._set_major_tick_params()

        # Make right and top line invisible
        self._make_top_right_ax_invisible()

        # Limits and ticks for x-axis
        self._set_x_axis(x_axis, y_axis)

        # Limits and ticks for y-axis
        self._set_y_axis(y_axis, x_axis)

        # Format labels
        self._format_labels()

    def _set_major_tick_params(self):

        # Distance between ticks and label of ticks
        self._ax.tick_params(
            axis="x",
            which="major",
            pad=PLOT_PROP.pad,
            length=PLOT_PROP.ticks_len,
            width=1,
            left="off",
            labelleft="off",
            direction="in",
        )

        self._ax.tick_params(
            axis="y",
            which="major",
            pad=PLOT_PROP.pad,
            length=PLOT_PROP.ticks_len,
            width=1,
            left="off",
            labelleft="off",
            direction="in",
        )

    def _format_labels(self):
        """
        Format labels
        """
        for label in self._ax.get_yticklabels():
            label.set_fontproperties(PLOT_PROP.font)

        for label in self._ax.get_xticklabels():
            label.set_fontproperties(PLOT_PROP.font)

    def _make_top_right_ax_invisible(self):
        """
        Make right and top line invisible
        """

        self._ax.spines["right"].set_visible(False)
        self._ax.spines["top"].set_visible(False)

    def _set_x_axis(self, x_axis, y_axis, *args, **kwargs):
        x_majors = np.arange(
            x_axis["min"], x_axis["max"] + x_axis["step"], x_axis["step"]
        )

        # Limits and ticks for x-axis
        self._ax.set_xlim(x_axis["min"], x_axis["max"] + x_axis["step"] * 1.01)
        self._ax.spines["bottom"].set_position(("data", y_axis["min"]))
        self._ax.spines["bottom"].set_bounds(x_axis["min"], x_axis["max"])

        if x_axis["max"] <= 1:
            labels = map(lambda x: "0" if x == 0 else "{:.1f}".format(x), x_majors)

        else:
            labels = map(lambda x: "0" if x == 0 else "{:.0f}".format(x), x_majors)

        self._ax.set_xticks(x_majors)
        self._ax.set_xticklabels(labels)

    def _set_y_axis(self, y_axis, x_axis, *args, **kwargs):
        y_majors = np.arange(
            y_axis["min"], y_axis["max"] + y_axis["step"], y_axis["step"]
        )

        # Limits and ticks for y-axis
        self._ax.set_ylim(y_axis["min"], y_axis["max"])
        self._ax.spines["left"].set_position(
            ("data", x_axis["min"] - x_axis["step"] / 2)
        )
        self._ax.spines["left"].set_bounds(y_axis["min"], y_axis["max"])

        if y_axis["max"] <= 1:
            labels = map(lambda x: "{:.1f}".format(x), y_majors)
        else:
            labels = map(lambda x: "{:.0f}".format(x), y_majors)

        self._ax.set_yticks(y_majors)
        self._ax.set_yticklabels(labels)


class AxesFormattingBarPlot(AxesFormatting):

    def format_axes(self, y_axis, x_majors, x_labels, max_bar_in_plot):
        # Ticks format
        self._set_major_tick_params()

        # Make right and top line invisible
        self._make_top_right_ax_invisible()

        # Limits and ticks for x-axis
        self._set_x_axis(x_majors, max_bar_in_plot, x_labels)

        # Limits and ticks for y-axis
        x_axis = {"min": x_majors[0], "step": x_majors[1] - x_majors[0]}
        self._set_y_axis(y_axis, x_axis)

        # Format labels
        self._format_labels()

    def _set_x_axis(self, x_majors, max_bar_in_plot, x_labels):
        # Limits and ticks for x-axis
        self._ax.set_xlim(0, max_bar_in_plot + 1)
        self._ax.spines["bottom"].set_bounds((0.4, len(x_majors) + 0.5))

        self._ax.set_xticks(x_majors)
        self._ax.set_xticklabels(x_labels)
        plt.xticks(rotation=90)


class AxesFormattingProfileSample(AxesFormatting):
    _C_YLIM = {"min": -0.05, "max": 1.05}

    def format_axes(self, x_axis, y_axis, *args, **kwargs):
        super().format_axes(x_axis, y_axis)
        self._set_minor_tick_params()

        self._ax.xaxis.set_minor_locator(
            AutoMinorLocator((x_axis["max"] - x_axis["min"]) // 10)
        )

        self._ax.invert_xaxis()

    def _set_y_axis(self, y_axis, x_axis, *args, **kwargs):
        y_majors = np.arange(
            y_axis["min"], y_axis["max"] + y_axis["step"], y_axis["step"]
        )

        # Limits and ticks for y-axis
        self._ax.set_ylim(self._C_YLIM["min"], self._C_YLIM["max"])
        self._ax.spines["left"].set_bounds(0, 1)
        self._ax.spines["left"].set_position(("data", x_axis["max"]))

        labels = map(lambda x: "{:.0f}".format(x), y_majors)
        self._ax.set_yticks(y_majors)
        self._ax.set_yticklabels(labels)

    def _set_minor_tick_params(self):
        self._ax.tick_params(
            axis="x",
            pad=PLOT_PROP.pad,
            size=PLOT_PROP.ticks_len / 2,
            which="minor",
            labelsize=PLOT_PROP.font_size,
        )

        self._ax.tick_params(
            axis="y",
            pad=PLOT_PROP.pad,
            size=PLOT_PROP.ticks_len / 2,
            which="minor",
            left=True,
            labelleft=True,
            labelsize=PLOT_PROP.font_size,
        )


class AxesFormattingProfileSampleHorizontal(AxesFormatting):
    _C_YLIM = {"min": -0.05, "max": 1.05}

    def format_axes(self, x_axis, y_axis, *args, **kwargs):
        super().format_axes(x_axis, y_axis)
        self._set_minor_tick_params()

        self._ax.invert_xaxis()
        self._ax.invert_yaxis()

    def _set_x_axis(self, x_axis, y_axis, *args, **kwargs):
        x_majors = np.arange(
            x_axis["min"], x_axis["max"] + x_axis["step"], x_axis["step"]
        )

        # Limits and ticks for x-axis
        self._ax.set_xlim(x_axis["min"], x_axis["max"] + x_axis["step"] * 1.1)
        self._ax.spines["bottom"].set_position(("data", y_axis["max"] + 0.05))
        self._ax.spines["bottom"].set_bounds(x_axis["min"], x_axis["max"])

        labels = map(lambda x: "0" if x == 0 else "{:.0f}".format(x), x_majors)
        self._ax.set_xticks(x_majors)
        self._ax.set_xticklabels(labels)

        # x_minor = np.arange(x_axis["min"], x_axis["max"] + 10, 10)
        # self._ax.xaxis.set_minor_locator(FixedLocator(x_minor))

    def _set_y_axis(self, y_axis, x_axis, *args, **kwargs):
        y_majors = np.arange(
            y_axis["min"], y_axis["max"] + y_axis["step"], y_axis["step"]
        )

        # Limits and ticks for y-axis
        self._ax.set_ylim(self._C_YLIM["min"], self._C_YLIM["max"])
        self._ax.spines["left"].set_bounds(0, 1)
        self._ax.spines["left"].set_position(("data", x_axis["max"]))

        labels = map(lambda x: "{:.0f}".format(x), y_majors)
        self._ax.set_yticks(y_majors)
        self._ax.set_yticklabels(labels)

    def _set_minor_tick_params(self):
        self._ax.tick_params(
            axis="x",
            pad=PLOT_PROP.pad,
            size=PLOT_PROP.ticks_len / 2,
            which="minor",
            labelsize=PLOT_PROP.font_size,
        )

        self._ax.tick_params(
            axis="y",
            pad=PLOT_PROP.pad,
            size=PLOT_PROP.ticks_len / 2,
            which="minor",
            left=True,
            labelleft=True,
            labelsize=PLOT_PROP.font_size,
        )


class AxesFormattingConfusionMatrix(AxesFormatting):

    def format_axes(self, x_axis, y_axis, xy_labels, small_font=True):
        # Ticks format
        self._set_major_tick_params()

        if small_font:
            font = PLOT_PROP.small_font
        else:
            font = PLOT_PROP.font

        # Limits and ticks for x-axis
        self._set_x_axis(x_axis, y_axis, xy_labels, font)

        # Limits and ticks for y-axis
        self._set_y_axis(y_axis, x_axis, xy_labels, font)

    def _set_major_tick_params(self):

        # Distance between ticks and label of ticks
        self._ax.tick_params(axis="x", which="major", pad=PLOT_PROP.pad, length=0)

        self._ax.tick_params(axis="y", which="major", pad=PLOT_PROP.pad, length=0)

    def _set_x_axis(self, x_axis, y_axis, xy_labels, font):
        x_majors = np.arange(x_axis["min"], x_axis["max"], x_axis["step"])

        # Limits and ticks for y-axis
        self._ax.set_xlim(
            x_axis["min"] - x_axis["step"] / 2, x_axis["max"] + x_axis["step"] / 2
        )
        self._ax.spines["bottom"].set_position(
            ("data", y_axis["max"] - y_axis["step"] / 2)
        )
        self._ax.spines["top"].set_position(
            ("data", y_axis["min"] - y_axis["step"] / 2)
        )
        self._ax.spines["bottom"].set_bounds(
            x_axis["min"] - x_axis["step"] / 2, x_axis["max"] - x_axis["step"] / 2
        )
        self._ax.spines["top"].set_bounds(
            x_axis["min"] - x_axis["step"] / 2, x_axis["max"] - x_axis["step"] / 2
        )

        self._ax.set_xticks(x_majors)
        self._ax.set_xticklabels(xy_labels, rotation=90, fontproperties=font)

    def _set_y_axis(self, y_axis, x_axis, xy_labels, font):
        y_majors = np.arange(y_axis["min"], y_axis["max"], y_axis["step"])

        # Limits and ticks for y-axis
        self._ax.set_ylim(
            y_axis["min"] - y_axis["step"] / 2, y_axis["max"] + y_axis["step"] / 2
        )
        self._ax.spines["left"].set_position(
            ("data", x_axis["min"] - x_axis["step"] / 2)
        )
        self._ax.spines["right"].set_position(
            ("data", x_axis["max"] - x_axis["step"] / 2)
        )
        self._ax.spines["left"].set_bounds(
            y_axis["min"] - y_axis["step"] / 2, y_axis["max"] - y_axis["step"] / 2
        )
        self._ax.spines["right"].set_bounds(
            y_axis["min"] - y_axis["step"] / 2, y_axis["max"] - y_axis["step"] / 2
        )

        self._ax.set_yticks(y_majors)
        self._ax.set_yticklabels(xy_labels, fontproperties=font)
        self._ax.invert_yaxis()


class AxesFormattingRegConfusionMatrix(AxesFormatting):
    def format_axes(self, x_axis, y_axis, xy_labels, matrix_len, small_font=True):
        # Ticks format
        self._set_major_tick_params()

        if small_font:
            font = PLOT_PROP.small_font
        else:
            font = PLOT_PROP.font

        for i in range(matrix_len - len(xy_labels)):
            xy_labels.append("")


        # Limits and ticks for x-axis
        self._set_x_axis(x_axis, y_axis, xy_labels, matrix_len, font)

        # Limits and ticks for y-axis
        self._set_y_axis(y_axis, x_axis, xy_labels, matrix_len, font)

    def _set_major_tick_params(self):

        # Distance between ticks and label of ticks
        self._ax.tick_params(axis="x", which="major", pad=PLOT_PROP.pad, length=0)

        self._ax.tick_params(axis="y", which="major", pad=PLOT_PROP.pad, length=0)

    def _set_x_axis(self, x_axis, y_axis, xy_labels, matrix_len, font):
        x_majors = np.arange(x_axis["min"], matrix_len, x_axis["step"])

        # Limits and ticks for y-axis
        self._ax.set_xlim(
            x_axis["min"] - x_axis["step"], matrix_len + x_axis["step"]
        )
        self._ax.spines["bottom"].set_position(
            ("data", y_axis["max"] - y_axis["step"] / 2)
        )
        self._ax.spines["top"].set_position(
            ("data", y_axis["min"] - y_axis["step"] / 2)
        )
        self._ax.spines["bottom"].set_bounds(
            x_axis["min"] - x_axis["step"] / 2, x_axis["max"] - x_axis["step"] / 2
        )
        self._ax.spines["top"].set_bounds(
            x_axis["min"] - x_axis["step"] / 2, x_axis["max"] - x_axis["step"] / 2
        )

        self._ax.set_xticks(x_majors)
        self._ax.set_xticklabels(xy_labels, rotation=90, fontproperties=font)

    def _set_y_axis(self, y_axis, x_axis, xy_labels, matrix_len, font):
        y_majors = np.arange(y_axis["min"], matrix_len, y_axis["step"])

        # Limits and ticks for y-axis
        self._ax.set_ylim(
            y_axis["min"] - y_axis["step"], matrix_len + y_axis["step"]
        )
        self._ax.spines["left"].set_position(
            ("data", x_axis["min"] - x_axis["step"] / 2)
        )
        self._ax.spines["right"].set_position(
            ("data", x_axis["max"] - x_axis["step"] / 2)
        )
        self._ax.spines["left"].set_bounds(
            y_axis["min"] - y_axis["step"] / 2, y_axis["max"] - y_axis["step"] / 2
        )
        self._ax.spines["right"].set_bounds(
            y_axis["min"] - y_axis["step"] / 2, y_axis["max"] - y_axis["step"] / 2
        )

        self._ax.set_yticks(y_majors)
        self._ax.set_yticklabels(xy_labels, fontproperties=font)
        self._ax.invert_yaxis()


class AxesFormattingBinaryConfusionMatrix(AxesFormatting):

    def format_axes(self, x_axis, y_axis, xy_labels):
        # Ticks format
        AxesFormattingConfusionMatrix._set_major_tick_params(self)

        # Limits and ticks for x-axis
        self._set_x_axis(x_axis, y_axis, xy_labels)

        # Limits and ticks for y-axis
        AxesFormattingConfusionMatrix._set_y_axis(self, y_axis, x_axis, xy_labels)

        # Format labels
        self._format_labels()

    def _set_x_axis(self, x_axis, y_axis, xy_labels):
        x_majors = np.arange(x_axis["min"], x_axis["max"], x_axis["step"])

        # Limits and ticks for y-axis
        self._ax.set_xlim(
            x_axis["min"] - x_axis["step"] / 2, x_axis["max"] + x_axis["step"] / 2
        )
        self._ax.spines["bottom"].set_position(
            ("data", y_axis["max"] - y_axis["step"] / 2)
        )
        self._ax.spines["top"].set_position(
            ("data", y_axis["min"] - y_axis["step"] / 2)
        )
        self._ax.spines["bottom"].set_bounds(
            x_axis["min"] - x_axis["step"] / 2, x_axis["max"] - x_axis["step"] / 2
        )
        self._ax.spines["top"].set_bounds(
            x_axis["min"] - x_axis["step"] / 2, x_axis["max"] - x_axis["step"] / 2
        )

        self._ax.set_xticks(x_majors)
        self._ax.set_xticklabels(xy_labels)


class AxesFormattingVerticalBarhPlot(AxesFormatting):
    _BARWIDTH = 0.6

    def format_axes(self, x_axis, y_axis, x_labels, y_labels, span_limits):
        # Ticks format
        self._set_major_tick_params()

        # Make right and top line invisible
        self._make_top_right_ax_invisible()

        # Limits and ticks for x-axis
        self._set_x_axis(x_axis, y_axis, x_labels, span_limits)

        # Limits and ticks for y-axis
        self._set_y_axis(y_axis, x_axis, y_labels, span_limits)

        # Format labels
        self._format_labels()

    def _set_major_tick_params(self):
        # Distance between ticks and label of ticks

        self._ax.tick_params(
            axis="y", which="major", pad=PLOT_PROP.pad, length=0, direction="in"
        )

        self._ax.tick_params(
            axis="x",
            which="major",
            pad=PLOT_PROP.pad,
            length=PLOT_PROP.ticks_len,
            direction="in",
        )

    def _set_x_axis(self, x_axis, y_axis, x_labels, span_limits):
        x_majors = np.arange(
            x_axis["min"], x_axis["max"] + x_axis["step"] / 2, x_axis["step"]
        )

        # Limits and ticks for x-axis
        self._ax.set_xlim(-0.6, x_axis["max"])
        self._ax.spines["bottom"].set_position(("data", span_limits["bottom"]))
        self._ax.spines["bottom"].set_bounds(x_axis["min"], x_axis["max"])

        self._ax.set_xticks(x_majors)
        self._ax.set_xticklabels(x_labels)

    def _set_y_axis(self, y_axis, x_axis, y_labels, span_limits):
        y_majors = np.arange(
            y_axis["min"], y_axis["max"] + y_axis["step"], y_axis["step"]
        )

        # Limits and ticks for y-axis
        self._ax.set_ylim(
            y_axis["min"] - y_axis["step"] / 2, y_axis["max"] + y_axis["step"] / 2
        )
        self._ax.spines["left"].set_position(("data", x_axis["min"]))
        self._ax.spines["left"].set_bounds(*span_limits["left_span"])

        self._ax.set_yticks(y_majors)
        self._ax.set_yticklabels(y_labels)


class LogHistAxesFormatting(AxesFormatting):

    def format_axes(self, x_axis, y_axis, *args, **kwargs):
        # Ticks format
        self._set_major_tick_params()

        # Make right and top line invisible
        self._make_top_right_ax_invisible()

        # Limits and ticks for x-axis
        self._set_x_axis(x_axis, y_axis)

        # Limits and ticks for y-axis
        self._set_y_axis(y_axis, x_axis)

        # Format labels
        self._format_labels()

    def _set_y_axis(self, y_axis, x_axis, *args, **kwargs):
        order = int(len(str(y_axis["max"]))) + 1
        y_majors = [10 ** i for i in range(1, order)]

        # Limits and ticks for y-axis
        self._ax.set_ylim(1, y_majors[-1])
        self._ax.spines["left"].set_position(("data", x_axis["min"]))
        self._ax.spines["left"].set_bounds(1, y_majors[-1])

        self._ax.set_yticks(y_majors)
