from copy import copy

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


class PlotProperties:
    """ """

    # Plot properties
    C_DPI = 300

    # Plot font properties
    DEFAULT_FONT_PATH = "Arial"
    DEFAULT_FONT_SIZE = 8
    DEFAULT_FONT_PROP = font_manager.FontProperties(
        family=DEFAULT_FONT_PATH, size=DEFAULT_FONT_SIZE
    )

    SMALL_FONT_PROP = font_manager.FontProperties(family=DEFAULT_FONT_PATH, size=4)

    LINEWIDTH = 1
    PAD = 1
    LABELPAD = 1
    TICKS_LEN = 2

    def __init__(self):
        pass

    @staticmethod
    def cm2inch(x):
        if type(x) is float or type(x) is int:
            return x * 0.39
        else:
            return tuple(_x * 0.39 for _x in x)

    @property
    def dpi(self):
        return copy(self.C_DPI)

    @property
    def font(self):
        return copy(self.DEFAULT_FONT_PROP)

    @property
    def small_font(self):
        return copy(self.SMALL_FONT_PROP)

    @property
    def font_size(self):
        return copy(self.DEFAULT_FONT_SIZE)

    @property
    def line_width(self):
        return copy(self.LINEWIDTH)

    @property
    def pad(self):
        return copy(self.PAD)

    @property
    def label_pad(self):
        return copy(self.LABELPAD)

    @property
    def ticks_len(self):
        return copy(self.TICKS_LEN)
