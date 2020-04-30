import pickle

import matplotlib.patches as mpatches

from neware_parser.Pickle import Pickle

# TODO(harvey) duplicate in plot.py
COLORS = [
    (.4, .4, .4),

    (1., 0., 0.),
    (0., 0., 1.),
    (0., 1., 0.),

    (.6, 0., .6),
    (0., .6, .6),
    (.6, .6, 0.),

    (1., 0., .5),
    (.5, 0., 1.),
    (0., 1., .5),
    (0., .5, 1.),
    (1., .5, 0.),
    (.5, 1., 0.),
]


class PlotEngine:

    @staticmethod
    def quantity_vs_capacity(
        filename: str, fig,
        name = "some quantity", subplot_count = 1, offset = 0,
    ) -> None:

        protocols, quantities, cycles = Pickle.load(filename, 3)
        ax1 = fig.add_subplot(subplot_count, 1, 1 + offset)
        ax1.set_ylabel(name)
        for count, quantity in enumerate(quantities):
            ax1.plot(cycles, quantity, c = COLORS[count])

    @staticmethod
    def resistance(filename: str, fig, offset: int) -> None:

        keys, resistances, cycles = Pickle.load(filename, 3)
        ax1 = fig.add_subplot(6, 1, 1 + offset)
        ax1.set_ylabel("resistance")
        for count, (k, resistance) in enumerate(zip(keys, resistances)):
            ax1.plot(cycles, resistance, c = COLORS[count])

    @staticmethod
    def shift(filename: str, fig, offset: int) -> None:

        keys, shifts, cycles = Pickle.load(filename, 3)
        ax1 = fig.add_subplot(6, 1, 1 + offset)
        ax1.set_ylabel("shift")
        for count, (k, shift) in enumerate(zip(keys, shifts)):
            ax1.plot(cycles, shift, c = COLORS[count])

    @staticmethod
    def scale(filename: str, fig, offset: int) -> None:
        """ Plot scale from the given pickle

        Args:
            filename (str): Filename (including path) to the pickle file
            fig: The figure on which to plot scale
            offset (int): The offset on the figure for the scale plot
        """

        patches, keys, scales, cycles = Pickle.load(filename, 4)
        ax1 = fig.add_subplot(6, 1, 1 + offset)

        for k_count, (k, scale) in enumerate(zip(keys, scales)):
            patches.append(
                mpatches.Patch(color = COLORS[k_count], label = make_legend(k))
            )
            ax1.plot(cycles, scale, c = COLORS[k_count])

        ax1.set_ylabel("scale")
        ax1.legend(
            handles = patches, fontsize = "small",
            bbox_to_anchor = (0.7, 1), loc = "upper left"
        )


# TODO(harvey) duplicate in plot.py
def bake_rate(rate_in):
    rate = round(20. * rate_in) / 20.
    if rate == .05:
        rate = "C/20"
    elif rate > 1.75:
        rate = "{}C".format(int(round(rate)))
    elif rate > 0.4:
        rate = round(2. * rate_in) / 2.
        if rate == 1.:
            rate = "1C"
        elif rate == 1.5:
            rate = "3C/2"
        elif rate == 0.5:
            rate = "C/2"
    elif rate > 0.09:
        if rate == 0.1:
            rate = "C/10"
        elif rate == 0.2:
            rate = "C/5"
        elif rate == 0.35:
            rate = "C/3"
        else:
            rate = "{:1.1f}C".format(rate)
    return rate


# TODO(harvey) duplicate in plot.py
def bake_voltage(vol_in):
    vol = round(10. * vol_in) / 10.
    if vol == 1. or vol == 2. or vol == 3. or vol == 4. or vol == 5.:
        vol = "{}".format(int(vol))
    else:
        vol = "{:1.1f}".format(vol)
    return vol


# TODO(harvey) duplicate in plot.py
def make_legend(key):
    constant_rate = key[0]
    constant_rate = bake_rate(constant_rate)
    end_rate_prev = key[1]
    end_rate_prev = bake_rate(end_rate_prev)
    end_rate = key[2]
    end_rate = bake_rate(end_rate)

    end_voltage = key[3]
    end_voltage = bake_voltage(end_voltage)

    end_voltage_prev = key[4]
    end_voltage_prev = bake_voltage(end_voltage_prev)

    template = "I {}:{}:{:5}   V {}:{}"
    return template.format(
        end_rate_prev, constant_rate, end_rate, end_voltage_prev, end_voltage
    )
