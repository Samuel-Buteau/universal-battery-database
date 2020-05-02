import matplotlib.patches as mpatches

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

    # TODO(harvey):
    #   We have now smaller plot functions. Need to make a generic plot function
    #   that calls all of these smaller plot functions
    #   (that are not yet complete). Make generic plot function when all smaller
    #   plot functions are done.

    @staticmethod
    def quantities_vs_capacity(
        barcode, barcode_count, cyc_grp_dict, svit_and_count,
        quantities, cycles, quantities_names,
    ):
        """ Generic plot for cycle number on the x-axis.

        Given a dictionaries of dictionaries, where each dictionary specifies a
        quantity to plot vs cycle.
        """

    @staticmethod
    def protocol_independent_vs_capacity(
        quantity, cycles, ax1, name = "some quantity",
    ) -> None:
        ax1.set_ylabel(name)
        ax1.plot(cycles, quantity, c = "k")

    @staticmethod
    def scale(
        cycles, scales, protocols, patches, ax1,
    ) -> None:

        for count, (protocol, scale) in enumerate(zip(protocols, scales)):
            patches.append(
                mpatches.Patch(
                    color = COLORS[count], label = make_legend(protocol)
                )
            )
            ax1.plot(cycles, scale, c = COLORS[count])

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
