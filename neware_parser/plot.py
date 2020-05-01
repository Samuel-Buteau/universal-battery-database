import os
import sys

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from neware_parser.Print import Print
from neware_parser.Key import Key
from neware_parser.PlotEngine import PlotEngine
from neware_parser.DataEngine import DataEngine
from neware_parser.Pickle import Pickle

matplotlib_axes_logger.setLevel("ERROR")

FIGSIZE = [6, 5]

# TODO(harvey) duplicate in PlotEngine.py
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


# TODO(harvey) duplicate in PlotEngine.py
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


# TODO(harvey) duplicate in PlotEngine.py
def bake_voltage(vol_in):
    vol = round(10. * vol_in) / 10.
    if vol == 1. or vol == 2. or vol == 3. or vol == 4. or vol == 5.:
        vol = "{}".format(int(vol))
    else:
        vol = "{:1.1f}".format(vol)
    return vol


# TODO(harvey) duplicate in PlotEngine.py
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


def get_svit_and_count(my_data, barcode):
    n_sign = len(my_data["sign_grid"])
    n_voltage = len(my_data["voltage_grid"])
    n_current = len(my_data["current_grid"])
    n_temperature = len(my_data["temperature_grid"])

    count_matrix = np.reshape(
        my_data[Key.ALL_DATA][barcode]["all_reference_mats"]
        [Key.COUNT_MATRIX][-1],
        [n_sign, n_voltage, n_current, n_temperature, 1]
    )

    svit_grid = np.concatenate(
        (
            np.tile(
                np.reshape(my_data["sign_grid"], [n_sign, 1, 1, 1, 1]),
                [1, n_voltage, n_current, n_temperature, 1]
            ),
            np.tile(
                np.reshape(my_data["voltage_grid"], [1, n_voltage, 1, 1, 1]),
                [n_sign, 1, n_current, n_temperature, 1]
            ),
            np.tile(
                np.reshape(my_data["current_grid"], [1, 1, n_current, 1, 1]),
                [n_sign, n_voltage, 1, n_temperature, 1]
            ),
            np.tile(
                np.reshape(
                    my_data["temperature_grid"],
                    [1, 1, 1, n_temperature, 1]
                ),
                [n_sign, n_voltage, n_current, 1, 1]
            ),
        ),
        axis = -1
    )
    return {Key.SVIT_GRID: svit_grid, Key.COUNT_MATRIX: count_matrix}


def plot_vq(plot_params, init_returns):
    barcodes\
        = plot_params["barcodes"][:plot_params[Key.FIT_ARGS]["barcode_show"]]
    protocol_count = plot_params["count"]
    fit_args = plot_params[Key.FIT_ARGS]

    degradation_model = init_returns[Key.MODEL]
    my_data = init_returns[Key.MY_DATA]
    cycle_m = init_returns[Key.CYC_M]
    cycle_v = init_returns[Key.CYC_V]
    x_lim = [-0.01, 1.01]
    y_lim = [2.95, 4.35]

    for barcode_count, barcode in enumerate(barcodes):

        fig, axs = plt.subplots(nrows = 3, figsize = [5, 10], sharex = True)
        cyc_grp_dict = my_data[Key.ALL_DATA][barcode][Key.CYC_GRP_DICT]

        for step, offset, mode, x_leg, y_leg in [
            ("dchg", 0, "cc", 0.5, 1),
            ("chg", 1, "cc", 0.5, 0.5),
            ("chg", 2, "cv", 0., 0.5)
        ]:
            protocols = get_protocols(cyc_grp_dict, step)

            patches = []
            ax = axs[offset]
            for protocol_count, protocol in enumerate(protocols):
                if protocol[-1] == "dchg":
                    sign_change = -1.
                else:
                    sign_change = +1.

                barcode_k = cyc_grp_dict[protocol][Key.MAIN]

                if mode == "cc":
                    capacity_tensor = barcode_k["cc_capacity_vector"]
                elif mode == "cv":
                    capacity_tensor = barcode_k["cv_capacity_vector"]
                else:
                    sys.exit("Unknown mode in plot_vq.")

                for vq_count, vq in enumerate(capacity_tensor):
                    cycle = barcode_k[Key.N][vq_count]

                    mult = 1. - (.5 * float(cycle) / 6000.)

                    if mode == "cc":
                        vq_mask = barcode_k["cc_mask_vector"][vq_count]
                        y_axis = barcode_k["cc_voltage_vector"][vq_count]
                        y_lim = [2.95, 4.35]
                    elif mode == "cv":
                        vq_mask = barcode_k["cv_mask_vector"][vq_count]
                        y_axis = barcode_k["cv_current_vector"][vq_count]
                        y_lim = [
                            min([protocol[2] for protocol in protocols]) - 0.05,
                            0.05 + max([protocol[0] for protocol in protocols])
                        ]
                    else:
                        sys.exit("Unknown mode in plot_vq.")

                    valids = vq_mask > .5

                    ax.set_xlim(x_lim)
                    ax.set_ylim(y_lim)

                    ax.scatter(
                        sign_change * vq[valids],
                        y_axis[valids],
                        c = [[
                            mult * COLORS[protocol_count][0],
                            mult * COLORS[protocol_count][1],
                            mult * COLORS[protocol_count][2]
                        ]],
                        s = 3
                    )

            cycles = [0, 6000 / 2, 6000]
            for protocol_count, protocol in enumerate(protocols):
                patches.append(mpatches.Patch(
                    color = COLORS[protocol_count],
                    label = make_legend(protocol)
                ))

                if protocol[-1] == "dchg":
                    sign_change = -1.
                else:
                    sign_change = +1.

                v_min = min(protocol[3], protocol[4])
                v_max = max(protocol[3], protocol[4])
                v_range = np.arange(v_min, v_max, 0.05)
                curr_min = abs(cyc_grp_dict[protocol][Key.I_CC_AVG])
                curr_max = abs(cyc_grp_dict[protocol]["avg_end_current"])

                if curr_min == curr_max:
                    current_range = np.array([curr_min])
                else:
                    current_range = sign_change * np.exp(
                        np.arange(
                            np.log(curr_min),
                            np.log(curr_max),
                            .05 * (np.log(curr_max) - np.log(curr_min))
                        )
                    )

                svit_and_count = get_svit_and_count(my_data, barcode)

                for cycle_count, cycle in enumerate(cycles):
                    scaled_cyc = ((float(cycle) - cycle_m) / tf.sqrt(cycle_v))
                    mult = 1. - (.5 * float(cycle) / 6000.)

                    test_results = test_all_voltages(
                        scaled_cyc,
                        cyc_grp_dict[protocol][Key.I_CC_AVG],
                        cyc_grp_dict[protocol][Key.I_PREV_END_AVG],
                        cyc_grp_dict[protocol][Key.V_PREV_END_AVG],
                        cyc_grp_dict[protocol][Key.V_END_AVG],
                        barcode_count,
                        degradation_model,
                        v_range,
                        current_range,
                        svit_and_count[Key.SVIT_GRID],
                        svit_and_count[Key.COUNT_MATRIX],
                    )

                    if mode == "cc":
                        pred_cap = tf.reshape(
                            test_results["pred_cc_capacity"], shape = [-1]
                        )
                        yrange = v_range
                    elif mode == "cv":
                        yrange = current_range
                        pred_cap = tf.reshape(
                            test_results["pred_cv_capacity"], shape = [-1]
                        )
                    else:
                        sys.exit("Unknown mode in plot_vq.")

                    ax.set_xlim(x_lim)
                    if mode == "cc":
                        ax.set_ylim(y_lim)
                    ax.plot(
                        sign_change * pred_cap,
                        yrange,
                        c = (
                            mult * COLORS[protocol_count][0],
                            mult * COLORS[protocol_count][1],
                            mult * COLORS[protocol_count][2],
                        ),
                    )

            ax.legend(
                handles = patches, fontsize = "small",
                bbox_to_anchor = (x_leg, y_leg), loc = "upper left"
            )
            ax.set_ylabel(step + "-" + mode)

        axs[2].set_xlabel("pred_cap")
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0)
        savefig("VQ_{}_Count_{}.png".format(barcode, protocol_count), fit_args)
        plt.close(fig)


def plot_measured(cyc_grp_dict, mode, protocols, patches, ax1):
    caps = []
    for count, protocol in enumerate(protocols):

        if protocol[-1] == "dchg":
            sign_change = -1.
        else:
            sign_change = +1.

        if mode == "cc":
            cap = cyc_grp_dict[protocol][Key.MAIN]["last_cc_capacity"]
        elif mode == "cv":
            cap = cyc_grp_dict[protocol][Key.MAIN]["last_cv_capacity"]
        else:
            sys.exit("Unknown mode in measured.")

        caps.append(sign_change * cap)

    for count, (protocol, cap) in enumerate(zip(protocols, caps)):
        patches.append(
            mpatches.Patch(color = COLORS[count], label = make_legend(protocol))
        )
        ax1.scatter(
            cyc_grp_dict[protocol][Key.MAIN][Key.N],
            cap,
            c = COLORS[count],
            s = 5,
            label = make_legend(protocol)
        )


def plot_predicted(
    cyc_grp_dict, mode, protocols, cycle_m, cycle_v, barcode_count,
    degradation_model, svit_and_count, ax1,
):
    cycles = [x for x in np.arange(0., 6000., 20.)]
    my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycles]
    caps = []

    for count, protocol in enumerate(protocols):

        if protocol[-1] == "dchg":
            sign_change = -1.
        else:
            sign_change = +1.

        if mode == "cc":
            target_voltage = cyc_grp_dict[protocol]["avg_last_cc_voltage"]
            target_currents = [cyc_grp_dict[protocol][Key.I_CC_AVG]]
        elif mode == "cv":
            target_voltage = cyc_grp_dict[protocol][Key.V_END_AVG]
            curr_min = abs(cyc_grp_dict[protocol][Key.I_CC_AVG])
            curr_max = abs(cyc_grp_dict[protocol]["avg_end_current"])

            if curr_min == curr_max:
                target_currents = np.array([curr_min])
            else:
                target_currents = sign_change * np.exp(
                    np.arange(
                        np.log(curr_min),
                        np.log(curr_max),
                        .05 * (np.log(curr_max) - np.log(curr_min))
                    )
                )
        else:
            sys.exit("Unknown mode in predicted.")

        test_results = test_single_voltage(
            my_cycle,
            target_voltage,
            cyc_grp_dict[protocol][Key.I_CC_AVG],
            cyc_grp_dict[protocol][Key.I_PREV_END_AVG],
            cyc_grp_dict[protocol][Key.V_PREV_END_AVG],
            cyc_grp_dict[protocol][Key.V_END_AVG],
            target_currents,
            barcode_count, degradation_model,
            svit_and_count[Key.SVIT_GRID],
            svit_and_count[Key.COUNT_MATRIX]
        )

        if mode == "cc":
            pred_cap = tf.reshape(
                test_results["pred_cc_capacity"], shape = [-1],
            )
        elif mode == "cv":
            pred_cap = test_results["pred_cv_capacity"].numpy()[:, -1]
        else:
            sys.exit("Unknown mode in predicted.")

        caps.append(sign_change * pred_cap)

    for count, pred_cap in enumerate(caps):
        ax1.plot(cycles, pred_cap, c = COLORS[count])


def plot_capacities(
    cyc_grp_dict, cycle_m, cycle_v, barcode_count,
    degradation_model, svit_and_count, fig,
):
    for typ, off, mode in [
        ("dchg", 0, "cc"), ("chg", 1, "cc"), ("chg", 2, "cv")
    ]:
        patches = []
        keys = get_protocols(cyc_grp_dict, typ)

        ax1 = fig.add_subplot(6, 1, 1 + off)
        ax1.set_ylabel(mode + "-" + typ + "-capacity")

        plot_measured(
            cyc_grp_dict, mode, keys, patches, ax1,
        )

        plot_predicted(
            cyc_grp_dict, mode, keys, cycle_m, cycle_v,
            barcode_count, degradation_model, svit_and_count,
            ax1,
        )

        ax1.legend(
            handles = patches,
            fontsize = "small",
            bbox_to_anchor = (0.7, 1),
            loc = "upper left",
        )


def plot_things_vs_cycle_number(plot_params, init_returns):
    barcodes\
        = plot_params["barcodes"][:plot_params[Key.FIT_ARGS]["barcode_show"]]
    count = plot_params["count"]
    fit_args = plot_params[Key.FIT_ARGS]

    degradation_model = init_returns[Key.MODEL]
    my_data = init_returns[Key.MY_DATA]
    cycle_m = init_returns[Key.CYC_M]
    cycle_v = init_returns[Key.CYC_V]

    # for each cell, plot the quantities of interest
    for barcode_count, barcode in enumerate(barcodes):
        cyc_grp_dict = my_data[Key.ALL_DATA][barcode][Key.CYC_GRP_DICT]
        svit_and_count = get_svit_and_count(my_data, barcode)

        """ Computing """

        scale_pickle_file = os.path.join(
            fit_args[Key.PATH_PLOTS],
            "scale_{}_count_{}.pickle".format(barcode, count),
        )
        scale_data = DataEngine.scale(
            degradation_model, barcode_count,
            cyc_grp_dict, cycle_m, cycle_v, svit_and_count
        )
        Pickle.dump(scale_pickle_file, scale_data)

        resistance_pickle_file = os.path.join(
            fit_args[Key.PATH_PLOTS],
            "resistance_{}_count_{}.pickle".format(barcode, count),
        )
        resistance_data = DataEngine.resistance(
            degradation_model, barcode_count,
            cyc_grp_dict, cycle_m, cycle_v, svit_and_count
        )
        Pickle.dump(resistance_pickle_file, resistance_data)

        shift_pickle_file = os.path.join(
            fit_args[Key.PATH_PLOTS],
            "shift_{}_count_{}.pickle".format(barcode, count),
        )
        shift_data = DataEngine.shift(
            degradation_model, barcode_count,
            cyc_grp_dict, cycle_m, cycle_v, svit_and_count
        )
        Pickle.dump(shift_pickle_file, shift_data)
        """ Plot Data """

        fig = plt.figure(figsize = [11, 10])

        # TODO(harvey) separate into DataEngine and PlotEngine
        plot_capacities(
            cyc_grp_dict, cycle_m, cycle_v, barcode_count,
            degradation_model, svit_and_count,
            fig,
        )

        scale_data = Pickle.load(scale_pickle_file)
        PlotEngine.scale(
            scale_data["cycles"], scale_data["scales"],
            scale_data["protocols"], scale_data["patches"],
            fig.add_subplot(6, 1, 4),
        )

        resistance_data = Pickle.load(resistance_pickle_file)
        PlotEngine.quantity_vs_capacity(
            resistance_data["resistances"], resistance_data["cycles"],
            fig.add_subplot(6, 1, 5), name = "resistance",
        )

        shift_data = Pickle.load(shift_pickle_file)
        PlotEngine.quantity_vs_capacity(
            shift_data["shifts"], shift_data["cycles"],
            fig.add_subplot(6, 1, 6), name = "shift",
        )

        savefig("Cap_{}_Count_{}.png".format(barcode, count), fit_args)
        plt.close(fig)


def test_all_voltages(
    cycle, constant_current, end_current_prev, end_voltage_prev, end_voltage,
    barcode_count, degradation_model,
    voltages, currents, svit_grid, count_matrix,
):
    expanded_cycle = tf.constant(cycle, shape = [1, 1])
    expanded_constant_current = tf.constant(constant_current, shape = [1, 1])
    expanded_end_current_prev = tf.constant(end_current_prev, shape = [1, 1])
    expanded_end_voltage_prev = tf.constant(end_voltage_prev, shape = [1, 1])
    expanded_end_voltage = tf.constant(end_voltage, shape = [1, 1])

    expanded_svit_grid = tf.constant([svit_grid])
    expanded_count_matrix = tf.constant([count_matrix])

    indices = tf.reshape(barcode_count, [1])

    return degradation_model(
        (
            expanded_cycle,
            expanded_constant_current,
            expanded_end_current_prev,
            expanded_end_voltage_prev,
            expanded_end_voltage,
            indices,
            tf.reshape(voltages, [1, len(voltages)]),
            tf.reshape(currents, [1, len(currents)]),
            expanded_svit_grid,
            expanded_count_matrix,
        ),
        training = False
    )


# TODO(harvey): duplicate in DataEngine.py
def test_single_voltage(
    cycle, target_voltage, constant_current, end_current_prev,
    end_voltage_prev, end_voltage, target_currents, barcode_count,
    degradation_model, svit_grid, count_matrix,
):
    expanded_cycle = tf.expand_dims(cycle, axis = 1)
    expanded_constant_current = tf.constant(
        constant_current, shape = [len(cycle), 1],
    )
    expanded_end_current_prev = tf.constant(
        end_current_prev, shape = [len(cycle), 1],
    )
    expanded_end_voltage_prev = tf.constant(
        end_voltage_prev, shape = [len(cycle), 1],
    )
    expanded_end_voltage = tf.constant(end_voltage, shape = [len(cycle), 1])

    indices = tf.tile(tf.expand_dims(barcode_count, axis = 0), [len(cycle)])

    expanded_svit_grid = tf.tile(
        tf.constant([svit_grid]), [len(cycle), 1, 1, 1, 1, 1],
    )
    expanded_count_matrix = tf.tile(
        tf.constant([count_matrix]), [len(cycle), 1, 1, 1, 1, 1],
    )

    return degradation_model(
        (
            expanded_cycle,
            expanded_constant_current,
            expanded_end_current_prev,
            expanded_end_voltage_prev,
            expanded_end_voltage,
            indices,
            tf.constant(target_voltage, shape = [len(cycle), 1]),
            tf.tile(
                tf.reshape(target_currents, shape = [1, len(target_currents)]),
                [len(cycle), 1]
            ),
            expanded_svit_grid,
            expanded_count_matrix,
        ),
        training = False
    )


def savefig(fig_name, fit_args):
    plt.savefig(
        os.path.join(fit_args[Key.PATH_PLOTS], fig_name), dpi = 300
    )


def set_tick_params(ax):
    ax.tick_params(
        direction = "in",
        length = 3,
        width = 1,
        labelsize = 12,
        bottom = True,
        top = True,
        left = True,
        right = True
    )


def get_nearest_point(xys, y):
    best = xys[0, :]
    best_distance = (best[1] - y) ** 2.
    for i in range(len(xys)):
        new_distance = (xys[i, 1] - y) ** 2.
        if best_distance > new_distance:
            best_distance = new_distance
            best = xys[i, :]

    return best


# TODO(harvey): duplicate in DataEngine.py
def get_protocols(cyc_grp_dict: dict, step: str) -> list:
    """
    Args:
        cyc_grp_dict (dict)
        step (str): Specifies charge or discharge
    Returns:
        list: Keys representing charge/discharge configurations
    """

    keys = [key for key in cyc_grp_dict.keys() if key[-1] == step]
    keys.sort(
        key = lambda k: (
            round(20. * k[0]), round(20. * k[1]), round(20. * k[2]),
            round(20. * k[3]), round(20. * k[4])
        )
    )
    return keys
