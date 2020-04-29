import os
import pickle

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from neware_parser.Print import Print
from neware_parser.Key import Key

matplotlib_axes_logger.setLevel("ERROR")

FIGSIZE = [6, 5]

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


def bake_voltage(vol_in):
    vol = round(10. * vol_in) / 10.
    if vol == 1. or vol == 2. or vol == 3. or vol == 4. or vol == 5.:
        vol = "{}".format(int(vol))
    else:
        vol = "{:1.1f}".format(vol)
    return vol


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
    count = plot_params["count"]
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

        for typ, off, mode, x_leg, y_leg in [
            ("dchg", 0, "cc", 0.5, 1),
            ("chg", 1, "cc", 0.5, 0.5),
            ("chg", 2, "cv", 0., 0.5)
        ]:
            list_of_keys = make_keys(cyc_grp_dict, typ)

            list_of_patches = []
            ax = axs[off]
            for k_count, k in enumerate(list_of_keys):
                if k[-1] == "dchg":
                    sign_change = -1.
                else:
                    sign_change = +1.

                barcode_k = cyc_grp_dict[k][Key.MAIN]

                if mode == "cc":
                    capacity_tensor = barcode_k["cc_capacity_vector"]
                elif mode == "cv":
                    capacity_tensor = barcode_k["cv_capacity_vector"]

                for vq_count, vq in enumerate(capacity_tensor):
                    cyc = barcode_k[Key.N][vq_count]

                    mult = 1. - (.5 * float(cyc) / 6000.)

                    if mode == "cc":
                        vq_mask = barcode_k["cc_mask_vector"][vq_count]
                        y_axis = barcode_k["cc_voltage_vector"][vq_count]
                        y_lim = [2.95, 4.35]
                    elif mode == "cv":
                        vq_mask = barcode_k["cv_mask_vector"][vq_count]
                        y_axis = barcode_k["cv_current_vector"][vq_count]
                        y_lim = [
                            min([key[2] for key in list_of_keys]) - 0.05,
                            0.05 + max([key[0] for key in list_of_keys])
                        ]

                    valids = vq_mask > .5

                    ax.set_xlim(x_lim)
                    ax.set_ylim(y_lim)

                    ax.scatter(
                        sign_change * vq[valids],
                        y_axis[valids],
                        c = [[
                            mult * COLORS[k_count][0],
                            mult * COLORS[k_count][1],
                            mult * COLORS[k_count][2]
                        ]],
                        s = 3
                    )

            cycle = [0, 6000 / 2, 6000]
            for k_count, k in enumerate(list_of_keys):
                list_of_patches.append(mpatches.Patch(
                    color = COLORS[k_count], label = make_legend(k)
                ))

                if k[-1] == "dchg":
                    sign_change = -1.
                else:
                    sign_change = +1.

                v_min = min(k[3], k[4])
                v_max = max(k[3], k[4])
                v_range = np.arange(v_min, v_max, 0.05)
                curr_min = abs(cyc_grp_dict[k][Key.I_CC_AVG])
                curr_max = abs(cyc_grp_dict[k]["avg_end_current"])

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

                for i, cyc in enumerate(cycle):
                    scaled_cyc = ((float(cyc) - cycle_m) / tf.sqrt(cycle_v))
                    mult = 1. - (.5 * float(cyc) / 6000.)

                    test_results = test_all_voltages(
                        scaled_cyc,
                        cyc_grp_dict[k][Key.I_CC_AVG],
                        cyc_grp_dict[k][Key.I_PREV_END_AVG],
                        cyc_grp_dict[k][Key.V_PREV_END_AVG],
                        cyc_grp_dict[k][Key.V_END_AVG],
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

                    ax.set_xlim(x_lim)
                    if mode == "cc":
                        ax.set_ylim(y_lim)
                    ax.plot(
                        sign_change * pred_cap,
                        yrange,
                        c = (
                            mult * COLORS[k_count][0],
                            mult * COLORS[k_count][1],
                            mult * COLORS[k_count][2],
                        ),
                    )

            ax.legend(
                handles = list_of_patches, fontsize = "small",
                bbox_to_anchor = (x_leg, y_leg), loc = "upper left"
            )
            ax.set_ylabel(typ + "-" + mode)

        axs[2].set_xlabel("pred_cap")
        fig.tight_layout()
        fig.subplots_adjust(hspace = 0)
        savefig("VQ_{}_Count_{}.png".format(barcode, count), fit_args)
        plt.close(fig)


def plot_measured(cyc_grp_dict, mode, list_of_keys, list_of_patches, ax1):
    for k_count, k in enumerate(list_of_keys):
        list_of_patches.append(
            mpatches.Patch(
                color = COLORS[k_count], label = make_legend(k)
            )
        )

        main_data = cyc_grp_dict[k][Key.MAIN]

        if k[-1] == "dchg":
            sign_change = -1.
        else:
            sign_change = +1.

        if mode == "cc":
            cap = sign_change * main_data["last_cc_capacity"]
        elif mode == "cv":
            cap = sign_change * main_data["last_cv_capacity"]

        ax1.scatter(
            main_data[Key.N],
            cap,
            c = COLORS[k_count],
            s = 5,
            label = make_legend(k)
        )


def plot_predicted(
    cyc_grp_dict, mode, list_of_keys, cycle_m, cycle_v, barcode_count,
    degradation_model, svit_and_count, ax1,
):
    for k_count, k in enumerate(list_of_keys):

        if k[-1] == "dchg":
            sign_change = -1.
        else:
            sign_change = +1.

        cycle = [x for x in np.arange(0., 6000., 20.)]

        my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycle]

        if mode == "cc":
            target_voltage = cyc_grp_dict[k]["avg_last_cc_voltage"]
            target_currents = [cyc_grp_dict[k][Key.I_CC_AVG]]
        elif mode == "cv":
            target_voltage = cyc_grp_dict[k][Key.V_END_AVG]
            curr_min = abs(cyc_grp_dict[k][Key.I_CC_AVG])
            curr_max = abs(cyc_grp_dict[k]["avg_end_current"])

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

        test_results = test_single_voltage(
            my_cycle,
            target_voltage,
            cyc_grp_dict[k][Key.I_CC_AVG],
            cyc_grp_dict[k][Key.I_PREV_END_AVG],
            cyc_grp_dict[k][Key.V_PREV_END_AVG],
            cyc_grp_dict[k][Key.V_END_AVG],
            target_currents,
            barcode_count, degradation_model,
            svit_and_count[Key.SVIT_GRID],
            svit_and_count[Key.COUNT_MATRIX]
        )

        if mode == "cc":
            pred_cap = tf.reshape(
                test_results["pred_cc_capacity"],
                shape = [-1]
            )
        elif mode == "cv":
            pred_cap = test_results["pred_cv_capacity"].numpy()[:, -1]

        ax1.plot(cycle, sign_change * pred_cap, c = COLORS[k_count])


def plot_capacities(
    cyc_grp_dict, cycle_m, cycle_v, barcode_count,
    degradation_model, svit_and_count, fig,
):
    for typ, off, mode in [
        ("dchg", 0, "cc"), ("chg", 1, "cc"), ("chg", 2, "cv")
    ]:
        list_of_patches = []
        list_of_keys = make_keys(cyc_grp_dict, typ)

        ax1 = fig.add_subplot(6, 1, 1 + off)
        ax1.set_ylabel(mode + "-" + typ + "-capacity")

        plot_measured(
            cyc_grp_dict, mode, list_of_keys, list_of_patches, ax1,
        )

        plot_predicted(
            cyc_grp_dict, mode, list_of_keys, cycle_m, cycle_v,
            barcode_count, degradation_model, svit_and_count,
            ax1,
        )

        ax1.legend(
            handles = list_of_patches,
            fontsize = "small",
            bbox_to_anchor = (0.7, 1),
            loc = "upper left",
        )


def compute_scale(
    degradation_model, barcode_count, cyc_grp_dict, cycle_m, cycle_v,
    svit_and_count, step, mode, filename,
) -> None:
    """ Compute the predicted scale and save it in a pickle file

    Args:
        cyc_grp_dict (dict)
        cycle_m: Cycle mean.
        cycle_v: Cycle variance.
        barcode_count: Number of barcodes.
        degradation_model
        svit_and_count: TODO(harvey)
        filename: Filename (including path) to save the generated pickle file
    """

    step, mode = "dchg", "cc"

    patches = []
    keys = make_keys(cyc_grp_dict, step)
    scales = []
    cycles = [x for x in np.arange(0., 6000., 20.)]

    for k_count, k in enumerate(keys):
        my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycles]

        target_voltage = cyc_grp_dict[k]["avg_last_cc_voltage"]
        target_currents = [cyc_grp_dict[k][Key.I_CC_AVG]]

        test_results = test_single_voltage(
            my_cycle,
            target_voltage,
            cyc_grp_dict[k][Key.I_CC_AVG],
            cyc_grp_dict[k][Key.I_PREV_END_AVG],
            cyc_grp_dict[k][Key.V_PREV_END_AVG],
            cyc_grp_dict[k][Key.V_END_AVG],
            target_currents,
            barcode_count, degradation_model,
            svit_and_count[Key.SVIT_GRID],
            svit_and_count[Key.COUNT_MATRIX],
        )
        scales.append(tf.reshape(test_results["pred_scale"], shape = [-1]))

    pickle_dump_scale(filename, patches, keys, scales, cycles)


def pickle_dump_scale(filename: str, patches, keys, scales, cycles) -> None:
    f = open(filename, "wb+")
    pickle.dump(patches, f)
    pickle.dump(keys, f)
    pickle.dump(scales, f)
    pickle.dump(cycles, f)
    f.close()


def pickle_load_scale(filename: str) -> tuple:
    f = open(filename, "rb")
    patches = pickle.load(f)
    keys = pickle.load(f)
    scales = pickle.load(f)
    cycles = pickle.load(f)
    f.close()

    return patches, keys, scales, cycles


def plot_scale(filename: str, fig, offset: int) -> None:
    """ Plot scale from the given pickle

    Args:
        filename (str): Filename (including path) to the pickle file
        fig: The figure on which to plot scale
        offset (int): The offset on the figure for the scale plot
    """

    patches, keys, scales, cycles = pickle_load_scale(filename)
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


def plot_resistance(
    cyc_grp_dict, cycle_m, cycle_v, barcode_count,
    degradation_model, svit_and_count, fig,
):
    for typ, off, mode in [("dchg", 4, "cc")]:

        list_of_keys = make_keys(cyc_grp_dict, typ)

        ax1 = fig.add_subplot(6, 1, 1 + off)
        ax1.set_ylabel("resistance")

        for k_count, k in enumerate(list_of_keys):
            cycle = [x for x in np.arange(0., 6000., 20.)]

            my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycle]

            target_voltage = cyc_grp_dict[k]["avg_last_cc_voltage"]
            target_currents = [cyc_grp_dict[k][Key.I_CC_AVG]]

            test_results = test_single_voltage(
                my_cycle,
                target_voltage,
                cyc_grp_dict[k][Key.I_CC_AVG],
                cyc_grp_dict[k][Key.I_PREV_END_AVG],
                cyc_grp_dict[k][Key.V_PREV_END_AVG],
                cyc_grp_dict[k][Key.V_END_AVG],
                target_currents,
                barcode_count, degradation_model,
                svit_and_count[Key.SVIT_GRID],
                svit_and_count[Key.COUNT_MATRIX]
            )

            pred_cap = tf.reshape(test_results["pred_R"], shape = [-1])

            ax1.plot(cycle, pred_cap, c = COLORS[k_count])


def plot_shift(
    cyc_grp_dict, cycle_m, cycle_v, barcode_count,
    degradation_model, svit_and_count,
    fig,
):
    for typ, off, mode in [("dchg", 5, "cc")]:

        keys = make_keys(cyc_grp_dict, typ)
        shifts = []
        # TODO(harvey): Why not use np array instead of casting to list?
        cycle = [x for x in np.arange(0., 6000., 20.)]
        my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycle]

        for k_count, k in enumerate(keys):
            test_results = test_single_voltage(
                my_cycle,
                cyc_grp_dict[k]["avg_last_cc_voltage"],  # target voltage
                cyc_grp_dict[k][Key.I_CC_AVG],
                cyc_grp_dict[k][Key.I_PREV_END_AVG],
                cyc_grp_dict[k][Key.V_PREV_END_AVG],
                cyc_grp_dict[k][Key.V_END_AVG],
                [cyc_grp_dict[k][Key.I_CC_AVG]],  # target currents
                barcode_count, degradation_model,
                svit_and_count[Key.SVIT_GRID],
                svit_and_count[Key.COUNT_MATRIX],
            )
            shifts.append(tf.reshape(test_results["pred_shift"], shape = [-1]))

        ax1 = fig.add_subplot(6, 1, 1 + off)
        ax1.set_ylabel("shift")
        for count, (k, shift) in enumerate(zip(keys, shifts)):
            ax1.plot(cycle, shift, c = COLORS[count])


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
        svit_and_count = get_svit_and_count(my_data, barcode)
        fig = plt.figure(figsize = [11, 10])
        cyc_grp_dict = my_data[Key.ALL_DATA][barcode][Key.CYC_GRP_DICT]

        plot_capacities(
            cyc_grp_dict, cycle_m, cycle_v, barcode_count,
            degradation_model, svit_and_count,
            fig,
        )

        scale_pickle_file = os.path.join(
            fit_args[Key.PATH_PLOTS],
            "scale_{}_count_{}.pickle".format(barcode, count),
        )
        compute_scale(
            degradation_model, barcode_count, cyc_grp_dict, cycle_m, cycle_v,
            svit_and_count, "dchg", "cc", scale_pickle_file,
        )
        plot_scale(filename = scale_pickle_file, fig = fig, offset = 3)
        plot_resistance(
            cyc_grp_dict, cycle_m, cycle_v, barcode_count,
            degradation_model, svit_and_count,
            fig,
        )
        plot_shift(
            cyc_grp_dict, cycle_m, cycle_v, barcode_count,
            degradation_model, svit_and_count,
            fig,
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

    indecies = tf.tile(tf.expand_dims(barcode_count, axis = 0), [len(cycle)])

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
            indecies,
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


def plot_v_curves(plot_params, init_returns):
    # for now, this is a 2 by 2 plot.
    barcodes\
        = plot_params["barcodes"][:plot_params[Key.FIT_ARGS]["barcode_show"]]
    count = plot_params["count"]
    fit_args = plot_params[Key.FIT_ARGS]
    degradation_model = init_returns[Key.MODEL]
    shift = np.linspace(start = -.2, stop = .2, num = 9, dtype = np.float32)
    v = np.linspace(start = 2., stop = 5.5, num = 64, dtype = np.float32)
    q = np.linspace(start = -0.25, stop = 1.25, num = 64, dtype = np.float32)
    for current in [0.05, 3.]:
        for barcode in barcodes:

            fig, axs = plt.subplots(
                nrows = 3, ncols = 3, figsize = [10, 10],
                sharex = True, sharey = True,
            )

            gathered_axs = []
            for ax_i in axs:
                for ax_j in ax_i:
                    gathered_axs.append(ax_j)

            res = degradation_model.get_v_curves(
                barcode = barcode,
                shift = tf.constant(shift),
                v = tf.constant(v),
                q = tf.constant(q),
                current = tf.constant(current, shape = [1, 1])
            )
            for j in range(len(shift)):
                ax = gathered_axs[j]

                ax.plot(q, res["v_plus"], label = "v_+")
                ax.plot(q, res["v_minus"][j], label = "v_-")
                ax.plot(q, res["v"][j], label = "v_full")
                ax.plot(res["q"][j], v, label = "q_full (inverted)")
                ax.axvline(shift[j], 0, 1)

            ax.legend()

            fig.tight_layout()
            savefig(
                "v_curves_{}_{}_I{}.png".format(barcode, count, current),
                fit_args,
            )
            plt.close(fig)


def make_keys(cyc_grp_dict: dict, step: str) -> list:
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
