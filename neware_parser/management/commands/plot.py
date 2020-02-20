import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import tensorflow as tf

from .colour_print import Print

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

FIGSIZE = [5, 5]

def plot_vq(plot_params, init_returns):
    barcodes = plot_params["barcodes"]
    count = plot_params["count"]
    fit_args = plot_params["fit_args"]

    degradation_model = init_returns["degradation_model"]
    test_object = init_returns["test_object"]
    all_data = init_returns["all_data"]
    cycles_m = init_returns["cycles_m"]
    cycles_v = init_returns["cycles_v"]
    vol_tensor = init_returns["vol_tensor"]

    x_lim = [-0.1, 1.1]
    y_lim = [2.95, 4.35]

    for barcode_count, barcode in enumerate(barcodes):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        colors = ['k', 'r', 'b', 'g', 'm', 'c']

        for k_count, k in enumerate(test_object[barcode_count].keys()):

            if k[2] == 'dchg':
                sign_change = -1.
            else:
                sign_change = +1.

            barcode_k = all_data[barcode][k][0]

            for vq_count, vq in enumerate(barcode_k['capacity_vector']):

                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.plot(sign_change * vq, vol_tensor.numpy(), c=colors[k_count])


        ax = fig.add_subplot(1, 2, 2)
        colors = [(1., 1., 1.), (1., 0., 0.), (0., 0., 1.),
                  (0., 1., 0.), (1., 0., 1.), (0., 1., 1.)]

        cycles = [0, 2000, 4000, 6000]
        for k_count, k in enumerate(test_object[barcode_count].keys()):

            if k[2] == 'dchg':
                sign_change = -1.
            else:
                sign_change = +1.

            for i, cyc in enumerate(cycles):
                cycle = ((float(cyc) - cycles_m) / tf.sqrt(cycles_v))
                test_results = test_all_voltages(
                    cycle,
                    all_data[barcode][k][1]['avg_constant_current'],
                    all_data[barcode][k][1]['avg_end_current_prev'],
                    all_data[barcode][k][1]['avg_end_voltage_prev'],
                    barcode_count, degradation_model, vol_tensor
                )

                pred_cap = tf.reshape(test_results["pred_cap"], shape = [-1])

                mult = (i + 4) / (len(cycles) + 5)

                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.scatter(
                    sign_change*pred_cap,
                    vol_tensor.numpy(),
                    c = (
                        mult * colors[k_count][0],
                        mult * colors[k_count][1],
                        mult * colors[k_count][2]
                    )
                )


        savefig('VQ_{}_Count_{}.png', fit_args, barcode, count)
        plt.close(fig)

def plot_test_rate_voltage(plot_params, init_returns):
    barcodes = plot_params["barcodes"]
    count = plot_params["count"]
    fit_args = plot_params["fit_args"]

    degradation_model = init_returns["degradation_model"]
    vol_tensor = init_returns["vol_tensor"]

    for barcode_count, barcode in enumerate(barcodes):
        results = []
        for k in [[0.1, x / 10.] for x in range(40)]:
            test_results = test_single_voltage(
                [0.], vol_tensor[0], k, barcode_count, degradation_model
            )
            results.append([k[1], test_results["pred_max_dchg_vol"]])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        results = np.array(results)
        ax.scatter(results[:, 0], results[:, 1])

        plt.savefig(
            os.path.join(
                fit_args['path_to_plots'],
                'Test_Rate_Voltage_{}_Count_{}.png'.format(barcode, count)
            )
        )
        plt.close(fig)

def plot_capacity(plot_params, init_returns):
    barcodes = plot_params["barcodes"]
    count = plot_params["count"]
    fit_args = plot_params["fit_args"]

    degradation_model = init_returns["degradation_model"]
    test_object = init_returns["test_object"]
    all_data = init_returns["all_data"]
    cycles_m = init_returns["cycles_m"]
    cycles_v = init_returns["cycles_v"]
    vol_tensor = init_returns["vol_tensor"]

    for barcode_count, barcode in enumerate(barcodes):
        fig = plt.figure()

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_ylabel("capacity")
        #ax1.set_ylim([0.58, 1.03])

        colors = ['k', 'r', 'b', 'g', 'm', 'c']
        for k_count, k in enumerate(test_object[barcode_count].keys()):

            if k[2] == 'dchg':
                sign_change = -1.
            else:
                sign_change = +1.

            ax1.scatter(
                all_data[barcode][k][0]['cycle_number'],
                sign_change*all_data[barcode][k][0]['capacity_vector'][:, 0],
                c=colors[k_count]
            )

            for cyc_i in [0, -1]:
                cyc = test_object[barcode_count][k][cyc_i]
                ax1.axvline(
                    x=cyc, color=colors[k_count], linestyle='--')

        for k_count, k in enumerate(test_object[barcode_count].keys()):

            if k[2] == 'dchg':
                sign_change = -1.
            else:
                sign_change = +1.

            cycles = test_object[barcode_count][k]
            min_c = min(cycles)
            max_c = max(cycles)
            cycles = [
                 x for x in np.arange(0., 7000., 20.)
            ]

            my_cycles = [
                (cyc - cycles_m) / tf.sqrt(cycles_v) for cyc in cycles
            ]

            test_results = test_single_voltage(
                my_cycles, vol_tensor[0],
                all_data[barcode][k][1]['avg_constant_current'],
                all_data[barcode][k][1]['avg_end_current_prev'],
                all_data[barcode][k][1]['avg_end_voltage_prev'],
                barcode_count, degradation_model
            )
            pred_cap = tf.reshape(test_results["pred_cap"], shape = [-1])

            ax1.plot(cycles, sign_change*pred_cap, c=colors[k_count])

        savefig('Cap_{}_Count_{}.png', fit_args, barcode, count)
        plt.close(fig)


def plot_shift(plot_params, init_returns):
    #TODO(sam): this needs to conform to the new dataset.
    barcodes = plot_params["barcodes"]
    count = plot_params["count"]
    fit_args = plot_params["fit_args"]

    degradation_model = init_returns["degradation_model"]
    test_object = init_returns["test_object"]
    all_data = init_returns["all_data"]
    cycles_m = init_returns["cycles_m"]
    cycles_v = init_returns["cycles_v"]
    vol_tensor = init_returns["vol_tensor"]

    for barcode_count, barcode in enumerate(barcodes):
        fig = plt.figure()

        ax1 = fig.add_subplot(1, 1, 1)

        colors = ['k', 'r', 'b', 'g', 'm', 'c']

        for k_count, k in enumerate(test_object[barcode_count].keys()):
            cycles = test_object[barcode_count][k]
            min_c = min(cycles)
            max_c = max(cycles)
            cycles = [
                float(min_c) + float(max_c - min_c)
                * x for x in np.arange(0., 1.1, 0.02)
            ]

            my_cycles = [
                (cyc - cycles_m) / tf.sqrt(cycles_v) for cyc in cycles
            ]

            test_results = test_single_voltage(
                my_cycles, vol_tensor[0], k, barcode_count, degradation_model
            )

            ax1.plot(cycles, test_results["shift"], c=colors[k_count])
            ax1.set_ylabel("shift")

        savefig('Shift_{}_Count_{}.png', fit_args, barcode, count)
        plt.close(fig)

def plot_eq_vol_and_r(plot_params, init_returns):
    #TODO(sam): this needs to conform to the new dataset.
    barcodes = plot_params["barcodes"]
    count = plot_params["count"]
    fit_args = plot_params["fit_args"]
    all_data = init_returns["all_data"]

    degradation_model = init_returns["degradation_model"]
    test_object = init_returns["test_object"]
    cycles_m = init_returns["cycles_m"]
    cycles_v = init_returns["cycles_v"]
    vol_tensor = init_returns["vol_tensor"]

    for barcode_count, barcode in enumerate(barcodes):
        fig = plt.figure()
        fig.subplots_adjust(wspace = 0.3)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        colors = ['k', 'r', 'b', 'g', 'm', 'c']

        for k_count, k in enumerate(test_object[barcode_count].keys()):

            cycles = test_object[barcode_count][k]
            min_c = min(cycles)
            max_c = max(cycles)
            cycles = [
                float(min_c) + float(max_c - min_c)
                * x for x in np.arange(0., 1.1, 0.02)
            ]

            my_cycles = [
                (cyc - cycles_m) / tf.sqrt(cycles_v) for cyc in cycles
            ]

            test_results = test_single_voltage(
                my_cycles, vol_tensor[0], k, barcode_count, degradation_model
            )
            pred_cap = tf.reshape(test_results["pred_cap"], shape = [-1])

            ax1.plot(cycles, test_results["pred_eq_vol"], c=colors[k_count])
            ax1.set_ylabel("eq_vol")
            ax1.plot(cycles, [4.3 for _ in cycles], c='0.5')
            set_tick_params(ax1)

            ax2.plot(cycles, test_results["pred_r"], c=colors[k_count])
            ax2.set_ylabel("r")
            ax2.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
            ax2.plot(cycles, [0.05 for _ in cycles], c='0.5')
            set_tick_params(ax2)

        savefig('Eq_{}_Count_{}.png', fit_args, barcode, count)
        plt.close(fig)

def test_all_voltages(cycle, constant_current, end_current_prev, end_voltage_prev, barcode_count, degradation_model, voltages):
    expanded_cycles = tf.constant(cycle, shape=[1, 1])
    expanded_constant_current = tf.constant(constant_current, shape=[1, 1])
    expanded_end_current_prev = tf.constant(end_current_prev, shape=[1, 1])
    expanded_end_voltage_prev = tf.constant(end_voltage_prev, shape=[1, 1])

    indecies = tf.reshape(barcode_count, [1])

    return degradation_model(
        (
            expanded_cycles,
            expanded_constant_current,
            expanded_end_current_prev,
            expanded_end_voltage_prev,
            indecies,
            voltages
        ),
        training=False
    )

def test_single_voltage(cycles, v, constant_current, end_current_prev, end_voltage_prev, barcode_count, degradation_model):
    expanded_cycles = tf.expand_dims(cycles, axis=1)
    expanded_constant_current = tf.constant(constant_current, shape=[len(cycles), 1])
    expanded_end_current_prev = tf.constant(end_current_prev, shape=[len(cycles), 1])
    expanded_end_voltage_prev = tf.constant(end_voltage_prev, shape=[len(cycles), 1])

    indecies = tf.tile(tf.expand_dims(barcode_count, axis=0), [len(cycles)])

    return degradation_model(
        (
            expanded_cycles,
            expanded_constant_current,
            expanded_end_current_prev,
            expanded_end_voltage_prev,
            indecies,
            tf.expand_dims(v, axis=0)),
        training=False
    )

def savefig(figname, fit_args, barcode, count):
    plt.savefig(
        os.path.join(
            fit_args['path_to_plots'],
            figname.format(barcode, count)
        ),
        dpi=300
    )

def set_tick_params(ax):
    ax.tick_params(
        direction = 'in',
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
