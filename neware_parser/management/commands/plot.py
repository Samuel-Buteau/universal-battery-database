import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from .colour_print import Print

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

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

    for barcode_count, barcode in enumerate(barcodes):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        colors = ['k', 'r', 'b', 'g', 'm', 'c']

        for k_count, k in enumerate(test_object[barcode_count].keys()):

            barcode_k = all_data[barcode][k]
            n_samples = len(barcode_k['capacity_vector'])

            for vq_count, vq in enumerate(barcode_k['capacity_vector']):

                ax.plot(vq, vol_tensor.numpy(), c=colors[k_count])

                if vq_count % int(n_samples / 10) == 0:
                    fused_vector = np.stack([vq, vol_tensor.numpy()], axis=1)
                    target_voltage = barcode_k['dchg_maximum_voltage'][vq_count]
                    best_point = get_nearest_point(fused_vector, target_voltage)

                    ax.scatter(
                        [best_point[0]],
                        [best_point[1]],
                        c=colors[k_count],
                        marker='o',
                        s=100
                    )

        ax = fig.add_subplot(1, 2, 2)
        colors = [(1., 1., 1.), (1., 0., 0.), (0., 0., 1.),
                  (0., 1., 0.), (1., 0., 1.), (0., 1., 1.)]

        cycles = [0, 2000, 4000, 6000]
        for k_count, k in enumerate(test_object[barcode_count].keys()):

            for i, cyc in enumerate(cycles):
                cycle = ((float(cyc) - cycles_m) / tf.sqrt(cycles_v))
                test_results = test_all_voltages(
                    cycle, k, barcode_count, degradation_model, vol_tensor
                )

                pred_cap = tf.reshape(test_results["pred_cap"], shape = [-1])

                mult = (i + 4) / (len(cycles) + 5)
                ax.plot(
                    pred_cap,
                    vol_tensor.numpy(),
                    c = (
                        mult * colors[k_count][0],
                        mult * colors[k_count][1],
                        mult * colors[k_count][2]
                    )
                )

                fused_vector = np.stack([pred_cap, vol_tensor.numpy()], axis=1)
                target_voltage = test_results["pred_max_dchg_vol"][0]
                best_point = get_nearest_point(fused_vector, target_voltage)
                ax.scatter(
                    [best_point[0]],
                    [best_point[1]],
                    marker = 'x',
                    s = 100,
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
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        colors = ['k', 'r', 'b', 'g', 'm', 'c']
        for k_count, k in enumerate(test_object[barcode_count].keys()):

            ax1.scatter(
                all_data[barcode][k]['cycle_number'],
                all_data[barcode][k]['capacity_vector'][:, 0],
                c=colors[k_count]
            )
            ax2.scatter(
                all_data[barcode][k]['cycle_number'],
                all_data[barcode][k]['dchg_maximum_voltage'],
                c=colors[k_count]
            )

            for cyc_i in [0, -1]:
                cyc = test_object[barcode_count][k][cyc_i]
                ax1.axvline(
                    x=cyc, color=colors[k_count], linestyle='--')

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
            pred_max_dchg_vol = test_results["pred_max_dchg_vol"]

            ax1.plot(cycles, pred_cap, c=colors[k_count])
            ax2.plot(cycles, pred_max_dchg_vol, c=colors[k_count])

        savefig('Cap_{}_Count_{}.png', fit_args, barcode, count)
        plt.close(fig)

def plot_shift(plot_params, init_returns):
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

            ax1.plot(cycles, test_results["shift"], c=colors[k_count])
            ax1.set_ylabel("shift")
            ax2.plot(cycles, test_results["pred_eq_vol"], c=colors[k_count])
            ax2.set_ylabel("eq_voltage_0")

        savefig('Shift_{}_Count_{}.png', fit_args, barcode, count)
        plt.close(fig)

def plot_eq_vol(plot_params, init_returns):
    barcodes = plot_params["barcodes"]
    count = plot_params["count"]
    fit_args = plot_params["fit_args"]

    degradation_model = init_returns["degradation_model"]
    test_object = init_returns["test_object"]
    cycles_m = init_returns["cycles_m"]
    cycles_v = init_returns["cycles_v"]
    vol_tensor = init_returns["vol_tensor"]

    for barcode_count, barcode in enumerate(barcodes):
        fig = plt.figure(figsize=[7., 5.])
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(2.)
            ax2.spines[axis].set_linewidth(2.)

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
            ax1.plot(cycles, [4.3 for _ in cycles], c='0.5')
            set_tick_params(ax1)

            ax2.plot(cycles, test_results["pred_r"], c=colors[k_count])
            ax2.plot(cycles, [0.05 for _ in cycles], c='0.5')
            set_tick_params(ax2)

        plt.tight_layout(pad=0.1)
        savefig('Eq_{}_Count_{}.png', fit_args, barcode, count)
        plt.close(fig)

def test_all_voltages(cycle, k, barcode_count, degradation_model, voltages):
    centers = tf.expand_dims(
        tf.concat(
            (tf.expand_dims(cycle, axis=0), k),
            axis=0
        ),
        axis=0
    )
    indecies = tf.reshape(barcode_count, [1])
    measured_cycles = tf.reshape(cycle, [1, 1])

    return degradation_model(
        (centers, indecies, measured_cycles, voltages),
        training=False
    )

def test_single_voltage(cycles, v, k, barcode_count, degradation_model):
    centers = tf.concat(
        (
            tf.expand_dims(cycles, axis=1),
            tf.tile(tf.expand_dims(k, axis=0), [len(cycles), 1])
        ),
        axis=1
    )
    indecies = tf.tile(tf.expand_dims(barcode_count, axis=0), [len(cycles)])
    measured_cycles = tf.expand_dims(cycles, axis=1)

    return degradation_model(
        (centers, indecies, measured_cycles, tf.expand_dims(v, axis=0)),
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
