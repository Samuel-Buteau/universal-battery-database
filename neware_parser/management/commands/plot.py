import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import tensorflow as tf

from .colour_print import Print

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import matplotlib.patches as mpatches
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
        rate = '0.05'
    elif rate > 1.75:
        rate = '{}'.format(int(round(rate)))
    elif rate > 0.4:
        rate = round(2. * rate_in) / 2.
        if rate == 1.:
            rate = '1'
        else:
            rate = '{:1.1f}'.format(rate)
    elif rate > 0.09:
        rate = '{:1.1f}'.format(rate)
    return rate


def bake_voltage(vol_in):
    vol = round(10. * vol_in) / 10.
    if vol == 1. or vol == 2. or vol == 3. or vol==4. or vol == 5.:
        vol = '{}'.format(int(vol))
    else:
        vol = '{:1.1f}'.format(vol)
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

    template = "I:{}:{} V:{}:{}"
    return template.format(end_rate_prev, constant_rate, end_voltage_prev, end_voltage)





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
        fig = plt.figure(figsize=[9,5])

        for typ, off in [('dchg',0), ('chg', 3)]:
            list_of_keys = [key for key in test_object[barcode_count].keys() if key[-1] == typ]
            list_of_keys.sort(key=lambda k: (round(20.*k[0]), round(20.*k[1]), round(20.*k[2]), round(20.*k[3]), round(20.*k[4])))
            print(list_of_keys)
            list_of_patches = []
            ax = fig.add_subplot(2, 3, 1+off)
            for k_count, k in enumerate(list_of_keys):

                if k[-1] == 'dchg':
                    sign_change = -1.
                else:
                    sign_change = +1.

                barcode_k = all_data[barcode][k][0]

                for vq_count, vq in enumerate(barcode_k['capacity_vector']):
                    cyc = all_data[barcode][k][0]['cycle_number'][vq_count]
                    mult = 1. - (.5 * cyc / 6000.)

                    vq_mask = barcode_k['vq_curve_mask'][vq_count]
                    valids = vq_mask> .5

                    ax.set_xlim(x_lim)
                    ax.set_ylim(y_lim)
                    ax.scatter(sign_change * vq[valids], vol_tensor.numpy()[valids],
                               c=(
                                   mult * COLORS[k_count][0],
                                   mult * COLORS[k_count][1],
                                   mult * COLORS[k_count][2]
                               ),
                               s=3)


            ax = fig.add_subplot(2, 3, 2+off)

            cycles = [0, 6000/2, 6000]
            for k_count, k in enumerate(list_of_keys):
                list_of_patches.append(mpatches.Patch(color=COLORS[k_count], label=make_legend(k)))

                if k[-1] == 'dchg':
                    sign_change = -1.
                else:
                    sign_change = +1.

                for i, cyc in enumerate(cycles):
                    cycle = ((float(cyc) - cycles_m) / tf.sqrt(cycles_v))
                    mult = 1.-(.5*cyc/6000.)

                    test_results = test_all_voltages(
                        cycle,
                        all_data[barcode][k][1]['avg_constant_current'],
                        all_data[barcode][k][1]['avg_end_current_prev'],
                        all_data[barcode][k][1]['avg_end_voltage_prev'],
                        barcode_count, degradation_model, vol_tensor
                    )

                    pred_cap = tf.reshape(test_results["pred_cap"], shape = [-1])


                    ax.set_xlim(x_lim)
                    ax.set_ylim(y_lim)
                    ax.scatter(
                        sign_change*pred_cap,
                        vol_tensor.numpy(),
                        c = (
                            mult * COLORS[k_count][0],
                            mult * COLORS[k_count][1],
                            mult * COLORS[k_count][2]
                        ),
                        s=3,

                    )

            ax = fig.add_subplot(2, 3, 3 + off)
            ax.axis('off')
            ax.legend(handles=list_of_patches)

        savefig('VQ_{}_Count_{}.png', fit_args, barcode, count)
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

    for barcode_count, barcode in enumerate(barcodes):
        fig = plt.figure(figsize=[5,5])

        for typ, off in [('dchg',0), ('chg', 2)]:
            list_of_patches = []
            list_of_keys = [key for key in test_object[barcode_count].keys() if key[-1] == typ]
            list_of_keys.sort(key=lambda k: (round(20.*k[0]), round(20.*k[1]), round(20.*k[2]), round(20.*k[3]), round(20.*k[4])))

            ax1 = fig.add_subplot(2, 2, 1 + off)
            ax1.set_ylabel("capacity")

            for k_count, k in enumerate(list_of_keys):
                list_of_patches.append(mpatches.Patch(color=COLORS[k_count], label=make_legend(k)))

                if k[-1] == 'dchg':
                    sign_change = -1.
                else:
                    sign_change = +1.

                ax1.scatter(
                    all_data[barcode][k][0]['cycle_number'],
                    sign_change*all_data[barcode][k][0]['last_cc_capacity'],
                    c=COLORS[k_count],
                    s=5,
                    label = make_legend(k)
                )


                for cyc_i in [0, -1]:
                    cyc = test_object[barcode_count][k][cyc_i]
                    ax1.axvline(
                        x=cyc, color=COLORS[k_count], linestyle='--')

            for k_count, k in enumerate(list_of_keys):

                if k[-1] == 'dchg':
                    sign_change = -1.
                else:
                    sign_change = +1.

                cycles = test_object[barcode_count][k]
                min_c = min(cycles)
                max_c = max(cycles)
                cycles = [
                     x for x in np.arange(0., 6000., 20.)
                ]

                my_cycles = [
                    (cyc - cycles_m) / tf.sqrt(cycles_v) for cyc in cycles
                ]

                test_results = test_single_voltage(
                    my_cycles, all_data[barcode][k][1]['avg_last_cc_voltage'],
                    all_data[barcode][k][1]['avg_constant_current'],
                    all_data[barcode][k][1]['avg_end_current_prev'],
                    all_data[barcode][k][1]['avg_end_voltage_prev'],
                    barcode_count, degradation_model
                )
                pred_cap = tf.reshape(test_results["pred_cap"], shape = [-1])

                ax1.plot(cycles, sign_change*pred_cap, c=COLORS[k_count])

            ax = fig.add_subplot(2, 2, 2 + off)
            ax.axis('off')
            ax.legend(handles=list_of_patches)

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
