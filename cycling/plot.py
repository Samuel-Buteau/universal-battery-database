import os

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from cycling.Key import Key

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

C_OVER_TWENTY_RULE = (0.038,0.062)
C_OVER_TWO_RULE = (0.38,0.62)
C_RULE = (0.85,1.3)
TWO_C_RULE = (1.6,2.4)
THREE_C_RULE = (2.5,3.5)


#TODO(sam): keep track of these in the database and allow users to modify.
Preferred_Legends = {
    (C_OVER_TWENTY_RULE, C_OVER_TWENTY_RULE, C_OVER_TWENTY_RULE, None, None):0,
    (C_OVER_TWENTY_RULE, C_OVER_TWO_RULE, C_OVER_TWO_RULE, None, None): 1,
    (C_OVER_TWENTY_RULE, C_RULE, C_RULE, None, None): 2,
    (C_OVER_TWENTY_RULE, TWO_C_RULE, TWO_C_RULE, None, None): 3,
    (C_OVER_TWENTY_RULE, THREE_C_RULE, THREE_C_RULE, None, None): 4,

    (C_OVER_TWENTY_RULE, C_OVER_TWENTY_RULE, C_OVER_TWENTY_RULE, None, None): 0,
    (C_OVER_TWENTY_RULE, C_RULE, C_OVER_TWENTY_RULE, None, None): 5,
    (C_OVER_TWO_RULE, C_RULE, None, None, None): 1,
    (C_RULE, C_RULE, C_OVER_TWENTY_RULE, None, None): 2,
    (TWO_C_RULE, C_RULE, C_OVER_TWENTY_RULE, None, None): 3,
    (THREE_C_RULE, C_RULE, C_OVER_TWENTY_RULE, None, None): 4,

}

def bake_rate(rate_in):
    rate = round(100. * rate_in) / 100.
    return rate

def bake_voltage(vol_in):
    vol = round(10. * vol_in) / 10.
    return vol


def make_legend_key(key):

    constant_rate = bake_rate(key[0])
    end_rate_prev= bake_rate(key[1])
    end_rate = bake_rate(key[2])

    end_voltage = bake_voltage(key[3])
    end_voltage_prev = bake_voltage(key[4])

    return (
        end_rate_prev, constant_rate, end_rate, end_voltage_prev, end_voltage
    )

def match_legend_key(legend_key, rule):
    match = True
    for i in range(len(legend_key)):
        if rule[i] is None:
            continue
        if rule[i][0] <= legend_key[i] <= rule[i][1]:
            continue
        else:
            match = False
            break
    return match

def make_legend(key):
    end_rate_prev, constant_rate, end_rate, end_voltage_prev, end_voltage = make_legend_key(key)
    template = "I {:3.2f}:{:3.2f}:{:3.2f} V {:2.1f}:{:2.1f}"
    return template.format(
        end_rate_prev, constant_rate, end_rate, end_voltage_prev, end_voltage
    )



def get_figsize(target):
    figsize = None
    if target == "generic_vs_capacity":
        figsize = [5, 10]
    elif target == "generic_vs_cycle":
        figsize = [11, 10]
    return figsize


#TODO(sam): make the interface more general
def plot_engine_direct(data_streams, target, todos, fit_args, filename):
    # open plot
    fig, axs = plt.subplots(nrows=len(todos), figsize=get_figsize(target), sharex=True)
    for i, todo in enumerate(todos):
        typ, mode = todo
        ax = axs[i]
        # options
        options = generate_options(mode, typ, target)
        list_of_target_data = []

        for source, data, _, max_cyc_n in data_streams:
            list_of_target_data.append(
                data_engine(
                    source,
                    target,
                    data,
                    typ,
                    mode,
                    max_cyc_n = max_cyc_n
                )
            )

        _, list_of_keys, _ = list_of_target_data[0]
        custom_colors = map_legend_to_color(list_of_keys)

        for j, target_data in enumerate(list_of_target_data):
            generic, _, generic_map = target_data
            # plot
            plot_generic(
                target,
                generic, list_of_keys, custom_colors,
                generic_map, ax,
                channel=data_streams[j][2],
                options=options,
            )

        produce_annotations(ax, get_list_of_patches(list_of_keys, custom_colors), options)

    # export
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    savefig(filename, fit_args)
    plt.close(fig)


def generate_options(mode, typ, target):
    # sign_change
    sign_change = get_sign_change(typ)

    if target == 'generic_vs_capacity':
        #label
        x_quantity = "Capacity"
        y_quantity = get_y_quantity(mode)
        #leg
        leg = {
            ("dchg", "cc"): (.5, 1.),
            ("chg", "cc"): (.5, .5),
            ("chg", "cv"): (0., .5),
        }

    elif target == "generic_vs_cycle":
        # label
        x_quantity = "Cycle"
        y_quantity = 'capacity'
        # leg
        leg = {
            ("dchg", "cc"): (.7, 1.),
            ("chg", "cc"):  (.7, 1.),
            ("chg", "cv"):  (.7, 1.),
        }

    ylabel = typ + "-" + mode + "\n" + y_quantity
    xlabel = x_quantity
    x_leg, y_leg = leg[(typ,mode)]
    return {
        "sign_change": sign_change,
        "x_leg": x_leg,
        "y_leg": y_leg,
        "xlabel": xlabel,
        "ylabel": ylabel
    }

def fetch_svit_keys_averages(compiled, barcode):
    svit_and_count = get_svit_and_count(compiled, barcode)
    keys = compiled[Key.ALL_DATA][barcode][Key.CYC_GRP_DICT].keys()
    averages = {}
    for k in keys:
        view = compiled[Key.ALL_DATA][barcode][Key.CYC_GRP_DICT][k]
        averages[k] = {}
        for t in [Key.I_CC_AVG, Key.I_PREV_END_AVG, Key.I_END_AVG, Key.V_PREV_END_AVG, Key.V_END_AVG,
                  Key.V_CC_LAST_AVG]:
            averages[k][t] = view[t]

    return svit_and_count, keys, averages



def get_sign_change(typ):
    if typ == "dchg":
        sign_change = -1.
    else:
        sign_change = 1.
    return sign_change

def get_y_quantity(mode):
    if mode == 'cc':
        y_quantity = 'voltage'
    elif mode == 'cv':
        y_quantity = 'current'
    return y_quantity

def get_generic_map(source, target, mode):
    quantity = get_y_quantity(mode)
    generic_map = {}
    if target == "generic_vs_cycle":
        generic_map = {
            'y': "last_{}_capacity".format(mode),
        }
    elif target == "generic_vs_capacity":
        generic_map = {
            'x': "{}_capacity_vector".format(mode),
            'y': "{}_{}_vector".format(mode, quantity),
        }
        if source == "compiled":
            generic_map['mask'] = "{}_mask_vector".format(mode)
    return generic_map

def data_engine(
        source,
        target,
        data,
        typ,
        mode,
        max_cyc_n
    ):
    sign_change=get_sign_change(typ)
    generic_map = get_generic_map(source, target, mode)
    if source == "model":
        degradation_model, barcode, cycle_m, cycle_v, svit_and_count, keys, averages = data
        list_of_keys = get_list_of_keys(keys, typ)
    elif source == "compiled":
        list_of_keys = get_list_of_keys(data.keys(), typ)
        needed_fields = [Key.N] + list(generic_map.values())

    generic = {}
    for k in list_of_keys:
        if source == 'compiled':
            actual_n = len(data[k][Key.MAIN])
            if actual_n > max_cyc_n:
                indecies = np.linspace(0, actual_n-1, max_cyc_n).astype(dtype=np.int32)
                generic[k] = data[k][Key.MAIN][needed_fields][indecies]
            else:
                generic[k] = data[k][Key.MAIN][needed_fields]
        elif source == 'model':
            generic[k] = compute_target(
                target,
                degradation_model,
                barcode,
                sign_change,
                mode,
                averages[k],
                generic_map,
                svit_and_count,
                cycle_m,
                cycle_v,
                max_cyc_n=max_cyc_n,
            )

    return generic, list_of_keys, generic_map



def map_legend_to_color(list_of_keys):
    #TODO: unnecessarily messy
    legends = Preferred_Legends
    custom_colors = {}
    colors_taken = []
    for k in list_of_keys:
        legend_key = make_legend_key(k)
        matched = False
        for legend_rule in legends.keys():
            if match_legend_key(legend_key, legend_rule):
                matched = True
                color_index = legends[legend_rule]
                if color_index in colors_taken:
                    possible_colors = [c_i for c_i in range(len(COLORS)) if c_i not in colors_taken]
                    if len(possible_colors) == 0:
                        color_index = 0
                    else:
                        color_index = sorted(possible_colors)[0]

                if not color_index in colors_taken:
                    colors_taken.append(color_index)
                custom_colors[k] = color_index
                break
        if not matched:
            continue

    for color_index in legends.values():
        if not color_index in colors_taken:
            colors_taken.append(color_index)

    for k in list_of_keys:
        if not k in custom_colors.keys():
            possible_colors = [c_i for c_i in range(len(COLORS)) if c_i not in colors_taken]
            if len(possible_colors) == 0:
                color_index = 0
            else:
                color_index = sorted(possible_colors)[0]

            if not color_index in colors_taken:
                colors_taken.append(color_index)
            custom_colors[k] = color_index

    for k in list_of_keys:
        custom_colors[k] = COLORS[custom_colors[k]]

    return custom_colors

def get_list_of_patches(list_of_keys, custom_colors):
    list_of_patches = []
    for k in list_of_keys:
        color = custom_colors[k]
        list_of_patches.append(mpatches.Patch(
            color=color, label=make_legend(k)
        ))
    return list_of_patches


def adjust_color(cyc, color, target_cycle=6000., target_ratio=.5):
    mult = 1. + (target_ratio-1.)*(float(cyc) / target_cycle)
    return (
            mult * color[0],
            mult * color[1],
            mult * color[2]
    )


def produce_annotations(ax, list_of_patches, options):
    ax.legend(
        handles = list_of_patches, fontsize = "small",
        bbox_to_anchor = (options["x_leg"], options["y_leg"]), loc = "upper left"
    )
    ax.set_ylabel(options["ylabel"])
    ax.set_xlabel(options["xlabel"])


def simple_plot(ax, x,y, color, channel):
    if channel == 'scatter':
        ax.scatter(
            x,
            y,
            c=[list(color)],
            s=3
        )
    elif channel == 'plot':
        ax.plot(
            x,
            y,
            c=color,
        )
    else:
        raise Exception("not yet implemented. channel = {}".format(channel))

def plot_generic(
        target,
        groups,list_of_keys,
        custom_colors, generic_map,
        ax,
        channel,
        options
    ):

    for k in list_of_keys:
        group = groups[k]
        if target == "generic_vs_cycle":
            x = group[Key.N]
            y = options["sign_change"] * group[generic_map['y']]
            color = custom_colors[k]
            simple_plot(ax, x, y, color, channel)
        elif target == "generic_vs_capacity":
            for i in range(len(group)):
                x_ = options["sign_change"] * group[generic_map['x']][i]
                y_ = group[generic_map['y']][i]
                if 'mask' in generic_map.keys():
                    valids = group[generic_map['mask']][i] > .5
                    x = x_[valids]
                    y = y_[valids]
                else:
                    x = x_
                    y = y_
                color =  adjust_color(group[Key.N][i], custom_colors[k])
                simple_plot(ax, x, y, color, channel)


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



def compute_target(target, degradation_model, barcode, sign_change, mode, averages, generic_map, svit_and_count,
                   cycle_m, cycle_v, cycle_min = 0, cycle_max = 6000 , max_cyc_n = 3):
    cycle = np.linspace(cycle_min, cycle_max, max_cyc_n)
    scaled_cyc = (cycle - cycle_m) / tf.sqrt(cycle_v)

    if target == 'generic_vs_capacity':
        v_range = np.ones((1), dtype=np.float32)
        current_range = np.ones((1), dtype=np.float32)
        if mode == 'cc':
            v_min = min(averages[Key.V_PREV_END_AVG], averages[Key.V_END_AVG])
            v_max = max(averages[Key.V_PREV_END_AVG], averages[Key.V_END_AVG])
            v_range = np.linspace(v_min, v_max, 32)
            y_n = 32
        elif mode == 'cv':
            curr_max = abs(averages[Key.I_CC_AVG])
            curr_min = abs(averages[Key.I_END_AVG])

            if curr_min == curr_max:
                current_range = np.array([curr_min])
                y_n = 1
            else:
                current_range = sign_change * np.exp(
                    np.linspace(
                        np.log(curr_min),
                        np.log(curr_max),
                        32
                    )
                )
                y_n = 32


        test_results = degradation_model.test_all_voltages(
            tf.constant(scaled_cyc, dtype=tf.float32),
            tf.constant(averages[Key.I_CC_AVG], dtype=tf.float32),
            tf.constant(averages[Key.I_PREV_END_AVG], dtype=tf.float32),
            tf.constant(averages[Key.V_PREV_END_AVG], dtype=tf.float32),
            tf.constant(averages[Key.V_END_AVG], dtype=tf.float32),
            tf.constant(degradation_model.cell_direct.id_dict[barcode], dtype=tf.int32),
            tf.constant(v_range, dtype=tf.float32),
            tf.constant(current_range, dtype=tf.float32),
            tf.constant(svit_and_count[Key.SVIT_GRID], dtype=tf.float32),
            tf.constant(svit_and_count[Key.COUNT_MATRIX], dtype=tf.float32),
        )

        if mode == "cc":
            yrange = v_range
            pred_capacity_label = "pred_cc_capacity"
        elif mode == "cv":
            yrange = current_range
            pred_capacity_label = "pred_cv_capacity"

        cap = tf.reshape(
            test_results[pred_capacity_label], shape=[max_cyc_n, -1]
        )

        if y_n == 1:
            y_n = (1,)

        generic = np.array(
            [(cyc, cap[i,:], yrange) for i, cyc in enumerate(cycle)],
            dtype= [
                (Key.N, 'f4'),
                (generic_map['x'], 'f4', y_n),
                (generic_map['y'], 'f4', y_n),
            ]
        )
    elif target == "generic_vs_cycle":
        if mode == "cc":
            target_voltage = averages["avg_last_cc_voltage"]
            target_currents = [averages[Key.I_CC_AVG]]
        elif mode == "cv":
            target_voltage = averages[Key.V_END_AVG]
            target_currents = [averages[Key.I_END_AVG]]

        test_results = degradation_model.test_single_voltage(
            tf.cast(scaled_cyc, dtype=tf.float32),
            tf.constant(target_voltage, dtype=tf.float32),
            tf.constant(averages[Key.I_CC_AVG], dtype=tf.float32),
            tf.constant(averages[Key.I_PREV_END_AVG], dtype=tf.float32),
            tf.constant(averages[Key.V_PREV_END_AVG], dtype=tf.float32),
            tf.constant(averages[Key.V_END_AVG], dtype=tf.float32),
            tf.constant(target_currents, dtype=tf.float32),
            tf.constant(degradation_model.cell_direct.id_dict[barcode], dtype=tf.int32),
            tf.constant(svit_and_count[Key.SVIT_GRID], dtype=tf.float32),
            tf.constant(svit_and_count[Key.COUNT_MATRIX], dtype=tf.float32)
        )
        if mode == "cc":
            pred_cap = tf.reshape(
                test_results["pred_cc_capacity"],
                shape=[-1]
            ).numpy()
        elif mode == "cv":
            pred_cap = test_results["pred_cv_capacity"].numpy()[:, -1]

        generic = np.array(
            list(zip(cycle, pred_cap)),
            dtype= [
                (Key.N, 'f4'),
                (generic_map['y'], 'f4'),
            ]

        )

    return generic






def plot_direct(target, plot_params, init_returns):
    if target == "generic_vs_capacity":
        compiled_max_cyc_n = 8
        model_max_cyc_n = 3
        header = "VQ"
    elif target == "generic_vs_cycle":
        compiled_max_cyc_n = 2000
        model_max_cyc_n = 200
        header = "Cap"

    barcodes = plot_params["barcodes"][:plot_params[Key.FIT_ARGS]["barcode_show"]]
    count = plot_params["count"]
    fit_args = plot_params[Key.FIT_ARGS]

    degradation_model = init_returns[Key.MODEL]
    my_data = init_returns[Key.MY_DATA]
    cycle_m = init_returns[Key.CYC_M]
    cycle_v = init_returns[Key.CYC_V]

    for barcode_count, barcode in enumerate(barcodes):
        compiled_groups = my_data[Key.ALL_DATA][barcode][Key.CYC_GRP_DICT]
        svit_and_count, keys, averages = fetch_svit_keys_averages(my_data, barcode)
        model_data = degradation_model, barcode, cycle_m, cycle_v, svit_and_count, keys, averages

        plot_engine_direct(
            data_streams=[
                ('compiled', compiled_groups, 'scatter', compiled_max_cyc_n),
                ('model', model_data, 'plot', model_max_cyc_n)
            ],
            target=target,
            todos=[
                ("dchg", "cc"),
                ("chg", "cc"),
                ("chg", "cv"),
            ],
            fit_args=fit_args,
            filename=header+"_{}_Count_{}.png".format(barcode, count)
        )




def savefig(figname, fit_args):
    plt.savefig(
        os.path.join(fit_args[Key.PATH_PLOTS], figname), dpi = 300
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




def get_list_of_keys(keys, typ):
    list_of_keys = [
        key for key in keys if key[-1] == typ
    ]
    list_of_keys.sort(
        key = lambda k: (
            round(40. * k[0]), round(40. * k[1]), round(40. * k[2]),
            round(10. * k[3]), round(10. * k[4])
        )
    )
    return list_of_keys
