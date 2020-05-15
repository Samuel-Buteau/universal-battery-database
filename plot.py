import os
import sys

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from Key import Key
from cycling.models import (
    compute_from_database, make_file_legends_and_vertical, get_byte_image,
)
from machine_learning.DegradationModelBlackbox import DegradationModel

matplotlib_axes_logger.setLevel("ERROR")
from plot_constants import *

FIGSIZE = [6, 5]


def bake_rate(rate_in):
    """ TODO(harvey) """
    rate = round(100. * rate_in) / 100.
    return rate


def bake_voltage(vol_in):
    """ TODO(harvey) """
    vol = round(10. * vol_in) / 10.
    return vol


def make_legend_key(key):
    """ TODO(harvey) """
    constant_rate = bake_rate(key[0])
    end_rate_prev = bake_rate(key[1])
    end_rate = bake_rate(key[2])

    end_voltage = bake_voltage(key[3])
    end_voltage_prev = bake_voltage(key[4])

    return (
        end_rate_prev, constant_rate, end_rate, end_voltage_prev, end_voltage
    )


def match_legend_key(legend_key, rule):
    """ TODO(harvey) """
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
    """ TODO(harvey) """
    (
        end_rate_prev, constant_rate, end_rate, end_voltage_prev, end_voltage,
    ) = make_legend_key(key)
    template = "I {:3.2f}:{:3.2f}:{:3.2f} V {:2.1f}:{:2.1f}"
    return template.format(
        end_rate_prev, constant_rate, end_rate, end_voltage_prev, end_voltage,
    )


def get_figsize(target):
    """ TODO(harvey) """
    figsize = None
    if target == "generic_vs_capacity":
        figsize = [5, 10]
    elif target == "generic_vs_cycle":
        figsize = [11, 10]
    return figsize


# TODO(sam): make the interface more general
def plot_engine_direct(
    data_streams, target: str, todos, fit_args, filename,
    lower_cycle = None, upper_cycle = None, vertical_barriers = None,
    list_all_options = None, show_invalid = False, figsize = None,
):
    """ TODO(harvey)
    Args: TODO(harvey)
        target: Plot type - "generic_vs_capacity" or "generic_vs_cycle".
    Returns: TODO(harvey)
    """
    # open plot
    if figsize is None:
        figsize = get_figsize(target)

    fig, axs = plt.subplots(
        nrows = len(todos), figsize = figsize, sharex = True,
    )
    for i, todo in enumerate(todos):
        typ, mode = todo
        if len(todos) == 1:
            ax = axs
        else:
            ax = axs[i]
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(3.)

        plot_options = generate_plot_options(mode, typ, target)
        list_of_target_data = []

        source_database = False
        cell_id = None
        for source, data, _, max_cyc_n in data_streams:
            list_of_target_data.append(
                data_engine(
                    source, target, data, typ, mode,
                    max_cyc_n = max_cyc_n, lower_cycle = lower_cycle,
                    upper_cycle = upper_cycle,
                )
            )
            if source == "database":
                source_database = True
                cell_id, valid = data

        list_of_keys = []
        for _, lok, _ in list_of_target_data:
            list_of_keys += lok
        list_of_keys = get_list_of_keys(list(set(list_of_keys)), typ)

        custom_colors = map_legend_to_color(list_of_keys)

        for j, target_data in enumerate(list_of_target_data):
            generic, _, generic_map = target_data

            plot_generic(
                target, generic, list_of_keys, custom_colors, generic_map, ax,
                channel = data_streams[j][2], plot_options = plot_options,
            )

        leg = produce_annotations(
            ax, get_list_of_patches(list_of_keys, custom_colors), plot_options
        )
        if source_database:
            make_file_legends_and_vertical(
                ax, cell_id, lower_cycle, upper_cycle, show_invalid,
                vertical_barriers, list_all_options, leg,
            )

    # export
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0)
    if source_database:
        # TODO(sam):
        send_to_file = False
        if vertical_barriers is None:
            quick = True
        else:
            quick = False

        if send_to_file:
            savefig(filename, fit_args)
            plt.close(fig)
        else:
            if quick:
                dpi = 50
            else:
                dpi = 300
            return get_byte_image(fig, dpi)

    else:
        savefig(filename, fit_args)
    plt.close(fig)


def generate_plot_options(mode: str, typ: str, target: str) -> dict:
    """ TODO(harvey)
    Args:
        mode: Specifies the mode of charge/discharge - constant-current ("cc")
            or constant-voltage ("cv").
    Returns: TODO(harvey)
    """
    # sign_change
    sign_change = get_sign_change(typ)

    if target == 'generic_vs_capacity':
        # label
        x_quantity = "Capacity"
        y_quantity = get_y_quantity(mode)
        # leg
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
            ("chg", "cc"): (.7, 1.),
            ("chg", "cv"): (.7, 1.),
        }

    else:
        sys.exit("Unknown `target` in `generate_options`!")

    y_label = typ + "-" + mode + "\n" + y_quantity
    x_label = x_quantity
    x_leg, y_leg = leg[(typ, mode)]
    return {
        "sign_change": sign_change,
        "x_leg": x_leg,
        "y_leg": y_leg,
        "xlabel": x_label,
        "ylabel": y_label,
    }


def fetch_svit_keys_averages(compiled, cell_id):
    svit_and_count = get_svit_and_count(compiled, cell_id)
    keys = compiled[Key.ALL_DATA][cell_id][Key.CYC_GRP_DICT].keys()
    averages = {}
    for k in keys:
        view = compiled[Key.ALL_DATA][cell_id][Key.CYC_GRP_DICT][k]
        averages[k] = {}
        for t in [
            Key.I_CC_AVG, Key.I_PREV_END_AVG, Key.I_END_AVG,
            Key.V_PREV_END_AVG, Key.V_END_AVG, Key.V_CC_LAST_AVG
        ]:
            averages[k][t] = view[t]

    return svit_and_count, keys, averages


def get_sign_change(typ: str) -> float:
    """ Get the sign change based on charge or discharge.
    Args:
        typ: Specifies charge ("chg") or discharge ("dchg").
    Returns:
        1 if type is charge, -1 if type is discharge.
    """
    if typ == "dchg":
        sign_change = -1.
    else:
        sign_change = 1.
    return sign_change


def get_y_quantity(mode: str) -> str:
    """ Get the dependent variable based on charge/discharge mode.
    Args:
        mode: Specifies the mode of charge/discharge - constant-current ("cc")
            or constant-voltage ("cv").
    Returns:
        "voltage" if mode is "cc", "current" if mode is "cv".
    """
    if mode == 'cc':
        y_quantity = 'voltage'
    elif mode == 'cv':
        y_quantity = 'current'
    else:
        sys.exit("Unknown `mode` in `get_y_quantity`!")
    return y_quantity


def get_generic_map(source, target: str, mode: str) -> dict:
    """ TODO(harvey)
    Args: TODO(harvey)
        target: Plot type - "generic_vs_capacity" or "generic_vs_cycle".
        mode: Specifies the mode of charge/discharge - constant-current ("cc")
            or constant-voltage ("cv").
    Returns: TODO(harvey)
    """
    quantity = get_y_quantity(mode)
    if target == "generic_vs_cycle":
        generic_map = {'y': "last_{}_capacity".format(mode)}
    elif target == "generic_vs_capacity":
        generic_map = {
            'x': "{}_capacity_vector".format(mode),
            'y': "{}_{}_vector".format(mode, quantity),
        }
        if source == "compiled":
            generic_map['mask'] = "{}_mask_vector".format(mode)
    else:
        sys.exit("Unknown `target` in `get_generic_map`!")
    return generic_map


def data_engine(
    source: str, target: str, data, typ, mode, max_cyc_n,
    lower_cycle = None, upper_cycle = None,
):
    """ TODO(harvey)
    Args: TODO(harvey)
        source: Specifies the source of the data to be plot: "model",
            "database", or "compiled".
        target: Plot type - "generic_vs_capacity" or "generic_vs_cycle".
    Returns: TODO(harvey)
    """
    generic = {}
    generic_map = get_generic_map(source, target, mode)
    if source == "model":
        (
            degradation_model, cell_id, cycle_m, cycle_v,
            svit_and_count, keys, averages,
        ) = data
        list_of_keys = get_list_of_keys(keys, typ)
        for k in list_of_keys:
            generic[k] = compute_target(
                target, degradation_model, cell_id, get_sign_change(typ),
                mode, averages[k], generic_map, svit_and_count,
                cycle_m, cycle_v, max_cyc_n = max_cyc_n,
            )
    elif source == "database":
        if typ != "dchg" or mode != "cc":
            return None, None, None
        cell_id, valid = data
        generic = compute_from_database(
            cell_id, lower_cycle, upper_cycle, valid,
        )
        list_of_keys = get_list_of_keys(generic.keys(), typ)

    elif source == "compiled":
        list_of_keys = get_list_of_keys(data.keys(), typ)
        needed_fields = [Key.N] + list(generic_map.values())
        for k in list_of_keys:
            actual_n = len(data[k][Key.MAIN])
            if actual_n > max_cyc_n:
                indices = np.linspace(0, actual_n - 1, max_cyc_n).astype(
                    dtype = np.int32
                )
                generic[k] = data[k][Key.MAIN][needed_fields][indices]
            else:
                generic[k] = data[k][Key.MAIN][needed_fields]
    else:
        sys.exit("Unknown `source` in `data_engine`!")

    return generic, list_of_keys, generic_map


def map_legend_to_color(list_of_keys):
    # TODO: unnecessarily messy
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
                    possible_colors = [
                        c_i for c_i in range(len(COLORS))
                        if c_i not in colors_taken
                    ]
                    if len(possible_colors) == 0:
                        color_index = 0
                    else:
                        color_index = sorted(possible_colors)[0]

                if color_index not in colors_taken:
                    colors_taken.append(color_index)
                custom_colors[k] = color_index
                break
        if not matched:
            continue

    for color_index in legends.values():
        if color_index not in colors_taken:
            colors_taken.append(color_index)

    for k in list_of_keys:
        if k not in custom_colors.keys():
            possible_colors = [
                c_i for c_i in range(len(COLORS)) if c_i not in colors_taken
            ]
            if len(possible_colors) == 0:
                color_index = 0
            else:
                color_index = sorted(possible_colors)[0]

            if color_index not in colors_taken:
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
            color = color, label = make_legend(k),
        ))
    return list_of_patches


def adjust_color(cyc, color, target_cycle = 6000., target_ratio = .5):
    mult = 1. + (target_ratio - 1.) * (float(cyc) / target_cycle)
    return (
        mult * color[0],
        mult * color[1],
        mult * color[2],
    )


def produce_annotations(ax, list_of_patches, plot_options):
    leg = ax.legend(
        handles = list_of_patches, fontsize = "small",
        bbox_to_anchor = (plot_options["x_leg"], plot_options["y_leg"]),
        loc = "upper left",
    )
    ax.set_ylabel(plot_options["ylabel"])
    ax.set_xlabel(plot_options["xlabel"])
    return leg


def simple_plot(ax, x, y, color, channel):
    if (
        channel == 'scatter'
        or channel == "scatter_valid"
        or channel == "scatter_invalid"
    ):
        if channel == 'scatter':
            s = 20
            marker = '.'
        elif channel == "scatter_valid":
            s = 100
            marker = '.'
        else:
            s = 5
            marker = 'x'

        ax.scatter(x, y, c = [list(color)], s = s, marker = marker)
    elif channel == 'plot':
        ax.plot(x, y, c = color, )
    else:
        raise Exception("not yet implemented. channel = {}".format(channel))


def plot_generic(
    target, groups, list_of_keys,
    custom_colors, generic_map, ax, channel, plot_options
):
    for k in list_of_keys:
        if k not in groups.keys():
            continue
        group = groups[k]
        if target == "generic_vs_cycle":
            x = group[Key.N]
            y = plot_options["sign_change"] * group[generic_map['y']]
            color = custom_colors[k]
            simple_plot(ax, x, y, color, channel)
        elif target == "generic_vs_capacity":
            for i in range(len(group)):
                x_ = plot_options["sign_change"] * group[generic_map['x']][i]
                y_ = group[generic_map['y']][i]
                if 'mask' in generic_map.keys():
                    valids = group[generic_map['mask']][i] > .5
                    x = x_[valids]
                    y = y_[valids]
                else:
                    x = x_
                    y = y_
                color = adjust_color(group[Key.N][i], custom_colors[k])
                simple_plot(ax, x, y, color, channel)


def get_svit_and_count(my_data, cell_id):
    n_sign = len(my_data["sign_grid"])
    n_voltage = len(my_data["voltage_grid"])
    n_current = len(my_data["current_grid"])
    n_temperature = len(my_data["temperature_grid"])

    count_matrix = np.reshape(
        my_data[Key.ALL_DATA][cell_id]["all_reference_mats"]
        [Key.COUNT_MATRIX][-1],
        [n_sign, n_voltage, n_current, n_temperature, 1],
    )

    svit_grid = np.concatenate(
        (
            np.tile(
                np.reshape(my_data["sign_grid"], [n_sign, 1, 1, 1, 1]),
                [1, n_voltage, n_current, n_temperature, 1],
            ),
            np.tile(
                np.reshape(my_data["voltage_grid"], [1, n_voltage, 1, 1, 1]),
                [n_sign, 1, n_current, n_temperature, 1],
            ),
            np.tile(
                np.reshape(my_data["current_grid"], [1, 1, n_current, 1, 1]),
                [n_sign, n_voltage, 1, n_temperature, 1],
            ),
            np.tile(
                np.reshape(
                    my_data["temperature_grid"], [1, 1, 1, n_temperature, 1],
                ),
                [n_sign, n_voltage, n_current, 1, 1]
            ),
        ),
        axis = -1,
    )
    return {Key.SVIT_GRID: svit_grid, Key.COUNT_MATRIX: count_matrix}


def compute_target(
    target: str, degradation_model: DegradationModel, cell_id,
    sign_change: float, mode: str, averages, generic_map, svit_and_count,
    cycle_m, cycle_v, cycle_min = 0, cycle_max = 6000, max_cyc_n = 3
):
    """
    Args: TODO(harvey)
        target: Plot type - "generic_vs_capacity" or "generic_vs_cycle".
        degradation_model: Machine learning model.
        sign_change: 1 if charge, -1 if discharge.
        mode: Charge/discharge mode - constant-current ("cc") or
            constant-voltage ("cv").
    """
    cycle = np.linspace(cycle_min, cycle_max, max_cyc_n)
    scaled_cyc = (cycle - cycle_m) / tf.sqrt(cycle_v)

    if target == 'generic_vs_capacity':
        v_range = np.ones(1, dtype = np.float32)
        current_range = np.ones(1, dtype = np.float32)
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
                    np.linspace(np.log(curr_min), np.log(curr_max), 32)
                )
                y_n = 32
        else:
            sys.exit("Unknown `mode` in `compute_target`!")

        test_results = degradation_model.test_all_voltages(
            tf.constant(scaled_cyc, dtype = tf.float32),
            tf.constant(averages[Key.I_CC_AVG], dtype = tf.float32),
            tf.constant(averages[Key.I_PREV_END_AVG], dtype = tf.float32),
            tf.constant(averages[Key.V_PREV_END_AVG], dtype = tf.float32),
            tf.constant(averages[Key.V_END_AVG], dtype = tf.float32),
            tf.constant(
                degradation_model.cell_direct.id_dict[cell_id],
                dtype = tf.int32,
            ),
            tf.constant(v_range, dtype = tf.float32),
            tf.constant(current_range, dtype = tf.float32),
            tf.constant(svit_and_count[Key.SVIT_GRID], dtype = tf.float32),
            tf.constant(svit_and_count[Key.COUNT_MATRIX], dtype = tf.float32),
        )

        if mode == "cc":
            yrange = v_range
            pred_capacity_label = Key.Pred.I_CC
        elif mode == "cv":
            yrange = current_range
            pred_capacity_label = Key.Pred.I_CV
        else:
            sys.exit("Unknown `mode` in `compute_target`!")

        cap = tf.reshape(
            test_results[pred_capacity_label], shape = [max_cyc_n, -1],
        )

        if y_n == 1:
            y_n = (1,)

        generic = np.array(
            [(cyc, cap[i, :], yrange) for i, cyc in enumerate(cycle)],
            dtype = [
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
        else:
            sys.exit("Unknown `mode` in `compute_target`!")

        test_results = degradation_model.test_single_voltage(
            tf.cast(scaled_cyc, dtype = tf.float32),
            tf.constant(target_voltage, dtype = tf.float32),
            tf.constant(averages[Key.I_CC_AVG], dtype = tf.float32),
            tf.constant(averages[Key.I_PREV_END_AVG], dtype = tf.float32),
            tf.constant(averages[Key.V_PREV_END_AVG], dtype = tf.float32),
            tf.constant(averages[Key.V_END_AVG], dtype = tf.float32),
            tf.constant(target_currents, dtype = tf.float32),
            tf.constant(
                degradation_model.cell_direct.id_dict[cell_id],
                dtype = tf.int32,
            ),
            tf.constant(svit_and_count[Key.SVIT_GRID], dtype = tf.float32),
            tf.constant(svit_and_count[Key.COUNT_MATRIX], dtype = tf.float32),
        )
        if mode == "cc":
            pred_cap = tf.reshape(
                test_results[Key.Pred.I_CC], shape = [-1],
            ).numpy()
        elif mode == "cv":
            pred_cap = test_results[Key.Pred.I_CV].numpy()[:, -1]
        else:
            sys.exit("Unknown `mode` in `compute_target`!")

        generic = np.array(
            list(zip(cycle, pred_cap)),
            dtype = [
                (Key.N, 'f4'),
                (generic_map['y'], 'f4'),
            ],
        )
    else:
        sys.exit("Unknown `target` in `compute_target`!")

    return generic


def plot_cycling_direct(
    cell_id, path_to_plots = None, lower_cycle = None, upper_cycle = None,
    show_invalid = False, vertical_barriers = None, list_all_options = None,
    figsize = None
):
    if show_invalid:
        data_streams = [
            ('database', (cell_id, True), 'scatter_valid', 100),
            ('database', (cell_id, False), 'scatter_invalid', 100),
        ]
    else:
        data_streams = [
            ('database', (cell_id, True), 'scatter_valid', 100),
        ]

    if path_to_plots is None:
        return plot_engine_direct(
            data_streams = data_streams,
            target = "generic_vs_cycle",
            todos = [("dchg", "cc")],
            fit_args = {'path_to_plots': path_to_plots},
            filename = "Initial_{}.png".format(cell_id),
            lower_cycle = lower_cycle,
            upper_cycle = upper_cycle,
            vertical_barriers = vertical_barriers,
            list_all_options = list_all_options,
            show_invalid = show_invalid,
            figsize = figsize,
        )
    else:
        plot_engine_direct(
            data_streams = data_streams,
            target = "generic_vs_cycle",
            todos = [("dchg", "cc")],
            fit_args = {'path_to_plots': path_to_plots},
            filename = "Initial_{}.png".format(cell_id),
            lower_cycle = lower_cycle,
            upper_cycle = upper_cycle,
            vertical_barriers = vertical_barriers,
            list_all_options = list_all_options,
            show_invalid = show_invalid,
            figsize = figsize,
        )


def plot_direct(target: str, plot_params: dict, init_returns: dict) -> None:
    """
    Args:
        target: Plot type - "generic_vs_capacity" or "generic_vs_cycle".
        plot_params: Parameters for plotting.
        init_returns: Return value of `ml_smoothing.initial_processing`.
    """
    if target == "generic_vs_capacity":
        compiled_max_cyc_n = 8
        model_max_cyc_n = 3
        header = "VQ"
    elif target == "generic_vs_cycle":
        compiled_max_cyc_n = 2000
        model_max_cyc_n = 200
        header = "Cap"
    else:
        sys.exit("Unknown `target` in `plot_direct`!")

    cell_ids\
        = plot_params["cell_ids"][:plot_params[Key.OPTIONS][Key.CELL_ID_SHOW]]
    count = plot_params["count"]
    fit_args = plot_params[Key.OPTIONS]

    degradation_model = init_returns[Key.MODEL]
    my_data = init_returns[Key.DATASET]
    cycle_m = init_returns[Key.CYC_M]
    cycle_v = init_returns[Key.CYC_V]

    for cell_id_count, cell_id in enumerate(cell_ids):
        compiled_groups = my_data[Key.ALL_DATA][cell_id][Key.CYC_GRP_DICT]
        svit_and_count, keys, averages = fetch_svit_keys_averages(
            my_data, cell_id,
        )
        model_data = (
            degradation_model, cell_id, cycle_m, cycle_v,
            svit_and_count, keys, averages,
        )

        plot_engine_direct(
            data_streams = [
                ('compiled', compiled_groups, 'scatter', compiled_max_cyc_n),
                ('model', model_data, 'plot', model_max_cyc_n),
            ],
            target = target,
            todos = [("dchg", "cc"), ("chg", "cc"), ("chg", "cv")],
            fit_args = fit_args,
            filename = header + "_{}_Count_{}.png".format(cell_id, count)
        )


def savefig(figname, options: dict):
    plt.savefig(os.path.join(options[Key.PATH_PLOTS], figname), dpi = 300)


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
    list_of_keys = [key for key in keys if key[-1] == typ]
    list_of_keys.sort(
        key = lambda k: (
            round(40. * k[0]), round(40. * k[1]), round(40. * k[2]),
            round(10. * k[3]), round(10. * k[4]),
        )
    )
    return list_of_keys
