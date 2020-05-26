import os
import sys



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from Key import Key

import base64
from io import BytesIO

matplotlib_axes_logger.setLevel("ERROR")
from plot_constants import *



def get_byte_image(fig, dpi):
    buf = BytesIO()
    plt.savefig(buf, format = "png", dpi = dpi)
    image_base64 = base64.b64encode(
        buf.getvalue()
    ).decode("utf-8").replace("\n", "")
    buf.close()
    plt.close(fig)
    return image_base64

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
    known_data_engines = None,
    make_file_legends_and_vertical = None,
    send_to_file = False,
    dpi = 300,
):
    """ TODO(harvey)
    Args: TODO(harvey)
        target: Plot type - "generic_vs_capacity" or "generic_vs_cycle".
    Returns: TODO(harvey)
    """
    if known_data_engines is None:
        known_data_engines = {}
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

        cell_id = None
        for source, data, _, max_cyc_n in data_streams:
            generic_map = get_generic_map(source, target, mode)
            if source not in known_data_engines.keys():
                continue
            generic, list_of_keys = known_data_engines[source](
                target, data, typ, generic_map, mode,
                max_cyc_n=max_cyc_n, lower_cycle=lower_cycle,
                upper_cycle=upper_cycle,
            )
            list_of_target_data.append(
                (generic, list_of_keys, generic_map)
            )

            if source == "database":
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

        if make_file_legends_and_vertical is not None:
            make_file_legends_and_vertical(
                ax, cell_id, lower_cycle, upper_cycle, show_invalid,
                vertical_barriers, list_all_options, leg,
            )

    # export
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0)


    if send_to_file:
        savefig(filename, fit_args)
        plt.close(fig)
    else:
        return get_byte_image(fig, dpi)



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
