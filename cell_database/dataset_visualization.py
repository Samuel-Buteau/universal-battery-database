import os
from cell_database.models import *
from cycling.models import *
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from plot import *
from cycling.plot import *

def escape_string_to_path(my_string):
    for bad, good in [
        ('\"', '_QUOTE_'),
        ('!', '_EXCLAMATION_'),
        ('#', '_DASH_'),
        ('$', '_DOLLAR_'),
        ('&', '_AMPERSAND_'),
        ("\'", '_HALFQUOTE_'),
        ("(", "_OPENP_"),
        (")", "_CLOSEP_"),
        ("*", "_STAR_"),
        (',', '_COMMA_'),
        (';', '_SEMICOLON_'),
        ('<', '_LESSTHAN_'),
        ('?', '_QUESTION_'),
        ('[', '_OPENB_'),
        (']', '_CLOSEB_'),
        ('=', '_EQUAL_'),
        ('\\', '_SLASH_'),
        ('^', '_HAT_'),
        ('`', '_BACKQUOTE_'),
        ("\{", '_OPENC_'),
        ("\}", '_CLOSEC_'),
        ('|', '_PIPE_'),
        ('~', '_TILDE_'),
        ('.', "_DOT_"),
        (':', "_COLON_"),
        ('/', '_OVER_'),
        # (' ', "_"),

    ]:
        my_string = my_string.replace(bad, good)

    return my_string





#TODO(sam): find better names for the functions that extract data in a plottable format
def compute_dataset(dataset, field_request):
    results = {}
    for wet_cell in dataset.wet_cells.order_by('cell_id'):
        rules = {}
        for filt in DatasetSpecificFilters.objects.filter(dataset=dataset, wet_cell=wet_cell):
            rules[filt.id] = filt.get_rule()
        results[wet_cell.cell_id] = compute_from_database2(wet_cell.cell_id, rules, field_request)
    return results


def hex_color_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16)/256 for i in (0, 2, 4))


def get_dataset_labels(dataset):
    filt_names = {}
    filt_colors = {}
    filt_pos = {}
    dataset_name = dataset.name
    wet_names = {}
    for wet_cell in dataset.wet_cells.order_by('cell_id'):
        wet_name, specified = wet_cell.get_specific_name_details(dataset)
        if not specified:
            wet_name = str(wet_cell.cell_id)
        wet_names[wet_cell.cell_id] = wet_name
        names = {}
        colors = {}
        pos = {}
        for filt in DatasetSpecificFilters.objects.filter(dataset=dataset, wet_cell=wet_cell):
            names[filt.id] = filt.name
            colors[filt.id] = hex_color_to_rgb(filt.color)
            pos[filt.id] = (filt.grid_position_x, filt.grid_position_y)
        filt_names[wet_cell.cell_id] = names
        filt_colors[wet_cell.cell_id] = colors
        filt_pos[wet_cell.cell_id] = pos

    return dataset_name, wet_names, filt_names, filt_colors, filt_pos




def output_dataset_to_csv(data, dataset_name, wet_names, filt_names, csv_format, output_dir):
    for cell_id in data.keys():
        for filt_id in data[cell_id].keys():
            path = os.path.join(
                output_dir,
                escape_string_to_path(dataset_name),
                escape_string_to_path(wet_names[cell_id])
            )

            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, '{}.csv'.format(escape_string_to_path(filt_names[cell_id][filt_id]))), 'w', newline='') as csv_f:
                writer = csv.writer(csv_f)
                header = [h for _, _, h in csv_format]
                writer.writerow(header)
                content = [[f(res[key]) for key, f, _ in csv_format] for res in data[cell_id][filt_id]]
                writer.writerows(content)


def output_dataset_to_plot(data, dataset_name, wet_names, filt_names, filt_colors, filt_pos, output_dir=None, dpi=300):
    positioned_filters = {}
    for cell_id in filt_pos.keys():
        for filt_id in filt_pos[cell_id].keys():
            if filt_pos[cell_id][filt_id] not in positioned_filters.keys():
                positioned_filters[filt_pos[cell_id][filt_id]] = [(cell_id, filt_id)]
            else:
                positioned_filters[filt_pos[cell_id][filt_id]].append((cell_id, filt_id))
    if len(positioned_filters.keys()) == 0:
        return None

    max_pos_x = max([pos_x for pos_x,pos_y in positioned_filters.keys()])
    max_pos_y = max([pos_y for pos_x, pos_y in positioned_filters.keys()])

    fig, axs = plt.subplots(
        nrows=max_pos_y, ncols=max_pos_x, figsize=[5*max_pos_x, 5*max_pos_y], sharex=True, sharey=True
    )
    for x_i in range(1, max_pos_x+1):
        for y_i in range(1, max_pos_y+1):
            if (x_i,y_i) not in positioned_filters.keys():
                continue
            if max_pos_x == 1 and max_pos_y == 1:
                ax = axs
            elif max_pos_x == 1:
                ax = axs[y_i-1]
            elif max_pos_y == 1:
                ax = axs[x_i-1]
            else:
                ax = axs[y_i-1][x_i-1]

            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(3.)


            list_of_keys = []
            custom_colors = {}
            legends = {}
            pos_data = {}
            for cell_id, filt_id in positioned_filters[(x_i,y_i)]:
                if cell_id not in data.keys() or filt_id not in data[cell_id].keys():
                    continue
                list_of_keys.append((cell_id, filt_id))
                custom_colors[(cell_id, filt_id)] = filt_colors[cell_id][filt_id]
                legends[(cell_id, filt_id)] = "{} {}".format(wet_names[cell_id], filt_names[cell_id][filt_id])
                pos_data[(cell_id, filt_id)]= data[cell_id][filt_id]

            plot_generic(
                "generic_vs_cycle", pos_data, list_of_keys, custom_colors,
                {"y":"total_discharge_capacity"}, ax,
                channel='scatter', plot_options={"sign_change":1.},
            )

            _ = produce_annotations(
                ax, get_list_of_patches(list_of_keys, custom_colors, legends), { "x_leg":0.5, "y_leg":0.5, "ylabel":"discharge cap (mAh)", "xlabel":"cycle number"}
            )


    # export
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)

    if output_dir is not None:
        path = os.path.join(
            output_dir,
        )
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, '{}.png'.format(escape_string_to_path(dataset_name))), dpi=dpi)
        plt.close(fig)
    else:
        return get_byte_image(fig, dpi)





'''
Now:
"Cycle Number",
 "Charge Capacity (mAh)",
 "Discharge Capacity (mAh)",

 "Average Charge Voltage (V)",
 "Average Discharge Voltage (V)",
 "Delta V (V)",

 "Charge Time (hours)",
"Discharge Time (hours)", \
"Cumulative Time (hours)",

Later:
"Normalized Charge Capacity",
"Normalized Discharge Capacity", "Zeroed Delta V (V)", "S (V)", "R (V)",
"Zeroed S (V)", "Zeroed R (V)"
'''

csv_format_default =[
    (Key.N, lambda x: str(int(x)), "Cycle Number"),
    ("total_charge_capacity", lambda x: "{:.3f}".format(x), "Total Charge Capacity (mAh)"),
    ("total_discharge_capacity", lambda x: "{:.3f}".format(x), "Total Discharge Capacity (mAh)"),

    ("avg_charge_voltage", lambda x: "{:.4f}".format(x), "Average Charge Voltage (V)"),
    ("avg_discharge_voltage", lambda x: "{:.4f}".format(x), "Average Discharge Voltage (V)"),
    ("delta_voltage", lambda x: "{:.6f}".format(x), "Delta Voltage (V)"),

    ("charge_time", lambda x: "{:.4f}".format(x), "Charge Time (hours)"),
    ("discharge_time", lambda x: "{:.4f}".format(x), "Discharge Time (hours)"),
    ("cumulative_time", lambda x: str(int(x)), "Cumulative Time (hours)"),

]


field_request_default = [
    (Key.N, 'f4', "CYCLE_NUMBER", None),
    ("total_charge_capacity", 'f4', "CUSTOM", lambda cyc: cyc.chg_total_capacity),
    ("total_discharge_capacity", 'f4', "CUSTOM", lambda cyc: cyc.dchg_total_capacity),

    ("avg_charge_voltage", 'f4', "CUSTOM", lambda cyc: cyc.chg_average_voltage),
    ("avg_discharge_voltage", 'f4', "CUSTOM", lambda cyc: cyc.dchg_average_voltage),
    ("delta_voltage", 'f4', "CUSTOM", lambda cyc: cyc.get_delta_v()),

    ("charge_time", 'f4', "CUSTOM", lambda cyc: cyc.chg_duration),
    ("discharge_time", 'f4', "CUSTOM", lambda cyc: cyc.dchg_duration),
    ("cumulative_time", 'f4', "CUMULATIVE_TIME", None),

]
