
import pickle

from django.core.management.base import BaseCommand
from cycling.plot import *
from cycling.Key import Key
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
Shortened Variable Names:
    vol -   voltage
    cap -   capacity
    dchg -  discharge
    neigh - neighbourhood
    der -   derivative
    pred -  predicted
    meas -  measured
    eval -  evaluation
    eq -    equilibrium
    res -   result
"""

# TODO(sam): For each barcode, needs a multigrid of (S, V, I, T) (current
#  needs to be adjusted)
# TODO(sam): Each cycle must have an index mapping to the nearest reference
#  cycle.
# TODO(sam): to evaluate a cycle, there must be the multigrid, the reference
#  cycle scaled by cycle number, the cell features, pasted together and ran
#  through a neural net.

NEIGH_MIN_CYC = 0
NEIGH_MAX_CYC = 1
NEIGH_RATE = 2
NEIGH_BARCODE = 3
NEIGH_ABSOLUTE_CYCLE = 4
NEIGH_VALID_CYC = 5
NEIGH_SIGN_GRID = 6
NEIGH_VOLTAGE_GRID = 7
NEIGH_CURRENT_GRID = 8
NEIGH_TEMPERATURE_GRID = 9
NEIGH_ABSOLUTE_REFERENCE = 10
NEIGH_REFERENCE = 11

NEIGH_TOTAL = 12


def quality_control(fit_args):

    if not os.path.exists(fit_args[Key.PATH_PLOTS]):
        os.makedirs(fit_args[Key.PATH_PLOTS])

    if not os.path.exists(fit_args["path_to_flags"]):
        os.makedirs(fit_args["path_to_flags"])

    dataset_path = os.path.join(
        fit_args[Key.PATH_DATASET],
        "dataset_ver_{}.file".format(fit_args[Key.DATA_VERSION])
    )


    if not os.path.exists(dataset_path):
        print("Path \"" + dataset_path + "\" does not exist.")
        return


    with open(dataset_path, "rb") as f:
        my_data = pickle.load(f)


    barcodes = list(my_data[Key.ALL_DATA].keys())



    initial_processing(
        my_data, barcodes,
        fit_args,
    )


def initial_processing(
    my_data: dict, barcodes,
    fit_args
) -> dict:


    errors = []

    max_cap = my_data[Key.Q_MAX]


    for barcode_count, barcode in enumerate(barcodes):
        all_data = my_data[Key.ALL_DATA][barcode]
        cyc_grp_dict = all_data[Key.CYC_GRP_DICT]

        dual_legends = {}
        for typ in ['chg', 'dchg']:
            legends = {}
            for k in cyc_grp_dict.keys():
                if k[-1] == typ:
                    legend = make_legend(k)
                    if legend in legends.keys():
                        legends[legend] +=1
                    else:
                        legends[legend] = 1

            dual_legends_typ = []
            for legend in legends:
                if legends[legend] > 1:
                    dual_legends_typ.append(legend)

            if len(dual_legends_typ) >0:
                dual_legends[typ] = dual_legends_typ

        for k_count, k in enumerate(cyc_grp_dict.keys()):
            if k[-1] in dual_legends.keys():
                legend = make_legend(k)
                if legend in dual_legends[k[-1]]:
                    print(k)
            caps = cyc_grp_dict[k][Key.MAIN][Key.Q_CC_VEC]
            cycs = cyc_grp_dict[k][Key.MAIN][Key.N]
            vols = cyc_grp_dict[k][Key.MAIN][Key.V_CC_VEC]
            masks = cyc_grp_dict[k][Key.MAIN][Key.MASK_CC_VEC]

            if any([
                abs(cyc_grp_dict[k][Key.I_PREV_END_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][Key.I_CC_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][Key.I_END_AVG]) < 1e-5,
                fit_args["voltage_grid_min_v"] > cyc_grp_dict[k][Key.V_PREV_END_AVG],
                fit_args["voltage_grid_max_v"] < cyc_grp_dict[k][Key.V_PREV_END_AVG],
                fit_args["voltage_grid_min_v"] > cyc_grp_dict[k][Key.V_END_AVG],
                fit_args["voltage_grid_max_v"] < cyc_grp_dict[k][Key.V_END_AVG],

            ]):
                for i in range(len(cycs)):
                    errors.append(
                        {
                            "type": 'bad_group_bounds',
                            "flag": {
                                "barcode": barcode,
                                "group": k,
                                "cycle": cycs[i],
                            },
                            "caps": caps[i],
                            "vols": vols[i],
                            "masks": masks[i]
                        }
                    )

            # Test for points of opposite polarity
            if k[-1] == 'dchg':
                f = lambda c: c > 1e-3
            elif k[-1] == 'chg':
                f = lambda c: c < -1e-3
            else:
                continue

            for i in range(len(cycs)):
                if any([f(c) for c in caps[i]]):
                    errors.append(
                        {
                            "type": 'opposite_polarity',
                            "flag":{
                                "barcode": barcode,
                                "group": k,
                                "cycle": cycs[i],
                            },
                            "caps": caps[i],
                            "vols": vols[i],
                            "masks": masks[i]
                        }
                    )
        if len(errors) > 0:
            print('Full error list:')
            print(errors)

        types = list(set([e["type"] for e in errors]))

        flags = {}
        for type in types:
            flags[type] = [e["flag"] for e in errors if e["type"] == type]

        if len(flags) > 0:
            print('Only the flags')
            print(flags)
        with open(os.path.join(fit_args["path_to_flags"], "FLAGS.file"), 'wb') as file:
            pickle.dump(flags, file, pickle.HIGHEST_PROTOCOL)

        plot_engine_direct(
            data_streams = [('compiled', cyc_grp_dict, 'scatter')],
            target = 'generic_vs_capacity',
            todos = [
                ("dchg", "cc"),
                ("chg", "cc"),
                ("chg", "cv"),
            ],
            barcode= barcode,
            fit_args = fit_args,
        )



#TODO(sam): make the interface more general
#TODO(sam): how to separate measured from predicted?
def plot_engine_direct(data_streams, target, todos, barcode, fit_args):
    # open plot
    fig, axs = plt.subplots(nrows=len(todos), figsize=[5, 10], sharex=True)

    for i, todo in enumerate(todos):
        typ, mode = todo
        ax = axs[i]

        # options
        options = generate_options(mode, typ)

        list_of_target_data = []

        for source, data, _ in data_streams:
            # data engine from compiled to generic_vs_capacity
            list_of_target_data.append(
                data_engine(
                    source,
                    target,
                    data,
                    typ,
                    mode
                )
            )

        _, list_of_keys, _ = list_of_target_data[0]
        custom_colors = map_legend_to_color(list_of_keys)

        for j, target_data in enumerate(list_of_target_data):
            generic_vs_capacity, _, generic_map = target_data
            # plot
            plot_generic_vs_capacity(
                generic_vs_capacity, list_of_keys, custom_colors,
                generic_map, ax,
                channel=data_streams[j][2],
                options=options,
            )

        produce_annotations(ax, get_list_of_patches(list_of_keys, custom_colors), options)

    # export
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    savefig("voltage_dependence_{}.png".format(barcode), fit_args)
    plt.close(fig)


def generate_options(mode, typ):
    #label
    y_quantity = ""
    if mode == 'cc':
        y_quantity = 'Voltage'
    elif mode == 'cv':
        y_quantity = 'Current'
    ylabel = typ + "-" + mode + "\n" + y_quantity
    xlabel = "Capacity"

    # sign_change
    if typ == "dchg":
        sign_change = -1.
    else:
        sign_change = +1.

    #leg
    leg = {
        ("dchg", "cc"): (.5, 1.),
        ("chg", "cc"): (.5, .5),
        ("chg", "cv"): (0., .5),
    }

    x_leg, y_leg = leg[(typ,mode)]

    return {
        "sign_change": sign_change,
        "x_leg": x_leg,
        "y_leg": y_leg,
        "xlabel": xlabel,
        "ylabel": ylabel
    }



def data_engine(
        source,
        target,
        data,
        typ,
        mode,
    ):
    if not (source == 'compiled' and target == 'generic_vs_capacity'):
        return None, None, None, None

    list_of_keys = get_list_of_keys(data, typ)
    needed_fields = []
    generic_map = {}
    if mode == 'cc':
        needed_fields = [Key.N, "cc_capacity_vector", "cc_voltage_vector", "cc_mask_vector"]
        generic_map = {
            'x': "cc_capacity_vector",
            'y': "cc_voltage_vector",
            'mask': "cc_mask_vector"
        }
    elif mode == 'cv':
        needed_fields = [Key.N, "cv_capacity_vector", "cv_current_vector", "cv_mask_vector"]
        generic_map = {
            'x': "cv_capacity_vector",
            'y': "cv_current_vector",
            'mask': "cv_mask_vector"
        }

    generic_vs_capacity = {}
    for k in list_of_keys:
        generic_vs_capacity[k] = data[k][Key.MAIN][needed_fields]

    return generic_vs_capacity, list_of_keys, generic_map


def map_legend_to_color(list_of_keys):
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
    return [[
            mult * color[0],
            mult * color[1],
            mult * color[2]
        ]]


def produce_annotations(ax, list_of_patches, options):
    ax.legend(
        handles = list_of_patches, fontsize = "small",
        bbox_to_anchor = (options["x_leg"], options["y_leg"]), loc = "upper left"
    )
    ax.set_ylabel(options["ylabel"])
    ax.set_xlabel(options["xlabel"])



def plot_generic_vs_capacity(
        groups,list_of_keys, custom_colors, generic_map,
        ax,
        channel,
        options
    ):
    if channel == 'scatter':
        plotter = ax.scatter
    else:
        raise Exception("not yet implemented. channel = {}".format(channel))

    for k in list_of_keys:
        group = groups[k]
        for i in range(len(group)):
            valids = group[generic_map['mask']][i] > .5
            plotter(
                options["sign_change"] * group[generic_map['x']][i][valids],
                group[generic_map['y']][i][valids],
                c = adjust_color(group[Key.N][i], custom_colors[k]),
                s = 3
            )



class Command(BaseCommand):

    def add_arguments(self, parser):

        required_args = [
            "--path_to_dataset",
            "--dataset_version",
            "--path_to_plots",
            "--path_to_flags"
        ]

        float_args = {
            "--voltage_grid_min_v": 2.5,
            "--voltage_grid_max_v": 5.0,

        }



        for arg in required_args:
            parser.add_argument(arg, required = True)
        for arg in float_args:
            parser.add_argument(arg, type=float, default=float_args[arg])

        # barcodes = [
        #     81602, 81603, 81604, 81605, 81606, 81607, 81608, 81609, 81610,
        #     81611, 81612, 81613, 81614, 81615, 81616, 81617, 81618, 81619,
        #     81620, 81621, 81622, 81623, 81624, 81625, 81626, 81627, 81712,
        #     81713, 82300, 82301, 82302, 82303, 82304, 82305, 82306, 82307,
        #     82308, 82309, 82310, 82311, 82406, 82407, 82410, 82411, 82769,
        #     82770, 82771, 82775, 82776, 82777, 82779, 82992, 82993, 83083,
        #     83092, 83101, 83106, 83107, 83220, 83221, 83222, 83223, 83224,
        #     83225, 83226, 83227, 83228, 83229, 83230, 83231, 83232, 83233,
        #     83234, 83235, 83236, 83237, 83239, 83240, 83241, 83242, 83243,
        #     83310, 83311, 83312, 83317, 83318, 83593, 83594, 83595, 83596,
        #     83741, 83742, 83743, 83744, 83745, 83746, 83747, 83748,
        # ]
        # 57706, 57707, 57710, 57711, 57714, 57715,64260,64268,83010, 83011,
        # 83012, 83013, 83014, 83015, 83016

        # parser.add_argument(
        #     "--wanted_barcodes", type = int, nargs = "+", default = barcodes,
        # )

    def handle(self, *args, **options):
        quality_control(options)
