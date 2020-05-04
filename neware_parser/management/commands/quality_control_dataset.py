import time
import pickle
import numpy as np
from django.core.management.base import BaseCommand
from neware_parser.plot import *
from neware_parser.Key import Key
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes._axes import _log as matplotlib_axes_logger

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

    max_cap = my_data[Key.MAX_Q]


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
                fit_args[Key.MIN_V_GRID] > cyc_grp_dict[k][Key.V_PREV_END_AVG],
                fit_args[Key.MAX_V_GRID] < cyc_grp_dict[k][Key.V_PREV_END_AVG],
                fit_args[Key.MIN_V_GRID] > cyc_grp_dict[k][Key.V_END_AVG],
                fit_args[Key.MAX_V_GRID] < cyc_grp_dict[k][Key.V_END_AVG],

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



        plot_1(cyc_grp_dict, barcode, fit_args, legends=Preferred_Legends)




def plot_1(cyc_grp_dict, barcode, fit_args, legends=None):
    # x_lim = [-0.01, 1.01]
    y_lim = [2.95, 4.35]
    if legends is None:
        legends = {}


    fig, axs = plt.subplots(nrows = 3, figsize = [5, 10], sharex = True)

    for typ, off, mode, x_leg, y_leg in [
        ("dchg", 0, "cc", 0.5, 1),
        ("chg", 1, "cc", 0.5, 0.5),
        ("chg", 2, "cv", 0., 0.5)
    ]:
        list_of_keys = get_list_of_keys(cyc_grp_dict, typ)

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



        list_of_patches = []
        ax = axs[off]
        for k_count, k in enumerate(list_of_keys):


            color = custom_colors[k]

            list_of_patches.append(mpatches.Patch(
                color=COLORS[color], label=make_legend(k)
            ))

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

                # ax.set_xlim(x_lim)
                # ax.set_ylim(y_lim)

                ax.scatter(
                    sign_change * vq[valids],
                    y_axis[valids],
                    c = [[
                        mult * COLORS[color][0],
                        mult * COLORS[color][1],
                        mult * COLORS[color][2]
                    ]],
                    s = 3
                )


        ax.legend(
            handles = list_of_patches, fontsize = "small",
            bbox_to_anchor = (x_leg, y_leg), loc = "upper left"
        )
        ax.set_ylabel(typ + "-" + mode)

    axs[2].set_xlabel("pred_cap")
    fig.tight_layout()
    fig.subplots_adjust(hspace = 0)
    savefig("voltage_dependence_{}.png".format(barcode), fit_args)
    plt.close(fig)



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
