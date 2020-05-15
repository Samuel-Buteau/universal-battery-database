import pickle

from django.core.management.base import BaseCommand
from plot import *
from Key import Key

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

# TODO(sam): For each cell_id, needs a multigrid of (S, V, I, T) (current
#  needs to be adjusted)
# TODO(sam): Each cycle must have an index mapping to the nearest reference
#  cycle.
# TODO(sam): to evaluate a cycle, there must be the multigrid, the reference
#  cycle scaled by cycle number, the cell features, pasted together and ran
#  through a neural net.

NEIGH_MIN_CYC = 0
NEIGH_MAX_CYC = 1
NEIGH_RATE = 2
NEIGH_CELL_ID = 3
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

    cell_ids = list(my_data[Key.ALL_DATA].keys())

    initial_processing(
        my_data, cell_ids,
        fit_args,
    )


def initial_processing(my_data: dict, cell_ids, fit_args) -> dict:
    errors = []

    max_cap = my_data[Key.Q_MAX]

    for cell_id_count, cell_id in enumerate(cell_ids):
        all_data = my_data[Key.ALL_DATA][cell_id]
        cyc_grp_dict = all_data[Key.CYC_GRP_DICT]

        dual_legends = {}
        for typ in ['chg', 'dchg']:
            legends = {}
            for k in cyc_grp_dict.keys():
                if k[-1] == typ:
                    legend = make_legend(k)
                    if legend in legends.keys():
                        legends[legend] += 1
                    else:
                        legends[legend] = 1

            dual_legends_typ = []
            for legend in legends:
                if legends[legend] > 1:
                    dual_legends_typ.append(legend)

            if len(dual_legends_typ) > 0:
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
                fit_args["voltage_grid_min_v"]
                > cyc_grp_dict[k][Key.V_PREV_END_AVG],
                fit_args["voltage_grid_max_v"]
                < cyc_grp_dict[k][Key.V_PREV_END_AVG],
                fit_args["voltage_grid_min_v"]
                > cyc_grp_dict[k][Key.V_END_AVG],
                fit_args["voltage_grid_max_v"]
                < cyc_grp_dict[k][Key.V_END_AVG],

            ]):
                for i in range(len(cycs)):
                    errors.append(
                        {
                            "type": 'bad_group_bounds',
                            "flag": {
                                "cell_id": cell_id,
                                "group": k,
                                Key.CYC: cycs[i],
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
                            "flag": {
                                "cell_id": cell_id,
                                "group": k,
                                Key.CYC: cycs[i],
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
        with open(
            os.path.join(fit_args["path_to_flags"], "FLAGS.file"), 'wb',
        ) as file:
            pickle.dump(flags, file, pickle.HIGHEST_PROTOCOL)

        plot_engine_direct(
            data_streams = [('compiled', cyc_grp_dict, 'scatter', 20000)],
            target = 'generic_vs_capacity',
            todos = [
                ("dchg", "cc"),
                ("chg", "cc"),
                ("chg", "cv"),
            ],
            fit_args = fit_args,
            filename = "voltage_dependence_{}.png".format(cell_id),

        )

        plot_engine_direct(
            data_streams = [('compiled', cyc_grp_dict, 'scatter', 20000)],
            target = 'generic_vs_cycle',
            todos = [
                ("dchg", "cc"),
                ("chg", "cc"),
                ("chg", "cv"),
            ],
            fit_args = fit_args,
            filename = "cycle_dependence_{}.png".format(cell_id),

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
            parser.add_argument(arg, type = float, default = float_args[arg])

        # cell_ids = [
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
        #     "--wanted_cell_ids", type = int, nargs = "+", default = cell_ids,
        # )

    def handle(self, *args, **options):
        quality_control(options)
