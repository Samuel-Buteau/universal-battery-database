import time

import numpy
import tensorflow as tf
from django.core.management.base import BaseCommand

from neware_parser.DegradationModel import DegradationModel
from neware_parser.LossRecord import LossRecord
from neware_parser.models import *
from neware_parser.plot import *
from neware_parser.Key import Key

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




    max_cap = my_data[Key.Q_MAX]

    for barcode_count, barcode in enumerate(barcodes):

        all_data = my_data[Key.ALL_DATA][barcode]
        cyc_grp_dict = all_data[Key.CYC_GRP_DICT]

        for k_count, k in enumerate(cyc_grp_dict.keys()):

            if any([
                abs(cyc_grp_dict[k][Key.I_PREV_END_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][Key.I_CC_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][Key.I_END_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][Key.V_PREV_END_AVG]) < 1e-1,
                abs(cyc_grp_dict[k][Key.V_END_AVG]) < 1e-1,
            ]):
                print(
                    cyc_grp_dict[k][Key.I_PREV_END_AVG],
                    cyc_grp_dict[k][Key.I_CC_AVG],
                    cyc_grp_dict[k][Key.I_END_AVG],
                    cyc_grp_dict[k][Key.V_PREV_END_AVG],
                    cyc_grp_dict[k][Key.V_END_AVG],
                )

            main_data = cyc_grp_dict[k][Key.MAIN]





class Command(BaseCommand):

    def add_arguments(self, parser):

        required_args = [
            "--path_to_dataset",
            "--dataset_version",
            "--path_to_plots",
        ]





        for arg in required_args:
            parser.add_argument(arg, required = True)

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
