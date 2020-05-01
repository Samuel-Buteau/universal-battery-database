import numpy as np
import tensorflow as tf

from neware_parser.Key import Key
from neware_parser.DegradationModel import DegradationModel


class DataEngine:

    @staticmethod
    def resistance(
        degradation_model: DegradationModel, barcode_count: int,
        cyc_grp_dict: dict, cycle_m, cycle_v, svit_and_count,
    ) -> dict:
        typ, off, mode = "dchg", 4, "cc"

        protocols = get_protocols(cyc_grp_dict, typ)
        resistances = []

        cycles = [x for x in np.arange(0., 6000., 20.)]
        my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycles]

        for count, protocol in enumerate(protocols):
            target_voltage = cyc_grp_dict[protocol]["avg_last_cc_voltage"]
            target_currents = [cyc_grp_dict[protocol][Key.I_CC_AVG]]

            test_results = test_single_voltage(
                my_cycle,
                target_voltage,
                cyc_grp_dict[protocol][Key.I_CC_AVG],
                cyc_grp_dict[protocol][Key.I_PREV_END_AVG],
                cyc_grp_dict[protocol][Key.V_PREV_END_AVG],
                cyc_grp_dict[protocol][Key.V_END_AVG],
                target_currents,
                barcode_count, degradation_model,
                svit_and_count[Key.SVIT_GRID],
                svit_and_count[Key.COUNT_MATRIX]
            )

            resistances.append(
                tf.reshape(test_results["pred_R"], shape = [-1])
            )
        return {
            "protocols": protocols,
            "resistances": resistances,
            "cycles": cycles
        }

    @staticmethod
    def shift(
        degradation_model: DegradationModel, barcode_count: int,
        cyc_grp_dict: dict, cycle_m, cycle_v, svit_and_count,
    ) -> dict:

        typ, off, mode = "dchg", 5, "cc"

        protocols = get_protocols(cyc_grp_dict, typ)
        shifts = []
        # TODO(harvey): Why not use np array instead of casting to list?
        cycles = [x for x in np.arange(0., 6000., 20.)]
        my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycles]

        for count, protocol in enumerate(protocols):
            test_results = test_single_voltage(
                my_cycle,
                cyc_grp_dict[protocol]["avg_last_cc_voltage"],  # target voltage
                cyc_grp_dict[protocol][Key.I_CC_AVG],
                cyc_grp_dict[protocol][Key.I_PREV_END_AVG],
                cyc_grp_dict[protocol][Key.V_PREV_END_AVG],
                cyc_grp_dict[protocol][Key.V_END_AVG],
                [cyc_grp_dict[protocol][Key.I_CC_AVG]],  # target currents
                barcode_count, degradation_model,
                svit_and_count[Key.SVIT_GRID],
                svit_and_count[Key.COUNT_MATRIX],
            )
            shifts.append(
                tf.reshape(test_results["pred_shift"], shape = [-1]),
            )
        return {
            "protocols": protocols,
            "shifts": shifts,
            "cycles": cycles
        }

    @staticmethod
    def scale(
        degradation_model: DegradationModel, barcode_count: int,
        cyc_grp_dict: dict, cycle_m, cycle_v, svit_and_count,
    ) -> dict:
        """ Compute the predicted scale and save it in a pickle file

        Args:
            cyc_grp_dict (dict)
            cycle_m: Cycle mean.
            cycle_v: Cycle variance.
            barcode_count: Number of barcodes.
            degradation_model
            svit_and_count: TODO(harvey)
            filename: Filename (including path) to save the generated pickle
                file
        """

        step, mode = "dchg", "cc"

        patches = []
        protocols = get_protocols(cyc_grp_dict, step)
        scales = []
        cycles = [x for x in np.arange(0., 6000., 20.)]
        my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycles]

        for count, protocol in enumerate(protocols):
            test_results = test_single_voltage(
                my_cycle,
                cyc_grp_dict[protocol]["avg_last_cc_voltage"],
                cyc_grp_dict[protocol][Key.I_CC_AVG],
                cyc_grp_dict[protocol][Key.I_PREV_END_AVG],
                cyc_grp_dict[protocol][Key.V_PREV_END_AVG],
                cyc_grp_dict[protocol][Key.V_END_AVG],
                [cyc_grp_dict[protocol][Key.I_CC_AVG]],
                barcode_count, degradation_model,
                svit_and_count[Key.SVIT_GRID],
                svit_and_count[Key.COUNT_MATRIX],
            )
            scales.append(tf.reshape(test_results["pred_scale"], shape = [-1]))
        return {
            "protocols": protocols,
            "scales": scales,
            "cycles": cycles,
            "patches": patches,
        }


# TODO(harvey): duplicate function in plot.py
def test_single_voltage(
    cycle, target_voltage, constant_current, end_current_prev,
    end_voltage_prev, end_voltage, target_currents, barcode_count,
    degradation_model, svit_grid, count_matrix,
):
    expanded_cycle = tf.expand_dims(cycle, axis = 1)
    expanded_constant_current = tf.constant(
        constant_current, shape = [len(cycle), 1],
    )
    expanded_end_current_prev = tf.constant(
        end_current_prev, shape = [len(cycle), 1],
    )
    expanded_end_voltage_prev = tf.constant(
        end_voltage_prev, shape = [len(cycle), 1],
    )
    expanded_end_voltage = tf.constant(end_voltage, shape = [len(cycle), 1])

    indecies = tf.tile(tf.expand_dims(barcode_count, axis = 0), [len(cycle)])

    expanded_svit_grid = tf.tile(
        tf.constant([svit_grid]), [len(cycle), 1, 1, 1, 1, 1],
    )
    expanded_count_matrix = tf.tile(
        tf.constant([count_matrix]), [len(cycle), 1, 1, 1, 1, 1],
    )

    return degradation_model(
        (
            expanded_cycle,
            expanded_constant_current,
            expanded_end_current_prev,
            expanded_end_voltage_prev,
            expanded_end_voltage,
            indecies,
            tf.constant(target_voltage, shape = [len(cycle), 1]),
            tf.tile(
                tf.reshape(target_currents, shape = [1, len(target_currents)]),
                [len(cycle), 1]
            ),
            expanded_svit_grid,
            expanded_count_matrix,
        ),
        training = False
    )


# TODO(harvey): duplicate function in plot.py
def get_protocols(cyc_grp_dict: dict, step: str) -> list:
    """
    Args:
        cyc_grp_dict (dict)
        step (str): Specifies charge or discharge
    Returns:
        list: Keys representing charge/discharge configurations
    """

    keys = [key for key in cyc_grp_dict.keys() if key[-1] == step]
    keys.sort(
        key = lambda k: (
            round(20. * k[0]), round(20. * k[1]), round(20. * k[2]),
            round(20. * k[3]), round(20. * k[4])
        )
    )
    return keys
