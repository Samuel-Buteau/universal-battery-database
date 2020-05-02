import numpy as np
import tensorflow as tf

from neware_parser.Key import Key
from neware_parser.DegradationModel import DegradationModel
from neware_parser.Print import Print


class DataEngine:

    @staticmethod
    def protocol_independent(
        degradation_model: DegradationModel, barcode_count: int,
        cyc_grp_dict: dict, cycle_m, cycle_v, svit_and_count,
    ) -> dict:
        typ, mode = "dchg", "cc"

        protocol = get_protocols(cyc_grp_dict, typ)[0]

        cycles = [x for x in np.arange(0., 6000., 20.)]
        my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycles]

        target_voltage = cyc_grp_dict[protocol]["avg_last_cc_voltage"]
        target_currents = [cyc_grp_dict[protocol][Key.I_CC_AVG]]

        model_predictions = test_single_voltage(
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

        return np.array(
            list(zip(
                cycles,
                tf.reshape(model_predictions["pred_R"], shape = [-1]),
                tf.reshape(model_predictions["pred_scale"], shape = [-1]),
                tf.reshape(model_predictions["pred_shift"], shape = [-1]),
            )),
            dtype = [
                (Key.N, "f4"),
                (Key.R, "f4"),
                (Key.SCALE, "f4"),
                (Key.SHIFT, "f4"),
            ]
        )


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
