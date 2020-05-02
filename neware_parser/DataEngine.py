import sys

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
    ):
        """
        Computes protocol-independent quantities vs capacity:
            - resistance
            - scale
            - shift
        """
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

    @staticmethod
    def measured_capacity(cyc_grp_dict, mode, protocols) -> dict:
        """
        Return:
            "caps": A list of structured arrays. One structured array consists of
            the cycle and capacity for one protocol. There the length of the list is
            the number of protocols.
        """

        caps = []
        for count, protocol in enumerate(protocols):

            if protocol[-1] == "dchg":
                sign_change = -1.
            else:
                sign_change = +1.

            if mode == "cc":
                cap = cyc_grp_dict[protocol][Key.MAIN]["last_cc_capacity"]
                cap_mode = Key.Q_CC
            elif mode == "cv":
                cap = cyc_grp_dict[protocol][Key.MAIN]["last_cv_capacity"]
                cap_mode = Key.Q_CV
            else:
                sys.exit("Unknown mode in measured.")

            caps.append(
                np.array(
                    list(zip(
                        cyc_grp_dict[protocol][Key.MAIN][Key.N],
                        sign_change * cap,
                    )),
                    dtype = [
                        (Key.N, "f4"),
                        (cap_mode, "f4"),
                    ]
                )
            )
        return {
            "mode": mode,
            "protocols": protocols,
            "caps": caps,
        }

    @staticmethod
    def predicted_capacity(
        cyc_grp_dict, mode, protocols, cycle_m, cycle_v, barcode_count,
        degradation_model, svit_and_count,
    ) -> dict:
        """
        Return:
            dict: With the following keys:

                "caps" - A list of structured arrays. One structured array consists
                    of the cycle and capacity for one protocol. There the length of
                    the list is the number of protocols.
        """

        cycles = [x for x in np.arange(0., 6000., 20.)]
        my_cycle = [(cyc - cycle_m) / tf.sqrt(cycle_v) for cyc in cycles]
        caps = []

        for count, protocol in enumerate(protocols):

            if protocol[-1] == "dchg":
                sign_change = -1.
            else:
                sign_change = +1.

            if mode == "cc":
                target_voltage = cyc_grp_dict[protocol]["avg_last_cc_voltage"]
                target_currents = [cyc_grp_dict[protocol][Key.I_CC_AVG]]
            elif mode == "cv":
                target_voltage = cyc_grp_dict[protocol][Key.V_END_AVG]
                curr_min = abs(cyc_grp_dict[protocol][Key.I_CC_AVG])
                curr_max = abs(cyc_grp_dict[protocol]["avg_end_current"])

                if curr_min == curr_max:
                    target_currents = np.array([curr_min])
                else:
                    target_currents = sign_change * np.exp(
                        np.arange(
                            np.log(curr_min),
                            np.log(curr_max),
                            .05 * (np.log(curr_max) - np.log(curr_min))
                        )
                    )
            else:
                sys.exit("Unknown mode in predicted.")

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

            if mode == "cc":
                pred_cap = tf.reshape(
                    test_results["pred_cc_capacity"], shape = [-1],
                )
            elif mode == "cv":
                pred_cap = test_results["pred_cv_capacity"].numpy()[:, -1]
            else:
                sys.exit("Unknown mode in predicted.")

            caps.append(sign_change * pred_cap)
        return {
            "mode": mode,
            "protocols": protocols,
            "caps": caps,
            "cycles": cycles,
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
