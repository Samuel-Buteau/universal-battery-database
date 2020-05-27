import tensorflow as tf
import numpy as np
from machine_learning.DegradationModelBlackbox import DegradationModel
from plot import *


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


def data_engine_compiled(
    target: str, data, typ, generic_map, mode, max_cyc_n,
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

    return generic, list_of_keys #TODO returned generic_map before



def data_engine_model(
    target: str, data, typ, generic_map, mode, max_cyc_n,
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

    return generic, list_of_keys #TODO returned generic_map before



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
            filename = header + "_{}_Count_{}.png".format(cell_id, count),
            known_data_engines={
                "model":data_engine_model,
                "compiled":data_engine_compiled,
            },
            send_to_file=True,
        )
