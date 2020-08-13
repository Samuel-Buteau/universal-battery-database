import time
import os
import pickle

import numpy as np
import tensorflow as tf

from django.core.management.base import BaseCommand

from Key import Key
from plot import plot_direct, plot_v_vs_q

from cycling.models import id_dict_from_id_list
from machine_learning.DegradationModelBlackbox import DegradationModel
from machine_learning.LossRecordBlackbox import LossRecord

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
NEIGH_ABSOLUTE_CYC = 4
NEIGH_VALID_CYC = 5
NEIGH_SIGN_GRID = 6
NEIGH_VOLTAGE_GRID = 7
NEIGH_CURRENT_GRID = 8
NEIGH_TMP_GRID = 9
NEIGH_ABSOLUTE_REFERENCE = 10
NEIGH_REFERENCE = 11

NEIGH_TOTAL = 12


def ml_smoothing(options):
    """
    The main function to carry out the machine learning training and evaluation
    procedure.

    Todo(harvey): Add more description about what this does.

    Args:
        options: Dictionary defining various fitting-related arguments.

    Returns: None
    """
    if len(tf.config.experimental.list_physical_devices("GPU")) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device = "/gpu:0")
    elif len(tf.config.experimental.list_physical_devices("GPU")) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")

    if not os.path.exists(options[Key.PATH_PLOTS]):
        os.makedirs(options[Key.PATH_PLOTS])

    with open(
        os.path.join(options[Key.PATH_PLOTS], "fit_args_log.txt"), "w",
    ) as f:
        my_str = ""
        for k in options:
            my_str = "{} \n {}: {}".format(my_str, k, str(options[k]))
        f.write(my_str)

    dataset_path = os.path.join(
        options[Key.PATH_DATASET],
        "dataset_ver_{}.file".format(options[Key.DATA_VERSION])
    )

    dataset_names_path = os.path.join(
        options[Key.PATH_DATASET],
        "dataset_ver_{}_names.file".format(options[Key.DATA_VERSION])
    )

    if not os.path.exists(dataset_path):
        print("Path \"" + dataset_path + "\" does not exist.")
        return

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    dataset_names = None
    if os.path.exists(dataset_names_path):
        with open(dataset_names_path, "rb") as f:
            dataset_names = pickle.load(f)

    cell_ids = list(dataset[Key.ALL_DATA].keys())

    if len(options[Key.CELL_IDS]) != 0:
        cell_ids = list(set(cell_ids).intersection(set(options[Key.CELL_IDS])))

    if len(cell_ids) == 0:
        print("no cell_ids")
        return

    train_and_evaluate(
        initial_processing(
            dataset, dataset_names, cell_ids, options, strategy = strategy,
        ),
        cell_ids,
        options,
    )


# TODO(sam): these huge tensors would be much easier to understand with
#  ragged tensors. Right now, I am just flattening everything.
def numpy_acc(dict, key, data):
    if key in dict.keys():
        dict[key] = np.concatenate((dict[key], data))
    else:
        dict[key] = data

    return dict


def three_level_flatten(iterables):
    for it1 in iterables:
        for it2 in it1:
            for element in it2:
                yield element


def initial_processing(
    dataset: dict, dataset_names, cell_ids, options: dict, strategy,
) -> dict:
    """ Handle the initial data processing

    Args:
        dataset: Contains the quantities given in the dataset.
        dataset_names: TODO(harvey)
        cell_ids: TODO(harvey)
        options: Parameters used to tune the machine learning fitting process.
        strategy: TODO(harvey)

    Returns:
        { Key.STRAT, Key.MODEL, Key.TENSORS, Key.TRAIN_DS, Key.CYC_M,
          Key.CYC_V, Key.OPT, Key.MY_DATA }

    """
    # TODO (harvey): Cleanup Docstring, maybe put detailed description elsewhere
    #   An appropriate place might be in the docstring for
    #   classes inside cycling.Key

    compiled_data = {}
    compiled_cycs_count, reference_cycs_count = 0, 0
    dataset[Key.Q_MAX] = 250
    max_cap = dataset[Key.Q_MAX]
    keys = [Key.V_GRID, Key.TEMP_GRID, Key.SIGN_GRID]
    for key in keys:
        numpy_acc(compiled_data, key, np.array([dataset[key]]))

    dataset[Key.I_GRID] = dataset[Key.I_GRID] - np.log(max_cap)
    # the current grid is adjusted by the max capacity of the cell_id. It is
    # in log space, so I/q becomes log(I) - log(q)
    numpy_acc(compiled_data, Key.I_GRID, np.array([dataset[Key.I_GRID]]))

    for cell_id_count, cell_id in enumerate(cell_ids):

        all_data = dataset[Key.ALL_DATA][cell_id]
        cyc_grp_dict = all_data[Key.CYC_GRP_DICT]

        for k_count, k in enumerate(cyc_grp_dict.keys()):

            if any([
                abs(cyc_grp_dict[k][Key.I_PREV_END_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][Key.I_CC_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][Key.I_END_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][Key.V_PREV_END_AVG]) < 1e-1,
                abs(cyc_grp_dict[k][Key.V_END_AVG]) < 1e-1,
            ]):
                continue

            main_data = cyc_grp_dict[k][Key.MAIN]

            # normalize capacity_vector with max_cap
            normalize_keys = [
                Key.Q_CC_VEC, Key.Q_CV_VEC, Key.Q_CC_LAST, Key.Q_CV_LAST,
                Key.I_CV_VEC, Key.I_CC, Key.I_PREV_END,
            ]
            for key in normalize_keys:
                main_data[key] = 1. / max_cap * main_data[key]

            normalize_keys = [Key.I_CC_AVG, Key.I_END_AVG, Key.I_PREV_END_AVG]
            for key in normalize_keys:
                cyc_grp_dict[k][key] = 1. / max_cap * cyc_grp_dict[k][key]

            # range of cycles which exist for this cycle group
            min_cyc, max_cyc = min(main_data[Key.N]), max(main_data[Key.N])

            """
            - now create neighborhoods, which contains the cycles,
              grouped by proximity
            - want to sample neighborhoods equally
            - neighborhoods have a central cycle and a delta on each side
            - to a first approximation, we want a delta_cyc = 300, but we have
              to vary this near the beginning of data and near the end.
            """
            number_of_centers = 10

            # the centers of neighborhoods we will try to create
            all_neigh_center_cycs = np.linspace(
                min_cyc, max_cyc, number_of_centers,
            )
            delta = (
                1.2 * (all_neigh_center_cycs[1] - all_neigh_center_cycs[0]) + 10
            )

            # check all tentative neighborhood centers and
            # commit the ones that contain good data to the dataset
            neigh_data = []
            valid_cycs = 0
            for cyc in all_neigh_center_cycs:
                # max_cyc and min_cyc are the limits of existing cycles.

                below_cyc, above_cyc = cyc - delta, cyc + delta

                # numpy array of True and False; same length as cyc_grp_dict[k]
                # False when cycle_number falls outside out of
                # [below_cyc, above_cyc] interval
                mask = np.logical_and(
                    below_cyc <= main_data[Key.N],
                    main_data[Key.N] <= above_cyc,
                )

                # the indices for the cyc_grp_dict[k] array which correspond
                # to a True mask
                all_valid_indices = np.arange(len(mask))[mask]

                # if there are less than 1 valid cycles, skip that neighborhood
                if len(all_valid_indices) == 0:
                    continue

                """
                at this point, we know that this neighborhood
                will be added to the dataset.
                """

                min_cyc_index = all_valid_indices[0]
                max_cyc_index = all_valid_indices[-1]

                valid_cycs += 1

                """
                this commits the neighborhood to the dataset

                - record the info about the center of the neighborhood
                  (cycle number, voltage, rate of charge, rate of discharge)
                - record the relative index (within the cycle group)
                  of the min cycle, max cycle
                - record a voltage index, and a cycle group index,
                  and a cell index
                - record the absolute index into the table of cycles
                  (len(cycles_full)).
                - keep a slot empty for later

                """

                neigh_data_i = np.zeros(NEIGH_TOTAL, dtype = np.int32)

                neigh_data_i[NEIGH_MIN_CYC] = min_cyc_index
                neigh_data_i[NEIGH_MAX_CYC] = max_cyc_index
                neigh_data_i[NEIGH_RATE] = k_count
                neigh_data_i[NEIGH_CELL_ID] = cell_id_count
                neigh_data_i[NEIGH_ABSOLUTE_CYC] = compiled_cycs_count
                # a weight based on prevalence. Set later
                neigh_data_i[NEIGH_VALID_CYC] = 0
                neigh_data_i[NEIGH_SIGN_GRID] = 0
                neigh_data_i[NEIGH_VOLTAGE_GRID] = 0
                neigh_data_i[NEIGH_CURRENT_GRID] = 0
                neigh_data_i[NEIGH_TMP_GRID] = 0

                center_cyc = float(cyc)
                reference_cycs = all_data[Key.REF_ALL_MATS][Key.N]

                index_of_closest_reference = np.argmin(
                    abs(center_cyc - reference_cycs)
                )

                neigh_data_i[NEIGH_ABSOLUTE_REFERENCE] = reference_cycs_count
                neigh_data_i[NEIGH_REFERENCE] = index_of_closest_reference

                neigh_data.append(neigh_data_i)

            if valid_cycs != 0:
                neigh_data = np.array(neigh_data, dtype = np.int32)

                # the empty slot becomes the count of added neighborhoods, which
                # are used to counterbalance the bias toward longer cycle life
                neigh_data[:, NEIGH_VALID_CYC] = valid_cycs

                numpy_acc(compiled_data, Key.NEIGH_DATA, neigh_data)

            compiled_cycs_count += len(main_data[Key.N])
            reference_cycs_count += len(all_data[Key.REF_ALL_MATS][Key.N])

            dict_to_acc = {
                Key.REF_CYC: all_data[Key.REF_ALL_MATS][Key.N],
                Key.COUNT_MATRIX: all_data[Key.REF_ALL_MATS][Key.COUNT_MATRIX],
                Key.CYC: main_data[Key.N],
                Key.V_CC_VEC: main_data[Key.V_CC_VEC],
                Key.Q_CC_VEC: main_data[Key.Q_CC_VEC],
                Key.MASK_CC_VEC: main_data[Key.MASK_CC_VEC],
                Key.I_CV_VEC: main_data[Key.I_CV_VEC],
                Key.Q_CV_VEC: main_data[Key.Q_CV_VEC],
                Key.MASK_CV_VEC: main_data[Key.MASK_CV_VEC],
                Key.I_CC: main_data[Key.I_CC],
                Key.I_PREV_END: main_data[Key.I_PREV_END],
                Key.V_PREV_END: main_data[Key.V_PREV_END],
                Key.V_END: main_data[Key.V_END],
            }

            for key in dict_to_acc:
                numpy_acc(compiled_data, key, dict_to_acc[key])

    neigh_data = tf.constant(compiled_data[Key.NEIGH_DATA])

    compiled_tensors = {}
    # cycles go from 0 to 6000, but nn prefers normally distributed variables
    # so cycle numbers is normalized with mean and variance
    cycle_tensor = tf.constant(compiled_data[Key.CYC])
    cycle_m, cycle_v = tf.nn.moments(cycle_tensor, axes = [0])
    cycle_m = 0.  # we shall leave the cycle 0 at 0
    cycle_v = cycle_v.numpy()
    cycle_tensor = (cycle_tensor - cycle_m) / tf.sqrt(cycle_v)
    compiled_tensors[Key.CYC] = cycle_tensor

    labels = [
        Key.V_CC_VEC, Key.Q_CC_VEC, Key.MASK_CC_VEC, Key.Q_CV_VEC, Key.I_CV_VEC,
        Key.MASK_CV_VEC, Key.I_CC, Key.I_PREV_END, Key.V_PREV_END, Key.V_END,
        Key.COUNT_MATRIX, Key.SIGN_GRID, Key.V_GRID, Key.I_GRID, Key.TEMP_GRID,
    ]
    for label in labels:
        compiled_tensors[label] = tf.constant(compiled_data[label])

    with strategy.scope():
        train_ds = strategy.experimental_distribute_dataset(
            tf.data.Dataset.from_tensor_slices(
                neigh_data
            ).repeat(2).shuffle(100000).batch(options[Key.BATCH])
        )

        teacher_model = DegradationModel(
            width = options[Key.TEACHER_WIDTH],
            depth = options[Key.TEACHER_DEPTH],
            n_sample = options[Key.N_SAMPLE],
            options = options,
            cell_dict = id_dict_from_id_list(np.array(cell_ids)),
            sigmas = {
                Key.Q_SIG: options[Key.T_Q_SIG],
                Key.Q_SIG_N: options[Key.T_Q_SIG_N],
                Key.Q_SIG_V: options[Key.T_Q_SIG_V],
                Key.Q_SIG_I: options[Key.T_Q_SIG_I],
            },
        )
        student_model = DegradationModel(
            width = options[Key.STUDENT_WIDTH],
            depth = options[Key.STUDENT_DEPTH],
            n_sample = options[Key.N_SAMPLE],
            options = options,
            cell_dict = id_dict_from_id_list(np.array(cell_ids)),
            sigmas = {
                Key.Q_SIG: options[Key.S_Q_SIG],
                Key.Q_SIG_N: options[Key.S_Q_SIG_N],
                Key.Q_SIG_V: options[Key.S_Q_SIG_V],
                Key.Q_SIG_I: options[Key.S_Q_SIG_I],
            },
        )

    return {
        Key.STRAT: strategy,
        Key.TEACHER_MODEL: teacher_model,
        Key.STUDENT_MODEL: student_model,
        Key.TENSORS: compiled_tensors,
        Key.TRAIN_DS: train_ds,
        Key.CYC_M: cycle_m,
        Key.CYC_V: cycle_v,
        Key.TEACHER_OPTIMIZER: tf.keras.optimizers.Adam(
            learning_rate = options[Key.TEACHER_LRN_RATE],
        ),
        Key.STUDENT_OPTIMIZER: tf.keras.optimizers.Adam(
            learning_rate = options[Key.STUDENT_LRN_RATE],
        ),
        Key.DATASET: dataset,
    }


def to_sorted_array(unsorted) -> np.array:
    """ Remove duplicates from a list or a view object (of a dictionary) and
        turn it into to a sorted array.

    Args:
        unsorted: Unsorted list or view object (of a dictionary).

    Returns:
        Sorted array of `unsorted`.
    """
    return np.array(sorted(list(set(unsorted))))


def train_and_evaluate(
    init_returns: dict, cell_ids: list, options: dict,
) -> None:
    """

    Args:
        init_returns: Return value of `initial_processing`.
        cell_ids: Specified cell IDs (identifiers for different cells).
        options:
    """
    strategy = init_returns[Key.STRAT]

    epochs = 100000
    count = 0

    end = time.time()

    train_step_params = {
        Key.TENSORS: init_returns[Key.TENSORS],
        Key.TEACHER_OPTIMIZER: init_returns[Key.TEACHER_OPTIMIZER],
        Key.STUDENT_OPTIMIZER: init_returns[Key.STUDENT_OPTIMIZER],
        Key.TEACHER_MODEL: init_returns[Key.TEACHER_MODEL],
        Key.STUDENT_MODEL: init_returns[Key.STUDENT_MODEL],
    }

    @tf.function
    def dist_train_step(strategy, neigh):
        return strategy.experimental_run_v2(
            lambda neigh: train_step(neigh, train_step_params, options),
            args = (neigh,),
        )

    # TODO(harvey): what is `l`?
    l = None
    loss_record = LossRecord()
    with strategy.scope():
        for epoch in range(epochs):
            sub_count = 0
            for neigh in init_returns[Key.TRAIN_DS]:
                count += 1
                sub_count += 1

                if l is None:
                    l = dist_train_step(strategy, neigh)
                else:
                    l += dist_train_step(strategy, neigh)

                if count != 0:
                    if (count % options[Key.PRINT_LOSS]) == 0:
                        tot = l / tf.cast(sub_count, dtype = tf.float32)
                        l = None
                        sub_count = 0
                        loss_record.record(count, tot.numpy())
                        loss_record.print_recent(options)

                    if (count % options[Key.VIS_FIT]) == 0:
                        plot_params = {
                            "cell_ids": cell_ids,
                            "count": count,
                            Key.OPTIONS: options,
                        }
                        start = time.time()
                        print("time to simulate: ", start - end)
                        loss_record.plot(count, options)
                        plot_direct(
                            "generic_vs_cycle", plot_params, init_returns,
                            teacher = False,
                        )
                        plot_direct(
                            "generic_vs_capacity", plot_params, init_returns,
                            teacher = False,
                        )

                        end = time.time()
                        print("time to plot: ", end - start)
                        print()

                if count >= options[Key.STOP]:
                    return


def train_step(neigh, train_params: dict, options: dict):
    """ One training step.

    Args:
        neigh: Neighbourhood.
        train_params: Contains all necessary parameters.
        options: Options for `ml_smoothing`.
    """
    # need to split the range
    batch_size2 = neigh.shape[0]
    n_sample = options[Key.N_SAMPLE]

    teacher_model = train_params[Key.TEACHER_MODEL]
    student_model = train_params[Key.STUDENT_MODEL]
    teacher_optimizer = train_params[Key.TEACHER_OPTIMIZER]
    student_optimizer = train_params[Key.STUDENT_OPTIMIZER]
    compiled_tensors = train_params[Key.TENSORS]

    sign_grid_tensor = compiled_tensors[Key.SIGN_GRID]
    voltage_grid_tensor = compiled_tensors[Key.V_GRID]
    current_grid_tensor = compiled_tensors[Key.I_GRID]
    tmp_grid_tensor = compiled_tensors[Key.TEMP_GRID]

    count_matrix_tensor = compiled_tensors[Key.COUNT_MATRIX]

    cycle_tensor = compiled_tensors[Key.CYC]
    constant_current_tensor = compiled_tensors[Key.I_CC]
    end_current_prev_tensor = compiled_tensors[Key.I_PREV_END]
    end_voltage_prev_tensor = compiled_tensors[Key.V_PREV_END]
    end_voltage_tensor = compiled_tensors[Key.V_END]

    cc_voltage_tensor = compiled_tensors[Key.V_CC_VEC]
    cc_capacity_tensor = compiled_tensors[Key.Q_CC_VEC]
    cc_mask_tensor = compiled_tensors[Key.MASK_CC_VEC]
    cv_capacity_tensor = compiled_tensors[Key.Q_CV_VEC]
    cv_current_tensor = compiled_tensors[Key.I_CV_VEC]
    cv_mask_tensor = compiled_tensors[Key.MASK_CV_VEC]

    """
    if you have the minimum cycle and maximum cycle for a neighborhood,
    you can sample cycle from this neighborhood by sampling real numbers
    x from [0,1] and computing min_cyc*(1.-x) + max_cyc*x,
    but here this computation is done in index space,
    then cycle numbers and vq curves are gathered
    """

    cyc_indices_lerp = tf.random.uniform(
        [batch_size2], minval = 0., maxval = 1., dtype = tf.float32,
    )
    cyc_indices = tf.cast(
        (1. - cyc_indices_lerp) * tf.cast(
            neigh[:, NEIGH_MIN_CYC] + neigh[:, NEIGH_ABSOLUTE_CYC],
            tf.float32,
        ) + cyc_indices_lerp * tf.cast(
            neigh[:, NEIGH_MAX_CYC] + neigh[:, NEIGH_ABSOLUTE_CYC],
            tf.float32,
        ),
        tf.int32,
    )

    sign_grid = tf.gather(
        sign_grid_tensor, indices = neigh[:, NEIGH_SIGN_GRID], axis = 0,
    )
    sign_grid_dim = sign_grid.shape[1]

    voltage_grid = tf.gather(
        voltage_grid_tensor, indices = neigh[:, NEIGH_VOLTAGE_GRID], axis = 0,
    )
    voltage_grid_dim = voltage_grid.shape[1]

    current_grid = tf.gather(
        current_grid_tensor, indices = neigh[:, NEIGH_CURRENT_GRID], axis = 0,
    )
    current_grid_dim = current_grid.shape[1]

    tmp_grid = tf.gather(
        tmp_grid_tensor, indices = neigh[:, NEIGH_TMP_GRID], axis = 0,
    )
    tmp_grid_dim = tmp_grid.shape[1]

    svit_tuple = (
        tf.tile(
            tf.reshape(
                sign_grid, [batch_size2, sign_grid_dim, 1, 1, 1, 1],
            ),
            [1, 1, voltage_grid_dim, current_grid_dim, tmp_grid_dim, 1],
        ),
        tf.tile(
            tf.reshape(
                voltage_grid, [batch_size2, 1, voltage_grid_dim, 1, 1, 1],
            ),
            [1, sign_grid_dim, 1, current_grid_dim, tmp_grid_dim, 1],
        ),
        tf.tile(
            tf.reshape(
                current_grid, [batch_size2, 1, 1, current_grid_dim, 1, 1],
            ),
            [1, sign_grid_dim, voltage_grid_dim, 1, tmp_grid_dim, 1],
        ),
        tf.tile(
            tf.reshape(
                tmp_grid, [batch_size2, 1, 1, 1, tmp_grid_dim, 1],
            ),
            [1, sign_grid_dim, voltage_grid_dim, current_grid_dim, 1, 1],
        ),
    )
    svit_grid = tf.concat(svit_tuple, axis = -1)

    count_matrix = tf.reshape(
        tf.gather(
            count_matrix_tensor,
            neigh[:, NEIGH_ABSOLUTE_REFERENCE] + neigh[:, NEIGH_REFERENCE],
            axis = 0,
        ),
        [
            batch_size2, sign_grid_dim, voltage_grid_dim, current_grid_dim,
            tmp_grid_dim, 1,
        ],
    )

    cycle = tf.gather(cycle_tensor, indices = cyc_indices, axis = 0)
    constant_current = tf.gather(
        constant_current_tensor, indices = cyc_indices, axis = 0,
    )
    end_current_prev = tf.gather(
        end_current_prev_tensor, indices = cyc_indices, axis = 0,
    )
    end_voltage_prev = tf.gather(
        end_voltage_prev_tensor, indices = cyc_indices, axis = 0,
    )
    end_voltage = tf.gather(
        end_voltage_tensor, indices = cyc_indices, axis = 0,
    )

    cc_capacity = tf.gather(cc_capacity_tensor, indices = cyc_indices)
    cc_voltage = tf.gather(cc_voltage_tensor, indices = cyc_indices)
    cc_mask = tf.gather(cc_mask_tensor, indices = cyc_indices)
    cc_mask_2 = tf.tile(
        tf.reshape(
            1. / tf.cast(neigh[:, NEIGH_VALID_CYC], tf.float32),
            [batch_size2, 1],
        ),
        [1, cc_voltage.shape[1]],
    )

    cv_capacity = tf.gather(cv_capacity_tensor, indices = cyc_indices)
    cv_current = tf.gather(cv_current_tensor, indices = cyc_indices)
    cv_mask = tf.gather(cv_mask_tensor, indices = cyc_indices)
    cv_mask_2 = tf.tile(
        tf.reshape(
            1. / tf.cast(neigh[:, NEIGH_VALID_CYC], tf.float32),
            [batch_size2, 1],
        ),
        [1, cv_current.shape[1]],
    )

    cell_indices = neigh[:, NEIGH_CELL_ID]

    with tf.GradientTape() as tape:
        teacher_train_results = teacher_model(
            {
                Key.CYC: tf.expand_dims(cycle, axis = 1),
                Key.I_CC: tf.expand_dims(constant_current, axis = 1),
                Key.I_PREV_END: tf.expand_dims(end_current_prev, axis = 1),
                Key.V_PREV_END: tf.expand_dims(end_voltage_prev, axis = 1),
                Key.V_END: tf.expand_dims(end_voltage, axis = 1),
                Key.INDICES: cell_indices,
                Key.V_TENSOR: cc_voltage,
                Key.I_TENSOR: cv_current,
                Key.SVIT_GRID: svit_grid,
                Key.COUNT_MATRIX: count_matrix,
            },
            training = True,
        )

        teacher_pred_cc_cap = teacher_train_results[Key.Pred.I_CC]
        teacher_pred_cv_cap = teacher_train_results[Key.Pred.I_CV]

        cc_capacity_loss = get_loss(
            cc_capacity, teacher_pred_cc_cap, cc_mask, cc_mask_2,
        )
        cv_capacity_loss = get_loss(
            cv_capacity, teacher_pred_cv_cap, cv_mask, cv_mask_2,
        )

        main_losses = (
            options[Key.Coeff.Q_CV] * cv_capacity_loss
            + options[Key.Coeff.Q_CC] * cc_capacity_loss
        )
        loss = main_losses + tf.stop_gradient(main_losses) * (
            options[Key.Coeff.Q] * teacher_train_results[Key.Loss.Q]
        )

    gradients = tape.gradient(loss, teacher_model.trainable_variables)

    gradients_no_nans = [
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), x) for x in gradients
    ]

    gradients_norm_clipped, _ = tf.clip_by_global_norm(
        gradients_no_nans, options[Key.GLB_NORM_CLIP],
    )

    teacher_optimizer.apply_gradients(
        zip(gradients_norm_clipped, teacher_model.trainable_variables)
    )

    sampled_constant_current = tf.exp(
        tf.random.uniform(
            minval = tf.math.log(0.001), maxval = tf.math.log(5.),
            shape = [n_sample, 1],
        )
    )
    sampled_constant_current_sign = tf.cast(
        tf.random.uniform(
            minval = 0, maxval = 1,
            shape = [n_sample, 1], dtype = tf.int32,
        ),
        dtype = tf.float32,
    )
    sampled_constant_current_sign = 2.0 * sampled_constant_current_sign - 1.

    sampled_feats_cell = student_model.cell_from_indices(
        indices = tf.random.uniform(
            maxval = student_model.cell_direct.num_keys,
            shape = [n_sample], dtype = tf.int32,
        ),
        training = False, sample = True,
    )
    student_samples = {
        Key.SAMPLE_V: tf.random.uniform(
            minval = 2.5, maxval = 5., shape = [n_sample, 1],
        ),
        Key.SAMPLE_CYC: tf.random.uniform(
            minval = -.1, maxval = 5., shape = [n_sample, 1],
        ),
        Key.SAMPLE_I: (
            sampled_constant_current_sign * sampled_constant_current
        ),
        Key.SAMPLE_CELL_FEAT: tf.stop_gradient(
            sampled_feats_cell
        ),
    }

    with tf.GradientTape() as student_tape:
        teacher_q = teacher_model(
            {
                Key.CYC: tf.expand_dims(cycle, axis = 1),
                Key.I_CC: tf.expand_dims(constant_current, axis = 1),
                Key.I_PREV_END: tf.expand_dims(end_current_prev, axis = 1),
                Key.V_PREV_END: tf.expand_dims(end_voltage_prev, axis = 1),
                Key.V_END: tf.expand_dims(end_voltage, axis = 1),
                Key.INDICES: cell_indices,
                Key.V_TENSOR: cc_voltage,
                Key.I_TENSOR: cv_current,
                Key.SVIT_GRID: svit_grid,
                Key.COUNT_MATRIX: count_matrix,
            },
            training = True,
        )[Key.Q]
        student_q = student_model.q_direct(
            cycle = student_samples[Key.SAMPLE_CYC],
            v = student_samples[Key.SAMPLE_V],
            current = student_samples[Key.SAMPLE_I],
            feats_cell = student_samples[Key.SAMPLE_CELL_FEAT],
        )

        student_loss = options[Key.Coeff.STUDENT_Q] * tf.reduce_mean(
            tf.square(
                tf.stop_gradient(teacher_q) - student_q
            )
        )

    gradients_student = student_tape.gradient(
        student_loss, student_model.trainable_variables,
    )
    for x in gradients_student:
        if x is None:
            gradients_student.remove(x)
    gradients_no_nans_student = [
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        for x in gradients_student
    ]
    gradients_norm_clipped_student, _ = tf.clip_by_global_norm(
        gradients_no_nans_student, options[Key.GLB_NORM_CLIP],
    )

    student_optimizer.apply_gradients(
        zip(gradients_norm_clipped_student, student_model.trainable_variables)
    )

    return tf.stack(
        [cc_capacity_loss, cv_capacity_loss, teacher_train_results[Key.Loss.Q]],
    )


def get_loss(measured, predicted, mask, mask_2):
    return tf.reduce_mean(
        (1e-10 + mask * mask_2) * tf.square(measured - predicted)
    ) / (1e-10 + tf.reduce_mean(mask * mask_2))


class Command(BaseCommand):

    def add_arguments(self, parser):

        required_args = [Key.PATH_DATASET, Key.DATA_VERSION, Key.PATH_PLOTS]

        float_args = {
            Key.GLB_NORM_CLIP: 10.,

            Key.TEACHER_LRN_RATE: 5e-4,
            Key.STUDENT_LRN_RATE: 5e-2,
            Key.MIN_LAT: 1,

            Key.Coeff.FEAT_CELL_DER: .001,
            Key.Coeff.FEAT_CELL_DER2: .01,

            Key.Coeff.CELL: 1.,
            Key.Coeff.CELL_OUT: .1,
            Key.Coeff.CELL_IN: .1,
            Key.Coeff.CELL_DER: .1,
            Key.Coeff.CELL_EQ: 10.,

            Key.Coeff.LYTE: 1.,
            Key.Coeff.LYTE_OUT: .1,
            Key.Coeff.LYTE_IN: .1,
            Key.Coeff.LYTE_DER: .1,
            Key.Coeff.LYTE_EQ: 10.,

            Key.Coeff.Q_CV: 1.,
            Key.Coeff.Q_CC: 1.,

            Key.Coeff.Q: 0.0001,
            Key.Coeff.STUDENT_Q: 0.01,
            Key.Coeff.Q_GEQ: 1.,
            Key.Coeff.Q_LEQ: 1.,
            Key.Coeff.Q_V_MONO: 0.,
            Key.Coeff.Q_DER3_V: 0.,
            Key.Coeff.Q_DER3_I: 0.,
            Key.Coeff.Q_DER3_N: 0.,
            Key.Coeff.Q_DER_I: 0.,
            Key.Coeff.Q_DER_N: 0.,

            Key.T_Q_SIG: 0.08,
            Key.T_Q_SIG_N: 1.5,
            Key.T_Q_SIG_V: 1.2,
            Key.T_Q_SIG_I: .5,

            Key.S_Q_SIG: 1,
            Key.S_Q_SIG_N: 1.5,
            Key.S_Q_SIG_V: 1.2,
            Key.S_Q_SIG_I: 1.5,
        }

        vis = 1000
        int_args = {
            Key.FOUR_FEAT: 1,
            Key.N_SAMPLE: 8 * 16,

            Key.TEACHER_DEPTH: 3,
            Key.TEACHER_WIDTH: 50,
            Key.STUDENT_DEPTH: 2,
            Key.STUDENT_WIDTH: 30,
            Key.BATCH: 4 * 16,

            Key.PRINT_LOSS: vis,
            Key.VIS_FIT: vis,
            Key.VIS_VQ: vis,

            Key.STOP: 16000,
            Key.CELL_ID_SHOW: 10,
        }

        for arg in required_args:
            parser.add_argument("--" + arg, required = True)
        for arg in float_args:
            parser.add_argument(
                "--" + arg, type = float, default = float_args[arg],
            )
        for arg in int_args:
            parser.add_argument("--" + arg, type = int, default = int_args[arg])

        cell_ids = [
            57706, 57707, 57710, 57711, 57714, 57715, 64260, 64268, 83010,
            83011, 83012, 83013, 83014, 83015, 83016, 81602, 81603, 81604,
            81605, 81606, 81607, 81608, 81609, 81610, 81611, 81612, 81613,
            81614, 81615, 81616, 81617, 81618, 81619, 81620, 81621, 81622,
            81623, 81624, 81625, 81626, 81627, 81712, 81713, 82300, 82301,
            82302, 82303, 82304, 82305, 82306, 82307, 82308, 82309, 82310,
            82311, 82406, 82407, 82410, 82411, 82769, 82770, 82771, 82775,
            82776, 82777, 82779, 82992, 82993, 83083, 83092, 83101, 83106,
            83107, 83220, 83221, 83222, 83223, 83224, 83225, 83226, 83227,
            83228, 83229, 83230, 83231, 83232, 83233, 83234, 83235, 83236,
            83237, 83239, 83240, 83241, 83242, 83243, 83310, 83311, 83312,
            83317, 83318, 83593, 83594, 83595, 83596, 83741, 83742, 83743,
            83744, 83745, 83746, 83747, 83748,
        ]

        parser.add_argument(
            "--" + Key.CELL_IDS, type = int, nargs = "+", default = cell_ids,
        )

    def handle(self, *args, **options):
        ml_smoothing(options)
