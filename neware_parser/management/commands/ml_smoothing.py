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


def ml_smoothing(fit_args):
    if len(tf.config.experimental.list_physical_devices("GPU")) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device = "/gpu:0")
    elif len(tf.config.experimental.list_physical_devices("GPU")) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")

    if not os.path.exists(fit_args[Key.PATH_PLOTS]):
        os.makedirs(fit_args[Key.PATH_PLOTS])

    with open(
        os.path.join(fit_args[Key.PATH_PLOTS], "fit_args_log.txt"), "w"
    ) as f:
        my_str = ""
        for k in fit_args:
            my_str = "{} \n {}: {}".format(my_str, k, str(fit_args[k]))
        f.write(my_str)

    dataset_path = os.path.join(
        fit_args[Key.PATH_DATASET],
        "dataset_ver_{}.file".format(fit_args[Key.DATA_VERSION])
    )

    dataset_names_path = os.path.join(
        fit_args[Key.PATH_DATASET],
        "dataset_ver_{}_names.file".format(fit_args[Key.DATA_VERSION])
    )

    v_curves_path = os.path.join(
        fit_args[Key.PATH_V_CURVES],
        'v_curves.file'
    )

    if not os.path.exists(dataset_path):
        print("Path \"" + dataset_path + "\" does not exist.")
        return

    if not os.path.exists(v_curves_path):
        print("Path \"" + v_curves_path + "\" does not exist.")
        return

    with open(v_curves_path, "rb") as f:
        my_v_curves = pickle.load(f)
        my_anode_v_curves = my_v_curves['anode']
        my_cathode_v_curves = my_v_curves['cathode']

    with open(dataset_path, "rb") as f:
        my_data = pickle.load(f)

    my_names = None
    if os.path.exists(dataset_names_path):
        with open(dataset_names_path, "rb") as f:
            my_names = pickle.load(f)

    barcodes = list(my_data[Key.ALL_DATA].keys())

    if len(fit_args[Key.BARCODES]) != 0:
        barcodes = list(
            set(barcodes).intersection(set(fit_args[Key.BARCODES])))

    if len(barcodes) == 0:
        print("no barcodes")
        return

    train_and_evaluate(
        initial_processing(
            my_data, my_names, barcodes, my_anode_v_curves, my_cathode_v_curves,
            fit_args, strategy = strategy,
        ),
        barcodes,
        fit_args
    )


# TODO(sam): these huge tensors would be much easier to understand with
#  ragged tensors.
# right now, I am just flattening everything.

def numpy_acc(my_dict, my_key, my_dat):
    if my_key in my_dict.keys():
        my_dict[my_key] = numpy.concatenate(
            (my_dict[my_key], my_dat)
        )
    else:
        my_dict[my_key] = my_dat

    return my_dict


def three_level_flatten(iterables):
    for it1 in iterables:
        for it2 in it1:
            for element in it2:
                yield element


def initial_processing(
    my_data: dict, my_names, barcodes, anode_v_curves, cathode_v_curves,
    fit_args, strategy
) -> dict:
    """ Handle the initial data processing

    Args:
        my_data (dictionary).
        my_names: TODO(harvey)
        barcodes: TODO(harvey)
        anode_v_curves: TODO(harvey)
        cathode_v_curves: TODO(harvey)
        fit_args: TODO(harvey)
        strategy: TODO(harvey)

    Returns:
        {
           Key.STRAT, Key.MODEL, Key.TENSORS, Key.TRAIN_DS, Key.CYC_M,
           Key.CYC_V, Key.OPT, Key.MY_DATA
        }

    """
    # TODO (harvey): Cleanup Docstring, maybe put detailed description elsewhere
    #                An appropriate place might be in the docstring for
    #                classes inside neware_parser.Key

    compiled_data = {}
    number_of_compiled_cycles = 0
    number_of_reference_cycles = 0

    my_data[Key.Q_MAX] = 250
    max_cap = my_data[Key.Q_MAX]

    keys = [Key.V_GRID, Key.TEMP_GRID, Key.SIGN_GRID]
    for key in keys:
        numpy_acc(compiled_data, key, numpy.array([my_data[key]]))

    my_data[Key.I_GRID] = my_data[Key.I_GRID] - numpy.log(max_cap)
    # the current grid is adjusted by the max capacity of the barcode. It is
    # in log space, so I/q becomes log(I) - log(q)
    numpy_acc(
        compiled_data, Key.I_GRID,
        numpy.array([my_data[Key.I_GRID]])
    )

    # TODO (harvey): simplify the following using loops
    cell_id_list = numpy.array(barcodes)
    cell_id_to_pos_id = {}
    cell_id_to_neg_id = {}
    cell_id_to_electrolyte_id = {}
    cell_id_to_dry_cell_id = {}
    dry_cell_id_to_meta = {}
    cell_id_to_latent = {}

    electrolyte_id_to_latent = {}
    electrolyte_id_to_solvent_id_weight = {}
    electrolyte_id_to_salt_id_weight = {}
    electrolyte_id_to_additive_id_weight = {}

    for cell_id in cell_id_list:
        if cell_id in my_data[Key.CELL_TO_POS].keys():
            cell_id_to_pos_id[cell_id]\
                = my_data[Key.CELL_TO_POS][cell_id]
        if cell_id in my_data[Key.CELL_TO_NEG].keys():
            cell_id_to_neg_id[cell_id]\
                = my_data[Key.CELL_TO_NEG][cell_id]
        if cell_id in my_data[Key.CELL_TO_ELE].keys():
            cell_id_to_electrolyte_id[cell_id]\
                = my_data[Key.CELL_TO_ELE][cell_id]
        if cell_id in my_data["cell_to_dry"].keys():
            dry_cell_id = my_data["cell_to_dry"][cell_id]
            cell_id_to_dry_cell_id[cell_id] = dry_cell_id

            if dry_cell_id in my_data["dry_to_meta"].keys():
                dry_cell_id_to_meta[dry_cell_id]\
                    = my_data["dry_to_meta"][dry_cell_id]

        if cell_id in my_data[Key.CELL_TO_LAT].keys():
            cell_id_to_latent[cell_id]\
                = my_data[Key.CELL_TO_LAT][cell_id]

        if cell_id_to_latent[cell_id] < 0.5:
            electrolyte_id = cell_id_to_electrolyte_id[cell_id]
            if electrolyte_id in my_data[Key.ELE_TO_SOL].keys():
                electrolyte_id_to_solvent_id_weight[electrolyte_id]\
                    = my_data[Key.ELE_TO_SOL][electrolyte_id]
            if electrolyte_id in my_data[Key.ELE_TO_SALT].keys():
                electrolyte_id_to_salt_id_weight[electrolyte_id]\
                    = my_data[Key.ELE_TO_SALT][electrolyte_id]
            if electrolyte_id in my_data[Key.ELE_TO_ADD].keys():
                electrolyte_id_to_additive_id_weight[electrolyte_id]\
                    = my_data[Key.ELE_TO_ADD][electrolyte_id]

            if electrolyte_id in my_data[Key.ELE_TO_LAT].keys():
                electrolyte_id_to_latent[electrolyte_id]\
                    = my_data[Key.ELE_TO_LAT][electrolyte_id]

    mess = [
        [
            [s[0] for s in siw] for siw in
            electrolyte_id_to_solvent_id_weight.values()
        ],
        [
            [s[0] for s in siw] for siw in
            electrolyte_id_to_salt_id_weight.values()
        ],
        [
            [s[0] for s in siw] for siw in
            electrolyte_id_to_additive_id_weight.values()
        ],
    ]

    molecule_id_list = numpy.array(
        # flatten, remove duplicates, then sort
        sorted(list(set(list(three_level_flatten(mess)))))
    )

    dry_cell_id_list = numpy.array(
        sorted(list(set(cell_id_to_dry_cell_id.values())))
    )
    pos_id_list = numpy.array(sorted(list(set(cell_id_to_pos_id.values()))))
    neg_id_list = numpy.array(sorted(list(set(cell_id_to_neg_id.values()))))
    electrolyte_id_list = numpy.array(
        sorted(list(set(cell_id_to_electrolyte_id.values())))
    )

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
                continue

            main_data = cyc_grp_dict[k][Key.MAIN]

            # normalize capacity_vector with max_cap
            normalize_keys = [
                Key.Q_CC_VEC, Key.Q_CV_VEC, Key.Q_CC_LAST,
                Key.Q_CV_LAST, Key.I_CV_VEC, Key.I_CC,
                Key.I_PREV
            ]
            for key in normalize_keys:
                main_data[key] = 1. / max_cap * main_data[key]

            normalize_keys = [
                Key.I_CC_AVG, Key.I_END_AVG,
                Key.I_PREV_END_AVG
            ]
            for key in normalize_keys:
                cyc_grp_dict[k][key] = 1. / max_cap * cyc_grp_dict[k][key]

            # range of cycles which exist for this cycle group
            min_cyc = min(main_data[Key.N])
            max_cyc = max(main_data[Key.N])

            """
            - now create neighborhoods, which contains the cycles,
              grouped by proximity
            - want to sample neighborhoods equally
            - neighborhoods have a central cycle and a delta on each side
            - to a first approximation, we want a delta_cyc = 300, but we have
              to vary this near the beginning of data and near the end.
            """

            # gives an absolute scale
            total_delta = max_cyc - min_cyc

            # the baseline, at least 200, but up to total_delta/5
            delta_cyc = max(200, int(float(total_delta) / 5.))

            # the centers of neighborhoods we will try to create
            all_neigh_center_cycles = list(filter(
                lambda x: x > min_cyc - 100,
                range(20, int(max_cyc + 50), 40))
            )

            # check all tentative neighborhood centers and
            # commit the ones that contain good data to the dataset
            neighborhood_data = []

            valid_cycles = 0
            for cyc in all_neigh_center_cycles:
                # max_cyc and min_cyc are the limits of existing cycles.

                # at least 200, but can extend up to the limit
                # starting from the current neighborhood center
                delta_up = max(max_cyc - cyc, 200)

                # same thing going down
                delta_down = max(cyc - min_cyc, 200)

                # the max symetric interval that fits into the
                # [cyc - delta_down, cyc + delta_up] interval is
                # [cyc - delta_actual, cyc + delta_actual]
                delta_actual = min(delta_up, delta_down)

                # choose the largest interval that fits both
                # [cyc - delta_actual, cyc + delta_actual] and
                # [cyc - delta_cyc, cyc + delta_cyc]
                combined_delta = min(delta_actual, delta_cyc)

                below_cyc = cyc - combined_delta
                above_cyc = cyc + combined_delta

                # numpy array of True and False; same length as cyc_grp_dict[k]
                # False when cycle_number falls outside out of
                # [below_cyc, above_cyc] interval
                mask = numpy.logical_and(
                    below_cyc <= main_data[Key.N],
                    main_data[Key.N] <= above_cyc
                )

                # the indices for the cyc_grp_dict[k] array which correspond
                # to a True mask
                all_valid_indices = numpy.arange(len(mask))[mask]

                # if there are less than 2 valid cycles, skip that neighborhood
                if len(all_valid_indices) < 2:
                    continue

                """
                at this point, we know that this neighborhood
                will be added to the dataset.
                """

                min_cyc_index = all_valid_indices[0]
                max_cyc_index = all_valid_indices[-1]

                valid_cycles += 1

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

                # TODO(sam): here, figure out where the reference cycle is,
                #            and note the index

                neighborhood_data_i = numpy.zeros(
                    NEIGH_TOTAL, dtype = numpy.int32
                )

                neighborhood_data_i[NEIGH_MIN_CYC] = min_cyc_index
                neighborhood_data_i[NEIGH_MAX_CYC] = max_cyc_index
                neighborhood_data_i[NEIGH_RATE] = k_count
                neighborhood_data_i[NEIGH_BARCODE] = barcode_count
                neighborhood_data_i[NEIGH_ABSOLUTE_CYCLE]\
                    = number_of_compiled_cycles
                # a weight based on prevalence. Set later
                neighborhood_data_i[NEIGH_VALID_CYC] = 0
                neighborhood_data_i[NEIGH_SIGN_GRID] = 0
                neighborhood_data_i[NEIGH_VOLTAGE_GRID] = 0
                neighborhood_data_i[NEIGH_CURRENT_GRID] = 0
                neighborhood_data_i[NEIGH_TEMPERATURE_GRID] = 0

                center_cycle = float(cyc)
                reference_cycles = all_data[Key.REF_ALL_MATS][Key.N]

                index_of_closest_reference = numpy.argmin(
                    abs(center_cycle - reference_cycles)
                )

                neighborhood_data_i[NEIGH_ABSOLUTE_REFERENCE]\
                    = number_of_reference_cycles
                neighborhood_data_i[NEIGH_REFERENCE]\
                    = index_of_closest_reference

                neighborhood_data.append(neighborhood_data_i)

            if valid_cycles != 0:
                neighborhood_data = numpy.array(
                    neighborhood_data, dtype = numpy.int32,
                )

                # the empty slot becomes the count of added neighborhoods, which
                # are used to counterbalance the bias toward longer cycle life
                neighborhood_data[:, NEIGH_VALID_CYC] = valid_cycles

                numpy_acc(compiled_data, "neighborhood_data", neighborhood_data)

            number_of_compiled_cycles += len(main_data[Key.N])
            number_of_reference_cycles\
                += len(all_data[Key.REF_ALL_MATS][Key.N])

            dict_to_acc = {
                "reference_cycle": all_data[Key.REF_ALL_MATS][Key.N],
                Key.COUNT_MATRIX:
                    all_data[Key.REF_ALL_MATS][Key.COUNT_MATRIX],
                "cycle": main_data[Key.N],
                Key.V_CC_VEC: main_data[Key.V_CC_VEC],
                Key.Q_CC_VEC: main_data[Key.Q_CC_VEC],
                Key.MASK_CC_VEC: main_data[Key.MASK_CC_VEC],
                Key.I_CV_VEC: main_data[Key.I_CV_VEC],
                Key.Q_CV_VEC: main_data[Key.Q_CV_VEC],
                Key.MASK_CV_VEC: main_data[Key.MASK_CV_VEC],
                Key.I_CC: main_data[Key.I_CC],
                Key.I_PREV: main_data[Key.I_PREV],
                Key.V_PREV_END: main_data[Key.V_PREV_END],
                Key.V_END: main_data[Key.V_END],
            }

            for key in dict_to_acc:
                numpy_acc(compiled_data, key, dict_to_acc[key])

    neighborhood_data = tf.constant(compiled_data["neighborhood_data"])

    compiled_tensors = {}
    # cycles go from 0 to 6000, but nn prefers normally distributed variables
    # so cycle numbers is normalized with mean and variance
    cycle_tensor = tf.constant(compiled_data["cycle"])
    cycle_m, cycle_v = tf.nn.moments(cycle_tensor, axes = [0])
    cycle_m = 0.  # we shall leave the cycle 0 at 0
    cycle_v = cycle_v.numpy()
    cycle_tensor = (cycle_tensor - cycle_m) / tf.sqrt(cycle_v)
    compiled_tensors["cycle"] = cycle_tensor

    compiled_tensors["anode_v_curve"] = tf.constant(
        list(anode_v_curves.values())[0]
    )
    compiled_tensors["cathode_v_curve"] = tf.constant(
        list(cathode_v_curves.values())[0]
    )
    labels = [
        Key.V_CC_VEC, Key.Q_CC_VEC, Key.MASK_CC_VEC, Key.Q_CV_VEC, Key.I_CV_VEC,
        Key.MASK_CV_VEC, Key.I_CC, Key.I_PREV, Key.V_PREV_END, Key.V_END,
        Key.COUNT_MATRIX, Key.SIGN_GRID, Key.V_GRID, Key.I_GRID, Key.TEMP_GRID,
    ]
    for label in labels:
        compiled_tensors[label] = tf.constant(compiled_data[label])

    batch_size = fit_args[Key.BATCH]

    with strategy.scope():
        train_ds_ = tf.data.Dataset.from_tensor_slices(
            neighborhood_data
        ).repeat(2).shuffle(100000).batch(batch_size)

        train_ds = strategy.experimental_distribute_dataset(train_ds_)

        dry_cell_to_dry_cell_name = {}
        pos_to_pos_name = {}
        neg_to_neg_name = {}
        electrolyte_to_electrolyte_name = {}
        molecule_to_molecule_name = {}

        if my_names is not None:
            pos_to_pos_name = my_names[Key.POS_TO_POS]
            neg_to_neg_name = my_names[Key.NEG_TO_NEG]
            electrolyte_to_electrolyte_name\
                = my_names[Key.ELE_TO_ELE]
            molecule_to_molecule_name = my_names[Key.MOL_TO_MOL]
            dry_cell_to_dry_cell_name = my_names["dry_to_dry_name"]

        degradation_model = DegradationModel(
            width = fit_args[Key.WIDTH],
            depth = fit_args[Key.DEPTH],
            cell_dict = id_dict_from_id_list(cell_id_list),
            pos_dict = id_dict_from_id_list(pos_id_list),
            neg_dict = id_dict_from_id_list(neg_id_list),
            electrolyte_dict = id_dict_from_id_list(electrolyte_id_list),
            molecule_dict = id_dict_from_id_list(molecule_id_list),
            dry_cell_dict = id_dict_from_id_list(dry_cell_id_list),

            cell_to_pos = cell_id_to_pos_id,
            cell_to_neg = cell_id_to_neg_id,
            cell_to_electrolyte = cell_id_to_electrolyte_id,
            cell_to_dry_cell = cell_id_to_dry_cell_id,
            dry_cell_to_meta = dry_cell_id_to_meta,

            cell_latent_flags = cell_id_to_latent,

            electrolyte_to_solvent = electrolyte_id_to_solvent_id_weight,
            electrolyte_to_salt = electrolyte_id_to_salt_id_weight,
            electrolyte_to_additive = electrolyte_id_to_additive_id_weight,
            electrolyte_latent_flags = electrolyte_id_to_latent,

            names = (
                pos_to_pos_name,
                neg_to_neg_name,
                electrolyte_to_electrolyte_name,
                molecule_to_molecule_name,
                dry_cell_to_dry_cell_name,
            ),
            n_sample = fit_args[Key.N_SAMPLE],
            incentive_coeffs = fit_args,
            min_latent = fit_args[Key.MIN_LAT]
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = fit_args[Key.LRN_RATE]
        )

    return {
        Key.STRAT: strategy,
        Key.MODEL: degradation_model,
        Key.TENSORS: compiled_tensors,
        Key.TRAIN_DS: train_ds,
        Key.CYC_M: cycle_m,
        Key.CYC_V: cycle_v,
        Key.OPT: optimizer,
        Key.MY_DATA: my_data
    }


def train_and_evaluate(init_returns, barcodes, fit_args):
    strategy = init_returns[Key.STRAT]

    epochs = 100000
    count = 0

    end = time.time()
    now_ker = None

    train_step_params = {
        Key.TENSORS: init_returns[Key.TENSORS],
        Key.OPT: init_returns[Key.OPT],
        Key.MODEL: init_returns[Key.MODEL],
    }

    @tf.function
    def dist_train_step(strategy, neighborhood):
        return strategy.experimental_run_v2(
            lambda neighborhood: train_step(
                neighborhood, train_step_params, fit_args
            ),
            args = (neighborhood,)
        )

    l = None
    loss_record = LossRecord()
    with strategy.scope():
        for epoch in range(epochs):
            sub_count = 0
            for neighborhood in init_returns[Key.TRAIN_DS]:
                count += 1
                sub_count += 1

                l_ = dist_train_step(strategy, neighborhood)
                if l is None:
                    l = l_
                else:
                    l += l_

                if count != 0:
                    if (count % fit_args[Key.PRINT_LOSS]) == 0:
                        tot = l / tf.cast(sub_count, dtype = tf.float32)
                        l = None
                        sub_count = 0
                        loss_record.record(count, tot.numpy())
                        loss_record.print_recent(fit_args)

                    plot_params = {
                        "barcodes": barcodes,
                        "count": count,
                        Key.FIT_ARGS: fit_args,
                    }

                    if (count % fit_args[Key.VIS]) == 0:

                        start = time.time()
                        print("time to simulate: ", start - end)
                        loss_record.plot(count, fit_args)
                        plot_things_vs_cycle_number(plot_params, init_returns)
                        plot_vq(plot_params, init_returns)
                        # plot_v_curves(plot_params, init_returns)
                        end = time.time()
                        print("time to plot: ", end - start)
                        ker = init_returns[
                            Key.MODEL
                        ].cell_direct.kernel.numpy()
                        prev_ker = now_ker
                        now_ker = ker

                        # TODO(sam):
                        # this analysis should be more thorough, but how to
                        # generalize to
                        # more than two barcodes? for now, there is very
                        # little use for that.
                        if now_ker is not None:
                            delta_cell_ker = numpy.abs(now_ker[0] - now_ker[1])
                            print(
                                "average difference between cells: ",
                                numpy.average(delta_cell_ker)
                            )
                        if prev_ker is not None:
                            delta_time_ker = numpy.abs(now_ker - prev_ker)
                            print(
                                "average difference between prev and now: ",
                                numpy.average(delta_time_ker)
                            )
                        print()

                if count >= fit_args[Key.STOP]:
                    return


def train_step(neighborhood, params, fit_args):
    sign_grid_tensor = params[Key.TENSORS][Key.SIGN_GRID]
    voltage_grid_tensor = params[Key.TENSORS][Key.V_GRID]
    current_grid_tensor = params[Key.TENSORS][Key.I_GRID]
    temperature_grid_tensor = params[Key.TENSORS][Key.TEMP_GRID]

    count_matrix_tensor = params[Key.TENSORS][Key.COUNT_MATRIX]

    cycle_tensor = params[Key.TENSORS]["cycle"]
    constant_current_tensor = params[Key.TENSORS][Key.I_CC]
    end_current_prev_tensor = params[Key.TENSORS][Key.I_PREV]
    end_voltage_prev_tensor = params[Key.TENSORS][Key.V_PREV_END]
    end_voltage_tensor = params[Key.TENSORS][Key.V_END]

    degradation_model = params[Key.MODEL]
    optimizer = params[Key.OPT]

    cc_voltage_tensor = params[Key.TENSORS][Key.V_CC_VEC]
    cc_capacity_tensor = params[Key.TENSORS][Key.Q_CC_VEC]
    cc_mask_tensor = params[Key.TENSORS][Key.MASK_CC_VEC]
    cv_capacity_tensor = params[Key.TENSORS][Key.Q_CV_VEC]
    cv_current_tensor = params[Key.TENSORS][Key.I_CV_VEC]
    cv_mask_tensor = params[Key.TENSORS][Key.MASK_CV_VEC]
    anode_v_curve = params[Key.TENSORS]["anode_v_curve"]
    cathode_v_curve = params[Key.TENSORS]["cathode_v_curve"]
    # need to split the range
    batch_size2 = neighborhood.shape[0]

    """
    if you have the minimum cycle and maximum cycle for a neighborhood,
    you can sample cycle from this neighborhood by sampling real numbers
    x from [0,1] and computing min_cyc*(1.-x) + max_cyc*x,
    but here this computation is done in index space,
    then cycle numbers and vq curves are gathered
    """

    cycle_indices_lerp = tf.random.uniform(
        [batch_size2], minval = 0., maxval = 1., dtype = tf.float32)
    cycle_indices = tf.cast(
        (1. - cycle_indices_lerp) * tf.cast(
            neighborhood[:, NEIGH_MIN_CYC]
            + neighborhood[:, NEIGH_ABSOLUTE_CYCLE],
            tf.float32
        ) + cycle_indices_lerp * tf.cast(
            neighborhood[:, NEIGH_MAX_CYC]
            + neighborhood[:, NEIGH_ABSOLUTE_CYCLE],
            tf.float32
        ),
        tf.int32
    )

    sign_grid = tf.gather(
        sign_grid_tensor,
        indices = neighborhood[:, NEIGH_SIGN_GRID], axis = 0
    )
    sign_grid_dim = sign_grid.shape[1]

    voltage_grid = tf.gather(
        voltage_grid_tensor,
        indices = neighborhood[:, NEIGH_VOLTAGE_GRID], axis = 0
    )
    voltage_grid_dim = voltage_grid.shape[1]

    current_grid = tf.gather(
        current_grid_tensor,
        indices = neighborhood[:, NEIGH_CURRENT_GRID], axis = 0
    )
    current_grid_dim = current_grid.shape[1]

    temperature_grid = tf.gather(
        temperature_grid_tensor,
        indices = neighborhood[:, NEIGH_TEMPERATURE_GRID], axis = 0
    )
    temperature_grid_dim = temperature_grid.shape[1]

    svit_tuple = (
        tf.tile(
            tf.reshape(
                sign_grid,
                [batch_size2, sign_grid_dim, 1, 1, 1, 1]
            ),
            [1, 1, voltage_grid_dim, current_grid_dim, temperature_grid_dim, 1],
        ),
        tf.tile(
            tf.reshape(
                voltage_grid,
                [batch_size2, 1, voltage_grid_dim, 1, 1, 1]
            ),
            [1, sign_grid_dim, 1, current_grid_dim, temperature_grid_dim, 1],
        ),
        tf.tile(
            tf.reshape(
                current_grid,
                [batch_size2, 1, 1, current_grid_dim, 1, 1]
            ),
            [1, sign_grid_dim, voltage_grid_dim, 1, temperature_grid_dim, 1],
        ),
        tf.tile(
            tf.reshape(
                temperature_grid,
                [batch_size2, 1, 1, 1, temperature_grid_dim, 1]
            ),
            [1, sign_grid_dim, voltage_grid_dim, current_grid_dim, 1, 1],
        ),
    )
    svit_grid = tf.concat(svit_tuple, axis = -1)

    count_matrix = tf.reshape(
        tf.gather(
            count_matrix_tensor,
            (
                neighborhood[:, NEIGH_ABSOLUTE_REFERENCE]
                + neighborhood[:, NEIGH_REFERENCE]
            ),
            axis = 0,
        ),
        [batch_size2, sign_grid_dim, voltage_grid_dim, current_grid_dim,
         temperature_grid_dim, 1],
    )

    cycle = tf.gather(cycle_tensor, indices = cycle_indices, axis = 0)
    constant_current = tf.gather(
        constant_current_tensor, indices = cycle_indices, axis = 0
    )
    end_current_prev = tf.gather(
        end_current_prev_tensor, indices = cycle_indices, axis = 0
    )
    end_voltage_prev = tf.gather(
        end_voltage_prev_tensor, indices = cycle_indices, axis = 0
    )
    end_voltage = tf.gather(
        end_voltage_tensor, indices = cycle_indices, axis = 0
    )

    cc_capacity = tf.gather(cc_capacity_tensor, indices = cycle_indices)
    cc_voltage = tf.gather(cc_voltage_tensor, indices = cycle_indices)
    cc_mask = tf.gather(cc_mask_tensor, indices = cycle_indices)
    cc_mask_2 = tf.tile(
        tf.reshape(
            1. / tf.cast(neighborhood[:, NEIGH_VALID_CYC], tf.float32),
            [batch_size2, 1],
        ),
        [1, cc_voltage.shape[1]],
    )

    cv_capacity = tf.gather(cv_capacity_tensor, indices = cycle_indices)
    cv_current = tf.gather(cv_current_tensor, indices = cycle_indices)
    cv_mask = tf.gather(cv_mask_tensor, indices = cycle_indices)
    cv_mask_2 = tf.tile(
        tf.reshape(
            1. / tf.cast(neighborhood[:, NEIGH_VALID_CYC], tf.float32),
            [batch_size2, 1]
        ),
        [1, cv_current.shape[1]],
    )

    cell_indices = neighborhood[:, NEIGH_BARCODE]

    with tf.GradientTape() as tape:
        train_results = degradation_model(
            (
                tf.expand_dims(cycle, axis = 1),
                tf.expand_dims(constant_current, axis = 1),
                tf.expand_dims(end_current_prev, axis = 1),
                tf.expand_dims(end_voltage_prev, axis = 1),
                tf.expand_dims(end_voltage, axis = 1),
                cell_indices,
                cc_voltage,
                cv_current,
                svit_grid,
                count_matrix,
                cc_capacity,
                cv_capacity,
                anode_v_curve,
                cathode_v_curve
            ),
            training = True
        )

        pred_cc_capacity = train_results["pred_cc_capacity"]
        pred_cv_capacity = train_results["pred_cv_capacity"]
        pred_cc_voltage = train_results["pred_cc_voltage"]
        pred_cv_voltage = train_results["pred_cv_voltage"]
        cv_voltage = tf.tile(
            tf.expand_dims(end_voltage, axis = 1),
            [1, cv_current.shape[1]],
        )

        cc_capacity_loss = get_loss(
            cc_capacity, pred_cc_capacity, cc_mask, cc_mask_2,
        )
        cv_capacity_loss = get_loss(
            cv_capacity, pred_cv_capacity, cv_mask, cv_mask_2,
        )
        cc_voltage_loss = get_loss(
            cc_voltage, pred_cc_voltage, cc_mask, cc_mask_2,
        )
        cv_voltage_loss = get_loss(
            cv_voltage, pred_cv_voltage, cv_mask, cv_mask_2,
        )

        loss = (
            tf.stop_gradient(
                fit_args[Key.Coeff.Q_CV] * cv_capacity_loss
                + fit_args[Key.Coeff.V_CC] * cc_voltage_loss
                + fit_args[Key.Coeff.Q_CC] * cc_capacity_loss
            )
            + fit_args[Key.Coeff.Q_CV] * cv_capacity_loss
            + fit_args[Key.Coeff.V_CC] * cc_voltage_loss
            + fit_args[Key.Coeff.Q_CC] * cc_capacity_loss
            * (
                fit_args[Key.Coeff.Q] * train_results[Key.Loss.Q]
                + fit_args[Key.Coeff.SCALE] * train_results[Key.Loss.SCALE]
                + fit_args[Key.Coeff.R] * train_results[Key.Loss.R]
                + fit_args[Key.Coeff.SHIFT] * train_results[Key.Loss.SHIFT]
                + fit_args[Key.Coeff.CELL] * train_results[Key.Loss.CELL]
                + fit_args[Key.Coeff.RECIP] * train_results[Key.Loss.RECIP]
                + fit_args[Key.Coeff.PROJ] * train_results[Key.Loss.PROJ]
                + fit_args[Key.Coeff.OOB] * train_results[Key.Loss.OOB]
            )
        )

    gradients = tape.gradient(
        loss,
        degradation_model.trainable_variables,
    )

    gradients_no_nans = [
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        for x in gradients
    ]

    gradients_norm_clipped, _ = tf.clip_by_global_norm(
        gradients_no_nans,
        fit_args[Key.GLB_NORM_CLIP],
    )

    optimizer.apply_gradients(
        zip(
            gradients_norm_clipped,
            degradation_model.trainable_variables,
        )
    )

    return tf.stack(
        [
            cc_capacity_loss,
            cv_capacity_loss,
            cc_voltage_loss,
            cv_voltage_loss,
            train_results[Key.Loss.Q],
            train_results[Key.Loss.SCALE],
            train_results[Key.Loss.R],
            train_results[Key.Loss.SHIFT],
            train_results[Key.Loss.CELL],
            train_results[Key.Loss.RECIP],

            train_results[Key.Loss.PROJ],
            train_results[Key.Loss.OOB],
        ],
    )


def get_loss(measured, predicted, mask, mask_2):
    return tf.reduce_mean(
        (1e-10 + mask * mask_2) * tf.square(measured - predicted)
    ) / (1e-10 + tf.reduce_mean(mask * mask_2))


class Command(BaseCommand):

    def add_arguments(self, parser):

        required_args = [
            "--path_to_dataset",
            "--dataset_version",
            "--path_to_plots",
            "--" + Key.PATH_V_CURVES,
        ]

        float_args = {
            '--global_norm_clip': 10.,

            '--learning_rate': 5e-4,
            '--min_latent': .05,

            '--coeff_d_features': .001,
            '--coeff_d2_features': .01,

            '--coeff_cell': 1.,
            '--coeff_cell_output': .1,
            '--coeff_cell_input': .1,
            '--coeff_cell_derivative': .1,
            '--coeff_cell_eq': 10.,

            '--coeff_electrolyte': 1.,
            '--coeff_electrolyte_output': .1,
            '--coeff_electrolyte_input': .1,
            '--coeff_electrolyte_derivative': .1,
            '--coeff_electrolyte_eq': 10.,

            '--coeff_cv_capacity': 1.,
            '--coeff_cv_voltage': 1.,
            '--coeff_cc_voltage': 1.,
            '--coeff_cc_capacity': 1.,

            '--coeff_q': 1.,
            '--coeff_q_small': .001,
            '--coeff_q_geq': 1.,
            '--coeff_q_leq': 1.,
            '--coeff_q_v_mono': 10.,
            '--coeff_q_d3_v': 1.,
            '--coeff_q_d3_shift': .01,
            '--coeff_q_d3_current': 10.,
            '--coeff_q_d3_cycle': 10.,
            '--coeff_q_d_current': 1.,
            '--coeff_q_d_cycle': 5.,

            '--coeff_scale': 5.,
            '--coeff_scale_geq': 5.,
            '--coeff_scale_leq': 5.,
            '--coeff_scale_eq': 5.,
            '--coeff_scale_mono': 1.,
            '--coeff_scale_d3_cycle': .02,

            '--coeff_r': 5.,
            '--coeff_r_geq': 100.,
            '--coeff_r_big': .0,
            '--coeff_r_d3_cycle': .02,

            '--coeff_shift': 1.,
            '--coeff_shift_geq': .5,
            '--coeff_shift_leq': .5,
            '--coeff_shift_small': .01,
            '--coeff_shift_d3_cycle': .02,
            '--coeff_shift_mono': .0,

            '--coeff_reciprocal': 10.,
            '--coeff_reciprocal_v': 10.,
            '--coeff_reciprocal_q': 10.,
            '--coeff_reciprocal_v_small': .001,
            '--coeff_reciprocal_v_geq': 1.,
            '--coeff_reciprocal_v_leq': .5,
            '--coeff_reciprocal_v_mono': 10.,
            '--coeff_reciprocal_d3_current': 10.,
            '--coeff_reciprocal_d_current_minus': 10.,
            '--coeff_reciprocal_d_current_plus': 2.,
            '--coeff_reciprocal_d3_cycle': 10.,
            '--coeff_reciprocal_d_cycle': 1.,
            '--coeff_anode_match': 5.,
            '--coeff_cathode_match': 1.,

            '--coeff_projection': 1.,
            '--coeff_projection_pos': 1.,
            '--coeff_projection_neg': 1.,
            '--coeff_projection_dry_cell': 1.,

            '--coeff_out_of_bounds': 10.,
            '--coeff_out_of_bounds_geq': 1.,
            '--coeff_out_of_bounds_leq': 1.,
        }

        vis = 10000
        int_args = {
            "--n_sample": 8 * 16,

            "--depth": 3,
            "--width": 50,
            "--batch_size": 4 * 16,

            "--print_loss_every": 500,
            "--visualize_fit_every": vis,
            "--visualize_vq_every": vis,

            "--stop_count": 1000004,
            "--barcode_show": 10,
        }

        for arg in required_args:
            parser.add_argument(arg, required = True)
        for arg in float_args:
            parser.add_argument(arg, type = float, default = float_args[arg])
        for arg in int_args:
            parser.add_argument(arg, type = int, default = int_args[arg])

        barcodes = [
            81602, 81603, 81604, 81605, 81606, 81607, 81608, 81609, 81610,
            81611, 81612, 81613, 81614, 81615, 81616, 81617, 81618, 81619,
            81620, 81621, 81622, 81623, 81624, 81625, 81626, 81627, 81712,
            81713, 82300, 82301, 82302, 82303, 82304, 82305, 82306, 82307,
            82308, 82309, 82310, 82311, 82406, 82407, 82410, 82411, 82769,
            82770, 82771, 82775, 82776, 82777, 82779, 82992, 82993, 83083,
            83092, 83101, 83106, 83107, 83220, 83221, 83222, 83223, 83224,
            83225, 83226, 83227, 83228, 83229, 83230, 83231, 83232, 83233,
            83234, 83235, 83236, 83237, 83239, 83240, 83241, 83242, 83243,
            83310, 83311, 83312, 83317, 83318, 83593, 83594, 83595, 83596,
            83741, 83742, 83743, 83744, 83745, 83746, 83747, 83748,
        ]
        # 57706, 57707, 57710, 57711, 57714, 57715,64260,64268,83010, 83011,
        # 83012, 83013, 83014, 83015, 83016

        parser.add_argument(
            "--wanted_barcodes", type = int, nargs = "+", default = barcodes,
        )

    def handle(self, *args, **options):
        ml_smoothing(options)
