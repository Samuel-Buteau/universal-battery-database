import time

import numpy
import tensorflow as tf
from django.core.management.base import BaseCommand

from neware_parser.DegradationModel import DegradationModel
from neware_parser.models import *
from neware_parser.plot import *

'''
Shortened Variable Names:
    vol -   voltage
    cap -   capacity
    dchg -  discharge
    neigh - neighbourhood
    der -   derivative
    pred -  predicted
    meas -  measured
    eval -  evaluation
    eq -    equillibrium
    res -   result
'''

# TODO(sam): For each barcode, needs a multigrid of (S, V, I, T) (current
#  needs to be adjusted)
# TODO(sam): Each cycle must have an index mapping to the nearest reference
#  cycle.
# TODO(sam): to evaluate a cycle, there must be the multigrid, the reference
#  cycle scaled by cycle number, the cell features, pasted together and ran
#  through a neural net.

NEIGHBORHOOD_MIN_CYC_INDEX = 0
NEIGHBORHOOD_MAX_CYC_INDEX = 1
NEIGHBORHOOD_RATE_INDEX = 2
NEIGHBORHOOD_BARCODE_INDEX = 3
NEIGHBORHOOD_ABSOLUTE_CYCLE_INDEX = 4
NEIGHBORHOOD_VALID_CYC_INDEX = 5
NEIGHBORHOOD_SIGN_GRID_INDEX = 6
NEIGHBORHOOD_VOLTAGE_GRID_INDEX = 7
NEIGHBORHOOD_CURRENT_GRID_INDEX = 8
NEIGHBORHOOD_TEMPERATURE_GRID_INDEX = 9
NEIGHBORHOOD_ABSOLUTE_REFERENCE_INDEX = 10
NEIGHBORHOOD_REFERENCE_INDEX = 11

NEIGHBORHOOD_TOTAL = 12


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


def initial_processing(my_data, my_names, barcodes, fit_args):
    """
    my_data has the following structure:
        my_data: a dictionary indexed by various data:
            - 'max_cap': a single number. the maximum capacity across the 
            dataset.
            - 'voltage_grid': 1D array of voltages
            - 'current_grid': 1D array of log currents
            - 'temperature_grid': 1D array of temperatures
            - 'sign_grid': 1D array of signs
            - 'cell_id_to_pos_id': a dictionary indexed by barcode yielding a 
            positive electrode id.
            - 'cell_id_to_neg_id': a dictionary indexed by barcode yielding a 
            positive electrode id.
            - 'cell_id_to_electrolyte_id': a dictionary indexed by barcode 
            yielding a positive electrode id.
            - 'cell_id_to_latent': a dictionary indexed by barcode yielding 
                         1 if the cell is latent, 
                         0 if made of known pos,neg,electrolyte
                
            - 'all_data': a dictionary indexed by barcode. 
               Each barcode yields:
                - 'all_reference_mats': structured array with dtype =
                    [
                        (
                            'cycle_number',
                            'f4'
                        ),
                        (
                            'count_matrix',
                            'f4',
                            (
                                len(sign_grid),
                                len(voltage_grid_degradation),
                                len(current_grid),
                                len(temperature_grid)
                            )
                        ),
                    ]

                - 'cyc_grp_dict': we know how this works.
                    basically groups of steps indexed by group averages of
                    (
                        end_current_prev,
                        constant_current,
                        end_current,
                        end_voltage_prev,
                        end_voltage,
                        sign
                    )

                    each group is a dictinary indexed by various quantities:
                        - 'main_data':  a numpy structured array with dtype:
                            [
                                ('cycle_number', 'f4'),
                                ('cc_voltage_vector', 'f4', len(voltage_grid)),
                                ('cc_capacity_vector', 'f4', len(voltage_grid)),
                                ('cc_mask_vector', 'f4', len(voltage_grid)),
                                ('cv_current_vector', 'f4', fit_args[
                                'current_max_n']),
                                ('cv_capacity_vector', 'f4', fit_args[
                                'current_max_n']),
                                ('cv_mask_vector', 'f4', fit_args[
                                'current_max_n']),
                                ('constant_current', 'f4'),
                                ('end_current_prev', 'f4'),
                                ('end_current', 'f4'),
                                ('end_voltage_prev', 'f4'),
                                ('end_voltage', 'f4'),
                                ('last_cc_voltage', 'f4'),
                                ('last_cc_capacity', 'f4'),
                                ('last_cv_capacity', 'f4'),
                                ('temperature', 'f4'),
                            ]

                        -     'avg_constant_current'
                        -     'avg_end_current_prev'
                        -     'avg_end_current'
                        -     'avg_end_voltage_prev'
                        -     'avg_end_voltage'
                        -     'avg_last_cc_voltage'





    """

    compiled_data = {}
    number_of_compiled_cycles = 0
    number_of_reference_cycles = 0

    max_cap = my_data['max_cap']

    numpy_acc(
        compiled_data,
        'voltage_grid',
        numpy.array([my_data['voltage_grid']])
    )
    numpy_acc(
        compiled_data,
        'temperature_grid',
        numpy.array([my_data['temperature_grid']])
    )
    numpy_acc(
        compiled_data,
        'sign_grid',
        numpy.array([my_data['sign_grid']])
    )

    my_data['current_grid'] = my_data['current_grid'] - numpy.log(max_cap)

    # the current grid is adjusted by the max capacity of the barcode. It is
    # in log space, so I/Q becomes log(I) - log(Q)
    numpy_acc(
        compiled_data,
        'current_grid',
        numpy.array([my_data['current_grid']])
    )

    cell_id_list = numpy.array(barcodes)
    cell_id_to_pos_id = {}
    cell_id_to_neg_id = {}
    cell_id_to_electrolyte_id = {}
    cell_id_to_latent = {}

    electrolyte_id_to_latent = {}
    electrolyte_id_to_solvent_id_weight = {}
    electrolyte_id_to_salt_id_weight = {}
    electrolyte_id_to_additive_id_weight = {}

    for cell_id in cell_id_list:
        if cell_id in my_data['cell_id_to_pos_id'].keys():
            cell_id_to_pos_id[cell_id] = my_data['cell_id_to_pos_id'][cell_id]
        if cell_id in my_data['cell_id_to_neg_id'].keys():
            cell_id_to_neg_id[cell_id] = my_data['cell_id_to_neg_id'][cell_id]
        if cell_id in my_data['cell_id_to_electrolyte_id'].keys():
            cell_id_to_electrolyte_id[cell_id] = \
                my_data['cell_id_to_electrolyte_id'][cell_id]
        if cell_id in my_data['cell_id_to_latent'].keys():
            cell_id_to_latent[cell_id] = my_data['cell_id_to_latent'][cell_id]

        if cell_id_to_latent[cell_id] < 0.5:
            electrolyte_id = cell_id_to_electrolyte_id[cell_id]
            if electrolyte_id in my_data[
                'electrolyte_id_to_solvent_id_weight'].keys():
                electrolyte_id_to_solvent_id_weight[electrolyte_id] = \
                    my_data['electrolyte_id_to_solvent_id_weight'][
                        electrolyte_id]
            if electrolyte_id in my_data[
                'electrolyte_id_to_salt_id_weight'].keys():
                electrolyte_id_to_salt_id_weight[electrolyte_id] = \
                    my_data['electrolyte_id_to_salt_id_weight'][
                        electrolyte_id]
            if electrolyte_id in my_data[
                'electrolyte_id_to_additive_id_weight'].keys():
                electrolyte_id_to_additive_id_weight[electrolyte_id] = \
                    my_data['electrolyte_id_to_additive_id_weight'][
                        electrolyte_id]

            if electrolyte_id in my_data['electrolyte_id_to_latent'].keys():
                electrolyte_id_to_latent[electrolyte_id] = \
                    my_data['electrolyte_id_to_latent'][electrolyte_id]

    mess = [
        [[s[0] for s in siw] for siw in
         electrolyte_id_to_solvent_id_weight.values()],
        [[s[0] for s in siw] for siw in
         electrolyte_id_to_salt_id_weight.values()],
        [[s[0] for s in siw] for siw in
         electrolyte_id_to_additive_id_weight.values()],
    ]

    molecule_id_list = numpy.array(
        sorted(
            list(
                set(
                    list(three_level_flatten(mess))
                )
            )
        )
    )

    pos_id_list = numpy.array(sorted(list(set(cell_id_to_pos_id.values()))))
    neg_id_list = numpy.array(sorted(list(set(cell_id_to_neg_id.values()))))
    electrolyte_id_list = numpy.array(
        sorted(list(set(cell_id_to_electrolyte_id.values()))))

    for barcode_count, barcode in enumerate(barcodes):

        cyc_grp_dict = my_data['all_data'][barcode]['cyc_grp_dict']

        for k_count, k in enumerate(cyc_grp_dict.keys()):
            main_data = cyc_grp_dict[k]['main_data']

            # normalize capacity_vector with max_cap
            main_data['cc_capacity_vector'] = (
                1. / max_cap * main_data[
                'cc_capacity_vector']
            )

            main_data['cv_capacity_vector'] = (
                1. / max_cap * main_data[
                'cv_capacity_vector']
            )

            main_data[
                'cv_current_vector'] = (
                1. / max_cap * main_data['cv_current_vector']
            )

            main_data[
                'last_cc_capacity'] = (
                1. / max_cap * main_data['last_cc_capacity']
            )

            main_data[
                'last_cv_capacity'] = (
                1. / max_cap * main_data['last_cv_capacity']
            )

            main_data[
                'constant_current'] = (
                1. / max_cap * main_data['constant_current']
            )
            main_data[
                'end_current_prev'] = (
                1. / max_cap * main_data['end_current_prev']
            )

            cyc_grp_dict[k][
                'avg_constant_current'] = (
                1. / max_cap * cyc_grp_dict[k]['avg_constant_current']
            )

            cyc_grp_dict[k][
                'avg_end_current'] = (
                1. / max_cap * cyc_grp_dict[k]['avg_end_current']
            )

            cyc_grp_dict[k][
                'avg_end_current_prev'] = (
                1. / max_cap * cyc_grp_dict[k]['avg_end_current_prev']
            )

            # range of cycles which exist for this cycle group
            min_cyc = min(main_data['cycle_number'])
            max_cyc = max(main_data['cycle_number'])

            '''
            - now create neighborhoods, which contains the cycles,
              grouped by proximity
            - want to sample neighborhoods equally
            - neighborhoods have a central cycle and a delta on each side
            - to a first approximation, we want a delta_cyc = 300, but we have
              to vary this near the beginning of data and near the end.
            '''

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
                    below_cyc <= main_data['cycle_number'],
                    main_data['cycle_number'] <= above_cyc
                )

                # the indecies for the cyc_grp_dict[k] array which correspond
                # to a True mask
                all_valid_indecies = numpy.arange(len(mask))[mask]

                # if there are less than 2 valid cycles, skip that neighborhood
                if len(all_valid_indecies) < 2:
                    continue

                '''
                at this point, we know that this neighborhood
                will be added to the dataset.
                '''

                min_cyc_index = all_valid_indecies[0]
                max_cyc_index = all_valid_indecies[-1]

                valid_cycles += 1

                '''
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
                
 
                '''
                # TODO(sam): here, figure out where the reference cycle is,
                #  and note the index

                neighborhood_data_i = numpy.zeros(NEIGHBORHOOD_TOTAL,
                                                  dtype = numpy.int32)

                neighborhood_data_i[NEIGHBORHOOD_MIN_CYC_INDEX] = min_cyc_index
                neighborhood_data_i[NEIGHBORHOOD_MAX_CYC_INDEX] = max_cyc_index
                neighborhood_data_i[NEIGHBORHOOD_RATE_INDEX] = k_count
                neighborhood_data_i[NEIGHBORHOOD_BARCODE_INDEX] = barcode_count
                neighborhood_data_i[
                    NEIGHBORHOOD_ABSOLUTE_CYCLE_INDEX] = \
                    number_of_compiled_cycles
                neighborhood_data_i[
                    NEIGHBORHOOD_VALID_CYC_INDEX] = 0  # a weight based on
                # prevalance. Set later
                neighborhood_data_i[NEIGHBORHOOD_SIGN_GRID_INDEX] = 0
                neighborhood_data_i[NEIGHBORHOOD_VOLTAGE_GRID_INDEX] = 0
                neighborhood_data_i[NEIGHBORHOOD_CURRENT_GRID_INDEX] = 0
                neighborhood_data_i[NEIGHBORHOOD_TEMPERATURE_GRID_INDEX] = 0

                center_cycle = float(cyc)
                reference_cycles = \
                    my_data['all_data'][barcode]['all_reference_mats'][
                        'cycle_number']

                index_of_closest_reference = numpy.argmin(
                    abs(center_cycle - reference_cycles)
                )

                neighborhood_data_i[
                    NEIGHBORHOOD_ABSOLUTE_REFERENCE_INDEX] = \
                    number_of_reference_cycles
                neighborhood_data_i[
                    NEIGHBORHOOD_REFERENCE_INDEX] = index_of_closest_reference

                neighborhood_data.append(
                    neighborhood_data_i
                )

            if valid_cycles != 0:
                neighborhood_data = numpy.array(
                    neighborhood_data, dtype = numpy.int32
                )

                # the empty slot becomes the count of added neighborhoods, which
                # are used to counterbalance the bias toward longer cycle life
                neighborhood_data[:, NEIGHBORHOOD_VALID_CYC_INDEX] = \
                    valid_cycles

                numpy_acc(compiled_data, 'neighborhood_data', neighborhood_data)

            number_of_compiled_cycles += len(
                main_data['cycle_number']
            )

            number_of_reference_cycles += len(
                my_data['all_data'][barcode]['all_reference_mats']['cycle_number'])
            numpy_acc(compiled_data, 'reference_cycle',
                      my_data['all_data'][barcode]['all_reference_mats'][
                          'cycle_number'])
            numpy_acc(compiled_data, 'count_matrix',
                      my_data['all_data'][barcode]['all_reference_mats'][
                          'count_matrix'])

            numpy_acc(compiled_data, 'cycle',
                      main_data['cycle_number'])
            numpy_acc(compiled_data, 'cc_voltage_vector',
                      main_data['cc_voltage_vector'])
            numpy_acc(compiled_data, 'cc_capacity_vector',
                      main_data['cc_capacity_vector'])
            numpy_acc(compiled_data, 'cc_mask_vector',
                      main_data['cc_mask_vector'])
            numpy_acc(compiled_data, 'cv_current_vector',
                      main_data['cv_current_vector'])
            numpy_acc(compiled_data, 'cv_capacity_vector',
                      main_data['cv_capacity_vector'])
            numpy_acc(compiled_data, 'cv_mask_vector',
                      main_data['cv_mask_vector'])
            numpy_acc(compiled_data, 'constant_current',
                      main_data['constant_current'])
            numpy_acc(compiled_data, 'end_current_prev',
                      main_data['end_current_prev'])
            numpy_acc(compiled_data, 'end_voltage_prev',
                      main_data['end_voltage_prev'])
            numpy_acc(compiled_data, 'end_voltage',
                      main_data['end_voltage'])

    neighborhood_data = tf.constant(compiled_data['neighborhood_data'])

    compiled_tensors = {}
    # cycles go from 0 to 6000, but nn prefers normally distributed variables
    # so cycle numbers is normalized with mean and variance
    cycle_tensor = tf.constant(compiled_data['cycle'])
    cycle_m, cycle_v = tf.nn.moments(cycle_tensor, axes = [0])
    cycle_m = 0.  # we shall leave the cycle 0 at 0
    cycle_v = cycle_v.numpy()
    cycle_tensor = (cycle_tensor - cycle_m) / tf.sqrt(cycle_v)
    compiled_tensors['cycle'] = cycle_tensor

    labels = [
        'cc_voltage_vector',
        'cc_capacity_vector',
        'cc_mask_vector',
        'cv_capacity_vector',
        'cv_current_vector',
        'cv_mask_vector',
        'constant_current',
        'end_current_prev',
        'end_voltage_prev',
        'end_voltage',

        'count_matrix',

        'sign_grid',
        'voltage_grid',
        'current_grid',
        'temperature_grid',
    ]
    for label in labels:
        compiled_tensors[label] = tf.constant(compiled_data[label])

    batch_size = fit_args['batch_size']
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        train_ds_ = tf.data.Dataset.from_tensor_slices(
            neighborhood_data
        ).repeat(2).shuffle(100000).batch(batch_size)

        train_ds = mirrored_strategy.experimental_distribute_dataset(
            train_ds_)

        pos_to_pos_name = {}
        neg_to_neg_name = {}
        electrolyte_to_electrolyte_name = {}
        if my_names is not None:
            pos_to_pos_name = my_names['pos_to_pos_name']
            neg_to_neg_name = my_names['neg_to_neg_name']
            electrolyte_to_electrolyte_name = my_names[
                'electrolyte_to_electrolyte_name']
            molecule_to_molecule_name = my_names['molecule_to_molecule_name']

        degradation_model = DegradationModel(
            width = fit_args['width'],
            depth = fit_args['depth'],
            cell_dict = id_dict_from_id_list(cell_id_list),
            pos_dict = id_dict_from_id_list(pos_id_list),
            neg_dict = id_dict_from_id_list(neg_id_list),
            electrolyte_dict = id_dict_from_id_list(electrolyte_id_list),
            molecule_dict = id_dict_from_id_list(molecule_id_list),

            cell_to_pos = cell_id_to_pos_id,
            cell_to_neg = cell_id_to_neg_id,
            cell_to_electrolyte = cell_id_to_electrolyte_id,
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
            )

        )

        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.002)

    return {
        "mirrored_strategy": mirrored_strategy,
        "degradation_model": degradation_model,
        "compiled_tensors": compiled_tensors,

        "train_ds": train_ds,
        "cycle_m": cycle_m,
        "cycle_v": cycle_v,

        "optimizer": optimizer,
        "my_data": my_data
    }


def train_and_evaluate(init_returns, barcodes, fit_args):
    mirrored_strategy = init_returns["mirrored_strategy"]

    EPOCHS = 100000
    count = 0

    template = 'Epoch {}, Count {}'
    end = time.time()
    now_ker = None
    with mirrored_strategy.scope():
        for epoch in range(EPOCHS):
            for neighborhood in init_returns["train_ds"]:
                count += 1

                train_step_params = {
                    "neighborhood": neighborhood,
                    "compiled_tensors": init_returns['compiled_tensors'],
                    "optimizer": init_returns['optimizer'],
                    "degradation_model": init_returns['degradation_model'],
                }

                dist_train_step(mirrored_strategy, train_step_params, fit_args)

                if count != 0:
                    if (count % fit_args['print_loss_every']) == 0:
                        print(template.format(epoch + 1, count, ))

                    plot_params = {
                        "barcodes": barcodes,
                        "count": count,
                        "fit_args": fit_args,
                    }

                    if (count % fit_args['visualize_fit_every']) == 0:
                        start = time.time()
                        print("time to simulate: ", start - end)
                        plot_things_vs_cycle_number(plot_params, init_returns)
                        plot_vq(plot_params, init_returns)
                        plot_v_curves(plot_params, init_returns)
                        end = time.time()
                        print("time to plot: ", end - start)
                        ker = init_returns[
                            "degradation_model"
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
                                'average difference between cells: ',
                                numpy.average(delta_cell_ker)
                            )
                        if prev_ker is not None:
                            delta_time_ker = numpy.abs(now_ker - prev_ker)
                            print(
                                'average difference between prev and now: ',
                                numpy.average(delta_time_ker)
                            )
                        print()

                if count >= fit_args['stop_count']:
                    return


def train_step(params, fit_args):
    neighborhood = params["neighborhood"]

    sign_grid_tensor = params["compiled_tensors"]['sign_grid']
    voltage_grid_tensor = params["compiled_tensors"]['voltage_grid']
    current_grid_tensor = params["compiled_tensors"]['current_grid']
    temperature_grid_tensor = params["compiled_tensors"]['temperature_grid']

    count_matrix_tensor = params["compiled_tensors"]['count_matrix']

    cycle_tensor = params["compiled_tensors"]['cycle']
    constant_current_tensor = params["compiled_tensors"]["constant_current"]
    end_current_prev_tensor = params["compiled_tensors"]["end_current_prev"]
    end_voltage_prev_tensor = params["compiled_tensors"]["end_voltage_prev"]
    end_voltage_tensor = params["compiled_tensors"]["end_voltage"]

    degradation_model = params["degradation_model"]
    optimizer = params["optimizer"]

    cc_voltage_tensor = params["compiled_tensors"]["cc_voltage_vector"]
    cc_capacity_tensor = params["compiled_tensors"]["cc_capacity_vector"]
    cc_mask_tensor = params["compiled_tensors"]["cc_mask_vector"]
    cv_capacity_tensor = params["compiled_tensors"]["cv_capacity_vector"]
    cv_current_tensor = params["compiled_tensors"]["cv_current_vector"]
    cv_mask_tensor = params["compiled_tensors"]["cv_mask_vector"]

    # need to split the range
    batch_size2 = neighborhood.shape[0]

    '''
    if you have the minimum cycle and maximum cycle for a neighborhood,
    you can sample cycle from this neighborhood by sampling real numbers
    x from [0,1] and computing min_cyc*(1.-x) + max_cyc*x,
    but here this computation is done in index space,
    then cycle numbers and vq curves are gathered
    '''

    cycle_indices_lerp = tf.random.uniform(
        [batch_size2], minval = 0., maxval = 1., dtype = tf.float32)
    cycle_indices = tf.cast(
        (1. - cycle_indices_lerp) * tf.cast(
            neighborhood[:, NEIGHBORHOOD_MIN_CYC_INDEX]
            + neighborhood[:, NEIGHBORHOOD_ABSOLUTE_CYCLE_INDEX],
            tf.float32
        ) + cycle_indices_lerp * tf.cast(
            neighborhood[:, NEIGHBORHOOD_MAX_CYC_INDEX]
            + neighborhood[:, NEIGHBORHOOD_ABSOLUTE_CYCLE_INDEX],
            tf.float32
        ),
        tf.int32
    )

    sign_grid = tf.gather(
        sign_grid_tensor,
        indices = neighborhood[:, NEIGHBORHOOD_SIGN_GRID_INDEX], axis = 0
    )
    sign_grid_dim = sign_grid.shape[1]

    voltage_grid = tf.gather(
        voltage_grid_tensor,
        indices = neighborhood[:, NEIGHBORHOOD_VOLTAGE_GRID_INDEX], axis = 0
    )
    voltage_grid_dim = voltage_grid.shape[1]
    current_grid = tf.gather(
        current_grid_tensor,
        indices = neighborhood[:, NEIGHBORHOOD_CURRENT_GRID_INDEX], axis = 0
    )
    current_grid_dim = current_grid.shape[1]

    temperature_grid = tf.gather(
        temperature_grid_tensor,
        indices = neighborhood[:, NEIGHBORHOOD_TEMPERATURE_GRID_INDEX], axis = 0
    )
    temperature_grid_dim = temperature_grid.shape[1]

    svit_grid = tf.concat(
        (
            tf.tile(
                tf.reshape(sign_grid, [batch_size2, sign_grid_dim, 1, 1, 1, 1]),
                [1, 1, voltage_grid_dim, current_grid_dim, temperature_grid_dim,
                 1],
            ),
            tf.tile(
                tf.reshape(voltage_grid,
                           [batch_size2, 1, voltage_grid_dim, 1, 1, 1]),
                [1, sign_grid_dim, 1, current_grid_dim, temperature_grid_dim,
                 1],
            ),
            tf.tile(
                tf.reshape(current_grid,
                           [batch_size2, 1, 1, current_grid_dim, 1, 1]),
                [1, sign_grid_dim, voltage_grid_dim, 1, temperature_grid_dim,
                 1],
            ),
            tf.tile(
                tf.reshape(temperature_grid,
                           [batch_size2, 1, 1, 1, temperature_grid_dim, 1]),
                [1, sign_grid_dim, voltage_grid_dim, current_grid_dim, 1, 1],
            ),
        ),
        axis = -1
    )

    count_matrix = tf.reshape(
        tf.gather(
            count_matrix_tensor,
            (
                neighborhood[:, NEIGHBORHOOD_ABSOLUTE_REFERENCE_INDEX]
                + neighborhood[:, NEIGHBORHOOD_REFERENCE_INDEX]
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
            1. / tf.cast(
                neighborhood[:, NEIGHBORHOOD_VALID_CYC_INDEX], tf.float32
            ),
            [batch_size2, 1]
        ),
        [1, cc_voltage.shape[1]]
    )

    cv_capacity = tf.gather(cv_capacity_tensor, indices = cycle_indices)
    cv_current = tf.gather(cv_current_tensor, indices = cycle_indices)
    cv_mask = tf.gather(cv_mask_tensor, indices = cycle_indices)
    cv_mask_2 = tf.tile(
        tf.reshape(
            1. / tf.cast(
                neighborhood[:, NEIGHBORHOOD_VALID_CYC_INDEX], tf.float32
            ),
            [batch_size2, 1]
        ),
        [1, cv_current.shape[1]]
    )

    cell_indices = neighborhood[:, NEIGHBORHOOD_BARCODE_INDEX]

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
                cc_capacity
            ),
            training = True
        )

        pred_cc_capacity = train_results["pred_cc_capacity"]
        pred_cv_capacity = train_results["pred_cv_capacity"]
        pred_cc_voltage = train_results["pred_cc_voltage"]

        cc_capacity_loss = (
            tf.reduce_mean(
                cc_mask_2 * cc_mask * tf.square(cc_capacity - pred_cc_capacity)
            ) / (1e-10 + tf.reduce_mean(cc_mask_2 * cc_mask))
        )
        cv_capacity_loss = (
            tf.reduce_mean(
                cv_mask_2 * cv_mask * tf.square(cv_capacity - pred_cv_capacity)
            ) / (1e-10 + tf.reduce_mean(cv_mask_2 * cv_mask))
        )
        cc_voltage_loss = (
            tf.reduce_mean(
                cc_mask_2 * cc_mask * tf.square(cc_voltage - pred_cc_voltage)
            ) / (1e-10 + tf.reduce_mean(cc_mask_2 * cc_mask))
        )

        loss = (
            0.05 * cv_capacity_loss + 1. * cc_voltage_loss + cc_capacity_loss
            + train_results["Q_loss"]
            + train_results["Q_scale_loss"]
            + train_results["r_loss"]
            + train_results["shift_loss"]
            + fit_args['z_cell_coeff'] * train_results["z_cell_loss"]
            + .1 * train_results["reciprocal_loss"]
            + .1 * train_results["projection_loss"]
            + .1 * train_results["out_of_bounds_loss"]
        )

    gradients = tape.gradient(loss, degradation_model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, degradation_model.trainable_variables)
    )


@tf.function
def dist_train_step(mirrored_strategy, train_step_params, fit_args):
    mirrored_strategy.experimental_run_v2(
        train_step, args = (train_step_params, fit_args)
    )


def ml_smoothing(fit_args):
    print(
        "Num GPUs Available: ",
        len(tf.config.experimental.list_physical_devices('GPU'))
    )

    if not os.path.exists(fit_args['path_to_plots']):
        os.mkdir(fit_args['path_to_plots'])

    dataset_path = os.path.join(
        fit_args['path_to_dataset'],
        'dataset_ver_{}.file'.format(fit_args['dataset_version'])
    )

    dataset_names_path = os.path.join(
        fit_args['path_to_dataset'],
        'dataset_ver_{}_names.file'.format(fit_args['dataset_version'])
    )

    if not os.path.exists(dataset_path):
        print("Path \"" + dataset_path + "\" does not exist.")
        return

    with open(dataset_path, 'rb') as f:
        my_data = pickle.load(f)

    my_names = None
    if os.path.exists(dataset_names_path):
        with open(dataset_names_path, 'rb') as f:
            my_names = pickle.load(f)

    barcodes = list(my_data['all_data'].keys())

    if len(fit_args['wanted_barcodes']) != 0:
        barcodes = list(
            set(barcodes).intersection(set(fit_args['wanted_barcodes'])))

    if len(barcodes) == 0:
        print('no barcodes')
        return

    train_and_evaluate(
        initial_processing(my_data, my_names, barcodes, fit_args), barcodes,
        fit_args)


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path_to_dataset', required = True)
        parser.add_argument('--dataset_version', required = True)
        parser.add_argument('--path_to_plots', required = True)
        parser.add_argument('--z_cell_coeff', type = float, default = 0.00001)
        parser.add_argument('--mono_coeff', type = float, default = .005)
        parser.add_argument('--smooth_coeff', type = float, default = .05)
        parser.add_argument('--const_f_coeff', type = float, default = .0)
        parser.add_argument('--smooth_f_coeff', type = float, default = .01)
        parser.add_argument('--depth', type = int, default = 5)
        parser.add_argument('--width', type = int, default = 48)
        parser.add_argument('--batch_size', type = int, default = 4 * 16)

        vis = 1000
        parser.add_argument('--print_loss_every', type = int, default = vis)
        parser.add_argument('--visualize_fit_every', type = int, default = vis)
        parser.add_argument('--visualize_vq_every', type = int, default = vis)

        parser.add_argument('--stop_count', type = int, default = 60000)
        parser.add_argument(
            '--wanted_barcodes', type = int, nargs = '+',
            default = [83220, 83083]
        )

    def handle(self, *args, **options):
        ml_smoothing(options)
