import time

import matplotlib.pyplot as plt

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

# Dictionary key names
Q_CC = 'cc_capacity_vector'
Q_CV = 'cv_capacity_vector'
Q_CC_LAST = 'last_cc_capacity'
Q_CV_LAST = 'last_cv_capacity'
Q_END_AVG = 'avg_end_current'

I_CV = 'cv_current_vector'
I_CC = 'constant_current'
I_PREV = 'end_current_prev'
I_CC_AVG = 'avg_constant_current'
I_PREV_END_AVG = 'avg_end_current_prev'

V_CC = 'cc_voltage_vector'
V_END = 'end_voltage'
V_END_AVG = 'avg_end_voltage'
V_PREV_END = 'end_voltage_prev'
V_PREV_END_AVG = 'avg_end_voltage_prev'

MASK_CC = 'cc_mask_vector'
MASK_CV = 'cv_mask_vector'

N = 'cycle_number'

# my_data key names
CELL_TO_POS = 'cell_id_to_pos_id'
CELL_TO_NEG = 'cell_id_to_neg_id'
CELL_TO_ELE = 'cell_id_to_electrolyte_id'
CELL_TO_LAT = 'cell_id_to_latent'

ELE_TO_SOL = 'electrolyte_id_to_solvent_id_weight'
ELE_TO_SALT = 'electrolyte_id_to_salt_id_weight'
ELE_TO_ADD = 'electrolyte_id_to_additive_id_weight'


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


def initial_processing(my_data, my_names, barcodes, fit_args, strategy):
    """
    my_data has the following structure:
        my_data: a dictionary indexed by various data:
            - 'max_cap': a single number. the maximum capacity across the
            dataset.
            - 'voltage_grid': 1D array of voltages
            - 'current_grid': 1D array of log currents
            - 'temperature_grid': 1D array of temperatures
            - 'sign_grid': 1D array of signs
            - CELL_TO_POS: a dictionary indexed by barcode yielding a
            positive electrode id.
            - CELL_TO_NEG: a dictionary indexed by barcode yielding a
            positive electrode id.
            - CELL_TO_ELE: a dictionary indexed by barcode
            yielding a positive electrode id.
            - CELL_TO_LAT: a dictionary indexed by barcode yielding
                         1 if the cell is latent,
                         0 if made of known pos,neg,electrolyte

            - 'all_data': a dictionary indexed by barcode.
               Each barcode yields:
                - 'all_reference_mats': structured array with dtype =
                    [
                        (
                            N,
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
                                (N, 'f4'),
                                (V_CC, 'f4', len(voltage_grid)),
                                (Q_CC, 'f4', len(voltage_grid)),
                                (MASK_CC, 'f4', len(voltage_grid)),
                                (I_CV, 'f4', fit_args[
                                'current_max_n']),
                                (Q_CV, 'f4', fit_args[
                                'current_max_n']),
                                (MASK_CV, 'f4', fit_args[
                                'current_max_n']),
                                (I_CC, 'f4'),
                                (I_PREV, 'f4'),
                                ('end_current', 'f4'),
                                (V_PREV_END, 'f4'),
                                (V_END, 'f4'),
                                ('last_cc_voltage', 'f4'),
                                (Q_CC_LAST, 'f4'),
                                (Q_CV_LAST, 'f4'),
                                ('temperature', 'f4'),
                            ]

                        -     I_CC_AVG
                        -     I_PREV_END_AVG
                        -     Q_END_AVG
                        -     V_PREV_END_AVG
                        -     V_END_AVG
                        -     'avg_last_cc_voltage'





    """

    compiled_data = {}
    number_of_compiled_cycles = 0
    number_of_reference_cycles = 0

    my_data['max_cap'] = 250
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
    # in log space, so I/q becomes log(I) - log(q)
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
        if cell_id in my_data[CELL_TO_POS].keys():
            cell_id_to_pos_id[cell_id] = my_data[CELL_TO_POS][cell_id]
        if cell_id in my_data[CELL_TO_NEG].keys():
            cell_id_to_neg_id[cell_id] = my_data[CELL_TO_NEG][cell_id]
        if cell_id in my_data[CELL_TO_ELE].keys():
            cell_id_to_electrolyte_id[cell_id] = my_data[CELL_TO_ELE][cell_id]
        if cell_id in my_data[CELL_TO_LAT].keys():
            cell_id_to_latent[cell_id] = my_data[CELL_TO_LAT][cell_id]

        if cell_id_to_latent[cell_id] < 0.5:
            electrolyte_id = cell_id_to_electrolyte_id[cell_id]
            if electrolyte_id in my_data[ELE_TO_SOL].keys():
                electrolyte_id_to_solvent_id_weight[electrolyte_id]\
                    = my_data[ELE_TO_SOL][electrolyte_id]
            if electrolyte_id in my_data[ELE_TO_SALT].keys():
                electrolyte_id_to_salt_id_weight[electrolyte_id]\
                    = my_data[ELE_TO_SALT][electrolyte_id]
            if electrolyte_id in my_data[ELE_TO_ADD].keys():
                electrolyte_id_to_additive_id_weight[electrolyte_id]\
                    = my_data[ELE_TO_ADD][electrolyte_id]

            if electrolyte_id in my_data['electrolyte_id_to_latent'].keys():
                electrolyte_id_to_latent[electrolyte_id]\
                    = my_data['electrolyte_id_to_latent'][electrolyte_id]

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

    pos_id_list = numpy.array(sorted(list(set(cell_id_to_pos_id.values()))))
    neg_id_list = numpy.array(sorted(list(set(cell_id_to_neg_id.values()))))
    electrolyte_id_list = numpy.array(
        sorted(list(set(cell_id_to_electrolyte_id.values())))
    )

    for barcode_count, barcode in enumerate(barcodes):

        all_data = my_data['all_data'][barcode]
        cyc_grp_dict = all_data['cyc_grp_dict']

        for k_count, k in enumerate(cyc_grp_dict.keys()):

            my_pass = any([
                abs(cyc_grp_dict[k][I_PREV_END_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][I_CC_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][Q_END_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][V_PREV_END_AVG]) < 1e-5,
                abs(cyc_grp_dict[k][V_END_AVG]) < 1e-5,
            ])
            if my_pass:
                continue
            main_data = cyc_grp_dict[k]['main_data']

            # normalize capacity_vector with max_cap
            main_data[Q_CC] = 1. / max_cap * main_data[Q_CC]
            main_data[Q_CV] = 1. / max_cap * main_data[Q_CV]
            main_data[I_CV] = 1. / max_cap * main_data[I_CV]
            main_data[Q_CC_LAST] = 1. / max_cap * main_data[Q_CC_LAST]
            main_data[Q_CV_LAST] = 1. / max_cap * main_data[Q_CV_LAST]
            main_data[I_CC] = 1. / max_cap * main_data[I_CC]
            main_data[I_PREV] = 1. / max_cap * main_data[I_PREV]

            cyc_grp_dict[k][I_CC_AVG] = 1. / max_cap * cyc_grp_dict[k][I_CC_AVG]
            cyc_grp_dict[k][Q_END_AVG]\
                = 1. / max_cap * cyc_grp_dict[k][Q_END_AVG]
            cyc_grp_dict[k][I_PREV_END_AVG]\
                = 1. / max_cap * cyc_grp_dict[k][I_PREV_END_AVG]

            # range of cycles which exist for this cycle group
            min_cyc = min(main_data[N])
            max_cyc = max(main_data[N])

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
                    below_cyc <= main_data[N],
                    main_data[N] <= above_cyc
                )

                # the indices for the cyc_grp_dict[k] array which correspond
                # to a True mask
                all_valid_indices = numpy.arange(len(mask))[mask]

                # if there are less than 2 valid cycles, skip that neighborhood
                if len(all_valid_indices) < 2:
                    continue

                '''
                at this point, we know that this neighborhood
                will be added to the dataset.
                '''

                min_cyc_index = all_valid_indices[0]
                max_cyc_index = all_valid_indices[-1]

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
                reference_cycles\
                    = all_data['all_reference_mats'][N]

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
                    neighborhood_data, dtype = numpy.int32
                )

                # the empty slot becomes the count of added neighborhoods, which
                # are used to counterbalance the bias toward longer cycle life
                neighborhood_data[:, NEIGH_VALID_CYC] = valid_cycles

                numpy_acc(compiled_data, 'neighborhood_data', neighborhood_data)

            number_of_compiled_cycles += len(main_data[N])
            number_of_reference_cycles += len(
                all_data['all_reference_mats'][N]
            )

            dict_to_acc = {
                'reference_cycle': all_data['all_reference_mats'][N],
                'count_matrix': all_data['all_reference_mats']['count_matrix'],
                'cycle': main_data[N],
                V_CC: main_data[V_CC],
                Q_CC: main_data[Q_CC],
                MASK_CC: main_data[MASK_CC],
                I_CV: main_data[I_CV],
                Q_CV: main_data[Q_CV],
                MASK_CV: main_data[MASK_CV],
                I_CC: main_data[I_CC],
                I_PREV: main_data[I_PREV],
                V_PREV_END: main_data[V_PREV_END],
                V_END: main_data[V_END],
            }

            for key in dict_to_acc:
                numpy_acc(compiled_data, key, dict_to_acc[key])

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
        V_CC,
        Q_CC,
        MASK_CC,
        Q_CV,
        I_CV,
        MASK_CV,
        I_CC,
        I_PREV,
        V_PREV_END,
        V_END,

        'count_matrix',

        'sign_grid',
        'voltage_grid',
        'current_grid',
        'temperature_grid',
    ]
    for label in labels:
        compiled_tensors[label] = tf.constant(compiled_data[label])

    batch_size = fit_args['batch_size']

    with strategy.scope():
        train_ds_ = tf.data.Dataset.from_tensor_slices(
            neighborhood_data
        ).repeat(2).shuffle(100000).batch(batch_size)

        train_ds = strategy.experimental_distribute_dataset(train_ds_)

        pos_to_pos_name = {}
        neg_to_neg_name = {}
        electrolyte_to_electrolyte_name = {}
        if my_names is not None:
            pos_to_pos_name = my_names['pos_to_pos_name']
            neg_to_neg_name = my_names['neg_to_neg_name']
            electrolyte_to_electrolyte_name\
                = my_names['electrolyte_to_electrolyte_name']
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
            ),
            n_sample = fit_args['n_sample'],
            incentive_coeffs = fit_args,
            min_latent = fit_args['min_latent']

        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate = fit_args['learning_rate'])

    return {
        "strategy": strategy,
        "degradation_model": degradation_model,
        "compiled_tensors": compiled_tensors,

        "train_ds": train_ds,
        "cycle_m": cycle_m,
        "cycle_v": cycle_v,

        "optimizer": optimizer,
        "my_data": my_data
    }


class LossRecord():
    def __init__(self):
        self.data = []
        self.labels = [
            'cc_capacity_loss',
            'cv_capacity_loss',
            'cc_voltage_loss',
            "q_loss",
            "scale_loss",
            "r_loss",
            "shift_loss",
            "cell_loss",
            "reciprocal_loss",
            "projection_loss",
            "out_of_bounds_loss"
        ]

    def record(self, count, losses):
        self.data.append((count, losses))

    def print_recent(self, fit_args):
        if len(self.data) > 0:
            count, losses = self.data[-1]
            print('Count {}:'.format(count))
            for i in range(len(losses)):
                print('\t{}:{}. coeff:{}'.format(
                    self.labels[i],
                    losses[i],
                    fit_args['coeff_' + self.labels[i].split('_loss')[0]]
                ))

    def plot(self, count, fit_args):
        fig = plt.figure(figsize = [11, 10])
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        for i in range(len(self.labels)):
            ax.plot(
                [s[0] for s in self.data],
                [s[1][i] for s in self.data],
                label = self.labels[i]
            )

        ax.legend()

        savefig('losses_Count_{}.png'.format(count), fit_args)
        plt.close(fig)


def train_and_evaluate(init_returns, barcodes, fit_args):
    strategy = init_returns["strategy"]

    epochs = 100000
    count = 0

    template = 'Epoch {}, Count {}, Loss {}'
    end = time.time()
    now_ker = None

    train_step_params = {
        "compiled_tensors": init_returns['compiled_tensors'],
        "optimizer": init_returns['optimizer'],
        "degradation_model": init_returns['degradation_model'],
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
            for neighborhood in init_returns["train_ds"]:
                count += 1
                sub_count += 1

                l_ = dist_train_step(strategy, neighborhood)
                if l is None:
                    l = l_
                else:
                    l += l_

                if count != 0:
                    if (count % fit_args['print_loss_every']) == 0:
                        tot = l / tf.cast(sub_count, dtype = tf.float32)
                        l = None
                        sub_count = 0
                        loss_record.record(count, tot.numpy())
                        loss_record.print_recent(fit_args)

                    plot_params = {
                        "barcodes": barcodes,
                        "count": count,
                        "fit_args": fit_args,
                    }

                    if (count % fit_args['visualize_fit_every']) == 0:

                        start = time.time()
                        print("time to simulate: ", start - end)
                        loss_record.plot(count, fit_args)
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


def train_step(neighborhood, params, fit_args):
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
            1. / tf.cast(
                neighborhood[:, NEIGH_VALID_CYC], tf.float32
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
                neighborhood[:, NEIGH_VALID_CYC], tf.float32
            ),
            [batch_size2, 1]
        ),
        [1, cv_current.shape[1]]
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
                cv_capacity
            ),
            training = True
        )

        pred_cc_capacity = train_results["pred_cc_capacity"]
        pred_cv_capacity = train_results["pred_cv_capacity"]
        pred_cc_voltage = train_results["pred_cc_voltage"]
        pred_cv_voltage = train_results["pred_cv_voltage"]
        cv_voltage = tf.tile(
            tf.expand_dims(end_voltage, axis=1),
            [1, cv_current.shape[1]],
        )

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

        cv_voltage_loss = (
                tf.reduce_mean(
                    cv_mask_2 * cv_mask * tf.square(cv_voltage - pred_cv_voltage)
                ) / (1e-10 + tf.reduce_mean(cv_mask_2 * cv_mask))
        )

        loss = (
            fit_args['coeff_cv_capacity'] * cv_capacity_loss
            + fit_args['coeff_cv_voltage'] * cv_voltage_loss
            + fit_args['coeff_cc_voltage'] * cc_voltage_loss
            + fit_args['coeff_cc_capacity'] * cc_capacity_loss
            + 1. * tf.stop_gradient(
                fit_args['coeff_cv_capacity'] * cv_capacity_loss
                + fit_args['coeff_cv_voltage'] * cv_voltage_loss
                + fit_args['coeff_cc_voltage'] * cc_voltage_loss
                + fit_args['coeff_cc_capacity'] * cc_capacity_loss
            ) * (
            fit_args['coeff_q'] *train_results["q_loss"]
            + fit_args['coeff_scale'] *train_results["scale_loss"]
            + fit_args['coeff_r'] * train_results["r_loss"]
            + fit_args['coeff_shift'] *train_results["shift_loss"]
            + fit_args['coeff_cell'] * train_results["cell_loss"]
            + fit_args['coeff_reciprocal'] * train_results["reciprocal_loss"]
            + fit_args['coeff_projection']  * train_results["projection_loss"]
            + fit_args['coeff_out_of_bounds'] * train_results["out_of_bounds_loss"]
        )

        )

    gradients = tape.gradient(
        loss,
        degradation_model.trainable_variables
    )

    gradients_no_nans = [
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        for x in gradients
    ]

    gradients_norm_clipped, _ = tf.clip_by_global_norm(
        gradients_no_nans,
        fit_args['global_norm_clip']
    )

    optimizer.apply_gradients(
        zip(
            gradients_norm_clipped,
            degradation_model.trainable_variables
        )
    )

    return tf.stack(
        [
            cc_capacity_loss,
            cv_capacity_loss,
            cc_voltage_loss,
            train_results["q_loss"],
            train_results["scale_loss"],
            train_results["r_loss"],

            train_results["shift_loss"],
            train_results["cell_loss"],
            train_results["reciprocal_loss"],

            train_results["projection_loss"],
            train_results["out_of_bounds_loss"]
        ],
    )


def ml_smoothing(fit_args):
    if len(tf.config.experimental.list_physical_devices('GPU')) == 1:
        strategy = tf.distribute.OneDeviceStrategy(
            device = '/gpu:0'
        )
    elif len(tf.config.experimental.list_physical_devices('GPU')) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(
            '/cpu:0'
        )

    if not os.path.exists(fit_args['path_to_plots']):
        os.mkdir(fit_args['path_to_plots'])

    with open(
        os.path.join(fit_args['path_to_plots'], 'fit_args_log.txt'), 'w'
    ) as f:
        my_str = ''
        for k in fit_args:
            my_str = '{} \n {}: {}'.format(my_str, k, str(fit_args[k]))
        f.write(my_str)

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
        initial_processing(
            my_data, my_names, barcodes, fit_args, strategy = strategy
        ), barcodes,
        fit_args)


class Command(BaseCommand):

    def add_arguments(self, parser):

        required_args = [
            '--path_to_dataset',
            '--dataset_version',
            '--path_to_plots'
        ]

        float_args = {
            '--global_norm_clip': 10.,

            '--learning_rate': 1e-4,
            '--min_latent': .1,

            '--coeff_d_features': .0001,
            '--coeff_d2_features': .0001,

            '--coeff_cell': .001,
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
            '--coeff_q_d3_current': .1,

            '--coeff_scale': 5.,
            '--coeff_scale_geq': 5.,
            '--coeff_scale_leq': 5.,
            '--coeff_scale_eq': 5.,
            '--coeff_scale_mono': 1.,
            '--coeff_scale_d3_cycle': .002,

            '--coeff_r': 5.,
            '--coeff_r_geq': 1.,
            '--coeff_r_big': .0,
            '--coeff_r_d3_cycle': .002,

            '--coeff_shift': 1.,
            '--coeff_shift_geq': .5,
            '--coeff_shift_leq': .5,
            '--coeff_shift_small': .1,
            '--coeff_shift_d3_cycle': .0,
            '--coeff_shift_mono': .0,
            '--coeff_shift_a_big': .001,

            '--coeff_reciprocal': 10.,
            '--coeff_reciprocal_v': 10.,
            '--coeff_reciprocal_q': 10.,
            '--coeff_reciprocal_v_small': .001,
            '--coeff_reciprocal_v_geq': 1.,
            '--coeff_reciprocal_v_leq': .5,
            '--coeff_reciprocal_v_mono': 10.,
            '--coeff_reciprocal_d3_current': 1.,
            '--coeff_reciprocal_d_current_minus': 10.,
            '--coeff_reciprocal_d_current_plus': 5.,

            '--coeff_projection': .01,
            '--coeff_projection_pos': 1.,
            '--coeff_projection_neg': 1.,

            '--coeff_out_of_bounds': 10.,
            '--coeff_out_of_bounds_geq': 1.,
            '--coeff_out_of_bounds_leq': 1.,
        }

        vis = 10000
        int_args = {
            '--n_sample': 8 * 16,

            '--depth': 3,
            '--width': 50,
            '--batch_size': 4 * 16,

            '--print_loss_every': 500,
            '--visualize_fit_every': vis,
            '--visualize_vq_every': vis,

            '--stop_count': 1000004,
            '--barcode_show': 10
        }

        for arg in required_args:
            parser.add_argument(arg, required = True)

        for arg in float_args:
            parser.add_argument(arg, type = float, default = float_args[arg])

        for arg in int_args:
            parser.add_argument(arg, type = int, default = float_args[arg])

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
            83741, 83742, 83743, 83744, 83745, 83746, 83747, 83748
        ]
        # 57706, 57707, 57710, 57711, 57714, 57715,64260,64268,83010, 83011,
        # 83012, 83013, 83014, 83015, 83016

        parser.add_argument(
            '--wanted_barcodes', type = int, nargs = '+', default = barcodes
        )

    def handle(self, *args, **options):
        ml_smoothing(options)
