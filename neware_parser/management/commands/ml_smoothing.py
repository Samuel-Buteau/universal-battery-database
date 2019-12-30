import os
import pickle
import numpy
#import argparse
#import math
#import re
#import copy

import tensorflow as tf

from django.core.management.base import BaseCommand
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, GlobalAveragePooling1D,
    BatchNormalization, Conv1D, Layer
)
from mpl_toolkits.mplot3d import Axes3D

from neware_parser.models import *
from .plot import plot_vq, plot_test_rate_voltage, plot_capacity, plot_eq_vol

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
'''

class Data:

    def __init__(self, vector):
        self.val = tf.cast(tf.constant(vector), dtype=tf.float32)
        self.len = len(vector)
        self.mean, self.var = tf.nn.moments(self.val, axes=[0])
        self.val = (self.val - self.mean) / tf.sqrt(self.var)

    def __call__(self):
        return self.val


# stores cell features
# key: index
# value: feature (matrix)
class DictionaryLayer(Layer):

    def __init__(self, num_features, num_keys):
        super(DictionaryLayer, self).__init__()
        self.num_features = num_features
        self.num_keys = num_keys
        self.kernel = self.add_weight(
            "kernel", shape=[self.num_keys, self.num_features * 2])

    def call(self, input, training=True):
        eps = tf.random.normal(
            shape=[self.num_keys, self.num_features])
        mean = self.kernel[:, :self.num_features]
        log_sig = self.kernel[:, self.num_features:]

        if not training:
            features = mean
        else:
            features = mean + tf.exp(log_sig / 2.) * eps

        # tf.gather: "fetching in the dictionary"
        fetched_features = tf.gather(features, input, axis=0)
        fetched_mean = tf.gather(mean, input, axis=0)
        fetched_log_sig = tf.gather(log_sig, input, axis=0)

        return fetched_features, fetched_mean, fetched_log_sig


################################################################################
# Begin: Degradation Model
################################################################################

class FeedforwardNeuralNetwork:

    def __init__(self, depth, width):
        self.initial = Dense(
            width,
            activation = 'relu',
            use_bias = True,
            bias_initializer = 'zeros'
        )

        self.bulk = [
            Dense(
                width,
                activation = 'relu',
                use_bias = True,
                bias_initializer = 'zeros'
            ) for _ in range(depth)
        ]

        self.final = Dense(
            1,
            activation = None,
            use_bias = True,
            bias_initializer = 'zeros',
            kernel_initializer = 'zeros'
        )


class DegradationModel(Model):

    def __init__(self, num_keys, depth, width):
        super(DegradationModel, self).__init__()

        cap_nn = FeedforwardNeuralNetwork(depth, width)
        eq_vol_nn = FeedforwardNeuralNetwork(depth, width)
        r_nn = FeedforwardNeuralNetwork(depth, width)

        self.feedforward_nn = {
            'cap': {
                'initial': cap_nn.initial,
                'bulk': cap_nn.bulk,
                'final': cap_nn.final
            },
            'eq_vol': {
                'initial': eq_vol_nn.initial,
                'bulk': eq_vol_nn.bulk,
                'final': eq_vol_nn.final
            },
            'r': {
                'initial': r_nn.initial,
                'bulk': r_nn.bulk,
                'final': r_nn.final
            },
        }

        self.dictionary = DictionaryLayer(num_features=width, num_keys=num_keys)

        self.width = width
        self.num_keys = num_keys

    def apply_nn(self, cycles, rates, features, nn):

        # two nn's are needed to implement the equation for discharge voltage
        # the keyword 'max_discharge_voltage' calls the networks
        # with the right inputs and plumbing
        if nn == 'max_dchg_vol':
            dchg_rate = rates[:, 1:2]
            eq_vol = self.apply_nn(cycles, rates, features, 'eq_vol')
            r = self.apply_nn(cycles, None, features, 'r')
            return {
                "max_dchg_vol": eq_vol - (dchg_rate * r),
                "eq_vol": eq_vol,
                "r": r
            }

        else:

            if nn == 'eq_vol':
                rates = rates[:, 0:1]
            if nn == 'r':
                rates = None

            if rates is None:
                centers = self.feedforward_nn[nn]['initial'](
                    tf.concat(
                        (
                            # readjust the cycles
                            cycles * (1e-10 + tf.exp(-features[:, 0:1])),
                            features[:, 1:]
                        ),
                        axis=1
                    )
                )

            else:
                centers = self.feedforward_nn[nn]['initial'](
                    tf.concat(
                        (
                            # readjust the cycles
                            cycles * (1e-10 + tf.exp(-features[:, 0:1])),
                            rates,
                            features[:, 1:]
                        ),
                        axis=1
                    )
                )
            for d in self.feedforward_nn[nn]['bulk']:
                centers = d(centers)
            return self.feedforward_nn[nn]['final'](centers)

    def create_derivatives(self, cycles, rates, features, nn):
        derivatives = {}
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch(cycles)
            tape3.watch(rates)
            tape3.watch(features)

            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(cycles)
                tape2.watch(rates)
                tape2.watch(features)

                if nn == 'max_dchg_vol':
                    nn_results = self.apply_nn(cycles, rates, features, nn)
                    res = tf.reshape(nn_results["max_dchg_vol"], [-1, 1])
                else:
                    res = tf.reshape(
                        self.apply_nn(cycles, rates, features, nn), [-1, 1])

            derivatives['dCyc'] = tape2.batch_jacobian(
                source=cycles, target=res)[:, 0, :]
            derivatives['dRates'] = tape2.batch_jacobian(
                source=rates, target=res)[:, 0, :]
            derivatives['dFeatures'] = tape2.batch_jacobian(
                source=features, target=res)[:, 0, :]
            del tape2

        derivatives['d2Cyc'] = tape3.batch_jacobian(
            source=cycles, target=derivatives['dCyc'])[:, 0, :]
        derivatives['d2Rates'] = tape3.batch_jacobian(
            source=rates, target=derivatives['dRates'])
        derivatives['d2Features'] = tape3.batch_jacobian(
            source=features, target=derivatives['dFeatures'])

        del tape3
        return res, derivatives

    def call(self, x, training=False):

        centers = x[0]  # batch of [cyc, k[0], k[1]]
        indecies = x[1]  # batch of index
        meas_cycles = x[2]  # batch of cycles
        vol_tensor = x[3]

        features, mean, log_sig = self.dictionary(indecies, training=training)
        cycles = centers[:, 0:1]
        rates = centers[:, 1:]

        # duplicate cycles and others for all the voltages
        # dimensions are now [batch, voltages, features]
        cycles_tiled = tf.tile(
            tf.expand_dims(cycles, axis=1), [1, vol_tensor.shape[0], 1])
        rates_tiled = tf.tile(
            tf.expand_dims(rates, axis=1), [1, vol_tensor.shape[0], 1])
        features_tiled = tf.tile(
            tf.expand_dims(features, axis=1), [1, vol_tensor.shape[0], 1])
        voltages_tiled = tf.tile(
            tf.expand_dims(
                tf.expand_dims(vol_tensor, axis=1), axis=0),
            [cycles.shape[0], 1, 1]
        )

        rates_concat = tf.concat((rates_tiled, voltages_tiled), axis=2)

        cycles_flat = tf.reshape(cycles_tiled, [-1, 1])
        rates_flat = tf.reshape(rates_concat, [-1, 3])
        features_flat = tf.reshape(features_tiled, [-1, self.width])

        # now every dimension works for concatenation

        if training:

            var_cyc = tf.expand_dims(meas_cycles, axis=1) - cycles
            var_cyc_squared = tf.square(var_cyc)

            ''' discharge capacity '''
            cap, cap_derivatives = self.create_derivatives(
                cycles_flat, rates_flat, features_flat, 'cap')
            cap = tf.reshape(cap, [-1, vol_tensor.shape[0]])

            pred_cap = (
                cap + var_cyc * tf.reshape(
                    cap_derivatives['dCyc'], [-1, vol_tensor.shape[0]])
                + var_cyc_squared * tf.reshape(
                    cap_derivatives['d2Cyc'], [-1, vol_tensor.shape[0]])
            )

            ''' discharge max voltage '''
            max_dchg_vol, max_dchg_vol_der = self.create_derivatives(
                cycles, rates, features, 'max_dchg_vol')
            max_dchg_vol = tf.reshape(max_dchg_vol, [-1])

            '''resistance derivatives '''
            r, r_der = self.create_derivatives(
                cycles, rates, features, 'r')
            r = tf.reshape(r, [-1])

            '''eq_vol derivatives '''
            eq_vol, eq_vol_der = self.create_derivatives(
                cycles, rates, features, 'eq_vol')
            eq_vol = tf.reshape(eq_vol, [-1])

            pred_max_dchg_vol = (
                max_dchg_vol + tf.reshape(max_dchg_vol_der['dCyc'], [-1])
                * tf.reshape(var_cyc, [-1])
                + tf.reshape(max_dchg_vol_der['d2Cyc'], [-1])
                * tf.reshape(var_cyc_squared, [-1])
            )


            return {
                "pred_cap": pred_cap,
                "pred_max_dchg_vol": tf.reshape(pred_max_dchg_vol, [-1]),
                "pred_eq_vol": tf.reshape(eq_vol, [-1]),
                "pred_r": tf.reshape(r, [-1]),
                "mean": mean,
                "log_sig": log_sig,
                "cap_der": cap_derivatives,
                "max_dchg_vol_der": max_dchg_vol_der,
                "r_der": r_der,
                "eq_vol_der": eq_vol_der,
            }

        else:
            nn_results = self.apply_nn(cycles, rates, features, 'max_dchg_vol')

            return {
                "pred_cap": tf.reshape(
                    self.apply_nn(
                        cycles_flat, rates_flat, features_flat, 'cap'),
                    [-1, vol_tensor.shape[0]]
                ),
                "pred_max_dchg_vol": nn_results["max_dchg_vol"],
                "pred_eq_vol": nn_results["eq_vol"],
                "pred_r": nn_results["r"]
            }


################################################################################
# End: Degradation Model
################################################################################

# ==== Begin: initial processing ===============================================

NEIGH_INT_MIN_CYC_INDEX = 0
NEIGH_INT_MAX_CYC_INDEX = 1
NEIGH_INT_RATE_INDEX = 2
NEIGH_INT_BARCODE_INDEX = 3
NEIGH_INT_ABSOLUTE_INDEX = 4
NEIGH_INT_VALID_CYC_INDEX = 5
NEIGH_FLOAT_DELTA = 0
NEIGH_FLOAT_CHG_RATE = 1
NEIGH_FLOAT_DCHG_RATE = 2


def initial_processing(my_data, barcodes, fit_args):
    all_cells_neigh_data_int, all_cycle_nums, all_dchg_vol = [], [], []
    all_cells_neigh_data_float, all_vq_curves, all_vq_curves_masks = [], [], []

    test_object = {}

    '''
    - cycles are grouped by their charge rates and discharge rates.
    - a cycle group contains many cycles
    - things are split up this way to sample each group equally
    - each barcode corresponds to a single cell
    '''
    for barcode_count, barcode in enumerate(barcodes):

        test_object[barcode_count] = {}

        # here we load as if it were the original data

        '''
        - dictionary indexed by charging and discharging rate (i.e. cycle group)
        - contains structured arrays of
            - cycle_number
            - capacity_vector: a vector where each element is a
              capacity associated with a given voltage
              [(voltage_vector[i], capacity_vector[i])
              is a voltage-capacity pair]
            - vq_curve_mask: a vector where each element is a weight
              corresponding to a voltage-capacity pair
              [this allows us to express the fact that sometimes a given
              voltage was not measured, so the capacity is meaningless.
              (mask of 0)]
        '''
        max_cap = 0.
        cyc_grp_dict = my_data['all_data'][barcode]
        # find largest cap measured for this cell (max over all cycle groups)
        for k in cyc_grp_dict.keys():
            max_cap = max(
                max_cap, max(cyc_grp_dict[k]['capacity_vector'][:, 0]))

        print("max_cap:", max_cap)

        cell_neigh_data_int, cell_neigh_data_float = [], []

        for k_count, k in enumerate(cyc_grp_dict.keys()):

            # normalize capacity_vector with max_cap
            my_data['all_data'][barcode][k]['capacity_vector'] = (
                1. / max_cap * cyc_grp_dict[k]['capacity_vector'])

            print("k:", k)

            # range of cycles which exist for this cycle group
            min_cyc = min(cyc_grp_dict[k]['cycle_number'])
            max_cyc = max(cyc_grp_dict[k]['cycle_number'])

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

            neigh_data_int, neigh_data_float = [], []

            # check all tentative neighborhood centers and
            # commit the ones that contain good data to the dataset
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
                    below_cyc <= cyc_grp_dict[k]['cycle_number'],
                    cyc_grp_dict[k]['cycle_number'] <= above_cyc
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

                # add the neighborhood
                # if no neighborhoods were added, initialize test_object
                if valid_cycles == 0:
                    test_object[barcode_count][k] = []

                test_object[barcode_count][k].append(cyc)
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

                neigh_data_int.append(
                    [min_cyc_index, max_cyc_index, k_count, barcode_count,
                     len(all_cycle_nums), 0]
                )
                neigh_data_float.append([combined_delta, k[0], k[1]])

            if valid_cycles != 0:
                neigh_data_int = numpy.array(
                    neigh_data_int, dtype=numpy.int32)

                # the empty slot becomes the count of added neighborhoods, which
                # are used to counterbalance the bias toward longer cycle life
                neigh_data_int[:, NEIGH_INT_VALID_CYC_INDEX] = valid_cycles
                neigh_data_float = numpy.array(
                    neigh_data_float, dtype=numpy.float32)

                cell_neigh_data_int.append(neigh_data_int)
                cell_neigh_data_float.append(neigh_data_float)

            else:
                print('name: ', barcode)
                print('rates: ', k)

            if len(all_cycle_nums) > 0:

                # giant array with all the cycle numbers
                all_cycle_nums = numpy.concatenate(
                    (all_cycle_nums, cyc_grp_dict[k]['cycle_number']))

                # giant array of all the vq_curves
                all_vq_curves = numpy.concatenate(
                    (all_vq_curves, cyc_grp_dict[k]['capacity_vector']))

                # giant array of all the vq_curves_mask
                all_vq_curves_masks = numpy.concatenate(
                    (all_vq_curves_masks, cyc_grp_dict[k]['vq_curve_mask']))

                all_dchg_vol = numpy.concatenate((
                    all_dchg_vol,
                    cyc_grp_dict[k]['dchg_maximum_voltage'])
                )

            else:
                all_cycle_nums = cyc_grp_dict[k]['cycle_number']
                all_vq_curves = cyc_grp_dict[k]['capacity_vector']
                all_vq_curves_masks = cyc_grp_dict[k]['vq_curve_mask']
                all_dchg_vol = cyc_grp_dict[k]['dchg_maximum_voltage']

        if len(cell_neigh_data_int) != 0:
            all_cells_neigh_data_int.append(cell_neigh_data_int)
            all_cells_neigh_data_float.append(cell_neigh_data_float)
        else:
            print("barcode: ", barcode)

    neigh_data_int = tf.constant(numpy.concatenate(
        [numpy.concatenate(cell_neigh_data_int, axis=0)
            for cell_neigh_data_int in all_cells_neigh_data_int],
        axis=0)
    )

    # cycles go from 0 to 6000, but nn prefers normally distributed variables
    # so cycle numbers is normalized with mean and variance
    cycles_tensor = tf.constant(all_cycle_nums)
    cycles_m, cycles_v = tf.nn.moments(cycles_tensor, axes=[0])
    cycles_m = cycles_m.numpy()
    cycles_v = cycles_v.numpy()
    cycles_tensor = (cycles_tensor - cycles_m) / tf.sqrt(cycles_v)

    # the voltages are also normalized
    vol_tensor = tf.cast(tf.constant(my_data['voltage_grid']), dtype=tf.float32)
    voltages_m, voltages_v = tf.nn.moments(vol_tensor, axes=[0])
    vol_tensor = (vol_tensor - voltages_m) / tf.sqrt(voltages_v)
    vq_curves = tf.constant(all_vq_curves)
    vq_curves_mask = tf.constant(all_vq_curves_masks)

    # max voltage is NOT normalized
    max_dchg_vol_tensor = tf.constant(all_dchg_vol)

    neigh_data_float = (numpy.concatenate(
        [numpy.concatenate(neigh_data_float_full, axis=0)
         for neigh_data_float_full
         in all_cells_neigh_data_float],
        axis=0)
    )

    # onvert the delta_cycles of each neighborhoods to the normalized units
    # (divide by standard deviation)
    neigh_data_float[:, NEIGH_FLOAT_DELTA] = (
        (neigh_data_float[:, NEIGH_FLOAT_DELTA]) / numpy.sqrt(cycles_v))

    neigh_data_float = tf.constant(neigh_data_float)

    batch_size = fit_args['batch_size']
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        train_ds_ = tf.data.Dataset.from_tensor_slices(
            (neigh_data_int, neigh_data_float)
        ).repeat(2).shuffle(100000).batch(batch_size)

        train_ds = mirrored_strategy.experimental_distribute_dataset(
            train_ds_)

        degradation_model = DegradationModel(
            num_keys=len(barcodes),
            width=fit_args['width'],
            depth=fit_args['depth'])

        optimizer = tf.keras.optimizers.Adam()

    return {
        "mirrored_strategy": mirrored_strategy,
        "degradation_model": degradation_model,

        "cycles_tensor": cycles_tensor,
        "vol_tensor": vol_tensor,
        "max_dchg_vol_tensor": max_dchg_vol_tensor,

        "train_ds": train_ds,
        "cycles_m": cycles_m,
        "cycles_v": cycles_v,

        "vq_curves": vq_curves,
        "vq_curves_mask": vq_curves_mask,
        "optimizer": optimizer,
        "test_object": test_object,
        "all_data": my_data['all_data'],
        "voltage_vector": my_data['voltage_grid'],
    }


# === End: initial processing ==================================================

# === Begin: train =============================================================

def train_and_evaluate(init_returns, barcodes, fit_args):
    mirrored_strategy = init_returns["mirrored_strategy"]

    EPOCHS = 100000
    count = 0

    template = 'Epoch {}, Count {}'

    with mirrored_strategy.scope():
        for epoch in range(EPOCHS):
            for neigh_int, neigh_float in init_returns["train_ds"]:
                count += 1

                train_step_params = {
                    "neigh_float": neigh_float,
                    "neigh_int": neigh_int,

                    "cycles_tensor": init_returns["cycles_tensor"],
                    "vol_tensor": init_returns["vol_tensor"],
                    "max_dchg_vol_tensor": init_returns["max_dchg_vol_tensor"],

                    "degradation_model": init_returns["degradation_model"],
                    "optimizer": init_returns["optimizer"],
                    "vq_curves": init_returns["vq_curves"],
                    "vq_curves_mask": init_returns["vq_curves_mask"],
                    "voltage_vector": init_returns["voltage_vector"]
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
                        plot_capacity(plot_params, init_returns)
                        plot_eq_vol(plot_params, init_returns)

                    if (count % fit_args['visualize_vq_every']) == 0:
                        plot_vq(plot_params, init_returns)

                if count == fit_args['stop_count']:
                    return


# === End: train ===============================================================

# === Begin: train step ========================================================

def train_step(params, fit_args):
    neigh_float = params["neigh_float"]
    neigh_int = params["neigh_int"]

    cycles_tensor = params["cycles_tensor"]
    vol_tensor = params["vol_tensor"]

    degradation_model = params["degradation_model"]
    optimizer = params["optimizer"]
    vq_curves = params["vq_curves"]
    vq_curves_mask = params["vq_curves_mask"]
    voltage_vector = params["voltage_vector"]

    # need to split the range
    batch_size2 = neigh_int.shape[0]

    '''
    find the actual cycle number by interpolation
    then offset the cycle number by the delta * randomness.
    '''

    # offset center cycles so the model is never evaluated at the same cycle
    center_cycle_offsets = tf.random.uniform(
        [batch_size2], minval=-1., maxval=1., dtype=tf.float32)

    '''
    if you have the minimum cycle and maximum cycle for a neighborhood,
    you can sample cycles from this neighborhood by sampling real numbers
    x from [0,1] and computing min_cyc*(1.-x) + max_cyc*x,
    but here this computation is done in index space,
    then cycle numbers and vq curves are gathered
    '''

    cycle_indecies_lerp = tf.random.uniform(
        [batch_size2], minval=0., maxval=1., dtype=tf.float32)
    cycle_indecies = tf.cast(
        (1. - cycle_indecies_lerp) * tf.cast(
            neigh_int[:, NEIGH_INT_MIN_CYC_INDEX]
            + neigh_int[:, NEIGH_INT_ABSOLUTE_INDEX],
            tf.float32)
        + (cycle_indecies_lerp) * tf.cast(
            neigh_int[:, NEIGH_INT_MAX_CYC_INDEX]
            + neigh_int[:, NEIGH_INT_ABSOLUTE_INDEX],
            tf.float32),
        tf.int32
    )

    meas_cycles = tf.gather(
        cycles_tensor, indices=cycle_indecies, axis=0)
    model_eval_cycles = (
        meas_cycles + center_cycle_offsets * neigh_float[:, NEIGH_FLOAT_DELTA])

    cap = tf.gather(vq_curves, indices=cycle_indecies)

    ws_cap = tf.gather(vq_curves_mask, indices=cycle_indecies)
    ws2_cap = tf.tile(
        tf.reshape(
            1. / (tf.cast(neigh_int[:, NEIGH_INT_VALID_CYC_INDEX], tf.float32)),
            [batch_size2, 1]
        ),
        [1, len(voltage_vector)]
    )

    ''' maximum discharge voltage '''

    meas_max_dchg_vol = tf.reshape(tf.gather(
        params["max_dchg_vol_tensor"], indices=cycle_indecies, axis=0), [-1])

    # weight for the prediction error
    # the more measurements you have for a cell, then less each one is worth
    # So that in expectation, you "care" about every cell equally.
    ws2_max_dchg_vol = 1. / (tf.cast(
        neigh_int[:, NEIGH_INT_VALID_CYC_INDEX], tf.float32))

    # the indecies are referring to the cell indecies
    indecies = neigh_int[:, NEIGH_INT_BARCODE_INDEX]

    centers = tf.concat(
        (tf.expand_dims(model_eval_cycles, axis=1), neigh_float[:, 1:]), axis=1)

    with tf.GradientTape() as tape:
        train_results = degradation_model(
            (centers, indecies, meas_cycles, vol_tensor), training=True)

        pred_cap = train_results["pred_cap"]
        pred_max_dchg_vol = train_results["pred_max_dchg_vol"]
        mean = train_results["mean"]
        log_sig = train_results["log_sig"]
        cap_der = train_results["cap_der"]
        max_dchg_vol_der = train_results["max_dchg_vol_der"]
        r = train_results["pred_r"]
        eq_vol = train_results["pred_eq_vol"]
        r_der = train_results["r_der"]
        eq_vol_der = train_results["eq_vol_der"]

        cap_loss = (
            tf.reduce_mean(ws2_cap * ws_cap * tf.square(cap - pred_cap))
            / (1e-10 + tf.reduce_mean(ws2_cap * ws_cap))
            + tf.reduce_mean(ws2_max_dchg_vol
            * tf.square(meas_max_dchg_vol - pred_max_dchg_vol))
            / (1e-10 + tf.reduce_mean(ws2_max_dchg_vol))
        )

        kl_loss = fit_args['kl_coeff'] * tf.reduce_mean(
            0.5 * (tf.exp(log_sig) + tf.square(mean) - 1. - log_sig))

        mono_loss = fit_args['mono_coeff'] * (
            tf.reduce_mean(tf.nn.relu(-cap))  # penalizes negative capacities
            + tf.reduce_mean(tf.nn.relu(cap_der['dCyc'])) # shouldn't increase
            + tf.reduce_mean(tf.nn.relu(cap_der['dRates'])) # shouldn't increase

            + 10.* (
                tf.reduce_mean(tf.nn.relu(-r))
                + tf.reduce_mean(tf.nn.relu(-eq_vol))
                # resistance should not decrease.
                + 10  * tf.reduce_mean(tf.abs(r_der['dCyc']))
                # equilibrium voltage should not change much
                + 10. * tf.reduce_mean(tf.abs(eq_vol_der['dCyc']))
                # equilibrium voltage should not change much
                + 10. * tf.reduce_mean(tf.abs(eq_vol_der['dRates']))
            )
        )

        smooth_loss = fit_args['smooth_coeff'] * (
            tf.reduce_mean(tf.square(tf.nn.relu(cap_der['d2Cyc']))
            + 0.02 * tf.square(tf.nn.relu(-cap_der['d2Cyc'])))
            + tf.reduce_mean(
                tf.square(tf.nn.relu(cap_der['d2Rates']))
                + 0.02 * tf.square(tf.nn.relu(-cap_der['d2Rates']))
            )

            # this enforces smoothness of resistance;
            # it is more ok to accelerate UPWARDS
            + 10. * tf.reduce_mean(tf.square(tf.nn.relu(-r_der['d2Cyc']))
            + 0.5 * tf.square(tf.nn.relu(r_der['d2Cyc'])))
            + 1.* tf.reduce_mean(tf.square((eq_vol_der['d2Rates'])))
            + 1.* tf.reduce_mean(tf.square((eq_vol_der['d2Cyc'])))
        )

        const_f_loss = fit_args['const_f_coeff'] * (
            tf.reduce_mean(tf.square(cap_der['dFeatures']))
            + tf.reduce_mean(tf.square(r_der['dFeatures']))
            + tf.reduce_mean(tf.square(eq_vol_der['dFeatures']))
        )

        smooth_f_loss = fit_args['smooth_f_coeff'] * (
            tf.reduce_mean(tf.square(cap_der['d2Features']))
            + tf.reduce_mean(tf.square(r_der['d2Features']))
            + tf.reduce_mean(tf.square(eq_vol_der['d2Features']))
        )

        loss = cap_loss + kl_loss + mono_loss + smooth_loss
        loss += const_f_loss + smooth_f_loss

    gradients = tape.gradient(loss, degradation_model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, degradation_model.trainable_variables))


# === End : train step =========================================================

@tf.function
def dist_train_step(mirrored_strategy, train_step_params, fit_args):
    mirrored_strategy.experimental_run_v2(
        train_step, args=(train_step_params, fit_args))

def ml_smoothing(fit_args):
    if not os.path.exists(fit_args['path_to_plots']):
        os.mkdir(fit_args['path_to_plots'])

    dataset_path = os.path.join(
        fit_args['path_to_dataset'],
        'dataset_ver_{}.file'.format(fit_args['dataset_version'])
    )

    if not os.path.exists(dataset_path):
        print("Path \"" + dataset_path + "\" does not exist.")
        return

    with open(dataset_path, 'rb') as f:
        my_data = pickle.load(f)

    barcodes = list(my_data['all_data'].keys())

    if len(fit_args['wanted_barcodes']) !=0:
        barcodes = list(
            set(barcodes).intersection(set(fit_args['wanted_barcodes'])))

    if len(barcodes) == 0:
        return

    train_and_evaluate(
        initial_processing(my_data, barcodes, fit_args), barcodes, fit_args)


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path_to_dataset', required=True)
        parser.add_argument('--dataset_version', required=True)
        parser.add_argument('--path_to_plots', required=True)
        parser.add_argument('--kl_coeff', type=float, default=0.00001)
        parser.add_argument('--mono_coeff', type=float, default=.005)
        parser.add_argument('--smooth_coeff', type=float, default=.05)
        parser.add_argument('--const_f_coeff', type=float, default=.0)
        parser.add_argument('--smooth_f_coeff', type=float, default=.01)
        parser.add_argument('--depth', type=int, default=3)
        parser.add_argument('--width', type=int, default=32)
        parser.add_argument('--batch_size', type=int, default=2 * 16)
        parser.add_argument('--print_loss_every', type=int, default=1000)
        parser.add_argument(
            '--visualize_fit_every', type=int, default=10000)
        parser.add_argument(
            '--visualize_vq_every', type=int, default=10000)

        parser.add_argument('--stop_count', type=int, default=80000)
        parser.add_argument(
            '--wanted_barcodes', type=int, nargs='+', default=[83220, 83083])

    def handle(self, *args, **options):
        ml_smoothing(options)



