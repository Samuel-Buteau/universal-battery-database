import argparse
import os
import pickle
import numpy
import math
import re
import copy
from neware_parser.models import *
from django.core.management.base import BaseCommand
import tensorflow as tf

from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, GlobalAveragePooling1D,
    BatchNormalization, Conv1D, Layer
)
from tensorflow.keras import Model

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Colour:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def cprint(c, s):
    print(c + str(s) + Colour.END)


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

class DegradationModel(Model):

    def __init__(self, num_keys, depth, width):
        super(DegradationModel, self).__init__()

        self.neural_network = {
            'cap':{
                'initial':Dense(width, activation='relu', use_bias=True, bias_initializer='zeros'),
                'bulk':[Dense(width, activation='relu', use_bias=True,
                      bias_initializer='zeros') for _ in range(depth)],
                'final':Dense(1, activation=None, use_bias=True,
                                 bias_initializer='zeros',
                                 kernel_initializer='zeros')

            },
            'vol':{
                'initial':Dense(width, activation='relu', use_bias=True, bias_initializer='zeros'),
                'bulk':[Dense(width, activation='relu', use_bias=True,
                      bias_initializer='zeros') for _ in range(depth)],
                'final':Dense(1, activation=None, use_bias=True,
                                 bias_initializer='zeros',
                                 kernel_initializer='zeros')

            },
        }

        self.dictionary = DictionaryLayer(num_features=width, num_keys=num_keys)

        self.width = width
        self.num_keys = num_keys

    # DONE: you must return predicted capacity and predicted voltage
    # even when not training! otherwise we can't see the predictions.

    def apply_nn(self, cycles, others, features, nn):
        centers = self.neural_network[nn]['initial'](
            tf.concat(
                (
                    # readjust the cycles
                    cycles * (1e-10 + tf.exp(-features[:, 0:1])),
                    others,
                    features[:, 1:]),
                axis=1))
        for d in self.neural_network[nn]['bulk']:
            centers = d(centers)
        return self.neural_network[nn]['final'](centers)

    def create_derivatives(self, cycles, others, features, nn):
        derivatives = {}
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch(cycles)
            tape3.watch(others)
            tape3.watch(features)

            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(cycles)
                tape2.watch(others)
                tape2.watch(features)

                res = tf.reshape(self.apply_nn(cycles, others, features, nn), [-1, 1])

            derivatives['dCyc'] = tape2.batch_jacobian(
                source=cycles, target=res)[:, 0, :]
            derivatives['dOthers'] = tape2.batch_jacobian(
                source=others, target=res)[:, 0, :]
            derivatives['dFeatures'] = tape2.batch_jacobian(
                source=features, target=res)[:, 0, :]
            del tape2

        derivatives['d2Cyc'] = tape3.batch_jacobian(
            source=cycles, target=derivatives['dCyc'])[:, 0, :]
        derivatives['d2Others'] = tape3.batch_jacobian(
            source=others, target=derivatives['dOthers'])
        derivatives['d2Features'] = tape3.batch_jacobian(
            source=features, target=derivatives['dFeatures'])

        del tape3
        return res, derivatives

    def call(self, x, training=False):

        centers = x[0]  # batch of [cyc, k[0], k[1]]
        indecies = x[1]  # batch of index
        measured_cycles = x[2]  # batch of cycles
        voltages = x[3]  # vector of voltages (doesn't change across the batch)

        features, mean, log_sig = self.dictionary(indecies, training=training)
        cycles = centers[:, 0:1]
        others = centers[:, 1:]

        # duplicate cycles and others for all the voltages
        # dimensions are now [batch, voltages, features]
        cycles_tiled = tf.tile(
            tf.expand_dims(cycles, axis=1), [1, voltages.shape[0], 1])
        others_tiled = tf.tile(
            tf.expand_dims(others, axis=1), [1, voltages.shape[0], 1])
        features_tiled = tf.tile(
            tf.expand_dims(features, axis=1), [1, voltages.shape[0], 1])
        voltages_tiled = tf.tile(
            tf.expand_dims(
                tf.expand_dims(voltages, axis=1), axis=0),
            [cycles.shape[0], 1, 1])

        others_concated = tf.concat((others_tiled, voltages_tiled), axis=2)


        cycles_flat= tf.reshape(cycles_tiled, [-1, 1])
        others_flat = tf.reshape(others_concated, [-1, 3])
        features_flat = tf.reshape(features_tiled, [-1, self.width])

        # now every dimension works for concatenation

        if training:
            cap, cap_derivatives = self.create_derivatives(cycles_flat, others_flat, features_flat, 'cap')

            vol, vol_derivatives = self.create_derivatives(cycles, others, features, 'vol')

            var_cyc = tf.expand_dims(measured_cycles, axis=1) - cycles
            var_cyc_squared = tf.square(var_cyc)

            cap = tf.reshape(cap, [-1, voltages.shape[0]])
            predicted_cap = (cap +
                             tf.reshape(cap_derivatives['dCyc'], [-1, voltages.shape[0]]) * var_cyc +
                             tf.reshape(cap_derivatives['d2Cyc'], [-1, voltages.shape[0]]) * var_cyc_squared
                             )

            vol = tf.reshape(vol, [-1])
            predicted_vol = (vol +tf.reshape(vol_derivatives['dCyc'], [-1]) * tf.reshape(var_cyc,[-1]) + tf.reshape(vol_derivatives['d2Cyc'], [-1]) * tf.reshape(var_cyc_squared, [-1]))

            return (predicted_cap, tf.reshape(predicted_vol, [-1]), mean, log_sig, cap_derivatives, vol_derivatives)

        else:
            predicted_cap = self.apply_nn(cycles_flat, others_flat, features_flat, 'cap')
            predicted_vol = self.apply_nn(cycles, others, features, 'vol')
            return tf.reshape(predicted_cap, [-1, voltages.shape[0]]), predicted_vol


################################################################################
# End: Degradation Model
################################################################################

# === Begin: clamp =============================================================

def clamp(a, x, b):
    print(Colour.PINK + "clamp" + Colour.END)

    x = min(x, b)
    x = max(x, a)
    print(Colour.GREEN + "clamp" + Colour.END)
    return x


# === End: clamp ===============================================================


# === Begin: make my barcodes ==================================================

WANTED_BARCODES = [83220, 83083]


def make_my_barcodes(args):
    if not os.path.exists(args['path_to_plots']):
        os.mkdir(args['path_to_plots'])

    my_barcodes = CyclingFile.objects.filter(
        database_file__deprecated=False,
        database_file__is_valid=True
    ).exclude(
        database_file__valid_metadata=None
    ).order_by(
        'database_file__valid_metadata__barcode'
    ).values_list(
        'database_file__valid_metadata__barcode',
        flat=True
    ).distinct()

    used_barcodes = []
    for b in my_barcodes:
        if CycleGroup.objects.filter(barcode=b).exists():
            used_barcodes.append(b)

    used_barcodes = list(set(used_barcodes).intersection(set(WANTED_BARCODES)))
    return used_barcodes


# === End: make my barcodes ====================================================

# === Begin: test inconsistent mu ==============================================

def test_inconsistent_voltages(my_barcodes):
    mu = None
    my_files = {}
    inconsistent_mu = False
    for barcode_count, barcode in enumerate(my_barcodes):
        my_files[barcode] = get_files_for_barcode(barcode)
        for f in my_files[barcode]:
            if mu is None:
                mu = f.get_uniform_voltages()
            else:
                new_mu = f.get_uniform_voltages()
                same = True
                if len(new_mu) != len(mu):
                    same = False
                else:
                    if not all(numpy.equal(new_mu, mu)):
                        same = False
                if not same:
                    print(f.filename,
                          " Has a different set of voltages! Please fix. "
                          + "Original voltages: ", mu,
                          " files voltages: ", f.get_uniform_voltages())
                    inconsistent_mu = True

    if inconsistent_mu:
        raise ("some voltages were different. Please fix.")

    return mu


# === End: test inconsistent mu ================================================

# ==== Begin: initial processing ===============================================
'''
neighborhood_data_int.append(
    [min_cyc_index, max_cyc_index, k_count, barcode_count, len(cycles_full), 0])
neighborhood_data_float.append([combined_delta, k[0], k[1]])
'''

NEIGH_INT_MIN_CYC_INDEX = 0
NEIGH_INT_MAX_CYC_INDEX = 1
NEIGH_INT_RATE_INDEX = 2
NEIGH_INT_BARCODE_INDEX = 3
NEIGH_INT_ABSOLUTE_INDEX = 4
NEIGH_INT_VALID_CYC_INDEX = 5
NEIGH_FLOAT_DELTA = 0
NEIGH_FLOAT_CHG_RATE = 1
NEIGH_FLOAT_DCHG_RATE = 2


def initial_processing(my_barcodes, args):
    neighborhood_data_int_full_full = []
    cycles_full = []
    discharge_voltages_full = []
    neighborhood_data_float_full_full = []
    vq_curves_full = []
    vq_curves_mask_full = []

    test_object = {}
    all_data = {}

    voltage_vector = test_inconsistent_voltages(my_barcodes)

    num_barcodes = len(my_barcodes)

    for barcode_count, barcode in enumerate(my_barcodes):
        # each barcode corresponds to a single cell
        test_object[barcode_count] = {}
        # here we load as if it were the original data
        cyc_grp_dict = {}

        for cyc_group in CycleGroup.objects.filter(
                barcode=barcode
        ).order_by('discharging_rate'):
            # cycles are grouped by their charge rates and discharge rates.

            # a cycle group contains many cycles

            # the point of splitting things up this way is that we want to
            # sample each group equally.

            # cyc_grp_dict is a dictionary indexed by charging rate and
            # discharging rate (i.e. cycle group)

            # it containts structured arrays of
            # cycle_number, capacity_vector, vq_curve_mask
            # - cycle_number is the cycle number ;)
            # - capacity_vector is a vector where each element is a
            #   capacity associated with a given voltage.
            #   we also have a voltage vector (called voltage_vector).
            #   If capacity_vector is a capacity vector and
            #   voltage_vector is a voltage vector, then (voltage_vector[i],
            #   capacity_vector[i]) is a voltage-capacity pair.
            # - vq_curve_mask is a vector where each element is a weight
            #   corresponding to a voltage-capacity pair.
            #   this allows us to express the fact that sometimes a given
            #   voltage was not measured, so the capacity is meaningless.
            #   (mask of 0)

            # DONE(harvey): dchg_maximum_voltage is a scalar for a given cycle.
            # So in dtype, you must just write ('dchg_maximum_voltage', 'f4')
            cyc_grp_dict[
                (cyc_group.charging_rate, cyc_group.discharging_rate)
            ] = numpy.array(
                [
                    (
                        cyc.get_offset_cycle(),
                        cyc.get_discharge_curve()[:, 0],
                        cyc.get_discharge_curve()[:, 1],
                        cyc.dchg_maximum_voltage
                    ) for cyc in cyc_group.cycle_set.order_by(
                    'cycle_number') if cyc.valid_cycle],
                dtype=[
                    ('cycle_number', 'f4'),
                    ('capacity_vector', 'f4', len(voltage_vector)),
                    ('vq_curve_mask', 'f4', len(voltage_vector)),
                    ('dchg_maximum_voltage', 'f4')])

        # all_data is indexed by barcode_count
        all_data[barcode_count] = cyc_grp_dict
        max_cap = 0.

        # we want to find to largest capacity measured for this cell.
        # (this is the max over all the cycle groups.)
        for k in cyc_grp_dict.keys():
            max_cap = max(
                max_cap, max(cyc_grp_dict[k]['capacity_vector'][:, 0]))

        print("max_cap:", max_cap)

        neighborhood_data_int_full, neighborhood_data_float_full = [], []

        for k_count, k in enumerate(cyc_grp_dict.keys()):

            # we modify the capacity_vector by normalizing by the max_cap.
            cyc_grp_dict[k]['capacity_vector'] = (
                    1. / max_cap * cyc_grp_dict[k]['capacity_vector'])
            print("k:", k)

            # min_cyc, max_cyc are the range of cycles
            # which exists for this cycle group.
            min_cyc = min(cyc_grp_dict[k]['cycle_number'])
            max_cyc = max(cyc_grp_dict[k]['cycle_number'])

            # Now we will create neighborhoods, which contains the cycles,
            # grouped by proximity.
            # We want to sample neighborhoods equally.

            # neighborhoods have a central cycle and a delta on each side.
            # to a first approximation, we want a delta_cyc = 300, but we have
            # to vary this near the beginning of data and near the end.

            # total_delta gives us an absolute scale
            total_delta = max_cyc - min_cyc
            # delta_cyc is the baseline, at least 200, but up to total_delta/5
            delta_cyc = max(200, int(float(total_delta) / 5.))

            # all_neighborhood_center_cycles are the centers of neighborhoods
            # we will try to create
            all_neighborhood_center_cycles = list(filter(
                lambda x: x > min_cyc - 100,
                range(20, int(max_cyc + 50), 40)))

            neighborhood_data_int, neighborhood_data_float = [], []

            # Now we will check all the tentative neighborhood centers and
            # commit the ones that contain good data to the dataset.
            valid_cycles = 0
            for cyc in all_neighborhood_center_cycles:
                # max_cyc and min_cyc are the limits of existing cycles.
                # delta_up is at least 200, but it can extend up to the limit
                # starting from the current neighborhood center.
                delta_up = max(max_cyc - cyc, 200)
                # same thing going down.
                delta_down = max(cyc - min_cyc, 200)
                # the max symetric interval that fits into the
                # [cyc - delta_down, cyc + delta_up] interval is
                # [cyc - delta_actual, cyc + delta_actual]
                delta_actual = min(delta_up, delta_down)
                # then, we choose the largest interval that fits both
                # [cyc - delta_actual, cyc + delta_actual] and
                # [cyc - delta_cyc, cyc + delta_cyc]
                combined_delta = min(delta_actual, delta_cyc)

                below_cyc = cyc - combined_delta
                above_cyc = cyc + combined_delta

                # mask is a numpy array of True and False of the same length as
                # cyc_grp_dict[k].
                # it is False when the cycle_number falls outside out of the
                # [below_cyc, above_cyc] interval.
                mask = numpy.logical_and(
                    below_cyc <= cyc_grp_dict[k]['cycle_number'],
                    cyc_grp_dict[k]['cycle_number'] <= above_cyc)

                # the indecies into the cyc_grp_dict[k] array which correspond
                # to a mask of True
                all_valid_indecies = numpy.arange(len(mask))[mask]

                # if there are less than 2 valid cycles,
                # just skip that neighborhood.
                if len(all_valid_indecies) < 2:
                    continue  # for now, we just skip.
                # at this point, we know that this neighborhood
                # will be added to the dataset.

                min_cyc_index = all_valid_indecies[0]
                max_cyc_index = all_valid_indecies[-1]

                # now we will add the neighborhood.
                # if no neighborhoods were added, initialize test_object
                if valid_cycles == 0:
                    test_object[barcode_count][k] = []

                test_object[barcode_count][k].append(cyc)
                valid_cycles += 1

                # this commits the neighborhood to the dataset

                # - record the info about the center of the neighborhood
                #   (cycle number, voltage, rate of charge, rate of discharge)
                # - record the relative index (within the cycle group)
                #   of the min cycle, max cycle
                # - record a voltage index, and a cycle group index,
                #   and a cell index
                # - record the absolute index into the table of cycles
                #   (len(cycles_full)).
                # - keep a slot empty for later
                neighborhood_data_int.append(
                    [min_cyc_index, max_cyc_index, k_count, barcode_count,
                     len(cycles_full), 0])
                neighborhood_data_float.append([combined_delta, k[0], k[1]])


            if valid_cycles != 0:
                neighborhood_data_int = numpy.array(
                    neighborhood_data_int, dtype=numpy.int32)
                # the empty slot becomes the count of added neighborhoods, which
                # we use to counterbalance the bias toward longer cycle life.
                neighborhood_data_int[:, NEIGH_INT_VALID_CYC_INDEX] = valid_cycles
                neighborhood_data_float = numpy.array(
                    neighborhood_data_float, dtype=numpy.float32)

                neighborhood_data_int_full.append(neighborhood_data_int)
                neighborhood_data_float_full.append(neighborhood_data_float)

            else:
                print('name: ', barcode)
                print('rates: ', k)

            if len(cycles_full) > 0:

                # a giant array with all the cycle numbers
                cycles_full = numpy.concatenate(
                    (cycles_full, cyc_grp_dict[k]['cycle_number']))

                # a giant array of all the vq_curves
                vq_curves_full = numpy.concatenate(
                    (vq_curves_full, cyc_grp_dict[k]['capacity_vector']))

                # a giant array of all the vq_curves_mask
                vq_curves_mask_full = numpy.concatenate(
                    (vq_curves_mask_full, cyc_grp_dict[k]['vq_curve_mask']))

                discharge_voltages_full = numpy.concatenate((
                    discharge_voltages_full,
                    cyc_grp_dict[k]['dchg_maximum_voltage']))

            else:
                cycles_full = cyc_grp_dict[k]['cycle_number']
                vq_curves_full = cyc_grp_dict[k]['capacity_vector']
                vq_curves_mask_full = cyc_grp_dict[k]['vq_curve_mask']
                discharge_voltages_full = cyc_grp_dict[k]['dchg_maximum_voltage']

        if len(neighborhood_data_int_full) != 0:
            neighborhood_data_int_full_full.append(neighborhood_data_int_full)
            neighborhood_data_float_full_full.append(neighborhood_data_float_full)
        else:
            print("barcode: ", barcode)

    neighborhood_data_int = tf.constant(numpy.concatenate(
        [numpy.concatenate(neighborhood_data_int_full, axis=0)
         for neighborhood_data_int_full in neighborhood_data_int_full_full],
        axis=0))

    # the cycles go from 0 to 6000, but neural networks much prefer normally
    # distributed variables. So we compute mean and variance, and we normalize
    # the cycle numbers.
    cycles = tf.constant(cycles_full)
    cycles_m, cycles_v = tf.nn.moments(cycles, axes=[0])

    cycles_m = cycles_m.numpy()
    cycles_v = cycles_v.numpy()
    # normalization happens here.
    cycles = (cycles - cycles_m) / tf.sqrt(cycles_v)

    # the voltages are also normalized
    voltages = tf.cast(tf.constant(voltage_vector), dtype=tf.float32)
    n_voltages = len(voltage_vector)
    voltages_m, voltages_v = tf.nn.moments(voltages, axes=[0])
    voltages = (voltages - voltages_m) / tf.sqrt(voltages_v)
    vq_curves = tf.constant(vq_curves_full)
    vq_curves_mask = tf.constant(vq_curves_mask_full)

    # DONE: you must define a tensor containing the max_discharge_voltages.
    # Also, you must normalize it by removing the voltage mean and variance.
    max_dchg_voltage_tensor = tf.constant(discharge_voltages_full)
    #dchg_voltages_m, dchg_voltages_v = tf.nn.moments(max_dchg_voltage_tensor, axes=[0])
    dchg_voltages_m = 0.
    dchg_voltages_v = 1.
    max_dchg_voltage_tensor = (
            (max_dchg_voltage_tensor - dchg_voltages_m)
            / tf.sqrt(dchg_voltages_v))

    neighborhood_data_float = (numpy.concatenate(
        [numpy.concatenate(neighborhood_data_float_full, axis=0)
         for neighborhood_data_float_full
         in neighborhood_data_float_full_full],
        axis=0))

    # note: this step is to convert the delta_cycles of each
    # neighborhoods to the normalized units (you divide by standard deviation).
    neighborhood_data_float[:, NEIGH_FLOAT_DELTA] = (
            (neighborhood_data_float[:, NEIGH_FLOAT_DELTA]) / numpy.sqrt(cycles_v))

    neighborhood_data_float = tf.constant(neighborhood_data_float)

    batch_size = args['batch_size']
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        train_ds_ = tf.data.Dataset.from_tensor_slices(
            (neighborhood_data_int, neighborhood_data_float)
        ).repeat(2).shuffle(100000).batch(batch_size)

        train_ds = mirrored_strategy.experimental_distribute_dataset(
            train_ds_)

        degradation_model = DegradationModel(
            num_keys=num_barcodes, width=args['width'], depth=args['depth'])

        optimizer = tf.keras.optimizers.Adam()

    return {
        "mirrored_strategy": mirrored_strategy,
        "train_ds": train_ds,
        "cycles_m": cycles_m,
        "cycles_v": cycles_v,
        "voltages_m": dchg_voltages_m,
        "voltages_v": dchg_voltages_v,

        "cycles": cycles,
        "voltages": voltages,
        "n_voltages": n_voltages,
        "vq_curves": vq_curves,
        "vq_curves_mask": vq_curves_mask,
        "degradation_model": degradation_model,
        "optimizer": optimizer,
        "test_object": test_object,
        "all_data": all_data,
        "voltage_vector": voltage_vector,
        # DONE: here you must return the tensor, not the numpy array.
        "max_dchg_voltage_tensor": max_dchg_voltage_tensor
    }


# === End: initial processing ==================================================

# === Begin: train =============================================================

def train_and_evaluate(init_returns, my_barcodes, args):
    mirrored_strategy = init_returns["mirrored_strategy"]
    train_ds = init_returns["train_ds"]
    cycles_m = init_returns["cycles_m"]
    cycles_v = init_returns["cycles_v"]
    voltages_m = init_returns["voltages_m"]
    voltages_v = init_returns["voltages_v"]

    cycles = init_returns["cycles"]
    voltages = init_returns["voltages"]
    n_voltages = init_returns["n_voltages"]
    vq_curves = init_returns["vq_curves"]
    vq_curves_mask = init_returns["vq_curves_mask"]
    degradation_model = init_returns["degradation_model"]
    optimizer = init_returns["optimizer"]
    test_object = init_returns["test_object"]
    all_data = init_returns["all_data"]
    voltage_vector = init_returns["voltage_vector"]
    max_dchg_voltage_tensor = init_returns["max_dchg_voltage_tensor"]

    EPOCHS = 100000
    count = 0

    template = 'Epoch {}, Count {}'

    with mirrored_strategy.scope():
        for epoch in range(EPOCHS):
            for neighborhood_int, neighborhood_float in train_ds:
                count += 1

                train_step_params = {
                    "degradation_model": degradation_model,
                    "neighborhood_float": neighborhood_float,
                    "neighborhood_int": neighborhood_int,
                    "cycles": cycles,
                    "vq_curves": vq_curves,
                    "vq_curves_mask": vq_curves_mask,
                    "voltages": voltages,
                    "n_voltages": n_voltages,
                    "optimizer": optimizer,
                    "max_dchg_voltage_tensor": max_dchg_voltage_tensor
                }

                dist_train_step(mirrored_strategy, train_step_params, args)

                if count != 0:
                    if (count % args['print_loss_every']) == 0:
                        print(template.format(epoch + 1, count, ))

                    plot_params = {
                        "my_barcodes": my_barcodes,
                        "count": count,
                        "args": args,
                        "cycles_m": cycles_m,
                        "cycles_v": cycles_v,
                        "voltages_m": voltages_m,
                        "voltages_v": voltages_v,

                        "degradation_model": degradation_model,
                        "voltages": voltages,
                        "test_object": test_object,
                        "all_data": all_data,
                    }

                    if (count % args['visualize_fit_every']) == 0:
                        plot_capacity(plot_params)

                    if (count % args['visualize_vq_every']) == 0 and count != 0:
                        plot_vq(plot_params, voltage_vector)

                if count == args['stop_count']:
                    return


# === End: train ===============================================================

# === Begin: train step ========================================================

def train_step(params, args):
    degradation_model = params["degradation_model"]
    neighborhood_float = params["neighborhood_float"]
    neighborhood_int = params["neighborhood_int"]
    cycles = params["cycles"]
    vq_curves = params["vq_curves"]
    vq_curves_mask = params["vq_curves_mask"]
    voltages = params["voltages"]
    n_voltages = params["n_voltages"]
    optimizer = params["optimizer"]
    max_dchg_voltage_tensor = params["max_dchg_voltage_tensor"]

    print(Colour.BLUE + "train step" + Colour.END)
    # need to split the range ourselves.
    batch_size2 = neighborhood_int.shape[0]

    # we will find the actual cycle number by interpolation.
    # we will then offset the cycle number by the delta * randomness.

    # we offset the center cycles so that we make sure that we never exactly
    # evaluate the model at the exact same cycle.
    center_cycle_offsets = tf.random.uniform(
        [batch_size2], minval=-1., maxval=1., dtype=tf.float32)

    # if you have the minimum cycle and maximum cycle for a neighborhood,
    # you can sample cycles from this neighborhood.
    # by sampling real numbers x from [0,1] and just compute
    # min_cyc*(1.-x) + max_cyc*x.
    # but here we do this computation in index space
    # and then gather the cycle numbers and vq curves.
    cycle_indecies_lerp = tf.random.uniform(
        [batch_size2], minval=0., maxval=1., dtype=tf.float32)

    cycle_indecies = tf.cast(
        (1. - cycle_indecies_lerp) * tf.cast(
            neighborhood_int[:, NEIGH_INT_MIN_CYC_INDEX]
            + neighborhood_int[:, NEIGH_INT_ABSOLUTE_INDEX],
            tf.float32)
        + (cycle_indecies_lerp) * tf.cast(
            neighborhood_int[:, NEIGH_INT_MAX_CYC_INDEX]
            + neighborhood_int[:, NEIGH_INT_ABSOLUTE_INDEX],
            tf.float32),
        tf.int32)

    # DONE: this is good, but you needed to pass in the tensor, not the numpy array.
    # also, you must ensure that this tensor has just one dimension to match what the model outputs.
    measured_discharge_voltages = tf.reshape(tf.gather(
        max_dchg_voltage_tensor,
        indices=cycle_indecies,
        axis=0), [-1])

    measured_cycles = tf.gather(
        cycles, indices=cycle_indecies, axis=0)

    model_evaluation_cycles = (
            measured_cycles
            + center_cycle_offsets * neighborhood_float[:, NEIGH_FLOAT_DELTA])

    cap = tf.gather(vq_curves, indices=cycle_indecies)

    ws = tf.gather(vq_curves_mask, indices=cycle_indecies)

    # question: What is this used for?
    # note: see in the loss function! it is a weight for the prediction error.
    # the more measurements you have for a cell, then less each one is worth.
    # So that in expectation, you "care" about every cell equally.
    ws2_discharge_voltages = 1. / (tf.cast(
        neighborhood_int[:, NEIGH_INT_VALID_CYC_INDEX], tf.float32))

    ws2 = tf.tile(
        tf.reshape(
            1. / (tf.cast(
                neighborhood_int[:, NEIGH_INT_VALID_CYC_INDEX],
                tf.float32)),
            [batch_size2, 1]),
        [1, n_voltages])

    # the indecies are referring to the cell indecies
    indecies = neighborhood_int[:, NEIGH_INT_BARCODE_INDEX]

    my_centers = tf.concat(
        (
            tf.expand_dims(model_evaluation_cycles, axis=1),
            neighborhood_float[:, 1:]
        ),
        axis=1
    )



    with tf.GradientTape() as tape:
        (predicted_cap, predicted_vol, mean, log_sig, cap_derivatives,vol_derivatives
         ) = degradation_model(
            (my_centers, indecies, measured_cycles, voltages),
            training=True)

        # question: looks like `res` changes size and changes into a
        # different dimension than `predicted_voltages`

        # note: yes. predictions_cap is 2D (batch_size, n_voltages)
        # whereas predicted_voltages should be 1D (batch_size).

        loss = (
                tf.reduce_mean(ws2 * ws * tf.square(cap - predicted_cap))
                / (1e-10 + tf.reduce_mean(ws2 * ws))
                + tf.reduce_mean(ws2_discharge_voltages
                                 * tf.square(measured_discharge_voltages-predicted_vol))
                / (1e-10 + tf.reduce_mean(ws2_discharge_voltages))
                + args['kl_coeff'] * tf.reduce_mean(
            0.5 * (tf.exp(log_sig) + tf.square(mean) - 1. - log_sig))
                # TODO(harvey): figure out what way the max discharge voltage should vary and enforce monotonicity in predictions :)
                + args['mono_coeff'] * (
                        tf.reduce_mean(tf.nn.relu(-cap))  #This penalizes negative capacities
                        + tf.reduce_mean(tf.nn.relu(cap_derivatives['dCyc']))
                        + tf.reduce_mean(tf.nn.relu(cap_derivatives['dOthers'])))
                # TODO(harvey): Do you understand what this is doing? it is enforcing smoothness,
                # but it cares much less when the predictions accelerate downwards.
                # you must figure out the equivalent for voltage predictions and enforce it.
                + args['smooth_coeff'] * (
                        tf.reduce_mean(tf.square(tf.nn.relu(cap_derivatives['d2Cyc']))
                                       + 0.02 * tf.square(tf.nn.relu(-cap_derivatives['d2Cyc'])))
                        + tf.reduce_mean(tf.square(tf.nn.relu(cap_derivatives['d2Others']))
                                         + 0.02 * tf.square(tf.nn.relu(-cap_derivatives['d2Others']))))
                # TODO(harvey): right now, the capacity function depends well on cell features.
                # we also want the voltage function to depend well on cell features. Do you know how to do it?
                + args['const_f_coeff'] * tf.reduce_mean(tf.square(cap_derivatives['dFeatures']))
                + args['smooth_f_coeff'] * tf.reduce_mean(
            tf.square(cap_derivatives['d2Features'])))
        cprint(Colour.RED, "LOSS calculated")
    # train_loss(loss)
    gradients = tape.gradient(loss, degradation_model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, degradation_model.trainable_variables))


# === End : train step =========================================================

@tf.function
def dist_train_step(mirrored_strategy, train_step_params, args):
    mirrored_strategy.experimental_run_v2(
        train_step, args=(train_step_params, args))

def get_nearest_point(xys, y):
    best = xys[0,:]
    best_distance = (best[1]-y)**2.
    for i in range(len(xys)):
        new_distance =(xys[i,1]-y)**2.
        if best_distance > new_distance:
            best_distance = new_distance
            best = xys[i,:]

    return best

def plot_vq(plot_params, voltage_vector):
    my_barcodes = plot_params["my_barcodes"]
    count = plot_params["count"]
    args = plot_params["args"]
    cycles_m = plot_params["cycles_m"]
    cycles_v = plot_params["cycles_v"]
    voltages_m = plot_params["voltages_m"]
    voltages_v = plot_params["voltages_v"]
    degradation_model = plot_params["degradation_model"]
    voltages = plot_params["voltages"]
    test_object = plot_params["test_object"]
    all_data = plot_params["all_data"]

    print(Colour.BLUE + "plot vq" + Colour.END)
    for barcode_count, barcode in enumerate(my_barcodes):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        colors = ['k', 'r', 'b', 'g', 'm', 'c']
        for k_count, k in enumerate(test_object[barcode_count].keys()):

            n_samples = len(all_data[barcode_count][k]['capacity_vector'])
            for vq_count, vq in enumerate(all_data[barcode_count][k]['capacity_vector']):
                ax.plot(vq, voltage_vector, c=colors[k_count])

                if vq_count % int(n_samples/10) == 0:
                    fused_vector = numpy.stack([vq, voltage_vector], axis =1 )
                    target_voltage = all_data[barcode_count][k]['dchg_maximum_voltage'][vq_count]
                    best_point = get_nearest_point(fused_vector, target_voltage)
                    ax.scatter([best_point[0]], [best_point[1]], c=colors[k_count], marker='o', s=100)

        ax = fig.add_subplot(1, 2, 2)
        colors = [(1., 1., 1.), (1., 0., 0.), (0., 0., 1.),
                  (0., 1., 0.), (1., 0., 1.), (0., 1., 1.)]

        for k_count, k in enumerate(
                test_object[barcode_count].keys()):
            # TODO(harvey): shouldn't that be a parameter that you pass in?
            cycles = [0, 2000, 4000, 6000]
            for i, cyc in enumerate(cycles):
                cycle = ((float(cyc) - cycles_m) / tf.sqrt(cycles_v))
                predicted_cap, predicted_vol = test_all_voltages(
                    cycle, k, barcode_count, degradation_model, voltages)

                mult = (i + 4) / (len(cycles) + 5)
                ax.plot(predicted_cap, voltage_vector, c=(
                    mult * colors[k_count][0],
                    mult * colors[k_count][1],
                    mult * colors[k_count][2]))
                fused_vector = numpy.stack([predicted_cap, voltage_vector], axis=1)
                target_voltage = (predicted_vol[0] * math.sqrt(voltages_v))+ voltages_m
                best_point = get_nearest_point(fused_vector, target_voltage)
                ax.scatter([best_point[0]], [best_point[1]], marker='x', s=100, c=(
                    mult * colors[k_count][0],
                    mult * colors[k_count][1],
                    mult * colors[k_count][2]))


        plt.savefig(os.path.join(
            args['path_to_plots'],
            'VQ_{}_Count_{}.png'.format(barcode, count)))
        plt.close(fig)


def plot_test_rate_voltage(plot_params):
    my_barcodes = plot_params["my_barcodes"]
    count = plot_params["count"]
    args = plot_params["args"]
    cycles_m = plot_params["cycles_m"]
    cycles_v = plot_params["cycles_v"]
    voltages_m = plot_params["voltages_m"]
    voltages_v = plot_params["voltages_v"]
    degradation_model = plot_params["degradation_model"]
    voltages = plot_params["voltages"]
    test_object = plot_params["test_object"]
    all_data = plot_params["all_data"]

    print(Colour.BLUE + "plot capacity" + Colour.END)
    for barcode_count, barcode in enumerate(my_barcodes):
        results = []
        for k in [[0.1, x/10.] for x in range(40)]:
            _, predicted_vol = test_single_voltage([0.], voltages[0],
                                                           k, barcode_count, degradation_model)
            results.append([k[1], predicted_vol])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        results = numpy.array(results)
        ax.scatter(results[:,0],
                   results[:,1])

        plt.savefig(os.path.join(
            args['path_to_plots'],
            'Cap_{}_Count_{}.png'.format(barcode, count)))
        plt.close(fig)

def plot_capacity(plot_params):
    my_barcodes = plot_params["my_barcodes"]
    count = plot_params["count"]
    args = plot_params["args"]
    cycles_m = plot_params["cycles_m"]
    cycles_v = plot_params["cycles_v"]
    voltages_m = plot_params["voltages_m"]
    voltages_v = plot_params["voltages_v"]
    degradation_model = plot_params["degradation_model"]
    voltages = plot_params["voltages"]
    test_object = plot_params["test_object"]
    all_data = plot_params["all_data"]

    print(Colour.BLUE + "plot capacity" + Colour.END)
    for barcode_count, barcode in enumerate(my_barcodes):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        colors = ['k', 'r', 'b', 'g', 'm', 'c']
        for k_count, k in enumerate(
                test_object[barcode_count].keys()):

            ax1.scatter(
                all_data[barcode_count][k]['cycle_number'],
                all_data[barcode_count][k]['capacity_vector'][:, 0],
                c=colors[k_count])

            ax2.scatter(
                all_data[barcode_count][k]['cycle_number'],
                all_data[barcode_count][k]['dchg_maximum_voltage'],
                c=colors[k_count])


            for cyc_i in [0, -1]:
                cyc = test_object[barcode_count][k][cyc_i]
                ax1.axvline(x=cyc, color=colors[k_count],
                            linestyle='--')

        for k_count, k in enumerate(
                test_object[barcode_count].keys()):
            cycles = test_object[barcode_count][k]
            min_c = min(cycles)
            max_c = max(cycles)
            cycles = [float(min_c) + float(max_c - min_c)
                      * x for x in numpy.arange(0., 1.1, 0.1)]

            my_cycles = [(cyc - cycles_m)
                         / tf.sqrt(cycles_v) for cyc in cycles]
            # TODO(harvey): make test_single_voltage return both the capacity predictions as well as the voltage predictions
            predicted_cap, predicted_vol = test_single_voltage(my_cycles, voltages[0],
                                        k, barcode_count, degradation_model)
            ax1.plot(cycles, predicted_cap, c=colors[k_count])
            # TODO(harvey): plot cycles, voltages.
            ax2.plot(cycles, (predicted_vol*math.sqrt(voltages_v))+voltages_m, c=colors[k_count])

        plt.savefig(os.path.join(
            args['path_to_plots'],
            'Cap_{}_Count_{}.png'.format(barcode, count)))
        plt.close(fig)


def test_all_voltages(cycle, k, barcode_count, degradation_model, voltages):
    centers = tf.expand_dims(tf.concat(
        (tf.expand_dims(cycle, axis=0), k), axis=0), axis=0)
    indecies = tf.reshape(barcode_count, [1])
    measured_cycles = tf.reshape(cycle, [1, 1])
    predicted_cap, predicted_vol = degradation_model(
        (centers, indecies, measured_cycles, voltages), training=False)

    predicted_cap = tf.reshape(predicted_cap, shape=[-1])
    return predicted_cap, predicted_vol


def test_single_voltage(cycles, v, k, barcode_count, degradation_model):
    centers = tf.concat(
        (
            tf.expand_dims(cycles, axis=1),
            tf.tile(tf.expand_dims(k, axis=0), [len(cycles), 1])
        ),
        axis=1)
    indecies = tf.tile(tf.expand_dims(barcode_count, axis=0), [len(cycles)])
    measured_cycles = tf.expand_dims(cycles,
                                     axis=1)
    predicted_cap, predicted_vol = degradation_model(
        (centers, indecies, measured_cycles, tf.expand_dims(v, axis=0)),
        training=False)
    predicted_cap = tf.reshape(predicted_cap, shape=[len(cycles)])

    return predicted_cap, predicted_vol


def fitting_level0(args):
    my_barcodes = make_my_barcodes(args)  # list

    # plot_initial(my_barcodes, args)

    train_and_evaluate(initial_processing(my_barcodes, args), my_barcodes, args)


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path_to_plots', required=True)
        parser.add_argument('--kl_coeff', type=float, default=0.00001)
        parser.add_argument('--mono_coeff', type=float, default=.005)
        parser.add_argument('--smooth_coeff', type=float, default=.005)
        parser.add_argument('--const_f_coeff', type=float, default=.0)
        parser.add_argument('--smooth_f_coeff', type=float, default=.01)
        parser.add_argument('--depth', type=int, default=2)
        parser.add_argument('--width', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=2 * 16)
        parser.add_argument('--print_loss_every', type=int, default=1000)
        parser.add_argument(
            '--visualize_fit_every', type=int, default=5000)
        parser.add_argument(
            '--visualize_vq_every', type=int, default=5000)

        parser.add_argument('--stop_count', type=int, default=20000)

    def handle(self, *args, **options):
        fitting_level0(options)
