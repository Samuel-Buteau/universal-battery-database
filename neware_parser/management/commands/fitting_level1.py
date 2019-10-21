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

# TODO: a tool to delete cycles by looking at the total capacity vs cycle number
# TODO: control derivatives w.r.t. voltage and cycle.

NUM_CYCLES = 5


class Colour:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


################################################################################
# Begin: Dictionary Layer
################################################################################

# stores cell features
# key: index
# value: feature (matrix)
class DictionaryLayer(Layer):

    # === Begin: Dictionary Layer | init =======================================

    def __init__(self, num_features, num_keys):
        super(DictionaryLayer, self).__init__()
        self.num_features = num_features
        self.num_keys = num_keys
        self.kernel = self.add_weight(
            "kernel", shape=[self.num_keys, self.num_features * 2])

    # === End: Dictionary Layer | init =========================================

    # === Begin: Dictionary Layer | call =======================================

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

    # === End: Dictionary Layer | call =========================================


################################################################################
# Begin: Degradation Model
################################################################################

class DegradationModel(Model):

    def __init__(self, num_keys, depth, width):
        super(DegradationModel, self).__init__()

        self.dense_initial = Dense(width, activation='relu', use_bias=True,
                                   bias_initializer='zeros')
        self.dense = []

        for _ in range(depth):
            self.dense.append(
                Dense(width, activation='relu', use_bias=True,
                      bias_initializer='zeros'))

        self.dense_final = Dense(1, activation=None, use_bias=True,
                                 bias_initializer='zeros',
                                 kernel_initializer='zeros')

        self.dictionary = DictionaryLayer(num_features=width,
                                          num_keys=num_keys)

        self.width = width
        self.num_keys = num_keys
    def call(self, x, training=False):
        """
        (centers, indecies, xs, mu) = x

        centers : batch of [cyc, k[0], k[1]]
        indecies : batch of index
        xs: batch of [cyc, cyc, ...]
        mu: this is the vector of voltages. It doesn't change across the batch.
        :param x:
        :param training:
        :return:
        """
        centers = x[0]
        indecies = x[1]
        xs = x[2]
        mu = x[3]

        features, mean, log_sig = self.dictionary(indecies, training=training)

        cycles = centers[:, 0:1]
        others = centers[:, 1:]

        # basically, we need to duplicate the cycles and others for all the voltages.
        # now dimentions are [batch, voltages, features]
        cycles_tiled = tf.tile(tf.expand_dims(cycles, axis=1), [1]+mu.shape[0:1]+[1])
        others_tiled = tf.tile(tf.expand_dims(others, axis=1), [1]+mu.shape[0:1]+[1])
        features_tiled = tf.tile(tf.expand_dims(features, axis=1), [1]+mu.shape[0:1]+[1])
        mu_tiled = tf.tile(tf.expand_dims(tf.expand_dims(mu, axis=1), axis=0), cycles.shape[0:1]+  [1, 1])
        others_tiled_2 = tf.concat(
            (others_tiled,
             mu_tiled,
             ),
            axis = 2
         )
        cycles_flat = tf.reshape(cycles_tiled, [-1, 1])
        others_flat = tf.reshape(others_tiled_2, [-1, 3])
        features_flat = tf.reshape(features_tiled, [-1, self.width])
        # now every dimension works for concatenation.
        if training:

            with tf.GradientTape(persistent=True) as tape3:

                tape3.watch(cycles_flat)
                tape3.watch(others_flat)
                tape3.watch(features_flat)

                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch(cycles_flat)
                    tape2.watch(others_flat)
                    tape2.watch(features_flat)

                    # cat two lists
                    centers2 = tf.concat(
                        (
                            # reajust the cycles
                            cycles_flat* (1e-10 + tf.exp(-features_flat[:, 0:1])),
                            others_flat,
                            features_flat[:, 1:]),
                        axis=1)


                    centers2 = self.dense_initial(centers2)

                    for d in self.dense:
                        centers2 = d(centers2)

                    coeffs = self.dense_final(centers2)
                    res1 = tf.reshape(coeffs, shape=[-1, 1])

                dresdcyc = tape2.batch_jacobian(
                    source=cycles_flat, target=res1)[:, 0, :]
                dresdothers = tape2.batch_jacobian(
                    source=others_flat, target=res1)[:, 0, :]
                dresdfeatures = tape2.batch_jacobian(
                    source=features_flat, target=res1)[:, 0, :]

                del tape2

            d2resdcyc = tape3.batch_jacobian(
                source=cycles_flat, target=dresdcyc)[:, 0, :]
            d2resdothers = tape3.batch_jacobian(
                source=others_flat, target=dresdothers)
            d2resdfeatures = tape3.batch_jacobian(
                source=features_flat, target=dresdfeatures)

            del tape3

            var_cyc = (xs - cycles)
            var_cyc2 = tf.square(var_cyc)

            cap = tf.reshape(res1, [-1, mu.shape[0], 1])
            dcapdcyc = tf.reshape(dresdcyc, [-1, mu.shape[0], 1])
            d2capdcyc = tf.reshape(d2resdcyc, [-1, mu.shape[0], 1])

            var_cyc_reshape= tf.expand_dims(var_cyc, axis=1)
            var_cyc2_reshape= tf.expand_dims(var_cyc2, axis=1)
            res = (cap +
                   dcapdcyc * var_cyc_reshape +
                   d2capdcyc * var_cyc2_reshape)

            print(Colour.BLUE + "Degradation Model - call (training)"
                  + Colour.END)

            return (res, mean, log_sig, coeffs, dresdcyc, d2resdcyc,
                    dresdothers, d2resdothers, dresdfeatures, d2resdfeatures)

        else:
            centers2 = tf.concat(
                (
                    # reajust the cycles
                    cycles_flat * (1e-10 + tf.exp(-features_flat[:, 0:1])),
                    others_flat,
                    features_flat[:, 1:]),
                axis=1)

            centers2 = self.dense_initial(centers2)

            for d in self.dense:
                centers2 = d(centers2)

            coeffs = self.dense_final(centers2)


            return tf.reshape(coeffs, shape=[-1, mu.shape[0]])


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
wanted_barcodes = [
    83220,
    83083,
    82012,
    82993,
    82410,
    82311,
    82306,
    81625,
    57706,
]
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

    used_barcodes = list(set(used_barcodes).intersection(set(wanted_barcodes)))
    return used_barcodes


# === End: make my barcodes ====================================================

# === Begin: test inconsistent mu ==============================================

def test_inconsistent_mu(my_barcodes):
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

def initial_processing(my_barcodes, args):
    neighborhood_data_int_full_full = []
    cycles_full = []
    neighborhood_data_float_full_full = []
    vq_curves_full = []
    vq_curves_mask_full = []

    test_object = {}
    all_data = {}

    mu = test_inconsistent_mu(my_barcodes)

    num_barcodes = len(my_barcodes)

    for barcode_count, barcode in enumerate(my_barcodes):
        # each barcode corresponds to a single cell
        test_object[barcode_count] = {}
        # here we load as if it were the original data
        data_da = {}

        for cyc_group in CycleGroup.objects.filter(
                barcode=barcode
        ).order_by('discharging_rate'):
            # cycles are grouped by their charge rates and discharge rates.
            # a cycle group contains many cycles
            # the point of splitting things up this way is that we want to sample each group equally.
            # data_da is a dictionary indexed by charging rate and discharging rate (i.e. cycle group)
            # it containts structured arrays of cycle_number, vq_curve, vq_curve_mask
            # - cycle_number is the cycle number ;)
            # - vq_curve is a vector where each element is a capacity associated with a given voltage.
            #   we also have a voltage vector (called mu). If vq_curve is a capacity vector and
            #   mu is a voltage vector, then (mu[i], vq_curve[i]) is a voltage-capacity pair.
            # - vq_curve_mask is a vector where each element is a weight corresponding to a voltage-capacity pair.
            #   this allows us to express the fact that sometimes a given voltage was not measured, so the capacity is meaningless. (mask of 0)
            data_da[
                (cyc_group.charging_rate, cyc_group.discharging_rate)
            ] = numpy.array(
                [(cyc.get_offset_cycle(), cyc.get_discharge_curve()[:, 0],
                  cyc.get_discharge_curve()[:, 1])
                 for cyc in cyc_group.cycle_set.order_by(
                    'cycle_number') if cyc.valid_cycle],
                dtype=[('cycle_number', 'f4'),
                       ('vq_curve', 'f4', len(mu)),
                       ('vq_curve_mask', 'f4', len(mu))])

        # all_data is indexed by barcode_count
        all_data[barcode_count] = data_da
        max_cap = 0.

        # we want to find to largest capacity measured for this cell. (this is the max over all the cycle groups.)
        for k in data_da.keys():
            max_cap = max(max_cap, max(data_da[k]['vq_curve'][:, 0]))

        print("max_cap:", max_cap)

        neighborhood_data_int_full, neighborhood_data_float_full = [], []

        for k_count, k in enumerate(data_da.keys()):
            # we modify the vq_curve by normalizing by the max_cap.
            data_da[k]['vq_curve'] = 1. / max_cap * data_da[k]['vq_curve']
            print("k:", k)

            # min_cyc, max_cyc are the range of cycles which exists for this cycle group.
            min_cyc = min(data_da[k]['cycle_number'])
            max_cyc = max(data_da[k]['cycle_number'])

            # Now we will create neighborhoods, which contains the cycles, grouped by proximity.
            # We want to sample neighborhoods equally.

            # neighborhoods have a central cycle and a delta on each side.
            # to a first approximation, we want a delta_cyc = 300, but we have to vary this near the beginning of data and near the end.

            # total_delta gives us an absolute scale
            total_delta = max_cyc - min_cyc
            # delta_cyc is the baseline, at least 200, but up to total_delta/5
            delta_cyc = max(200, int(float(total_delta) / 5.))

            # all_neighborhood_center_cycles are the centers of neighborhoods we will try to create
            all_neighborhood_center_cycles = list(filter(
                lambda x: x > min_cyc - 100,
                range(20, int(max_cyc + 50), 40)))


            neighborhood_data_int, neighborhood_data_float = [], []

            # Now we will check all the tentative neighborhood centers and commit the ones that contain good data to the dataset.
            valid_cycles = 0
            for cyc in all_neighborhood_center_cycles:
                # max_cyc and min_cyc are the limits of existing cycles.
                # delta_up is at least 200, but it can extend up to the limit starting from the current neighborhood center.
                delta_up = max(max_cyc - cyc, 200)
                # same thing going down.
                delta_down = max(cyc - min_cyc, 200)
                # the max symetric interval that fits into the [cyc - delta_down, cyc + delta_up] interval is
                # [cyc - delta_actual, cyc + delta_actual]
                delta_actual = min(delta_up, delta_down)
                # then, we choose the largest interval that fits both
                # [cyc - delta_actual, cyc + delta_actual] and [cyc - delta_cyc, cyc + delta_cyc]
                combined_delta = min(delta_actual, delta_cyc)

                below_cyc = cyc - combined_delta
                above_cyc = cyc + combined_delta

                # mask is a numpy array of True and False of the same length as data_da[k].
                # it is False when the cycle_number falls outside out of the [below_cyc, above_cyc] interval.
                mask = numpy.logical_and(
                    below_cyc <= data_da[k]['cycle_number'],
                    data_da[k]['cycle_number'] <= above_cyc)

                # the indecies into the data_da[k] array which correspond to a mask of True
                all_valid_indecies = numpy.arange(len(mask))[mask]

                # if there are less than 2 valid cycles, just skip that neighborhood.
                if len(all_valid_indecies) < 2:
                    continue  # for now, we just skip.
                # at this point, we know that this neighborhood will be added to the dataset.

                min_cyc_index = all_valid_indecies[0]
                max_cyc_index = all_valid_indecies[-1]

                # now we will add the neighborhood.
                # if no neighborhoods were added, initialize test_object
                if valid_cycles == 0:
                    test_object[barcode_count][k] = []

                test_object[barcode_count][k].append(cyc)
                valid_cycles += 1

                # this commits the neighborhood to the dataset

                # record the info about the center of the neighborhood (cycle number, voltage, rate of charge, rate of discharge)
                # record the relative index (within the cycle group) of the min cycle, max cycle
                # record a voltage index, and a cycle group index, and a cell index
                # record the absolute index into the table of cycles (len(cycles_full)).
                # keep a slot empty for later
                neighborhood_data_int.append([min_cyc_index, max_cyc_index, k_count, barcode_count,
                                   len(cycles_full), 0])
                neighborhood_data_float.append([cyc, k[0], k[1]])

            if valid_cycles != 0:
                neighborhood_data_int = numpy.array(neighborhood_data_int, dtype=numpy.int32)
                # the empty slot becomes the count of added neighborhoods,
                # which we use to counterbalance the bias toward longer cycle life.
                neighborhood_data_int[:, -1] = valid_cycles
                neighborhood_data_float = numpy.array(neighborhood_data_float,
                                            dtype=numpy.float32)

                neighborhood_data_int_full.append(neighborhood_data_int)
                neighborhood_data_float_full.append(neighborhood_data_float)

            else:
                print('name: ', barcode)
                print('rates: ', k)

            if len(cycles_full) > 0:
                # cycles_full is a giant array with all the cycle numbers
                cycles_full = numpy.concatenate(
                    (cycles_full, data_da[k]['cycle_number']))
                # vq_curves_full is a giant array of all the vq_curves ...
                vq_curves_full = numpy.concatenate(
                    (vq_curves_full, data_da[k]['vq_curve']))
                # vq_curves_mask_full is a giant array of all the vq_curves_mask ...
                vq_curves_mask_full = numpy.concatenate(
                    (vq_curves_mask_full, data_da[k]['vq_curve_mask']))
            else:
                cycles_full = data_da[k]['cycle_number']
                vq_curves_full = data_da[k]['vq_curve']
                vq_curves_mask_full = data_da[k]['vq_curve_mask']

        if len(neighborhood_data_int_full) != 0:
            neighborhood_data_int_full_full.append(neighborhood_data_int_full)
            neighborhood_data_float_full_full.append(neighborhood_data_float_full)
        else:
            print("barcode: ", barcode)

    neighborhood_data_int = tf.constant(numpy.concatenate(
        [numpy.concatenate(neighborhood_data_int_full, axis=0)
         for neighborhood_data_int_full in neighborhood_data_int_full_full],
        axis=0))

    # the cycles go from 0 to 6000, but neural networks much prefer normally distributed
    # variables. So we compute mean and variance, and we normalize the cycle numbers.
    cycles = tf.constant(cycles_full)
    cycles_m, cycles_v = tf.nn.moments(cycles, axes=[0])

    cycles_m = cycles_m.numpy()
    cycles_v = cycles_v.numpy()
    # normalization happens here.
    cycles = (cycles - cycles_m) / tf.sqrt(cycles_v)

    # the voltages are also normalized
    voltages = tf.cast(tf.constant(mu), dtype=tf.float32)
    n_voltages = len(mu)
    voltages_m, voltages_v = tf.nn.moments(voltages, axes=[0])
    voltages = (voltages - voltages_m) / tf.sqrt(voltages_v)
    vq_curves = tf.constant(vq_curves_full)
    vq_curves_mask = tf.constant(vq_curves_mask_full)
    neighborhood_data_float = (numpy.concatenate(
        [numpy.concatenate(
            neighborhood_data_float_full,
            axis=0) for neighborhood_data_float_full in neighborhood_data_float_full_full],
        axis=0))

    # the centers of neighborhoods are normalized in the same way.
    neighborhood_data_float[:, 0] = ((neighborhood_data_float[:, 0] - cycles_m)
                           / numpy.sqrt(cycles_v))

    neighborhood_data_float = tf.constant(neighborhood_data_float)

    batch_size = 16 * 32
    num_cycles = NUM_CYCLES
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        train_ds_ = tf.data.Dataset.from_tensor_slices(
            (neighborhood_data_int, neighborhood_data_float)
        ).repeat(2).shuffle(100000).batch(batch_size)

        train_ds = mirrored_strategy.experimental_distribute_dataset(
            train_ds_)

        degradation_model = DegradationModel(num_keys=num_barcodes, width=args['width'],
                                             depth=args['depth'])

        optimizer = tf.keras.optimizers.Adam()

    return (mirrored_strategy, train_ds, cycles_m, cycles_v, num_cycles,
            cycles, voltages, n_voltages, vq_curves, vq_curves_mask, degradation_model,
            optimizer, test_object, all_data, mu)


# === End: initial processing ==================================================

# === Begin: train =============================================================

def train(my_barcodes, mirrored_strategy, train_ds, degradation_model,
          cycles, num_cycles, cycles_m, cycles_v, test_object, all_data,
          mu, vq_curves, vq_curves_mask, optimizer, voltages, n_voltages, args):
    EPOCHS = 100000
    count = 0

    template = 'Epoch {}, Count {}'

    with mirrored_strategy.scope():
        for epoch in range(EPOCHS):
            for neighborhood_int, neighborhood_float in train_ds:
                count += 1
                dist_train_step(neighborhood_float, neighborhood_int, mirrored_strategy, cycles_v,
                                num_cycles, cycles, vq_curves, vq_curves_mask,
                                voltages,
                                n_voltages,
                                degradation_model, args, optimizer)

                if (count % args['print_loss_every']) == 0 and count != 0:
                    print(template.format(epoch + 1, count, ))
                if (count % args['visualize_fit_every']) == 0 and count != 0:
                    plot_capacity(count, args, my_barcodes, test_object,
                                  all_data, cycles_m, cycles_v, voltages, degradation_model)
                if (count % args['visualize_vq_every']) == 0 and count != 0:
                    plot_vq(cycles_m, cycles_v, degradation_model, voltages, mu,
                            count, args, my_barcodes, test_object, all_data)

                if count == args['stop_count']:
                    return


# === End: train ===============================================================

# === Begin: train step ========================================================

def train_step(centers, index_d, cycles_v, num_cycles, cycles, vq_curves,
               vq_curves_mask, voltages, n_voltages, degradation_model, args, optimizer):
    print(Colour.BLUE + "train step" + Colour.END)
    # need to split the range ourselves.
    batch_size2 = index_d.shape[0]

    # we offset the center cycles so that we make sure that we never exactly evaluate the model at the exact same cycle.
    center_cycle_offsets = tf.random.uniform(
        [batch_size2, 1],
        minval=-5.0 / tf.sqrt(cycles_v),
        maxval=5. / tf.sqrt(cycles_v),
        dtype=tf.float32)

    # if you have the minimum cycle and maximum cycle for a neighborhood, you can sample cycles from this neighborhood
    # by sampling real numbers x from [0,1] and just compute min_cyc*(1.-x) + max_cyc*x.
    # but here we do this computation in index space and then gather the cycle numbers and vq curves.
    cycle_indecies_lerp = tf.random.uniform(
        [batch_size2, num_cycles], minval=0., maxval=1., dtype=tf.float32)

    A = tf.expand_dims(
        tf.cast(index_d[:, 0] + index_d[:, 5], tf.float32), axis=1)

    B = tf.expand_dims(
        tf.cast(index_d[:, 1] + index_d[:, 5], tf.float32), axis=1)

    cycle_indecies_float = (
            A * (1. - cycle_indecies_lerp) + B * (cycle_indecies_lerp))

    cycle_indecies = tf.cast(cycle_indecies_float, tf.int32)

    flat_cycle_indecies = tf.reshape(
        cycle_indecies, [batch_size2 * num_cycles])

    my_cycles_flat = tf.gather(
        cycles, indices=flat_cycle_indecies, axis=0)

    my_cycles = tf.reshape(my_cycles_flat, [batch_size2, num_cycles])

    xs = my_cycles



    ys = tf.reshape(
        tf.gather(vq_curves, indices=flat_cycle_indecies),
        [-1, n_voltages, num_cycles])

    ws = tf.reshape(
        tf.gather(vq_curves_mask, indices=flat_cycle_indecies),
        [-1, n_voltages, num_cycles])

    ws2 = tf.tile(
        tf.reshape(1. / (tf.cast(index_d[:, -1], tf.float32)), [-1, 1, 1]),
        [1, n_voltages, num_cycles])

    # the indecies are referring to the cell indecies
    indecies = index_d[:, 3]

    my_centers = tf.concat(
        [centers[:, 0:1] + center_cycle_offsets, centers[:, 1:]], axis=1)

    with tf.GradientTape() as tape:
        (predictions_ys, mean, log_sig, res,
         dcyc, ddcyc, dother, ddother, dfeatures, ddfeatures
         ) = degradation_model((my_centers, indecies, xs, voltages), training=True)

        loss = (
                tf.reduce_mean(ws2 * ws * tf.square(ys - predictions_ys))
                / (1e-10 + tf.reduce_mean(ws2 * ws))
                + args['kl_coeff'] * tf.reduce_mean(
            0.5 * (tf.exp(log_sig) + tf.square(mean) - 1. - log_sig))
                + args['mono_coeff'] * (
                        tf.reduce_mean(tf.nn.relu(-res))
                        + tf.reduce_mean(tf.nn.relu(dcyc))
                        + tf.reduce_mean(tf.nn.relu(dother)))
                + args['smooth_coeff'] * (
                        tf.reduce_mean(
                            tf.square(tf.nn.relu(ddcyc))
                            + 0.02 * tf.square(tf.nn.relu(-ddcyc)))
                        + tf.reduce_mean(
                    tf.square(tf.nn.relu(ddother))
                    + 0.02 * tf.square(tf.nn.relu(-ddother))))
                + args['const_f_coeff'] * tf.reduce_mean(tf.square(dfeatures))
                + args['smooth_f_coeff'] * tf.reduce_mean(
            tf.square(ddfeatures)))

    # train_loss(loss)
    gradients = tape.gradient(loss, degradation_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, degradation_model.trainable_variables))


# === End : train step =========================================================

# === Begin: dist train step ===================================================

@tf.function
def dist_train_step(centers, index_d, mirrored_strategy, cycles_v, num_cycles,
                    cycles, vq_curves, vq_curves_mask, voltages, n_voltages, degradation_model, args, optimizer):
    mirrored_strategy.experimental_run_v2(
        train_step, args=(centers, index_d, cycles_v, num_cycles, cycles,
                          vq_curves, vq_curves_mask, voltages,n_voltages, degradation_model, args, optimizer))


# === End: dist train step =====================================================

# === Begin: plot vq ===========================================================

def plot_vq(cycles_m, cycles_v, degradation_model, voltages, mu,
            count, args, my_barcodes, test_object, all_data):
    print(Colour.BLUE + "plot vq" + Colour.END)
    for barcode_count, barcode in enumerate(my_barcodes):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        colors = ['k', 'r', 'b', 'g', 'm', 'c']
        for k_count, k in enumerate(test_object[barcode_count].keys()):

            for vq in all_data[barcode_count][k]['vq_curve']:
                ax.plot(vq, mu, c=colors[k_count])

        ax = fig.add_subplot(1, 2, 2)
        colors = [(1., 1., 1.), (1., 0., 0.), (0., 0., 1.),
                  (0., 1., 0.), (1., 0., 1.), (0., 1., 1.)]

        for k_count, k in enumerate(
                test_object[barcode_count].keys()):
            cycles = [0, 2000, 4000, 6000]
            for i, cyc in enumerate(cycles):
                cycle = ((float(cyc) - cycles_m) / tf.sqrt(cycles_v))
                preds = test_all_voltages(
                    cycle, k, barcode_count, degradation_model, voltages)

                mult = (i + 4) / (len(cycles) + 5)
                ax.plot(preds, mu, c=(
                    mult * colors[k_count][0],
                    mult * colors[k_count][1],
                    mult * colors[k_count][2]))

        plt.savefig(os.path.join(
            args['path_to_plots'],
            'VQ_{}_Count_{}.png'.format(barcode, count)))
        plt.close(fig)


# === End: plot vq =============================================================

# === Begin: plot capacity =====================================================

def plot_capacity(count, args, my_barcodes, test_object, all_data,
                  cycles_m, cycles_v, voltages, degradation_model):
    print(Colour.BLUE + "plot capacity" + Colour.END)
    for barcode_count, barcode in enumerate(my_barcodes):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        colors = ['k', 'r', 'b', 'g', 'm', 'c']
        for k_count, k in enumerate(
                test_object[barcode_count].keys()):

            ax.scatter(
                all_data[barcode_count][k]['cycle_number'],
                all_data[barcode_count][k]['vq_curve'][:, 0],
                c=colors[k_count])

            for cyc_i in [0, -1]:
                cyc = test_object[barcode_count][k][cyc_i]
                plt.axvline(x=cyc, color=colors[k_count],
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

            preds = test_single_voltage(my_cycles, voltages[0],
                                        k, barcode_count, degradation_model)
            ax.plot(cycles, preds, c=colors[k_count])

        plt.savefig(os.path.join(
            args['path_to_plots'],
            'Cap_{}_Count_{}.png'.format(barcode, count)))
        plt.close(fig)


# ==== End: plot capacity ======================================================

# === Begin: test all voltages =================================================


def test_all_voltages(cycle, k, barcode_count, degradation_model, voltages):
    centers = tf.expand_dims(tf.concat(
        (tf.expand_dims(cycle, axis=0),
         k), axis=0), axis =0)
    indecies = tf.expand_dims(barcode_count, axis=0)
    xs = tf.reshape(cycle, [1,1])
    predictions_ys = degradation_model((centers, indecies, xs,voltages), training=False)

    predictions_ys = tf.reshape(predictions_ys, shape=[-1])
    return predictions_ys


# === End: test all voltages ===================================================

# === Begin: test single voltage ===============================================

def test_single_voltage(cycles, v, k, barcode_count, degradation_model):
    centers = tf.concat(
        (
            tf.expand_dims(cycles, axis=1),
            tf.tile(tf.expand_dims(k, axis=0), [len(cycles), 1])
        ),
        axis = 1)
    indecies = tf.tile(tf.expand_dims(barcode_count, axis=0), [len(cycles)])
    xs = tf.expand_dims(cycles,
                        axis=1)
    predictions_ys = degradation_model((centers, indecies, xs, tf.expand_dims(v, axis=0)), training=False)
    predictions_ys = tf.reshape(predictions_ys, shape=[len(cycles)])
    return predictions_ys


# === End: test single voltage =================================================


################################################################################
# Begin: fitting level 0
################################################################################

def fitting_level0(args):
    my_barcodes = make_my_barcodes(args)  # list
    # plot_initial(my_barcodes, args)

    (
        # tensorflow.python.distribute.mirrored_strategy.MirroredStrategy
        mirrored_strategy,
        train_ds,  # tensorflow.python.distribute.input_lib.DistributedDataset
        cycles_m,  # numpy.float32
        cycles_v,  # numpy.float32
        num_cycles,  # int
        cycles,  # tensorflow.python.framework.ops.EagerTensor
        voltages,  # tensorflow.python.framework.ops.EagerTensor
        n_voltages,
        vq_curves,  # tensorflow.python.framework.ops.EagerTensor
        vq_curves_mask,  # tensorflow.python.framework.ops.EagerTensor
        # neware_parser.management.commands.fitting_level0.DegradationModel
        degradation_model,
        optimizer,  # tensorflow.python.keras.optimizer_v2.adam.Adam
        test_object,  # dict
        all_data,  # dict
        mu  # numpy.ndarray
    ) = initial_processing(my_barcodes, args)

    train(my_barcodes, mirrored_strategy, train_ds, degradation_model,
          cycles, num_cycles, cycles_m, cycles_v, test_object, all_data,
          mu, vq_curves, vq_curves_mask, optimizer, voltages,n_voltages, args)


################################################################################
# End: fitting level 0
################################################################################


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path_to_plots', required=True)
        parser.add_argument('--kl_coeff', type=float, default=0.00001)
        parser.add_argument('--mono_coeff', type=float, default=.005)
        parser.add_argument('--smooth_coeff', type=float, default=.005)
        parser.add_argument('--const_f_coeff', type=float, default=.0)
        parser.add_argument('--smooth_f_coeff', type=float, default=.01)
        parser.add_argument('--depth', type=int, default=2)
        parser.add_argument('--width', type=int, default=1 * 16)
        parser.add_argument('--print_loss_every', type=int, default=5000)
        parser.add_argument(
            '--visualize_fit_every', type=int, default=50000)
        parser.add_argument(
            '--visualize_vq_every', type=int, default=100000)

        parser.add_argument('--stop_count', type=int, default=300000)

    def handle(self, *args, **options):
        fitting_level0(options)
