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
import time
from django.db.models import Max,Min
from tensorflow.keras.layers import Dense, Flatten, Conv2D,GlobalAveragePooling1D, BatchNormalization, Conv1D, Layer
from tensorflow.keras import Model


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# TODO: a tool to delete cycles by looking at the total capacity vs cycle number.
# TODO: control derivatives w.r.t. voltage and cycle.

NUM_CYCLES = 5



class DictionaryLayer(Layer):
    def __init__(self, num_features, num_keys):
        super(DictionaryLayer, self).__init__()
        self.num_features = num_features
        self.num_keys= num_keys
        self.kernel = self.add_weight(
            "kernel",
            shape=[
                self.num_keys,
                self.num_features*2
            ]
        )

    def call(self, input, training=True):

        eps = tf.random.normal(
            shape=[
                self.num_keys,
                self.num_features
            ]
        )
        mean = self.kernel[:, :self.num_features]
        log_sig = self.kernel[:, self.num_features:]

        if not training:
            features = mean
        else:
            features = mean + tf.exp(log_sig / 2.) * eps




        fetched_features = tf.gather(
            features,
            input,
            axis=0
        )
        fetched_mean = tf.gather(
            mean,
            input,
            axis=0
        )
        fetched_log_sig = tf.gather(
            log_sig,
            input,
            axis=0
        )




        return fetched_features, fetched_mean, fetched_log_sig





class MyModel(Model):
    def __init__(self, num_keys, depth, width):
        super(MyModel, self).__init__()
        self.dense_initial = Dense(
            width, activation='relu',
            use_bias=True,
            bias_initializer='zeros',
        )
        self.dense = []
        for _ in range(depth):
            self.dense.append(
                    Dense(
                    width, activation='relu',
                    use_bias=True,
                    bias_initializer='zeros',
                )
            )
        self.dense_final = Dense(

            1, activation=None,
            use_bias=True,
            bias_initializer='zeros',
            kernel_initializer='zeros'
        )

        self.dictionary = DictionaryLayer(
            num_features=width,
            num_keys=num_keys
        )


    def call(self, x, training=False):
        """
        (centers, indecies, xs) = x

        centers : batch of [cyc, v, k[0], k[1]]
        indecies : batch of index
        xs: batch of [cyc, cyc,...]
        :param x:
        :param training:
        :return:
        """
        centers = x[0]
        indecies = x[1]
        xs = x[2]
        features, mean, log_sig = self.dictionary(indecies, training=training)

        cycles = centers[:, 0:1]
        others = centers[:, 1:]
        if training:
            with tf.GradientTape(persistent=True) as tape3:
                tape3.watch(cycles)
                tape3.watch(others)
                tape3.watch(features)
                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch(cycles)
                    tape2.watch(others)
                    tape2.watch(features)

                    centers2 = tf.concat(
                        (
                            cycles*(1e-10+tf.exp(-features[:, 0:1])),
                            others,
                            features[:, 1:]
                        ),
                        axis=1
                    )


                    centers2 = self.dense_initial(
                                        centers2
                                    )

                    for d in self.dense:
                        centers2 = d(centers2)

                    coeffs = self.dense_final(centers2)
                    res1 = tf.reshape(coeffs, shape=[-1, 1])




                dresdcyc = tape2.batch_jacobian(source=cycles, target=res1)[:, 0, :]
                dresdothers = tape2.batch_jacobian(source=centers, target=res1)[:, 0, :]
                dresdfeatures = tape2.batch_jacobian(source=features, target=res1)[:,0,:]





                del tape2

            d2resdcyc = tape3.batch_jacobian(source=cycles, target=dresdcyc)[:, 0, :]
            d2resdothers = tape3.batch_jacobian(source=others, target=dresdothers)
            d2resdfeatures = tape3.batch_jacobian(source=features, target=dresdfeatures)


            del tape3


            var_cyc = (xs -cycles)
            var_cyc2 = tf.square(var_cyc)


            res = (
                    res1 +
                    dresdcyc* var_cyc +
                    d2resdcyc * var_cyc2
            )

            return res, mean, log_sig, coeffs, dresdcyc, d2resdcyc, dresdothers, d2resdothers, dresdfeatures, d2resdfeatures
        else:

            centers2 = tf.concat(
                (
                    cycles * (1e-10 + tf.exp(-features[:, 0:1])),
                    others,
                    features[:, 1:]
                ),
                axis=1
            )

            centers2 = self.dense_initial(
                centers2
            )

            for d in self.dense:
                centers2 = d(centers2)

            coeffs = self.dense_final(centers2)
            return tf.reshape(coeffs, shape=[-1])

def clamp(a, x, b):
    x = min(x, b)
    x = max(x, a)
    return x



def fitting_level0(args):
    if not os.path.exists(args['path_to_plots']):
        os.mkdir(args['path_to_plots'])

    my_barcodes = CyclingFile.objects.filter(database_file__deprecated=False).order_by('database_file__valid_metadata__barcode').values_list(
        'database_file__valid_metadata__barcode', flat=True).distinct()
    used_barcodes = []
    for b in my_barcodes:
        if CycleGroup.objects.filter(barcode=b).exists():
            used_barcodes.append(b)

    my_barcodes = used_barcodes





    for barcode in  my_barcodes:
        plot_barcode(barcode, path_to_plots=args['path_to_plots'])


    index_data_full_full = [
    ]
    cycles_full = []
    metadata_data_full_full = []
    vq_curves_full = []
    vq_curves_mask_full = []
    all_rates = []
    all_cycles_full_full = []
    test_object = {}
    all_data = {}
    my_files = {}
    mu = None
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
                    print(f.filename, " Has a different set of voltages! please fix. original voltages: ", mu, " files voltages: ", f.get_uniform_voltages())
                    inconsistent_mu = True

    if inconsistent_mu:
        raise("some voltages were different. Please fix.")

    num_barcodes = len(my_barcodes)
    for barcode_count, barcode in enumerate(my_barcodes):
        test_object[barcode_count] = {}
        #here we load as if it were the original data
        data_da = {}
        for cyc_group in CycleGroup.objects.filter(barcode=barcode).order_by('discharging_rate'):
            data_da[(cyc_group.charging_rate, cyc_group.discharging_rate)] = numpy.array(
                [(cyc.get_offset_cycle(), cyc.get_discharge_curve()[:,0],cyc.get_discharge_curve()[:,1] )
                 for cyc in cyc_group.cycle_set.order_by('cycle_number') if cyc.valid_cycle ],
                dtype=[('cycle_number', 'f4'), ('vq_curve', 'f4', len(mu)), ('vq_curve_mask', 'f4', len(mu))]
            )

        all_data[barcode_count] = data_da
        all_rates.append(list(data_da.keys()))
        max_cap = 0.
        for k in data_da.keys():
            max_cap = max(max_cap, max(data_da[k]['vq_curve'][:, 0]))

        print(max_cap)

        index_data_full = []
        metadata_data_full = []

        all_cycles_full=[]
        for k_count, k in enumerate(data_da.keys()):

            data_da[k]['vq_curve'] = 1./max_cap * data_da[k]['vq_curve']
            print(k)
            min_cyc = min(data_da[k]['cycle_number'])
            max_cyc = max(data_da[k]['cycle_number'])
            #delta_cyc = 300
            total_delta = max_cyc - min_cyc
            delta_cyc = max(200, int(float(total_delta)/5.))

            all_cycles = list(filter(lambda x: x > min_cyc - 100,
                              range(20, int(max_cyc + 50), 40)))

            all_cycles_full.append(all_cycles)
            index_data = []
            metadata_data = []

            valid_cycles = 0
            for cyc in all_cycles:

                delta_up = max(max_cyc - cyc, 200)
                delta_down = max(cyc- min_cyc, 200)
                delta_actual = min(delta_up, delta_down)
                combined_delta = min(delta_actual, delta_cyc)

                below_cyc = cyc - combined_delta
                above_cyc = cyc + combined_delta

                mask = numpy.logical_and(
                    below_cyc <=data_da[k]['cycle_number'],
                    data_da[k]['cycle_number'] <= above_cyc
                )
                all_valid_indecies = numpy.arange(len(mask))[mask]

                if len(all_valid_indecies) < 2:
                    #for now, we just skip.
                    continue

                min_cyc_index = all_valid_indecies[0]
                max_cyc_index = all_valid_indecies[-1]
                if valid_cycles == 0:
                    test_object[barcode_count][k] = []
                test_object[barcode_count][k].append(cyc)
                valid_cycles += 1
                for v_count, v in enumerate(mu):

                    index_data.append([min_cyc_index, max_cyc_index, v_count, k_count, barcode_count, len(cycles_full), 0])
                    metadata_data.append([cyc, v, k[0], k[1]])

            if valid_cycles !=0:
                index_data = numpy.array(index_data, dtype=numpy.int32)
                index_data[:, -1] = valid_cycles
                metadata_data = numpy.array(metadata_data, dtype=numpy.float32)

                index_data_full.append(index_data)
                metadata_data_full.append(metadata_data)

            else:
                print('name: ', barcode)
                print('rates: ', k)

            if len(cycles_full) > 0 :
                cycles_full = numpy.concatenate((cycles_full,data_da[k]['cycle_number']))
                vq_curves_full = numpy.concatenate((vq_curves_full, data_da[k]['vq_curve']))
                vq_curves_mask_full = numpy.concatenate((vq_curves_mask_full, data_da[k]['vq_curve_mask']))
            else:
                cycles_full =  data_da[k]['cycle_number']
                vq_curves_full =  data_da[k]['vq_curve']
                vq_curves_mask_full = data_da[k]['vq_curve_mask']


        if len(index_data_full) != 0:
            index_data_full_full.append(index_data_full)
            metadata_data_full_full.append(metadata_data_full)
            all_cycles_full_full.append(all_cycles_full)
        else:
            print(barcode)


    index_data = tf.constant(numpy.concatenate([numpy.concatenate(index_data_full, axis=0) for index_data_full in index_data_full_full], axis=0))

    cycles = tf.constant(cycles_full)
    cycles_m, cycles_v = tf.nn.moments(cycles, axes=[0])

    cycles_m = cycles_m.numpy()
    cycles_v = cycles_v.numpy()
    cycles = (cycles - cycles_m)/tf.sqrt(cycles_v)
    voltages = tf.cast(tf.constant(mu), dtype=tf.float32)
    voltages_m, voltages_v = tf.nn.moments(voltages, axes=[0])
    voltages = (voltages - voltages_m) / tf.sqrt(voltages_v)
    vq_curves = tf.constant(vq_curves_full)
    vq_curves_mask = tf.constant(vq_curves_mask_full)
    metadata_data = (numpy.concatenate(
        [numpy.concatenate(metadata_data_full, axis=0) for metadata_data_full in metadata_data_full_full], axis=0))


    metadata_data[:, 0] = (metadata_data[:,0] - cycles_m)/numpy.sqrt(cycles_v)
    metadata_data[:, 1] = (metadata_data[:,1] - voltages_m.numpy())/numpy.sqrt(voltages_v.numpy())
    metadata_data = tf.constant(metadata_data)

    batch_size = 32*16*32
    num_cycles = NUM_CYCLES
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        train_ds_ = tf.data.Dataset.from_tensor_slices(
            (
                index_data,
                metadata_data,
            )
        ).repeat(2).shuffle(100000).batch(batch_size)
        train_ds = mirrored_strategy.experimental_distribute_dataset(train_ds_)

        model = MyModel(num_keys=num_barcodes, width=args['width'], depth=args['depth'])

        optimizer = tf.keras.optimizers.Adam()



        def train_step(centers,  index_d):
            # need to split the range ourselves.
            batch_size2 = index_d.shape[0]



            center_cycle_offsets = tf.random.uniform(
                [batch_size2,1],
                minval=-5.0/ tf.sqrt(cycles_v),
                maxval=5./ tf.sqrt(cycles_v),
                dtype=tf.float32,
            )



            cycle_indecies_lerp = tf.random.uniform(
                [batch_size2, num_cycles],
                minval=0.,
                maxval=1.,
                dtype=tf.float32,
            )
            #
            A = tf.expand_dims(tf.cast(index_d[:,0] + index_d[:, 5], tf.float32), axis=1)

            B = tf.expand_dims(tf.cast(index_d[:,1] + index_d[:, 5], tf.float32), axis=1)

            cycle_indecies_float = A * (1.-cycle_indecies_lerp) + B * (cycle_indecies_lerp)

            cycle_indecies = tf.cast(cycle_indecies_float, tf.int32)

            flat_cycle_indecies =tf.reshape(cycle_indecies, [batch_size2 * num_cycles])


            my_cycles_flat = \
                tf.gather(
                    cycles,
                    indices = flat_cycle_indecies,
                    axis=0
                )

            my_cycles = tf.reshape(my_cycles_flat, [batch_size2, num_cycles])

            xs = my_cycles

            flat_voltage_indecies = tf.reshape(
                tf.tile(
                index_d[:, 2:2+1],
                multiples=[1, num_cycles]
                ),
                [-1]
            )

            flat_vq_indecies = tf.stack(

                [flat_cycle_indecies,
                 flat_voltage_indecies
                ],
                axis=1
            )
            ys = tf.reshape(
                    tf.gather_nd(
                        vq_curves,
                        indices=flat_vq_indecies
                    ),
                [-1, num_cycles]
            )

            ws = tf.reshape(
                    tf.gather_nd(
                        vq_curves_mask,
                        indices=flat_vq_indecies
                    ),
                [-1, num_cycles]
            )

            ws2 = tf.tile(
                tf.reshape(
                1./ (tf.cast(index_d[:, 6], tf.float32)),
                [-1, 1]
                ),
                [1, num_cycles]
                )

            indecies = index_d[:, 4]

            my_centers = tf.concat(
                [
                    centers[:,0:1] + center_cycle_offsets,
                    centers[:,1:]
                ],
                axis=1

            )

            with tf.GradientTape() as tape:
                predictions_ys, mean, log_sig, res, dcyc, ddcyc, dother, ddother, dfeatures, ddfeatures = model((my_centers, indecies, xs), training=True)


                loss = (tf.reduce_mean(ws2*ws*tf.square(ys - predictions_ys))/(1e-10 + tf.reduce_mean(ws2*ws)) +
                        args['kl_coeff'] * tf.reduce_mean(0.5* (tf.exp(log_sig) + tf.square(mean)-1.- log_sig))
                        + args['mono_coeff']* (tf.reduce_mean( tf.nn.relu(-res))+ tf.reduce_mean(tf.nn.relu(dcyc)) + tf.reduce_mean( tf.nn.relu(dother)))
                        + args['smooth_coeff'] * (tf.reduce_mean(tf.square(tf.nn.relu(ddcyc))+ 0.02*tf.square(tf.nn.relu(-ddcyc))) + tf.reduce_mean(tf.square(tf.nn.relu(ddother)) + 0.02*tf.square(tf.nn.relu(-ddother))))
                        + args['const_f_coeff'] * tf.reduce_mean(tf.square(dfeatures))
                        + args['smooth_f_coeff'] * tf.reduce_mean(tf.square(ddfeatures))
                        )

            #train_loss(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        @tf.function
        def dist_train_step(centers,  index_d):
            mirrored_strategy.experimental_run_v2(train_step, args=(centers, index_d))




    def test_all_voltages(cycle, k, barcode_count):

        centers = tf.stack(

            [
                tf.tile([cycle], multiples=[len(mu)]),
                voltages,
                tf.tile([k[0]], multiples=[len(mu)]),
                tf.tile([k[1]], multiples=[len(mu)]),
            ],
            axis=1
        )
        indecies = tf.tile([barcode_count], multiples=[len(mu)])
        xs = tf.tile([[cycle]], multiples=[len(mu), 1])
        predictions_ys = model((centers, indecies, xs), training=False)

        predictions_ys = tf.reshape(predictions_ys, shape=[len(mu)])
        return predictions_ys



    def test_single_voltage(cycles, v, k, barcode_count):

        centers = tf.stack(

            [
                cycles,
                tf.tile([v], multiples=[len(cycles)]),
                tf.tile([k[0]], multiples=[len(cycles)]),
                tf.tile([k[1]], multiples=[len(cycles)]),
            ],
            axis=1
        )
        indecies = tf.tile([barcode_count], multiples=[len(cycles)])
        xs = tf.expand_dims(cycles, axis=1)
        predictions_ys = model((centers, indecies, xs), training=False)

        predictions_ys = tf.reshape(predictions_ys, shape=[len(cycles)])
        return predictions_ys


    EPOCHS = 100000
    count = 0
    grads = None

    template = 'Epoch {}, Count {}'

    with mirrored_strategy.scope():
        for epoch in range(EPOCHS):
            for index, metadat in train_ds:
                count += 1

                dist_train_step(metadat, index)

                count +=1




                if (count % args['visualize_vq_every']) == 0 and count != 0:
                    for barcode_count, barcode in enumerate(my_barcodes):
                        fig = plt.figure()
                        ax = fig.add_subplot(1,2,1)
                        colors = ['k', 'r', 'b', 'g', 'm', 'c']
                        for k_count, k in enumerate(test_object[barcode_count].keys()):

                            for vq in all_data[barcode_count][k]['vq_curve']:
                                ax.plot(vq, mu, c=colors[k_count])

                        ax = fig.add_subplot(1, 2, 2)
                        colors = [(1.,1.,1.), (1.,0.,0.), (0.,0.,1.), (0.,1.,0.), (1.,0.,1.), (0.,1.,1.)]
                        for k_count, k in enumerate(test_object[barcode_count].keys()):
                            cycles = [0,2000,4000,6000]
                            for i, cyc in enumerate(cycles):
                                my_cyc = (float(cyc) - cycles_m) / tf.sqrt(cycles_v)
                                preds = test_all_voltages(my_cyc, k, barcode_count)

                                mult = (i+4)/(len(cycles)+5)
                                ax.plot(preds, mu, c=(mult*colors[k_count][0],mult*colors[k_count][1],mult*colors[k_count][2]))

                        plt.savefig(os.path.join(args['path_to_plots'], 'VQ_{}_Count_{}.png'.format(barcode, count)))
                        plt.close(fig)

                if (count % args['visualize_fit_every']) == 0 and count != 0:
                    for barcode_count, barcode in enumerate(my_barcodes):
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        colors = ['k', 'r', 'b', 'g', 'm', 'c']
                        for k_count, k in enumerate(test_object[barcode_count].keys()):


                             ax.scatter(all_data[barcode_count][k]['cycle_number'], all_data[barcode_count][k]['vq_curve'][:,0], c=colors[k_count])
                             for cyc_i in [0,-1]:
                                 cyc = test_object[barcode_count][k][cyc_i]
                                 plt.axvline(x=cyc, color=colors[k_count], linestyle='--')

                        for k_count, k in enumerate(test_object[barcode_count].keys()):
                            cycles = test_object[barcode_count][k]
                            min_c = min(cycles)
                            max_c = max(cycles)
                            cycles = [float(min_c) + float(max_c-min_c)*x for x in numpy.arange(0.,1.1,0.1)]

                            my_cycles = [(cyc - cycles_m)/ tf.sqrt(cycles_v) for cyc in cycles]

                            preds = test_single_voltage(my_cycles, voltages[0], k, barcode_count)
                            ax.plot(cycles, preds, c=colors[k_count])

                        plt.savefig(
                            os.path.join(args['path_to_plots'], 'Cap_{}_Count_{}.png'.format(barcode, count)))
                        plt.close(fig)

                if (count % args['print_loss_every']) == 0 and count != 0:

                    print(
                        template.format(
                            epoch + 1,
                            count,
                        )
                    )



class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--path_to_plots', required=True)
        parser.add_argument('--kl_coeff', type=float, default=0.00001)
        parser.add_argument('--mono_coeff', type=float, default=.005)
        parser.add_argument('--smooth_coeff', type=float, default=.005)
        parser.add_argument('--const_f_coeff', type=float, default=.0)
        parser.add_argument('--smooth_f_coeff', type=float, default=.01)
        parser.add_argument('--depth', type=int, default=2)
        parser.add_argument('--width', type=int, default=1*16)
        parser.add_argument('--print_loss_every', type=int, default=1000)
        parser.add_argument('--visualize_fit_every', type=int, default=10000)
        parser.add_argument('--visualize_vq_every', type=int, default=100000)
    def handle(self, *args, **options):
        fitting_level0(options)


