import argparse
import os
import pickle
from Process_Everything import plot_degradation,plot_voltage_curves
import numpy
import math

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D,GlobalAveragePooling1D, BatchNormalization, Conv1D
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def get_masks_and_max(counts):
    max_counts = tf.reduce_max(counts)

    masks_logical = tf.sequence_mask(
        lengths=counts,
        maxlen=max_counts,
    )

    return masks_logical, max_counts


class Resnet1DIdentityBlock(Model):
  def __init__(self, kernel_size, filters):
    super(Resnet1DIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = Conv1D(filters1, 1)
    #self.bn2a = BatchNormalization()

    self.conv2b = Conv1D(filters2, kernel_size, padding='same')
    #self.bn2b = BatchNormalization()

    self.conv2c = tf.keras.layers.Conv1D(filters3, 1)
    '''
    self.bn2c = tf.keras.layers.BatchNormalization(
        scale=True,
        renorm=True,
        gamma_initializer=tf.zeros_initializer(),
        axis=2
    )
    '''

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    #x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    #x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    #x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)



#TODO:
class MyModel(Model):
  def __init__(self, num_res_enc, num_res_dec, dense=False):
    super(MyModel, self).__init__()
    self.has_dense = dense
    if dense:
        self.flatten = Flatten()
        self.dense = Dense(32, activation='relu',
                           use_bias=True,
                           #kernel_initializer='zeros',
                           bias_initializer='zeros',
                           )
    else:
        self.conv_enc_initial = Conv1D(32, 3, activation='relu' )
        self.res_enc = []
        for _ in range(num_res_enc):
            self.res_enc.append(Resnet1DIdentityBlock(kernel_size=3, filters=(32,32,32)))
        self.average = GlobalAveragePooling1D()

    self.conv_dec_initial = Conv1D(32, 1, activation='relu')
    self.res_dec = []
    for _ in range(num_res_dec):
        self.res_dec.append(Resnet1DIdentityBlock(kernel_size=1, filters=(32, 32, 32)))

    self.conv_dec_final = Conv1D(3, 1,
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer='zeros',
                                 bias_initializer='zeros',
                                 )





  def call(self, x, training=False):

    q = x[0]
    outcomes_x = x[1]
    counts = x[2]
    masks_logical, max_counts = get_masks_and_max(counts)
    cropped_outcomes_x= outcomes_x[:, :max_counts, :]

    #q = self.conv1(q)
    #q_compressed = self.bn(self.average(q))

    if self.has_dense:
        q = self.dense(self.flatten(q))
    else:
        q = self.conv_enc_initial(q)
        for enc in self.res_enc:
            q = enc(q, training=training)

        q = self.average(q)
    q_compressed = q

    #print(q_compressed)
    q_compressed = tf.expand_dims(q_compressed, axis=1)
    #print(q_compressed)
    q_compressed = tf.tile(q_compressed, multiples=[1, max_counts, 1])
    #print(q_compressed)
    embedded_cropped_outcomes_x = tf.concat(
        (cropped_outcomes_x, q_compressed),
        axis=2
    )

    enc_x = self.conv_dec_initial(embedded_cropped_outcomes_x)
    for dec in self.res_dec:
        enc_x = dec(enc_x, training=training)
    outcomes = self.conv_dec_final(enc_x)


    return outcomes



"""
TODO:
In order to correctly predict capacity degradation, there are two aspects:
the relative starting capacity of C/20, C/2, 1C, 2C, 3C.

For repeatability, normalize to C/20 cap (or lowest charge/discharge rate)

Also, the degradation for each rate should be seen relative to baseline.



TODO:
add correction mechanism to specify exceptions.


TODO:
The golden standard must be padded



TODO:
in order to model multiple rates, you can use a 'gaussian process' style model.
Basically, you stack voltage curves by rate, 
add the rate in the matrix, and only apply convolutions of kernel size 1 in the rate direction. 
alternately, you can increase the batch dimension, then apply 1d convolutions, then restack.
Then, at the end, simply average across rate dimension.

if this is not powerful enough, you can use transformer architecture.


TODO: 
investigate a gaussian process model which stacks all initial segments for various temperatures and upper cutoff voltages,
then treat each independently for the initial processing, but then averages (or transformer) to produce a single representation.
This representation is then queried for all available (cyc, rate, upper_cutoff, temp) to produce predictions.



TODO:
the delta Q features are NOT!!!!!!!!!!!!!!!!! working well. 
    - We need to investigate how far this architecture can be taken with good practices.
    - We need to add the average Q as well as the difference. 
    - We can compute these quantities for all rates.
    - We can define an extractor which takes the set of voltage curves and 
    produces a model of the constant and linear term as a function of voltage and rate.
 
 
 
TODO:
we need to determine how good a C + cyc*A + sqrt(t)*B a fit is. 
- cut connection from input.
- define from rate to C,A,B
- define sqrt(cyc), 
- have only one element in the batch  
"""

mu=.05 * numpy.array(range(2 * 30, 2 * 46))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_degradation_analysis', required=True)
    parser.add_argument('--common_path', default='')
    parser.add_argument('--num_res_enc', type=int, default=5)
    parser.add_argument('--num_res_dec', type=int, default=5)

    args = parser.parse_args()


    da_suffix = 'DegradationAnalysis.file'
    all_path_filenames_DegradationAnalysis = []

    dac_suffix = 'DegradationAnalysisCurves_initial_segment.file'
    all_path_filenames_DegradationAnalysisCurves = []


    for root, dirs, filenames in os.walk(os.path.join(args.common_path, args.path_to_degradation_analysis)):
        for file in filenames:
            if file.endswith(da_suffix):
                all_path_filenames_DegradationAnalysis.append(os.path.join(root, file.split(da_suffix)[0]))
            if file.endswith(dac_suffix):
                all_path_filenames_DegradationAnalysisCurves.append(os.path.join(root, file.split(dac_suffix)[0]))


    all_path_filenames_DegradationAnalysis = set(all_path_filenames_DegradationAnalysis)
    all_path_filenames_DegradationAnalysisCurves = set(all_path_filenames_DegradationAnalysisCurves)

    all_path_filenames = all_path_filenames_DegradationAnalysis.intersection(all_path_filenames_DegradationAnalysisCurves)

    all_data = []
    for f in all_path_filenames:
        with open(f+da_suffix, 'rb') as fi:
            data_da = pickle.load(fi)

        with open(f+dac_suffix, 'rb') as fi:
            data_dac = pickle.load(fi)


        list_of_degradation_points = []
        for k in data_da['data'].keys():
            for elem in data_da['data'][k]:
                list_of_degradation_points.append(
                    [
                        elem['cycle_number'][0]/1000.,
                        elem['discharge_c_rate'][0],
                        math.exp(elem['discharge_cap'][0]) - 1.,
                        elem['s_norm'][0],
                        elem['discharge_r'][0]
                    ]
                )

        list_of_degradation_points = numpy.array(list_of_degradation_points)

        '''
        plot_degradation(
            filename = f,
            summary_data = data_da['data'],
            output_filename=None,
            direct=True
        )
        '''

        for k in data_dac.keys():
            for i in range(len(data_dac[k])):
                data_dac[k][i]['vq_curve'] = (1./math.exp(data_da['theoretical_capacity'])*data_dac[k][i]['vq_curve'][0],1./math.exp(data_da['theoretical_capacity'])*data_dac[k][i]['vq_curve'][1])
        for k in data_dac.keys():
            if len(data_dac[k]) > 4:
                # found the 1C cycles.
                # for now, assume the cycles are ordered.
                list_of_q_ij = []
                for i in range(len(data_dac[k])):
                    list_of_q_ij.append(
                        numpy.stack(
                            (
                                    data_dac[k][i]['cycle_number'][0] * numpy.ones(len(mu), dtype=numpy.float32),
                                    mu,
                                    data_dac[k][i]['vq_curve'][0]
                            ),
                            axis=1
                        )
                    )
                list_of_q_ij = numpy.array(list_of_q_ij)


        '''
        plot_voltage_curves(
            filename = f,
            summary_data = data_dac,
            mu=.05 * numpy.array(range(2 * 30, 2 * 46)),
            output_filename=None,
            direct=True
        )
        '''
        all_data.append((list_of_q_ij, list_of_degradation_points))

    max_degradation_points = 0
    for _, ad in all_data:
        max_degradation_points = max(max_degradation_points, len(ad))

    number_of_data = len(all_data)

    #here we enforce that the shape is 10, len(mu), 3
    q_tensor = numpy.zeros(
        (number_of_data, len(mu), 3),
         dtype=numpy.float32
    )

    degradation_points_x_tensor = numpy.zeros(
        (number_of_data, max_degradation_points, 2),
        dtype=numpy.float32
    )

    degradation_points_y_tensor = numpy.zeros(
        (number_of_data, max_degradation_points, 3),
        dtype=numpy.float32
    )

    degradation_points_len_tensor = numpy.zeros(
        number_of_data,
        dtype=numpy.int32
    )

    BATCH_SIZE = 128

    sortlist = []
    for i in range(number_of_data):
        degradation_points_len_tensor[i] = len(all_data[i][1])
        degradation_points_x_tensor[i, :degradation_points_len_tensor[i], :] = all_data[i][1][:,:2]
        degradation_points_y_tensor[i, :degradation_points_len_tensor[i], :] = all_data[i][1][:,2:]
        #This computes the delta Q feature
        q_tensor[i, :, 0] = all_data[i][0][-1,:, 2] - all_data[i][0][0,:,2]
        sortlist.append(numpy.sum(all_data[i][0][-1,:,2] - all_data[i][0][0,:,2]))


    indecies = numpy.argsort(sortlist)
    number_of_colors =2
    colors = plt.get_cmap('jet')(numpy.linspace(0,1,number_of_colors))
    for k in range(50):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)

        for counter, i in enumerate(indecies[k*number_of_colors:(k+1)*number_of_colors]):
            ax.plot(mu,q_tensor[i,:,0], color=colors[counter])


        ax = fig.add_subplot(1,2,2, projection='3d')
        for counter, i in enumerate(indecies[k*number_of_colors:(k+1)*number_of_colors]):
            ax.scatter(
                degradation_points_x_tensor[i, :degradation_points_len_tensor[i], 0],
                degradation_points_x_tensor[i, :degradation_points_len_tensor[i], 1],

                degradation_points_y_tensor[i, :degradation_points_len_tensor[i], 0],
                color=colors[counter]
             )
        plt.show()
    0/0
    train_ds = tf.data.Dataset.from_tensor_slices(
        (
            q_tensor[:BATCH_SIZE],
            degradation_points_x_tensor[:BATCH_SIZE],
            degradation_points_y_tensor[:BATCH_SIZE],
            degradation_points_len_tensor[:BATCH_SIZE],
        )
    ).batch(BATCH_SIZE)


    model = MyModel(num_res_enc=args.num_res_enc, num_res_dec=args.num_res_dec, dense=False)

    loss_object = MeanSquaredError()

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')



    def train_step(qs, xs, ys, counts):
        masks, max_counts = get_masks_and_max(counts)

        cropped_xs = xs[:, :max_counts, :]
        xs_tab = tf.reshape(cropped_xs, [-1, 2])



        '''
        TODO:
        
        we want to incur derivative penalities, figuring out how things should vary with rate and cycle number
        
        '''
        with tf.GradientTape() as tape:
            with tf.GradientTape(persistent=True) as gg:
                gg.watch(xs_tab)
                with tf.GradientTape(persistent=True) as ggg:

                    ggg.watch(xs_tab)
                    xs_new = tf.reshape(xs_tab, [-1,max_counts, 2])

                    predictions_ys = model((qs, xs_new, counts))
                    cropped_ys = ys[:, :max_counts, :]



                    rescaling = tf.cast(max_counts, dtype=tf.float32)/tf.reduce_mean(tf.cast(counts,
                            dtype=tf.float32))


                    weights = tf.cast(
                        masks,
                        dtype=tf.float32
                    )
                    first_predictions = tf.reshape(predictions_ys[:,:,:1], [-1, 1])

                delta_ys = ggg.batch_jacobian(first_predictions, xs_tab, experimental_use_pfor=False)
                delta_ys_reshaped = tf.reshape(delta_ys, [-1, max_counts , 2])
                del ggg

                delta_ys_cyc = tf.reshape(delta_ys, [-1, 2])[:, :1]

            dd_ys = gg.batch_jacobian(delta_ys_cyc, xs_tab, experimental_use_pfor=False)
            dd_ys_reshaped = tf.reshape(dd_ys, [-1, max_counts, 2])
            del gg

            d2_ys = tf.reduce_mean(delta_ys_reshaped * delta_ys_reshaped * tf.expand_dims(weights, axis=2))
            delta_ys_cycles =delta_ys_reshaped[:, :max_counts, 0]
            d2_ys_cycles = tf.reduce_mean(tf.nn.relu(delta_ys_cycles) * weights)

            dd2_ys = tf.reduce_mean(dd_ys_reshaped * dd_ys_reshaped * tf.expand_dims(weights, axis=2))
            dd_ys_cycles = dd_ys_reshaped[:, :max_counts, 0]
            dd2_ys_cycles = tf.reduce_mean(tf.nn.relu(dd_ys_cycles) * weights)

            loss = loss_object(cropped_ys, predictions_ys, sample_weight=rescaling*weights) #+ 0.1 * d2_ys + 0.1 * dd2_ys + 10. * d2_ys_cycles + 10. * dd2_ys_cycles

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    @tf.function
    def train_step2(qs, xs, ys, counts):
        masks, max_counts = get_masks_and_max(counts)

        with tf.GradientTape() as tape:

            predictions_ys = model((qs, xs, counts), training=True)
            cropped_ys = ys[:, :max_counts, :]

            rescaling = tf.cast(max_counts, dtype=tf.float32) / tf.reduce_mean(tf.cast(counts,
                                                                                       dtype=tf.float32))

            weights = tf.cast(
                masks,
                dtype=tf.float32
            )


            loss = loss_object(cropped_ys, predictions_ys,
                               sample_weight=rescaling * weights)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)


    def test_step(qs, xs, ys, counts):

        predictions_ys = model((qs, xs, counts), training=False)

        masks, max_counts = get_masks_and_max(counts)
        cropped_ys = ys[:, :max_counts, :]
        weights = tf.cast(
            masks,
            dtype=tf.float32
        )

        loss = numpy.average((tf.abs(cropped_ys[:,:,0]-predictions_ys[:,:,0])).numpy(), weights=weights.numpy())

        print('test loss: ', loss)
        '''
        fig = plt.figure()
        n = len(xs.numpy())
        for j in range(1):
            ax = fig.add_subplot(1, 1, j + 1, projection='3d')
            for index in range(n):
                x = xs[index, :counts[index],:].numpy()
                pred_y = predictions_ys[index, :counts[index],:].numpy()
                orig_y = cropped_ys[index, :counts[index],:].numpy()


                ax.scatter(x[:, 0], x[:, 1], pred_y[:,j], c='r')
                ax.scatter(x[:, 0], x[:, 1], orig_y[:,j], c='k')
        plt.show()
        '''

    EPOCHS = 10000


    for epoch in range(EPOCHS):
        first = True
        for qs, xs, ys, counts in train_ds:
            if (epoch % 1000) == 0 and first:
                test_step(qs, xs, ys, counts)
            first = False
            train_step2(qs, xs, ys, counts)
            template = 'Epoch {}, Loss: {}'


        print (
            template.format(
                epoch+1,
                train_loss.result(),
            )
        )
