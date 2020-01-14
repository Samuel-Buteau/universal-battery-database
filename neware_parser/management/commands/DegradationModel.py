import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer



def feedforward_nn_parameters(depth, width):
    initial = Dense(
        width,
        activation = 'relu',
        use_bias = True,
        bias_initializer = 'zeros'
    )

    bulk = [
        Dense(
            width,
            activation = 'relu',
            use_bias = True,
            bias_initializer = 'zeros'
        ) for _ in range(depth)
    ]

    final = Dense(
        1,
        activation = None,
        use_bias = True,
        bias_initializer = 'zeros',
        kernel_initializer = 'zeros'
    )
    return {'initial':initial, 'bulk':bulk, 'final':final}



class DegradationModel(Model):

    def __init__(self, num_keys, depth, width):
        super(DegradationModel, self).__init__()


        self.feedforward_nn = {
            'cap': feedforward_nn_parameters(depth, width),
            'eq_vol': feedforward_nn_parameters(depth, width),
            'r': feedforward_nn_parameters(depth, width),
        }

        self.dictionary = DictionaryLayer(num_features=width, num_keys=num_keys)

        self.width = width
        self.num_keys = num_keys

    def dchg_rate(self, rates):
        return rates[:, 1:2]

    def apply_nn(self, cycles, rates, features, nn):
        if nn == "dchg_rate":
            return self.dchg_rate(rates)
        if nn == 'r':
            return self.r(cycles, features)

        # Convention: any nn always gets called with all the inputs if possible.
        # and is responsible for only using the appropriate ones.
        # For instance, nn 'r' doesn't use rates, so it will be called with the rates
        # and set it to None before running the neural net code.

        # two nn's are needed to implement the equation for discharge voltage
        # the keyword 'max_discharge_voltage' calls the networks
        # with the right inputs and plumbing

        # First, if the symbol refers to other symbols, first load the other symbols

        dependencies = {
            'max_dchg_vol': ['dchg_rate', 'eq_vol', 'r']
        }
        equations = {
            'max_dchg_vol': lambda: preloaded['eq_vol'] - (preloaded['dchg_rate'] * preloaded['r']),
        }

        #This code runs
        preloaded = {}
        if nn in dependencies.keys():
            for arg in dependencies[nn]:
                preloaded[arg] = self.apply_nn(cycles, rates, features, arg)
        #this evaluates compound expressions and returns the result.
        if nn in equations.keys():
            return equations[nn]()

        else:
            #primitive (with a nn)
            if nn == 'eq_vol':
                rates = rates[:, 0:1]

            centers = (self.feedforward_nn[nn]['initial'])(
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
            return (self.feedforward_nn[nn]['final'])(centers)

    def r(self, cycles, features):
        centers = (self.feedforward_nn['r']['initial'])(
            tf.concat(
                (
                    # readjust the cycles
                    cycles * (1e-10 + tf.exp(-features[:, 0:1])),
                    features[:, 1:]
                ),
                axis=1
            )
        )
        for d in self.feedforward_nn['r']['bulk']:
            centers = d(centers)
        return (self.feedforward_nn['r']['final'])(centers)


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
            pred_max_dchg_vol = self.apply_nn(cycles, rates, features, 'max_dchg_vol')
            pred_eq_vol = self.apply_nn(cycles, rates, features, 'eq_vol')
            pred_r = self.apply_nn(cycles, rates, features, 'r')

            return {
                "pred_cap": tf.reshape(
                    self.apply_nn(
                        cycles_flat, rates_flat, features_flat, 'cap'),
                    [-1, vol_tensor.shape[0]]
                ),
                "pred_max_dchg_vol": pred_max_dchg_vol,
                "pred_eq_vol": pred_eq_vol,
                "pred_r": pred_r
            }




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
