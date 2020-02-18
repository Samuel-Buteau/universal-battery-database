import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

from .colour_print import Print

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
    return {'initial': initial, 'bulk': bulk, 'final': final}

"""
The call convention for the neural networks is a bit complex but it makes the code easier to use.

First, there are some parameters that are from the real data (e.g. cycle number, voltage, etc.)
these are passed to every call as a dictionary.

Second, all the parameters that are used in the body of the function can be overridden by passing them. If they are passed as None, 
the default dictionary will be used

"""


class DegradationModel(Model):

    def __init__(self, num_keys, depth, width):
        super(DegradationModel, self).__init__()

        self.nn_cap = feedforward_nn_parameters(depth, width)
        #self.nn_eq_vol = feedforward_nn_parameters(depth, width)
        self.nn_r = feedforward_nn_parameters(depth, width)

        self.nn_theoretical_cap = feedforward_nn_parameters(depth, width)
        self.nn_soc = feedforward_nn_parameters(depth, width)
        self.nn_soc_0 = feedforward_nn_parameters(depth, width)
        self.nn_init_soc = feedforward_nn_parameters(depth, width)

        self.nn_soc_part2 = feedforward_nn_parameters(depth, width)
        #self.nn_eq_voltage_0 = feedforward_nn_parameters(depth, width)

        self.nn_shift = feedforward_nn_parameters(depth, width)
        self.nn_soc_0_part3 = feedforward_nn_parameters(depth, width)
        self.nn_soc_1_part3 = feedforward_nn_parameters(depth, width)

        self.dictionary = DictionaryLayer(num_features=width, num_keys=num_keys)

        self.width = width
        self.num_keys = num_keys

    # Begin: nn application functions ==========================================

    def norm_constant(self, params, over_params=None, scalar = True):
        if over_params is None:
            over_params = {}
        if 'features' in over_params.keys():
            features = over_params['features']
        else:
            features_label = "features"
            if not scalar:
                features_label += "_flat"
            features = params[features_label]

        return features[:, 0:1]


    def cell_features(self, params, over_params=None, scalar = True):
        if over_params is None:
            over_params = {}
        if 'features' in over_params.keys():
            features = over_params['features']
        else:
            features_label = "features"
            if not scalar:
                features_label += "_flat"
            features = params[features_label]

        return features[:, 1:]


    def norm_cycle(self, params, over_params=None, scalar = True):
        if over_params is None:
            over_params = {}
        if 'cycles' in over_params.keys():
            cycles = over_params['cycles']
        else:
            cycles_label = "cycles"
            if not scalar:
                cycles_label += "_flat"
            cycles = params[cycles_label]

        return cycles * (1e-10 + tf.exp(-self.norm_constant(params, over_params, scalar)))

    # theoretical_cap(cycles, constant_current, cell_feat)
    def theoretical_cap(self, params, over_params=None):
        if over_params is None:
            over_params = {}
        norm_cycle = self.norm_cycle(params, over_params)

        if 'constant_current' in over_params.keys():
            constant_current = over_params["constant_current"]
        else:
            constant_current = params['constant_current']

        cell_feat = self.cell_features(params, over_params)

        dependencies = (
            norm_cycle,
            tf.abs(constant_current),
            cell_feat
        )
        return tf.exp(self.nn_call(self.nn_theoretical_cap, dependencies))

    def r(self, params, over_params=None):
        if over_params is None:
            over_params = {}
        dependencies = (
            self.norm_cycle(params, over_params, scalar=True),
            self.cell_features(params, over_params,scalar=True)
        )
        return tf.exp(self.nn_call(self.nn_r, dependencies))



    def eq_voltage_0(self, params, over_params=None):
        if over_params is None:
            over_params = {}
        if 'end_voltage_prev' in over_params.keys():
            end_voltage_prev = over_params["end_voltage_prev"]
        else:
            end_voltage_prev = params["end_voltage_prev"]
        if 'end_current_prev' in over_params.keys():
            end_current_prev = over_params["end_current_prev"]
        else:
            end_current_prev = params["end_current_prev"]

        return end_voltage_prev - end_current_prev * self.r(params, over_params)

    def eq_voltage_1(self, params, over_params=None):
        if over_params is None:
            over_params = {}
        if 'voltage' in over_params.keys():
            voltage = over_params["voltage"]
        else:
            voltage = params['voltage_flat']

        if 'constant_current' in over_params.keys():
            constant_current = over_params['constant_current']
        else:
            constant_current = params["constant_current_flat"]

        r_flat = self.add_volt_dep(self.r(params,over_params), params)
        return voltage - constant_current * r_flat

    # sco_1 = soc(voltage = eq_voltage_1, cell_feat)
    def soc(self, params, over_params=None, scalar = True):
        if over_params is None:
            over_params = {}

        cell_feat = self.cell_features(params, over_params, scalar)
        if 'voltage' in over_params.keys():
            voltage = over_params['voltage']
        else:
            raise Exception("tried to call soc without any voltage. please add a 'voltage':value to over_param")

        dependencies = (
            voltage,
            cell_feat
        )
        return tf.nn.elu(self.nn_call(self.nn_soc, dependencies))

    ''' Part 4 ------------------------------------------------------------- '''

    ''' Part 3 ------------------------------------------------------------- '''

    # def dchg_cap_part3(self, params):
    #     theoretical_cap = self.add_volt_dep(
    #         self.theoretical_cap(params),
    #         params
    #     )
    #
    #     shift = self.shift(params)
    #
    #     soc_0 = self.add_volt_dep(
    #         self.soc_part3(params, shift, self.eq_voltage_0(params)),
    #         params
    #     )
    #     # "whereever you are in the voltage curve"
    #     soc_1 = self.soc_part3(
    #         params,
    #         self.add_volt_dep(shift, params),
    #         self.eq_voltage_1(params),
    #         scalar = False
    #     )
    #
    #     return -theoretical_cap * (soc_1 - soc_0)
    #
    # def shift(self, params):
    #     dependencies = (
    #         self.norm_cycle(params),
    #         params["end_current_prev"],
    #         params["constant_current"],
    #         params["cell_feat"]
    #     )
    #     return self.nn_call(self.nn_shift, dependencies)
    #
    # def soc_part3(self, params, shift, voltage, scalar = True):
    #     cell_feat = "cell_feat"
    #     if not scalar:
    #         cell_feat = "cell_feat_flat"
    #
    #     dependencies = (
    #         shift,
    #         voltage,
    #         params[cell_feat]
    #     )
    #     return self.nn_call(self.nn_soc_0_part3, dependencies)

    ''' Part 2 ------------------------------------------------------------- '''

    def dchg_cap_part2(self, params, over_params=None):
        if over_params is None:
            over_params = {}

        theoretical_cap = self.add_volt_dep(
            self.theoretical_cap(params,over_params),
            params
        )
        new_over_param = {}
        for key in over_params.keys():
            new_over_param[key] = over_params[key]
        new_over_param['voltage'] = self.eq_voltage_0(params, over_params)
        soc_0 = self.add_volt_dep(
            self.soc(params, new_over_param , scalar = True),
            params
        )
        new_over_param = {}
        for key in over_params.keys():
            new_over_param[key] = over_params[key]
        new_over_param['voltage'] = self.eq_voltage_1(params, over_params)

        soc_1 = self.soc(params, new_over_param, scalar = False)

        return theoretical_cap * (soc_1 - soc_0)

        # eq_voltage_0(cycle, cell_feat)


    ''' Part 1 ------------------------------------------------------------- '''


    ''' End ---------------------------------------------------------------- '''


    # Unstructured variables ---------------------------------------------------

    def nn_call(self, nn_func, dependencies):
        centers = nn_func['initial'](
            tf.concat(dependencies, axis=1)
        )
        for d in nn_func['bulk']:
            centers = d(centers)
        return nn_func['final'](centers)



    # End: nn application functions ============================================

    def create_derivatives(self, nn, params, over_params={}, scalar = True,
                           no_vol_der=False,
                           no_cyc_der=False,
                           no_features_der=False,
                           no_constant_current_der=False,
                           no_end_current_prev_der=False
                           ):
        derivatives = {}

        cycles_label = "cycles"
        end_current_prev_label = "end_current_prev"
        constant_current_label = "constant_current"
        features_label = "features"
        if not scalar:
            cycles_label = "cycles_flat"
            end_current_prev_label = "end_current_prev_flat"
            constant_current_label = "constant_current_flat"
            features_label = "features_flat"

        if 'cycles' in over_params.keys():
            cycles = over_params['cycles']
        else:
            cycles = params[cycles_label]
        if 'end_current_prev' in over_params.keys():
            end_current_prev = over_params['end_current_prev']
        else:
            end_current_prev = params[end_current_prev_label]
        if 'constant_current' in over_params.keys():
            constant_current = over_params['constant_current']
        else:
            constant_current = params[constant_current_label]
        if 'features' in over_params.keys():
            features = over_params['features']
        else:
            features = params[features_label]

        if 'voltage' in over_params.keys():
            voltage = over_params['voltage']
        else:
            voltage = params["voltage_flat"]
        with tf.GradientTape(persistent=True) as tape3:
            if not no_cyc_der:
                tape3.watch(cycles)
            if not no_end_current_prev_der:
                tape3.watch(end_current_prev)
            if not no_constant_current_der:
                tape3.watch(constant_current)
            if not no_features_der:
                tape3.watch(features)
            if not no_vol_der:
                tape3.watch(voltage)

            with tf.GradientTape(persistent=True) as tape2:
                if not no_cyc_der:
                    tape2.watch(cycles)
                if not no_end_current_prev_der:
                    tape2.watch(end_current_prev)
                if not no_constant_current_der:
                    tape2.watch(constant_current)
                if not no_features_der:
                    tape2.watch(features)
                if not no_vol_der:
                    tape2.watch(voltage)

                res = tf.reshape(nn(params, over_params), [-1, 1])

            if not no_cyc_der:
                derivatives['dCyc'] = tape2.batch_jacobian(
                    source=cycles,
                    target=res
                )[:, 0, :]
            if not no_end_current_prev_der:
                derivatives['d_end_current_prev'] = tape2.batch_jacobian(
                    source=end_current_prev,
                    target=res
                )[:, 0, :]

            if not no_constant_current_der:
                derivatives['d_constant_current'] = tape2.batch_jacobian(
                    source=constant_current,
                    target=res
                )[:, 0, :]
            if not no_features_der:
                derivatives['dFeatures'] = tape2.batch_jacobian(
                    source=features,
                    target=res
                )[:, 0, :]
            if not no_vol_der:
                derivatives['dVol'] = tape2.batch_jacobian(
                    source=voltage,
                    target=res
                )[:, 0, :]

            del tape2

        if not no_cyc_der:
            derivatives['d2Cyc'] = tape3.batch_jacobian(
                source=cycles,
                target=derivatives['dCyc']
            )[:, 0, :]
        if not no_end_current_prev_der:
            derivatives['d2_end_current_prev'] = tape3.batch_jacobian(
                source=end_current_prev,
                target=derivatives['d_end_current_prev']
            )[:, 0, :]
        if not no_constant_current_der:
            derivatives['d2_constant_current'] = tape3.batch_jacobian(
                source=constant_current,
                target=derivatives['d_constant_current']
            )[:, 0, :]
        if not no_features_der:
            derivatives['d2Features'] = tape3.batch_jacobian(
                source=features,
                target=derivatives['dFeatures']
            )
        if not no_vol_der:
            derivatives['d2Vol'] = tape3.batch_jacobian(
                source=voltage,
                target=derivatives['dVol']
            )[:, 0, :]

        del tape3
        return res, derivatives

    # add voltage dependence ([cyc] -> [cyc, vol])
    def add_volt_dep(self, thing, params, dim = 1):
        return tf.reshape(
            tf.tile(
                tf.expand_dims(
                    thing,
                    axis=1
                ),
                [1, params["voltage_count"],1]
            ),
            [params["batch_count"] * params["voltage_count"], dim]
        )

    # add cycle dependence ([vol] -> [cyc, vol])
    def add_cyc_dep(self, thing, params, dim = 1):
        return tf.reshape(
            tf.tile(
                tf.expand_dims(
                    thing,
                    axis=0
                ),
                [params["batch_count"], 1,1]
            ),
            [params["batch_count"] * params["voltage_count"], dim]
        )

    def call(self, x, training=False):

        cycles = x[0]  # matrix; dim: [batch, 1]
        constant_current = x[1]  # matrix; dim: [batch, 1]
        end_current_prev = x[2]  # matrix; dim: [batch, 1]
        end_voltage_prev = x[3]  # matrix; dim: [batch, 1]

        indecies = x[4]  # batch of index; dim: [batch]
        meas_cycles = x[5]  # batch of cycles; dim: [batch]
        voltage_vector = x[6] # dim: [voltages]

        features, mean, log_sig = self.dictionary(indecies, training=training)
        # duplicate cycles and others for all the voltages
        # dimensions are now [batch, voltages, features]
        batch_count = cycles.shape[0]
        voltage_count = voltage_vector.shape[0]
        count_dict = {
            "batch_count": batch_count,
            "voltage_count": voltage_count
        }

        params = {
            "batch_count": batch_count,
            "voltage_count": voltage_count,
            "cycles_flat": self.add_volt_dep(
                cycles,
                count_dict
            ),
            "constant_current_flat": self.add_volt_dep(
                constant_current,
                count_dict
            ),
            "end_current_prev_flat": self.add_volt_dep(
                end_current_prev,
                count_dict
            ),
            "end_voltage_prev_flat": self.add_volt_dep(
                end_voltage_prev,
                count_dict
            ),
            "features_flat": self.add_volt_dep(
                features,
                count_dict,
                dim = self.width
            ),
            "voltage_flat": self.add_cyc_dep(
                tf.expand_dims(voltage_vector, axis = 1),
                count_dict
            ),

            "cycles": cycles,
            "constant_current": constant_current,
            "end_current_prev": end_current_prev,
            "end_voltage_prev": end_voltage_prev,
            "features": features
        }

        if training:

            var_cyc = tf.expand_dims(meas_cycles, axis=1) - cycles
            var_cyc_squared = tf.square(var_cyc)

            ''' discharge capacity '''
            cap, cap_der = self.create_derivatives(
                self.dchg_cap_part2,
                params,
                over_params={},
                scalar = False
            )
            cap = tf.reshape(cap, [-1, voltage_count])

            pred_cap = (
                cap + var_cyc * tf.reshape(
                    cap_der['dCyc'], [-1, voltage_count]
                ) + var_cyc_squared * tf.reshape(
                    cap_der['d2Cyc'], [-1, voltage_count]
                )
            )


            '''resistance derivatives '''
            r, r_der = self.create_derivatives(self.r, params, over_params={}, no_vol_der=True)
            r = tf.reshape(r, [-1])

            #NOTE(sam): this is an example of a forall. (for all voltages, and all cell features)

            n_sample = 64
            sampled_voltages = tf.random.uniform(minval=2.5, maxval=5., shape=[n_sample, 1])
            sampled_cycles = tf.random.uniform(minval=-10., maxval=10., shape=[n_sample, 1])
            sampled_constant_current = tf.random.uniform(minval=-10., maxval=10., shape=[n_sample, 1])

            sampled_features = self.dictionary.sample(n_sample)
            soc_1, soc_1_der = self.create_derivatives(
                self.soc,
                params,
                over_params={
                    'voltage':sampled_voltages,
                    'features': sampled_features,
                },
                scalar=True,
                no_cyc_der=True,
                no_constant_current_der=True,
                no_end_current_prev_der=True,
            )

            soc_loss = .001 * (
                    tf.reduce_mean(tf.square(soc_1)) +
                    1000. * tf.reduce_mean(tf.nn.relu(-soc_1))+
                    tf.reduce_mean(tf.nn.relu(-soc_1_der['dVol']))
            )
            theoretical_cap = self.theoretical_cap(
                params,
                over_params={
                    'cycles': sampled_cycles,
                    'constant_current':sampled_constant_current,
                    'features':sampled_features
                },
            )

            theo_cap_loss = 1. * (
                    tf.reduce_mean(tf.nn.relu(theoretical_cap-1.))
            )


            const_f_loss =  (
                    tf.reduce_mean(tf.square(cap_der['dFeatures']))
                    + tf.reduce_mean(tf.square(r_der['dFeatures']))

            )

            smooth_f_loss =  (
                    tf.reduce_mean(tf.square(cap_der['d2Features']))
                    + tf.reduce_mean(tf.square(r_der['d2Features']))

            )

            kl_loss = tf.reduce_mean(
                0.5 * (tf.exp(log_sig) + tf.square(mean) - 1. - log_sig)
            )


            # TODO(sam): figure out how to do foralls with derivatives.
            # mono_loss = fit_args['mono_coeff'] * (
            #     tf.reduce_mean(tf.nn.relu(-cap))  # penalizes negative capacities
            #     + tf.reduce_mean(tf.nn.relu(cap_der['dCyc'])) # shouldn't increase
            #     + tf.reduce_mean(tf.nn.relu(cap_der['d_chg_rate']))
            #     + tf.reduce_mean(tf.nn.relu(cap_der['d_dchg_rate']))
            #     + tf.reduce_mean(tf.nn.relu(cap_der['dVol']))
            #
            #     + 10. * (
            #         tf.reduce_mean(tf.nn.relu(-r))
            #         + tf.reduce_mean(tf.nn.relu(-eq_vol))
            #         # resistance should not decrease.
            #         + 10  * tf.reduce_mean(tf.abs(r_der['dCyc']))
            #         + 10. * (
            #             tf.reduce_mean(tf.abs(eq_vol_der['dCyc']))
            #             # equilibrium voltage should not change much
            #             # TODO is this correct?
            #             + tf.reduce_mean(tf.abs(eq_vol_der["d_chg_rate"]))
            #             + tf.reduce_mean(tf.abs(eq_vol_der["d_dchg_rate"]))
            #         )
            #     )
            # )

            # smooth_loss = fit_args['smooth_coeff'] * (
            #     tf.reduce_mean(tf.square(tf.nn.relu(cap_der['d2Cyc']))
            #     + 0.02 * tf.square(tf.nn.relu(-cap_der['d2Cyc'])))
            #     + tf.reduce_mean(
            #         tf.square(tf.nn.relu(cap_der['d2_chg_rate']))
            #         + 0.02 * tf.square(tf.nn.relu(-cap_der['d2_chg_rate']))
            #         + tf.square(tf.nn.relu(cap_der['d2_dchg_rate']))
            #         + 0.02 * tf.square(tf.nn.relu(-cap_der['d2_dchg_rate']))
            #         + tf.square(tf.nn.relu(cap_der['d2Vol']))
            #         + 0.02 * tf.square(tf.nn.relu(-cap_der['d2Vol']))
            #     )
            #
            #     # enforces smoothness of resistance;
            #     # more ok to accelerate UPWARDS
            #     + 10. * tf.reduce_mean(tf.square(tf.nn.relu(-r_der['d2Cyc']))
            #     + 0.5 * tf.square(tf.nn.relu(r_der['d2Cyc'])))
            #     + 1. * tf.reduce_mean(tf.square((eq_vol_der["d_chg_rate"])))
            #     + 1. * tf.reduce_mean(tf.square((eq_vol_der["d_dchg_rate"])))
            #     + 1. * tf.reduce_mean(tf.square((eq_vol_der['d2Cyc'])))
            # )

            return {
                "pred_cap": pred_cap,
                "soc_loss": soc_loss,
                "theo_cap_loss": theo_cap_loss,
                "const_f_loss": const_f_loss,
                "smooth_f_loss": smooth_f_loss,
                "kl_loss":kl_loss,
                "pred_r": tf.reshape(r, [-1]),
                "mean": mean,
                "log_sig": log_sig,
                "cap_der": cap_der,
            }

        else:
            pred_cap = tf.reshape(
                self.dchg_cap_part2(params),
                [-1, voltage_vector.shape[0]]
            )
            pred_r = self.r(params)
            #shift = self.shift(params)

            return {
                "pred_cap": pred_cap,
                "pred_r": pred_r,
                #"shift": shift,
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


    def sample(self, n_sample):
        eps = tf.random.normal(
            shape=[n_sample, self.num_features])
        mean = self.kernel[:, :self.num_features]
        log_sig = self.kernel[:, self.num_features:]
        indecies = tf.random.uniform(maxval=self.num_keys, shape=[n_sample],dtype=tf.int32)

        fetched_mean = tf.gather(mean, indecies, axis=0)
        fetched_log_sig = tf.gather(log_sig, indecies, axis=0)
        fetched_features = fetched_mean + tf.exp(fetched_log_sig / 2.) * eps
        return tf.stop_gradient(fetched_features)
