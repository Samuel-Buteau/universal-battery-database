import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer
from enum import Enum

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
class Inequality(Enum):
    LessThan = 1
    GreaterThan = 2
    Equals = 3

class Level(Enum):
    Strong = 1
    Proportional = 2

class Target(Enum):
    Small = 1
    Big = 2


def incentive_inequality(A,symbol, B, level):
    """

    :param A: The first object
    :param symbol: the relationship we want
    (either Inequality.LessThan or Inequality.GreaterThan or Inequality.Equal)
        Inequality.LessThan (i.e. A < B) means that A should be less than B,
        Inequality.GreaterThan (i.e. A > B) means that A should be greater than B.
        Inequality.Equals (i.e. A = B) means that A should be equal to B.

    :param B: The second object
    :param level:  determines the relationship between the incentive strength and the values of A and B.
    (either Level.Strong or Level.Proportional)
        Level.Strong means that we take the L1 norm, so the gradient trying to satisfy
        'A symbol B' will be constant no matter how far from 'A symbol B' we are.
        Level.Proportional means that we take the L2 norm, so the gradient trying to satisfy
        'A symbol B' will be proportional to how far from 'A symbol B' we are.


    :return:
    Returns a loss which will give the model an incentive to satisfy 'A symbol B', with level.


    """
    if symbol == Inequality.LessThan:
        intermediate = tf.nn.relu(A - B)
    elif symbol == Inequality.GreaterThan:
        intermediate = tf.nn.relu(B - A)
    elif symbol == Inequality.Equals:
        intermediate = tf.abs(A - B)
    else:
        raise Exception("not yet implemented inequality symbol {}.".format(symbol))

    if level == Level.Strong:
        return intermediate
    elif level == Level.Proportional:
        return tf.square(intermediate)
    else:
        raise Exception("not yet implemented incentive level {}.".format(level))

def incentive_magnitude(A, target, level):
    """

        :param A: The object
        :param target: the direction we want
        (either Target.Small or Target.Big)
            Target.Small  means that the norm of A should be as small as possible,
            Target.Big  means that the norm of A should be as big as possible,

        :param level:  determines the relationship between the incentive strength and the value of A.
        (either Level.Strong or Level.Proportional)
            Level.Strong means that we take the L1 norm,
            so the gradient trying to push the absolute value of A to target will be constant.
            Level.Proportional means that we take the L2 norm,
            so the gradient trying to push the absolute value of A to target will be proportional to the absolute value of A.



        :return:
        Returns a loss which will give the model an incentive to push the absolute value of A to target.


        """

    A_prime = tf.abs(A)

    if target == Target.Small:
        multiplier = 1.
    elif target == Target.Big:
        multiplier = -1.

    else:
        raise Exception('not yet implemented target {}'.format(target))

    if level == Level.Strong:
        A_prime =  A_prime
    elif level == Level.Proportional:
        A_prime = tf.square(A_prime)
    else:
        raise Exception('not yet implemented level {}'.format(level))

    return multiplier * A_prime


def incentive_combine(As):
    """

        :param As: A list of tuples. Each tuple contains a coefficient and a tensor of losses corresponding to incentives.

        :return:
        Returns a combined loss (single number) which will incentivize all the individual incentive tensors with weights given by the coefficients.

    """

    return sum([a[0]* tf.reduce_mean(a[1]) for a in As])

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
        return self.theoretical_cap_direct(params, over_params={'norm_cycle':norm_cycle, 'constant_current':constant_current, 'cell_feat':cell_feat})

    def theoretical_cap_direct(self, params, over_params=None):
        if over_params is None:
            over_params = {}

        if 'norm_cycle' in over_params.keys():
            norm_cycle = over_params["norm_cycle"]

        if 'constant_current' in over_params.keys():
            constant_current = over_params["constant_current"]

        if 'cell_feat' in over_params.keys():
            cell_feat = over_params["cell_feat"]


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

        return self.eq_voltage(
            params,
            over_params={
                'voltage':end_voltage_prev,
                'current':end_current_prev,
                'resistance':self.r(params, over_params),
            }
        )

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
        return self.eq_voltage(
            params,
            over_params={
                'voltage':voltage,
                'current':constant_current,
                'resistance':r_flat,
            }
        )

    def eq_voltage(self, params, over_params=None):
        if over_params is None:
            over_params = {}
        if 'voltage' in over_params.keys():
            voltage = over_params["voltage"]

        if 'current' in over_params.keys():
            current = over_params['current']
        if 'resistance' in over_params.keys():
            resistance = over_params['resistance']

        return voltage - current * resistance


    def soc(self, params, over_params=None, scalar = True):
        if over_params is None:
            over_params = {}

        cell_feat = self.cell_features(params, over_params, scalar)
        if 'voltage' in over_params.keys():
            voltage = over_params['voltage']
        else:
            raise Exception("tried to call soc without any voltage. please add a 'voltage':value to over_param")

        return self.soc_direct(params, over_params = {'voltage':voltage, 'cell_feat':cell_feat})

    def soc_direct(self, params, over_params=None):
        if over_params is None:
            over_params = {}

        if 'voltage' in over_params.keys():
            voltage = over_params['voltage']

        if 'cell_feat' in over_params.keys():
            cell_feat = over_params['cell_feat']

        dependencies = (
            voltage,
            cell_feat
        )
        return tf.nn.elu(self.nn_call(self.nn_soc, dependencies))



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


    def cc_capacity_part2(self, params, over_params=None):
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


    def cv_capacity(self, params, over_params=None):
        if over_params is None:
            over_params = {}

        theoretical_cap_cc = self.theoretical_cap(params, {})

        new_over_param = {}
        for key in over_params.keys():
            new_over_param[key] = over_params[key]
        new_over_param['voltage'] = self.eq_voltage_0(
            params,
            over_params
        )
        cc_soc_0 = self.soc(params, new_over_param, scalar=True)

        new_over_param = {}
        for key in over_params.keys():
            new_over_param[key] = over_params[key]
        new_over_param['voltage'] = self.eq_voltage(
            params,
            over_params = {
                'voltage':params['end_voltage'],
                'current':params['constant_current'],
                'resistance':self.r(params, over_params)
            }
        )

        cc_soc_1 = self.soc(params, new_over_param, scalar=True)

        cc_capacity = theoretical_cap_cc * (cc_soc_1 - cc_soc_0)

        cell_feat = self.cell_features(params, {})
        theoretical_cap_cv = self.theoretical_cap_direct(
            params,
            over_params={
                'norm_cycle':self.add_current_dep(self.norm_cycle(params, {}), params),
                'constant_current':params['cv_current_flat'],
                'cell_feat': self.add_current_dep(cell_feat, params, cell_feat.shape[1]),
            }
        )

        theoretical_cap_cv_reshaped = tf.reshape(theoretical_cap_cv, [-1, params['current_count']])

        cv_soc_1 = self.soc_direct(
            params,
            over_params={
                'voltage':self.eq_voltage(
                    params,
                    over_params = {
                        'voltage':self.add_current_dep(params['end_voltage'], params),
                        'current':params['cv_current_flat'],
                        'resistance':self.add_current_dep(self.r(params, over_params), params),

                    }
                ),
                'cell_feat':self.add_current_dep(self.cell_features(params, {}),params, cell_feat.shape[1]),
            },
        )

        cv_soc_1_reshaped = tf.reshape(cv_soc_1, [-1, params['current_count']])

        cv_soc_0_reshaped = tf.concat(
            (
            cc_soc_1,
            cv_soc_1_reshaped[:,1:],
            ),
            axis=1,
        )

        cv_capacity_deltas = theoretical_cap_cv_reshaped * (cv_soc_1_reshaped - cv_soc_0_reshaped)
        cv_capacity_cumulative = tf.cumsum(cv_capacity_deltas, axis=1)

        return tf.reshape(cv_capacity_cumulative, [-1,1]) + self.add_current_dep(cc_capacity, params)





    def nn_call(self, nn_func, dependencies):
        centers = nn_func['initial'](
            tf.concat(dependencies, axis=1)
        )
        for d in nn_func['bulk']:
            centers = d(centers)
        return nn_func['final'](centers)




    def create_derivatives(self, nn, params, over_params={}, scalar = True,
                           voltage_der=0,
                           cycles_der=0,
                           features_der=0,
                           constant_current_der=0,
                           end_current_prev_der=0
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
        with tf.GradientTape(persistent=True) as tape_d3:
            if cycles_der >= 3:
                tape_d3.watch(cycles)
            if end_current_prev_der >= 3:
                tape_d3.watch(end_current_prev)
            if constant_current_der >= 3:
                tape_d3.watch(constant_current)
            if features_der >= 3:
                tape_d3.watch(features)
            if voltage_der >= 3:
                tape_d3.watch(voltage)


            with tf.GradientTape(persistent=True) as tape_d2:
                if cycles_der >= 2:
                    tape_d2.watch(cycles)
                if end_current_prev_der >= 2:
                    tape_d2.watch(end_current_prev)
                if constant_current_der >= 2:
                    tape_d2.watch(constant_current)
                if features_der >= 2:
                    tape_d2.watch(features)
                if voltage_der >= 2:
                    tape_d2.watch(voltage)

                with tf.GradientTape(persistent=True) as tape_d1:
                    if cycles_der >= 1:
                        tape_d1.watch(cycles)
                    if end_current_prev_der >= 1:
                        tape_d1.watch(end_current_prev)
                    if constant_current_der >= 1:
                        tape_d1.watch(constant_current)
                    if features_der >= 1:
                        tape_d1.watch(features)
                    if voltage_der >= 1:
                        tape_d1.watch(voltage)

                    res = tf.reshape(nn(params, over_params), [-1, 1])

                if cycles_der >= 1:
                    derivatives['d_cycles'] = tape_d1.batch_jacobian(
                        source=cycles,
                        target=res
                    )[:, 0, :]
                if end_current_prev_der >= 1:
                    derivatives['d_end_current_prev'] = tape_d1.batch_jacobian(
                        source=end_current_prev,
                        target=res
                    )[:, 0, :]

                if constant_current_der >= 1:
                    derivatives['d_constant_current'] = tape_d1.batch_jacobian(
                        source=constant_current,
                        target=res
                    )[:, 0, :]
                if features_der >= 1:
                    derivatives['d_features'] = tape_d1.batch_jacobian(
                        source=features,
                        target=res
                    )[:, 0, :]
                if voltage_der >= 1:
                    derivatives['d_voltage'] = tape_d1.batch_jacobian(
                        source=voltage,
                        target=res
                    )[:, 0, :]

                del tape_d1

            if cycles_der >= 2:
                derivatives['d2_cycles'] = tape_d2.batch_jacobian(
                    source=cycles,
                    target=derivatives['d_cycles']
                )[:, 0, :]
            if end_current_prev_der >= 2:
                derivatives['d2_end_current_prev'] = tape_d2.batch_jacobian(
                    source=end_current_prev,
                    target=derivatives['d_end_current_prev']
                )[:, 0, :]
            if constant_current_der >= 2:
                derivatives['d2_constant_current'] = tape_d2.batch_jacobian(
                    source=constant_current,
                    target=derivatives['d_constant_current']
                )[:, 0, :]
            if features_der >= 2:
                derivatives['d2_features'] = tape_d2.batch_jacobian(
                    source=features,
                    target=derivatives['d_features']
                )
            if voltage_der >= 2:
                derivatives['d2_voltage'] = tape_d2.batch_jacobian(
                    source=voltage,
                    target=derivatives['d_voltage']
                )[:, 0, :]

            del tape_d2

        if cycles_der >= 3:
            derivatives['d3_cycles'] = tape_d3.batch_jacobian(
                source=cycles,
                target=derivatives['d2_cycles']
            )[:, 0, :]
        if end_current_prev_der >= 3:
            derivatives['d3_end_current_prev'] = tape_d3.batch_jacobian(
                source=end_current_prev,
                target=derivatives['d2_end_current_prev']
            )[:, 0, :]
        if constant_current_der >= 3:
            derivatives['d3_constant_current'] = tape_d3.batch_jacobian(
                source=constant_current,
                target=derivatives['d2_constant_current']
            )[:, 0, :]
        if features_der >= 3:
            derivatives['d3_features'] = tape_d3.batch_jacobian(
                source=features,
                target=derivatives['d2_features']
            )
        if voltage_der >= 3:
            derivatives['d3_voltage'] = tape_d3.batch_jacobian(
                source=voltage,
                target=derivatives['d2_voltage']
            )[:, 0, :]

        del tape_d3

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

    def add_current_dep(self, thing, params, dim = 1):
        return tf.reshape(
            tf.tile(
                tf.expand_dims(
                    thing,
                    axis=1
                ),
                [1, params["current_count"],1]
            ),
            [params["batch_count"] * params["current_count"], dim]
        )


    def call(self, x, training=False):

        cycles = x[0]  # matrix; dim: [batch, 1]
        constant_current = x[1]  # matrix; dim: [batch, 1]
        end_current_prev = x[2]  # matrix; dim: [batch, 1]
        end_voltage_prev = x[3]  # matrix; dim: [batch, 1]
        end_voltage = x[4]  # matrix; dim: [batch, 1]

        indecies = x[5]  # batch of index; dim: [batch]
        voltage_tensor = x[6] # dim: [batch, voltages]
        current_tensor = x[7]  # dim: [batch, voltages]

        features, mean, log_sig = self.dictionary(indecies, training=training)
        # duplicate cycles and others for all the voltages
        # dimensions are now [batch, voltages, features]
        batch_count = cycles.shape[0]
        voltage_count = voltage_tensor.shape[1]
        current_count = current_tensor.shape[1]
        count_dict = {
            "batch_count": batch_count,
            "voltage_count": voltage_count,
            "current_count": current_count,
        }

        params = {
            "batch_count": batch_count,
            "voltage_count": voltage_count,
            "current_count": current_count,

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
            "voltage_flat": tf.reshape(voltage_tensor, [-1, 1]),
            "cv_current_flat": tf.reshape(current_tensor, [-1, 1]),

            "cycles": cycles,
            "constant_current": constant_current,
            "end_current_prev": end_current_prev,
            "end_voltage_prev": end_voltage_prev,
            "features": features,
            "end_voltage": end_voltage,

        }
        cc_capacity = self.cc_capacity_part2(params, over_params={})
        pred_cc_capacity = tf.reshape(cc_capacity, [-1, voltage_count])

        cv_capacity = self.cv_capacity(params, over_params={})
        pred_cv_capacity = tf.reshape(cv_capacity, [-1, current_count])

        if training:


            #NOTE(sam): this is an example of a forall. (for all voltages, and all cell features)
            n_sample = 64
            sampled_voltages = tf.random.uniform(minval=2.5, maxval=5., shape=[n_sample, 1])
            sampled_cycles = tf.random.uniform(minval=-10., maxval=10., shape=[n_sample, 1])
            sampled_constant_current = tf.random.uniform(minval=0.001, maxval=10., shape=[n_sample, 1])
            sampled_features = self.dictionary.sample(n_sample)

            soc_1, soc_1_der = self.create_derivatives(
                self.soc,
                params,
                over_params={
                    'voltage':sampled_voltages,
                    'features': sampled_features,
                },
                scalar=True,
                voltage_der=2,
                features_der=2,
            )

            soc_loss = .001 * incentive_combine([
                (1., incentive_magnitude(
                            soc_1,
                            Target.Small,
                            Level.Proportional
                        )
                ),

                (1000., incentive_inequality(
                            soc_1, Inequality.GreaterThan, 0,
                            Level.Strong
                        )
                ),
                (1., incentive_inequality(
                            soc_1_der['d_voltage'], Inequality.GreaterThan, 0,
                            Level.Strong
                        )
                ),
                (.01, incentive_magnitude(
                            soc_1_der['d2_voltage'],
                            Target.Small,
                            Level.Proportional
                        )
                ),
                (.01, incentive_magnitude(
                            soc_1_der['d_features'],
                            Target.Small,
                            Level.Proportional
                        )
                ),
                (.01, incentive_magnitude(
                            soc_1_der['d2_features'],
                            Target.Small,
                            Level.Strong
                        )
                )
            ]

            )

            theoretical_cap, theoretical_cap_der = self.create_derivatives(
                self.theoretical_cap,
                params,
                over_params={
                    'cycles': sampled_cycles,
                    'constant_current':sampled_constant_current,
                    'features':sampled_features
                },
                scalar=True,
                cycles_der=3,
                constant_current_der=2,
                features_der=2,
            )

            theo_cap_loss = .001 * incentive_combine(
                [
                    (1.,incentive_inequality(
                            theoretical_cap, Inequality.GreaterThan, 0,
                            Level.Strong
                        )
                    ),
                    (1.,incentive_inequality(
                            theoretical_cap, Inequality.LessThan, 1,
                            Level.Strong
                        )
                    ),
                    (1.,incentive_inequality(
                            theoretical_cap_der['d_cycles'], Inequality.LessThan, 0,
                            Level.Proportional
                        ) # we want cap to diminish with cycle number
                    ),
                    (.1, incentive_inequality(
                        theoretical_cap_der['d2_cycles'], Inequality.LessThan, 0,
                        Level.Proportional
                    )  # we want cap to diminish with cycle number
                     ),

                    (1.,incentive_inequality(
                            theoretical_cap_der['d_constant_current'], Inequality.LessThan, 0,
                            Level.Proportional
                        ) # we want cap to diminish with constant_current (if constant current is positive)
                    ),
                    (100., incentive_magnitude(
                        theoretical_cap_der['d_cycles'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),

                    (100.,incentive_magnitude(
                            theoretical_cap_der['d2_cycles'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (100., incentive_magnitude(
                            theoretical_cap_der['d3_cycles'],
                            Target.Small,
                            Level.Proportional
                        )
                     ),

                    (1.,incentive_magnitude(
                            theoretical_cap_der['d2_constant_current'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (1.,incentive_magnitude(
                            theoretical_cap_der['d_features'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (1.,incentive_magnitude(
                            theoretical_cap_der['d2_features'],
                            Target.Small,
                            Level.Strong
                        )
                    )
                ]
            )

            r, r_der = self.create_derivatives(
                self.r,
                params,
                over_params={
                    'cycles': sampled_cycles,
                    'features': sampled_features
                },
                cycles_der=3,
                features_der=2,
                scalar=True,
            )

            r_loss = .001 * incentive_combine(
                [
                    (1.,incentive_inequality(
                            r, Inequality.GreaterThan, 0,
                            Level.Strong
                        )
                    ),
                    (10.,incentive_magnitude(
                            r_der['d2_cycles'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),

                    (100., incentive_magnitude(
                            r_der['d3_cycles'],
                            Target.Small,
                            Level.Proportional
                        )
                     ),

                    (1.,incentive_magnitude(
                            r_der['d_features'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (1.,incentive_magnitude(
                            r_der['d2_features'],
                            Target.Small,
                            Level.Strong
                        )
                    )
                ]
            )


            kl_loss = tf.reduce_mean(
                0.5 * (tf.exp(log_sig) + tf.square(mean) - 1. - log_sig)
            )



            return {
                "pred_cc_capacity": pred_cc_capacity,
                "pred_cv_capacity": pred_cv_capacity,
                "soc_loss": soc_loss,
                "theo_cap_loss": theo_cap_loss,
                "r_loss": r_loss,
                "kl_loss":kl_loss,
            }

        else:

            pred_r = self.r(params)

            return {
                "pred_cc_capacity": pred_cc_capacity,
                "pred_cv_capacity": pred_cv_capacity,
                "pred_r": pred_r,
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
