from enum import Enum

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
    return {'initial': initial, 'bulk': bulk, 'final': final}


"""
The call convention for the neural networks is a bit complex but it makes the 
code easier to use.

First, there are some parameters that are from the real data (e.g. cycle 
number, voltage, etc.)
these are passed to every call as a dictionary.

Second, all the parameters that are used in the body of the function can be 
overridden by passing them. If they are passed as None, 
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


def incentive_inequality(A, symbol, B, level):
    """

    :param A: The first object
    :param symbol: the relationship we want
    (either Inequality.LessThan or Inequality.GreaterThan or Inequality.Equal)
        Inequality.LessThan (i.e. A < B) means that A should be less than B,
        Inequality.GreaterThan (i.e. A > B) means that A should be greater
        than B.
        Inequality.Equals (i.e. A = B) means that A should be equal to B.

    :param B: The second object
    :param level:  determines the relationship between the incentive strength
    and the values of A and B.
    (either Level.Strong or Level.Proportional)
        Level.Strong means that we take the L1 norm, so the gradient trying
        to satisfy
        'A symbol B' will be constant no matter how far from 'A symbol B' we
        are.
        Level.Proportional means that we take the L2 norm, so the gradient
        trying to satisfy
        'A symbol B' will be proportional to how far from 'A symbol B' we are.


    :return:
    Returns a loss which will give the model an incentive to satisfy 'A
    symbol B', with level.


    """
    if symbol == Inequality.LessThan:
        intermediate = tf.nn.relu(A - B)
    elif symbol == Inequality.GreaterThan:
        intermediate = tf.nn.relu(B - A)
    elif symbol == Inequality.Equals:
        intermediate = tf.abs(A - B)
    else:
        raise Exception(
            "not yet implemented inequality symbol {}.".format(symbol))

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
            Target.Small  means that the norm of A should be as small as
            possible,
            Target.Big  means that the norm of A should be as big as possible,

        :param level:  determines the relationship between the incentive
        strength and the value of A.
        (either Level.Strong or Level.Proportional)
            Level.Strong means that we take the L1 norm,
            so the gradient trying to push the absolute value of A to target
            will be constant.
            Level.Proportional means that we take the L2 norm,
            so the gradient trying to push the absolute value of A to target
            will be proportional to the absolute value of A.



        :return:
        Returns a loss which will give the model an incentive to push the
        absolute value of A to target.


        """

    A_prime = tf.abs(A)

    if target == Target.Small:
        multiplier = 1.
    elif target == Target.Big:
        multiplier = -1.

    else:
        raise Exception('not yet implemented target {}'.format(target))

    if level == Level.Strong:
        A_prime = A_prime
    elif level == Level.Proportional:
        A_prime = tf.square(A_prime)
    else:
        raise Exception('not yet implemented level {}'.format(level))

    return multiplier * A_prime


def incentive_combine(As):
    """

        :param As: A list of tuples. Each tuple contains a coefficient and a
        tensor of losses corresponding to incentives.

        :return:
        Returns a combined loss (single number) which will incentivize all
        the individual incentive tensors with weights given by the coefficients.

    """

    return sum([a[0] * tf.reduce_mean(a[1]) for a in As])


class DegradationModel(Model):

    def __init__(self, num_keys, depth, width):
        super(DegradationModel, self).__init__()

        self.nn_r = feedforward_nn_parameters(depth, width)
        self.nn_theoretical_cap = feedforward_nn_parameters(depth, width)
        self.nn_soc = feedforward_nn_parameters(depth, width)
        self.nn_shift = feedforward_nn_parameters(depth, width)

        self.dictionary = DictionaryLayer(num_features = width,
                                          num_keys = num_keys)

        self.width = width
        self.num_keys = num_keys

    # Begin: nn application functions ==========================================

    def eq_voltage_direct(self, voltage, current, resistance):
        return voltage - current * resistance

    def soc_direct(self, voltage, shift, cell_features):
        dependencies = (
            voltage,
            shift,
            cell_features
        )
        return tf.nn.elu(self.nn_call(self.nn_soc, dependencies))

    def theoretical_cap_direct(self, norm_cycle, current, cell_features):
        dependencies = (
            norm_cycle,
            tf.abs(current),
            cell_features
        )
        return tf.nn.elu(self.nn_call(self.nn_theoretical_cap, dependencies))

    def shift_direct(self, norm_cycle, current, cell_features):
        dependencies = (
            norm_cycle,
            tf.abs(current),
            cell_features
        )
        return (self.nn_call(self.nn_shift, dependencies))

    def r_direct(self, norm_cycle, cell_features):
        dependencies = (
            norm_cycle,
            cell_features,
        )
        return tf.nn.elu(self.nn_call(self.nn_r, dependencies))

    def norm_constant_direct(self, features):
        return features[:, 0:1]

    def cell_features_direct(self, features):
        return features[:, 1:]

    def norm_cycle_direct(self, cycle, norm_constant):
        return cycle * (1e-10 + tf.exp(-norm_constant))

    def norm_cycle(self, params):
        return self.norm_cycle_direct(
            norm_constant = self.norm_constant_direct(params['features']),
            cycle = params['cycle']
        )

    def soc_for_derivative(self, params):
        return self.soc_direct(
            cell_features = self.cell_features_direct(
                features = params['features']),
            voltage = params['voltage'],
            shift = params['shift']
        )

    def theoretical_cap_for_derivative(self, params):
        return self.theoretical_cap_direct(
            norm_cycle = self.norm_cycle(params = {
                'cycle':    params['cycle'],
                'features': params['features']
            }
            ),
            current = params['current'],
            cell_features = self.cell_features_direct(
                features = params['features'])
        )

    def shift_for_derivative(self, params):
        return self.shift_direct(
            norm_cycle = self.norm_cycle(params = {
                'cycle':    params['cycle'],
                'features': params['features']
            }
            ),
            current = params['current'],
            cell_features = self.cell_features_direct(
                features = params['features']),
        )

    def r_for_derivative(self, params):
        return self.r_direct(
            norm_cycle = self.norm_cycle(params = {
                'cycle':    params['cycle'],
                'features': params['features']
            }
            ),
            cell_features = self.cell_features_direct(
                features = params['features']),
        )

    def cc_capacity_part2(self, params):
        norm_constant = self.norm_constant_direct(features = params['features'])
        norm_cycle = self.norm_cycle_direct(
            cycle = params['cycle'],
            norm_constant = norm_constant
        )

        cell_features = self.cell_features_direct(features = params['features'])
        theoretical_cap = self.theoretical_cap_direct(
            norm_cycle = norm_cycle,
            current = params['constant_current'],
            cell_features = cell_features,
        )
        shift = self.shift_direct(
            norm_cycle = norm_cycle,
            current = params['constant_current'],
            cell_features = cell_features
        )

        resistance = self.r_direct(
            norm_cycle = norm_cycle,
            cell_features = cell_features,
        )

        eq_voltage_0 = self.eq_voltage_direct(
            voltage = params['end_voltage_prev'],
            current = params['end_current_prev'],
            resistance = resistance,
        )

        soc_0 = self.soc_direct(
            voltage = eq_voltage_0,
            shift = shift,
            cell_features = cell_features
        )

        eq_voltage_1 = self.eq_voltage_direct(
            voltage = params['voltage'],
            current = self.add_volt_dep(params['constant_current'], params),
            resistance = self.add_volt_dep(resistance, params),
        )

        soc_1 = self.soc_direct(
            voltage = eq_voltage_1,
            shift = self.add_volt_dep(shift, params),
            cell_features = self.add_volt_dep(cell_features, params,
                                              cell_features.shape[1]),
        )

        return self.add_volt_dep(theoretical_cap, params) * (
            soc_1 - self.add_volt_dep(soc_0, params))

    def cv_capacity(self, params):
        norm_constant = self.norm_constant_direct(features = params['features'])
        norm_cycle = self.norm_cycle_direct(
            cycle = params['cycle'],
            norm_constant = norm_constant
        )

        cell_features = self.cell_features_direct(features = params['features'])
        cc_shift = self.shift_direct(
            norm_cycle = norm_cycle,
            current = params['constant_current'],
            cell_features = cell_features
        )

        resistance = self.r_direct(
            norm_cycle = norm_cycle,
            cell_features = cell_features,
        )

        eq_voltage_0 = self.eq_voltage_direct(
            voltage = params['end_voltage_prev'],
            current = params['end_current_prev'],
            resistance = resistance,
        )

        soc_0 = self.soc_direct(
            voltage = eq_voltage_0,
            shift = cc_shift,
            cell_features = cell_features
        )

        theoretical_cap = self.theoretical_cap_direct(
            norm_cycle = self.add_current_dep(norm_cycle, params),
            current = params['cv_current'],
            cell_features = self.add_current_dep(cell_features, params,
                                                 cell_features.shape[1]),
        )

        eq_voltage_1 = self.eq_voltage_direct(
            voltage = self.add_current_dep(params['end_voltage'], params),
            current = params['cv_current'],
            resistance = self.add_current_dep(resistance, params),
        )

        cv_shift = self.shift_direct(
            norm_cycle = self.add_current_dep(norm_cycle, params),
            current = params['cv_current'],
            cell_features = self.add_current_dep(cell_features, params,
                                                 cell_features.shape[1]),
        )

        soc_1 = self.soc_direct(
            voltage = eq_voltage_1,
            shift = cv_shift,
            cell_features = self.add_current_dep(cell_features, params,
                                                 cell_features.shape[1]),
        )

        return theoretical_cap * (soc_1 - self.add_current_dep(soc_0, params))

    def nn_call(self, nn_func, dependencies):
        centers = nn_func['initial'](
            tf.concat(dependencies, axis = 1)
        )
        for d in nn_func['bulk']:
            centers = d(centers)
        return nn_func['final'](centers)

    def create_derivatives(self, nn,
                           params,
                           voltage_der = 0,
                           cycle_der = 0,
                           features_der = 0,
                           current_der = 0,
                           shift_der = 0
                           ):
        """
        derivatives will only be taken inside forall statements.
        if auxiliary variables must be given, create a lambda.


        :param nn:
        :param params:
        :param voltage_der:
        :param cycles_der:
        :param features_der:
        :param current_der:
        :param shift_der:
        :return:
        """
        derivatives = {}
        if cycle_der >= 1:
            cycle = params['cycle']
        if voltage_der >= 1:
            voltage = params['voltage']
        if current_der >= 1:
            current = params['current']
        if features_der >= 1:
            features = params['features']
        if shift_der >= 1:
            shift = params['shift']

        with tf.GradientTape(persistent = True) as tape_d3:
            if cycle_der >= 3:
                tape_d3.watch(cycle)
            if voltage_der >= 3:
                tape_d3.watch(voltage)
            if current_der >= 3:
                tape_d3.watch(current)
            if features_der >= 3:
                tape_d3.watch(features)
            if shift_der >= 3:
                tape_d3.watch(shift)

            with tf.GradientTape(persistent = True) as tape_d2:
                if cycle_der >= 2:
                    tape_d2.watch(cycle)
                if voltage_der >= 2:
                    tape_d2.watch(voltage)
                if current_der >= 2:
                    tape_d2.watch(current)
                if features_der >= 2:
                    tape_d2.watch(features)
                if shift_der >= 2:
                    tape_d2.watch(shift)

                with tf.GradientTape(persistent = True) as tape_d1:
                    if cycle_der >= 1:
                        tape_d1.watch(cycle)
                    if voltage_der >= 1:
                        tape_d1.watch(voltage)
                    if current_der >= 1:
                        tape_d1.watch(current)
                    if features_der >= 1:
                        tape_d1.watch(features)
                    if shift_der >= 1:
                        tape_d1.watch(shift)

                    res = tf.reshape(nn(params), [-1, 1])

                if cycle_der >= 1:
                    derivatives['d_cycle'] = tape_d1.batch_jacobian(
                        source = cycle,
                        target = res
                    )[:, 0, :]
                if voltage_der >= 1:
                    derivatives['d_voltage'] = tape_d1.batch_jacobian(
                        source = voltage,
                        target = res
                    )[:, 0, :]
                if current_der >= 1:
                    derivatives['d_current'] = tape_d1.batch_jacobian(
                        source = current,
                        target = res
                    )[:, 0, :]
                if shift_der >= 1:
                    derivatives['d_shift'] = tape_d1.batch_jacobian(
                        source = shift,
                        target = res
                    )[:, 0, :]
                if features_der >= 1:
                    derivatives['d_features'] = tape_d1.batch_jacobian(
                        source = features,
                        target = res
                    )[:, 0, :]

                del tape_d1

            if cycle_der >= 2:
                derivatives['d2_cycle'] = tape_d2.batch_jacobian(
                    source = cycle,
                    target = derivatives['d_cycle']
                )[:, 0, :]
            if voltage_der >= 2:
                derivatives['d2_voltage'] = tape_d2.batch_jacobian(
                    source = voltage,
                    target = derivatives['d_voltage']
                )[:, 0, :]
            if current_der >= 2:
                derivatives['d2_current'] = tape_d2.batch_jacobian(
                    source = current,
                    target = derivatives['d_current']
                )[:, 0, :]
            if shift_der >= 2:
                derivatives['d2_shift'] = tape_d2.batch_jacobian(
                    source = shift,
                    target = derivatives['d_shift']
                )[:, 0, :]

            if features_der >= 2:
                derivatives['d2_features'] = tape_d2.batch_jacobian(
                    source = features,
                    target = derivatives['d_features']
                )

            del tape_d2

        if cycle_der >= 3:
            derivatives['d3_cycle'] = tape_d3.batch_jacobian(
                source = cycle,
                target = derivatives['d2_cycle']
            )[:, 0, :]
        if voltage_der >= 3:
            derivatives['d3_voltage'] = tape_d3.batch_jacobian(
                source = voltage,
                target = derivatives['d2_voltage']
            )[:, 0, :]
        if current_der >= 3:
            derivatives['d3_current'] = tape_d3.batch_jacobian(
                source = current,
                target = derivatives['d2_current']
            )[:, 0, :]
        if shift_der >= 3:
            derivatives['d3_shift'] = tape_d3.batch_jacobian(
                source = shift,
                target = derivatives['d2_shift']
            )[:, 0, :]

        if features_der >= 3:
            derivatives['d3_features'] = tape_d3.batch_jacobian(
                source = features,
                target = derivatives['d2_features']
            )

        del tape_d3

        return res, derivatives

    # add voltage dependence ([cyc] -> [cyc, vol])
    def add_volt_dep(self, thing, params, dim = 1):
        return tf.reshape(
            tf.tile(
                tf.expand_dims(
                    thing,
                    axis = 1
                ),
                [1, params["voltage_count"], 1]
            ),
            [params["batch_count"] * params["voltage_count"], dim]
        )

    def add_current_dep(self, thing, params, dim = 1):
        return tf.reshape(
            tf.tile(
                tf.expand_dims(
                    thing,
                    axis = 1
                ),
                [1, params["current_count"], 1]
            ),
            [params["batch_count"] * params["current_count"], dim]
        )

    def call(self, x, training = False):

        cycle = x[0]  # matrix; dim: [batch, 1]
        constant_current = x[1]  # matrix; dim: [batch, 1]
        end_current_prev = x[2]  # matrix; dim: [batch, 1]
        end_voltage_prev = x[3]  # matrix; dim: [batch, 1]
        end_voltage = x[4]  # matrix; dim: [batch, 1]

        indecies = x[5]  # batch of index; dim: [batch]
        voltage_tensor = x[6]  # dim: [batch, voltages]
        current_tensor = x[7]  # dim: [batch, voltages]

        features, mean, log_sig = self.dictionary(indecies, training = training)
        # duplicate cycles and others for all the voltages
        # dimensions are now [batch, voltages, features]
        batch_count = cycle.shape[0]
        voltage_count = voltage_tensor.shape[1]
        current_count = current_tensor.shape[1]

        params = {
            "batch_count":      batch_count,
            "voltage_count":    voltage_count,
            "current_count":    current_count,

            "voltage":          tf.reshape(voltage_tensor, [-1, 1]),
            "cv_current":       tf.reshape(current_tensor, [-1, 1]),

            "cycle":            cycle,
            "constant_current": constant_current,
            "end_current_prev": end_current_prev,
            "end_voltage_prev": end_voltage_prev,
            "features":         features,
            "end_voltage":      end_voltage,

        }
        cc_capacity = self.cc_capacity_part2(params)
        pred_cc_capacity = tf.reshape(cc_capacity, [-1, voltage_count])

        cv_capacity = self.cv_capacity(params)
        pred_cv_capacity = tf.reshape(cv_capacity, [-1, current_count])

        if training:

            # NOTE(sam): this is an example of a forall. (for all voltages,
            # and all cell features)
            n_sample = 64
            sampled_voltages = tf.random.uniform(minval = 2.5, maxval = 5.,
                                                 shape = [n_sample, 1])
            sampled_cycles = tf.random.uniform(minval = -10., maxval = 10.,
                                               shape = [n_sample, 1])
            sampled_constant_current = tf.random.uniform(minval = 0.001,
                                                         maxval = 10.,
                                                         shape = [n_sample, 1])
            sampled_features = self.dictionary.sample(n_sample)
            sampled_shift = tf.random.uniform(minval = -1., maxval = 1.,
                                              shape = [n_sample, 1])

            soc, soc_der = self.create_derivatives(
                self.soc_for_derivative,
                params = {
                    'voltage':  sampled_voltages,
                    'features': sampled_features, 'shift': sampled_shift
                },
                voltage_der = 3,
                features_der = 2,
                shift_der = 3,
            )

            soc_loss = .001 * incentive_combine([
                (1., incentive_magnitude(
                    soc,
                    Target.Small,
                    Level.Proportional
                )
                 ),

                (1000., incentive_inequality(
                    soc, Inequality.GreaterThan, 0,
                    Level.Strong
                )
                 ),
                (1., incentive_inequality(
                    soc_der['d_voltage'], Inequality.GreaterThan, 0,
                    Level.Strong
                )
                 ),
                (.1, incentive_magnitude(
                    soc_der['d3_voltage'],
                    Target.Small,
                    Level.Proportional
                )
                 ),
                (.01, incentive_magnitude(
                    soc_der['d_features'],
                    Target.Small,
                    Level.Proportional
                )
                 ),
                (.01, incentive_magnitude(
                    soc_der['d2_features'],
                    Target.Small,
                    Level.Strong
                )
                 ),

                (100., incentive_magnitude(
                    soc_der['d_shift'],
                    Target.Small,
                    Level.Proportional
                )
                 ),

                (100., incentive_magnitude(
                    soc_der['d2_shift'],
                    Target.Small,
                    Level.Proportional
                )
                 ),
                (100., incentive_magnitude(
                    soc_der['d3_shift'],
                    Target.Small,
                    Level.Proportional
                )
                 ),

            ]

            )

            theoretical_cap, theoretical_cap_der = self.create_derivatives(
                self.theoretical_cap_for_derivative,
                params = {
                    'cycle':    sampled_cycles,
                    'current':  sampled_constant_current,
                    'features': sampled_features
                },
                cycle_der = 3,
                current_der = 0,
                features_der = 2,
            )

            theo_cap_loss = .001 * incentive_combine(
                [
                    (100., incentive_inequality(
                        theoretical_cap, Inequality.GreaterThan, 0.01,
                        Level.Strong
                    )
                     ),
                    (100., incentive_inequality(
                        theoretical_cap, Inequality.LessThan, 1,
                        Level.Strong
                    )
                     ),
                    (1., incentive_inequality(
                        theoretical_cap_der['d_cycle'], Inequality.LessThan, 0,
                        Level.Proportional
                    )  # we want cap to diminish with cycle number
                     ),
                    (.1, incentive_inequality(
                        theoretical_cap_der['d2_cycle'], Inequality.LessThan, 0,
                        Level.Proportional
                    )  # we want cap to diminish with cycle number
                     ),

                    # (1.,incentive_inequality(
                    #         theoretical_cap_der['d_constant_current'],
                    #         Inequality.LessThan, 0,
                    #         Level.Proportional
                    #     ) # we want cap to diminish with constant_current (
                    #     if constant current is positive)
                    # ),
                    (100., incentive_magnitude(
                        theoretical_cap_der['d_cycle'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),

                    (100., incentive_magnitude(
                        theoretical_cap_der['d2_cycle'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),
                    (100., incentive_magnitude(
                        theoretical_cap_der['d3_cycle'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),

                    # (100.,incentive_magnitude(
                    #         theoretical_cap_der['d_constant_current'],
                    #         Target.Small,
                    #         Level.Proportional
                    #     )
                    # ),
                    (1., incentive_magnitude(
                        theoretical_cap_der['d_features'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),
                    (1., incentive_magnitude(
                        theoretical_cap_der['d2_features'],
                        Target.Small,
                        Level.Strong
                    )
                     )
                ]
            )

            shift, shift_der = self.create_derivatives(
                self.shift_for_derivative,
                params = {
                    'cycle':    sampled_cycles,
                    'current':  sampled_constant_current,
                    'features': sampled_features
                },
                cycle_der = 3,
                current_der = 0,
                features_der = 2,
            )

            shift_loss = .001 * incentive_combine(
                [
                    (100., incentive_inequality(
                        shift, Inequality.GreaterThan, -1,
                        Level.Strong
                    )
                     ),
                    (100., incentive_inequality(
                        shift, Inequality.LessThan, 1,
                        Level.Strong
                    )
                     ),
                    # (1.,incentive_inequality(
                    #         shift_der['d_constant_current'],
                    #         Inequality.LessThan, 0,
                    #         Level.Proportional
                    #     ) # we want cap to diminish with constant_current (
                    #     if constant current is positive)
                    # ),
                    (100., incentive_magnitude(
                        shift_der['d_cycle'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),

                    (100., incentive_magnitude(
                        shift_der['d2_cycle'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),
                    (100., incentive_magnitude(
                        shift_der['d3_cycle'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),

                    # (100.,incentive_magnitude(
                    #         shift_der['d_constant_current'],
                    #         Target.Small,
                    #         Level.Proportional
                    #     )
                    # ),
                    (1., incentive_magnitude(
                        shift_der['d_features'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),
                    (1., incentive_magnitude(
                        shift_der['d2_features'],
                        Target.Small,
                        Level.Strong
                    )
                     )
                ]
            )

            r, r_der = self.create_derivatives(
                self.r_for_derivative,
                params = {
                    'cycle':    sampled_cycles,
                    'features': sampled_features
                },
                cycle_der = 3,
                features_der = 2,

            )

            r_loss = .001 * incentive_combine(
                [
                    (100., incentive_inequality(
                        r, Inequality.GreaterThan, 0.01,
                        Level.Strong
                    )
                     ),
                    (10., incentive_magnitude(
                        r_der['d2_cycle'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),

                    (100., incentive_magnitude(
                        r_der['d3_cycle'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),

                    (1., incentive_magnitude(
                        r_der['d_features'],
                        Target.Small,
                        Level.Proportional
                    )
                     ),
                    (1., incentive_magnitude(
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
                "soc_loss":         soc_loss,
                "theo_cap_loss":    theo_cap_loss,
                "r_loss":           r_loss,
                "shift_loss":       shift_loss,
                "kl_loss":          kl_loss,
            }

        else:

            pred_r = self.r_direct(
                norm_cycle = self.norm_cycle(params),
                cell_features = self.cell_features_direct(
                    features = params['features']),
            )

            pred_theo_cap = self.theoretical_cap_direct(
                norm_cycle = self.norm_cycle(params),
                current = params['constant_current'],
                cell_features = self.cell_features_direct(
                    features = params['features'])
            )

            return {
                "pred_cc_capacity":   pred_cc_capacity,
                "pred_cv_capacity":   pred_cv_capacity,
                "pred_r":             pred_r,
                "pred_theo_capacity": pred_theo_cap,
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
            "kernel", shape = [self.num_keys, self.num_features * 2])

    def call(self, input, training = True):
        eps = tf.random.normal(
            shape = [self.num_keys, self.num_features])
        mean = self.kernel[:, :self.num_features]
        log_sig = self.kernel[:, self.num_features:]

        if not training:
            features = mean
        else:
            features = mean + tf.exp(log_sig / 2.) * eps

        # tf.gather: "fetching in the dictionary"
        fetched_features = tf.gather(features, input, axis = 0)
        fetched_mean = tf.gather(mean, input, axis = 0)
        fetched_log_sig = tf.gather(log_sig, input, axis = 0)

        return fetched_features, fetched_mean, fetched_log_sig

    def sample(self, n_sample):
        eps = tf.random.normal(
            shape = [n_sample, self.num_features])
        mean = self.kernel[:, :self.num_features]
        log_sig = self.kernel[:, self.num_features:]
        indecies = tf.random.uniform(maxval = self.num_keys, shape = [n_sample],
                                     dtype = tf.int32)

        fetched_mean = tf.gather(mean, indecies, axis = 0)
        fetched_log_sig = tf.gather(log_sig, indecies, axis = 0)
        fetched_features = fetched_mean + tf.exp(fetched_log_sig / 2.) * eps
        return tf.stop_gradient(fetched_features)
