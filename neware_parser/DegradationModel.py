from enum import Enum

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

#TODO(sam): for now, remove incentives/derivatives wrt cycle.
# implement R, shift, Q_scale in terms of Strain.
# should treat strain like a vector of cycles maybe.
# One of the problems with incentives vs regular loss is if high dimentions,
# measure of data is zero, so either the incentives are overwhelming everywhere,
# or they are ignored on the data subspace.
# it is important to sample around the data subspace relatively densely.


#TODO(sam): making the StressToStrain into a layer has advantages,
# but how to set the training flag easily?
# right now, everything takes training flag and passes it to all the children

#TODO(sam): how to constrain the cycle dependence of R, shift, Q_scale
# without having to always go through StressToStrain?
# one way is to express R = R_0(cell_features) * R(strain),
# Q_shift = Q_shift0(cell_features) + Q_shift(strain),
# Q_scale = Q_scale0(cell_features) + Q_scale(strain)
# More generally, we don't know how the final value depends on the initial value.
# What we can ask for, however is that Q_scale = Q_scale(Q_scale0, strain), and Q_scale(Q_scale0, 0) = Q_scale0

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

    :param symbol: the relationship we want (either Inequality.LessThan or
    Inequality.GreaterThan or Inequality.Equal)

        Inequality.LessThan (i.e. A < B) means that A should be less than B,

        Inequality.GreaterThan (i.e. A > B) means that A should be greater
        than B.

        Inequality.Equals (i.e. A = B) means that A should be equal to B.

    :param B: The second object

    :param level:  determines the relationship between the incentive strength
    and the values of A and B.
    (either Level.Strong or Level.Proportional)

        Level.Strong means that we take the L1 norm, so the gradient trying
        to satisfy 'A symbol B' will be constant no matter how far from 'A
        symbol B' we
        are.

        Level.Proportional means that we take the L2 norm, so the gradient
        trying to satisfy 'A symbol B' will be proportional to how far from
        'A symbol B' we are.

    :return: A loss which will give the model an incentive to satisfy 'A
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
            "not yet implemented inequality symbol {}.".format(symbol)
        )

    if level == Level.Strong:
        return intermediate
    elif level == Level.Proportional:
        return tf.square(intermediate)
    else:
        raise Exception("not yet implemented incentive level {}.".format(level))


def incentive_magnitude(A, target, level):
    """
    :param A: The object

    :param target: The direction we want (either Target.Small or Target.Big)

        Target.Small means that the norm of A should be as small as possible

        Target.Big means that the norm of A should be as big as
        possible,

    :param level: Determines the relationship between the incentive strength
    and the value of A. (either Level.Strong or Level.Proportional)

        Level.Strong means that we take the L1 norm, so the gradient trying
        to push the absolute value of A to target
        will be constant.

        Level.Proportional means that we take the L2 norm,
        so the gradient trying to push the absolute value of A to target
        will be proportional to the absolute value of A.

    :return: A loss which will give the model an incentive to push the
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

    :return: A combined loss (single number) which will incentivize all the
    individual incentive tensors with weights given by the coefficients.
    """

    return sum([a[0] * tf.reduce_mean(a[1]) for a in As])


class DegradationModel(Model):

    def __init__(self, num_keys, depth, width, n_channels=16):
        super(DegradationModel, self).__init__()

        self.nn_r = feedforward_nn_parameters(depth, width)
        self.nn_theoretical_cap = feedforward_nn_parameters(depth, width)
        self.nn_soc = feedforward_nn_parameters(depth, width)
        self.nn_shift = feedforward_nn_parameters(depth, width)


        self.nn_r_strainless = feedforward_nn_parameters(depth, width)
        self.nn_theoretical_cap_strainless = feedforward_nn_parameters(depth, width)
        self.nn_shift_strainless = feedforward_nn_parameters(depth, width)


        self.dictionary = DictionaryLayer(
            num_features = width,
            num_keys = num_keys
        )

        self.stress_to_strain_layer = StressToStrainLayer(
            num_features = width,
            n_channels=n_channels
        )

        self.width = width
        self.num_keys = num_keys
        self.n_channels = n_channels

    # Begin: nn application functions ==========================================

    def eq_voltage_direct(self, voltage, current, resistance, training=True):
        return voltage - current * resistance

    def soc_direct(self, voltage, shift, cell_features, training=True):
        dependencies = (
            voltage,
            shift,
            cell_features
        )
        return tf.nn.elu(self.nn_call(self.nn_soc, dependencies))

    def theoretical_cap_strainless_direct(self,cell_features, training=True):
        dependencies = (
            cell_features
        )
        return tf.nn.elu(self.nn_call(self.nn_theoretical_cap_strainless, dependencies))

    def shift_strainless_direct(self, current, cell_features, training=True):
        dependencies = (
            tf.abs(current),
            cell_features
        )
        return self.nn_call(self.nn_shift_strainless, dependencies)

    def r_strainless_direct(self, cell_features, training=True):
        dependencies = (
            cell_features,
        )

        return tf.nn.elu(self.nn_call(self.nn_r_strainless, dependencies))


    def theoretical_cap_direct(self, strain, current, theoretical_cap_strainless, training=True):
        dependencies = (
            strain,
            # tf.abs(current),
            theoretical_cap_strainless,
        )
        return tf.nn.elu(self.nn_call(self.nn_theoretical_cap, dependencies))

    def shift_direct(self, strain, current, shift_strainless, training=True):
        dependencies = (
            strain,
            tf.abs(current),
            shift_strainless
        )
        return self.nn_call(self.nn_shift, dependencies)

    def r_direct(self, strain, r_strainless, training=True):
        dependencies = (
            strain,
            r_strainless,
        )
        return tf.nn.elu(self.nn_call(self.nn_r, dependencies))




    def norm_constant_direct(self, features, training=True):
        return features[:, 0:1]

    def cell_features_direct(self, features, training=True):
        return features[:, 1:]

    def norm_cycle_direct(self, cycle, norm_constant, training=True):
        return cycle * (1e-10 + tf.exp(-norm_constant))

    def norm_cycle(self, params, training=True):
        return self.norm_cycle_direct(
            norm_constant = self.norm_constant_direct(params['features'], training=training),
            cycle = params['cycle']
        )

    def stress_to_strain_direct(self, norm_cycle, cell_features, svit_grid, count_matrix, training=True):
        return self.stress_to_strain_layer(
            (
                norm_cycle,
                cell_features,
                svit_grid,
                count_matrix,
            ),
            training = training
        )

    def soc_for_derivative(self, params, training=True):
        return self.soc_direct(
            cell_features = self.cell_features_direct(
                features = params['features'],
                training=training
            ),
            voltage = params['voltage'],
            shift = params['shift']
        )

    '''
    def theoretical_cap_for_derivative(self, params, training=True):
        return self.theoretical_cap_direct(
            norm_cycle = self.norm_cycle(
                params = {
                    'cycle':    params['cycle'],
                    'features': params['features']
                },
                training=training
            ),
            current = params['current'],
            cell_features = self.cell_features_direct(
                features = params['features'],
                training=training
            ),
            training=training
        )

    def shift_for_derivative(self, params, training=True):
        return self.shift_direct(
            norm_cycle = self.norm_cycle(
                params = {
                    'cycle':    params['cycle'],
                    'features': params['features']
                },
                training=training
            ),
            current = params['current'],
            cell_features = self.cell_features_direct(
                features = params['features'],
                training=training
            ),
            training=training
        )

    def r_for_derivative(self, params, training=True):
        return self.r_direct(
            norm_cycle = self.norm_cycle(
                params = {
                    'cycle':    params['cycle'],
                    'features': params['features']
                },
                training=training
            ),
            cell_features = self.cell_features_direct(
                features = params['features'],
                training=training
            ),
            training=training
        )

    '''

    def cc_capacity_part2(self, params, training=True):
        norm_constant = self.norm_constant_direct(features = params['features'], training=training)
        norm_cycle = self.norm_cycle_direct(
            cycle = params['cycle'],
            norm_constant = norm_constant,
            training=training
        )

        cell_features = self.cell_features_direct(features = params['features'], training=training)
        strain = self.stress_to_strain_direct(
            norm_cycle = norm_cycle,
            cell_features= cell_features,
            svit_grid=params['svit_grid'],
            count_matrix=params['count_matrix'],
            training=training
        )


        theoretical_cap_strainless = self.theoretical_cap_strainless_direct(
            cell_features = cell_features,
            training=training
        )


        shift_0_strainless = self.shift_strainless_direct(
            current = params['end_current_prev'],
            cell_features = cell_features,
            training=training
        )

        resistance_strainless = self.r_strainless_direct(
            cell_features = cell_features,
            training=training
        )


        theoretical_cap = self.theoretical_cap_direct(
            strain=strain,
            current = params['constant_current'],
            theoretical_cap_strainless = theoretical_cap_strainless,
            training=training
        )


        shift_0 = self.shift_direct(
            strain=strain,
            current = params['end_current_prev'],
            shift_strainless = shift_0_strainless,
            training=training
        )

        resistance = self.r_direct(
            strain=strain,
            r_strainless= resistance_strainless,
            training=training
        )

        eq_voltage_0 = self.eq_voltage_direct(
            voltage = params['end_voltage_prev'],
            current = params['end_current_prev'],
            resistance = resistance,
            training=training
        )

        soc_0 = self.soc_direct(
            voltage = eq_voltage_0,
            shift = shift_0,
            cell_features = cell_features,
            training=training
        )

        eq_voltage_1 = self.eq_voltage_direct(
            voltage = params['voltage'],
            current = self.add_volt_dep(params['constant_current'], params),
            resistance = self.add_volt_dep(resistance, params),
            training=training
        )

        shift_1_strainless = self.shift_strainless_direct(
            current=params['constant_current'],
            cell_features=cell_features,
            training=training
        )
        shift_1 = self.shift_direct(
            strain=strain,
            current=params['constant_current'],
            shift_strainless=shift_1_strainless,
            training=training
        )


        soc_1 = self.soc_direct(
            voltage = eq_voltage_1,
            shift = self.add_volt_dep(shift_1, params),
            cell_features = self.add_volt_dep(
                cell_features, params,
                cell_features.shape[1]
            ),
            training=training
        )

        return self.add_volt_dep(theoretical_cap, params) * (
            soc_1 - self.add_volt_dep(soc_0, params))

    def cv_capacity(self, params, training=True):
        norm_constant = self.norm_constant_direct(features = params['features'], training=training)
        norm_cycle = self.norm_cycle_direct(
            cycle = params['cycle'],
            norm_constant = norm_constant,
            training=training
        )

        cell_features = self.cell_features_direct(features = params['features'], training=training)

        strain = self.stress_to_strain_direct(
            norm_cycle = norm_cycle,
            cell_features= cell_features,
            svit_grid=params['svit_grid'],
            count_matrix=params['count_matrix'],
            training=training
        )


        cc_shift_strainless = self.shift_strainless_direct(
            current=params['end_current_prev'],
            cell_features=cell_features,
            training=training
        )
        cc_shift = self.shift_direct(
            strain=strain,
            current=params['end_current_prev'],
            shift_strainless=cc_shift_strainless,
            training=training
        )

        resistance_strainless = self.r_strainless_direct(
            cell_features=cell_features,
            training=training
        )

        resistance = self.r_direct(
            strain=strain,
            r_strainless=resistance_strainless,
            training=training
        )


        eq_voltage_0 = self.eq_voltage_direct(
            voltage = params['end_voltage_prev'],
            current = params['end_current_prev'],
            resistance = resistance,
            training=training
        )

        soc_0 = self.soc_direct(
            voltage = eq_voltage_0,
            shift = cc_shift,
            cell_features = cell_features,
            training=training
        )

        #NOTE(sam): if there truly is no dependency on current for theoretical_cap,
        # then we can restructure the code below.
        theoretical_cap_strainless = self.theoretical_cap_strainless_direct(
            cell_features = cell_features,
            training=training
        )

        theoretical_cap = self.theoretical_cap_direct(
            strain= self.add_current_dep(
                strain,
                params,
                strain.shape[1]
            ),
            current = params['cv_current'],
            theoretical_cap_strainless = self.add_current_dep(
                theoretical_cap_strainless,
                params
            ),
            training=training
        )


        eq_voltage_1 = self.eq_voltage_direct(
            voltage = self.add_current_dep(params['end_voltage'], params),
            current = params['cv_current'],
            resistance = self.add_current_dep(resistance, params),
            training=training
        )

        cv_shift_strainless = self.shift_strainless_direct(
            current=params['cv_current'],
            cell_features=self.add_current_dep(
                cell_features,
                params,
                cell_features.shape[1]
            ),
            training=training
        )

        cv_shift = self.shift_direct(
            self.add_current_dep(
                strain,
                params,
                strain.shape[1]
            ),
            current=params['cv_current'],
            shift_strainless=cv_shift_strainless,
            training=training
        )


        soc_1 = self.soc_direct(
            voltage = eq_voltage_1,
            shift = cv_shift,
            cell_features = self.add_current_dep(
                cell_features,
                params,
                cell_features.shape[1]
            ),
            training=training
        )

        return theoretical_cap * (soc_1 - self.add_current_dep(soc_0, params))


    def nn_call(self, nn_func, dependencies):
        centers = nn_func['initial'](
            tf.concat(dependencies, axis = 1)
        )
        for d in nn_func['bulk']:
            centers = d(centers)
        return nn_func['final'](centers)

    def create_derivatives(
        self,
        nn,
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
        :param cycle_der:
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
        svit_grid = x[8]
        count_matrix = x[9]


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

            "svit_grid":        svit_grid,
            "count_matrix":     count_matrix,
        }
        cc_capacity = self.cc_capacity_part2(params, training=training)
        pred_cc_capacity = tf.reshape(cc_capacity, [-1, voltage_count])

        cv_capacity = self.cv_capacity(params, training=training)
        pred_cv_capacity = tf.reshape(cv_capacity, [-1, current_count])

        if training:

            # NOTE(sam): this is an example of a forall. (for all voltages,
            # and all cell features)
            n_sample = 64
            sampled_voltages = tf.random.uniform(
                minval = 2.5,
                maxval = 5.,
                shape = [n_sample, 1]
            )
            sampled_cycles = tf.random.uniform(
                minval = -10.,
                maxval = 10.,
                shape = [n_sample, 1]
            )
            sampled_constant_current = tf.random.uniform(
                minval = 0.001,
                maxval = 10.,
                shape = [n_sample, 1]
            )
            sampled_features = self.dictionary.sample(n_sample)
            sampled_shift = tf.random.uniform(
                minval = -1.,
                maxval = 1.,
                shape = [n_sample, 1]
            )

            '''
            soc, soc_der = self.create_derivatives(
                self.soc_for_derivative,
                params = {
                    'voltage':  sampled_voltages,
                    'features': sampled_features,
                    'shift':    sampled_shift
                },
                voltage_der = 3,
                features_der = 2,
                shift_der = 3,
            )

            soc_loss = .0001 * incentive_combine(
                [
                    (
                        1.,
                        incentive_magnitude(
                            soc,
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        100.,
                        incentive_inequality(
                            soc, Inequality.GreaterThan, 0,
                            Level.Strong
                        )
                    ),
                    (
                        10000.,
                        incentive_inequality(
                            soc_der['d_voltage'], Inequality.GreaterThan, 0.05,
                            Level.Strong
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            soc_der['d3_voltage'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        .01,
                        incentive_magnitude(
                            soc_der['d_features'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        .01,
                        incentive_magnitude(
                            soc_der['d2_features'],
                            Target.Small,
                            Level.Strong
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            soc_der['d_shift'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),

                    (
                        100.,
                        incentive_magnitude(
                            soc_der['d2_shift'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
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

            theo_cap_loss = .0001 * incentive_combine(
                [
                    (
                        100.,
                        incentive_inequality(
                            theoretical_cap, Inequality.GreaterThan, 0.01,
                            Level.Strong
                        )
                    ),
                    (
                        100.,
                        incentive_inequality(
                            theoretical_cap, Inequality.LessThan, 1,
                            Level.Strong
                        )
                    ),
                    (
                        1.,
                        incentive_inequality(
                            theoretical_cap_der['d_cycle'], Inequality.LessThan,
                            0,
                            Level.Proportional
                        )  # we want cap to diminish with cycle number
                    ),
                    (
                        .1,
                        incentive_inequality(
                            theoretical_cap_der['d2_cycle'],
                            Inequality.LessThan,
                            0,
                            Level.Proportional
                        )  # we want cap to diminish with cycle number
                    ),

                    (
                        100.,
                        incentive_magnitude(
                            theoretical_cap_der['d_cycle'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),

                    (
                        100.,
                        incentive_magnitude(
                            theoretical_cap_der['d2_cycle'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            theoretical_cap_der['d3_cycle'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),

                    (
                        1.,
                        incentive_magnitude(
                            theoretical_cap_der['d_features'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        1.,
                        incentive_magnitude(
                            theoretical_cap_der['d2_features'],
                            Target.Small,
                            Level.Strong
                        )
                    ),
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
                current_der = 3,
                features_der = 2,
            )

            shift_loss = .0001 * incentive_combine(
                [
                    (
                        100.,
                        incentive_inequality(
                            shift, Inequality.GreaterThan, -1,
                            Level.Strong
                        )
                    ),
                    (
                        100.,
                        incentive_inequality(
                            shift, Inequality.LessThan, 1,
                            Level.Strong
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            shift_der['d_current'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            shift_der['d2_current'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            shift_der['d3_current'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        1.,
                        incentive_magnitude(
                            shift_der['d_features'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        1.,
                        incentive_magnitude(
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

            r_loss = .0001 * incentive_combine(
                [
                    (
                        100.,
                        incentive_inequality(
                            r,
                            Inequality.GreaterThan,
                            0.01,
                            Level.Strong
                        )
                    ),

                    (
                        100.,
                        incentive_inequality(
                            r,
                            Inequality.GreaterThan,
                            0.01,
                            Level.Strong
                        )
                    ),
                    (
                        10.,
                        incentive_magnitude(
                            r_der['d2_cycle'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),

                    (
                        100.,
                        incentive_magnitude(
                            r_der['d3_cycle'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),

                    (
                        1.,
                        incentive_magnitude(
                            r_der['d_features'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        1.,
                        incentive_magnitude(
                            r_der['d2_features'],
                            Target.Small,
                            Level.Strong
                        )
                    )
                ]
            )
            '''
            kl_loss = tf.reduce_mean(
                0.5 * (tf.exp(log_sig) + tf.square(mean) - 1. - log_sig)
            )

            return {
                "pred_cc_capacity": pred_cc_capacity,
                "pred_cv_capacity": pred_cv_capacity,
                "soc_loss":         0.,#soc_loss,
                "theo_cap_loss":    0.,#theo_cap_loss,
                "r_loss":           0.,#r_loss,
                "shift_loss":       0.,#shift_loss,
                "kl_loss":          kl_loss,
            }

        else:

            norm_constant = self.norm_constant_direct(features=params['features'], training=training)

            norm_cycle = self.norm_cycle_direct(
                cycle=params['cycle'],
                norm_constant=norm_constant,
                training=training
            )

            cell_features = self.cell_features_direct(features=params['features'], training=training)

            strain = self.stress_to_strain_direct(
                norm_cycle=norm_cycle,
                cell_features=cell_features,
                svit_grid=params['svit_grid'],
                count_matrix=params['count_matrix'],
                training=training
            )

            theoretical_cap_strainless = self.theoretical_cap_strainless_direct(
                cell_features=cell_features,
                training=training
            )

            shift_0_strainless = self.shift_strainless_direct(
                current=params['constant_current'],
                cell_features=cell_features,
                training=training
            )

            resistance_strainless = self.r_strainless_direct(
                cell_features=cell_features,
                training=training
            )

            theoretical_cap = self.theoretical_cap_direct(
                strain=strain,
                current=params['constant_current'],
                theoretical_cap_strainless=theoretical_cap_strainless,
                training=training
            )

            shift = self.shift_direct(
                strain=strain,
                current=params['constant_current'],
                shift_strainless=shift_0_strainless,
                training=training
            )

            resistance = self.r_direct(
                strain=strain,
                r_strainless=resistance_strainless,
                training=training
            )


            return {

                "pred_cc_capacity":   pred_cc_capacity,
                "pred_cv_capacity":   pred_cv_capacity,
                "pred_r":             resistance,
                "pred_theo_capacity": theoretical_cap,
                "pred_shift":         shift,
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
            "kernel", shape = [self.num_keys, self.num_features * 2]
        )

    def call(self, input, training = True):
        eps = tf.random.normal(
            shape = [self.num_keys, self.num_features]
        )
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



class StressToStrainLayer(Layer):
    def __init__(self, num_features, n_channels):
        super(StressToStrainLayer, self).__init__()
        self.num_features = num_features
        self.n_channels = n_channels
        self.input_kernel = self.add_weight(
            "input_kernel", shape=[1, 1, 1, 4 + 1 + self.num_features - 1, self.n_channels]
        )

        self.v_i_kernel_1 = self.add_weight(
            "v_i_kernel_1", shape=[3, 3, 1, self.n_channels, self.n_channels]
        )

        self.v_i_kernel_2 = self.add_weight(
            "v_i_kernel_2", shape=[3, 3, 1, self.n_channels, self.n_channels]
        )

        self.t_kernel = self.add_weight(
            "t_kernel", shape=[1, 1, 3, self.n_channels, self.n_channels]
        )

        self.output_kernel = self.add_weight(
            "output_kernel", shape=[1, 1, 1, self.n_channels, self.n_channels]
        )


    def call(self, input, training = True):
        norm_cycle = input[0]  # matrix; dim: [batch, 1]
        cell_features = input[1]  # matrix; dim: [batch, n_features]
        svit_grid = input[2] # tensor; dim: [batch, n_sign, n_voltage, n_current, n_temperature, 4]
        count_matrix = input[3] # tensor; dim: [batch, n_sign, n_voltage, n_current, n_temperature, 1]

        n_batch = count_matrix.shape[0]
        n_voltage = count_matrix.shape[2]
        n_current = count_matrix.shape[3]
        n_temperature = count_matrix.shape[4]

        total_count_matrix_0 = tf.reshape(norm_cycle, [n_batch, 1, 1, 1, 1]) * count_matrix[:,0,:,:,:,:]
        total_count_matrix_1 = tf.reshape(norm_cycle, [n_batch, 1, 1, 1, 1]) * count_matrix[:,1,:,:,:,:]

        svit_grid_0 = svit_grid[:,0,:,:,:,:]
        svit_grid_1 = svit_grid[:,1,:,:,:,:]

        cell_features_grid = tf.tile(
            tf.reshape(cell_features, [n_batch, 1, 1, 1, -1]),
            [1, n_voltage, n_current, n_temperature, 1]
        )



        val_0 = tf.concat(
            (
                svit_grid_0,
                total_count_matrix_0,
                cell_features_grid
            ),
            axis = -1
        )
        val_1 = tf.concat(
            (
                svit_grid_1,
                total_count_matrix_1,
                cell_features_grid
            ),
            axis=-1
        )

        filters=[
            (self.input_kernel, 'none'),
            (self.v_i_kernel_1, 'relu'),
            (self.t_kernel, 'relu'),
            (self.v_i_kernel_2,'relu'),
            (self.output_kernel,'none')
        ]

        for fil,activ in filters:
            val_0 = tf.nn.convolution(input=val_0, filters=fil, padding='SAME')
            val_1 = tf.nn.convolution(input=val_1, filters=fil, padding='SAME')

            if activ is 'relu':
                val_0 = tf.nn.relu(val_0)
                val_1 = tf.nn.relu(val_1)

        # each entry is scaled by its count.
        val_0 = val_0 * total_count_matrix_0
        val_1 = val_1 * total_count_matrix_1

        # then we take the average over all the grid.
        val_0 = tf.reduce_mean(val_0, axis=[1, 2, 3], keepdims=False)
        val_1 = tf.reduce_mean(val_1, axis=[1, 2, 3], keepdims=False)

        return val_0 + val_1