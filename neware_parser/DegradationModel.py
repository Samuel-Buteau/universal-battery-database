from enum import Enum

import numpy
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer


# TODO(sam): for now, remove incentives/derivatives wrt cycle.
# implement R, shift, q_scale in terms of Strain.
# should treat strain like a vector of cycles maybe.
# One of the problems with incentives vs regular loss is if high dimentions,
# measure of data is zero, so either the incentives are overwhelming everywhere,
# or they are ignored on the data subspace.
# it is important to sample around the data subspace relatively densely.


# TODO(sam): making the StressToStrain into a layer has advantages,
# but how to set the training flag easily?
# right now, everything takes training flag and passes it to all the children

# TODO(sam): how to constrain the cycle dependence of R, shift, q_scale
# without having to always go through StressToStrain?
# one way is to express R = R_0(cell_features) * R(strain),
# q_shift = q_shift0(cell_features) + q_shift(strain),
# q_scale = q_scale0(cell_features) + q_scale(strain)
# More generally, we don't know how the final value depends on the initial
# value.
# What we can ask for, however is that q_scale = q_scale(q_scale0, strain),
# and q_scale(q_scale0, 0) = q_scale0

def feedforward_nn_parameters(depth, width, last = None):
    if last is None:
        last = 1

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
        last,
        activation = None,
        use_bias = True,
        bias_initializer = 'zeros',
        kernel_initializer = 'zeros'
    )
    return {'initial': initial, 'bulk': bulk, 'final': final}


def nn_call(nn_func, dependencies, training = True):
    centers = nn_func['initial'](
        tf.concat(dependencies, axis = 1),
        training = training,
    )
    for d in nn_func['bulk']:
        centers = d(centers, training = training)
    return nn_func['final'](centers, training = training)


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


def print_cell_info(
    cell_latent_flags, cell_to_pos, cell_to_neg, cell_to_electrolyte,

    electrolyte_to_solvent, electrolyte_to_salt, electrolyte_to_additive,
    electrolyte_latent_flags, names
):
    pos_to_pos_name, neg_to_neg_name = names[0], names[1]
    electrolyte_to_electrolyte_name = names[2]
    molecule_to_molecule_name = names[3]
    print('\ncell_id: Known Components (Y/N):\n')
    for k in cell_latent_flags.keys():
        known = 'Y'
        if cell_latent_flags[k] > .5:
            known = 'N'
        print('{}\t\t{}'.format(k, known))
        if known == 'Y':
            pos_id = cell_to_pos[k]
            if pos_id in pos_to_pos_name.keys():
                print('\tcathode:\t\t\t{}'.format(pos_to_pos_name[pos_id]))
            else:
                print('\tcathode id:\t\t\t{}'.format(pos_id))
            neg_id = cell_to_neg[k]
            if neg_id in neg_to_neg_name.keys():
                print('\tanode:\t\t\t\t{}'.format(neg_to_neg_name[neg_id]))
            else:
                print('\tanode id:\t\t\t{}'.format(neg_id))
            electrolyte_id = cell_to_electrolyte[k]
            if electrolyte_id in electrolyte_to_electrolyte_name.keys():
                print('\telectrolyte:\t\t\t{}'.format(
                    electrolyte_to_electrolyte_name[electrolyte_id]))
            else:
                print('\telectrolyte id:\t\t\t{}'.format(electrolyte_id))

            electrolyte_known = 'Y'
            if electrolyte_latent_flags[electrolyte_id] > .5:
                electrolyte_known = 'N'
            print('\tKnown Electrolyte Components:\t{}'.format(
                electrolyte_known))
            if electrolyte_known == 'Y':
                for st, electrolyte_to in [
                    ('solvents', electrolyte_to_solvent),
                    ('salts', electrolyte_to_salt),
                    ('additive', electrolyte_to_additive),
                ]:
                    print('\t{}:'.format(st))
                    components = electrolyte_to[electrolyte_id]
                    for s, w in components:
                        if s in molecule_to_molecule_name.keys():
                            print('\t\t{} {}'.format(
                                w, molecule_to_molecule_name[s])
                            )
                        else:
                            print('\t\t{} id {}'.format(w, s))
        print()


def add_volt_dep(thing, params, dim = 1):
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


def add_current_dep(thing, params, dim = 1):
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


def get_norm_constant(features):
    return features[:, 0:1]


def get_cell_features(features):
    return features[:, 1:]


def get_norm_cycle_direct(cycle, norm_constant):
    return cycle * (1e-10 + tf.exp(-norm_constant))


def calculate_eq_voltage(voltage, current, resistance):
    return voltage - current * resistance


def get_norm_cycle(params):
    return get_norm_cycle_direct(
        norm_constant = get_norm_constant(params['features']),
        cycle = params['cycle']
    )


class DegradationModel(Model):

    def __init__(
        self, depth, width,
        cell_dict, pos_dict, neg_dict, electrolyte_dict, molecule_dict,
        cell_latent_flags, cell_to_pos, cell_to_neg, cell_to_electrolyte,

        electrolyte_to_solvent, electrolyte_to_salt, electrolyte_to_additive,
        electrolyte_latent_flags, names,

        n_channels = 16
    ):
        super(DegradationModel, self).__init__()

        print_cell_info(
            cell_latent_flags, cell_to_pos, cell_to_neg, cell_to_electrolyte,
            electrolyte_to_solvent, electrolyte_to_salt,
            electrolyte_to_additive, electrolyte_latent_flags, names
        )

        self.num_features = width

        """ feedforward neural networks """

        self.nn_r = feedforward_nn_parameters(depth, width)
        self.nn_q_scale = feedforward_nn_parameters(depth, width)
        self.nn_q = feedforward_nn_parameters(depth, width)
        self.nn_shift = feedforward_nn_parameters(depth, width)
        self.nn_v_plus = feedforward_nn_parameters(depth, width)
        self.nn_v_minus = feedforward_nn_parameters(depth, width)

        self.nn_r_strainless = feedforward_nn_parameters(depth, width)
        self.nn_q_scale_strainless = feedforward_nn_parameters(depth, width)
        self.nn_shift_strainless = feedforward_nn_parameters(depth, width)

        self.nn_pos_projection = feedforward_nn_parameters(
            depth, width, last = self.num_features
        )
        self.nn_neg_projection = feedforward_nn_parameters(
            depth, width, last = self.num_features
        )

        """ Primitive Dictionary Layer variables """

        self.cell_direct = PrimitiveDictionaryLayer(
            num_features = self.num_features, id_dict = cell_dict
        )
        self.pos_direct = PrimitiveDictionaryLayer(
            num_features = self.num_features, id_dict = pos_dict
        )
        self.neg_direct = PrimitiveDictionaryLayer(
            num_features = self.num_features, id_dict = neg_dict
        )
        self.electrolyte_direct = PrimitiveDictionaryLayer(
            num_features = self.num_features, id_dict = electrolyte_dict
        )
        self.molecule_direct = PrimitiveDictionaryLayer(
            num_features = self.num_features, id_dict = molecule_dict
        )

        self.num_keys = self.cell_direct.num_keys

        # cell_latent_flags is a dict with barcodes as keys.
        # latent_flags is a numpy array such that the indecies match cell_dict
        latent_flags = numpy.ones(
            (self.cell_direct.num_keys, 1),
            dtype = numpy.float32
        )

        for cell_id in self.cell_direct.id_dict.keys():
            if cell_id in cell_latent_flags.keys():
                latent_flags[self.cell_direct.id_dict[cell_id], 0] =\
                    cell_latent_flags[cell_id]

        self.cell_latent_flags = tf.constant(latent_flags)

        cell_pointers = numpy.zeros(
            shape = (self.cell_direct.num_keys, 3), dtype = numpy.int32
        )

        for cell_id in self.cell_direct.id_dict.keys():
            if cell_id in cell_to_pos.keys():
                cell_pointers[self.cell_direct.id_dict[cell_id], 0] =\
                    pos_dict[cell_to_pos[cell_id]]
            if cell_id in cell_to_neg.keys():
                cell_pointers[self.cell_direct.id_dict[cell_id], 1] =\
                    neg_dict[cell_to_neg[cell_id]]
            if cell_id in cell_to_electrolyte.keys():
                cell_pointers[self.cell_direct.id_dict[cell_id], 2] =\
                    electrolyte_dict[cell_to_electrolyte[cell_id]]

        self.cell_pointers = tf.constant(cell_pointers)
        self.cell_indirect = feedforward_nn_parameters(
            depth, width, last = self.num_features
        )

        self.n_solvent_max = numpy.max(
            [len(v) for v in electrolyte_to_solvent.values()]
        )
        self.n_salt_max = numpy.max(
            [len(v) for v in electrolyte_to_salt.values()]
        )
        self.n_additive_max = numpy.max(
            [len(v) for v in electrolyte_to_additive.values()]
        )

        # electrolyte latent flags
        latent_flags = numpy.ones(
            (self.electrolyte_direct.num_keys, 1), dtype = numpy.float32
        )

        for electrolyte_id in self.electrolyte_direct.id_dict.keys():
            if electrolyte_id in electrolyte_latent_flags.keys():
                latent_flags[
                    self.electrolyte_direct.id_dict[electrolyte_id], 0
                ] = electrolyte_latent_flags[electrolyte_id]

        self.electrolyte_latent_flags = tf.constant(latent_flags)

        # electrolyte pointers and weights

        pointers = numpy.zeros(
            shape = (
                self.electrolyte_direct.num_keys,
                self.n_solvent_max + self.n_salt_max + self.n_additive_max
            ),
            dtype = numpy.int32,
        )
        weights = numpy.zeros(
            shape = (
                self.electrolyte_direct.num_keys,
                self.n_solvent_max + self.n_salt_max + self.n_additive_max
            ),
            dtype = numpy.float32,
        )

        for electrolyte_id in self.electrolyte_direct.id_dict.keys():
            for reference_index, electrolyte_to in [
                (0, electrolyte_to_solvent),
                (self.n_solvent_max, electrolyte_to_salt),
                (self.n_solvent_max + self.n_salt_max, electrolyte_to_additive)
            ]:
                if electrolyte_id in electrolyte_to.keys():
                    my_components = electrolyte_to[electrolyte_id]
                    for i in range(len(my_components)):
                        molecule_id, weight = my_components[i]
                        pointers[
                            self.electrolyte_direct.id_dict[electrolyte_id],
                            i + reference_index
                        ] = molecule_dict[molecule_id]
                        weights[
                            self.electrolyte_direct.id_dict[electrolyte_id],
                            i + reference_index
                        ] = weight

        self.electrolyte_pointers = tf.constant(pointers)
        self.electrolyte_weights = tf.constant(weights)

        self.electrolyte_indirect = feedforward_nn_parameters(
            depth, width, last = self.num_features
        )

        self.stress_to_encoded_layer = StressToEncodedLayer(
            n_channels = n_channels
        )
        self.nn_strain = feedforward_nn_parameters(depth, width)

        self.width = width
        self.n_channels = n_channels

    def z_cell_from_indices(
        self,
        indices, training = True, sample = False, compute_derivatives = False
    ):

        features_cell_direct, loss_cell = self.cell_direct(
            indices,
            training = training,
            sample = False
        )

        fetched_latent_cell = tf.gather(
            self.cell_latent_flags,
            indices,
            axis = 0
        )
        fetched_pointers_cell = tf.gather(
            self.cell_pointers,
            indices,
            axis = 0
        )

        pas_indices = fetched_pointers_cell[:, 0]
        neg_indices = fetched_pointers_cell[:, 1]
        electrolyte_indices = fetched_pointers_cell[:, 2]

        features_pos, loss_pos = self.pos_direct(
            pas_indices,
            training = training,
            sample = sample
        )

        features_neg, loss_neg = self.neg_direct(
            neg_indices,
            training = training,
            sample = sample
        )

        features_electrolyte_direct, loss_electrolyte_direct\
            = self.electrolyte_direct(
            electrolyte_indices,
            training = training,
            sample = sample
        )

        fetched_latent_electrolyte = tf.gather(
            self.electrolyte_latent_flags,
            electrolyte_indices,
            axis = 0
        )
        fetched_pointers_electrolyte = tf.gather(
            self.electrolyte_pointers,
            electrolyte_indices,
            axis = 0
        )
        fetched_weights_electrolyte = tf.gather(
            self.electrolyte_weights,
            electrolyte_indices,
            axis = 0
        )
        fetched_pointers_electrolyte_reshaped = tf.reshape(
            fetched_pointers_electrolyte,
            [-1]
        )

        features_molecule, loss_molecule = self.molecule_direct(
            fetched_pointers_electrolyte_reshaped,
            training = training,
            sample = sample
        )
        features_molecule_reshaped = tf.reshape(
            features_molecule,
            [
                -1,
                self.n_solvent_max + self.n_salt_max + self.n_additive_max,
                self.molecule_direct.num_features
            ]
        )

        if training:
            loss_molecule_reshaped = tf.reshape(
                loss_molecule,
                [
                    -1,
                    self.n_solvent_max + self.n_salt_max + self.n_additive_max,
                    1
                ]
            )

        fetched_molecule_weights = tf.reshape(
            fetched_weights_electrolyte,
            [-1, self.n_solvent_max + self.n_salt_max + self.n_additive_max, 1]
        ) * features_molecule_reshaped

        total_solvent = 1. / (1e-10 + tf.reduce_sum(
            fetched_weights_electrolyte[:, 0:self.n_solvent_max],
            axis = 1
        ))

        features_solvent = tf.reshape(total_solvent, [-1, 1]) * tf.reduce_sum(
            fetched_molecule_weights[:, 0:self.n_solvent_max, :],
            axis = 1
        )
        features_salt = tf.reduce_sum(
            fetched_molecule_weights[
            :, self.n_solvent_max:self.n_solvent_max + self.n_salt_max, :
            ],
            axis = 1
        )
        features_additive = tf.reduce_sum(
            fetched_molecule_weights[
            :,
            self.n_solvent_max + self.n_salt_max:
            self.n_solvent_max + self.n_salt_max + self.n_additive_max,
            :
            ],
            axis = 1
        )

        if training:
            fetched_molecule_loss_weights = tf.reshape(
                fetched_weights_electrolyte,
                [
                    -1,
                    self.n_solvent_max + self.n_salt_max + self.n_additive_max,
                    1
                ]
            ) * loss_molecule_reshaped
            loss_solvent = tf.reshape(total_solvent, [-1, 1]) * tf.reduce_sum(
                fetched_molecule_loss_weights[:, 0:self.n_solvent_max, :],
                axis = 1
            )
            loss_salt = tf.reduce_sum(
                fetched_molecule_loss_weights[:,
                self.n_solvent_max:self.n_solvent_max + self.n_salt_max, :],
                axis = 1
            )
            loss_additive = tf.reduce_sum(
                fetched_molecule_loss_weights[
                :,
                self.n_solvent_max + self.n_salt_max:
                self.n_solvent_max + self.n_salt_max + self.n_additive_max,
                :
                ],
                axis = 1
            )

        derivatives = {}

        if compute_derivatives:

            with tf.GradientTape(persistent = True) as tape_d1:
                tape_d1.watch(
                    features_solvent
                )
                tape_d1.watch(
                    features_salt
                )
                tape_d1.watch(
                    features_additive
                )

                electrolyte_dependencies = (
                    features_solvent,
                    features_salt,
                    features_additive,
                )

                features_electrolyte_indirect = nn_call(
                    self.electrolyte_indirect,
                    electrolyte_dependencies,
                    training = training
                )

            derivatives['d_features_solvent'] = tape_d1.batch_jacobian(
                source = features_solvent,
                target = features_electrolyte_indirect
            )
            derivatives['d_features_salt'] = tape_d1.batch_jacobian(
                source = features_salt,
                target = features_electrolyte_indirect
            )
            derivatives['d_features_additive'] = tape_d1.batch_jacobian(
                source = features_additive,
                target = features_electrolyte_indirect
            )

            del tape_d1
        else:
            electrolyte_dependencies = (
                features_solvent,
                features_salt,
                features_additive,
            )

            features_electrolyte_indirect = nn_call(
                self.electrolyte_indirect,
                electrolyte_dependencies,
                training = training
            )

        features_electrolyte = (
            (fetched_latent_electrolyte * features_electrolyte_direct) +
            ((1. - fetched_latent_electrolyte) * features_electrolyte_indirect)
        )

        if compute_derivatives:

            with tf.GradientTape(persistent = True) as tape_d1:
                tape_d1.watch(
                    features_pos
                )
                tape_d1.watch(
                    features_neg
                )
                tape_d1.watch(
                    features_electrolyte
                )

                cell_dependencies = (
                    features_pos,
                    features_neg,
                    features_electrolyte,
                )

                features_cell_indirect = nn_call(
                    self.cell_indirect,
                    cell_dependencies,
                    training = training
                )

            derivatives['d_features_pos'] = tape_d1.batch_jacobian(
                source = features_pos,
                target = features_cell_indirect
            )
            derivatives['d_features_neg'] = tape_d1.batch_jacobian(
                source = features_neg,
                target = features_cell_indirect
            )
            derivatives['d_features_electrolyte'] = tape_d1.batch_jacobian(
                source = features_electrolyte,
                target = features_cell_indirect
            )

            del tape_d1
        else:
            cell_dependencies = (
                features_pos,
                features_neg,
                features_electrolyte,
            )

            features_cell_indirect = nn_call(
                self.cell_indirect,
                cell_dependencies,
                training = training
            )

        features_cell = (
            (fetched_latent_cell * features_cell_direct) +
            ((1. - fetched_latent_cell) * features_cell_indirect)
        )

        if training:
            loss_output_cell = .1 * incentive_magnitude(
                features_cell,
                Target.Small,
                Level.Proportional
            )
            loss_output_cell = tf.reduce_mean(
                loss_output_cell,
                axis = 1,
                keepdims = True
            )

            loss_output_electrolyte = .1 * incentive_magnitude(
                features_electrolyte,
                Target.Small,
                Level.Proportional
            )
            loss_output_electrolyte = tf.reduce_mean(
                loss_output_electrolyte,
                axis = 1,
                keepdims = True
            )

        else:
            loss_output_cell = None
            loss_output_electrolyte = None

        if sample:
            eps = tf.random.normal(
                shape = [features_cell.shape[0], self.num_features]
            )
            features_cell += self.cell_direct.sample_epsilon * eps

        if training:
            loss_input_electrolyte_indirect = (
                (1. - fetched_latent_electrolyte) * loss_solvent +
                (1. - fetched_latent_electrolyte) * loss_salt +
                (1. - fetched_latent_electrolyte) * loss_additive
            )
            if compute_derivatives:
                l_solvent = tf.reduce_mean(
                    incentive_magnitude(
                        derivatives['d_features_solvent'],
                        Target.Small,
                        Level.Proportional
                    ),
                    axis = [1, 2]
                )
                l_salt = tf.reduce_mean(incentive_magnitude(
                    derivatives['d_features_salt'],
                    Target.Small,
                    Level.Proportional
                ),
                    axis = [1, 2]
                )
                l_additive = tf.reduce_mean(incentive_magnitude(
                    derivatives['d_features_additive'],
                    Target.Small,
                    Level.Proportional
                ),
                    axis = [1, 2]
                )

                mult = (1. - tf.reshape(fetched_latent_electrolyte, [-1]))
                loss_derivative_electrolyte_indirect = tf.reshape(
                    (
                        mult * l_solvent +
                        mult * l_salt +
                        mult * l_additive
                    )
                    ,
                    [-1, 1]
                )
            else:
                loss_derivative_electrolyte_indirect = 0.

            loss_electrolyte = (
                loss_output_electrolyte +
                loss_input_electrolyte_indirect +
                loss_derivative_electrolyte_indirect
            )

            loss_input_cell_indirect = ((1. - fetched_latent_cell) * loss_pos +
                                        (1. - fetched_latent_cell) * loss_neg +
                                        (
                                            1. - fetched_latent_cell) *
                                        loss_electrolyte
                                        )

            if compute_derivatives:
                l_pos = incentive_magnitude(
                    derivatives['d_features_pos'],
                    Target.Small,
                    Level.Proportional
                )
                l_neg = incentive_magnitude(
                    derivatives['d_features_neg'],
                    Target.Small,
                    Level.Proportional
                )
                l_electrolyte = incentive_magnitude(
                    derivatives['d_features_electrolyte'],
                    Target.Small,
                    Level.Proportional
                )
                mult = (1. - tf.reshape(fetched_latent_cell, [-1, 1, 1]))
                loss_derivative_cell_indirect = (
                    mult * l_pos +
                    mult * l_neg +
                    mult * l_electrolyte
                )
            else:
                loss_derivative_cell_indirect = 0.

        else:
            loss_input_cell_indirect = None
            loss_derivative_cell_indirect = None

        if training:
            loss = .1 * incentive_combine(
                [
                    (1., loss_output_cell),
                    (1., loss_input_cell_indirect),
                    (1., loss_derivative_cell_indirect),
                ]
            )
        else:
            loss = 0.

        return features_cell, loss, features_pos, features_neg,\
               fetched_latent_cell

    # TODO (Harvey): Group general/direct/derivative functions sensibly
    # (new Classes?)

    """ General variable methods """

    def reciprocal_q(self, params, training = True):
        cell_features = get_cell_features(features = params['features'])
        v, out_of_bounds_loss = self.v_direct(
            q = params['q'], shift = params['shift'],
            cell_features = cell_features, training = training
        )
        return self.q_direct(
            voltage = v, shift = params['shift'],
            cell_features = cell_features, training = training
        ), out_of_bounds_loss

    def reciprocal_v(self, params, training = True):
        cell_features = get_cell_features(features = params['features'])
        q = self.q_direct(
            voltage = params['voltage'], shift = params['shift'],
            cell_features = cell_features, training = training
        )
        return self.v_direct(
            q = q, shift = params['shift'],
            cell_features = cell_features, training = training
        )

    def cc_capacity(self, params, training = True):

        norm_cycle = get_norm_cycle_direct(
            cycle = params['cycle'],
            norm_constant = get_norm_constant(features = params['features'])
        )

        cell_features = get_cell_features(features = params['features'])

        strain = self.stress_to_strain_direct(
            norm_cycle = norm_cycle,
            cell_features = cell_features,
            encoded_stress = self.stress_to_encoded_direct(
                svit_grid = params['svit_grid'],
                count_matrix = params['count_matrix'],
            ),
            training = training
        )

        q_scale = self.q_scale_direct(
            strain = strain,
            current = params['constant_current'],
            q_scale_strainless = self.q_scale_strainless_direct(
                cell_features = cell_features,
                training = training
            ),
            training = training
        )

        shift_0 = self.shift_direct(
            strain = strain,
            current = params['end_current_prev'],
            shift_strainless = self.shift_strainless_direct(
                current = params['end_current_prev'],
                cell_features = cell_features,
                training = training
            ),
            training = training
        )

        resistance = self.r_direct(
            strain = strain,
            r_strainless = self.r_strainless_direct(
                cell_features = cell_features,
                training = training
            ),
            training = training
        )

        eq_voltage_0 = calculate_eq_voltage(
            voltage = params['end_voltage_prev'],
            current = params['end_current_prev'],
            resistance = resistance
        )

        q_0 = self.q_direct(
            voltage = eq_voltage_0,
            shift = shift_0,
            cell_features = cell_features,
            training = training
        )

        eq_voltage_1 = calculate_eq_voltage(
            voltage = params['voltage'],
            current = add_volt_dep(params['constant_current'], params),
            resistance = add_volt_dep(resistance, params)
        )

        shift_1 = self.shift_direct(
            strain = strain,
            current = params['constant_current'],
            shift_strainless = self.shift_strainless_direct(
                current = params['constant_current'],
                cell_features = cell_features,
                training = training
            ),
            training = training
        )

        q_1 = self.q_direct(
            voltage = eq_voltage_1,
            shift = add_volt_dep(shift_1, params),
            cell_features = add_volt_dep(
                cell_features, params,
                cell_features.shape[1]
            ),
            training = training
        )

        return add_volt_dep(q_scale, params) * (q_1 - add_volt_dep(q_0, params))

    def cc_voltage(self, params, training = True):
        norm_constant = get_norm_constant(features = params['features'])
        norm_cycle = get_norm_cycle_direct(
            cycle = params['cycle'],
            norm_constant = norm_constant,
        )

        cell_features = get_cell_features(features = params['features'])
        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params['svit_grid'],
            count_matrix = params['count_matrix'],
        )
        strain = self.stress_to_strain_direct(
            norm_cycle = norm_cycle,
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            training = training
        )
        q_scale_strainless = self.q_scale_strainless_direct(
            cell_features = cell_features,
            training = training
        )
        shift_0_strainless = self.shift_strainless_direct(
            current = params['end_current_prev'],
            cell_features = cell_features,
            training = training
        )
        q_scale = self.q_scale_direct(
            strain = strain,
            current = params['constant_current'],
            q_scale_strainless = q_scale_strainless,
            training = training
        )
        shift_0 = self.shift_direct(
            strain = strain,
            current = params['end_current_prev'],
            shift_strainless = shift_0_strainless,
            training = training
        )
        resistance = self.r_direct(
            strain = strain,
            r_strainless = self.r_strainless_direct(
                cell_features = cell_features,
                training = training
            ),
            training = training
        )
        eq_voltage_0 = calculate_eq_voltage(
            voltage = params['end_voltage_prev'],
            current = params['end_current_prev'],
            resistance = resistance,
        )
        q_0 = self.q_direct(
            voltage = eq_voltage_0,
            shift = shift_0,
            cell_features = cell_features,
            training = training
        )
        q_over_q = tf.reshape(params['cc_capacity'], [-1, 1]) / (
            1e-5 + tf.abs(add_volt_dep(q_scale, params))
        )
        shift_1_strainless = self.shift_strainless_direct(
            current = params['constant_current'],
            cell_features = cell_features,
            training = training
        )
        shift_1 = self.shift_direct(
            strain = strain,
            current = params['constant_current'],
            shift_strainless = shift_1_strainless,
            training = training
        )
        voltage, out_of_bounds_loss = self.v_direct(
            q = q_over_q + add_volt_dep(q_0, params),
            shift = add_volt_dep(shift_1, params),
            cell_features = add_volt_dep(
                cell_features,
                params,
                cell_features.shape[1]
            ),
            training = training
        )

        cc_voltage = voltage + add_volt_dep(
            resistance * params['constant_current'], params
        )

        return cc_voltage, out_of_bounds_loss

    def cv_capacity(self, params, training = True):
        norm_constant = get_norm_constant(features = params['features'])
        norm_cycle = get_norm_cycle_direct(
            cycle = params['cycle'], norm_constant = norm_constant
        )

        cell_features = get_cell_features(features = params['features'])

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params['svit_grid'],
            count_matrix = params['count_matrix'],
        )

        strain = self.stress_to_strain_direct(
            norm_cycle = norm_cycle,
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            training = training
        )

        cc_shift_strainless = self.shift_strainless_direct(
            current = params['end_current_prev'],
            cell_features = cell_features,
            training = training
        )
        cc_shift = self.shift_direct(
            strain = strain,
            current = params['end_current_prev'],
            shift_strainless = cc_shift_strainless,
            training = training
        )

        resistance_strainless = self.r_strainless_direct(
            cell_features = cell_features,
            training = training
        )

        resistance = self.r_direct(
            strain = strain,
            r_strainless = resistance_strainless,
            training = training
        )

        eq_voltage_0 = calculate_eq_voltage(
            voltage = params['end_voltage_prev'],
            current = params['end_current_prev'],
            resistance = resistance,
        )

        q_0 = self.q_direct(
            voltage = eq_voltage_0,
            shift = cc_shift,
            cell_features = cell_features,
            training = training
        )

        # NOTE (sam): if there truly is no dependency on current for q_scale,
        # then we can restructure the code below.
        q_scale_strainless = self.q_scale_strainless_direct(
            cell_features = cell_features,
            training = training
        )

        q_scale = self.q_scale_direct(
            strain = add_current_dep(
                strain,
                params,
                strain.shape[1]
            ),
            current = params['cv_current'],
            q_scale_strainless = add_current_dep(
                q_scale_strainless,
                params
            ),
            training = training
        )

        eq_voltage_1 = calculate_eq_voltage(
            voltage = add_current_dep(params['end_voltage'], params),
            current = params['cv_current'],
            resistance = add_current_dep(resistance, params),
        )

        cv_shift_strainless = self.shift_strainless_direct(
            current = params['cv_current'],
            cell_features = add_current_dep(
                cell_features,
                params,
                cell_features.shape[1]
            ),
            training = training
        )

        cv_shift = self.shift_direct(
            add_current_dep(
                strain,
                params,
                strain.shape[1]
            ),
            current = params['cv_current'],
            shift_strainless = cv_shift_strainless,
            training = training
        )

        q_1 = self.q_direct(
            voltage = eq_voltage_1,
            shift = cv_shift,
            cell_features = add_current_dep(
                cell_features,
                params,
                cell_features.shape[1]
            ),
            training = training
        )

        return q_scale * (q_1 - add_current_dep(q_0, params))

    """ Direct variable methods """

    def pos_projection_direct(self, cell_features, training = True):
        dependencies = (
            cell_features
        )
        return nn_call(
            self.nn_pos_projection, dependencies, training = training
        )

    def neg_projection_direct(self, cell_features, training = True):
        dependencies = (
            cell_features
        )
        return nn_call(
            self.nn_neg_projection, dependencies, training = training
        )

    def v_plus_direct(self, q, cell_features, training = True):
        pos_cell_features = self.pos_projection_direct(
            cell_features = cell_features,
            training = training
        )
        dependencies = (
            q,
            pos_cell_features
        )

        out_of_bound_loss = incentive_combine([
            (
                1.,
                incentive_inequality(q, Inequality.LessThan, 1., Level.Strong)
            ),
            (
                1.,
                incentive_inequality(
                    q, Inequality.GreaterThan, 0., Level.Strong
                )
            ),
        ])
        return nn_call(
            self.nn_v_plus, dependencies, training = training
        ), out_of_bound_loss

    def v_minus_direct(self, q, cell_features, training = True):
        neg_cell_features = self.neg_projection_direct(
            cell_features = cell_features,
            training = training
        )

        out_of_bound_loss = incentive_combine(
            [
                (1.,
                 incentive_inequality(
                     q, Inequality.LessThan, 1.,
                     Level.Strong
                 )),
                (1.,
                 incentive_inequality(
                     q, Inequality.GreaterThan, 0.,
                     Level.Strong
                 )
                 ),
            ]
        )

        dependencies = (
            q,
            neg_cell_features
        )

        return nn_call(
            self.nn_v_minus, dependencies, training = training
        ), out_of_bound_loss

    def v_direct(self, q, shift, cell_features, training = True):
        v_plus, loss_plus = self.v_plus_direct(
            q = q,
            cell_features = cell_features,
            training = training
        )
        v_minus, loss_minus = self.v_minus_direct(
            q = (q - shift),
            cell_features = cell_features,
            training = training
        )

        return v_plus - v_minus, loss_minus + loss_plus

    def q_direct(self, voltage, shift, cell_features, training = True):
        pos_cell_features = self.pos_projection_direct(
            cell_features = cell_features,
            training = training
        )
        neg_cell_features = self.neg_projection_direct(
            cell_features = cell_features,
            training = training
        )
        dependencies = (
            voltage,
            shift,
            pos_cell_features,
            neg_cell_features
        )
        return tf.nn.elu(nn_call(self.nn_q, dependencies, training = training))

    def q_scale_strainless_direct(self, cell_features, training = True):
        dependencies = (
            cell_features
        )
        return tf.nn.elu(nn_call(
            self.nn_q_scale_strainless, dependencies, training = training)
        )

    def shift_strainless_direct(self, current, cell_features, training = True):
        dependencies = (
            tf.abs(current),
            cell_features
        )
        return nn_call(
            self.nn_shift_strainless, dependencies, training = training
        )

    def r_strainless_direct(self, cell_features, training = True):
        dependencies = (
            cell_features,
        )

        return tf.nn.elu(
            nn_call(self.nn_r_strainless, dependencies, training = training))

    def q_scale_direct(
        self, strain, current, q_scale_strainless, training = True
    ):
        dependencies = (
            strain,
            # tf.abs(current),
            q_scale_strainless,
        )
        return tf.nn.elu(
            nn_call(self.nn_q_scale, dependencies, training = training))

    def shift_direct(self, strain, current, shift_strainless, training = True):
        dependencies = (
            strain,
            tf.abs(current),
            shift_strainless
        )
        return nn_call(self.nn_shift, dependencies, training = training)

    def r_direct(self, strain, r_strainless, training = True):
        dependencies = (
            strain,
            r_strainless,
        )
        return tf.nn.elu(nn_call(self.nn_r, dependencies, training = training))

    def stress_to_strain_direct(
        self, norm_cycle, cell_features, encoded_stress, training = True
    ):
        dependencies = (
            norm_cycle,
            cell_features,
            encoded_stress,
        )

        return nn_call(self.nn_strain, dependencies)

    def stress_to_encoded_direct(
        self, svit_grid, count_matrix, training = True
    ):
        return self.stress_to_encoded_layer(
            (
                svit_grid,
                count_matrix,
            ),
            training = training
        )

    """ For derivative variable methods """

    def v_plus_for_derivative(self, params, training = True):
        v_plus, loss = self.v_plus_direct(
            cell_features = get_cell_features(features = params['features']),
            q = params['q'],
            training = training
        )
        return v_plus

    def v_minus_for_derivative(self, params, training = True):
        v_m, loss = self.v_minus_direct(
            cell_features = get_cell_features(features = params['features']),
            q = params['q'],
            training = training
        )
        return v_m

    def q_for_derivative(self, params, training = True):
        return self.q_direct(
            cell_features = get_cell_features(features = params['features']),
            voltage = params['voltage'],
            shift = params['shift']
        )

    def q_scale_for_derivative(self, params, training = True):
        norm_cycle = get_norm_cycle(
            params = {
                'cycle': params['cycle'],
                'features': params['features']
            }
        )
        cell_features = get_cell_features(features = params['features'])

        strainless = self.q_scale_strainless_direct(cell_features,
                                                    training = training)

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params['svit_grid'],
            count_matrix = params['count_matrix'],
        )

        strain = self.stress_to_strain_direct(
            norm_cycle = norm_cycle,
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            training = training
        )

        return self.q_scale_direct(
            strain = strain,
            current = params['current'],
            q_scale_strainless = strainless,
            training = training
        )

    def shift_for_derivative(self, params, training = True):
        norm_cycle = get_norm_cycle(
            params = {
                'cycle': params['cycle'],
                'features': params['features']
            }
        )
        cell_features = get_cell_features(features = params['features'])

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params['svit_grid'],
            count_matrix = params['count_matrix'],
        )

        strain = self.stress_to_strain_direct(
            norm_cycle = norm_cycle,
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            training = training
        )

        strainless = self.shift_strainless_direct(
            current = params['current'],
            cell_features = cell_features,
            training = training
        )

        return self.shift_direct(
            strain = strain,
            current = params['current'],
            shift_strainless = strainless,
            training = training
        )

    def r_for_derivative(self, params, training = True):
        norm_cycle = get_norm_cycle(
            params = {
                'cycle': params['cycle'],
                'features': params['features']
            }
        )
        cell_features = get_cell_features(features = params['features'])

        strainless = self.r_strainless_direct(
            cell_features = cell_features,
            training = training
        )

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params['svit_grid'],
            count_matrix = params['count_matrix'],
        )

        strain = self.stress_to_strain_direct(
            norm_cycle = norm_cycle,
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            training = training
        )

        return self.r_direct(
            strain = strain,
            r_strainless = strainless,
            training = training
        )

    def create_derivatives(
        self,
        nn,
        params,
        der_params
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

        with tf.GradientTape(persistent = True) as tape_d3:
            for k in der_params.keys():
                if der_params[k] >= 3:
                    tape_d3.watch(params[k])

            with tf.GradientTape(persistent = True) as tape_d2:
                for k in der_params.keys():
                    if der_params[k] >= 2:
                        tape_d2.watch(params[k])

                with tf.GradientTape(persistent = True) as tape_d1:
                    for k in der_params.keys():
                        if der_params[k] >= 1:
                            tape_d1.watch(params[k])

                    res = tf.reshape(nn(params), [-1, 1])

                for k in der_params.keys():
                    if der_params[k] >= 1:
                        derivatives['d_' + k] = tape_d1.batch_jacobian(
                            source = params[k],
                            target = res
                        )[:, 0, :]

                del tape_d1

            for k in der_params.keys():
                if der_params[k] >= 2:
                    derivatives['d2_' + k] = tape_d2.batch_jacobian(
                        source = params[k],
                        target = derivatives['d_' + k]
                    )
                    if k != 'features':
                        derivatives['d2_' + k] = derivatives['d2_' + k][:, 0, :]

            del tape_d2

        for k in der_params.keys():
            if der_params[k] >= 3:
                derivatives['d3_' + k] = tape_d3.batch_jacobian(
                    source = params[k],
                    target = derivatives['d2_' + k]
                )
                if k != 'features':
                    derivatives['d3_' + k] = derivatives['d3_' + k][:, 0, :]

        del tape_d3

        return res, derivatives

    # add voltage dependence ([cyc] -> [cyc, vol])

    def get_v_curves(self, barcode, shift, voltage, q):
        """

        :param barcode:
        a barcode
        :param shift:
        a 1-D array of shifts
        :param voltage:
        a 1-D array of voltages
        :param q:
        a 1-D array of qs
        :param training:
        :return:
        returns a dictionary containing keys:
            - 'v_plus': a 1-D matrix such that v_plus[j]
                      corresponds to q[j]
            - 'v_minus': a 2-D matrix such that v_minus[i][j]
                      corresponds to shift[i] and q[j]
            - 'v': a 2-D matrix such that v[i][j]
                      corresponds to shift[i] and q[j]
            - 'q': a 2-D matrix such that q[i][k]
                      corresponds to shift[i] and voltage[k]
        """
        training = False
        indecies = tf.constant(
            self.cell_direct.id_dict[barcode],
            shape = [1]
        )

        features, _, _, _, _ = self.z_cell_from_indices(
            indices = indecies,
            training = training,
            sample = False
        )
        cell_features = get_cell_features(features = features)

        v_plus, _ = self.v_plus_direct(
            q = tf.reshape(q, [-1, 1]),
            cell_features = tf.tile(cell_features, [q.shape[0], 1]),
            training = training
        )
        v_plus = tf.reshape(v_plus, [-1])

        q_big = tf.reshape(
            tf.tile(tf.reshape(q, [1, -1, 1]), [shift.shape[0], 1, 1]), [-1, 1])
        shift_big = tf.reshape(
            tf.tile(tf.reshape(shift, [-1, 1, 1]), [1, q.shape[0], 1]), [-1, 1])

        cell_features_big = tf.reshape(
            tf.tile(tf.reshape(cell_features, [1, 1, -1]),
                    [shift.shape[0], q.shape[0], 1]),
            [-1, cell_features.shape[1]])
        v_minus, _ = self.v_minus_direct(
            q = q_big - shift_big,
            cell_features = cell_features_big,
            training = training
        )

        v_minus = tf.reshape(v_minus, [shift.shape[0], q.shape[0]])

        v, _ = self.v_direct(
            q = q_big,
            shift = shift_big,
            cell_features = cell_features_big,
            training = training,
        )

        v = tf.reshape(v, [shift.shape[0], q.shape[0]])

        voltage_big = tf.reshape(
            tf.tile(tf.reshape(voltage, [1, -1, 1]), [shift.shape[0], 1, 1]),
            [-1, 1])
        shift_big = tf.reshape(
            tf.tile(tf.reshape(shift, [-1, 1, 1]), [1, voltage.shape[0], 1]),
            [-1, 1])

        cell_features_big = tf.reshape(
            tf.tile(tf.reshape(cell_features, [1, 1, -1]),
                    [shift.shape[0], voltage.shape[0], 1]),
            [-1, cell_features.shape[1]])

        q = self.q_direct(
            voltage = voltage_big,
            shift = shift_big,
            cell_features = cell_features_big,
            training = training
        )
        q = tf.reshape(q, [shift.shape[0], voltage.shape[0]])

        return {
            'v_plus': v_plus,
            'v_minus': v_minus,
            'v': v,
            'q': q,
        }

    def call(self, x, training = False):

        cycle = x[0]  # matrix; dim: [batch, 1]
        constant_current = x[1]  # matrix; dim: [batch, 1]
        end_current_prev = x[2]  # matrix; dim: [batch, 1]
        end_voltage_prev = x[3]  # matrix; dim: [batch, 1]
        end_voltage = x[4]  # matrix; dim: [batch, 1]
        indices = x[5]  # batch of index; dim: [batch]
        voltage_tensor = x[6]  # dim: [batch, voltages]
        current_tensor = x[7]  # dim: [batch, voltages]
        svit_grid = x[8]
        count_matrix = x[9]

        features, _, _, _, _ = self.z_cell_from_indices(
            indices = indices,
            training = training,
            sample = False
        )

        # duplicate cycles and others for all the voltages
        # dimensions are now [batch, voltages, features]
        batch_count = cycle.shape[0]
        voltage_count = voltage_tensor.shape[1]
        current_count = current_tensor.shape[1]

        params = {
            "batch_count": batch_count,
            "voltage_count": voltage_count,
            "current_count": current_count,

            "voltage": tf.reshape(voltage_tensor, [-1, 1]),
            "cv_current": tf.reshape(current_tensor, [-1, 1]),

            "cycle": cycle,
            "constant_current": constant_current,
            "end_current_prev": end_current_prev,
            "end_voltage_prev": end_voltage_prev,
            "features": features,
            "end_voltage": end_voltage,

            "svit_grid": svit_grid,
            "count_matrix": count_matrix,
        }
        cc_capacity = self.cc_capacity(params, training = training)
        pred_cc_capacity = tf.reshape(cc_capacity, [-1, voltage_count])

        cv_capacity = self.cv_capacity(params, training = training)
        pred_cv_capacity = tf.reshape(cv_capacity, [-1, current_count])

        if training:
            cc_capacity = x[10]
            params['cc_capacity'] = cc_capacity
            cc_voltage, out_of_bounds_loss = self.cc_voltage(params,
                                                             training =
                                                             training)
            pred_cc_voltage = tf.reshape(cc_voltage, [-1, voltage_count])

            # NOTE(sam): this is an example of a forall. (for all voltages,
            # and all cell features)
            n_sample = 4 * 32
            sampled_voltages = tf.random.uniform(
                minval = 2.5,
                maxval = 5.,
                shape = [n_sample, 1]
            )
            sampled_qs = tf.random.uniform(
                minval = -.25,
                maxval = 1.25,
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
            sampled_features, _, sampled_pos, sampled_neg, sampled_latent\
                = self.z_cell_from_indices(
                indices = tf.random.uniform(
                    maxval = self.cell_direct.num_keys,
                    shape = [n_sample],
                    dtype = tf.int32,
                ),
                training = False,
                sample = True
            )
            sampled_features = tf.stop_gradient(sampled_features)

            sampled_shift = tf.random.uniform(
                minval = -.5,
                maxval = .5,
                shape = [n_sample, 1]
            )
            sampled_svit_grid = tf.gather(
                svit_grid,
                indices = tf.random.uniform(
                    minval = 0,
                    maxval = batch_count,
                    shape = [n_sample],
                    dtype = tf.int32,
                ),
                axis = 0
            )
            sampled_count_matrix = tf.gather(
                count_matrix,
                indices = tf.random.uniform(
                    minval = 0,
                    maxval = batch_count,
                    shape = [n_sample],
                    dtype = tf.int32,
                ),
                axis = 0
            )

            sampled_cell_features = get_cell_features(
                features = sampled_features
            )
            predicted_pos = self.pos_projection_direct(
                cell_features = sampled_cell_features,
                training = training
            )

            predicted_neg = self.neg_projection_direct(
                cell_features = sampled_cell_features,
                training = training
            )

            projection_loss = .1 * incentive_combine(
                [
                    (10., ((1. - sampled_latent) * incentive_inequality(
                        sampled_pos, Inequality.Equals, predicted_pos,
                        Level.Proportional
                    )
                           )),
                    (10., ((1. - sampled_latent) * incentive_inequality(
                        sampled_neg, Inequality.Equals, predicted_neg,
                        Level.Proportional
                    )
                           )),
                ]
            )

            reciprocal_q, oob_loss_1 = self.reciprocal_q(
                params = {
                    'q': sampled_qs,
                    'features': sampled_features,
                    'shift': sampled_shift
                },
                training = training
            )
            reciprocal_v, oob_loss_2 = self.reciprocal_v(
                params = {
                    'voltage': sampled_voltages,
                    'features': sampled_features,
                    'shift': sampled_shift
                },
                training = training
            )

            v_plus, v_plus_der = self.create_derivatives(
                self.v_plus_for_derivative,
                params = {
                    'q': sampled_qs,
                    'features': sampled_features,
                },
                der_params = {
                    'q': 1,
                }
            )
            v_minus, v_minus_der = self.create_derivatives(
                self.v_minus_for_derivative,
                params = {
                    'q': sampled_qs,
                    'features': sampled_features,
                },
                der_params = {
                    'q': 1,
                }
            )

            oob_loss_3 = incentive_combine(
                [
                    (
                        1.,
                        incentive_inequality(
                            reciprocal_q, Inequality.GreaterThan, 0.,
                            Level.Strong
                        )
                    ),
                    (
                        1.,
                        incentive_inequality(
                            reciprocal_q, Inequality.LessThan, 1.,
                            Level.Strong
                        )
                    ),
                ]
            )

            reciprocal_loss = .5 * incentive_combine(
                [
                    (
                        2.,
                        incentive_inequality(
                            sampled_voltages, Inequality.Equals, reciprocal_v,
                            Level.Proportional
                        )
                    ),
                    (
                        2.,
                        incentive_inequality(
                            sampled_qs, Inequality.Equals, reciprocal_q,
                            Level.Proportional
                        )
                    ),
                    (
                        .01,
                        incentive_magnitude(
                            v_minus, Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        .01,
                        incentive_magnitude(
                            v_plus, Target.Small,
                            Level.Proportional
                        )
                    ),

                    (
                        10.,
                        incentive_inequality(
                            v_minus, Inequality.GreaterThan, 0.,
                            Level.Strong
                        )
                    ),

                    (
                        10.,
                        incentive_inequality(
                            v_minus, Inequality.LessThan, 5.,
                            Level.Strong
                        )
                    ),

                    (
                        1.,
                        incentive_inequality(
                            v_minus_der['d_q'], Inequality.LessThan, 0.,
                            Level.Strong
                        )
                    ),

                    (
                        10.,
                        incentive_inequality(
                            v_plus, Inequality.GreaterThan, 0.,
                            Level.Strong
                        )
                    ),
                    (
                        10.,
                        incentive_inequality(
                            v_plus, Inequality.LessThan, 5.,
                            Level.Strong
                        )
                    ),

                    (
                        1.,
                        incentive_inequality(
                            v_plus_der['d_q'], Inequality.GreaterThan, 0.,
                            Level.Strong
                        )
                    ),

                ]
            )

            q, q_der = self.create_derivatives(
                self.q_for_derivative,
                params = {
                    'voltage': sampled_voltages,
                    'features': sampled_features,
                    'shift': sampled_shift
                },
                der_params = {
                    'voltage': 3,
                    'features': 2,
                    'shift': 3,
                }
            )

            q_loss = .0001 * incentive_combine(
                [
                    (
                        1.,
                        incentive_magnitude(
                            q,
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        100.,
                        incentive_inequality(
                            q, Inequality.GreaterThan, 0,
                            Level.Strong
                        )
                    ),
                    (
                        10000.,
                        incentive_inequality(
                            q_der['d_voltage'], Inequality.GreaterThan, 0.01,
                            Level.Strong
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            q_der['d3_voltage'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        .01,
                        incentive_magnitude(
                            q_der['d_features'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        .01,
                        incentive_magnitude(
                            q_der['d2_features'],
                            Target.Small,
                            Level.Strong
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            q_der['d_shift'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),

                    (
                        100.,
                        incentive_magnitude(
                            q_der['d2_shift'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            q_der['d3_shift'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                ]
            )

            q_scale, q_scale_der = self.create_derivatives(
                self.q_scale_for_derivative,
                params = {
                    'cycle': sampled_cycles,
                    'current': sampled_constant_current,
                    'features': sampled_features,
                    'svit_grid': sampled_svit_grid,
                    'count_matrix': sampled_count_matrix
                },
                der_params = {
                    'cycle': 3,
                    'current': 0,
                    'features': 2,
                }
            )

            q_scale_loss = .0001 * incentive_combine(
                [
                    (
                        10000.,
                        incentive_inequality(
                            q_scale, Inequality.GreaterThan, 0.1,
                            Level.Strong
                        )
                    ),
                    (
                        20000.,
                        incentive_inequality(
                            q_scale, Inequality.LessThan, 1.2,
                            Level.Strong
                        )
                    ),

                    (
                        10.,
                        incentive_inequality(
                            q_scale, Inequality.Equals, 1,
                            Level.Proportional
                        )
                    ),
                    (
                        1.,
                        incentive_inequality(
                            q_scale_der['d_cycle'], Inequality.LessThan, 0,
                            Level.Proportional
                        )
                    ),
                    (
                        .1,
                        incentive_inequality(
                            q_scale_der['d2_cycle'], Inequality.LessThan, 0,
                            Level.Proportional
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            q_scale_der['d_cycle'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),

                    (
                        100.,
                        incentive_magnitude(
                            q_scale_der['d2_cycle'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            q_scale_der['d3_cycle'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),

                    (
                        1.,
                        incentive_magnitude(
                            q_scale_der['d_features'],
                            Target.Small,
                            Level.Proportional
                        )
                    ),
                    (
                        1.,
                        incentive_magnitude(
                            q_scale_der['d2_features'],
                            Target.Small,
                            Level.Strong
                        )
                    ),
                ]
            )

            shift, shift_der = self.create_derivatives(
                self.shift_for_derivative,
                params = {
                    'cycle': sampled_cycles,
                    'current': sampled_constant_current,
                    'features': sampled_features,
                    'svit_grid': sampled_svit_grid,
                    'count_matrix': sampled_count_matrix
                },
                der_params = {
                    'cycle': 3,
                    'current': 3,
                    'features': 3,
                }
            )

            shift_loss = .0001 * incentive_combine(
                [
                    (
                        10000.,
                        incentive_inequality(
                            shift, Inequality.GreaterThan, -1,
                            Level.Strong
                        )
                    ),
                    (
                        10000.,
                        incentive_inequality(
                            shift, Inequality.LessThan, 1,
                            Level.Strong
                        )
                    ),
                    (
                        100.,
                        incentive_magnitude(
                            shift,
                            Target.Small,
                            Level.Proportional
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
                    'cycle': sampled_cycles,
                    'features': sampled_features,
                    'svit_grid': sampled_svit_grid,
                    'count_matrix': sampled_count_matrix
                },
                der_params = {
                    'cycle': 3,
                    'features': 2,

                }
            )

            r_loss = .0001 * incentive_combine(
                [
                    (
                        10000.,
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

            _, z_cell_loss, _, _, _ = self.z_cell_from_indices(
                indices = tf.range(
                    self.cell_direct.num_keys,
                    dtype = tf.int32,
                ),
                training = True,
                sample = False,
                compute_derivatives = True,
            )

            return {
                "pred_cc_capacity": pred_cc_capacity,
                "pred_cv_capacity": pred_cv_capacity,
                "pred_cc_voltage": pred_cc_voltage,
                "q_loss": q_loss,
                "q_scale_loss": q_scale_loss,
                "r_loss": r_loss,
                "shift_loss": shift_loss,
                "z_cell_loss": z_cell_loss,
                "reciprocal_loss": reciprocal_loss,
                "projection_loss": projection_loss,
                "out_of_bounds_loss"\
                    : out_of_bounds_loss + oob_loss_1 + oob_loss_2 + oob_loss_3,
            }

        else:

            norm_constant = get_norm_constant(features = params['features'])

            norm_cycle = get_norm_cycle_direct(
                cycle = params['cycle'], norm_constant = norm_constant,
            )

            cell_features = get_cell_features(features = params['features'])

            encoded_stress = self.stress_to_encoded_direct(
                svit_grid = params['svit_grid'],
                count_matrix = params['count_matrix'],
            )

            strain = self.stress_to_strain_direct(
                norm_cycle = norm_cycle,
                cell_features = cell_features,
                encoded_stress = encoded_stress,
                training = training
            )

            q_scale_strainless = self.q_scale_strainless_direct(
                cell_features = cell_features,
                training = training
            )

            shift_0_strainless = self.shift_strainless_direct(
                current = params['constant_current'],
                cell_features = cell_features,
                training = training
            )

            resistance_strainless = self.r_strainless_direct(
                cell_features = cell_features,
                training = training
            )

            q_scale = self.q_scale_direct(
                strain = strain,
                current = params['constant_current'],
                q_scale_strainless = q_scale_strainless,
                training = training
            )

            shift = self.shift_direct(
                strain = strain,
                current = params['constant_current'],
                shift_strainless = shift_0_strainless,
                training = training
            )

            resistance = self.r_direct(
                strain = strain,
                r_strainless = resistance_strainless,
                training = training
            )

            return {

                "pred_cc_capacity": pred_cc_capacity,
                "pred_cv_capacity": pred_cv_capacity,
                "pred_R": resistance,
                "pred_q_scale": q_scale,
                "pred_shift": shift,
            }


# stores cell features
# key: index
# value: feature (matrix)
class PrimitiveDictionaryLayer(Layer):

    def __init__(self, num_features, id_dict):
        super(PrimitiveDictionaryLayer, self).__init__()
        self.num_features = num_features
        self.num_keys = 1 + max(id_dict.values())
        self.id_dict = id_dict
        self.kernel = self.add_weight(
            "kernel", shape = [self.num_keys, self.num_features]
        )
        self.sample_epsilon = 0.05

    def get_main_ker(self):
        return self.kernel.numpy()

    def __call__(self, input, training = True, sample = False):
        fetched_features = tf.gather(self.kernel, input, axis = 0)
        if training:
            features_loss = .1 * incentive_magnitude(
                fetched_features,
                Target.Small,
                Level.Proportional
            )
            features_loss = tf.reduce_mean(
                features_loss,
                axis = 1,
                keepdims = True
            )

        else:
            features_loss = None

        if sample:
            eps = tf.random.normal(
                shape = [input.shape[0], self.num_features]
            )
            fetched_features = fetched_features + self.sample_epsilon * eps

        return fetched_features, features_loss


class StressToEncodedLayer(Layer):
    def __init__(self, n_channels):
        super(StressToEncodedLayer, self).__init__()
        self.n_channels = n_channels
        self.input_kernel = self.add_weight(
            "input_kernel", shape = [1, 1, 1, 4 + 1, self.n_channels]
        )

        self.v_i_kernel_1 = self.add_weight(
            "v_i_kernel_1", shape = [3, 3, 1, self.n_channels, self.n_channels]
        )

        self.v_i_kernel_2 = self.add_weight(
            "v_i_kernel_2", shape = [3, 3, 1, self.n_channels, self.n_channels]
        )

        self.t_kernel = self.add_weight(
            "t_kernel", shape = [1, 1, 3, self.n_channels, self.n_channels]
        )

        self.output_kernel = self.add_weight(
            "output_kernel", shape = [1, 1, 1, self.n_channels, self.n_channels]
        )

    def __call__(self, input, training = True):
        svit_grid = input[
            0]  # tensor; dim: [batch, n_sign, n_voltage, n_current,
        # n_temperature, 4]
        count_matrix = input[
            1]  # tensor; dim: [batch, n_sign, n_voltage, n_current,
        # n_temperature, 1]

        count_matrix_0 = count_matrix[:, 0, :, :, :, :]
        count_matrix_1 = count_matrix[:, 1, :, :, :, :]

        svit_grid_0 = svit_grid[:, 0, :, :, :, :]
        svit_grid_1 = svit_grid[:, 1, :, :, :, :]

        val_0 = tf.concat(
            (
                svit_grid_0,
                count_matrix_0,
            ),
            axis = -1
        )
        val_1 = tf.concat(
            (
                svit_grid_1,
                count_matrix_1,
            ),
            axis = -1
        )

        filters = [
            (self.input_kernel, 'none'),
            (self.v_i_kernel_1, 'relu'),
            (self.t_kernel, 'relu'),
            (self.v_i_kernel_2, 'relu'),
            (self.output_kernel, 'none')
        ]

        for fil, activ in filters:
            val_0 = tf.nn.convolution(input = val_0, filters = fil,
                                      padding = 'SAME')
            val_1 = tf.nn.convolution(input = val_1, filters = fil,
                                      padding = 'SAME')

            if activ is 'relu':
                val_0 = tf.nn.relu(val_0)
                val_1 = tf.nn.relu(val_1)

        # each entry is scaled by its count.
        val_0 = val_0 * count_matrix_0
        val_1 = val_1 * count_matrix_1

        # then we take the average over all the grid.
        val_0 = tf.reduce_mean(val_0, axis = [1, 2, 3], keepdims = False)
        val_1 = tf.reduce_mean(val_1, axis = [1, 2, 3], keepdims = False)

        return (val_0 + val_1)
