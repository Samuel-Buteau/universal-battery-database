import numpy
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from machine_learning.PrimitiveDictionaryLayer import PrimitiveDictionaryLayer
from machine_learning.StressToEncodedLayer import StressToEncodedLayer
from machine_learning.loss_calculator import *
from Key import Key

main_activation = tf.keras.activations.relu


def feedforward_nn_parameters(
    depth: int, width: int, last = None, finalize = False
):
    """ Create and returns a new neural network

    Returns:
        dict: Indexed by "initial", "bulk", and "final" for each part of
            neural network.
    """
    if last is None:
        last = 1

    initial = Dense(
        width,
        activation = main_activation,
        use_bias = True,
        bias_initializer = "zeros"
    )

    bulk = [
        [
            Dense(
                width,
                activation = activation,
                use_bias = True,
                bias_initializer = "zeros"
            ) for activation in ["relu", None]
        ]
        for _ in range(depth)
    ]

    if finalize:
        final = Dense(
            last,
            activation = None,
            use_bias = True,
            bias_initializer = "zeros",
            kernel_initializer = "zeros"
        )
    else:
        final = Dense(
            last,
            activation = None,
            use_bias = True,
            bias_initializer = "zeros",
        )
    return {"initial": initial, "bulk": bulk, "final": final}


def nn_call(nn_func, dependencies, training = True):
    centers = nn_func["initial"](
        tf.concat(dependencies, axis = 1),
        training = training,
    )

    for dd in nn_func["bulk"]:
        centers_prime = centers
        centers_prime = tf.nn.relu(centers_prime)
        for d in dd:
            centers_prime = d(centers_prime, training = training)
        centers = centers + centers_prime  # This is a skip connection

    return nn_func["final"](centers, training = training)


def print_cell_info(
    cell_latent_flags, cell_to_pos, cell_to_neg, cell_to_electrolyte,
    cell_to_dry_cell, dry_cell_to_meta,
    electrolyte_to_solvent, electrolyte_to_salt, electrolyte_to_additive,
    electrolyte_latent_flags, names,
):
    """ Print cell information upon the initialization of the Model """

    #TODO: names being a tuple is really dumb. use some less error prone way.
    pos_to_pos_name, neg_to_neg_name = names[0], names[1]
    electrolyte_to_electrolyte_name = names[2]
    molecule_to_molecule_name = names[3]
    dry_cell_to_dry_cell_name = names[4]

    print("\ncell_id: Known Components (Y/N):\n")
    for k in cell_latent_flags.keys():
        known = "Y"
        if cell_latent_flags[k] > .5:
            known = "N"
        print("{}\t\t{}".format(k, known))
        if known == "Y":
            pos_id = cell_to_pos[k]
            if pos_id in pos_to_pos_name.keys():
                print("\tcathode:\t\t\t{}".format(pos_to_pos_name[pos_id]))
            else:
                print("\tcathode id:\t\t\t{}".format(pos_id))
            neg_id = cell_to_neg[k]
            if neg_id in neg_to_neg_name.keys():
                print("\tanode:\t\t\t\t{}".format(neg_to_neg_name[neg_id]))
            else:
                print("\tanode id:\t\t\t{}".format(neg_id))

            dry_cell_id = cell_to_dry_cell[k]
            if dry_cell_id in dry_cell_to_dry_cell_name.keys():
                print("\tdry cell:\t\t\t{}".format(
                    dry_cell_to_dry_cell_name[dry_cell_id]))
            else:
                print("\tdry cell id:\t\t\t{}".format(dry_cell_id))

            if dry_cell_id in dry_cell_to_meta.keys():
                my_meta = dry_cell_to_meta[dry_cell_id]
            else:
                my_meta = {}

            todo= [
                ("cathode_loading", "cathode_loading",),
                ("cathode_density", "cathode_density",),
                ("cathode_thickness", "cathode_thickness",),
                ("anode_loading", "anode_loading",),
                ("anode_density", "anode_density",),
                ("anode_thickness", "anode_thickness",),

            ]

            for label, key in todo:
                val = "?"
                if key in my_meta.keys():
                    val = "{:.5f}".format(my_meta[key])

                print("\t\t{}:\t\t\t{}".format(label, val))

            electrolyte_id = cell_to_electrolyte[k]
            if electrolyte_id in electrolyte_to_electrolyte_name.keys():
                print("\telectrolyte:\t\t\t{}".format(
                    electrolyte_to_electrolyte_name[electrolyte_id]))
            else:
                print("\telectrolyte id:\t\t\t{}".format(electrolyte_id))



            electrolyte_known = "Y"
            if electrolyte_latent_flags[electrolyte_id] > .5:
                electrolyte_known = "N"
            print("\tKnown Electrolyte Components:\t{}".format(
                electrolyte_known))
            if electrolyte_known == "Y":
                for st, electrolyte_to in [
                    ("solvents", electrolyte_to_solvent),
                    ("salts", electrolyte_to_salt),
                    ("additive", electrolyte_to_additive),
                ]:
                    print("\t{}:".format(st))
                    components = electrolyte_to[electrolyte_id]
                    for s, w in components:
                        if s in molecule_to_molecule_name.keys():
                            print("\t\t{} {}".format(
                                w, molecule_to_molecule_name[s])
                            )
                        else:
                            print("\t\t{} id {}".format(w, s))
        print()


def add_v_dep(thing, params, dim = 1):
    """ Add voltage dependence: [cyc] -> [cyc, vol] """

    return tf.reshape(
        tf.tile(
            tf.expand_dims(thing, axis = 1),
            [1, params[Key.COUNT_V], 1]
        ),
        [params[Key.COUNT_BATCH] * params[Key.COUNT_V], dim]
    )


def add_current_dep(thing, params, dim = 1):
    return tf.reshape(
        tf.tile(
            tf.expand_dims(thing, axis = 1),
            [1, params[Key.COUNT_I], 1]
        ),
        [params[Key.COUNT_BATCH] * params[Key.COUNT_I], dim]
    )


def get_norm_constant(features):
    return features[:, 0:1]


def get_cell_features(features):
    return features[:, :]


def get_norm_cycle_direct(cycle, norm_constant):
    return cycle  # * (1e-10 + tf.exp(-norm_constant))


def calculate_equilibrium_voltage(v, current, resistance):
    return v - current * resistance


def get_norm_cycle(params):
    return get_norm_cycle_direct(
        norm_constant = get_norm_constant(params["features"]),
        cycle = params[Key.CYC]
    )


def create_derivatives(nn, params, der_params, internal_loss = False):
    """
    Derivatives will only be taken inside forall statements.
    If auxiliary variables must be given, create a lambda.

    Args:
        nn: The neural network for which to compute derivatives.
        params: The network"s parameters

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

                if internal_loss:
                    res, loss = nn(params)
                    res = tf.reshape(res, [-1, 1])
                else:
                    res = tf.reshape(nn(params), [-1, 1])

            for k in der_params.keys():
                if der_params[k] >= 1:
                    derivatives["d_" + k] = tape_d1.batch_jacobian(
                        source = params[k],
                        target = res
                    )[:, 0, :]

            del tape_d1

        for k in der_params.keys():
            if der_params[k] >= 2:
                derivatives["d2_" + k] = tape_d2.batch_jacobian(
                    source = params[k],
                    target = derivatives["d_" + k]
                )
                if not k in ["features", "encoded_stress"]:
                    derivatives["d2_" + k] = derivatives["d2_" + k][:, 0, :]

        del tape_d2

    for k in der_params.keys():
        if der_params[k] >= 3:
            derivatives["d3_" + k] = tape_d3.batch_jacobian(
                source = params[k],
                target = derivatives["d2_" + k]
            )
            if not k in ["features", "encoded_stress"]:
                derivatives["d3_" + k] = derivatives["d3_" + k][:, 0, :]

    del tape_d3
    if internal_loss:
        return res, derivatives, loss
    else:
        return res, derivatives


class DegradationModel(Model):
    """ Something

    Attributes:
        nn_r (dict): Neural network for R.
    """

    def __init__(
        self, depth, width,
        cell_dict, pos_dict, neg_dict, electrolyte_dict, molecule_dict,
        dry_cell_dict,
        cell_latent_flags, cell_to_pos, cell_to_neg,
        cell_to_electrolyte,
        cell_to_dry_cell,
        dry_cell_to_meta,

        electrolyte_to_solvent, electrolyte_to_salt, electrolyte_to_additive,
        electrolyte_latent_flags, names,

        n_sample,
        incentive_coeffs,
        n_channels = 16,
        min_latent = 0.1,

    ):
        super(DegradationModel, self).__init__()

        print_cell_info(
            cell_latent_flags, cell_to_pos, cell_to_neg, cell_to_electrolyte,
            cell_to_dry_cell,
            dry_cell_to_meta,
            electrolyte_to_solvent, electrolyte_to_salt,
            electrolyte_to_additive, electrolyte_latent_flags, names
        )

        self.min_latent = min_latent

        self.n_sample = n_sample
        self.incentive_coeffs = incentive_coeffs
        self.num_features = width

        """ feedforward neural networks """

        self.nn_r = feedforward_nn_parameters(depth, width)
        self.nn_scale = feedforward_nn_parameters(depth, width)
        self.nn_q = feedforward_nn_parameters(depth, width, finalize = True)
        self.nn_shift = feedforward_nn_parameters(depth, width)

        self.nn_v_plus = feedforward_nn_parameters(
            depth, width, finalize = True
        )
        self.nn_v_minus = feedforward_nn_parameters(
            depth, width, finalize = True
        )

        self.nn_pos_projection = feedforward_nn_parameters(
            depth, width, last = self.num_features
        )
        self.nn_neg_projection = feedforward_nn_parameters(
            depth, width, last = self.num_features
        )

        self.nn_dry_cell_projection = feedforward_nn_parameters(
            depth, width, last=6
        )

        """ Primitive Dictionary Layer variables """
        self.dry_cell_direct = PrimitiveDictionaryLayer(
            num_feats =6, id_dict=dry_cell_dict
        )

        self.dry_cell_latent_flags = numpy.ones(
            (self.dry_cell_direct.num_keys, 6),
            dtype = numpy.float32
        )
        self.dry_cell_given = numpy.zeros(
            (self.dry_cell_direct.num_keys, 6),
            dtype=numpy.float32
        )

        for dry_cell_id in self.dry_cell_direct.id_dict.keys():
            if dry_cell_id in dry_cell_to_meta.keys():
                todo = [
                    "cathode_loading",
                    "cathode_density",
                    "cathode_thickness",
                    "anode_loading",
                    "anode_density",
                    "anode_thickness",
                ]
                for i, key in enumerate(todo):
                    if key in dry_cell_to_meta[dry_cell_id].keys():
                        val = dry_cell_to_meta[dry_cell_id][key]
                        self.dry_cell_given[self.dry_cell_direct.id_dict[dry_cell_id], i] = val
                        self.dry_cell_latent_flags[self.dry_cell_direct.id_dict[dry_cell_id], i] = 0.

        self.dry_cell_given = tf.constant(self.dry_cell_given)
        self.dry_cell_latent_flags = tf.constant(self.dry_cell_latent_flags)

        self.cell_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_features, id_dict = cell_dict
        )
        self.pos_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_features, id_dict = pos_dict
        )
        self.neg_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_features, id_dict = neg_dict
        )
        self.electrolyte_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_features, id_dict = electrolyte_dict
        )
        self.molecule_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_features, id_dict = molecule_dict
        )

        self.num_keys = self.cell_direct.num_keys

        # cell_latent_flags is a dict with cell_ids as keys.
        # latent_flags is a numpy array such that the indecies match cell_dict
        latent_flags = numpy.ones(
            (self.cell_direct.num_keys, 1),
            dtype = numpy.float32
        )

        for cell_id in self.cell_direct.id_dict.keys():
            if cell_id in cell_latent_flags.keys():
                latent_flags[self.cell_direct.id_dict[cell_id], 0]= cell_latent_flags[cell_id]

        self.cell_latent_flags = tf.constant(latent_flags)

        cell_pointers = numpy.zeros(
            shape = (self.cell_direct.num_keys, 4), dtype = numpy.int32
        )

        # TODO(sam): dry_cell
        for cell_id in self.cell_direct.id_dict.keys():
            if cell_id in cell_to_pos.keys():
                cell_pointers[self.cell_direct.id_dict[cell_id], 0]= pos_dict[cell_to_pos[cell_id]]
            if cell_id in cell_to_neg.keys():
                cell_pointers[self.cell_direct.id_dict[cell_id], 1] = neg_dict[cell_to_neg[cell_id]]
            if cell_id in cell_to_electrolyte.keys():
                cell_pointers[self.cell_direct.id_dict[cell_id], 2] = electrolyte_dict[cell_to_electrolyte[cell_id]]
            if cell_id in cell_to_dry_cell.keys():
                cell_pointers[self.cell_direct.id_dict[cell_id], 3] = dry_cell_dict[cell_to_dry_cell[cell_id]]


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

        self.width = width
        self.n_channels = n_channels

    def cell_from_indices(
        self,
        indices, training = True, sample = False, compute_derivatives = False,
    ):
        """ Cell from indices """
        # TODO(sam): dry_cell
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

        fetched_latent_cell= self.min_latent + (1 - self.min_latent) * fetched_latent_cell
        fetched_pointers_cell = tf.gather(
            self.cell_pointers,
            indices,
            axis = 0
        )

        pos_indices = fetched_pointers_cell[:, 0]
        neg_indices = fetched_pointers_cell[:, 1]
        electrolyte_indices = fetched_pointers_cell[:, 2]
        dry_cell_indices = fetched_pointers_cell[:, 3]

        features_pos, loss_pos = self.pos_direct(
            pos_indices,
            training = training,
            sample = sample
        )

        features_neg, loss_neg = self.neg_direct(
            neg_indices,
            training = training,
            sample = sample
        )

        features_dry_cell_unknown, loss_dry_cell_unknown = self.dry_cell_direct(
            dry_cell_indices,
            training = training,
            sample = sample
        )

        latent_dry_cell = tf.gather(
            self.dry_cell_latent_flags,
            dry_cell_indices,
            axis=0
        )

        features_dry_cell_given = tf.gather(
            self.dry_cell_given,
            dry_cell_indices,
            axis=0
        )

        features_dry_cell = latent_dry_cell*features_dry_cell_unknown + (1. - latent_dry_cell)*features_dry_cell_given
        loss_dry_cell = loss_dry_cell_unknown # TODO(sam): this is not quite right


        (
            features_electrolyte_direct, loss_electrolyte_direct
        ) = self.electrolyte_direct(
            electrolyte_indices,
            training = training,
            sample = sample
        )

        fetched_latent_electrolyte = tf.gather(
            self.electrolyte_latent_flags,
            electrolyte_indices,
            axis = 0
        )
        fetched_latent_electrolyte = (
            self.min_latent + (1 - self.min_latent) * fetched_latent_electrolyte
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

            derivatives["d_features_solvent"] = tape_d1.batch_jacobian(
                source = features_solvent,
                target = features_electrolyte_indirect
            )
            derivatives["d_features_salt"] = tape_d1.batch_jacobian(
                source = features_salt,
                target = features_electrolyte_indirect
            )
            derivatives["d_features_additive"] = tape_d1.batch_jacobian(
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

        loss_electrolyte_eq = tf.reduce_mean(
            (1. - fetched_latent_electrolyte) * incentive_inequality(
                features_electrolyte_direct, Inequality.Equals,
                features_electrolyte_indirect, Level.Proportional,
            )
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

                tape_d1.watch(
                    features_dry_cell
                )

                cell_dependencies = (
                    features_pos,
                    features_neg,
                    features_electrolyte,
                    features_dry_cell
                )

                features_cell_indirect = nn_call(
                    self.cell_indirect,
                    cell_dependencies,
                    training = training
                )

            derivatives["d_features_pos"] = tape_d1.batch_jacobian(
                source = features_pos,
                target = features_cell_indirect
            )
            derivatives["d_features_neg"] = tape_d1.batch_jacobian(
                source = features_neg,
                target = features_cell_indirect
            )
            derivatives["d_features_electrolyte"] = tape_d1.batch_jacobian(
                source = features_electrolyte,
                target = features_cell_indirect
            )
            derivatives["d_features_dry_cell"] = tape_d1.batch_jacobian(
                source=features_dry_cell,
                target=features_cell_indirect
            )


            del tape_d1
        else:
            cell_dependencies = (
                features_pos,
                features_neg,
                features_electrolyte,
                features_dry_cell,
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
        loss_cell_eq = tf.reduce_mean(
            (1. - fetched_latent_cell) * incentive_inequality(
                features_cell_direct, Inequality.Equals, features_cell_indirect,
                Level.Proportional,
            )
        )

        if training:
            loss_output_cell = incentive_magnitude(
                features_cell,
                Target.Small,
                Level.Proportional
            )
            loss_output_cell = tf.reduce_mean(
                loss_output_cell,
                axis = 1,
                keepdims = True
            )

            loss_output_electrolyte = incentive_magnitude(
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
                        derivatives["d_features_solvent"],
                        Target.Small,
                        Level.Proportional
                    ),
                    axis = [1, 2]
                )
                l_salt = tf.reduce_mean(incentive_magnitude(
                    derivatives["d_features_salt"],
                    Target.Small,
                    Level.Proportional
                ),
                    axis = [1, 2]
                )
                l_additive = tf.reduce_mean(incentive_magnitude(
                    derivatives["d_features_additive"],
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
                self.incentive_coeffs["coeff_electrolyte_output"]
                * loss_output_electrolyte +
                self.incentive_coeffs["coeff_electrolyte_input"]
                * loss_input_electrolyte_indirect +
                self.incentive_coeffs["coeff_electrolyte_derivative"]
                * loss_derivative_electrolyte_indirect +
                self.incentive_coeffs["coeff_electrolyte_eq"]
                * loss_electrolyte_eq
            )

            loss_input_cell_indirect = (
                (1. - fetched_latent_cell) * loss_pos +
                (1. - fetched_latent_cell) * loss_neg +
                (1. - fetched_latent_cell) * loss_dry_cell +
                (1. - fetched_latent_cell) *
                self.incentive_coeffs["coeff_electrolyte"] * loss_electrolyte
            )

            if compute_derivatives:
                l_pos = incentive_magnitude(
                    derivatives["d_features_pos"],
                    Target.Small,
                    Level.Proportional
                )
                l_neg = incentive_magnitude(
                    derivatives["d_features_neg"],
                    Target.Small,
                    Level.Proportional
                )
                l_electrolyte = incentive_magnitude(
                    derivatives["d_features_electrolyte"],
                    Target.Small,
                    Level.Proportional
                )
                l_dry_cell = incentive_magnitude(
                    derivatives["d_features_dry_cell"],
                    Target.Small,
                    Level.Proportional
                )
                mult = (1. - tf.reshape(fetched_latent_cell, [-1, 1]))
                loss_derivative_cell_indirect = (
                    mult * tf.reduce_mean(l_pos, axis=2) +
                    mult * tf.reduce_mean(l_neg, axis=2) +
                    mult *tf.reduce_mean(l_electrolyte, axis=2) +
                    mult * tf.reduce_mean(l_dry_cell, axis=2)
                )
            else:
                loss_derivative_cell_indirect = 0.

        else:
            loss_input_cell_indirect = None
            loss_derivative_cell_indirect = None

        if training:
            loss = incentive_combine([
                (
                    self.incentive_coeffs["coeff_cell_output"],
                    loss_output_cell
                ),
                (
                    self.incentive_coeffs["coeff_cell_input"],
                    loss_input_cell_indirect
                ),
                (
                    self.incentive_coeffs["coeff_cell_derivative"],
                    loss_derivative_cell_indirect
                ),
                (
                    self.incentive_coeffs["coeff_cell_eq"],
                    loss_cell_eq
                )
            ])
        else:
            loss = 0.

        return (
            features_cell, loss, features_pos,
            features_neg, features_dry_cell, fetched_latent_cell,
        )

    def sample(self, svit_grid, batch_count, count_matrix, n_sample = 4 * 32):

        # NOTE(sam): this is an example of a forall.
        # (for all voltages, and all cell features)
        sampled_vs = tf.random.uniform(
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
            minval = -.1,
            maxval = 5.,
            shape = [n_sample, 1]
        )
        sampled_constant_current = tf.random.uniform(
            minval = tf.math.log(0.001),
            maxval = tf.math.log(5.),
            shape = [n_sample, 1]
        )
        sampled_constant_current = tf.exp(sampled_constant_current)
        sampled_constant_current_sign = tf.random.uniform(
            minval=0,
            maxval=1,
            shape=[n_sample, 1],
            dtype = tf.int32,

        )
        sampled_constant_current_sign = tf.cast(sampled_constant_current_sign, dtype=tf.float32)
        sampled_constant_current_sign = 1. * (sampled_constant_current_sign) + (-1.)*(1.-sampled_constant_current_sign)

        sampled_constant_current = sampled_constant_current_sign * sampled_constant_current

        sampled_features, _, sampled_pos, sampled_neg, sampled_dry_cell, sampled_latent = self.cell_from_indices(
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

        sampled_encoded_stress = self.stress_to_encoded_direct(
                svit_grid = sampled_svit_grid,
                count_matrix = sampled_count_matrix,
            )
        sampled_norm_constant = get_norm_constant(sampled_features)
        sampled_norm_cycle = get_norm_cycle_direct(sampled_cycles, norm_constant=sampled_norm_constant)

        return (
            sampled_vs,
            sampled_qs,
            sampled_cycles,
            sampled_constant_current,
            sampled_features,
            sampled_pos,
            sampled_neg,
            sampled_dry_cell,
            sampled_latent,
            sampled_features,
            sampled_shift,
            sampled_svit_grid,
            sampled_count_matrix,
            sampled_cell_features,
            sampled_encoded_stress,
            sampled_norm_cycle,
        )

    # TODO(Harvey): Group general/direct/derivative functions sensibly
    #               (new Classes?)

    """ General variable methods """

    def cc_capacity(self, params, training = True):

        norm_cycle = get_norm_cycle_direct(
            cycle = params[Key.CYC],
            norm_constant = get_norm_constant(features = params["features"])
        )

        cell_features = get_cell_features(features = params["features"])
        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params[Key.SVIT_GRID],
            count_matrix = params[Key.COUNT_MATRIX],
        )

        scale = self.scale_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )

        shift, _ = self.shift_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )

        resistance = self.r_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )

        v_eq_0 = calculate_equilibrium_voltage(
            v = params[Key.V_PREV_END],
            current = params[Key.I_PREV_END],
            resistance = resistance
        )

        q_0 = self.q_direct(
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            v = v_eq_0,
            shift = shift,
            cell_features = cell_features,
            current = params[Key.I_PREV_END],
            training = training
        )

        v_eq_1 = calculate_equilibrium_voltage(
            v = params[Key.V],
            current = add_v_dep(params[Key.I_CC], params),
            resistance = add_v_dep(resistance, params)
        )

        q_1 = self.q_direct(
            encoded_stress=add_v_dep(
                encoded_stress,
                params,
                encoded_stress.shape[1]
            ),
            norm_cycle=add_v_dep(norm_cycle, params),
            v = v_eq_1,
            shift = add_v_dep(shift, params),
            cell_features = add_v_dep(
                cell_features, params,
                cell_features.shape[1]
            ),
            current = add_v_dep(
                params[Key.I_CC],
                params
            ),
            training = training
        )

        return add_v_dep(scale, params) * (q_1 - add_v_dep(q_0, params))

    # NOTE: ONLY USED DURING TRAINING RIGHT NOW
    def cc_voltage(self, params, training = True):
        norm_constant = get_norm_constant(features = params["features"])
        norm_cycle = get_norm_cycle_direct(
            cycle = params[Key.CYC],
            norm_constant = norm_constant,
        )

        cell_features = get_cell_features(features = params["features"])
        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params[Key.SVIT_GRID],
            count_matrix = params[Key.COUNT_MATRIX],
        )
        scale = self.scale_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )
        shift, _ = self.shift_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )
        resistance = self.r_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )
        v_eq_0 = calculate_equilibrium_voltage(
            v = params[Key.V_PREV_END],
            current = params[Key.I_PREV_END],
            resistance = resistance,
        )
        q_0 = self.q_direct(
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            v = v_eq_0,
            shift = shift,
            cell_features = cell_features,
            current = params[Key.I_PREV_END],
            training = training
        )
        q_over_q = tf.reshape(params["cc_capacity"], [-1, 1]) / (
            1e-5 + tf.abs(add_v_dep(scale, params))
        )

        v, out_of_bounds_loss = self.v_direct(
            encoded_stress=add_v_dep(
                encoded_stress,
                params,
                encoded_stress.shape[1]
            ),
            norm_cycle=add_v_dep(norm_cycle, params),
            q = q_over_q + add_v_dep(q_0, params),
            shift = add_v_dep(shift, params),
            cell_features = add_v_dep(
                cell_features, params, cell_features.shape[1]
            ),
            current = add_v_dep(
                params[Key.I_CC],
                params,
            ),
            training = training
        )

        cc_v = v + add_v_dep(
            resistance * params[Key.I_CC], params
        )

        return cc_v, out_of_bounds_loss

    def cv_voltage(self, params, training = True):
        norm_constant = get_norm_constant(features = params["features"])
        norm_cycle = get_norm_cycle_direct(
            cycle = params[Key.CYC],
            norm_constant = norm_constant,
        )

        cell_features = get_cell_features(features = params["features"])
        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params[Key.SVIT_GRID],
            count_matrix = params[Key.COUNT_MATRIX],
        )
        scale = self.scale_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )
        shift, _ = self.shift_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )
        resistance = self.r_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )
        v_eq_0 = calculate_equilibrium_voltage(
            v = params[Key.V_PREV_END],
            current = params[Key.I_PREV_END],
            resistance = resistance,
        )
        q_0 = self.q_direct(
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            v = v_eq_0,
            shift = shift,
            cell_features = cell_features,
            current = params[Key.I_PREV_END],
            training = training
        )
        q_over_q = tf.reshape(params["cv_capacity"], [-1, 1]) / (
            1e-5 + tf.abs(add_current_dep(scale, params))
        )

        v, out_of_bounds_loss = self.v_direct(
            encoded_stress=add_current_dep(
                encoded_stress,
                params,
                encoded_stress.shape[1]
            ),
            norm_cycle=add_current_dep(norm_cycle, params),
            q = q_over_q + add_current_dep(q_0, params),
            shift = add_current_dep(shift, params),
            cell_features = add_current_dep(
                cell_features, params, cell_features.shape[1]
            ),
            current = params[Key.I_CV],
            training = training
        )

        cv_v = (v +
                (params[Key.I_CV] * add_current_dep(resistance, params))
                )

        return cv_v, out_of_bounds_loss

    def cv_capacity(self, params, training = True):
        norm_constant = get_norm_constant(features = params["features"])
        norm_cycle = get_norm_cycle_direct(
            cycle = params[Key.CYC], norm_constant = norm_constant
        )

        cell_features = get_cell_features(features = params["features"])

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params[Key.SVIT_GRID],
            count_matrix = params[Key.COUNT_MATRIX],
        )

        shift, _ = self.shift_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )

        resistance = self.r_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )

        v_eq_0 = calculate_equilibrium_voltage(
            v = params[Key.V_PREV_END],
            current = params[Key.I_PREV_END],
            resistance = resistance,
        )

        q_0 = self.q_direct(
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            v = v_eq_0,
            shift = shift,
            cell_features = cell_features,
            current = params[Key.I_PREV_END],
            training = training
        )

        # NOTE (sam): if there truly is no dependency on current for scale,
        # then we can restructure the code below.

        scale = self.scale_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )

        v_eq_1 = calculate_equilibrium_voltage(
            v = add_current_dep(params[Key.V_END], params),
            current = params[Key.I_CV],
            resistance = add_current_dep(resistance, params),
        )

        q_1 = self.q_direct(
            encoded_stress=add_current_dep(encoded_stress, params, encoded_stress.shape[1]),
            norm_cycle=add_current_dep(norm_cycle, params),
            v = v_eq_1,
            shift = add_current_dep(shift, params),
            cell_features = add_current_dep(
                cell_features, params, cell_features.shape[1]
            ),
            current = params[Key.I_CV],
            training = training
        )

        return (add_current_dep(scale, params)
                * (q_1 - add_current_dep(q_0, params)))

    def reciprocal_q(self, encoded_stress, cycle, features, q, shift, current, training = True):
        norm_constant = get_norm_constant(features=features)
        norm_cycle = get_norm_cycle_direct(
            cycle=cycle, norm_constant=norm_constant
        )
        cell_features = get_cell_features(features = features)
        v, out_of_bounds_loss = self.v_direct(
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            q = q,
            shift = shift,
            cell_features = cell_features,
            training = training,
            current = current
        )
        return self.q_direct(
            encoded_stress=encoded_stress,
            norm_cycle=norm_cycle,
            v = v,
            shift = shift,
            cell_features = cell_features,
            training = training,
            current = current
        ), out_of_bounds_loss

    def reciprocal_v(self, encoded_stress, cycle, features, v, shift, current, training = True):
        norm_constant = get_norm_constant(features=features)
        norm_cycle = get_norm_cycle_direct(
            cycle=cycle, norm_constant=norm_constant
        )
        cell_features = get_cell_features(features = features)
        q = self.q_direct(
            encoded_stress=encoded_stress,
            norm_cycle=norm_cycle,
            v = v, shift = shift,
            cell_features = cell_features,
            current = current,
            training = training
        )
        return self.v_direct(
            encoded_stress=encoded_stress,
            norm_cycle=norm_cycle,
            q = q, shift = shift,
            cell_features = cell_features,
            current = current,
            training = training
        )

    """ Stress variable methods """

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

    def dry_cell_projection_direct(self, cell_features, training = True):
        dependencies = (
            cell_features
        )
        return nn_call(
            self.nn_dry_cell_projection, dependencies, training = training
        )


    def r_direct(
        self, cell_features, encoded_stress, norm_cycle, training = True
    ):
        dependencies = (
            cell_features,
            encoded_stress,
            norm_cycle,
        )
        return tf.nn.elu(nn_call(self.nn_r, dependencies, training = training))

    #TODO(sam): not sure if this is needed
    def scale_direct(
        self, cell_features, encoded_stress, norm_cycle, training = True
    ):
        dependencies = (
            cell_features,
            encoded_stress,
            norm_cycle,
        )
        return 1. + tf.nn.tanh(
            nn_call(self.nn_scale, dependencies, training = training)
        )

    def q_direct(self, encoded_stress, norm_cycle, v, shift, cell_features, current, training = True):
        pos_cell_features = self.pos_projection_direct(
            cell_features = cell_features,
            training = training
        )
        neg_cell_features = self.neg_projection_direct(
            cell_features = cell_features,
            training = training
        )
        dependencies = (
            encoded_stress,
            norm_cycle,
            v,
            shift,
            pos_cell_features,
            neg_cell_features,
            current
        )
        return tf.nn.elu(nn_call(self.nn_q, dependencies, training = training))

    def shift_direct(
        self, cell_features, encoded_stress, norm_cycle, training = True
    ):

        dependencies = (
            cell_features,
            encoded_stress,
            norm_cycle
        )
        shift = tf.exp(
            nn_call(self.nn_shift, dependencies, training = training)
        )

        # Compute Loss
        loss = 0.

        return shift, loss

    def v_plus_direct(self, encoded_stress, norm_cycle, q, cell_features, current, training = True):
        pos_features = self.pos_projection_direct(
            cell_features = cell_features,
            training = training
        )

        dry_cell_features = self.dry_cell_projection_direct(
            cell_features=cell_features,
            training=training
        )
        dependencies = (
            encoded_stress,
            norm_cycle,
            q,
            pos_features,
            dry_cell_features,
            current
        )

        out_of_bounds_loss = calculate_out_of_bounds_loss(q=q, incentive_coeffs=self.incentive_coeffs)


        return nn_call(
            self.nn_v_plus, dependencies, training = training
        ), out_of_bounds_loss

    def v_minus_direct(self, encoded_stress, norm_cycle, q, cell_features, current, training = True):
        neg_features = self.neg_projection_direct(
            cell_features = cell_features,
            training = training
        )
        dry_cell_features = self.dry_cell_projection_direct(
            cell_features=cell_features,
            training=training
        )
        out_of_bounds_loss = calculate_out_of_bounds_loss(q=q, incentive_coeffs=self.incentive_coeffs)

        dependencies = (
            encoded_stress,
            norm_cycle,
            q,
            neg_features,
            dry_cell_features,
            current
        )

        return nn_call(
            self.nn_v_minus, dependencies, training = training
        ), out_of_bounds_loss

    def v_direct(self, encoded_stress, norm_cycle, q, shift, cell_features, current, training = True):
        v_plus, loss_plus = self.v_plus_direct(
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            q = q,
            cell_features = cell_features,
            current = current,
            training = training
        )
        v_minus, loss_minus = self.v_minus_direct(
            encoded_stress=encoded_stress,
            norm_cycle=norm_cycle,
            q = (q - shift),
            cell_features = cell_features,
            current = current,
            training = training
        )

        return v_plus - v_minus, loss_minus + loss_plus

    """ For derivative variable methods """

    def v_plus_for_derivative(self, params, training = True):
        norm_cycle = get_norm_cycle(
            params={Key.CYC: params[Key.CYC], "features": params["features"]}
        )

        v_plus, loss = self.v_plus_direct(
            encoded_stress = params["encoded_stress"],
            norm_cycle = norm_cycle,
            cell_features = get_cell_features(features = params["features"]),
            q = params["q"],
            current = params["current"],
            training = training
        )
        return v_plus

    def v_minus_for_derivative(self, params, training = True):
        norm_cycle = get_norm_cycle(
            params={Key.CYC: params[Key.CYC], "features": params["features"]}
        )

        v_m, loss = self.v_minus_direct(
            encoded_stress=params["encoded_stress"],
            norm_cycle=norm_cycle,
            cell_features = get_cell_features(features = params["features"]),
            q = params["q"],
            current = params["current"],
            training = training
        )
        return v_m

    def q_for_derivative(self, params, training = True):
        norm_cycle = get_norm_cycle(
            params={Key.CYC: params[Key.CYC], "features": params["features"]}
        )
        return self.q_direct(
            encoded_stress=params["encoded_stress"],
            norm_cycle=norm_cycle,
            cell_features = get_cell_features(features = params["features"]),
            v = params[Key.V],
            shift = params["shift"],
            current = params["current"],
            training = training,
        )

    def r_for_derivative(self, params, training = True):
        norm_cycle = get_norm_cycle(
            params = {Key.CYC: params[Key.CYC], "features": params["features"]}
        )
        cell_features = get_cell_features(features = params["features"])

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params[Key.SVIT_GRID],
            count_matrix = params[Key.COUNT_MATRIX],
        )

        return self.r_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )

    def scale_for_derivative(self, params, training = True):
        norm_cycle = get_norm_cycle(
            params = {Key.CYC: params[Key.CYC], "features": params["features"]}
        )
        cell_features = get_cell_features(features = params["features"])

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params[Key.SVIT_GRID],
            count_matrix = params[Key.COUNT_MATRIX],
        )

        return self.scale_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )

    def shift_for_derivative(self, params, training = True):
        norm_cycle = get_norm_cycle(
            params = {
                Key.CYC: params[Key.CYC],
                "features": params["features"]
            }
        )
        cell_features = get_cell_features(features = params["features"])

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params[Key.SVIT_GRID],
            count_matrix = params[Key.COUNT_MATRIX],
        )

        return self.shift_direct(
            cell_features = cell_features,
            encoded_stress = encoded_stress,
            norm_cycle = norm_cycle,
            training = training
        )

    def get_v_curves(self, cell_id, shift, v, q, current):
        #TODO(sam): this no longer works.
        """
        :param cell_id: a cell_id
        :param shift: a 1-D array of shifts
        :param v: a 1-D array of vs
        :param q: a 1-D array of qs
        :return:
        returns a dictionary containing keys:
            - "v_plus": a 1-D matrix such that v_plus[j]
                      corresponds to q[j]
            - "v_minus": a 2-D matrix such that v_minus[i][j]
                      corresponds to shift[i] and q[j]
            - Key.V: a 2-D matrix such that v[i][j]
                      corresponds to shift[i] and q[j]
            - "q": a 2-D matrix such that q[i][k]
                      corresponds to shift[i] and v[k]
        """
        training = False
        indices = tf.constant(
            self.cell_direct.id_dict[cell_id],
            shape = [1]
        )

        features, _, _, _, _, _ = self.cell_from_indices(
            indices = indices,
            training = training,
            sample = False
        )
        cell_features = get_cell_features(features = features)

        v_plus, _ = self.v_plus_direct(
            q = tf.reshape(q, [-1, 1]),
            cell_features = tf.tile(cell_features, [q.shape[0], 1]),
            current = tf.tile(current, [q.shape[0], 1]),
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
            current = tf.tile(current, [shift.shape[0] * q.shape[0], 1]),
            training = training
        )

        v_minus = tf.reshape(v_minus, [shift.shape[0], q.shape[0]])

        v_out, _ = self.v_direct(
            q = q_big,
            shift = shift_big,
            cell_features = cell_features_big,
            current = tf.tile(current, [shift.shape[0] * q.shape[0], 1]),
            training = training,
        )

        v_out = tf.reshape(v_out, [shift.shape[0], q.shape[0]])

        v_big = tf.reshape(
            tf.tile(tf.reshape(v, [1, -1, 1]), [shift.shape[0], 1, 1]),
            [-1, 1])
        shift_big = tf.reshape(
            tf.tile(tf.reshape(shift, [-1, 1, 1]), [1, v.shape[0], 1]),
            [-1, 1])

        cell_features_big = tf.reshape(
            tf.tile(tf.reshape(cell_features, [1, 1, -1]),
                    [shift.shape[0], v.shape[0], 1]),
            [-1, cell_features.shape[1]])

        q_out = self.q_direct(
            v = v_big,
            shift = shift_big,
            cell_features = cell_features_big,
            current = tf.tile(current, [shift.shape[0] * v.shape[0], 1]),
            training = training
        )
        q_out = tf.reshape(q_out, [shift.shape[0], v.shape[0]])

        return {
            "v_plus": v_plus,
            "v_minus": v_minus,
            Key.V: v_out,
            "q": q_out,
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

        features, _, _, _, _, _ = self.cell_from_indices(
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
            Key.COUNT_BATCH: batch_count,
            Key.COUNT_V: voltage_count,
            Key.COUNT_I: current_count,

            Key.V: tf.reshape(voltage_tensor, [-1, 1]),
            Key.I_CV: tf.reshape(current_tensor, [-1, 1]),

            Key.CYC: cycle,
            Key.I_CC: constant_current,
            Key.I_PREV_END: end_current_prev,
            Key.V_PREV_END: end_voltage_prev,
            "features": features,
            Key.V_END: end_voltage,

            Key.SVIT_GRID: svit_grid,
            Key.COUNT_MATRIX: count_matrix,
        }
        cc_capacity = self.cc_capacity(params, training = training)
        pred_cc_capacity = tf.reshape(cc_capacity, [-1, voltage_count])

        cv_capacity = self.cv_capacity(params, training = training)
        pred_cv_capacity = tf.reshape(cv_capacity, [-1, current_count])

        if training:
            cc_capacity = x[10]
            params["cc_capacity"] = cc_capacity
            cc_voltage, out_of_bounds_loss = self.cc_voltage(
                params, training = training
            )
            pred_cc_voltage = tf.reshape(cc_voltage, [-1, voltage_count])

            cv_capacity = x[11]
            params["cv_capacity"] = cv_capacity
            cv_voltage, out_of_bounds_loss = self.cv_voltage(
                params, training = training
            )
            pred_cv_voltage = tf.reshape(cv_voltage, [-1, current_count])
            (
                sampled_vs,
                sampled_qs,
                sampled_cycles,
                sampled_constant_current,
                sampled_features,
                sampled_pos,
                sampled_neg,
                sampled_dry_cell,
                sampled_latent,
                sampled_features,
                sampled_shift,
                sampled_svit_grid,
                sampled_count_matrix,
                sampled_cell_features,
                sampled_encoded_stress,
                sampled_norm_cycle,
            ) = self.sample(
                svit_grid, batch_count, count_matrix, n_sample = self.n_sample
            )

            predicted_pos = self.pos_projection_direct(
                cell_features = sampled_cell_features, training = training
            )

            predicted_neg = self.neg_projection_direct(
                cell_features = sampled_cell_features, training = training
            )

            predicted_dry_cell = self.dry_cell_projection_direct(
                cell_features=sampled_cell_features, training=training
            )

            projection_loss = calculate_projection_loss(
                sampled_latent, sampled_pos, sampled_neg, sampled_dry_cell,
                predicted_pos, predicted_neg,
                predicted_dry_cell,
                incentive_coeffs = self.incentive_coeffs
            )

            reciprocal_q, out_of_bounds_loss_1 = self.reciprocal_q(
                encoded_stress = sampled_encoded_stress,
                cycle = sampled_cycles,
                q = sampled_qs,
                features = sampled_features,
                shift = sampled_shift,
                current = sampled_constant_current,
                training = training
            )

            reciprocal_v, out_of_bounds_loss_2 = self.reciprocal_v(
                encoded_stress=sampled_encoded_stress,
                cycle=sampled_cycles,
                v = sampled_vs,
                features = sampled_features,
                shift = sampled_shift,
                current = sampled_constant_current,
                training = training
            )

            v_plus, v_plus_der = create_derivatives(
                self.v_plus_for_derivative,
                params = {
                    Key.CYC:sampled_cycles,
                    "encoded_stress":sampled_encoded_stress,
                    "q": sampled_qs,
                    "features": sampled_features,
                    "current": sampled_constant_current
                },
                der_params = {"q": 1, "current": 3, Key.CYC: 3}
            )
            v_minus, v_minus_der = create_derivatives(
                self.v_minus_for_derivative,
                params = {
                    Key.CYC: sampled_cycles,
                    "encoded_stress": sampled_encoded_stress,
                    "q": sampled_qs,
                    "features": sampled_features,
                    "current": sampled_constant_current
                },
                der_params = {"q": 1, "current": 3, Key.CYC: 3}
            )



            out_of_bounds_loss_3 = calculate_out_of_bounds_loss(q = reciprocal_q,
                                                                incentive_coeffs = self.incentive_coeffs)

            reciprocal_loss = calculate_reciprocal_loss(
                sampled_vs, sampled_qs,
                v_plus, v_minus, v_plus_der, v_minus_der,
                reciprocal_v, reciprocal_q,
                incentive_coeffs = self.incentive_coeffs
            )

            q, q_der = create_derivatives(
                self.q_for_derivative,
                params = {
                    Key.CYC: sampled_cycles,
                    "encoded_stress": sampled_encoded_stress,
                    Key.V: sampled_vs,
                    "features": sampled_features,
                    "shift": sampled_shift,
                    "current": sampled_constant_current
                },
                der_params = {Key.V: 3, "features": 2, "shift": 3, "current": 3, Key.CYC: 3}
            )

            q_loss = calculate_q_loss(q, q_der,
                                      incentive_coeffs = self.incentive_coeffs)

            scale, scale_der = create_derivatives(
                self.scale_for_derivative,
                params = {
                    Key.CYC: sampled_cycles,
                    "features": sampled_features,
                    Key.SVIT_GRID: sampled_svit_grid,
                    Key.COUNT_MATRIX: sampled_count_matrix
                },
                der_params = {Key.CYC: 3, "features": 2}
            )

            scale_loss = calculate_scale_loss(scale, scale_der,
                                              incentive_coeffs = self.incentive_coeffs)

            shift, shift_der, shift_internal_loss = create_derivatives(
                self.shift_for_derivative,
                params = {
                    Key.CYC: sampled_cycles,
                    "features": sampled_features,
                    Key.SVIT_GRID: sampled_svit_grid,
                    Key.COUNT_MATRIX: sampled_count_matrix
                },
                der_params = {"features": 2, Key.CYC: 3},
                internal_loss = True
            )
            shift_loss = calculate_shift_loss(shift, shift_der,
                                              incentive_coeffs = self.incentive_coeffs) + shift_internal_loss

            r, r_der = create_derivatives(
                self.r_for_derivative,
                params = {
                    Key.CYC: sampled_cycles,
                    "features": sampled_features,
                    Key.SVIT_GRID: sampled_svit_grid,
                    Key.COUNT_MATRIX: sampled_count_matrix
                },
                der_params = {Key.CYC: 3, "features": 2}
            )
            r_loss = calculate_r_loss(r, r_der,
                                      incentive_coeffs = self.incentive_coeffs)

            _, cell_loss, _, _, _, _ = self.cell_from_indices(
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
                "pred_cv_voltage": pred_cv_voltage,

                "q_loss": q_loss,
                "scale_loss": scale_loss,
                "r_loss": r_loss,
                "shift_loss": shift_loss,
                "cell_loss": cell_loss,
                "reciprocal_loss": reciprocal_loss,
                "projection_loss": projection_loss,
                "out_of_bounds_loss": out_of_bounds_loss + out_of_bounds_loss_1 + out_of_bounds_loss_2 + out_of_bounds_loss_3,
            }

        else:

            norm_constant = get_norm_constant(features = params["features"])

            norm_cycle = get_norm_cycle_direct(
                cycle = params[Key.CYC], norm_constant = norm_constant,
            )

            cell_features = get_cell_features(features = params["features"])

            encoded_stress = self.stress_to_encoded_direct(
                svit_grid = params[Key.SVIT_GRID],
                count_matrix = params[Key.COUNT_MATRIX],
            )

            scale = self.scale_direct(
                cell_features = cell_features,
                encoded_stress = encoded_stress,
                norm_cycle = norm_cycle,
                training = training
            )

            shift, _ = self.shift_direct(
                cell_features = cell_features,
                encoded_stress = encoded_stress,
                norm_cycle = norm_cycle,
                training = training
            )

            resistance = self.r_direct(
                cell_features = cell_features,
                encoded_stress = encoded_stress,
                norm_cycle = norm_cycle,
                training = training
            )

            return {

                "pred_cc_capacity": pred_cc_capacity,
                "pred_cv_capacity": pred_cv_capacity,
                "pred_R": resistance,
                "pred_scale": scale,
                "pred_shift": shift,
            }
