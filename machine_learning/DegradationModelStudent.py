import tensorflow as tf
import numpy as np

from Key import Key
from machine_learning.DegradationModelBlackbox import DegradationModel
from machine_learning.PrimitiveDictionaryLayer import (
    Inequality, Level, PrimitiveDictionaryLayer,
    Target, incentive_combine, incentive_inequality, incentive_magnitude,
)
from machine_learning.StressToEncodedLayer import StressToEncodedLayer
from machine_learning.fnn_functions import (
    add_current_dep, add_v_dep, create_derivatives,
    feedforward_nn_parameters, nn_call,
)
from machine_learning.loss_calculator_blackbox import calculate_q_loss
from machine_learning.tf_wrappers import gather0, sum1


class DegradationModelStudent(DegradationModel):
    """ Subclass of DegradationModel for student; includes cell chemistry. """

    def __init__(
        self, depth: int, width: int, bottleneck: int, n_sample: int,
        options: dict, cell_dict: dict, random_matrix_q,
        pos_dict: dict, neg_dict: dict, lyte_dict: dict, mol_dict: dict,
        dry_cell_dict: dict, cell_latent_flags, cell_to_pos, cell_to_neg,
        cell_to_lyte, cell_to_dry_cell, dry_cell_to_meta,
        lyte_to_solvent, lyte_to_salt, lyte_to_additive, lyte_latent_flags,
        names,
        n_channels = 16, min_latent = 0.1
    ):
        super().__init__(
            depth, width, bottleneck, n_sample, options, cell_dict,
            random_matrix_q, 4, n_channels,
        )

        self.num_feats = width
        self.min_latent = min_latent

        self.dry_cell_direct = PrimitiveDictionaryLayer(
            num_feats = 6, id_dict = dry_cell_dict,
        )

        self.dry_cell_latent_flags = np.ones(
            (self.dry_cell_direct.num_keys, 6), dtype = np.float32,
        )

        self.dry_cell_given = np.zeros(
            (self.dry_cell_direct.num_keys, 6), dtype = np.float32,
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
                        self.dry_cell_given[
                            self.dry_cell_direct.id_dict[dry_cell_id], i,
                        ] = val
                        self.dry_cell_latent_flags[
                            self.dry_cell_direct.id_dict[dry_cell_id], i,
                        ] = 0.

        self.dry_cell_given = tf.constant(self.dry_cell_given)
        self.dry_cell_latent_flags = tf.constant(self.dry_cell_latent_flags)

        self.cell_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_feats, id_dict = cell_dict,
        )
        self.pos_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_feats, id_dict = pos_dict,
        )
        self.neg_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_feats, id_dict = neg_dict,
        )
        self.lyte_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_feats, id_dict = lyte_dict,
        )
        self.mol_direct = PrimitiveDictionaryLayer(
            num_feats = self.num_feats, id_dict = mol_dict,
        )

        self.num_keys = self.cell_direct.num_keys

        # cell_latent_flags is a dict with cell_ids as keys.
        # latent_flags is a numpy array such that the indecies match cell_dict
        latent_flags = np.ones(
            (self.cell_direct.num_keys, 1), dtype = np.float32,
        )

        for cell_id in self.cell_direct.id_dict.keys():
            if cell_id in cell_latent_flags.keys():
                latent_flags[
                    self.cell_direct.id_dict[cell_id], 0,
                ] = cell_latent_flags[cell_id]

        self.cell_latent_flags = tf.constant(latent_flags)

        cell_pointers = np.zeros(
            shape = (self.cell_direct.num_keys, 4), dtype = np.int32
        )

        for cell_id in self.cell_direct.id_dict.keys():
            if cell_id in cell_to_pos.keys():
                cell_pointers[
                    self.cell_direct.id_dict[cell_id], 0,
                ] = pos_dict[cell_to_pos[cell_id]]
            if cell_id in cell_to_neg.keys():
                cell_pointers[
                    self.cell_direct.id_dict[cell_id], 1,
                ] = neg_dict[cell_to_neg[cell_id]]
            if cell_id in cell_to_lyte.keys():
                cell_pointers[
                    self.cell_direct.id_dict[cell_id], 2,
                ] = lyte_dict[cell_to_lyte[cell_id]]
            if cell_id in cell_to_dry_cell.keys():
                cell_pointers[
                    self.cell_direct.id_dict[cell_id], 3,
                ] = dry_cell_dict[cell_to_dry_cell[cell_id]]

        self.cell_pointers = tf.constant(cell_pointers)
        self.cell_indirect = feedforward_nn_parameters(
            depth, width, last = self.num_feats,
        )

        self.n_solvent_max = np.max([len(v) for v in lyte_to_solvent.values()])

        self.n_salt_max = np.max([len(v) for v in lyte_to_salt.values()])
        self.n_additive_max = np.max(
            [len(v) for v in lyte_to_additive.values()]
        )

        # electrolyte latent flags
        latent_flags = np.ones(
            (self.lyte_direct.num_keys, 1), dtype = np.float32,
        )

        for lyte_id in self.lyte_direct.id_dict.keys():
            if lyte_id in lyte_latent_flags.keys():
                latent_flags[
                    self.lyte_direct.id_dict[lyte_id], 0,
                ] = lyte_latent_flags[lyte_id]

        self.lyte_latent_flags = tf.constant(latent_flags)

        # electrolyte pointers and weights

        combined_max = self.n_solvent_max + self.n_salt_max
        all_max = combined_max + self.n_additive_max
        pointers = np.zeros(
            shape = (self.lyte_direct.num_keys, all_max), dtype = np.int32,
        )
        weights = np.zeros(
            shape = (self.lyte_direct.num_keys, all_max), dtype = np.float32,
        )

        for lyte_id in self.lyte_direct.id_dict.keys():
            for ref_index, lyte_to in [
                (0, lyte_to_solvent),
                (self.n_solvent_max, lyte_to_salt),
                (combined_max, lyte_to_additive),
            ]:
                if lyte_id in lyte_to.keys():
                    my_components = lyte_to[lyte_id]
                    for i in range(len(my_components)):
                        mol_id, weight = my_components[i]
                        dict_id = self.lyte_direct.id_dict[lyte_id]
                        pointers[dict_id, i + ref_index] = mol_dict[mol_id]
                        weights[dict_id, i + ref_index] = weight

        self.lyte_pointers = tf.constant(pointers)
        self.lyte_weights = tf.constant(weights)

        self.lyte_indirect = feedforward_nn_parameters(
            depth, width, last = self.num_feats,
        )

        self.stress_to_encoded_layer = StressToEncodedLayer(
            n_channels = n_channels,
        )

    def cell_from_indices(
        self, indices,
        training = True, sample = False, compute_derivatives = False,
    ):
        """
        Returns:
            feats_cell, loss, fetched_latent_cell,
        """

        print("Student cell called")
        feats_cell_direct, loss_cell = self.cell_direct(
            indices, training = training,
        )

        lat_cell = gather0(self.cell_latent_flags, indices)
        fetched_latent_cell = self.min_latent + lat_cell * (1 - self.min_latent)
        fetched_pointers_cell = gather0(self.cell_pointers, indices)

        pos_indices = fetched_pointers_cell[:, 0]
        neg_indices = fetched_pointers_cell[:, 1]
        lyte_indices = fetched_pointers_cell[:, 2]
        dry_cell_indices = fetched_pointers_cell[:, 3]

        feats_pos, loss_pos = self.pos_direct(pos_indices, training, sample)

        feats_neg, loss_neg = self.neg_direct(neg_indices, training, sample)

        feats_dry_cell_unknown, loss_dry_cell_unknown = self.dry_cell_direct(
            dry_cell_indices, training, sample,
        )

        latent_dry_cell = gather0(self.dry_cell_latent_flags, dry_cell_indices)
        feats_dry_cell_given = gather0(self.dry_cell_given, dry_cell_indices)

        feats_dry_cell = (
            latent_dry_cell * feats_dry_cell_unknown
            + (1. - latent_dry_cell) * feats_dry_cell_given
        )
        # TODO(sam): this is not quite right
        loss_dry_cell = loss_dry_cell_unknown

        feats_lyte_direct, loss_lyte_direct = self.lyte_direct(
            lyte_indices, training, sample,
        )

        lat_lyte = gather0(self.lyte_latent_flags, lyte_indices)
        fetched_latent_lyte = self.min_latent + lat_lyte * (1 - self.min_latent)

        fetched_weights_lyte = gather0(self.lyte_weights, lyte_indices)
        fetched_pointers_lyte = tf.reshape(
            gather0(self.lyte_pointers, lyte_indices), [-1],
        )

        feats_mol, loss_mol = self.mol_direct(
            fetched_pointers_lyte, training, sample
        )

        combined_max = self.n_solvent_max + self.n_salt_max
        all_max = combined_max + self.n_additive_max
        feats_mol_reshaped = tf.reshape(
            feats_mol, [-1, all_max, self.mol_direct.num_features],
        )

        loss_mol_reshaped, loss_solvent = None, None
        loss_salt, loss_additive = None, None
        if training:
            loss_mol_reshaped = tf.reshape(loss_mol, [-1, all_max, 1])

        fetched_mol_weights = (
            tf.reshape(fetched_weights_lyte, [-1, all_max, 1])
            * feats_mol_reshaped
        )

        total_solvent = 1. / (1e-10 + sum1(
            fetched_weights_lyte[:, 0:self.n_solvent_max],
        ))

        feats_solvent = tf.reshape(total_solvent, [-1, 1]) * sum1(
            fetched_mol_weights[:, 0:self.n_solvent_max, :],
        )
        feats_salt = sum1(
            fetched_mol_weights[:, self.n_solvent_max:combined_max, :],
        )
        feats_additive = sum1(fetched_mol_weights[:, combined_max:all_max, :])

        if training:
            fetched_mol_loss_weights = (
                tf.reshape(fetched_weights_lyte, [-1, all_max, 1])
                * loss_mol_reshaped
            )
            loss_solvent = tf.reshape(total_solvent, [-1, 1]) * sum1(
                fetched_mol_loss_weights[:, 0:self.n_solvent_max, :],
            )
            loss_salt = sum1(
                fetched_mol_loss_weights[:, self.n_solvent_max:combined_max, :],
            )
            loss_additive = sum1(
                fetched_mol_loss_weights[:, combined_max: all_max, :],
            )

        derivatives = {}

        if compute_derivatives:

            with tf.GradientTape(persistent = True) as tape_d1:
                tape_d1.watch(feats_solvent)
                tape_d1.watch(feats_salt)
                tape_d1.watch(feats_additive)

                lyte_dependencies = (feats_solvent, feats_salt, feats_additive)

                feats_lyte_indirect = nn_call(
                    self.lyte_indirect, lyte_dependencies, training = training,
                )

            derivatives["d_features_solvent"] = tape_d1.batch_jacobian(
                source = feats_solvent, target = feats_lyte_indirect,
            )
            derivatives["d_features_salt"] = tape_d1.batch_jacobian(
                source = feats_salt, target = feats_lyte_indirect,
            )
            derivatives["d_features_additive"] = tape_d1.batch_jacobian(
                source = feats_additive, target = feats_lyte_indirect,
            )

            del tape_d1
        else:
            lyte_dependencies = (feats_solvent, feats_salt, feats_additive)

            feats_lyte_indirect = nn_call(
                self.lyte_indirect, lyte_dependencies, training = training,
            )

        feats_lyte = (
            fetched_latent_lyte * feats_lyte_direct
            + (1. - fetched_latent_lyte) * feats_lyte_indirect
        )

        loss_lyte_eq = tf.reduce_mean(
            (1. - fetched_latent_lyte) * incentive_inequality(
                feats_lyte_direct, Inequality.Equals,
                feats_lyte_indirect, Level.Proportional,
            )
        )

        if compute_derivatives:

            with tf.GradientTape(persistent = True) as tape_d1:
                tape_d1.watch(feats_pos)
                tape_d1.watch(feats_neg)
                tape_d1.watch(feats_lyte)

                tape_d1.watch(feats_dry_cell)

                cell_dependencies = (
                    feats_pos, feats_neg, feats_lyte, feats_dry_cell,
                )

                feats_cell_indirect = nn_call(
                    self.cell_indirect, cell_dependencies, training = training,
                )

            derivatives["d_features_pos"] = tape_d1.batch_jacobian(
                source = feats_pos, target = feats_cell_indirect,
            )
            derivatives["d_features_neg"] = tape_d1.batch_jacobian(
                source = feats_neg, target = feats_cell_indirect,
            )
            derivatives["d_features_electrolyte"] = tape_d1.batch_jacobian(
                source = feats_lyte, target = feats_cell_indirect,
            )
            derivatives["d_features_dry_cell"] = tape_d1.batch_jacobian(
                source = feats_dry_cell, target = feats_cell_indirect,
            )

            del tape_d1
        else:
            cell_dependencies = (
                feats_pos, feats_neg, feats_lyte, feats_dry_cell,
            )

            feats_cell_indirect = nn_call(
                self.cell_indirect, cell_dependencies, training = training,
            )

        feats_cell = (
            fetched_latent_cell * feats_cell_direct
            + (1. - fetched_latent_cell) * feats_cell_indirect
        )
        loss_cell_eq = tf.reduce_mean(
            (1. - fetched_latent_cell) * incentive_inequality(
                feats_cell_direct, Inequality.Equals, feats_cell_indirect,
                Level.Proportional,
            )
        )

        if training:
            loss_output_cell = incentive_magnitude(
                feats_cell, Target.Small, Level.Proportional,
            )
            loss_output_cell = tf.reduce_mean(
                loss_output_cell, axis = 1, keepdims = True,
            )

            loss_output_lyte = incentive_magnitude(
                feats_lyte, Target.Small, Level.Proportional,
            )
            loss_output_lyte = tf.reduce_mean(
                loss_output_lyte, axis = 1, keepdims = True,
            )

        else:
            loss_output_cell = None
            loss_output_lyte = None

        if sample:
            eps = tf.random.normal(
                shape = [feats_cell.shape[0], self.num_feats]
            )
            feats_cell += self.cell_direct.sample_epsilon * eps

        if training:
            loss_input_lyte_indirect = (
                (1. - fetched_latent_lyte) * loss_solvent
                + (1. - fetched_latent_lyte) * loss_salt
                + (1. - fetched_latent_lyte) * loss_additive
            )
            if compute_derivatives:
                l_solvent = tf.reduce_mean(
                    incentive_magnitude(
                        derivatives["d_features_solvent"], Target.Small,
                        Level.Proportional,
                    ),
                    axis = [1, 2],
                )
                l_salt = tf.reduce_mean(
                    incentive_magnitude(
                        derivatives["d_features_salt"], Target.Small,
                        Level.Proportional,
                    ),
                    axis = [1, 2],
                )
                l_additive = tf.reduce_mean(
                    incentive_magnitude(
                        derivatives["d_features_additive"], Target.Small,
                        Level.Proportional,
                    ),
                    axis = [1, 2],
                )

                mult = (1. - tf.reshape(fetched_latent_lyte, [-1]))
                loss_der_lyte_indirect = tf.reshape(
                    mult * l_solvent + mult * l_salt + mult * l_additive,
                    [-1, 1],
                )
            else:
                loss_der_lyte_indirect = 0.

            loss_lyte = (
                self.options["coeff_electrolyte_output"]
                * loss_output_lyte
                + self.options["coeff_electrolyte_input"]
                * loss_input_lyte_indirect
                + self.options["coeff_electrolyte_derivative"]
                * loss_der_lyte_indirect
                + self.options["coeff_electrolyte_eq"]
                * loss_lyte_eq
            )

            loss_input_cell_indirect = (
                (1. - fetched_latent_cell) * loss_pos +
                (1. - fetched_latent_cell) * loss_neg +
                (1. - fetched_latent_cell) * loss_dry_cell +
                (1. - fetched_latent_cell) *
                self.options["coeff_electrolyte"] * loss_lyte
            )

            if compute_derivatives:
                l_pos = incentive_magnitude(
                    derivatives["d_features_pos"], Target.Small,
                    Level.Proportional,
                )
                l_neg = incentive_magnitude(
                    derivatives["d_features_neg"], Target.Small,
                    Level.Proportional,
                )
                l_lyte = incentive_magnitude(
                    derivatives["d_features_electrolyte"], Target.Small,
                    Level.Proportional,
                )
                l_dry_cell = incentive_magnitude(
                    derivatives["d_features_dry_cell"], Target.Small,
                    Level.Proportional,
                )
                mult = (1. - tf.reshape(fetched_latent_cell, [-1, 1]))
                loss_derivative_cell_indirect = (
                    mult * tf.reduce_mean(l_pos, axis = 2)
                    + mult * tf.reduce_mean(l_neg, axis = 2)
                    + mult * tf.reduce_mean(l_lyte, axis = 2)
                    + mult * tf.reduce_mean(l_dry_cell, axis = 2)
                )
            else:
                loss_derivative_cell_indirect = 0.

            loss = incentive_combine([
                (
                    self.options["coeff_cell_output"],
                    loss_output_cell,
                ), (
                    self.options["coeff_cell_input"],
                    loss_input_cell_indirect,
                ), (
                    self.options["coeff_cell_derivative"],
                    loss_derivative_cell_indirect,
                ), (
                    self.options["coeff_cell_eq"],
                    loss_cell_eq,
                )
            ])
        else:
            loss = 0.

        return (
            feats_cell, loss, fetched_latent_cell,
        )

    def cc_capacity(self, params: dict, training = True):
        """ Compute constant-current capacity during training or evaluation.

        Args:
            params: Contains the parameters of constant-current capacity.
            training: Flag for training or evaluation.
        Returns:
            Computed constant-current capacity.
        """
        print("Student cc cap called")
        encoded_stress = params[Key.STRESS]
        q_0 = self.q_with_stress_direct(
            cycle = params[Key.CYC],
            v = params[Key.V_PREV_END],
            current = params[Key.I_PREV_END],
            feats_cell = params[Key.CELL_FEAT],
            encoded_stress = encoded_stress,
            training = training,
        )

        q_1 = self.q_with_stress_direct(
            cycle = add_v_dep(params[Key.CYC], params),
            v = params[Key.V],
            current = add_v_dep(params[Key.I_CC], params),
            feats_cell = add_v_dep(
                params[Key.CELL_FEAT], params, params[Key.CELL_FEAT].shape[1],
            ),
            encoded_stress = add_v_dep(
                encoded_stress, params, encoded_stress.shape[1],
            ),
            training = training,
        )

        return q_1 - add_v_dep(q_0, params)

    def cv_capacity(self, params: dict, training = True):
        """ Compute constant-voltage capacity during training or evaluation.

        Args:
            params: Parameters for computing constant-voltage (cv) capacity.
            training: Flag for training or evaluation.
        Returns:
            Computed constant-voltage capacity.
        """
        print("Student cv cap called")
        encoded_stress = params[Key.STRESS]

        q_0 = self.q_with_stress_direct(
            cycle = params[Key.CYC],
            v = params[Key.V_PREV_END],
            current = params[Key.I_PREV_END],
            feats_cell = params[Key.CELL_FEAT],
            encoded_stress = encoded_stress,
            training = training,
        )

        # NOTE (sam): if there truly is no dependency on current for scale,
        # then we can restructure the code below.

        q_1 = self.q_with_stress_direct(
            cycle = add_current_dep(params[Key.CYC], params),
            v = add_current_dep(params[Key.V_END], params),
            current = params[Key.I_CV],
            feats_cell = add_current_dep(
                params[Key.CELL_FEAT], params, params[Key.CELL_FEAT].shape[1],
            ),
            encoded_stress = add_current_dep(
                encoded_stress, params, encoded_stress.shape[1],
            ),
            training = training,
        )

        return q_1 - add_current_dep(q_0, params)

    def stress_to_encoded_direct(
        self, svit_grid, count_matrix, training = True,
    ):
        """ Compute stress directly
         Returns:
            (svit_grid, count_matrix) and the training flag.
        """
        print("Student stress called")
        return tf.tile(
            self.stress_to_encoded_layer(
                (svit_grid, count_matrix), training = training,
            ),
            [8, 1],
        )

    def q_with_stress_direct(
        self, cycle, v, feats_cell, current, encoded_stress,
        training = True, get_bottleneck = False,
    ):
        """ `q_direct` but with stress
        Returns:
            If get_bottleneck, then q and bottleneck; just q if not.
        """
        print("Student q called")
        inputs = [cycle, v, current]
        if self.fourier_features:
            b, d, f = len(cycle), self.q_param_count, self.f
            input_vector = tf.concat(inputs, axis = 1)
            dot_product = tf.einsum(
                'bd,df->bf',
                input_vector,
                self.random_matrix_q,
            )

            dependencies = (
                tf.math.sin(dot_product),
                tf.math.cos(dot_product),
                encoded_stress,
                feats_cell,
            )
        else:
            dependencies = tuple(inputs)

        return nn_call(
            self.fnn_q, dependencies,
            training = training, get_bottleneck = get_bottleneck,
        )

    def q_with_stress_for_derivative(self, params: dict, training = True):
        """ Wrapper function calling `q_direct`, to be passed in to
        `create_derivatives` to compute state of charge and its first
        derivative.

        Args:
            params: Contains input parameters for computing state of charge (q).
            training: Flag for training or evaluation.
        Returns:
            Computed state of charge; same as that for `q_direct`.
        """
        encoded_stress = params[Key.STRESS]
        if 'get_bottleneck' in params.keys() and params["get_bottleneck"]:
            q, bottleneck = self.q_with_stress_direct(
                cycle = params[Key.CYC],
                feats_cell = params[Key.CELL_FEAT],
                v = params[Key.V],
                current = params[Key.I],
                encoded_stress = encoded_stress,
                training = training,
                get_bottleneck = params["get_bottleneck"]
            )

            return tf.reduce_mean(
                bottleneck * params["PROJ"], axis = -1, keepdims = True,
            )
        else:
            return self.q_with_stress_direct(
                cycle = params[Key.CYC],
                feats_cell = params[Key.CELL_FEAT],
                v = params[Key.V],
                encoded_stress = encoded_stress,
                current = params[Key.I],
                training = training,
            )

    def transfer_q_with_stress(
        self, cycle, voltage, cell_feat, current, encoded_stress, proj,
        get_bottleneck = False,
    ):
        """ `transfer_ q` but calls `q_with_stress_for_derivative` instead.
        Returns:
            `q` and `q_der`
        """
        return create_derivatives(
            self.q_with_stress_for_derivative,
            params = {
                Key.CYC: cycle,
                Key.V: voltage,
                Key.CELL_FEAT: cell_feat,
                Key.I: current,
                Key.STRESS: encoded_stress,
                "get_bottleneck": get_bottleneck,
                "PROJ": proj,
            },
            der_params = {Key.V: 2, Key.CELL_FEAT: 0, Key.I: 2, Key.CYC: 2},
        )
