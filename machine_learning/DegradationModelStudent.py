import tensorflow as tf
import numpy as np

from Key import Key
from machine_learning.DegradationModelBlackbox import DegradationModel
from machine_learning.PrimitiveDictionaryLayer import (
    Inequality, Level, PrimitiveDictionaryLayer,
    Target, incentive_combine, incentive_inequality, incentive_magnitude
)
from machine_learning.StressToEncodedLayer import StressToEncodedLayer
from machine_learning.fnn_functions import (
    create_derivatives,
    feedforward_nn_parameters, nn_call
)
from machine_learning.loss_calculator_blackbox import calculate_q_loss


class DegradationModelStudent(DegradationModel):

    def __init__(
        self, depth: int, width: int, bottleneck: int, n_sample: int,
        options: dict, cell_dict: dict, random_matrix_q,
        pos_dict: dict, neg_dict: dict, lyte_dict: dict, mol_dict: dict,
        dry_cell_dict: dict, cell_latent_flags, cell_to_pos, cell_to_neg,
        cell_to_lyte, cell_to_dry_cell, dry_cell_to_meta,
        lyte_to_solvent, lyte_to_salt, lyte_to_additive, lyte_latent_flags,
        names,
        n_channels = 16,
    ):
        super(DegradationModel, self).__init__(
            depth, width, bottleneck, n_sample, options, cell_dict,
            random_matrix_q, n_channels,
        )

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

        pointers = np.zeros(
            shape = (
                self.lyte_direct.num_keys,
                self.n_solvent_max + self.n_salt_max + self.n_additive_max,
            ),
            dtype = np.int32,
        )
        weights = np.zeros(
            shape = (
                self.lyte_direct.num_keys,
                self.n_solvent_max + self.n_salt_max + self.n_additive_max,
            ),
            dtype = np.float32,
        )

        for lyte_id in self.lyte_direct.id_dict.keys():
            for reference_index, lyte_to in [
                (0, lyte_to_solvent),
                (self.n_solvent_max, lyte_to_salt),
                (self.n_solvent_max + self.n_salt_max, lyte_to_additive),
            ]:
                if lyte_id in lyte_to.keys():
                    my_components = lyte_to[lyte_id]
                    for i in range(len(my_components)):
                        mol_id, weight = my_components[i]
                        pointers[
                            self.lyte_direct.id_dict[lyte_id],
                            i + reference_index,
                        ] = mol_dict[mol_id]
                        weights[
                            self.lyte_direct.id_dict[lyte_id],
                            i + reference_index,
                        ] = weight

        self.lyte_pointers = tf.constant(pointers)
        self.lyte_weights = tf.constant(weights)

        self.lyte_indirect = feedforward_nn_parameters(
            depth, width, last = self.num_feats,
        )

        self.stress_to_encoded_layer = StressToEncodedLayer(
            n_channels = n_channels,
        )

    def call(self, call_params: dict, training = False) -> dict:

        cycle = call_params[Key.CYC]  # matrix; dim: [batch, 1]
        voltage_tensor = call_params[Key.V_TENSOR]  # dim: [batch, voltages]
        current_tensor = call_params[Key.I_TENSOR]  # dim: [batch, voltages]
        svit_grid = call_params[Key.SVIT_GRID]
        count_matrix = call_params[Key.COUNT_MATRIX]

        feats_cell = self.cell_from_indices(
            indices = call_params[Key.INDICES],  # batch of index; dim: [batch]
            training = training, sample = False,
        )

        # duplicate cycles and others for all the voltages
        # dimensions are now [batch, voltages, features_cell]
        batch_count = cycle.shape[0]
        voltage_count = voltage_tensor.shape[1]
        current_count = current_tensor.shape[1]

        capacity_params = {
            Key.COUNT_BATCH: batch_count,
            Key.COUNT_V: voltage_count,
            Key.COUNT_I: current_count,

            Key.V: tf.reshape(voltage_tensor, [-1, 1]),
            Key.I_CV: tf.reshape(current_tensor, [-1, 1]),

            Key.CYC: cycle,

            # The following matrices all have dimensions [batch, 1]
            Key.I_CC: call_params[Key.I_CC],
            Key.I_PREV_END: call_params[Key.I_PREV_END],
            Key.V_PREV_END: call_params[Key.V_PREV_END],
            Key.CELL_FEAT: feats_cell,
            Key.V_END: call_params[Key.V_END],

            Key.SVIT_GRID: svit_grid,
            Key.COUNT_MATRIX: count_matrix,
        }

        # assert_current_sign(call_params, current_tensor)

        cc_capacity = self.cc_capacity(capacity_params, training = training)
        pred_cc_capacity = tf.reshape(cc_capacity, [-1, voltage_count])

        cv_capacity = self.cv_capacity(capacity_params, training = training)
        pred_cv_capacity = tf.reshape(cv_capacity, [-1, current_count])

        returns = {
            Key.Pred.I_CC: pred_cc_capacity,
            Key.Pred.I_CV: pred_cv_capacity,
        }

        if training:
            (
                sampled_vs,
                sampled_qs,
                sampled_cycles,
                sampled_constant_current,
                sampled_feats_cell,
                sampled_latent,
                sampled_svit_grid,
                sampled_count_matrix,
                sampled_encoded_stress,
            ) = self.sample(
                svit_grid, batch_count, count_matrix, n_sample = self.n_sample,
            )

            q, q_der = create_derivatives(
                self.q_for_derivative,
                params = {
                    Key.CYC: sampled_cycles,
                    Key.STRESS: sampled_encoded_stress,
                    Key.V: sampled_vs,
                    Key.CELL_FEAT: sampled_feats_cell,
                    Key.I: sampled_constant_current,
                },
                der_params = {Key.V: 3, Key.CELL_FEAT: 0, Key.I: 3, Key.CYC: 3}
            )

            q_loss = calculate_q_loss(q, q_der, options = self.options)

            _, cell_loss, _ = self.cell_from_indices(
                indices = tf.range(self.cell_direct.num_keys, dtype = tf.int32),
                training = True,
                sample = False,
                compute_derivatives = True,
            )

            returns[Key.Q] = q
            returns[Key.Loss.Q] = q_loss

        return returns

    def cell_from_indices(
        self, indices,
        training = True, sample = False, compute_derivatives = False,
    ):
        feats_cell_direct, loss_cell = self.cell_direct(
            indices, training = training, sample = False,
        )

        fetched_latent_cell = tf.gather(
            self.cell_latent_flags, indices, axis = 0,
        )

        fetched_latent_cell = (
            self.min_latent + (1 - self.min_latent) * fetched_latent_cell
        )
        fetched_pointers_cell = tf.gather(
            self.cell_pointers, indices, axis = 0,
        )

        pos_indices = fetched_pointers_cell[:, 0]
        neg_indices = fetched_pointers_cell[:, 1]
        lyte_indices = fetched_pointers_cell[:, 2]
        dry_cell_indices = fetched_pointers_cell[:, 3]

        feats_pos, loss_pos = self.pos_direct(
            pos_indices, training = training, sample = sample,
        )

        feats_neg, loss_neg = self.neg_direct(
            neg_indices, training = training, sample = sample,
        )

        feats_dry_cell_unknown, loss_dry_cell_unknown = self.dry_cell_direct(
            dry_cell_indices, training = training, sample = sample,
        )

        latent_dry_cell = tf.gather(
            self.dry_cell_latent_flags, dry_cell_indices, axis = 0,
        )

        feats_dry_cell_given = tf.gather(
            self.dry_cell_given, dry_cell_indices, axis = 0,
        )

        feats_dry_cell = (
            latent_dry_cell * feats_dry_cell_unknown
            + (1. - latent_dry_cell) * feats_dry_cell_given
        )
        # TODO(sam): this is not quite right
        loss_dry_cell = loss_dry_cell_unknown

        feats_lyte_direct, loss_lyte_direct = self.lyte_direct(
            lyte_indices, training = training, sample = sample,
        )

        fetched_latent_lyte = tf.gather(
            self.lyte_latent_flags, lyte_indices, axis = 0,
        )
        fetched_latent_lyte = (
            self.min_latent + (1 - self.min_latent) * fetched_latent_lyte
        )

        fetched_pointers_lyte = tf.gather(
            self.lyte_pointers, lyte_indices, axis = 0,
        )
        fetched_weights_lyte = tf.gather(
            self.lyte_weights, lyte_indices, axis = 0,
        )
        fetched_pointers_lyte_reshaped = tf.reshape(
            fetched_pointers_lyte, [-1],
        )

        feats_mol, loss_mol = self.mol_direct(
            fetched_pointers_lyte_reshaped,
            training = training, sample = sample
        )

        feats_mol_reshaped = tf.reshape(
            feats_mol,
            [
                -1,
                self.n_solvent_max + self.n_salt_max + self.n_additive_max,
                self.mol_direct.num_features,
            ],
        )

        if training:
            loss_mol_reshaped = tf.reshape(
                loss_mol,
                [
                    -1,
                    self.n_solvent_max + self.n_salt_max + self.n_additive_max,
                    1,
                ],
            )

        fetched_mol_weights = tf.reshape(
            fetched_weights_lyte,
            [-1, self.n_solvent_max + self.n_salt_max + self.n_additive_max, 1],
        ) * feats_mol_reshaped

        total_solvent = 1. / (1e-10 + tf.reduce_sum(
            fetched_weights_lyte[:, 0:self.n_solvent_max], axis = 1,
        ))

        feats_solvent = tf.reshape(total_solvent, [-1, 1]) * tf.reduce_sum(
            fetched_mol_weights[:, 0:self.n_solvent_max, :], axis = 1,
        )
        combined_max = self.n_solvent_max + self.n_salt_max
        feats_salt = tf.reduce_sum(
            fetched_mol_weights[:, self.n_solvent_max:combined_max, :],
            axis = 1,
        )
        all_max = combined_max + self.n_additive_max
        feats_additive = tf.reduce_sum(
            fetched_mol_weights[:, combined_max:all_max, :],
            axis = 1,
        )

        if training:
            fetched_mol_loss_weights = tf.reshape(
                fetched_weights_lyte,
                [
                    -1,
                    self.n_solvent_max + self.n_salt_max + self.n_additive_max,
                    1,
                ],
            ) * loss_mol_reshaped
            loss_solvent = tf.reshape(total_solvent, [-1, 1]) * tf.reduce_sum(
                fetched_mol_loss_weights[:, 0:self.n_solvent_max, :],
                axis = 1,
            )
            loss_salt = tf.reduce_sum(
                fetched_mol_loss_weights[
                :, self.n_solvent_max:self.n_solvent_max + self.n_salt_max, :,
                ],
                axis = 1,
            )
            loss_additive = tf.reduce_sum(
                fetched_mol_loss_weights[
                :,
                self.n_solvent_max + self.n_salt_max:
                self.n_solvent_max + self.n_salt_max + self.n_additive_max,
                :
                ],
                axis = 1,
            )

        derivatives = {}

        if compute_derivatives:

            with tf.GradientTape(persistent = True) as tape_d1:
                tape_d1.watch(feats_solvent)
                tape_d1.watch(feats_salt)
                tape_d1.watch(feats_additive)

                lyte_dependencies = (
                    feats_solvent,
                    feats_salt,
                    feats_additive,
                )

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
            lyte_dependencies = (
                feats_solvent,
                feats_salt,
                feats_additive,
            )

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
                    feats_pos,
                    feats_neg,
                    feats_lyte,
                    feats_dry_cell,
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
                feats_pos,
                feats_neg,
                feats_lyte,
                feats_dry_cell,
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

        else:
            loss_input_cell_indirect = None
            loss_derivative_cell_indirect = None

        if training:
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
