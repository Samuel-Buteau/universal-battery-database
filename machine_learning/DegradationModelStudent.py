import tensorflow as tf
import numpy as np

from machine_learning.DegradationModelBlackbox import DegradationModel
from machine_learning.PrimitiveDictionaryLayer import PrimitiveDictionaryLayer
from machine_learning.StressToEncodedLayer import StressToEncodedLayer
from machine_learning.fnn_functions import feedforward_nn_parameters


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
