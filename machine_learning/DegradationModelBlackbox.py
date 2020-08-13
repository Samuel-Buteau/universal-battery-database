import sys

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model

from Key import Key

from machine_learning.PrimitiveDictionaryLayer import PrimitiveDictionaryLayer
from machine_learning.loss_calculator_blackbox import calculate_q_loss
from machine_learning.fnn_functions import (
    feedforward_nn_parameters, nn_call, create_derivatives,
    add_v_dep, add_current_dep,
)


def assert_current_sign(call_params, current_tensor):
    """ Assert that the currents are given with the right sign """

    positive_cc_current = tf.math.greater_equal(call_params[Key.I_CC], 0.)
    negative_cc_current = tf.math.greater_equal(0., call_params[Key.I_CC])
    positive_prev_current = tf.math.greater_equal(
        call_params[Key.I_PREV_END], 0.,
    )
    negative_prev_current = tf.math.greater_equal(
        0., call_params[Key.I_PREV_END],
    )
    positive_cv_current = tf.reduce_all(
        tf.math.greater_equal(current_tensor, 0.), axis = 1, keepdims = True,
    )
    negative_cv_current = tf.reduce_all(
        tf.math.greater_equal(0., current_tensor), axis = 1, keepdims = True,
    )

    valid_charge = tf.math.logical_and(
        positive_cc_current,
        tf.math.logical_and(positive_cv_current, negative_prev_current),
    )
    valid_discharge = tf.math.logical_and(
        negative_cc_current,
        tf.math.logical_and(negative_cv_current, positive_prev_current),
    )

    tf.debugging.Assert(
        tf.reduce_all(
            tf.math.logical_or(valid_charge, valid_discharge),
            axis = [0, 1],
        ),
        [call_params[Key.I_CC], call_params[Key.I_PREV_END], current_tensor],
    )


def build_random_matrix(sigma, var_sigmas: list, d, f):
    if not len(var_sigmas) == d:
        raise Exception(
            "`var_sigmas` is of length {} but `d` is {}".format(
                len(var_sigmas), d,
            )
        )
    random_matrix = np.random.normal(0, sigma, (d, f))
    for i, var_sigma in enumerate(var_sigmas):
        random_matrix[i, :] *= var_sigma
    return 2 * np.pi * tf.constant(random_matrix, dtype = tf.float32)


class DegradationModel(Model):
    """
    The machine learning model of the long-term cycling data in this project.

    Notes:
        This version of Degradation Model has almost no internal structure.

    See Also:
        See the `call` method in this class for further information on using
            this class.
    """

    def __init__(
        self, depth: int, width: int, bottleneck:int, n_sample: int, options: dict,
        cell_dict: dict, random_matrix_q,
        n_channels = 16,
    ):
        """
        Args:
            options: Used to access the incentive coefficients.
        """
        super(DegradationModel, self).__init__()

        self.sample_count = n_sample  # number of samples
        self.options = options  # incentive coefficients
        self.feature_count = width  # number of features

        self.fnn_q = feedforward_nn_parameters(depth, width, finalize = True, bottleneck=bottleneck)
        self.fnn_v = feedforward_nn_parameters(depth, width, finalize = True)

        self.cell_direct = PrimitiveDictionaryLayer(
            num_feats = self.feature_count, id_dict = cell_dict,
        )

        self.width = width
        self.n_channels = n_channels

        self.fourier_features = bool(options[Key.FOUR_FEAT])

        self.q_param_count = 3
        self.v_param_count = 4
        self.f = 32

        self.random_matrix_q = random_matrix_q
        
    def transfer_q(self, CYC, V, CELL_FEAT, I, PROJ):
        q, q_der = create_derivatives(
                self.q_for_derivative,
                params = {
                    Key.CYC: CYC,
                    Key.V: V,
                    Key.CELL_FEAT: CELL_FEAT,
                    Key.I: I,
                    "get_bottleneck":True,
                    "PROJ":PROJ,
                },
                der_params = {Key.V: 1, Key.CELL_FEAT: 0, Key.I: 1, Key.CYC: 1}
            )
        return q, q_der
    
    def call(self, call_params: dict, training = False) -> dict:
        """ Call function for the Model during training or evaluation.

        Args:
            call_params: Contains -
                Cycle,
                Constant current6252 Quantum information and quantum computing,
                The end current of the previous step,
                The end voltage of the previous step,
                The end voltage of the current step,
                Indices,
                Voltage,
                Current
                S.V.I.T. grid,
                Count.
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            `{ Key.Pred.I_CC, Key.Pred.I_CV }`. During training, the
                dictionary also includes Key.Loss.Q.
        """
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
            samples = self.sample(n_sample = self.sample_count)

            q, q_der = create_derivatives(
                self.q_for_derivative,
                params = {
                    Key.CYC: samples["cycles"],
                    Key.V: samples["vs"],
                    Key.CELL_FEAT: samples["cell_feats"],
                    Key.I: samples["constant_current"],
                },
                der_params = {Key.V: 3, Key.CELL_FEAT: 2, Key.I: 3, Key.CYC: 3}
            )

            q_loss = calculate_q_loss(q, q_der, options = self.options)

            returns[Key.Q] = q
            returns[Key.Loss.Q] = q_loss

        return returns

    def cell_from_indices(self, indices, training = True, sample = False):
        """
        TODO(harvey): Need detailed explanation for what this function does.
            What are `indices`? What do the flags do?
        """
        feats_cell_direct, loss_cell = self.cell_direct(
            indices, training = training, sample = False,
        )

        feats_cell = feats_cell_direct

        if sample:
            eps = tf.random.normal(
                shape = [feats_cell.shape[0], self.feature_count],
            )
            feats_cell += self.cell_direct.sample_epsilon * eps

        return feats_cell

    def sample(self, n_sample = 4 * 32):
        """ Sample from all possible values of different variables.

        Args: TODO(harvey)
            n_sample:

        Returns:
            Sample values.
        """

        # This is an example of a forall: for all voltages and all cell features

        sampled_constant_current = tf.exp(
            tf.random.uniform(
                minval = tf.math.log(0.001), maxval = tf.math.log(5.),
                shape = [n_sample, 1],
            )
        )
        sampled_constant_current_sign = tf.cast(
            tf.random.uniform(
                minval = 0, maxval = 1, shape = [n_sample, 1], dtype = tf.int32,
            ),
            dtype = tf.float32,
        )
        sampled_constant_current_sign = 2.0 * sampled_constant_current_sign - 1.

        sampled_feats_cell = self.cell_from_indices(
            indices = tf.random.uniform(
                maxval = self.cell_direct.num_keys,
                shape = [n_sample], dtype = tf.int32,
            ),
            training = False, sample = True,
        )

        return {
            "vs": tf.random.uniform(
                minval = 2.5, maxval = 5., shape = [n_sample, 1],
            ),
            "cycles": tf.random.uniform(
                minval = -.1, maxval = 5., shape = [n_sample, 1],
            ),
            "constant_current": (
                sampled_constant_current_sign * sampled_constant_current
            ),
            "cell_feats": tf.stop_gradient(
                sampled_feats_cell
            ),
        }

    def cc_capacity(self, params: dict, training = True):
        """ Compute constant-current capacity during training or evaluation.

        Args:
            params: Contains the parameters of constant-current capacity.
            training: Flag for training or evaluation:
                True for training and False for evaluation.

        Returns:
            Computed constant-current capacity.
        """

        q_0 = self.q_direct(
            cycle = params[Key.CYC],
            v = params[Key.V_PREV_END],
            feats_cell = params[Key.CELL_FEAT],
            current = params[Key.I_PREV_END],
            training = training,
        )

        q_1 = self.q_direct(
            cycle = add_v_dep(params[Key.CYC], params),
            v = params[Key.V],
            feats_cell = add_v_dep(
                params[Key.CELL_FEAT], params, params[Key.CELL_FEAT].shape[1],
            ),
            current = add_v_dep(params[Key.I_CC], params),
            training = training,
        )

        return q_1 - add_v_dep(q_0, params)

    def cv_capacity(self, params: dict, training = True):
        """ Compute constant-voltage capacity during training or evaluation.

        Args:
            params: Parameters for computing constant-voltage (cv) capacity.
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            Computed constant-voltage capacity.
        """

        q_0 = self.q_direct(
            cycle = params[Key.CYC],
            v = params[Key.V_PREV_END],
            feats_cell = params[Key.CELL_FEAT],
            current = params[Key.I_PREV_END],
            training = training,
        )

        # NOTE (sam): if there truly is no dependency on current for scale,
        # then we can restructure the code below.

        q_1 = self.q_direct(
            cycle = add_current_dep(params[Key.CYC], params),
            v = add_current_dep(params[Key.V_END], params),
            feats_cell = add_current_dep(
                params[Key.CELL_FEAT], params, params[Key.CELL_FEAT].shape[1],
            ),
            current = params[Key.I_CV],
            training = training,
        )

        return q_1 - add_current_dep(q_0, params)

    def q_direct(
        self, cycle, v, feats_cell, current, training = True, get_bottleneck = False
    ):
        """
        Compute state of charge directly (receiving arguments directly without
        using `params`).

        Args: TODO(harvey)
            cycle: Cycle, often Key.CYC.
            v: Voltage
            feats_cell: Cell features.
            current: Current.
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            Computed state of charge.
        """

        if self.fourier_features:
            b, d, f = len(cycle), self.q_param_count, self.f
            input_vector = tf.concat(
                [cycle, v, current],
                axis = 1,
            )
            dot_product = tf.einsum(
                'bd,df->bf',
                input_vector,
                self.random_matrix_q,
            )

            dependencies = (
                tf.math.sin(dot_product),
                tf.math.cos(dot_product),
                feats_cell,
            )
        else:
            dependencies = (cycle, v, current, feats_cell)

        if get_bottleneck:
            res, bottleneck = nn_call(self.fnn_q, dependencies, training = training, get_bottleneck=get_bottleneck)
            return res, bottleneck
        else:
            res = nn_call(self.fnn_q, dependencies, training = training, get_bottleneck=get_bottleneck)
            return res

    def prev_voltage_direct(
        self, cycle, prev_end_current, constant_current, end_voltage,
        feats_cell, training = True,
    ):
        """
        Compute state of charge directly (receiving arguments directly without
        using `params`).

        Args: TODO(harvey)
            cycle: Cycle, often Key.CYC.
            v: Voltage
            feats_cell: Cell features.
            current: Current.
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            Computed state of charge.
        """

        input_dependencies = [
            cycle,
            prev_end_current,
            constant_current,
            end_voltage,
        ]
        if self.fourier_features:
            b, d, f = len(prev_end_current), self.v_param_count, self.f
            input_vector = tf.concat(input_dependencies, axis = 1)

            dot_product = tf.einsum(
                'bd,df->bf',
                input_vector,
                self.random_matrix_v,
            )

            dependencies = (
                tf.math.sin(dot_product),
                tf.math.cos(dot_product),
                feats_cell,
            )

        else:
            input_dependencies.append(feats_cell)
            dependencies = tuple(input_dependencies)

        return nn_call(self.fnn_v, dependencies, training = training)

    def q_for_derivative(self, params: dict, training = True):
        """
        Wrapper function calling `q_direct`, to be passed in to
        `create_derivatives` to compute state of charge and its first
        derivative.

        Examples:
            ```python
            # state of charge and its first derivative
            q, q_der = create_derivatives(
                self.q_for_derivative,
                params = {
                    Key.CYC: sampled_cycles,
                    Key.V: sampled_vs,
                    Key.CELL_FEAT: sampled_features_cell,
                    Key.I: sampled_constant_current
                },
                der_params = {
                    Key.V: 3, Key.CELL_FEAT: 2, Key.I: 3, Key.CYC: 3,
                }
            )
            ```

        Args:
            params: Contains input parameters for computing state of charge (q).
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            Computed state of charge; same as that for `q_direct`.
        """
        if 'get_bottleneck' in params.keys() and params["get_bottleneck"]:
            q, bottleneck = self.q_direct(
                cycle=params[Key.CYC],
                feats_cell=params[Key.CELL_FEAT],
                v=params[Key.V],
                current=params[Key.I],
                training=training,
                get_bottleneck=params["get_bottleneck"]
            )
            b = tf.reduce_mean(bottleneck * params["PROJ"], axis=-1)
            return tf.concat([q, b], axis=-1)
        else:
            return self.q_direct(
                cycle = params[Key.CYC],
                feats_cell = params[Key.CELL_FEAT],
                v = params[Key.V],
                current = params[Key.I],
                training = training,
            )

    @tf.function
    def test_all_voltages(
        self, cycle, constant_current, end_current_prev, end_voltage_prev,
        end_voltage, cell_id_index, voltages, currents, svit_grid, count_matrix,
    ):

        return self.call(
            {
                Key.CYC: tf.expand_dims(cycle, axis = 1),
                Key.I_CC: tf.tile(
                    tf.reshape(constant_current, [1, 1]),
                    [cycle.shape[0], 1],
                ),
                Key.I_PREV_END: tf.tile(
                    tf.reshape(end_current_prev, [1, 1]),
                    [cycle.shape[0], 1],
                ),
                Key.V_PREV_END: tf.tile(
                    tf.reshape(end_voltage_prev, [1, 1]),
                    [cycle.shape[0], 1],
                ),
                Key.V_END: tf.tile(
                    tf.reshape(end_voltage, [1, 1]),
                    [cycle.shape[0], 1],
                ),
                Key.INDICES: tf.tile(
                    tf.expand_dims(cell_id_index, axis = 0),
                    [cycle.shape[0]],
                ),
                Key.V_TENSOR: tf.tile(
                    tf.reshape(voltages, [1, -1]), [cycle.shape[0], 1],
                ),
                Key.I_TENSOR: tf.tile(
                    tf.reshape(currents, shape = [1, -1]), [cycle.shape[0], 1],
                ),
                Key.SVIT_GRID: tf.tile(
                    tf.expand_dims(svit_grid, axis = 0),
                    [cycle.shape[0], 1, 1, 1, 1, 1],
                ),
                Key.COUNT_MATRIX: tf.tile(
                    tf.expand_dims(count_matrix, axis = 0),
                    [cycle.shape[0], 1, 1, 1, 1, 1],
                ),
            },
            training = False,
        )

    @tf.function
    def test_single_voltage(
        self, cycle, v, constant_current, end_current_prev, end_voltage_prev,
        end_voltage, currents, cell_id_index, svit_grid, count_matrix
    ):

        return self.call(
            {
                Key.CYC: tf.expand_dims(cycle, axis = 1),
                Key.I_CC: tf.tile(
                    tf.reshape(constant_current, [1, 1]),
                    [cycle.shape[0], 1],
                ),
                Key.I_PREV_END: tf.tile(
                    tf.reshape(end_current_prev, [1, 1]),
                    [cycle.shape[0], 1],
                ),
                Key.V_PREV_END: tf.tile(
                    tf.reshape(end_voltage_prev, [1, 1]),
                    [cycle.shape[0], 1],
                ),
                Key.V_END: tf.tile(
                    tf.reshape(end_voltage, [1, 1]),
                    [cycle.shape[0], 1],
                ),
                Key.INDICES: tf.tile(
                    tf.expand_dims(cell_id_index, axis = 0),
                    [cycle.shape[0]],
                ),
                Key.V_TENSOR: tf.tile(
                    tf.reshape(v, [1, 1]),
                    [cycle.shape[0], 1],
                ),
                Key.I_TENSOR: tf.tile(
                    tf.reshape(currents, shape = [1, -1]),
                    [cycle.shape[0], 1],
                ),
                Key.SVIT_GRID: tf.tile(
                    tf.expand_dims(svit_grid, axis = 0),
                    [cycle.shape[0], 1, 1, 1, 1, 1],
                ),
                Key.COUNT_MATRIX: tf.tile(
                    tf.expand_dims(count_matrix, axis = 0),
                    [cycle.shape[0], 1, 1, 1, 1, 1],
                ),
            },
            training = False,
        )

    @tf.function
    def test_q(
        self, cycle, constant_current, end_current_prev, end_voltage_prev,
        end_voltage, cell_id_index, voltages, currents, svit_grid, count_matrix,
    ):

        call_params = {
            Key.CYC: tf.expand_dims(cycle, axis = 1),
            Key.I_CC: tf.tile(
                tf.reshape(constant_current, [1, 1]),
                [cycle.shape[0], 1],
            ),
            Key.I_PREV_END: tf.tile(
                tf.reshape(end_current_prev, [1, 1]),
                [cycle.shape[0], 1],
            ),
            Key.V_PREV_END: tf.tile(
                tf.reshape(end_voltage_prev, [1, 1]),
                [cycle.shape[0], 1],
            ),
            Key.V_END: tf.tile(
                tf.reshape(end_voltage, [1, 1]),
                [cycle.shape[0], 1],
            ),
            Key.INDICES: tf.tile(
                tf.expand_dims(cell_id_index, axis = 0),
                [cycle.shape[0]],
            ),
            Key.V_TENSOR: tf.tile(
                tf.reshape(voltages, [1, -1]), [cycle.shape[0], 1],
            ),
            Key.I_TENSOR: tf.tile(
                tf.reshape(currents, shape = [1, -1]), [cycle.shape[0], 1],
            ),
            Key.SVIT_GRID: tf.tile(
                tf.expand_dims(svit_grid, axis = 0),
                [cycle.shape[0], 1, 1, 1, 1, 1],
            ),
            Key.COUNT_MATRIX: tf.tile(
                tf.expand_dims(count_matrix, axis = 0),
                [cycle.shape[0], 1, 1, 1, 1, 1],
            ),
        }

        cycle = call_params[Key.CYC]  # matrix; dim: [batch, 1]
        voltage_tensor = call_params[Key.V_TENSOR]  # dim: [batch, voltages]
        current_tensor = call_params[Key.I_TENSOR]  # dim: [batch, voltages]
        svit_grid = call_params[Key.SVIT_GRID]
        count_matrix = call_params[Key.COUNT_MATRIX]

        feats_cell = self.cell_from_indices(
            indices = call_params[Key.INDICES],  # batch of index; dim: [batch]
            training = False, sample = False,
        )

        # duplicate cycles and others for all the voltages
        # dimensions are now [batch, voltages, features_cell]
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

            # The following matrices all have dimensions [batch, 1]
            Key.I_CC: call_params[Key.I_CC],
            Key.I_PREV_END: call_params[Key.I_PREV_END],
            Key.V_PREV_END: call_params[Key.V_PREV_END],
            Key.CELL_FEAT: feats_cell,
            Key.V_END: call_params[Key.V_END],

            Key.SVIT_GRID: svit_grid,
            Key.COUNT_MATRIX: count_matrix,
        }

        return {
            "q": tf.reshape(
                self.q_direct(
                    cycle = add_v_dep(params[Key.CYC], params),
                    v = params[Key.V],
                    feats_cell = add_v_dep(
                        params[Key.CELL_FEAT],
                        params,
                        params[Key.CELL_FEAT].shape[1],
                    ),
                    current = add_v_dep(params[Key.I_CC], params),
                    training = False,
                ),
                [-1, voltage_count],
            ),
            "q_prev": tf.reshape(
                add_v_dep(
                    self.q_direct(
                        cycle = params[Key.CYC],
                        v = self.prev_voltage_direct(
                            cycle = params[Key.CYC],
                            prev_end_current = params[Key.I_PREV_END],
                            constant_current = params[Key.I_CC],
                            end_voltage = params[Key.V_END],
                            feats_cell = params[Key.CELL_FEAT],
                        ),
                        feats_cell = params[Key.CELL_FEAT],
                        current = params[Key.I_PREV_END],
                        training = False,
                    ),
                    params,
                ),
                [-1, current_count],
            ),
        }
