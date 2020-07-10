import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from Key import Key

from machine_learning.PrimitiveDictionaryLayer import PrimitiveDictionaryLayer
from machine_learning.StressToEncodedLayer import StressToEncodedLayer
from machine_learning.loss_calculator_blackbox import calculate_q_loss
from machine_learning.incentives import (
    Inequality, Level, Target,
    incentive_inequality, incentive_magnitude, incentive_combine,
)

main_activation = tf.keras.activations.relu


def feedforward_nn_parameters(
    depth: int, width: int, last = None, finalize = False
):
    """ Create a new feedforward neural network

    Args:
        depth: The depth the feedforward neural network
        width: The width of the feedforward neural network
        last: TODO(harvey, confusion)
        finalize: TODO(harvey, confusion)

    Returns:
        { "initial", "bulk", and "final" }, each key corresponds to a component
            of the neural network.
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
                bias_initializer = "zeros",
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
            kernel_initializer = "zeros",
        )
    else:
        final = Dense(
            last,
            activation = None,
            use_bias = True,
            bias_initializer = "zeros",
        )
    return {"initial": initial, "bulk": bulk, "final": final}


def nn_call(nn_func: dict, dependencies: tuple, training = True):
    """ Call a feedforward neural network

    Examples:
        nn_call(self.lyte_indirect, lyte_dependencies, training = training)

    Args:
        nn_func: The neural network to call.
        dependencies: The dependencies of the neural network.
        training: Flag for training or evaluation.
            True for training; False for evaluation.

    Returns:
        The output of the neural network.
    """
    centers = nn_func["initial"](
        tf.concat(dependencies, axis = 1), training = training,
    )

    for dd in nn_func["bulk"]:
        centers_prime = centers
        centers_prime = tf.nn.relu(centers_prime)
        for d in dd:
            centers_prime = d(centers_prime, training = training)
        centers = centers + centers_prime  # This is a skip connection

    return nn_func["final"](centers, training = training)


def add_v_dep(
    voltage_independent: tf.Tensor, params: dict, dim = 1,
) -> tf.Tensor:
    """ Add voltage dependence: [cyc] -> [cyc, vol]

    Args:
        voltage_independent: Some voltage-independent quantity.
        params: Contains all parameters.
        dim: Dimension.

    Returns:
        The previously voltage-independent quantity with an "extra" voltage
            dimension, of shape
            `[params[Key.COUNT_BATCH] * params[Key.COUNT_V], dim]`
    """

    return tf.reshape(
        tf.tile(
            tf.expand_dims(voltage_independent, axis = 1),
            [1, params[Key.COUNT_V], 1],
        ),
        [params[Key.COUNT_BATCH] * params[Key.COUNT_V], dim]
    )


def add_current_dep(
    current_independent: tf.Tensor, params: dict, dim = 1,
) -> tf.Tensor:
    """ Add current dependence: [vol] -> [cyc, vol]

    Args:
        current_independent: Some current-independent quantity.
        params: Contains all parameters.
        dim: Dimension.

    Returns:
        The previously current-independent quantity with an "extra" current
            dimension, of shape
            `[params[Key.COUNT_BATCH] * params[Key.COUNT_I], dim]`
    """
    return tf.reshape(
        tf.tile(
            tf.expand_dims(current_independent, axis = 1),
            [1, params[Key.COUNT_I], 1],
        ),
        [params[Key.COUNT_BATCH] * params[Key.COUNT_I], dim]
    )


def create_derivatives(
    nn, params: dict, der_params: dict, internal_loss = False
):
    """
    Given a feedforward neural network `nn`,
        compute its value and its first derivative.

    TODO(harvey, confusion): Is "forall" explained somewhere? Maybe an entry
        in a wiki would be helpful.
    Derivatives will only be taken inside forall statements.
    If auxiliary variables must be given, create a lambda.

    Examples:
        ```python
        # state of charge and its first derivative
        q, q_der = create_derivatives(
            self.q_for_derivative,
            params = {
                Key.CYC: sampled_cycles,
                Key.STRESS: sampled_encoded_stress,
                Key.V: sampled_vs,
                Key.CELL_FEAT: sampled_features_cell,
                Key.I: sampled_constant_current
            },
            der_params = {Key.V: 3, Key.CELL_FEAT: 2, Key.I: 3, Key.CYC: 3}
        )
        ```

    Args:
        nn: A DegradationModel `for_derivative` method;
            it specifies the quantity to compute and derive.
        params: Contains parameters for computing the given quantity.
        der_params: Contains parameters for computing the first derivative of
            the given quantity.
        internal_loss: TODO(harvey, confusion)

    Returns:
        The evaluated quantity and it first derivative. If the `internal_loss`
            flag is on, then also the loss.
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
                        source = params[k], target = res,
                    )[:, 0, :]

            del tape_d1

        for k in der_params.keys():
            if der_params[k] >= 2:
                derivatives["d2_" + k] = tape_d2.batch_jacobian(
                    source = params[k], target = derivatives["d_" + k],
                )
                if k not in [Key.CELL_FEAT, Key.STRESS]:
                    derivatives["d2_" + k] = derivatives["d2_" + k][:, 0, :]

        del tape_d2

    for k in der_params.keys():
        if der_params[k] >= 3:
            derivatives["d3_" + k] = tape_d3.batch_jacobian(
                source = params[k], target = derivatives["d2_" + k]
            )
            if k not in [Key.CELL_FEAT, Key.STRESS]:
                derivatives["d3_" + k] = derivatives["d3_" + k][:, 0, :]

    del tape_d3
    if internal_loss:
        return res, derivatives, loss
    else:
        return res, derivatives


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
        self, depth, width, n_sample, options,
        cell_dict, dry_cell_dict, cell_latent_flags,
        cell_to_dry_cell, dry_cell_to_meta,
        n_channels = 16, min_latent = 0.1,
    ):
        """
        Args:
            options: Used to access the incentive coefficients.
        """
        super(DegradationModel, self).__init__()

        # minimum latent
        self.min_latent = min_latent

        # TODO(harvey): decide the style of "number of x" for the whole project
        # number of samples
        self.n_sample = n_sample
        # incentive coefficients
        self.options = options

        # TODO(harvey): decide the style of "number of x" for the whole project
        # number of features
        self.num_feats = width

        # feedforward neural network for capacity
        self.nn_q = feedforward_nn_parameters(depth, width, finalize = True)

        """ Primitive Dictionary Layer variables """
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
            shape = (self.cell_direct.num_keys, 4), dtype = np.int32,
        )

        for cell_id in self.cell_direct.id_dict.keys():
            if cell_id in cell_to_dry_cell.keys():
                cell_pointers[
                    self.cell_direct.id_dict[cell_id], 3,
                ] = dry_cell_dict[cell_to_dry_cell[cell_id]]

        self.cell_pointers = tf.constant(cell_pointers)
        self.cell_indirect = feedforward_nn_parameters(
            depth, width, last = self.num_feats,
        )

        self.stress_to_encoded_layer = StressToEncodedLayer(
            n_channels = n_channels
        )

        self.width = width
        self.n_channels = n_channels

    def call(self, params: dict, training = False) -> dict:
        """
        Call function for the Model during training or evaluation.

        Examples: TODO(harvey)

        Args:
            params: Contains -
                Cycle,
                Constant current,
                The end current of the previous step,
                The end voltage of the previous step,
                The end voltage of the current step,
                Indices,
                Voltage,
                Current
                S.V.I.T. grid,
                Count
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            `{ Key.Pred.I_CC, Key.Pred.I_CV }`. During training, the
                dictionary also includes `{ Key.Loss.Q, Key.Loss.CELL }`.
        """

        cycle = params[Key.CYC]  # matrix; dim: [batch, 1]
        constant_current = params[Key.I_CC]  # matrix; dim: [batch, 1]
        end_current_prev = params[Key.I_PREV_END]  # matrix; dim: [batch, 1]
        end_voltage_prev = params[Key.V_PREV_END]  # matrix; dim: [batch, 1]
        end_voltage = params[Key.V_END]  # matrix; dim: [batch, 1]
        indices = params[Key.INDICES]  # batch of index; dim: [batch]
        voltage_tensor = params[Key.V_TENSOR]  # dim: [batch, voltages]
        current_tensor = params[Key.I_TENSOR]  # dim: [batch, voltages]
        svit_grid = params[Key.SVIT_GRID]
        count_matrix = params[Key.COUNT_MATRIX]

        feats_cell, _, _ = self.cell_from_indices(
            indices = indices, training = training, sample = False,
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
            Key.I_CC: constant_current,
            Key.I_PREV_END: end_current_prev,
            Key.V_PREV_END: end_voltage_prev,
            Key.CELL_FEAT: feats_cell,
            Key.V_END: end_voltage,

            Key.SVIT_GRID: svit_grid,
            Key.COUNT_MATRIX: count_matrix,
        }
        cc_capacity = self.cc_capacity(params, training = training)
        pred_cc_capacity = tf.reshape(cc_capacity, [-1, voltage_count])

        cv_capacity = self.cv_capacity(params, training = training)
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
                der_params = {Key.V: 3, Key.CELL_FEAT: 2, Key.I: 3, Key.CYC: 3}
            )

            q_loss = calculate_q_loss(
                q, q_der, options = self.options
            )

            _, cell_loss, _ = self.cell_from_indices(
                indices = tf.range(self.cell_direct.num_keys, dtype = tf.int32),
                training = True,
                sample = False,
                compute_derivatives = True,
            )

            returns[Key.Loss.Q] = q_loss
            returns[Key.Loss.CELL] = cell_loss

        return returns

    def cell_from_indices(
        self, indices,
        training = True, sample = False, compute_derivatives = False,
    ):
        """ Cell from indices
        TODO(harvey, confusion): Need detailed explanation for what this
            function does.
            What are `indices`? What do the flags do?
        """
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

        dry_cell_indices = fetched_pointers_cell[:, 3]

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

        derivatives = {}

        if compute_derivatives:

            with tf.GradientTape(persistent = True) as tape_d1:
                tape_d1.watch(feats_dry_cell)

                cell_dependencies = (feats_dry_cell,)

                feats_cell_indirect = nn_call(
                    self.cell_indirect, cell_dependencies, training = training,
                )

            derivatives["d_features_dry_cell"] = tape_d1.batch_jacobian(
                source = feats_dry_cell, target = feats_cell_indirect,
            )

            del tape_d1
        else:
            cell_dependencies = (feats_dry_cell,)

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

        else:
            loss_output_cell = None

        if sample:
            eps = tf.random.normal(
                shape = [feats_cell.shape[0], self.num_feats]
            )
            feats_cell += self.cell_direct.sample_epsilon * eps

        if training:

            loss_input_cell_indirect = (
                (1. - fetched_latent_cell) * loss_dry_cell
            )

            if compute_derivatives:
                l_dry_cell = incentive_magnitude(
                    derivatives["d_features_dry_cell"], Target.Small,
                    Level.Proportional,
                )
                mult = (1. - tf.reshape(fetched_latent_cell, [-1, 1]))
                loss_derivative_cell_indirect = (
                    mult * tf.reduce_mean(l_dry_cell, axis = 2)
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

    def sample(self, svit_grid, batch_count, count_matrix, n_sample = 4 * 32):
        """
        Sample from all possible values of different variables.

        Args: TODO(harvey)
            svit_grid: multi-grid of (S, V, I, T).
            batch_count:
            count_matrix:
            n_sample:

        Returns:
            Sample values - voltages, capacities, cycles,
                constant current, cell features, latent, svit_grid,
                count_matrix, encoded_stress.
        """

        # NOTE(sam): this is an example of a forall.
        # (for all voltages, and all cell features)
        sampled_vs = tf.random.uniform(
            minval = 2.5, maxval = 5., shape = [n_sample, 1],
        )
        sampled_qs = tf.random.uniform(
            minval = -.25, maxval = 1.25, shape = [n_sample, 1],
        )
        sampled_cycles = tf.random.uniform(
            minval = -.1, maxval = 5., shape = [n_sample, 1],
        )
        sampled_constant_current = tf.random.uniform(
            minval = tf.math.log(0.001), maxval = tf.math.log(5.),
            shape = [n_sample, 1],
        )
        sampled_constant_current = tf.exp(sampled_constant_current)
        sampled_constant_current_sign = tf.random.uniform(
            minval = 0, maxval = 1, shape = [n_sample, 1], dtype = tf.int32,
        )
        sampled_constant_current_sign = tf.cast(
            sampled_constant_current_sign, dtype = tf.float32,
        )
        sampled_constant_current_sign = (
            1. * sampled_constant_current_sign
            - (1. - sampled_constant_current_sign)
        )

        sampled_constant_current = (
            sampled_constant_current_sign * sampled_constant_current
        )

        sampled_feats_cell, _, sampled_latent = self.cell_from_indices(
            indices = tf.random.uniform(
                maxval = self.cell_direct.num_keys,
                shape = [n_sample], dtype = tf.int32,
            ),
            training = False,
            sample = True,
        )
        sampled_feats_cell = tf.stop_gradient(sampled_feats_cell)

        sampled_svit_grid = tf.gather(
            svit_grid,
            indices = tf.random.uniform(
                minval = 0, maxval = batch_count,
                shape = [n_sample], dtype = tf.int32,
            ),
            axis = 0,
        )
        sampled_count_matrix = tf.gather(
            count_matrix,
            indices = tf.random.uniform(
                minval = 0, maxval = batch_count,
                shape = [n_sample], dtype = tf.int32,
            ),
            axis = 0,
        )

        sampled_encoded_stress = self.stress_to_encoded_direct(
            svit_grid = sampled_svit_grid,
            count_matrix = sampled_count_matrix,
        )

        return (
            sampled_vs,
            sampled_qs,
            sampled_cycles,
            sampled_constant_current,
            sampled_feats_cell,
            sampled_latent,
            sampled_svit_grid,
            sampled_count_matrix,
            sampled_encoded_stress,
        )

    def cc_capacity(self, params: dict, training = True):
        """
        Compute constant-current capacity during training or evaluation.

        Args:
            params: Contains the parameters of constant-current capacity.
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            Computed constant-current capacity.
        """

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params[Key.SVIT_GRID],
            count_matrix = params[Key.COUNT_MATRIX],
        )

        q_0 = self.q_direct(
            encoded_stress = encoded_stress,
            cycle = params[Key.CYC],
            v = params[Key.V_PREV_END],
            feats_cell = params[Key.CELL_FEAT],
            current = params[Key.I_PREV_END],
            training = training,
        )

        q_1 = self.q_direct(
            encoded_stress = add_v_dep(
                encoded_stress, params, encoded_stress.shape[1],
            ),
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
        """
        Compute constant-voltage capacity during training or evaluation.

        Args:
            params: Parameters for computing constant-voltage (cv) capacity.
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            Computed constant-voltage capacity.
        """

        encoded_stress = self.stress_to_encoded_direct(
            svit_grid = params[Key.SVIT_GRID],
            count_matrix = params[Key.COUNT_MATRIX],
        )

        q_0 = self.q_direct(
            encoded_stress = encoded_stress,
            cycle = params[Key.CYC],
            v = params[Key.V_PREV_END],
            feats_cell = params[Key.CELL_FEAT],
            current = params[Key.I_PREV_END],
            training = training,
        )

        # NOTE (sam): if there truly is no dependency on current for scale,
        # then we can restructure the code below.

        q_1 = self.q_direct(
            encoded_stress = add_current_dep(
                encoded_stress, params, encoded_stress.shape[1],
            ),
            cycle = add_current_dep(params[Key.CYC], params),
            v = add_current_dep(params[Key.V_END], params),
            feats_cell = add_current_dep(
                params[Key.CELL_FEAT], params, params[Key.CELL_FEAT].shape[1],
            ),
            current = params[Key.I_CV],
            training = training,
        )

        return q_1 - add_current_dep(q_0, params)

    def stress_to_encoded_direct(
        self, svit_grid, count_matrix, training = True,
    ):
        """
        Compute stress directly
            (receiving arguments directly without using `params`)

        Args: TODO(harvey)
            svit_grid: Multi-grid of (S, V, I, T).
            count_matrix:
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            (svit_grid, count_matrix) and the training flag.
        """
        return self.stress_to_encoded_layer(
            (svit_grid, count_matrix), training = training,
        )

    def q_direct(
        self, encoded_stress, cycle, v, feats_cell, current, training = True,
    ):
        """
        Compute state of charge directly
            (receiving arguments directly without using `params`)

        Args: TODO(harvey)
            encoded_stress
            cycle: Cycle, often Key.CYC.
            v: Voltage
            feats_cell: Cell features.
            current: Current.
            training: Flag for training or evaluation.
                True for training; False for evaluation.

        Returns:
            Computed state of charge.
        """
        dependencies = (
            encoded_stress,
            cycle,
            v,
            feats_cell,
            current,
        )
        return tf.nn.elu(nn_call(self.nn_q, dependencies, training = training))

    def q_for_derivative(self, params: dict, training = True):
        """
        Wrapper function calling `q_direct`, to be passed in to
            `create_derivatives` to compute state of charge
            and its first derivative.

        Examples:
            ```python
            # state of charge and its first derivative
            q, q_der = create_derivatives(
                self.q_for_derivative,
                params = {
                    Key.CYC: sampled_cycles,
                    Key.STRESS: sampled_encoded_stress,
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

        return self.q_direct(
            encoded_stress = params[Key.STRESS],
            cycle = params[Key.CYC],
            feats_cell = params[Key.CELL_FEAT],
            v = params[Key.V],
            current = params[Key.I],
            training = training,
        )

    @tf.function
    def test_all_voltages(
        self,
        cycle, constant_current, end_current_prev, end_voltage_prev,
        end_voltage,
        cell_id_index,
        voltages, currents, svit_grid, count_matrix,
    ):

        expanded_cycle = tf.expand_dims(cycle, axis = 1)
        expanded_constant_current = tf.tile(
            tf.reshape(constant_current, [1, 1]),
            [cycle.shape[0], 1],
        )
        expanded_end_current_prev = tf.tile(
            tf.reshape(end_current_prev, [1, 1]),
            [cycle.shape[0], 1],
        )
        expanded_end_voltage_prev = tf.tile(
            tf.reshape(end_voltage_prev, [1, 1]),
            [cycle.shape[0], 1],
        )
        expanded_end_voltage = tf.tile(
            tf.reshape(end_voltage, [1, 1]),
            [cycle.shape[0], 1],
        )

        indices = tf.tile(
            tf.expand_dims(cell_id_index, axis = 0),
            [cycle.shape[0]]
        )

        expanded_svit_grid = tf.tile(
            tf.expand_dims(svit_grid, axis = 0),
            [cycle.shape[0], 1, 1, 1, 1, 1],
        )
        expanded_count_matrix = tf.tile(
            tf.expand_dims(count_matrix, axis = 0),
            [cycle.shape[0], 1, 1, 1, 1, 1],
        )

        return self.call(
            {
                Key.CYC: expanded_cycle,
                Key.I_CC: expanded_constant_current,
                Key.I_PREV_END: expanded_end_current_prev,
                Key.V_PREV_END: expanded_end_voltage_prev,
                Key.V_END: expanded_end_voltage,
                Key.INDICES: indices,
                Key.V_TENSOR: tf.tile(
                    tf.reshape(voltages, [1, -1]),
                    [cycle.shape[0], 1],
                ),
                Key.I_TENSOR: tf.tile(
                    tf.reshape(currents, shape = [1, -1]),
                    [cycle.shape[0], 1]
                ),
                Key.SVIT_GRID: expanded_svit_grid,
                Key.COUNT_MATRIX: expanded_count_matrix,
            },
            training = False,
        )

    @tf.function
    def test_single_voltage(
        self,
        cycle, v, constant_current, end_current_prev, end_voltage_prev,
        end_voltage,
        currents, cell_id_index, svit_grid, count_matrix
    ):

        expanded_cycle = tf.expand_dims(cycle, axis = 1)
        expanded_constant_current = tf.tile(
            tf.reshape(constant_current, [1, 1]),
            [cycle.shape[0], 1],
        )
        expanded_end_current_prev = tf.tile(
            tf.reshape(end_current_prev, [1, 1]),
            [cycle.shape[0], 1],
        )
        expanded_end_voltage_prev = tf.tile(
            tf.reshape(end_voltage_prev, [1, 1]),
            [cycle.shape[0], 1],
        )
        expanded_end_voltage = tf.tile(
            tf.reshape(end_voltage, [1, 1]),
            [cycle.shape[0], 1],
        )

        indices = tf.tile(
            tf.expand_dims(cell_id_index, axis = 0),
            [cycle.shape[0]],
        )

        expanded_svit_grid = tf.tile(
            tf.expand_dims(svit_grid, axis = 0),
            [cycle.shape[0], 1, 1, 1, 1, 1],
        )
        expanded_count_matrix = tf.tile(
            tf.expand_dims(count_matrix, axis = 0),
            [cycle.shape[0], 1, 1, 1, 1, 1],
        )

        return self.call(
            {
                Key.CYC: expanded_cycle,
                Key.I_CC: expanded_constant_current,
                Key.I_PREV_END: expanded_end_current_prev,
                Key.V_PREV_END: expanded_end_voltage_prev,
                Key.V_END: expanded_end_voltage,
                Key.INDICES: indices,
                Key.V_TENSOR: tf.tile(tf.reshape(v, [1, 1]),
                                      [cycle.shape[0], 1]),
                Key.I_TENSOR: tf.tile(
                    tf.reshape(currents, shape = [1, -1]),
                    [cycle.shape[0], 1],
                ),
                Key.SVIT_GRID: expanded_svit_grid,
                Key.COUNT_MATRIX: expanded_count_matrix,
            },
            training = False
        )
