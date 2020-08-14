import tensorflow as tf

from tensorflow.keras.layers import Dense

from Key import Key

main_activation = tf.keras.activations.elu


def feedforward_nn_parameters(
    depth: int, width: int, last = None, finalize = False, bottleneck = None,
):
    """ Create a new feedforward neural network

    Args:
        depth: The depth the feedforward neural network
        width: The width of the feedforward neural network
        last:
        finalize:
        bottleneck:

    Returns:
        { "initial", "bulk", and "final" }, each key corresponds to a component
            of the neural network.
    """
    if last is None:
        last = 1
    if bottleneck is None:
        bottleneck = width

    if bottleneck != width:
        projector = Dense(
            bottleneck,
            activation = None,
            use_bias = True,
            bias_initializer = "zeros",
        )

    initial = Dense(
        width,
        activation = main_activation,
        use_bias = True,
        bias_initializer = "zeros"
    )

    widths = [width for _ in range(depth)]
    widths[-1] = bottleneck

    bulk = [
        [
            Dense(
                widths[i],
                activation = activation,
                use_bias = True,
                bias_initializer = "zeros",
            ) for activation in ["elu", None]
        ]
        for i in range(depth)
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
    ret = {"initial": initial, "bulk": bulk, "final": final}
    if bottleneck != width:
        ret['projector'] = projector
    return ret


def nn_call(
    nn_func: dict, dependencies: tuple, training = True, get_bottleneck = False,
):
    """ Call a feedforward neural network

    Examples:
        nn_call(self.lyte_indirect, lyte_dependencies, training = training)

    Args:
        nn_func: The neural network to call.
        dependencies: The dependencies of the neural network.
        training: Flag for training or evaluation.
            True for training; False for evaluation.
        get_bottleneck:

    Returns:
        The output of the neural network.
    """
    centers = nn_func["initial"](
        tf.concat(dependencies, axis = 1), training = training,
    )

    for dd in nn_func["bulk"][:-1]:
        centers_prime = centers
        centers_prime = tf.nn.elu(centers_prime)
        for d in dd:
            centers_prime = d(centers_prime, training = training)
        centers = centers + centers_prime  # This is a skip connection

    dd = nn_func["bulk"][-1]
    centers_prime = centers
    centers_prime = tf.nn.elu(centers_prime)
    for d in dd:
        centers_prime = d(centers_prime, training = training)
    if 'projector' in nn_func.keys():
        centers = nn_func['projector'](
            centers, training = training,
        ) + centers_prime  # This is a skip connection
    else:
        centers = centers + centers_prime

    if get_bottleneck:
        return nn_func["final"](centers, training = training), centers
    else:
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

    Derivatives will only be taken inside forall statements.
    If auxiliary variables must be given, create a lambda.

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
            der_params = {Key.V: 3, Key.CELL_FEAT: 2, Key.I: 3, Key.CYC: 3}
        )
        ```

    Args:
        nn: A DegradationModel `for_derivative` method;
            it specifies the quantity to compute and derive.
        params: Contains parameters for computing the given quantity.
        der_params: Contains parameters for computing the first derivative of
            the given quantity.
        internal_loss:

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
