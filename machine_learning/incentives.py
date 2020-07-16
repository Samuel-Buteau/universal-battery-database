from enum import Enum

import tensorflow as tf


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


def incentive_inequality(x, symbol, y, level):
    """
    Args:
        x: The first object

        symbol: The relationship we want. This is one of the following:

            Inequality.LessThan (x < y): x should be less than y,

            Inequality.GreaterThan (x > y): x should be greater than y.

            Inequality.Equals (x = y): x should be equal to y.

        y: The second object

        level: Determines the relationship between the incentive strength and
            values of x and y. This is one of the following:

            Level.Strong: we take the L1 norm, so the gradient trying to satisfy
                "x symbol y" will be constant no matter how far from
                "x symbol y" we are.

            Level.Proportional: we take the L2 norm, so the gradient trying to
                satisfy "x symbol y" will be proportional to how far from
                "x symbol y" we are.

    Returns:
        A loss which will give the model an incentive to satisfy "x symbol y",
            with level.
    """

    if symbol == Inequality.LessThan:
        intermediate = tf.nn.relu(x - y)
    elif symbol == Inequality.GreaterThan:
        intermediate = tf.nn.relu(y - x)
    elif symbol == Inequality.Equals:
        intermediate = tf.abs(x - y)
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


def incentive_magnitude(x, target, level):
    """
    Args:
        x: The object

        target: The direction we want. This is one of the following:

            Target.Small: the norm of x should be as small as possible

            Target.Big: the norm of x should be as big as possible.

        level: Determines the relationship between the incentive strength
        and the value of x. This is one of the following:

            Level.Strong: we take the L1 norm, so the gradient trying to push
                the absolute value of x to target will be constant.

            Level.Proportional: we take the L2 norm, so the gradient trying to
                push the absolute value of x to target will be proportional to
                the absolute value of x.

    Returns:
        A loss which will give the model an incentive to push the absolute value
            of x to target.
    """

    x_prime = tf.abs(x)

    if target == Target.Small:
        multiplier = 1.
    elif target == Target.Big:
        multiplier = -1.

    else:
        raise Exception("not yet implemented target {}".format(target))

    if level == Level.Strong:
        x_prime = x_prime
    elif level == Level.Proportional:
        x_prime = tf.square(x_prime)
    else:
        raise Exception("not yet implemented level {}".format(level))

    return multiplier * x_prime


def incentive_combine(xs):
    """
    Args:
        xs: A list of tuples. Each tuple contains a coefficient and a tensor of
            losses corresponding to incentives.

    Returns:
        A combined loss (single number) which will incentivize all the
            individual incentive tensors with weights given by the coefficients.
    """

    return sum([a[0] * tf.reduce_mean(a[1]) for a in xs])


def incentive_relative_equality(y, y_pred, min_val, max_val):
    return (y - y_pred) ** 2 / tf.clip_by_value(
        0.5 * tf.stop_gradient(tf.abs(y) + tf.abs(y_pred)),
        min_val, max_val,
    ) ** 2
