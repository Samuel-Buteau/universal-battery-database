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
    :param x: The first object

    :param symbol: the relationship we want (either Inequality.LessThan or
    Inequality.GreaterThan or Inequality.Equal)

        Inequality.LessThan (i.e. x < y) means that x should be less than B,

        Inequality.GreaterThan (i.e. x > B) means that x should be greater
        than B.

        Inequality.Equals (i.e. x = B) means that x should be equal to B.

    :param y: The second object

    :param level:  determines the relationship between the incentive strength
    and the values of x and B.
    (either Level.Strong or Level.Proportional)

        Level.Strong means that we take the L1 norm, so the gradient trying
        to satisfy "x symbol B" will be constant no matter how far from "x
        symbol B" we
        are.

        Level.Proportional means that we take the L2 norm, so the gradient
        trying to satisfy "x symbol B" will be proportional to how far from
        "x symbol B" we are.

    :return: A loss which will give the model an incentive to satisfy "x
    symbol B", with level.
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
    :param x: The object

    :param target: The direction we want (either Target.Small or Target.Big)

        Target.Small means that the norm of x should be as small as possible

        Target.Big means that the norm of x should be as big as
        possible,

    :param level: Determines the relationship between the incentive strength
    and the value of x. (either Level.Strong or Level.Proportional)

        Level.Strong means that we take the L1 norm, so the gradient trying
        to push the absolute value of x to target
        will be constant.

        Level.Proportional means that we take the L2 norm,
        so the gradient trying to push the absolute value of x to target
        will be proportional to the absolute value of x.

    :return: A loss which will give the model an incentive to push the
    absolute value of x to target.
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
    :param xs: A list of tuples. Each tuple contains a coefficient and a
    tensor of losses corresponding to incentives.

    :return: A combined loss (single number) which will incentivize all the
    individual incentive tensors with weights given by the coefficients.
    """

    return sum([a[0] * tf.reduce_mean(a[1]) for a in xs])
