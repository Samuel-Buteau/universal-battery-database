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
