from Key import Key

from machine_learning.incentives import (
    Inequality, Level, Target,
    incentive_inequality, incentive_magnitude, incentive_combine,
)


def calculate_q_loss(q, q_der, options):
    """ Compute the loss functions for capacity.

    Args:
        q: Computed capacity.
        q_der: Computed first derivative of capacity.
        options: Used to access the incentive coefficients.
    """
    return incentive_combine([
        (
            options[Key.Coeff.Q_CENTERED],
            incentive_magnitude(
                q, Target.Small, Level.Proportional,
            ),
        ),
        (
            options[Key.Coeff.Q_V_MONO],
            incentive_inequality(
                q_der[Key.D_V], Inequality.GreaterThan, 0, Level.Strong,
            ),
        ), (
            options[Key.Coeff.Q_DER3_V],
            incentive_magnitude(
                q_der[Key.D3_V], Target.Small, Level.Proportional,
            ),
        ), (
            options[Key.Coeff.Q_DER3_I],
            incentive_magnitude(
                q_der[Key.D3_I], Target.Small, Level.Proportional,
            ),
        ), (
            options[Key.Coeff.Q_DER3_N],
            incentive_magnitude(
                q_der[Key.D3_CYC], Target.Small, Level.Proportional,
            ),
        ), (
            options[Key.Coeff.Q_DER_I],
            incentive_magnitude(
                q_der[Key.D_I], Target.Small, Level.Proportional,
            ),
        ), (
            options[Key.Coeff.Q_DER_N],
            incentive_magnitude(
                q_der[Key.D_CYC], Target.Small, Level.Proportional,
            ),
        ),
    ])
