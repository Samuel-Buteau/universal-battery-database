from machine_learning.incentives import *
from Key import Key


def calculate_q_loss(q, q_der, options):
    """ Compute the loss functions for capacity.

    Args:
        q: Computed capacity.
        q_der: Computed first derivative of capacity.
        options: Used to access the incentive coefficients.
    """
    return incentive_combine([

        (
            options[Key.COEFF_Q_GEQ],
            incentive_inequality(q, Inequality.GreaterThan, 0, Level.Strong),
        ), (
            options[Key.COEFF_Q_LEQ],
            incentive_inequality(q, Inequality.LessThan, 1, Level.Strong),
        ), (
            options[Key.COEFF_Q_V_MONO],
            incentive_inequality(
                q_der["d_v"], Inequality.GreaterThan, 0, Level.Strong,
            ),
        ), (
            options[Key.COEFF_Q_DER3_V],
            incentive_magnitude(
                q_der["d3_v"], Target.Small, Level.Proportional,
            ),
        ), (
            options[Key.COEFF_Q_DER3_I],
            incentive_magnitude(
                q_der["d3_current"], Target.Small, Level.Proportional,
            ),
        ), (
            options[Key.COEFF_Q_DER3_N],
            incentive_magnitude(
                q_der["d3_cycle"], Target.Small, Level.Proportional,
            ),
        ), (
            options[Key.COEFF_Q_DER_I],
            incentive_magnitude(
                q_der["d_current"], Target.Small, Level.Proportional,
            ),
        ), (
            options[Key.COEFF_Q_DER_N],
            incentive_magnitude(
                q_der["d_cycle"], Target.Small, Level.Proportional,
            ),
        ), (
            options["coeff_d_features_cell"],
            incentive_magnitude(
                q_der["d_features_cell"], Target.Small, Level.Proportional,
            ),
        ), (
            options["coeff_d2_features_cell"],
            incentive_magnitude(
                q_der["d2_features_cell"], Target.Small, Level.Strong,
            ),
        ),
    ])
