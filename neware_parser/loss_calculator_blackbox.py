from neware_parser.Key import Key
from neware_parser.incentives import *


def calculate_q_loss(q, q_der, incentive_coeffs):
    return incentive_combine([

        (
            incentive_coeffs[Key.COEFF_Q_GEQ],
            incentive_inequality(q, Inequality.GreaterThan, 0, Level.Strong)
        ), (
            incentive_coeffs[Key.COEFF_Q_LEQ],
            incentive_inequality(q, Inequality.LessThan, 1, Level.Strong)
        ), (
            incentive_coeffs[Key.COEFF_Q_V_MONO],
            incentive_inequality(
                q_der["d_v"], Inequality.GreaterThan, 0, Level.Strong,
            )
        ), (
            incentive_coeffs[Key.COEFF_Q_DER3_V],
            incentive_magnitude(q_der["d3_v"], Target.Small, Level.Proportional)
        ), (
            incentive_coeffs[Key.COEFF_Q_DER3_I],
            incentive_magnitude(
                q_der["d3_current"], Target.Small, Level.Proportional,
            )
        ), (
            incentive_coeffs[Key.COEFF_Q_DER3_N],
            incentive_magnitude(
                q_der["d3_cycle"], Target.Small, Level.Proportional,
            )
        ), (
            incentive_coeffs[Key.COEFF_Q_DER_I],
            incentive_magnitude(
                q_der["d_current"], Target.Small, Level.Proportional,
            )
        ), (
            incentive_coeffs[Key.COEFF_Q_DER_N],
            incentive_magnitude(
                q_der["d_cycle"], Target.Small, Level.Proportional,
            )
        ), (
            incentive_coeffs[Key.COEFF_FEAT_CELL_DER],
            incentive_magnitude(
                q_der["d_features_cell"], Target.Small, Level.Proportional,
            )
        ), (
            incentive_coeffs[Key.COEFF_FEAT_CELL_DER2],
            incentive_magnitude(
                q_der["d2_features_cell"], Target.Small, Level.Strong,
            )
        ),
    ])
