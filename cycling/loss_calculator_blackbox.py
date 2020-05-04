from cycling.incentives import *



def calculate_q_loss(q, q_der, incentive_coeffs):
    return incentive_combine([

        (
            incentive_coeffs["coeff_q_geq"],
            incentive_inequality(
                q, Inequality.GreaterThan, 0,
                Level.Strong
            )
        ),
        (
            incentive_coeffs["coeff_q_leq"],
            incentive_inequality(
                q, Inequality.LessThan, 1,
                Level.Strong
            )
        ),


        (
            incentive_coeffs["coeff_q_v_mono"],
            incentive_inequality(
                q_der["d_v"], Inequality.GreaterThan, 0,
                Level.Strong
            )
        ),
        (
            incentive_coeffs["coeff_q_d3_v"],
            incentive_magnitude(
                q_der["d3_v"],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs["coeff_q_d3_current"],
            incentive_magnitude(
                q_der["d3_current"],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs["coeff_q_d3_cycle"],
            incentive_magnitude(
                q_der["d3_cycle"],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs["coeff_q_d_current"],
            incentive_magnitude(
                q_der["d_current"],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs["coeff_q_d_cycle"],
            incentive_magnitude(
                q_der["d_cycle"],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs["coeff_d_features_cell"],
            incentive_magnitude(
                q_der["d_features_cell"],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs["coeff_d2_features_cell"],
            incentive_magnitude(
                q_der["d2_features_cell"],
                Target.Small,
                Level.Strong
            )
        ),

    ])

