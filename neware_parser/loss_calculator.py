from neware_parser.incentives import *


def calculate_projection_loss(
    sampled_latent, sampled_pos, sampled_neg,
    predicted_pos, predicted_neg,
    incentive_coeffs
):
    return incentive_combine([
        (
            incentive_coeffs['coeff_projection_pos'],
            (1. - sampled_latent) * incentive_inequality(
                sampled_pos, Inequality.Equals, predicted_pos,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_projection_neg'],
            (1. - sampled_latent) * incentive_inequality(
                sampled_neg, Inequality.Equals, predicted_neg,
                Level.Proportional
            )
        ),
    ])


def calculate_oob_loss(reciprocal_q, incentive_coeffs=None):
    return incentive_combine([
        (
            incentive_coeffs['coeff_oob_geq'],
            incentive_inequality(
                reciprocal_q, Inequality.GreaterThan, 0., Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_oob_leq'],
            incentive_inequality(
                reciprocal_q, Inequality.LessThan, 1., Level.Strong
            )
        ),
    ])


def calculate_reciprocal_loss(
    sampled_voltages, sampled_qs,
    v_plus, v_minus, v_plus_der, v_minus_der,
    reciprocal_v, reciprocal_q,
        incentive_coeffs
):
    return incentive_combine([
        (
            incentive_coeffs['coeff_reciprocal_v'],
            incentive_inequality(
                sampled_voltages, Inequality.Equals, reciprocal_v,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_q'],
            incentive_inequality(
                sampled_qs, Inequality.Equals, reciprocal_q, Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_v_small'],
            incentive_magnitude(
                v_minus, Target.Small, Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_v_small'],
            incentive_magnitude(
                v_plus, Target.Small, Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_v_geq'],
            incentive_inequality(
                v_minus, Inequality.GreaterThan, 0., Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_v_leq'],
            incentive_inequality(
                v_minus, Inequality.LessThan, 5., Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_v_mono'],
            incentive_inequality(
                v_minus_der['d_q'], Inequality.LessThan, 0., Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_d3_current'],
            incentive_magnitude(
                v_plus_der['d3_current'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_d3_current'],
            incentive_magnitude(
                v_minus_der['d3_current'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_d_current_minus'],
            incentive_magnitude(
                v_minus_der['d_current'],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs['coeff_reciprocal_d_current_plus'],
            incentive_magnitude(
                v_plus_der['d_current'],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs['coeff_reciprocal_v_geq'],
            incentive_inequality(
                v_plus, Inequality.GreaterThan, 0., Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_v_leq'],
            incentive_inequality(
                v_plus, Inequality.LessThan, 5., Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_reciprocal_v_mono'],
            incentive_inequality(
                v_plus_der['d_q'], Inequality.GreaterThan, 0., Level.Strong
            )
        ),
    ])


def calculate_q_loss(q, q_der, incentive_coeffs):
    return incentive_combine([
        (
            incentive_coeffs['coeff_q_small'],
            incentive_magnitude(
                q,
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_q_geq'],
            incentive_inequality(
                q, Inequality.GreaterThan, 0,
                Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_q_leq'],
            incentive_inequality(
                q, Inequality.LessThan, 1,
                Level.Strong
            )
        ),


        (
            incentive_coeffs['coeff_q_v_mono'],
            incentive_inequality(
                q_der['d_voltage'], Inequality.GreaterThan, 0.01,
                Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_q_d3_v'],
            incentive_magnitude(
                q_der['d3_voltage'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_d_features'],
            incentive_magnitude(
                q_der['d_features'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_d2_features'],
            incentive_magnitude(
                q_der['d2_features'],
                Target.Small,
                Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_q_d3_shift'],
            incentive_magnitude(
                q_der['d3_shift'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_q_d3_current'],
            incentive_magnitude(
                q_der['d3_current'],
                Target.Small,
                Level.Proportional
            )
        ),
    ])


def calculate_q_scale_loss(q_scale, q_scale_der, incentive_coeffs):
    return incentive_combine([
        (
            incentive_coeffs['coeff_q_scale_geq'],
            incentive_inequality(
                q_scale, Inequality.GreaterThan, 0.1,
                Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_q_scale_leq'],
            incentive_inequality(
                q_scale, Inequality.LessThan, 1.,
                Level.Strong
            )
        ),

        (
            incentive_coeffs['coeff_q_scale_eq'],
            incentive_inequality(
                q_scale, Inequality.Equals, 1.,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs['coeff_q_scale_mono'],
            incentive_inequality(
                q_scale_der['d_cycle'], Inequality.LessThan, 0,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_q_scale_d3_cycle'],
            incentive_magnitude(
                q_scale_der['d3_cycle'],
                Target.Small,
                Level.Proportional
            )
        ),


        (
            incentive_coeffs['coeff_d_features'],
            incentive_magnitude(
                q_scale_der['d_features'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_d2_features'],
            incentive_magnitude(
                q_scale_der['d2_features'],
                Target.Small,
                Level.Strong
            )
        ),
    ])


def calculate_shift_loss(shift, shift_der, incentive_coeffs):
    return incentive_combine([
        (
            incentive_coeffs['coeff_shift_geq'],
            incentive_inequality(
                shift, Inequality.GreaterThan, -0.5,
                Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_shift_leq'],
            incentive_inequality(
                shift, Inequality.LessThan, 0.5,
                Level.Strong
            )
        ),
        (
            incentive_coeffs['coeff_shift_small'],
            incentive_magnitude(
                shift,
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_shift_d3_cycle'],
            incentive_magnitude(
                shift_der['d3_cycle'],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs['coeff_d_features'],
            incentive_magnitude(
                shift_der['d_features'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_d2_features'],
            incentive_magnitude(
                shift_der['d2_features'],
                Target.Small,
                Level.Strong
            )
        )
    ])


def calculate_r_loss(r, r_der, incentive_coeffs):
    return incentive_combine([
        (
            incentive_coeffs['coeff_r_geq'],
            incentive_inequality(
                r, Inequality.GreaterThan, 0.01,
                Level.Strong
            )
        ),

        (
            incentive_coeffs['coeff_r_big'],
            incentive_magnitude(
                r, Target.Big,
                Level.Proportional
            )
        ),

        (
            incentive_coeffs['coeff_r_d3_cycle'],
            incentive_magnitude(
                r_der['d3_cycle'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_d_features'],
            incentive_magnitude(
                r_der['d_features'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            incentive_coeffs['coeff_d2_features'],
            incentive_magnitude(
                r_der['d2_features'],
                Target.Small,
                Level.Strong
            )
        )
    ])
