from neware_parser.incentives import *


def calculate_projection_loss(
    sampled_latent, sampled_pos, sampled_neg,
    predicted_pos, predicted_neg
):
    return incentive_combine([
        (
            10.,
            (1. - sampled_latent) * incentive_inequality(
                sampled_pos, Inequality.Equals, predicted_pos,
                Level.Proportional
            )
        ),
        (
            10.,
            (1. - sampled_latent) * incentive_inequality(
                sampled_neg, Inequality.Equals, predicted_neg,
                Level.Proportional
            )
        ),
    ])


def calculate_oob_loss(reciprocal_q):
    return incentive_combine([
        (
            1.,
            incentive_inequality(
                reciprocal_q, Inequality.GreaterThan, 0., Level.Strong
            )
        ),
        (
            1.,
            incentive_inequality(
                reciprocal_q, Inequality.LessThan, 1., Level.Strong
            )
        ),
    ])


def calculate_reciprocal_loss(
    sampled_voltages, sampled_qs,
    v_plus, v_minus, v_plus_der, v_minus_der,
    reciprocal_v, reciprocal_q,
):
    return incentive_combine([
        (
            2.,
            incentive_inequality(
                sampled_voltages, Inequality.Equals, reciprocal_v,
                Level.Proportional
            )
        ),
        (
            2.,
            incentive_inequality(
                sampled_qs, Inequality.Equals, reciprocal_q, Level.Proportional
            )
        ),
        (
            .01,
            incentive_magnitude(
                v_minus, Target.Small, Level.Proportional
            )
        ),
        (
            .01,
            incentive_magnitude(
                v_plus, Target.Small, Level.Proportional
            )
        ),
        (
            10.,
            incentive_inequality(
                v_minus, Inequality.GreaterThan, 0., Level.Strong
            )
        ),
        (
            10.,
            incentive_inequality(
                v_minus, Inequality.LessThan, 5., Level.Strong
            )
        ),
        (
            1.,
            incentive_inequality(
                v_minus_der['d_q'], Inequality.LessThan, 0., Level.Strong
            )
        ),

        (
            10.,
            incentive_inequality(
                v_plus, Inequality.GreaterThan, 0., Level.Strong
            )
        ),
        (
            10.,
            incentive_inequality(
                v_plus, Inequality.LessThan, 5., Level.Strong
            )
        ),
        (
            1.,
            incentive_inequality(
                v_plus_der['d_q'], Inequality.GreaterThan, 0., Level.Strong
            )
        ),

    ])


def calculate_q_loss(q, q_der):
    return incentive_combine([
        (
            1.,
            incentive_magnitude(
                q,
                Target.Small,
                Level.Proportional
            )
        ),
        (
            100.,
            incentive_inequality(
                q, Inequality.GreaterThan, 0,
                Level.Strong
            )
        ),
        (
            10000.,
            incentive_inequality(
                q_der['d_voltage'], Inequality.GreaterThan, 0.01,
                Level.Strong
            )
        ),
        (
            100.,
            incentive_magnitude(
                q_der['d3_voltage'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            .01,
            incentive_magnitude(
                q_der['d_features'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            .01,
            incentive_magnitude(
                q_der['d2_features'],
                Target.Small,
                Level.Strong
            )
        ),
        (
            100.,
            incentive_magnitude(
                q_der['d_shift'],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            100.,
            incentive_magnitude(
                q_der['d2_shift'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            100.,
            incentive_magnitude(
                q_der['d3_shift'],
                Target.Small,
                Level.Proportional
            )
        ),
    ])


def calculate_q_scale_loss(q_scale, q_scale_der):
    return incentive_combine([
        (
            10000.,
            incentive_inequality(
                q_scale, Inequality.GreaterThan, 0.1,
                Level.Strong
            )
        ),
        (
            20000.,
            incentive_inequality(
                q_scale, Inequality.LessThan, 1.2,
                Level.Strong
            )
        ),

        (
            10.,
            incentive_inequality(
                q_scale, Inequality.Equals, 1,
                Level.Proportional
            )
        ),
        (
            1.,
            incentive_inequality(
                q_scale_der['d_cycle'], Inequality.LessThan, 0,
                Level.Proportional
            )
        ),
        (
            .1,
            incentive_inequality(
                q_scale_der['d2_cycle'], Inequality.LessThan, 0,
                Level.Proportional
            )
        ),
        (
            100.,
            incentive_magnitude(
                q_scale_der['d_cycle'],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            100.,
            incentive_magnitude(
                q_scale_der['d2_cycle'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            100.,
            incentive_magnitude(
                q_scale_der['d3_cycle'],
                Target.Small,
                Level.Proportional
            )
        ),

        (
            1.,
            incentive_magnitude(
                q_scale_der['d_features'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            1.,
            incentive_magnitude(
                q_scale_der['d2_features'],
                Target.Small,
                Level.Strong
            )
        ),
    ])


def calculate_shift_loss(shift, shift_der):
    return incentive_combine([
        (
            10000.,
            incentive_inequality(
                shift, Inequality.GreaterThan, -1,
                Level.Strong
            )
        ),
        (
            10000.,
            incentive_inequality(
                shift, Inequality.LessThan, 1,
                Level.Strong
            )
        ),
        (
            100.,
            incentive_magnitude(
                shift,
                Target.Small,
                Level.Proportional
            )
        ),

        (
            100.,
            incentive_magnitude(
                shift_der['d_current'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            100.,
            incentive_magnitude(
                shift_der['d2_current'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            100.,
            incentive_magnitude(
                shift_der['d3_current'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            1.,
            incentive_magnitude(
                shift_der['d_features'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            1.,
            incentive_magnitude(
                shift_der['d2_features'],
                Target.Small,
                Level.Strong
            )
        )
    ])


def calculate_r_loss(r, r_der):
    return incentive_combine([
        (
            10000.,
            incentive_inequality(
                r,
                Inequality.GreaterThan,
                0.01,
                Level.Strong
            )
        ),
        (
            10.,
            incentive_magnitude(
                r_der['d2_cycle'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            100.,
            incentive_magnitude(
                r_der['d3_cycle'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            1.,
            incentive_magnitude(
                r_der['d_features'],
                Target.Small,
                Level.Proportional
            )
        ),
        (
            1.,
            incentive_magnitude(
                r_der['d2_features'],
                Target.Small,
                Level.Strong
            )
        )
    ])
