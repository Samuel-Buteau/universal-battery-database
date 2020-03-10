import numpy
from django.core.management.base import BaseCommand
from django.db.models import Max, Min

from neware_parser.models import *
from neware_parser.neware_processing_functions import *

'''
Shortened Variable Names:
    vol -   voltage
    cap -   capacity
    dchg -  discharge
    neigh - neighbourhood
    der -   derivative
    pred -  predicted
    meas -  measured
    eval -  evaluation
    eq -    equillibrium
'''


# === Begin: make my barcodes ==================================================

def make_my_barcodes(fit_args):
    my_barcodes = CyclingFile.objects.filter(
        database_file__deprecated = False,
        database_file__is_valid = True
    ).exclude(
        database_file__valid_metadata = None
    ).order_by(
        'database_file__valid_metadata__barcode'
    ).values_list(
        'database_file__valid_metadata__barcode',
        flat = True
    ).distinct()

    used_barcodes = []
    for b in my_barcodes:
        if (ChargeCycleGroup.objects.filter(barcode = b).exists()
            or DischargeCycleGroup.objects.filter(barcode = b).exists()
        ):
            used_barcodes.append(b)

    if len(fit_args['wanted_barcodes']) == 0:
        return used_barcodes
    else:
        return list(
            set(used_barcodes).intersection(
                set(fit_args['wanted_barcodes'])
            )
        )


# === End: make my barcodes ====================================================

# ==== Begin: initial processing ===============================================




def initial_processing(my_barcodes, fit_args):
    all_data = {}
    voltage_grid = make_voltage_grid(
        fit_args['voltage_grid_min_v'],
        fit_args['voltage_grid_max_v'],
        fit_args['voltage_grid_n_samples'],
        my_barcodes
    )

    voltage_grid_degradation = make_voltage_grid(
        fit_args['voltage_grid_min_v'],
        fit_args['voltage_grid_max_v'],
        int(fit_args['voltage_grid_n_samples']/4),
        my_barcodes
    )

    current_grid = make_current_grid(
        fit_args['current_grid_min_v'],
        fit_args['current_grid_max_v'],
        fit_args['current_grid_n_samples'],
        my_barcodes
    )

    temperature_grid = make_temperature_grid(
        fit_args['temperature_grid_min_v'],
        fit_args['temperature_grid_max_v'],
        fit_args['temperature_grid_n_samples'],
        my_barcodes
    )
    sign_grid = make_sign_grid()
    '''
    - cycles are grouped by their charge rates and discharge rates.
    - a cycle group contains many cycles
    - things are split up this way to sample each group equally
    - each barcode corresponds to a single cell
    '''
    for barcode in my_barcodes:
        '''
        - dictionary indexed by charging and discharging rate (i.e. cycle group)
        - contains structured arrays of
            - cycle_number
            - capacity_vector: a vector where each element is a
              capacity associated with a given voltage
              [(voltage_grid[i], capacity_vector[i])
              is a voltage-capacity pair]
            - vq_curve_mask: a vector where each element is a weight
              corresponding to a voltage-capacity pair
              [this allows us to express the fact that sometimes a given
              voltage was not measured, so the capacity is meaningless.
              (mask of 0)]
        '''

        files = get_files_for_barcode(barcode)

        all_mats = []
        for cyc in Cycle.objects.filter(cycling_file__in=files).order_by('cycle_number'):
            count_matrix = get_count_matrix(cyc, voltage_grid_degradation, current_grid, temperature_grid, sign_grid)
            true_cycle = cyc.get_offset_cycle()
            # for each cycle, call COUNT_MATRIX, and get (true_cyc, COUNT_MATRIX) list
            if count_matrix is None:
                continue
            all_mats.append((true_cycle, count_matrix))

        all_mats = numpy.array(all_mats, dtype = [('cycle_number', 'f4'),

                        ('count_matrix', 'f4', (len(sign_grid), len(voltage_grid_degradation), len(current_grid), len(temperature_grid))),
                                                  ])

        min_cycle = numpy.min(all_mats['cycle_number'])
        max_cycle = numpy.max(all_mats['cycle_number'])

        cycle_span = max_cycle - min_cycle


        delta_cycle = cycle_span / float(fit_args['reference_cycles_n'])

        reference_cycles = [min_cycle + i*delta_cycle for i in numpy.arange(1, fit_args['reference_cycles_n'] + 1)]
        all_reference_mats = []
        # then for reference cycle, mask all cycles < reference cycle compute the average.
        for reference_cycle in reference_cycles:
            prev_matrices = all_mats['count_matrix'][all_mats['cycle_number'] <= reference_cycle]
            avg_matrices = numpy.average(prev_matrices)
            all_reference_mats.append((reference_cycle, avg_matrices))
            # each step points to the nearest reference cycle

        all_reference_mats = numpy.array(all_reference_mats, dtype=[('cycle_number', 'f4'),

                                                ('count_matrix', 'f4', (
                                                len(sign_grid), len(voltage_grid_degradation), len(current_grid), len(temperature_grid))),
                                                ])


        cyc_grp_dict = {}
        for typ in ['chg', 'dchg']:
            if typ == 'dchg':
                sign = -1.
            else:
                sign = 1.

            if typ == 'dchg':
                groups = DischargeCycleGroup.objects.filter(
                    barcode = barcode
                ).order_by('constant_rate')
            else:
                groups = ChargeCycleGroup.objects.filter(
                    barcode = barcode
                ).order_by('constant_rate')
            for cyc_group in groups:
                result = []

                for cyc in cyc_group.cycle_set.order_by('cycle_number'):
                    if cyc.valid_cycle:
                        post_process_results = \
                            machine_learning_post_process_cycle(
                                cyc,
                                voltage_grid,
                                typ,
                                current_max_n = fit_args['current_max_n']
                            )

                        if post_process_results is None:
                            continue

                        result.append((
                            cyc.get_offset_cycle(),
                            post_process_results['cc_voltages'],
                            post_process_results['cc_capacities'],
                            post_process_results['cc_masks'],
                            sign * post_process_results['cv_currents'],
                            post_process_results['cv_capacities'],
                            post_process_results['cv_masks'],
                            sign * post_process_results['constant_current'],
                            -1. * sign * post_process_results[
                                'end_current_prev'
                            ],
                            sign * post_process_results['end_current'],
                            post_process_results['end_voltage_prev'],
                            post_process_results['end_voltage'],
                            post_process_results['last_cc_voltage'],
                            post_process_results['last_cc_capacity'],
                            post_process_results['last_cv_capacity'],
                            cyc.get_temperature()
                        ))

                res = numpy.array(
                    result,
                    dtype = [
                        ('cycle_number', 'f4'),

                        ('cc_voltage_vector', 'f4', len(voltage_grid)),
                        ('cc_capacity_vector', 'f4', len(voltage_grid)),
                        ('cc_mask_vector', 'f4', len(voltage_grid)),

                        ('cv_current_vector', 'f4', fit_args['current_max_n']),
                        ('cv_capacity_vector', 'f4', fit_args['current_max_n']),
                        ('cv_mask_vector', 'f4', fit_args['current_max_n']),

                        ('constant_current', 'f4'),
                        ('end_current_prev', 'f4'),
                        ('end_current', 'f4'),
                        ('end_voltage_prev', 'f4'),
                        ('end_voltage', 'f4'),
                        ('last_cc_voltage', 'f4'),
                        ('last_cc_capacity', 'f4'),
                        ('last_cv_capacity', 'f4'),
                        ('temperature', 'f4'),
                    ]
                )

                cyc_grp_dict[
                    (cyc_group.constant_rate, cyc_group.end_rate_prev,
                     cyc_group.end_rate, cyc_group.end_voltage,
                     cyc_group.end_voltage_prev, typ)
                ] = (
                    res,
                    {
                        'avg_constant_current': numpy.average(
                            res['constant_current']
                        ),
                        'avg_end_current_prev': numpy.average(
                            res['end_current_prev']
                        ),
                        'avg_end_current':      numpy.average(
                            res['end_current']
                        ),
                        'avg_end_voltage_prev': numpy.average(
                            res['end_voltage_prev']
                        ),
                        'avg_end_voltage':      numpy.average(
                            res['end_voltage']
                        ),
                        'avg_last_cc_voltage':  numpy.average(
                            res['last_cc_voltage']
                        ),
                    }
                )

        all_data[barcode] = {'cyc_grp_dict':cyc_grp_dict, 'all_reference_mats':all_reference_mats}

    return {'all_data':all_data, 'voltage_grid':voltage_grid_degradation, 'current_grid':current_grid, 'temperature_grid':temperature_grid, 'sign_grid':sign_grid}


# === End: initial processing ==================================================
def compile_dataset(fit_args):
    if not os.path.exists(fit_args['path_to_dataset']):
        os.mkdir(fit_args['path_to_dataset'])
    my_barcodes = make_my_barcodes(fit_args)
    pick = initial_processing(my_barcodes, fit_args)
    with open(
        os.path.join(
            fit_args['path_to_dataset'],
            'dataset_ver_{}.file'.format(fit_args['dataset_version'])
        ),
        'wb'
    ) as f:
        pickle.dump(pick, f, pickle.HIGHEST_PROTOCOL)


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path_to_dataset', required = True)
        parser.add_argument('--dataset_version', required = True)
        parser.add_argument('--voltage_grid_min_v', type = float, default = 2.5)
        parser.add_argument('--voltage_grid_max_v', type = float, default = 5.0)
        parser.add_argument('--voltage_grid_n_samples', type = int,
                            default = 32)

        parser.add_argument('--current_grid_min_v', type=float, default=1.)
        parser.add_argument('--current_grid_max_v', type=float, default=1000.)
        parser.add_argument('--current_grid_n_samples', type=int, default=8)

        parser.add_argument('--temperature_grid_min_v', type=float, default=-20.)
        parser.add_argument('--temperature_grid_max_v', type=float, default=80.)
        parser.add_argument('--temperature_grid_n_samples', type=int, default=3)

        parser.add_argument('--current_max_n', type = int, default = 8)

        parser.add_argument('--reference_cycles_n', type=int, default=10)
        parser.add_argument(
            '--wanted_barcodes',
            type = int,
            nargs = '+',
            default = [83220, 83083]
        )

    def handle(self, *args, **options):
        compile_dataset(options)
