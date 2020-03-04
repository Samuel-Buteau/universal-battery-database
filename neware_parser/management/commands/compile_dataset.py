import numpy
from django.core.management.base import BaseCommand
from django.db.models import Max, Min

from neware_parser.models import *
from neware_parser.neware_processing_functions import \
    machine_learning_post_process_cycle

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


def clamp(a, x, b):
    x = min(x, b)
    x = max(x, a)
    return x


def make_voltage_grid(min_v, max_v, n_samples, my_barcodes):
    if n_samples < 2:
        n_samples = 2
    all_cycs = Cycle.objects.filter(
        discharge_group__barcode__in = my_barcodes,
        valid_cycle = True
    )
    my_max = max(
        all_cycs.aggregate(Max('chg_maximum_voltage'))[
            'chg_maximum_voltage__max'
        ],
        all_cycs.aggregate(Max('dchg_maximum_voltage'))[
            'dchg_maximum_voltage__max'
        ]
    )
    my_min = min(
        all_cycs.aggregate(Min('chg_minimum_voltage'))[
            'chg_minimum_voltage__min'
        ],
        all_cycs.aggregate(Min('dchg_minimum_voltage'))[
            'dchg_minimum_voltage__min'
        ]
    )

    my_max = clamp(min_v, my_max, max_v)
    my_min = clamp(min_v, my_min, max_v)

    delta = (my_max - my_min) / float(n_samples - 1.)
    return numpy.array([my_min + delta * float(i) for i in range(n_samples)])


def initial_processing(my_barcodes, fit_args):
    all_data = {}
    voltage_grid = make_voltage_grid(
        fit_args['voltage_grid_min_v'],
        fit_args['voltage_grid_max_v'],
        fit_args['voltage_grid_n_samples'],
        my_barcodes
    )

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

                for cyc in cyc_group.cycle_set.order_by(
                    'cycle_number'):
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

        all_data[barcode] = cyc_grp_dict

    return all_data


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
        parser.add_argument('--voltage_grid_max_v', type = float, default = 4.5)
        parser.add_argument('--voltage_grid_n_samples', type = int,
                            default = 32)
        parser.add_argument('--current_max_n', type = int, default = 8)
        parser.add_argument('--wanted_barcodes', type = int, nargs = '+',
                            default = [83220, 83083])

    def handle(self, *args, **options):
        compile_dataset(options)
