import argparse
import os
import pickle
import numpy
import math
import re
import copy
from neware_parser.models import *
from neware_parser.neware_processing_functions import machine_learning_post_process_cycle
from django.core.management.base import BaseCommand
import tensorflow as tf
from django.db.models import Max,Min
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, GlobalAveragePooling1D,
    BatchNormalization, Conv1D, Layer
)
from tensorflow.keras import Model

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
        database_file__deprecated=False,
        database_file__is_valid=True
    ).exclude(
        database_file__valid_metadata=None
    ).order_by(
        'database_file__valid_metadata__barcode'
    ).values_list(
        'database_file__valid_metadata__barcode',
        flat=True
    ).distinct()

    used_barcodes = []
    for b in my_barcodes:
        if CycleGroup.objects.filter(barcode=b).exists():
            used_barcodes.append(b)

    if len(fit_args['wanted_barcodes'])==0:
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
    all_cycs = Cycle.objects.filter(group__barcode__in=my_barcodes, valid_cycle=True)
    my_max = max(all_cycs.aggregate(Max('chg_maximum_voltage'))['chg_maximum_voltage__max'],
        all_cycs.aggregate(Max('dchg_maximum_voltage'))['dchg_maximum_voltage__max'])
    my_min= min(all_cycs.aggregate(Min('chg_minimum_voltage'))['chg_minimum_voltage__min'],
        all_cycs.aggregate(Min('dchg_minimum_voltage'))['dchg_minimum_voltage__min'])

    my_max = clamp(min_v, my_max, max_v)
    my_min = clamp(min_v, my_min, max_v)

    delta = (my_max - my_min)/float(n_samples-1.)
    return numpy.array([my_min + delta*float(i) for i in range(n_samples)])


def initial_processing(my_barcodes, fit_args):
    all_data = {}
    voltage_grid = make_voltage_grid(fit_args['voltage_grid_min_v'], fit_args['voltage_grid_max_v'], fit_args['voltage_grid_n_samples'], my_barcodes)

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

        #TODO(sam): add the aux vars:
        # - the lower cutoff of discharge (for previous cycle)
        # - the rate of discharge (for previous cycle)
        # - cutoff rate of charge (for current cycle)
        # - cutoff rate of discharge (for previous cycle)
        fs = get_files_for_barcode(barcode)
        for f in fs:
            steps = Step.objects.filter(cycle__cycling_file=f).order_by('cycle__cycle_number', 'step_number')
            if len(steps) == 0:
                continue
            first_step = steps[0]

            if 'CC_' in first_step.step_type:
                sign = +1.
                if 'DChg' in first_step.step_type:
                    sign = -1.

                first_step.end_current_prev = 0.
                first_step.constant_current = sign * first_step.average_current_by_capacity
                first_step.end_current = first_step.constant_current
                if sign > 0:
                    first_step.end_voltage = first_step.maximum_voltage
                    first_step.end_voltage_prev = first_step.minimum_voltage
                else:
                    first_step.end_voltage = first_step.minimum_voltage
                    first_step.end_voltage_prev = first_step.maximum_voltage

            if 'CCCV_' in first_step.step_type:
                sign = +1.
                if 'DChg' in first_step.step_type:
                    sign = -1.

                first_step.end_current_prev = 0.
                first_step.constant_current = sign * first_step.maximum_current
                first_step.end_current = sign * first_step.minimum_current
                if sign > 0:
                    first_step.end_voltage = first_step.maximum_voltage
                    first_step.constant_voltage = first_step.maximum_voltage
                    first_step.end_voltage_prev = first_step.minimum_voltage

                else:

                    first_step.end_voltage = first_step.minimum_voltage
                    first_step.constant_voltage = first_step.minimum_voltage
                    first_step.end_voltage_prev = first_step.maximum_voltage

            first_step.save()

            if len(steps) == 1:
                continue
            for i in range(1, len(steps)):
                step = steps[i]
                if 'CC_' in step.step_type:
                    sign = +1.
                    if 'DChg' in step.step_type:
                        sign = -1.
                    step.end_current_prev = steps[i-1].end_current
                    step.end_voltage_prev = steps[i-1].end_voltage
                    step.constant_current = sign * step.average_current_by_capacity
                    step.end_current = step.constant_current
                    if sign > 0:
                        step.end_voltage = step.maximum_voltage
                    else:
                        step.end_voltage = step.minimum_voltage

                if 'CCCV_' in step.step_type:
                    sign = +1.
                    if 'DChg' in step.step_type:
                        sign = -1.

                    step.end_current_prev = steps[i - 1].end_current
                    step.end_voltage_prev = steps[i - 1].end_voltage
                    step.constant_current = sign * step.maximum_current
                    step.end_current = sign * step.minimum_current
                    if sign > 0:
                        step.end_voltage = step.maximum_voltage
                        step.constant_voltage = step.maximum_voltage

                    else:

                        step.end_voltage = step.minimum_voltage
                        step.constant_voltage = step.minimum_voltage
                step.save()

        for cyc_group in CycleGroup.objects.filter(
                barcode=barcode
        ).order_by('discharging_rate'):
            result_dchg = []
            result_chg = []

            for cyc in cyc_group.cycle_set.order_by(
                    'cycle_number'):
                if cyc.valid_cycle:
                    vq_curve_dchg, vq_mask_dchg, step_info_dchg = machine_learning_post_process_cycle(cyc, voltage_grid,
                                                                                                      'dchg')
                    if vq_curve_dchg is None:
                        print('discharge:', vq_curve_dchg)

                        continue

                    vq_curve_chg, vq_mask_chg, step_info_chg = machine_learning_post_process_cycle(cyc, voltage_grid,
                                                                                                      'chg')
                    if vq_curve_chg is None:
                        print('charge:', step_info_chg)
                        print([s.step_type for s in cyc.step_set.order_by('cycle__cycle_number', 'step_number')])


                        continue

                    result_dchg.append((
                        cyc.get_offset_cycle(),
                        vq_curve_dchg,
                        vq_mask_dchg,
                        -1.* step_info_dchg['constant_current'],
                        step_info_dchg['end_current_prev'],
                        step_info_dchg['end_voltage_prev'],
                        cyc.get_temperature()
                    ))

                    result_chg.append((
                        cyc.get_offset_cycle(),
                        vq_curve_chg,
                        vq_mask_chg,
                        step_info_chg['constant_current'],
                        -1.*step_info_chg['end_current_prev'],
                        step_info_chg['end_voltage_prev'],
                        cyc.get_temperature()
                    ))

            res_dchg = numpy.array(
                result_dchg,
                dtype=[
                    ('cycle_number', 'f4'),
                    ('capacity_vector', 'f4', len(voltage_grid)),
                    ('vq_curve_mask', 'f4', len(voltage_grid)),
                    ('constant_current', 'f4'),
                    ('end_current_prev', 'f4'),
                    ('end_voltage_prev', 'f4'),
                    ('temperature', 'f4'),
                ])
            res_chg = numpy.array(
                result_chg,
                dtype=[
                    ('cycle_number', 'f4'),
                    ('capacity_vector', 'f4', len(voltage_grid)),
                    ('vq_curve_mask', 'f4', len(voltage_grid)),
                    ('constant_current', 'f4'),
                    ('end_current_prev', 'f4'),
                    ('end_voltage_prev', 'f4'),
                    ('temperature', 'f4'),
                ])

            cyc_grp_dict[
                (cyc_group.charging_rate, cyc_group.discharging_rate, 'dchg')
            ] = (res_dchg, {'avg_constant_current': numpy.average(res_dchg['constant_current']),
                       'avg_end_current_prev': numpy.average(res_dchg['end_current_prev']),
                       'avg_end_voltage_prev': numpy.average(res_dchg['end_voltage_prev']),}
                 )
            cyc_grp_dict[
                (cyc_group.charging_rate, cyc_group.discharging_rate, 'chg')
            ] = (res_chg, {'avg_constant_current': numpy.average(res_chg['constant_current']),
                            'avg_end_current_prev': numpy.average(res_chg['end_current_prev']),
                            'avg_end_voltage_prev': numpy.average(res_chg['end_voltage_prev']), }
                 )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k in cyc_grp_dict.keys():
            ax.plot(voltage_grid, numpy.average(cyc_grp_dict[k][0]['vq_curve_mask'], axis=0))
        plt.savefig(os.path.join(fit_args['path_to_dataset'], 'masks_version_{}_barcode_{}.png'.format(fit_args['dataset_version'], barcode)))
        plt.close(fig)
        all_data[barcode] = cyc_grp_dict

    return {'voltage_grid':voltage_grid, 'all_data':all_data}


# === End: initial processing ==================================================
def compile_dataset(fit_args):
    my_barcodes = make_my_barcodes(fit_args)
    pick = initial_processing(my_barcodes, fit_args)
    if not os.path.exists(fit_args['path_to_dataset']):
        os.mkdir(fit_args['path_to_dataset'])
    with open(os.path.join(fit_args['path_to_dataset'], 'dataset_ver_{}.file'.format(fit_args['dataset_version'])), 'wb') as f:
        pickle.dump(pick, f, pickle.HIGHEST_PROTOCOL)

class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path_to_dataset', required=True)
        parser.add_argument('--dataset_version', required=True)
        parser.add_argument('--voltage_grid_min_v', type=float, default=2.5)
        parser.add_argument('--voltage_grid_max_v', type=float, default=4.5)
        parser.add_argument('--voltage_grid_n_samples', type=int, default=32)
        parser.add_argument('--wanted_barcodes', type=int, nargs='+', default=[83220, 83083])

    def handle(self, *args, **options):
        compile_dataset(options)



