from neware_parser.models import *
import os
from django.utils import timezone
import collections
import datetime
import os.path
from itertools import repeat
from itertools import starmap
from django.db.models import Max, Q, F
import re
import pytz
from django.db import transaction
import numpy
from scipy.interpolate import PchipInterpolator
import math

halifax_timezone = pytz.timezone("America/Halifax")
UNIFORM_VOLTAGES = .05 * numpy.array(range(2 * 30, 2 * 46))


import FileNameHelper.models as filename_models


def get_good_neware_files():
    exp_type = filename_models.ExperimentType.objects.get(
        category=filename_models.Category.objects.get(name='cycling'),
        subcategory=filename_models.SubCategory.objects.get(name='neware')
    )

    return filename_models.DatabaseFile.objects.filter(
        is_valid=True, deprecated=False).exclude(
        valid_metadata=None).filter(
        valid_metadata__experiment_type=exp_type)




def get_barcodes():
    exp_type = filename_models.ExperimentType.objects.get(
        category=filename_models.Category.objects.get(name='cycling'),
        subcategory=filename_models.SubCategory.objects.get(name='neware')
    )

    all_current_barcodes = filename_models.DatabaseFile.objects.filter(
        is_valid=True, deprecated=False).exclude(
        valid_metadata=None).filter(valid_metadata__experiment_type=exp_type).values_list(
        'valid_metadata__barcode', flat=True).distinct()

    return all_current_barcodes

def strip(string, sub):
    if string.endswith(sub):
        return strip(string[:-1], sub)
    else:
        return string

def parse_time( my_realtime_string):
    matchObj1 = re.match(r'(\d\d\d\d)-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})', my_realtime_string)
    matchObj2 = re.match(r'(\d{1,2})/(\d{1,2})/(\d\d\d\d) (\d{1,2}):(\d{1,2}):(\d{1,2})', my_realtime_string)
    matchObj3 = re.match(r'(\d{1,2})/(\d{1,2})/(\d\d\d\d) (\d{1,2}):(\d{1,2})', my_realtime_string)


    second_accuracy = True
    if matchObj1:
        int_year = int(matchObj1.group(1))
        int_month = int(matchObj1.group(2))
        int_day = int(matchObj1.group(3))
        int_hour = int(matchObj1.group(4))
        int_minute = int(matchObj1.group(5))
        int_second = int(matchObj1.group(6))
    elif matchObj2:
        int_month = int(matchObj2.group(1))
        int_day = int(matchObj2.group(2))
        int_year = int(matchObj2.group(3))
        int_hour = int(matchObj2.group(4))
        int_minute = int(matchObj2.group(5))
        int_second = int(matchObj2.group(6))
    elif matchObj3:
        int_month = int(matchObj3.group(1))
        int_day = int(matchObj3.group(2))
        int_year = int(matchObj3.group(3))
        int_hour = int(matchObj3.group(4))
        int_minute = int(matchObj3.group(5))
        int_second = 0
        second_accuracy = False
    else:
        raise Exception('tried to parse time {}, but only known formats are YYYY-MM-DD hh:mm:ss, MM/DD/YYYY hh:mm:ss, MM/DD/YYYY hh:mm'.format(my_realtime_string))

    return datetime.datetime(int_year, int_month, int_day, hour=int_hour, minute=int_minute, second=int_second), second_accuracy


def identify_variable_position(separated, label, this_line):
    if not label in separated:
        raise Exception('This format is unknown! {}'.format(this_line))
    return separated.index(label)


def test_occupied_position(separated, pos):
    if len(separated) <= pos:
        return False
    elif separated[pos]:
        return True
    else:
        return False


def read_neware(path, last_imported_cycle=-1, CapacityUnits=1.0, VoltageUnits=(1.0 / 1000.0),
                CurrentUnits=1.0):
    print(path)
    '''
    First, determine which format it is.
    Open the file, see if it is nested or separated.
    
    Then, 
    if nested, parse the headers, then compile the data.
    else, go to Step data, parse that header, then compile all step data.
          then, go to Record data, parse that header, then compile all record data.
    
    '''

    with open(path, 'r', errors='ignore') as myfile:
        this_line = myfile.readline()
        separated = this_line.split('\n')[0].split('\t')
        if len(separated) == 1:
            nested = False
            print('NOT nested', separated)
        elif separated[0]:
            nested = True
            print('nested', separated)
        else:
            raise Exception('This format is unknown. {}'.format(this_line))

    with open(path, 'r', errors='ignore') as myfile:
        def get_header_line():
            this_line = myfile.readline()
            separated = [strip(h.split('(')[0], ' ')
                         for h in this_line.split('\n')[0].split('\t')]
            stripped = [h for h in separated if h]
            return this_line, separated, stripped

        def get_normal_line():
            this_line = myfile.readline()
            separated = this_line.split('\n')[0].split('\t')
            stripped = [s for s in separated if s]
            return this_line, separated, stripped

        def parse_step_header(position, nested=True):
            # nested
            this_line, separated, stripped = get_header_line()

            # Verify that this is the step header
            if nested and separated[0]:
                raise Exception('This format is unknown! {}'.format(this_line))

            if not nested:
                position['cycle_id'] = identify_variable_position(separated, label='Cycle ID', this_line=this_line)
            position['step_id'] = identify_variable_position(separated, label='Step ID', this_line=this_line)

            if 'Step Name' in separated:
                position['step_type'] = identify_variable_position(stripped, label='Step Name', this_line=this_line)
            elif 'Step Type' in separated:
                position['step_type'] = identify_variable_position(stripped, label='Step Type', this_line=this_line)
            else:
                raise Exception('This format is unknown! {}'.format(this_line))


        def parse_record_header(position, nested=True):
            this_line, separated, stripped = get_header_line()
            # Verify that this is the record header
            if nested and (separated[0] or separated[position['step_id']]):
                raise Exception('This format is unknown! {}'.format(this_line))

            if not nested:
                position['cycle_id'] = identify_variable_position(separated, label='Cycle ID', this_line=this_line)
                position['step_id'] = identify_variable_position(separated, label='Step ID', this_line=this_line)

            position['record_id'] = identify_variable_position(separated, label='Record ID', this_line=this_line)

            if 'Vol' in stripped:
                iter_dat = [
                    ('voltage','Vol'),
                    ('current','Cur'),
                    ('capacity','Cap'),
                    ('time','Time'),
                    ('realtime','Realtime'),
                ]
            elif 'Voltage' in stripped:
                iter_dat = [
                    ('voltage','Voltage'),
                    ('current','Current'),
                    ('capacity','Capacity'),
                    ('time','Time'),
                    ('realtime','Realtime'),
                ]

            for id, s in iter_dat:
                position[id] = identify_variable_position(
                    stripped,
                    label=s,
                    this_line=this_line
                )

        def parse_normal_step(position, current_cycle, current_step, imported_data, separated, stripped, nested):
            if nested:
                current_step = int(separated[position['step_id']])
            else:
                if test_occupied_position(separated, position['cycle_id']):
                    current_cycle = int(separated[position['cycle_id']])
                else:
                    return current_cycle, current_step

                if (not nested) and (current_cycle <= last_imported_cycle):
                    return current_cycle, current_step

                if test_occupied_position(separated, position['step_id']):
                    current_step = int(separated[position['step_id']])
                else:
                    return current_cycle, current_step

                if not current_cycle in imported_data.keys():
                    imported_data[current_cycle] = collections.OrderedDict([])

            if test_occupied_position(stripped, position['step_type']):
                step_type = stripped[position['step_type']]
                imported_data[current_cycle][current_step] = (step_type, [])

            else:
                return current_cycle, current_step

            return current_cycle, current_step

        def parse_normal_record(position, current_cycle, current_step, imported_data, separated, stripped, nested):
            if not nested:
                if test_occupied_position(separated, position['cycle_id']):
                    current_cycle = int(separated[position['cycle_id']])
                else:
                    return

                if current_cycle <= last_imported_cycle:
                    return

                if test_occupied_position(separated, position['step_id']):
                    current_step = int(separated[position['step_id']])
                else:
                    return

            my_extracted_strings = {}

            for i in ['realtime', 'capacity', 'current', 'voltage']:

                if test_occupied_position(stripped, position[i]):
                    my_extracted_strings[i] = stripped[position[i]]
                else:
                    continue

            my_cap_float = CapacityUnits * float(my_extracted_strings['capacity'])
            my_cur_float = CurrentUnits * float(my_extracted_strings['current'])
            my_vol_float = VoltageUnits * float(my_extracted_strings['voltage'])
            my_time, second_accuracy = parse_time(my_extracted_strings['realtime'])

            imported_data[current_cycle][current_step][1].append(
                [my_vol_float, my_cur_float, my_cap_float, my_time, second_accuracy])

        current_cycle = -1
        current_step = -1
        imported_data = collections.OrderedDict([])
        position = {}

        if nested:
            #Cycle
            this_line, separated, stripped = get_header_line()
            position['cycle_id'] = identify_variable_position(
                separated, label='Cycle ID', this_line = this_line)
            #Step
            parse_step_header(position, nested=nested)
            #Record
            parse_record_header(position, nested=nested)

            for i in range(1000000000):
                this_line, separated, stripped = get_normal_line()
                if this_line == '':
                    break

                if test_occupied_position(separated, position['cycle_id']):
                    current_cycle = int(separated[position['cycle_id']])
                    if current_cycle <= last_imported_cycle:
                        continue
                    imported_data[current_cycle] = collections.OrderedDict([])
                    if (current_cycle % 100) == 0:
                        print('cyc: ', current_cycle)
                else:
                    if current_cycle <= last_imported_cycle:
                        continue

                    if test_occupied_position(separated, position['step_id']):
                        current_cycle, current_step = parse_normal_step(
                            position, current_cycle,
                            current_step, imported_data,
                            separated, stripped,
                            nested
                        )

                    elif test_occupied_position(separated, position['record_id']):
                        parse_normal_record(
                            position, current_cycle,
                            current_step, imported_data,
                            separated, stripped,
                            nested
                        )

                    else:
                        continue


        else:
            #not nested
            for i in range(10000000000):
                this_line, separated, stripped = get_normal_line()
                if separated[0] == 'Step Data':
                    break

            #This is the step data header
            parse_step_header(position, nested=nested)

            #This is the step data
            for i in range(10000000000):
                this_line, separated, stripped = get_normal_line()
                if separated[0] == 'Record Data':
                    break

                current_cycle, current_step= parse_normal_step(
                        position, current_cycle,
                        current_step,imported_data,
                        separated, stripped,
                        nested
                    )

            # This is the record data header
            parse_record_header(position, nested=nested)


            # This is the record data
            for i in range(10000000000):
                this_line, separated, stripped = get_normal_line()
                if this_line == '':
                    break

                parse_normal_record(
                    position, current_cycle,
                    current_step, imported_data,
                    separated, stripped,
                    nested
                )

        return imported_data



def import_single_file(database_file, DEBUG=False):
    time_of_running_script = timezone.now()
    error_message = {}
    if not database_file.is_valid or database_file.deprecated or database_file.valid_metadata is None:
        return error_message

    full_path = os.path.join(database_file.root, database_file.filename)

    error_message['filepath'] = full_path
    already_cached = CyclingFile.objects.filter(database_file=database_file).exists()
    if not already_cached:
        error_message['cached'] = 'None'

    if already_cached:
        f = CyclingFile.objects.get(database_file=database_file)
        time_origin = database_file.last_modified
        time_cached = f.import_time

        if time_origin > time_cached:
            already_cached = False
            error_message['cached'] = 'Stale'
        else:
            error_message['cached'] = 'Valid'

    if already_cached:
        error_message['error'] = False
        return error_message

    with transaction.atomic():
        f, created = CyclingFile.objects.get_or_create(database_file=database_file)
        if created:
            f.set_uniform_voltages(UNIFORM_VOLTAGES)

        def get_last_cycle():
            if f.cycle_set.exists():
                last_imported_cycle = f.cycle_set.aggregate(Max('cycle_number'))
                last_imported_cycle = last_imported_cycle['cycle_number__max']
            else:
                last_imported_cycle = -1

            return last_imported_cycle

        last_imported_cycle = get_last_cycle()
        print(last_imported_cycle)




        def write_to_database(data_table):

            cycles = []
            if len(data_table) != 0:
                for cyc in list(data_table.keys())[:-1]:
                    if cyc > last_imported_cycle:
                        if len(data_table[cyc]) > 0:
                            passed = False
                            for step in data_table[cyc].keys():
                                if len(data_table[cyc][step][1]) > 0:
                                    passed = True
                                    break
                            if passed:
                                cycles.append(
                                    Cycle(cycling_file=f, cycle_number=cyc)
                                )

            Cycle.objects.bulk_create(cycles)
            steps = []
            for cyc in f.cycle_set.filter(cycle_number__gt=last_imported_cycle).order_by('cycle_number'):
                cyc_steps = data_table[cyc.cycle_number]
                for step in cyc_steps.keys():
                    if len(cyc_steps[step][1]) == 0:
                        continue

                    step_type, data = cyc_steps[step]
                    start_time = min([d[3] for d in data])
                    second_accuracy = all([d[4] for d in data])

                    steps.append(
                        Step(
                            cycle=cyc,
                            step_number=step,
                            step_type=cyc_steps[step][0],
                            start_time=halifax_timezone.localize(start_time),
                            second_accuracy=second_accuracy,

                        )
                    )

                    steps[-1].set_v_c_q_t_data(
                        numpy.array(
                            [
                                d[:3] + [(d[3] - start_time).total_seconds() / (60. * 60.)]
                                for d in data
                            ]
                        )

                    )

            Step.objects.bulk_create(steps)
            f.import_time = time_of_running_script
            f.save()

        if DEBUG:
            data_table = read_neware(full_path, last_imported_cycle=last_imported_cycle)
            write_to_database(data_table)

            error_message['error'] = False
            return error_message

        else:
            try:
                data_table = read_neware(full_path, last_imported_cycle=last_imported_cycle)


            except Exception as e:
                error_message['error'] = True
                error_message['error type'] = 'ReadNeware'
                error_message['error verbatum'] = e
                return error_message


            try:
                write_to_database(data_table)

                error_message['error'] = False
                return error_message

            except:
                error_message['error'] = True
                error_message['error type'] = 'WriteCache'
                return error_message


def bulk_import(barcodes=None, DEBUG=False):
    if barcodes is not None:
        neware_files = get_good_neware_files().filter(valid_metadata__barcode__in=barcodes)

    else:
        neware_files = get_good_neware_files()
    errors = list(map(lambda x: import_single_file(x, DEBUG), neware_files))
    return list(filter(lambda x: x['error'], errors))



def is_monotonically_decreasing(qs):
    mono = True
    for i in range(1, len(qs) - 1):
        if qs[i] > qs[i - 1]:
            mono = False
            break
    return mono


def average_data(data_source_, val_keys, sort_val, weight_func=None, weight_exp_func=None, compute_std=False):
    if weight_func is not None:
        weights, works = weight_func(data_source_)
    else:
        weights, works = numpy.ones(len(data_source_)), numpy.ones(len(data_source_), dtype=numpy.bool)

    weights = weights[works]
    data_source = data_source_[works]

    if weight_exp_func is not None:
        weights_exp = weight_exp_func(data_source)
    else:
        weights_exp = numpy.zeros(len(data_source))

    vals = data_source[sort_val]

    all_ = numpy.stack([vals, weights, weights_exp] + [data_source[s_v] for s_v in val_keys], axis=1)
    all_ = numpy.sort(all_, axis=0)
    if len(all_) >= 15:
        all_ = all_[2:-2]
    elif len(all_) >= 9:
        all_ = all_[1:-1]

    weights = all_[:, 1]
    weights_exp = all_[:, 2]
    vals = all_[:, 3:]
    max_weights_exp = numpy.max(weights_exp)
    actual_weights = weights * numpy.exp(
        weights_exp - max_weights_exp)
    if sum(actual_weights) == 0.0:
        actual_weights = None
    avg = numpy.average(vals, weights=actual_weights, axis=0)
    if not compute_std:
        return {val_keys[i]: avg[i] for i in range(len(val_keys))}
    else:
        var = numpy.average(numpy.square(vals - avg), weights=actual_weights, axis=0)
        if actual_weights is not None:
            actual_weights = (1. / numpy.sum(actual_weights) * actual_weights) + 1e-10
            actual_weights = numpy.expand_dims(actual_weights, axis=1)
            var = numpy.sum(actual_weights * numpy.square(vals - avg), axis=0) / (1e-10 + numpy.abs(
                numpy.sum(actual_weights, axis=0) - (numpy.sum(numpy.square(actual_weights), axis=0) / (
                        1e-10 + numpy.sum(actual_weights, axis=0)))))
        std = numpy.sqrt(var)
        return {val_keys[i]: (avg[i], std[i]) for i in range(len(val_keys))}


def default_deprecation(barcode):
    with transaction.atomic():
        files = get_good_neware_files().filter(valid_metadata__barcode=barcode)
        if files.count() == 0:
            return
        start_cycles = files.order_by('valid_metadata__start_cycle').values_list(
        'valid_metadata__start_cycle', flat=True).distinct()
        for start_cycle in start_cycles:
            files_start = files.filter(valid_metadata__start_cycle=start_cycle)
            if files_start.count() <= 1:
                continue
            last_modified_max = files_start.aggregate(Max('last_modified'))[
                                'last_modified__max']

            filesize_max = files_start.aggregate(Max('filesize'))[
                                            'filesize__max']

            winners = files_start.filter(last_modified=last_modified_max, filesize=filesize_max)
            if winners.count() == 0:
                continue
            winner_id = winners[0].id

            for f in files_start.exclude(id=winner_id):
                f.set_deprecated(True)
                f.save()




def process_barcode(barcode, NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS=10):
    print(barcode)
    with transaction.atomic():
        CycleGroup.objects.filter(barcode=barcode).delete()
        files = get_good_neware_files().filter(valid_metadata__barcode=barcode)
        new_data = []
        for cyc in Cycle.objects.filter(
                cycling_file__database_file__in = files, valid_cycle=True).order_by('cycle_number'):
            new_data.append(
                (
                    cyc.id,  # id
                    float(cyc.cycle_number + cyc.cycling_file.database_file.valid_metadata.start_cycle),  # cyc

                    math.log(1e-10 + abs(cyc.chg_maximum_current)),  # charge current
                    math.log(1e-10 + abs(cyc.dchg_minimum_current)),  # discharge current

                    math.log(1e-10 + cyc.chg_total_capacity),  # charge capacity
                    math.log(1e-10 + cyc.dchg_total_capacity),  # discharge capacity
                    cyc.discharge_curve_minimum_voltage,
                    cyc.discharge_curve_maximum_voltage,
                )
            )

        if len(new_data) > NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS:
            new_data = numpy.array(
                new_data, dtype=[
                    ('cycle_id', int),
                    ('cycle_number', 'f4'),
                    ('charge_c_rate', 'f4'),
                    ('discharge_c_rate', 'f4'),

                    ('charge_cap', 'f4'),
                    ('discharge_cap', 'f4'),
                    ('min_v', 'f4'),
                    ('max_v', 'f4'),
                ])
            discharge_rates = numpy.sort(new_data['discharge_c_rate'])
            typical_discharge_rate = numpy.median(discharge_rates)
            print('typical discharge rate: {}'.format(math.exp(typical_discharge_rate)))
            charge_rates = numpy.sort(new_data['charge_c_rate'])
            typical_charge_rate = numpy.median(charge_rates)
            print('typical charge rate: {}'.format(math.exp(typical_charge_rate)))

            theoretical_dict = average_data(
                new_data,
                val_keys=['discharge_cap'],
                sort_val='discharge_cap',
                weight_exp_func=lambda x: -0.1 * numpy.abs(5 - x['cycle_number']) - 0.1 * numpy.abs(
                    typical_discharge_rate - x['discharge_c_rate']) - 0.1 * numpy.abs(
                    typical_charge_rate - x['charge_c_rate'])
            )

            theoretical_cap_no_rate = theoretical_dict['discharge_cap']

            normalized_data = new_data
            normalized_data['discharge_c_rate'] = normalized_data['discharge_c_rate'] - theoretical_cap_no_rate
            normalized_data['charge_c_rate'] = normalized_data['charge_c_rate'] - theoretical_cap_no_rate
            normalized_data['discharge_cap'] = normalized_data['discharge_cap'] - theoretical_cap_no_rate

            def separate_data(data_table, splitting_var='discharge_c_rate'):
                rate_step_full = .1 * .5
                rate_step_cut = .075 * .5
                min_rate_full = numpy.min(data_table[splitting_var])
                min_rate_full = round(max(min_rate_full, math.log(.005)) - rate_step_full, ndigits=1)

                max_rate_full = numpy.max(data_table[splitting_var])
                max_rate_full = round(min(max_rate_full, math.log(500.)) + rate_step_full, ndigits=1)

                number_of_rate_steps = int((max_rate_full - min_rate_full) / rate_step_full) + 2

                split_data = {}

                prev_rate_mask = numpy.zeros(len(data_table), dtype=numpy.bool)
                prev_avg_rate = 0
                for i2 in range(number_of_rate_steps):
                    avg_rate = (rate_step_full * i2) + min_rate_full
                    min_rate = avg_rate - rate_step_cut
                    max_rate = avg_rate + rate_step_cut

                    rate_mask = numpy.logical_and(
                        min_rate <= data_table[splitting_var],
                        data_table[splitting_var] <= max_rate
                    )
                    if not numpy.any(rate_mask):
                        continue

                    intersection_mask = numpy.logical_and(rate_mask, prev_rate_mask)

                    if numpy.array_equal(rate_mask, prev_rate_mask) or numpy.array_equal(intersection_mask,
                                                                                         rate_mask):
                        prev_rate_mask = rate_mask
                        prev_avg_rate = avg_rate
                        continue

                    elif numpy.array_equal(intersection_mask, prev_rate_mask) and numpy.any(prev_rate_mask):
                        if prev_avg_rate in split_data.keys():
                            del split_data[prev_avg_rate]

                    split_data[avg_rate] = rate_mask
                    prev_rate_mask = rate_mask
                    prev_avg_rate = avg_rate

                sorted_keys = list(split_data.keys())
                sorted_keys.sort()
                avg_sorted_keys = {}
                for sk in sorted_keys:
                    avg_sorted_keys[sk] = numpy.mean(
                        data_table[split_data[sk]][splitting_var])

                grouped_rates = {}
                for sk in sorted_keys:
                    found = False
                    for k in grouped_rates.keys():
                        if abs(avg_sorted_keys[k] - avg_sorted_keys[sk]) < 0.13:
                            grouped_rates[k].append(sk)
                            found = True
                            break
                    if not found:
                        grouped_rates[sk] = [sk]

                split_data2 = {}

                for k in grouped_rates.keys():
                    split_data2[k] = numpy.zeros(len(data_table), dtype=numpy.bool)
                    for kk in grouped_rates[k]:
                        split_data2[k] = numpy.logical_or(split_data[kk], split_data2[k])

                return split_data2

            split_data2 = separate_data(normalized_data, splitting_var='discharge_c_rate')

            summary_data = {}
            for k in split_data2.keys():

                # print(k)
                rate_normalized_data = normalized_data[split_data2[k]]

                split_data3 = separate_data(rate_normalized_data, splitting_var='charge_c_rate')
                for kk in split_data3.keys():
                    rate_rate_normalized_data = rate_normalized_data[split_data3[kk]]

                    avg_discharge_rate = numpy.mean(rate_rate_normalized_data['discharge_c_rate'])
                    avg_charge_rate = numpy.mean(rate_rate_normalized_data['charge_c_rate'])

                    summary_data[(avg_charge_rate, avg_discharge_rate)] = rate_rate_normalized_data

            for k in summary_data.keys():

                cyc_group = CycleGroup(barcode=barcode, charging_rate=math.exp(k[0]), discharging_rate=math.exp(k[1]))
                cyc_group.save()

                for cyc_id in (summary_data[k]['cycle_id']):
                    cyc_group.cycle_set.add(Cycle.objects.get(id=cyc_id))


            bn, _ = BarcodeNode.objects.get_or_create(barcode=barcode)
            bn.last_modified = timezone.now()
            bn.save()

def process_single_file(f,DEBUG=False):
    error_message = {'filename': f.database_file.filename}
    print(f.database_file.filename)

    def thing_to_try():
        with transaction.atomic():
            if f.process_time <= f.import_time:
                # must process the step data to summarize it.
                uniform_voltages = f.get_uniform_voltages()
                for cyc in f.cycle_set.filter(processed=False):

                    for step in cyc.step_set.all():
                        dat = step.get_v_c_q_t_data()
                        v_min = numpy.nanmin(dat[:, 0])
                        v_max = numpy.nanmax(dat[:, 0])
                        cur_min = numpy.nanmin(dat[:, 1])
                        cur_max = numpy.nanmax(dat[:, 1])
                        delta_t = numpy.max(dat[:, -1]) - numpy.min(dat[:, -1])
                        capacity = numpy.max(dat[:, 2])
                        '''
                        sum i -> n-1 : 0.5*(vol[i] + vol[i+1]) * (cap[i+1] - cap[i])
                        sum i -> n-1 : (cap[i+1] - cap[i])
                        '''
                        if len(dat) > 1:
                            capacity_differences = numpy.absolute(numpy.diff(dat[:, 2]))
                            voltage_differences = numpy.absolute(numpy.diff(dat[:, 0]))

                            voltage_averages = .5 * (dat[:-1, 0] + dat[1:, 0])
                            current_averages = .5 * (dat[:-1, 1] + dat[1:, 1])
                            if numpy.nanmax(capacity_differences) > 0:
                                cur_avg_by_cap = numpy.average(current_averages, weights=capacity_differences)
                                vol_avg_by_cap = numpy.average(voltage_averages, weights=capacity_differences)
                            else:
                                cur_avg_by_cap = .5 * (cur_min + cur_max)
                                vol_avg_by_cap = .5 * (v_min + v_max)

                            if numpy.nanmax(voltage_differences) > 0:
                                cur_avg_by_vol = numpy.average(current_averages, weights=voltage_differences)
                            else:
                                cur_avg_by_vol = .5 * (cur_min + cur_max)

                        else:
                            cur_avg_by_cap = .5 * (cur_min + cur_max)
                            vol_avg_by_cap = .5 * (v_min + v_max)
                            cur_avg_by_vol = .5 * (cur_min + cur_max)

                        step.total_capacity = capacity
                        step.average_voltage = vol_avg_by_cap
                        step.minimum_voltage = v_min
                        step.maximum_voltage = v_max
                        step.average_current_by_capacity = cur_avg_by_cap
                        step.average_current_by_voltage = cur_avg_by_vol
                        step.minimum_current = cur_min
                        step.maximum_current = cur_max
                        step.duration = delta_t

                        step.save()

                    # process the cycle
                    discharge_query = Q(step_type__contains='DChg')
                    charge_query = ~Q(step_type__contains='DChg') & Q(step_type__contains='Chg')
                    discharge_data = numpy.array(
                        [
                            (step.total_capacity,
                             step.average_voltage,
                             step.minimum_voltage,
                             step.maximum_voltage,
                             step.average_current_by_capacity,
                             step.average_current_by_voltage,
                             step.minimum_current,
                             step.maximum_current,
                             step.duration,

                             )
                            for step in cyc.step_set.filter(discharge_query)
                        ],
                        dtype=numpy.dtype(
                            [
                                ('total_capacity', float),
                                ('average_voltage', float),
                                ('minimum_voltage', float),
                                ('maximum_voltage', float),
                                ('average_current_by_capacity', float),
                                ('average_current_by_voltage', float),
                                ('minimum_current', float),
                                ('maximum_current', float),
                                ('duration', float),
                            ]
                        )
                    )
                    charge_data = numpy.array(
                        [
                            (step.total_capacity,
                             step.average_voltage,
                             step.minimum_voltage,
                             step.maximum_voltage,
                             step.average_current_by_capacity,
                             step.average_current_by_voltage,
                             step.minimum_current,
                             step.maximum_current,
                             step.duration,
                             )
                            for step in cyc.step_set.filter(charge_query)
                        ],
                        dtype=numpy.dtype(
                            [
                                ('total_capacity', float),
                                ('average_voltage', float),
                                ('minimum_voltage', float),
                                ('maximum_voltage', float),
                                ('average_current_by_capacity', float),
                                ('average_current_by_voltage', float),
                                ('minimum_current', float),
                                ('maximum_current', float),
                                ('duration', float),
                            ]
                        )
                    )

                    # charge agregation
                    if len(charge_data) != 0:
                        cyc.chg_total_capacity = numpy.sum(charge_data['total_capacity'])
                        cyc.chg_duration = numpy.sum(charge_data['duration'])
                        cyc.chg_minimum_voltage = numpy.min(charge_data['minimum_voltage'])
                        cyc.chg_maximum_voltage = numpy.max(charge_data['maximum_voltage'])
                        cyc.chg_minimum_current = numpy.min(charge_data['minimum_current'])
                        cyc.chg_maximum_current = numpy.max(charge_data['maximum_current'])
                        cyc.chg_average_voltage = numpy.average(
                            charge_data['average_voltage'],
                            weights=1e-6 + charge_data['total_capacity']
                        )
                        cyc.chg_average_current_by_capacity = numpy.average(
                            charge_data['average_current_by_capacity'],
                            weights=1e-6 + charge_data['total_capacity']
                        )
                        cyc.chg_average_current_by_voltage = numpy.average(
                            charge_data['average_current_by_voltage'],
                            weights=1e-6 + charge_data['maximum_voltage'] - charge_data['minimum_voltage']
                        )
                    else:
                        cyc.delete()
                        continue

                    if len(discharge_data) != 0:
                        cyc.dchg_total_capacity = numpy.sum(discharge_data['total_capacity'])
                        cyc.dchg_duration = numpy.sum(discharge_data['duration'])
                        cyc.dchg_minimum_voltage = numpy.min(discharge_data['minimum_voltage'])
                        cyc.dchg_maximum_voltage = numpy.max(discharge_data['maximum_voltage'])
                        cyc.dchg_minimum_current = numpy.min(discharge_data['minimum_current'])
                        cyc.dchg_maximum_current = numpy.max(discharge_data['maximum_current'])
                        cyc.dchg_average_voltage = numpy.average(
                            discharge_data['average_voltage'],
                            weights=1e-6 + discharge_data['total_capacity']
                        )
                        cyc.dchg_average_current_by_capacity = numpy.average(
                            discharge_data['average_current_by_capacity'],
                            weights=1e-6 + discharge_data['total_capacity']
                        )
                        cyc.dchg_average_current_by_voltage = numpy.average(
                            discharge_data['average_current_by_voltage'],
                            weights=1e-6 + discharge_data['maximum_voltage'] - discharge_data['minimum_voltage']
                        )
                    else:
                        cyc.delete()
                        continue

                    # clean up the vq curves
                    if cyc.cycle_number < 1:
                        cyc.delete()
                        continue

                    if cyc.cycle_number == 1:
                        cyc.valid_cycle = False

                    if abs(cyc.dchg_total_capacity) < 1e-1:
                        cyc.delete()
                        continue

                    steps = cyc.step_set.filter(step_type__contains='DChg').order_by('cycle__cycle_number','step_number')
                    if len(steps) == 0:
                        cyc.delete()
                        continue
                    else:
                        first_step = steps[0]
                        vcqt_curve = first_step.get_v_c_q_t_data()
                        v_min = first_step.minimum_voltage
                        v_max = first_step.maximum_voltage
                        curve = vcqt_curve[:, [0, 2]]

                    cursor = numpy.array([-1, len(curve)], dtype=numpy.int32)
                    limits_v = numpy.array([10., -10.], dtype=numpy.float32)
                    limits_q = numpy.array([-10, 1e6], dtype=numpy.float32)
                    masks = numpy.zeros(len(curve), dtype=numpy.bool)

                    never_added_up = True
                    never_added_down = True
                    if len(curve) < 3:
                        cyc.delete()
                        continue

                    while True:
                        if cursor[0] + 1 >= cursor[1]:
                            break

                        can_add_upward = False
                        delta_upward = 0.
                        can_add_downward = False
                        delta_downward = 0.

                        # Test upward addition
                        dv1 = limits_v[0] - curve[cursor[0] + 1, 0]
                        dq1 = curve[cursor[0] + 1, 1] - limits_q[0]
                        dv2 = curve[cursor[0] + 1, 0] - limits_v[1]
                        dq2 = limits_q[1] - curve[cursor[0] + 1, 1]

                        if dv1 >= 0.0001 and dq1 >= -0.001 and dv2 >= 0.0001 and dq2 >= -0.001:

                            delta_upward = dv1 + dq1 / 100.
                            if dq1 < 0.0001:
                                dq1 = 0.0001
                            delta_upward += abs(dv1) / dq1
                            if abs(dv1) / dq1 < 10.:
                                if never_added_up:
                                    dv1 = curve[cursor[0] + 1, 0] - curve[cursor[0] + 2, 0]
                                    dq1 = curve[cursor[0] + 2, 1] - curve[cursor[0] + 1, 1]
                                    if dv1 >= -0.001 and dq1 >= -0.001:
                                        can_add_upward = True
                                else:
                                    can_add_upward = True

                        # Test downward addition
                        dv1 = limits_v[0] - curve[cursor[1] - 1, 0]
                        dq1 = curve[cursor[1] - 1, 1] - limits_q[0]
                        dv2 = curve[cursor[1] - 1, 0] - limits_v[1]
                        dq2 = limits_q[1] - curve[cursor[1] - 1, 1]

                        if dv1 >= 0.0001 and dq1 >= -0.001 and dv2 >= 0.0001 and dq2 >= -0.001:

                            delta_downward = dv2 + dq2 / 100.

                            if dq2 < 0.0001:
                                dq2 = 0.0001
                            delta_upward += abs(dv2) / dq2
                            if abs(dv2) / dq2 < 10.:
                                if never_added_down:
                                    dv2 = curve[cursor[1] - 2, 0] - curve[cursor[1] - 1, 0]
                                    dq2 = curve[cursor[1] - 1, 1] - curve[cursor[1] - 2, 1]
                                    if dv2 >= -0.001 and dq2 >= -0.001:
                                        can_add_downward = True
                                else:
                                    can_add_downward = True

                        if not can_add_upward and not can_add_downward:
                            cursor[0] += 1
                            cursor[1] += -1
                            continue

                        if can_add_upward and can_add_downward:
                            if delta_upward > delta_downward:
                                my_add = 'Down'
                            else:
                                my_add = 'Up'

                        elif can_add_upward and not can_add_downward:
                            my_add = 'Up'
                        elif not can_add_upward and can_add_downward:
                            my_add = 'Down'

                        if my_add == 'Up':
                            never_added_up = False
                            masks[cursor[0] + 1] = True
                            limits_v[0] = min(limits_v[0], curve[cursor[0] + 1, 0])
                            limits_q[0] = max(limits_q[0], curve[cursor[0] + 1, 1])

                            cursor[0] += 1

                        elif my_add == 'Down':
                            never_added_down = False
                            masks[cursor[1] - 1] = True
                            limits_v[1] = max(limits_v[1], curve[cursor[1] - 1, 0])
                            limits_q[1] = min(limits_q[1], curve[cursor[1] - 1, 1])

                            cursor[1] += -1

                    valid_curve = curve[masks]
                    invalid_curve = curve[~masks]
                    if len(invalid_curve) > 5:
                        cyc.delete()
                        continue

                    # uniformly sample it

                    v = valid_curve[:, 0]
                    q = valid_curve[:, 1]

                    sorted_ind = numpy.argsort(v)
                    v = v[sorted_ind]
                    q = q[sorted_ind]

                    last = v[-1]

                    added_v = numpy.arange(last + 0.01, 4.6, 0.01)
                    added_q = 0. * numpy.arange(last + 0.01, 4.6, 0.01)
                    v = numpy.concatenate((v, added_v), axis=0)
                    q = numpy.concatenate((q, added_q), axis=0)
                    spline = PchipInterpolator(v, q)
                    res = spline(uniform_voltages)

                    if not is_monotonically_decreasing(res):
                        cyc.delete()
                        continue

                    cyc.discharge_curve_minimum_voltage = v_min
                    cyc.discharge_curve_maximum_voltage = v_max
                    cyc.set_discharge_curve(
                        numpy.stack(
                            (
                                res,  # charge curve
                                numpy.where(
                                    numpy.logical_and(
                                        uniform_voltages <= v_min,
                                        uniform_voltages >= v_max
                                    ),
                                    numpy.ones(len(uniform_voltages), dtype=numpy.float32),
                                    0.001 * numpy.ones(len(uniform_voltages), dtype=numpy.float32),
                                )
                            ),
                            axis=1
                        )
                    )

                    cyc.processed = True
                    cyc.save()

                if not f.cycle_set.exists():
                    f.delete()
                    return

                f.process_time = timezone.now()
                f.save()

    if DEBUG:
        thing_to_try()
    else:
        try:
            thing_to_try()

        except Exception as e:
            error_message['error'] = True
            error_message['error data'] = e
            error_message['error step'] = 'processing raw'
            return error_message

    # after all processing.
    error_message['error'] = False
    return error_message

def bulk_process(DEBUG=False, NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS=10):
    errors = list(map(lambda x: process_single_file(x, DEBUG),
                      CyclingFile.objects.filter(database_file__deprecated=False, process_time__lte = F('import_time'))))
    all_current_barcodes = CyclingFile.objects.filter(
        database_file__deprecated=False).values_list(
        'database_file__valid_metadata__barcode', flat=True).distinct()

    print(list(all_current_barcodes))
    for barcode in all_current_barcodes:
        process_barcode(
            barcode,
            NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS)

    return list(filter(lambda x: x['error'], errors))



def bulk_deprecate():
    all_current_barcodes = get_good_neware_files().values_list(
        'valid_metadata__barcode', flat=True).distinct()
    print(list(all_current_barcodes))
    for barcode in all_current_barcodes:
        default_deprecation(
            barcode)


