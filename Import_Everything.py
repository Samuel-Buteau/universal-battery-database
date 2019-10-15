import argparse
import collections
import copy
import datetime
import os.path
import pickle
from itertools import repeat
from itertools import starmap
from multiprocessing import Pool

import numpy
import re

def raise_stripped(stripped, index):
    if len(stripped) < index:
        raise Exception(
            'stripped: {} does not have enough elements. Index: {}'.format(stripped, index))


def is_number(string, call=float):
    try:
        call(string)
        return True
    except ValueError:
        return False


def from_time(string, call='(H:M:S.ms)'):
    if call == '(H:M:S.ms)':
        return sum(
            numpy.array([1.0, 1.0 / 60.0, 1.0 / (60.0 * 60.0)]) * numpy.array([float(s) for s in string.split(':')]))
    elif call == '(H:M:S:ms)':
        return sum(numpy.array([1.0, 1.0 / 60.0, 1.0 / (60.0 * 60.0), 1.0 / (60.0 * 60.0 * 1000.0)]) * numpy.array(
            [float(s) for s in string.split(':')]))


def is_time(string, call='(H:M:S:ms)'):
    splitted = string.split(':')
    if call == '(H:M:S:ms)':
        if len(splitted) == 4:
            if all([is_number(s, call=float) for s in splitted]):
                return True
    elif call == '(H:M:S.ms)':
        if len(splitted) == 3:
            if all([is_number(s, call=float) for s in splitted]):
                return True

    return False


def strip(string, sub):
    if string.endswith(sub):
        return strip(string[:-1], sub)
    else:
        return string


def from_string_to_time(my_realtime_string):
    matchObj1 = re.match(r'(\d\d\d\d)-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})', my_realtime_string)
    matchObj2 = re.match(r'(\d{1,2})/(\d{1,2})/(\d\d\d\d) (\d{1,2}):(\d{1,2}):(\d{1,2})', my_realtime_string)
    matchObj3 = re.match(r'(\d{1,2})/(\d{1,2})/(\d\d\d\d) (\d{1,2}):(\d{1,2})', my_realtime_string)

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
    else:
        raise Exception('tried to parse time {}, but only known formats are YYYY-MM-DD hh:mm:ss, MM/DD/YYYY hh:mm:ss, MM/DD/YYYY hh:mm'.format(my_realtime_string))

    start_time = datetime.datetime(int_year, int_month, int_day, hour=int_hour, minute=int_minute, second=int_second)
    return start_time


Step_Level_Output_Header = ['Cycle Number', 'Charge Capacity (mAh)', 'Discharge Capacity (mAh)',
                            'Charge Average Voltage (V)', 'Discharge Average Voltage (V)', 'Charge Time (hours)',
                            'Discharge Time (hours)', 'Charge Min Voltage (V)', 'Discharge Min Voltage (V)',
                            'Charge Max Voltage (V)', 'Discharge Max Voltage (V)']


def read_neware(path, my_cycle_id_position=-1, my_step_id_position=-1, my_record_id_position=-1,
                TimeUnits='format_driven', CapacityUnits=1.0, EnergyUnits=1.0, VoltageUnits=(1.0 / 1000.0),
                cumulative_time=False, CurrentUnits=1.0):
    print(path)
    new_file_content = ''
    with open(path, 'r') as myfile:
        my_prev_time_float = 0.0
        my_format = -1
        full_data_dict = collections.OrderedDict([])
        dict_step = {}
        charge_Q = []
        charge_V_avg = []
        charge_Time = []
        charge_V_min = []
        charge_V_max = []
        discharge_Q = []
        discharge_V_avg = []
        discharge_Time = []
        discharge_V_min = []
        discharge_V_max = []
        first_record_next = True
        step_all_tags = []
        record_all_tags = []
        seen_header = False
        seen_step_header = False
        seen_record_header = False
        current_cycle = -1
        current_step = -1
        previous_step = -1
        previous_cycle = -1
        previous_step_cycle = -1

        cycle_id_position = my_cycle_id_position
        step_id_position = my_step_id_position
        record_id_position = my_record_id_position
        data_started = False
        step_parse_over = True
        record_data_started = False

        for i in range(10000000):
            this_line = myfile.readline()
            this_line_is_empty = (this_line == '\n' or this_line == '')

            separated = this_line.split('\n')[0].split('\t')
            if my_format == 3:

                if not data_started and not record_data_started:

                    if not seen_step_header and separated[0] == 'Step Data':
                        seen_step_header = True
                        continue

                    if not seen_record_header and separated[0] == 'Record Data':
                        seen_record_header = True
                        continue

                    if seen_step_header and len(step_all_tags) == 0:

                        separated = [strip(h.split('(')[0], ' ') for h in this_line.split('\n')[0].split('\t')]
                        step_all_tags = [tag for tag in separated if tag]

                        if 'Step Name' in separated:
                            dict_step = {'Step Name': 'Step Name', 'Step Time': 'Step Time', 'Capacity': 'Capacity',
                                         'Energy': 'Energy', 'Start Voltage': 'Start Voltage',
                                         'End Voltage': 'End Voltage'}
                        elif 'Step Type' in separated:
                            dict_step = {'Step Name': 'Step Type', 'Step Time': 'Step Time', 'Capacity': 'Cap',
                                         'Energy': 'Energy', 'Start Voltage': 'Start Vol', 'End Voltage': 'End Vol'}
                        data_started = True
                        continue

                    if seen_record_header and len(record_all_tags) == 0:

                        separated = [strip(h.split('(')[0], ' ') for h in this_line.split('\n')[0].split('\t')]
                        record_all_tags = [tag for tag in separated if tag]

                        if 'Vol' in separated:
                            dict_record = {'Vol': 'Vol', 'Cur': 'Cur', 'Cap': 'Cap'}
                        elif 'Voltage' in separated:
                            dict_record = {'Vol': 'Voltage', 'Cur': 'Current', 'Cap': 'Capacity'}
                        record_data_started = True
                        continue
                    continue

                if not record_data_started and data_started:
                    stripped = [s for s in separated if s]
                    if not this_line_is_empty:
                        raise_stripped(stripped, step_all_tags.index('Cycle ID'))
                        temp_cycle_id = int(stripped[step_all_tags.index('Cycle ID')])
                        raise_stripped(stripped, step_all_tags.index('Step ID'))
                        temp_step_id = int(stripped[step_all_tags.index('Step ID')])

                    if this_line_is_empty or not temp_cycle_id == current_cycle:

                        # finish cycle
                        # if this is the first cycle, nothing to do.
                        if not current_cycle == -1:
                            # if there is a previous cycle, finish that cycle and record it.
                            full_data_dict[current_cycle] = copy.deepcopy(current_cycle_data_dict)
                        if this_line_is_empty:
                            step_parse_over = True
                            data_started = False
                            current_cycle = -1
                            current_step = -1
                            continue
                        # start a new cycle.
                        # init the charge, discharge

                        current_cycle = temp_cycle_id
                        current_cycle_data_dict = collections.OrderedDict([])

                    # this is a step. we will extract stuff from it
                    # labels 'Step Name' 'Step Time' 'Capacity' 'Energy' 'Start Voltage' 'End Voltage'

                    raise_stripped(stripped, step_all_tags.index(dict_step['Step Name']))
                    my_name_string = stripped[step_all_tags.index(dict_step['Step Name'])]
                    raise_stripped(stripped, step_all_tags.index(dict_step['Step Time']))
                    my_time_string = stripped[step_all_tags.index(dict_step['Step Time'])]
                    my_TimeUnits = TimeUnits
                    if my_TimeUnits == 'format_driven':
                        if is_time(my_time_string, '(H:M:S:ms)'):
                            my_TimeUnits = '(H:M:S:ms)'
                        if is_time(my_time_string, '(H:M:S.ms)'):
                            my_TimeUnits = '(H:M:S.ms)'
                        elif is_number(my_time_string):
                            my_TimeUnits = 'days'

                    bad_time = False
                    if my_TimeUnits in ['(H:M:S:ms)', '(H:M:S.ms)']:
                        if is_time(my_time_string, my_TimeUnits):
                            my_time_float = from_time(my_time_string, my_TimeUnits)
                        else:
                            bad_time = True
                    elif my_TimeUnits == 'days':
                        if is_number(my_time_string):
                            my_time_float = 24.0 * float(my_time_string)
                        else:
                            bad_time = True
                    else:
                        raise Exception('invalid TimeUnits {}. stripped {}. step_all_tags {}. my_time_string {}'.format(
                            my_TimeUnits, stripped, step_all_tags, my_time_string))
                    if bad_time:
                        raise Exception('could not read my_time_string {}'.format(my_time_string))

                    if cumulative_time:
                        actual_time = my_time_float - my_prev_time_float
                        my_prev_time_float = my_time_float
                        my_time_float = actual_time

                    raise_stripped(stripped, step_all_tags.index(dict_step['Capacity']))
                    my_cap_string = stripped[step_all_tags.index(dict_step['Capacity'])]
                    my_cap_float = CapacityUnits * float(my_cap_string)

                    raise_stripped(stripped, step_all_tags.index(dict_step['Energy']))
                    my_energy_string = stripped[step_all_tags.index(dict_step['Energy'])]
                    my_energy_float = EnergyUnits * float(my_energy_string)

                    raise_stripped(stripped, step_all_tags.index(dict_step['Start Voltage']))
                    my_start_voltage_string = stripped[step_all_tags.index(dict_step['Start Voltage'])]
                    my_start_voltage_float = VoltageUnits * float(my_start_voltage_string)

                    raise_stripped(stripped, step_all_tags.index(dict_step['End Voltage']))
                    my_end_voltage_string = stripped[step_all_tags.index(dict_step['End Voltage'])]
                    my_end_voltage_float = VoltageUnits * float(my_end_voltage_string)

                    if my_cap_float == 0.0:
                        my_avg_voltage_float = 0.5 * (my_start_voltage_float + my_end_voltage_float)
                    else:
                        my_avg_voltage_float = my_energy_float / my_cap_float

                    current_cycle_data_dict[temp_step_id] = [my_name_string, my_cap_float, my_avg_voltage_float,
                                                             min(my_start_voltage_float, my_end_voltage_float),
                                                             max(my_start_voltage_float, my_end_voltage_float),
                                                             my_time_float]
                    continue

                if record_data_started and not data_started:
                    stripped = [s for s in separated if s]
                    if not (this_line_is_empty or len(stripped) == 0):
                        raise_stripped(stripped, step_all_tags.index('Cycle ID'))
                        temp_cycle_id = int(stripped[step_all_tags.index('Cycle ID')])

                        raise_stripped(stripped, step_all_tags.index('Step ID'))
                        temp_step_id = int(stripped[step_all_tags.index('Step ID')])

                    if this_line_is_empty or not temp_step_id == current_step or not temp_cycle_id == current_cycle:

                        # finish step
                        # if this is the first step, nothing to do.
                        if not current_step == -1:
                            # if there is a previous step, finish that step and record it.

                            # figure out the time of the step (based on current time and previous time)
                            if not this_line_is_empty:
                                raise_stripped(stripped, record_all_tags.index('Realtime'))
                                my_realtime_string = stripped[record_all_tags.index('Realtime')]
                                new_start_time = from_string_to_time(my_realtime_string)

                            else:
                                new_start_time = end_time

                            full_data_dict[current_cycle][current_step][5] = float(
                                (new_start_time - start_time).total_seconds()) / (60.0 * 60.0)
                            full_data_dict[current_cycle][current_step].extend(
                                [numpy.array(step_all_voltages), numpy.array(step_all_currents),
                                 numpy.array(step_all_capacities), numpy.array(step_all_times)])
                            # time index = 5
                        else:
                            raise_stripped(stripped, record_all_tags.index('Realtime'))
                            my_realtime_string = stripped[record_all_tags.index('Realtime')]
                            first_start_time = from_string_to_time(my_realtime_string)
                        if this_line_is_empty:
                            break
                        # start a new cycle.
                        # init the charge, discharge

                        current_cycle = temp_cycle_id
                        current_step = temp_step_id
                        step_all_voltages = []
                        step_all_currents = []
                        step_all_capacities = []
                        step_all_times = []
                        # YYYY-MM-DD HH:MM:SS
                        raise_stripped(stripped, record_all_tags.index('Realtime'))
                        my_realtime_string = stripped[record_all_tags.index('Realtime')]
                        start_time = from_string_to_time(my_realtime_string)
                        end_time = start_time

                    # this is a record. we will extract stuff from it
                    # labels 'Vol' 'Cap' 'Cur'
                    raise_stripped(stripped, record_all_tags.index('Realtime'))
                    my_realtime_string = stripped[record_all_tags.index('Realtime')]
                    end_time = from_string_to_time(my_realtime_string)

                    raise_stripped(stripped, record_all_tags.index(dict_record['Cap']))
                    my_cap_string = stripped[record_all_tags.index(dict_record['Cap'])]
                    my_cap_float = CapacityUnits * float(my_cap_string)

                    raise_stripped(stripped, record_all_tags.index(dict_record['Cur']))
                    my_cur_string = stripped[record_all_tags.index(dict_record['Cur'])]
                    my_cur_float = CurrentUnits * float(my_cur_string)

                    raise_stripped(stripped, record_all_tags.index(dict_record['Vol']))
                    my_vol_string = stripped[record_all_tags.index(dict_record['Vol'])]
                    my_vol_float = VoltageUnits * float(my_vol_string)

                    step_all_capacities.append(my_cap_float)
                    step_all_voltages.append(my_vol_float)
                    step_all_currents.append(my_cur_float)

                    my_delta_time = float(
                        (start_time - first_start_time).total_seconds()) / (60.0 * 60.0)

                    raise_stripped(stripped, record_all_tags.index('Time'))
                    my_time_string = stripped[record_all_tags.index('Time')]
                    my_TimeUnits = TimeUnits
                    if my_TimeUnits == 'format_driven':
                        if is_time(my_time_string, '(H:M:S:ms)'):
                            my_TimeUnits = '(H:M:S:ms)'
                        if is_time(my_time_string, '(H:M:S.ms)'):
                            my_TimeUnits = '(H:M:S.ms)'
                        elif is_number(my_time_string):
                            my_TimeUnits = 'days'

                    if my_TimeUnits == '(H:M:S:ms)':
                        my_time_float = from_time(my_time_string, '(H:M:S:ms)')
                    elif my_TimeUnits == '(H:M:S.ms)':
                        my_time_float = from_time(my_time_string, '(H:M:S.ms)')
                    elif my_TimeUnits == 'days':
                        my_time_float = 24.0 * float(my_time_string)
                    else:
                        raise Exception('invalid TimeUnits {}, stripped {}, step_all_tags {}, my_time_string {}'.format(
                            my_TimeUnits, stripped, step_all_tags, my_time_string))

                    step_all_times.append(my_time_float + my_delta_time)

                    continue

            if not seen_header and len(separated) == 1:

                # this means we have format 3
                my_format = 3
                seen_header = True
                continue

            if not seen_header and separated[0]:
                # this means there are many items on first line.
                # this is format 1
                seen_header = True
                my_format = 1
                separated = [strip(h.split('(')[0], ' ') for h in this_line.split('\n')[0].split('\t')]
                cycle_all_tags = [tag for tag in separated if tag]
                if cycle_id_position == -1:
                    cycle_id_position = separated.index('Cycle ID')
                continue
            if my_format == 1:
                if not seen_step_header and not separated[0]:
                    # this means the first entry is empty and we are at the step header
                    seen_step_header = True
                    separated = [strip(h.split('(')[0], ' ') for h in this_line.split('\n')[0].split('\t')]
                    step_all_tags = [tag for tag in separated if tag]

                    if step_id_position == -1:
                        step_id_position = separated.index('Step ID')
                    if 'Step Name' in separated:
                        dict_step = {'Step Name': 'Step Name', 'Step Time': 'Step Time', 'Capacity': 'Capacity',
                                     'Energy': 'Energy', 'Start Voltage': 'Start Voltage', 'End Voltage': 'End Voltage'}
                    elif 'Step Type' in separated:
                        dict_step = {'Step Name': 'Step Type', 'Step Time': 'Step Time', 'Capacity': 'Cap',
                                     'Energy': 'Energy', 'Start Voltage': 'Start Vol', 'End Voltage': 'End Vol'}
                    continue

                if not seen_record_header and not separated[0] and not separated[step_id_position]:
                    seen_record_header = True

                    separated = [strip(h.split('(')[0], ' ') for h in this_line.split('\n')[0].split('\t')]
                    record_all_tags = [tag for tag in separated if tag]

                    if record_id_position == -1:
                        record_id_position = separated.index('Record ID')
                    if 'Vol' in separated:
                        dict_record = {'Vol': 'Vol', 'Cur': 'Cur', 'Cap': 'Cap'}
                    elif 'Voltage' in separated:
                        dict_record = {'Vol': 'Voltage', 'Cur': 'Current', 'Cap': 'Capacity'}
                    continue

                stripped = [s for s in separated if s]
                if not this_line or this_line_is_empty or separated[cycle_id_position]:
                    # This is a cycle

                    previous_cycle = current_cycle
                    if (not this_line) or (this_line_is_empty):
                        new_start_time = end_time
                        full_data_dict[current_cycle][current_step][5] = float(
                            (new_start_time - start_time).total_seconds()) / (60.0 * 60.0)
                        full_data_dict[current_cycle][current_step].extend(
                            [numpy.array(step_all_voltages), numpy.array(step_all_currents),
                             numpy.array(step_all_capacities),
                             numpy.array(step_all_times)])
                        break
                    # start a new cycle.
                    # init the charge, discharge
                    raise_stripped(stripped, cycle_all_tags.index('Cycle ID'))
                    temp_cycle_id = int(stripped[cycle_all_tags.index('Cycle ID')])
                    current_cycle = temp_cycle_id
                    full_data_dict[current_cycle] = collections.OrderedDict([])
                    continue


                elif separated[step_id_position]:

                    first_record_next = True
                    # this is a step. we will extract stuff from it
                    # labels 'Step Name' 'Step Time' 'Capacity' 'Energy' 'Start Voltage' 'End Voltage'

                    raise_stripped(stripped, step_all_tags.index(dict_step['Step Name']))

                    my_name_string = stripped[step_all_tags.index(dict_step['Step Name'])]
                    raise_stripped(stripped, step_all_tags.index(dict_step['Step Time']))
                    my_time_string = stripped[step_all_tags.index(dict_step['Step Time'])]
                    my_TimeUnits = TimeUnits
                    if my_TimeUnits == 'format_driven':
                        if is_time(my_time_string, '(H:M:S:ms)'):
                            my_TimeUnits = '(H:M:S:ms)'
                        if is_time(my_time_string, '(H:M:S.ms)'):
                            my_TimeUnits = '(H:M:S.ms)'
                        elif is_number(my_time_string):
                            my_TimeUnits = 'days'

                    if my_TimeUnits == '(H:M:S:ms)':
                        my_time_float = from_time(my_time_string, '(H:M:S:ms)')
                    elif my_TimeUnits == '(H:M:S.ms)':
                        my_time_float = from_time(my_time_string, '(H:M:S.ms)')
                    elif my_TimeUnits == 'days':
                        my_time_float = 24.0 * float(my_time_string)
                    else:
                        raise Exception('invalid TimeUnits {}, stripped {}, step_all_tags {}, my_time_string {}'.format(
                            my_TimeUnits, stripped, step_all_tags, my_time_string))
                    if cumulative_time:
                        actual_time = my_time_float - my_prev_time_float
                        my_prev_time_float = my_time_float
                        my_time_float = actual_time

                    raise_stripped(stripped, step_all_tags.index(dict_step['Capacity']))
                    my_cap_string = stripped[step_all_tags.index(dict_step['Capacity'])]
                    my_cap_float = CapacityUnits * float(my_cap_string)

                    raise_stripped(stripped, step_all_tags.index(dict_step['Energy']))
                    my_energy_string = stripped[step_all_tags.index(dict_step['Energy'])]
                    my_energy_float = EnergyUnits * float(my_energy_string)

                    raise_stripped(stripped, step_all_tags.index(dict_step['Start Voltage']))
                    my_start_voltage_string = stripped[step_all_tags.index(dict_step['Start Voltage'])]
                    my_start_voltage_float = VoltageUnits * float(my_start_voltage_string)

                    raise_stripped(stripped, step_all_tags.index(dict_step['End Voltage']))
                    my_end_voltage_string = stripped[step_all_tags.index(dict_step['End Voltage'])]
                    my_end_voltage_float = VoltageUnits * float(my_end_voltage_string)

                    if my_cap_float == 0.0:
                        my_avg_voltage_float = 0.5 * (my_start_voltage_float + my_end_voltage_float)
                    else:
                        my_avg_voltage_float = my_energy_float / my_cap_float

                    raise_stripped(stripped, step_all_tags.index('Step ID'))
                    temp_step_id = int(stripped[step_all_tags.index('Step ID')])
                    previous_step = current_step
                    previous_step_cycle = previous_cycle
                    previous_cycle = current_cycle
                    current_step = temp_step_id
                    full_data_dict[current_cycle][current_step] = [my_name_string, my_cap_float, my_avg_voltage_float,
                                                                   min(my_start_voltage_float, my_end_voltage_float),
                                                                   max(my_start_voltage_float, my_end_voltage_float),
                                                                   my_time_float]

                    continue

                elif separated[record_id_position]:
                    if first_record_next:
                        if not previous_step == -1:

                            raise_stripped(stripped, record_all_tags.index('Realtime'))
                            my_realtime_string = stripped[record_all_tags.index('Realtime')]
                            new_start_time = from_string_to_time(my_realtime_string)
                            full_data_dict[previous_step_cycle][previous_step][5] = float(
                                (new_start_time - start_time).total_seconds()) / (60.0 * 60.0)
                            full_data_dict[previous_step_cycle][previous_step].extend(
                                [numpy.array(step_all_voltages), numpy.array(step_all_currents),
                                 numpy.array(step_all_capacities),
                                 numpy.array(step_all_times)])
                        else:
                            raise_stripped(stripped, record_all_tags.index('Realtime'))
                            my_realtime_string = stripped[record_all_tags.index('Realtime')]
                            first_start_time = from_string_to_time(my_realtime_string)

                        step_all_voltages = []
                        step_all_currents = []
                        step_all_capacities = []
                        step_all_times = []
                        # YYYY-MM-DD HH:MM:SS
                        raise_stripped(stripped, record_all_tags.index('Realtime'))
                        my_realtime_string = stripped[record_all_tags.index('Realtime')]
                        start_time = from_string_to_time(my_realtime_string)
                        first_record_next = False

                    raise_stripped(stripped, record_all_tags.index('Realtime'))
                    my_realtime_string = stripped[record_all_tags.index('Realtime')]
                    end_time = from_string_to_time(my_realtime_string)

                    raise_stripped(stripped, record_all_tags.index(dict_record['Cap']))
                    my_cap_string = stripped[record_all_tags.index(dict_record['Cap'])]
                    my_cap_float = CapacityUnits * float(my_cap_string)

                    raise_stripped(stripped, record_all_tags.index(dict_record['Cur']))
                    my_cur_string = stripped[record_all_tags.index(dict_record['Cur'])]
                    my_cur_float = CurrentUnits * float(my_cur_string)

                    raise_stripped(stripped, record_all_tags.index(dict_record['Vol']))
                    my_vol_string = stripped[record_all_tags.index(dict_record['Vol'])]
                    my_vol_float = VoltageUnits * float(my_vol_string)

                    step_all_capacities.append(my_cap_float)
                    step_all_voltages.append(my_vol_float)
                    step_all_currents.append(my_cur_float)

                    my_delta_time = float(
                        (start_time - first_start_time).total_seconds()) / (60.0 * 60.0)

                    raise_stripped(stripped, record_all_tags.index('Time'))
                    my_time_string = stripped[record_all_tags.index('Time')]
                    my_TimeUnits = TimeUnits
                    if my_TimeUnits == 'format_driven':
                        if is_time(my_time_string, '(H:M:S:ms)'):
                            my_TimeUnits = '(H:M:S:ms)'
                        if is_time(my_time_string, '(H:M:S.ms)'):
                            my_TimeUnits = '(H:M:S.ms)'
                        elif is_number(my_time_string):
                            my_TimeUnits = 'days'

                    if my_TimeUnits == '(H:M:S:ms)':
                        my_time_float = from_time(my_time_string, '(H:M:S:ms)')
                    elif my_TimeUnits == '(H:M:S.ms)':
                        my_time_float = from_time(my_time_string, '(H:M:S.ms)')
                    elif my_TimeUnits == 'days':
                        my_time_float = 24.0 * float(my_time_string)
                    else:
                        raise Exception('invalid TimeUnits {}, stripped {}, step_all_tags {}, my_time_string {}'.format(
                            my_TimeUnits, stripped, step_all_tags, my_time_string))

                    step_all_times.append(my_time_float + my_delta_time)

                    continue

        full_data_dict = copy.deepcopy(full_data_dict)

        return full_data_dict


def run_script_on(path_to_file, filename, args):
    error_message = {}
    new_file = filename.split('.txt')[0] + '_Imported.file'

    cache_path = os.path.join(args.common_path, args.path_to_cache, new_file)
    origin_path = os.path.join(path_to_file, filename)

    error_message['filepath'] = origin_path
    already_cached = os.path.isfile(cache_path)
    if not already_cached:
        error_message['cached'] = 'None'

    if already_cached:
        time_cached = os.path.getmtime(cache_path)
        time_origin = os.path.getmtime(origin_path)

        if time_origin > time_cached:
            already_cached = False
            error_message['cached'] = 'Stale'
        else:
            error_message['cached'] = 'Valid'

    if already_cached:
        error_message['error'] = False
        return error_message

    if args.DEBUG:
        # read neware

        if 'NEWARE40' in origin_path:
            error_message['mode'] = 'NEWARE40'
            data_table = read_neware(origin_path, cumulative_time=True)
        elif 'NEWARE308C' in origin_path or 'NEWARE323B' in origin_path:
            error_message['mode'] = 'NEWARE308C'
            data_table = read_neware(origin_path, EnergyUnits=1000.0)
        else:
            error_message['mode'] = 'else'
            data_table = read_neware(origin_path)

        with open(cache_path, 'wb') as f:
            pickle.dump(data_table, f, pickle.HIGHEST_PROTOCOL)

        error_message['error'] = False
        return error_message

    try:
        # read neware

        if 'NEWARE40' in origin_path:
            error_message['mode'] = 'NEWARE40'
            data_table = read_neware(origin_path, cumulative_time=True)
        elif 'NEWARE308C' in origin_path or 'NEWARE323B' in origin_path:
            error_message['mode'] = 'NEWARE308C'
            data_table = read_neware(origin_path, EnergyUnits=1000.0)
        else:
            error_message['mode'] = 'else'
            data_table = read_neware(origin_path)
    except Exception as e:
        error_message['error'] = True
        error_message['error type'] = 'ReadNeware'
        error_message['error verbatum'] = e
        return error_message
        pass

    try:
        # write
        with open(cache_path, 'wb') as f:
            pickle.dump(data_table, f, pickle.HIGHEST_PROTOCOL)

        error_message['error'] = False
        return error_message

    except:
        error_message['error'] = True
        error_message['error type'] = 'WriteCache'
        return error_message
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--single_thread', dest='single_thread', action='store_true')
    parser.add_argument('--no-single_thread', dest='single_thread', action='store_false')
    parser.set_defaults(single_thread=False)


    parser.add_argument('--DEBUG', type=bool, default=False)
    parser.add_argument('--path_to_file', required=True)
    parser.add_argument('--path_to_cache', required=True)
    parser.add_argument('--common_path', default='')
    parser.add_argument('--path_to_log', default='log_import')
    parser.add_argument('--debug_single_file', default='')
    parser.add_argument('--max_processing', type=int, default=128)
    parser.add_argument('--path_to_filter', default='')

    args = parser.parse_args()

    if len(args.path_to_filter) > 0:
        with open(args.path_to_filter,'rb') as f:
            barcodes = pickle.load(f)
            match_string = r'|'.join([r'({})'.format(b) for b in barcodes.keys()])


        all_path_filenames = []
        for root, dirs, filenames in os.walk(os.path.join(args.common_path, args.path_to_file)):
            for file in filenames:
                if file.endswith('.txt') and file.endswith(args.debug_single_file):
                    print(file)
                    if re.search(match_string, file):
                        all_path_filenames.append((root, file))

    else:
        all_path_filenames = []
        for root, dirs, filenames in os.walk(os.path.join(args.common_path, args.path_to_file)):
            for file in filenames:
                if file.endswith('.txt') and file.endswith(args.debug_single_file):
                    all_path_filenames.append((root, file))

    all_paths, all_filenames = zip(*all_path_filenames)
    data = list(zip(all_paths, all_filenames, repeat(args)))

    n = len(all_paths)
    # this might be too many files.
    # Chunk the list of data into lists of max_processing
    # Then, do all the processing and write to log after each chunk.
    # then, at the end, paste all the logs together.
    if args.max_processing*(1+ int(float(n) / float(args.max_processing))) < n:
        raise Exception('check math: might not process all the files. {} is < {}'.format(args.max_processing*(1+int(float(n) / float(args.max_processing))), n))
    for chunk_index in range(1 + int(float(n) / float(args.max_processing))):
        if args.single_thread:
            errors = list(starmap(run_script_on, data[args.max_processing*chunk_index: min(n, args.max_processing*(chunk_index+1))]))
        else:
            with Pool(processes=4) as pool:
                errors = pool.starmap(run_script_on, data[args.max_processing*chunk_index: min(n, args.max_processing*(chunk_index+1))], chunksize=4)

        non_null_errors = list(filter(lambda x: x['error'], errors))

        with open(os.path.join(args.common_path, args.path_to_log + '_{}'.format(chunk_index) + '.txt'), 'w') as f:
            if len(non_null_errors) > 0:
                for error in non_null_errors:
                    print(error, file=f)
