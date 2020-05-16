from cycling.models import *
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
from scipy.interpolate import PchipInterpolator
import math
from background_task import background
from scipy import special

import numpy as np

halifax_timezone = pytz.timezone("America/Halifax")

import filename_database.models as filename_models


def get_good_neware_files():
    exp_type = filename_models.ExperimentType.objects.get(
        category = filename_models.Category.objects.get(name = "cycling"),
        subcategory = filename_models.SubCategory.objects.get(name = "neware"),
    )

    return filename_models.DatabaseFile.objects.filter(
        is_valid = True, deprecated = False,
    ).exclude(
        valid_metadata = None
    ).filter(valid_metadata__experiment_type = exp_type)


def get_cell_ids():
    exp_type = filename_models.ExperimentType.objects.get(
        category = filename_models.Category.objects.get(name = "cycling"),
        subcategory = filename_models.SubCategory.objects.get(name = "neware"),
    )

    all_current_cell_ids = filename_models.DatabaseFile.objects.filter(
        is_valid = True, deprecated = False,
    ).exclude(
        valid_metadata = None
    ).filter(
        valid_metadata__experiment_type = exp_type
    ).values_list(
        "valid_metadata__cell_id", flat = True,
    ).distinct()

    return all_current_cell_ids


def strip(string, sub):
    if string.endswith(sub):
        return strip(string[:-1], sub)
    else:
        return string


def parse_time(my_realtime_string):
    matchObj1 = re.match(
        r"(\d\d\d\d)-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})",
        my_realtime_string,
    )
    matchObj2 = re.match(
        r"(\d{1,2})/(\d{1,2})/(\d\d\d\d) (\d{1,2}):(\d{1,2}):(\d{1,2})",
        my_realtime_string,
    )
    matchObj3 = re.match(
        r"(\d{1,2})/(\d{1,2})/(\d\d\d\d) (\d{1,2}):(\d{1,2})",
        my_realtime_string,
    )

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
        raise Exception(
            "tried to parse time {}, but only known formats are YYYY-MM-DD hh:mm:ss, MM/DD/YYYY hh:mm:ss, MM/DD/YYYY hh:mm".format(
                my_realtime_string
            )
        )

    return datetime.datetime(
        int_year, int_month, int_day,
        hour = int_hour, minute = int_minute, second = int_second,
    ), second_accuracy


def identify_variable_position(separated, label, this_line):
    if label not in separated:
        raise Exception("This format is unknown! {}".format(this_line))
    return separated.index(label)


def test_occupied_position(separated, pos):
    if len(separated) <= pos:
        return False
    elif separated[pos]:
        return True
    else:
        return False


def read_neware(
    path, last_imported_cycle = -1, capacity_units = 1.0,
    voltage_units = 1.0 / 1000.0, current_units = 1.0,
):
    print("\tREADING FILE AS NEWARE FORMAT. {}".format(path))
    """
    TEMPDOC: 
    this is a fairly general function to process a neware file no matter which format it has been saved in.
    
    First, determine which format it is.
    Open the file, see if it is nested or separated.

    Then,
    if nested, parse the headers, then compile the data.
    else, go to Step data, parse that header, then compile all step data.
          then, go to Record data, parse that header, then compile all record data.

    """

    with open(path, "r", errors = "ignore") as my_file:
        this_line = my_file.readline()
        separated = this_line.split("\n")[0].split("\t")
        if len(separated) == 1:
            nested = False
            # print("NOT nested", separated)
        elif separated[0]:
            nested = True
            # print("nested", separated)
        else:
            raise Exception("This format is unknown. {}".format(this_line))

    with open(path, "r", errors = "ignore") as my_file:
        def get_header_line():
            this_line = my_file.readline()
            separated = [strip(h.split("(")[0], " ")
                         for h in this_line.split("\n")[0].split("\t")]
            stripped = [h for h in separated if h]
            return this_line, separated, stripped

        def get_normal_line():
            this_line = my_file.readline()
            separated = this_line.split("\n")[0].split("\t")
            stripped = [s for s in separated if s]
            return this_line, separated, stripped

        def parse_step_header(position, nested = True):
            # nested
            this_line, separated, stripped = get_header_line()

            # Verify that this is the step header
            if nested and separated[0]:
                raise Exception("This format is unknown! {}".format(this_line))

            if not nested:
                position["cycle_id"] = identify_variable_position(
                    separated, label = "Cycle ID", this_line = this_line,
                )
            position["step_id"] = identify_variable_position(
                separated, label = "Step ID", this_line = this_line,
            )

            if "Step Name" in separated:
                position["step_type"] = identify_variable_position(
                    stripped, label = "Step Name", this_line = this_line,
                )
            elif "Step Type" in separated:
                position["step_type"] = identify_variable_position(
                    stripped, label = "Step Type", this_line = this_line,
                )
            else:
                raise Exception("This format is unknown! {}".format(this_line))

        def parse_record_header(position, nested = True):
            this_line, separated, stripped = get_header_line()
            # Verify that this is the record header
            if nested and (separated[0] or separated[position["step_id"]]):
                raise Exception("This format is unknown! {}".format(this_line))

            if not nested:
                position["cycle_id"] = identify_variable_position(
                    separated, label = "Cycle ID", this_line = this_line,
                )
                position["step_id"] = identify_variable_position(
                    separated, label = "Step ID", this_line = this_line,
                )

            position["record_id"] = identify_variable_position(
                separated, label = "Record ID", this_line = this_line,
            )

            if "Vol" in stripped:
                iter_dat = [
                    ("voltage", "Vol"),
                    ("current", "Cur"),
                    ("capacity", "Cap"),
                    ("time", "Time"),
                    ("realtime", "Realtime"),
                ]
            elif "Voltage" in stripped:
                iter_dat = [
                    ("voltage", "Voltage"),
                    ("current", "Current"),
                    ("capacity", "Capacity"),
                    ("time", "Time"),
                    ("realtime", "Realtime"),
                ]

            for id, s in iter_dat:
                position[id] = identify_variable_position(
                    stripped,
                    label = s, this_line = this_line,
                )

        def parse_normal_step(
            position, current_cycle, current_step,
            imported_data, separated, stripped, nested,
        ):
            if nested:
                current_step = int(separated[position["step_id"]])
            else:
                if test_occupied_position(separated, position["cycle_id"]):
                    current_cycle = int(separated[position["cycle_id"]])
                else:
                    return current_cycle, current_step

                if (not nested) and (current_cycle <= last_imported_cycle):
                    return current_cycle, current_step

                if test_occupied_position(separated, position["step_id"]):
                    current_step = int(separated[position["step_id"]])
                else:
                    return current_cycle, current_step

                if not current_cycle in imported_data.keys():
                    imported_data[current_cycle] = collections.OrderedDict([])

            if test_occupied_position(stripped, position["step_type"]):
                step_type = stripped[position["step_type"]]
                imported_data[current_cycle][current_step] = (step_type, [])

            else:
                return current_cycle, current_step

            return current_cycle, current_step

        def parse_normal_record(
            position, current_cycle, current_step,
            imported_data, separated, stripped, nested,
        ):
            if not nested:
                if test_occupied_position(separated, position["cycle_id"]):
                    current_cycle = int(separated[position["cycle_id"]])
                else:
                    return

                if current_cycle <= last_imported_cycle:
                    return

                if test_occupied_position(separated, position["step_id"]):
                    current_step = int(separated[position["step_id"]])
                else:
                    return

            my_extracted_strings = {}

            for i in ["realtime", "capacity", "current", "voltage"]:

                if test_occupied_position(stripped, position[i]):
                    my_extracted_strings[i] = stripped[position[i]]
                else:
                    continue

            my_cap_float = capacity_units * float(
                my_extracted_strings["capacity"])
            my_cur_float = current_units * float(
                my_extracted_strings["current"])
            my_vol_float = voltage_units * float(
                my_extracted_strings["voltage"])
            my_time, second_accuracy = parse_time(
                my_extracted_strings["realtime"])

            imported_data[current_cycle][current_step][1].append([
                my_vol_float, my_cur_float, my_cap_float, my_time,
                second_accuracy,
            ])

        current_cycle = -1
        current_step = -1
        imported_data = collections.OrderedDict([])
        position = {}

        if nested:
            # Cycle
            this_line, separated, stripped = get_header_line()
            position["cycle_id"] = identify_variable_position(
                separated, label = "Cycle ID", this_line = this_line,
            )
            # Step
            parse_step_header(position, nested = nested)
            # Record
            parse_record_header(position, nested = nested)

            for i in range(1000000000):
                this_line, separated, stripped = get_normal_line()
                if this_line == "":
                    break

                if test_occupied_position(separated, position["cycle_id"]):
                    current_cycle = int(separated[position["cycle_id"]])
                    if current_cycle <= last_imported_cycle:
                        continue
                    imported_data[current_cycle] = collections.OrderedDict([])
                    if (current_cycle % 100) == 0:
                        print("\t\tREAD CYCLES UP TO {}".format(current_cycle))
                else:
                    if current_cycle <= last_imported_cycle:
                        continue

                    if test_occupied_position(separated, position["step_id"]):
                        current_cycle, current_step = parse_normal_step(
                            position, current_cycle, current_step,
                            imported_data, separated, stripped, nested,
                        )

                    elif test_occupied_position(separated,
                                                position["record_id"]):
                        parse_normal_record(
                            position, current_cycle, current_step,
                            imported_data, separated, stripped, nested,
                        )

                    else:
                        continue


        else:
            # not nested
            for i in range(10000000000):
                this_line, separated, stripped = get_normal_line()
                if separated[0] == "Step Data":
                    break

            # This is the step data header
            parse_step_header(position, nested = nested)

            # This is the step data
            for i in range(10000000000):
                this_line, separated, stripped = get_normal_line()
                if separated[0] == "Record Data":
                    break

                current_cycle, current_step = parse_normal_step(
                    position, current_cycle, current_step, imported_data,
                    separated, stripped, nested,
                )

            # This is the record data header
            parse_record_header(position, nested = nested)

            # This is the record data
            for i in range(10000000000):
                this_line, separated, stripped = get_normal_line()
                if this_line == "":
                    break

                parse_normal_record(
                    position, current_cycle, current_step, imported_data,
                    separated, stripped, nested,
                )

        return imported_data


def import_single_file(database_file, debug = False):
    """
    checks based on timestamps if any work is required
    calls read_neware
    then pushes the data to the database

    TODO(sam): the demarcation between neware-specific and cycling-general is not clear

    Args:
        database_file:
        debug:
    """
    print("IMPORTING FILE {}".format(database_file))
    time_of_running_script = timezone.now()
    error_message = {}
    if (
        not database_file.is_valid
        or database_file.deprecated
        or database_file.valid_metadata is None
    ):
        return error_message

    full_path = os.path.join(database_file.root, database_file.filename)

    error_message["filepath"] = full_path
    already_cached = CyclingFile.objects.filter(
        database_file = database_file
    ).exists()
    if not already_cached:
        error_message["cached"] = "None"

    if already_cached:
        f = CyclingFile.objects.get(database_file = database_file)
        time_origin = database_file.last_modified
        time_cached = f.import_time

        if time_origin > time_cached:
            already_cached = False
            error_message["cached"] = "Stale"
        else:
            error_message["cached"] = "Valid"

    if already_cached:
        error_message["error"] = False
        return error_message

    with transaction.atomic():
        f, created = CyclingFile.objects.get_or_create(
            database_file = database_file)

        def get_last_cycle():
            if f.cycle_set.exists():
                last_imported_cycle = f.cycle_set.aggregate(Max("cycle_number"))
                last_imported_cycle = last_imported_cycle["cycle_number__max"]
            else:
                last_imported_cycle = -1

            return last_imported_cycle

        last_imported_cycle = get_last_cycle()
        print("\tLAST CYCLE ALREADY IMPORTED: {}".format(last_imported_cycle))

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
                                    Cycle(cycling_file = f, cycle_number = cyc)
                                )

            Cycle.objects.bulk_create(cycles)
            steps = []
            for cyc in f.cycle_set.filter(
                cycle_number__gt = last_imported_cycle
            ).order_by("cycle_number"):
                cyc_steps = data_table[cyc.cycle_number]
                for step in cyc_steps.keys():
                    if len(cyc_steps[step][1]) == 0:
                        continue

                    step_type, data = cyc_steps[step]
                    start_time = min([d[3] for d in data])
                    second_accuracy = all([d[4] for d in data])

                    steps.append(
                        Step(
                            cycle = cyc,
                            step_number = step,
                            step_type = cyc_steps[step][0],
                            start_time = halifax_timezone.localize(start_time),
                            second_accuracy = second_accuracy,
                        )
                    )

                    steps[-1].set_v_c_q_t_data(np.array([
                        d[:3] + [
                            (d[3] - start_time).total_seconds() / (60. * 60.)
                        ]
                        for d in data
                    ]))

            Step.objects.bulk_create(steps)
            f.import_time = time_of_running_script
            f.save()

        if debug:
            data_table = read_neware(
                full_path, last_imported_cycle = last_imported_cycle,
            )
            write_to_database(data_table)

            error_message["error"] = False
            return error_message

        else:
            try:
                data_table = read_neware(
                    full_path, last_imported_cycle = last_imported_cycle,
                )

            except Exception as e:
                error_message["error"] = True
                error_message["error type"] = "ReadNeware"
                error_message["error verbatim"] = e
                return error_message

            try:
                write_to_database(data_table)

                error_message["error"] = False
                return error_message

            except:
                error_message["error"] = True
                error_message["error type"] = "WriteCache"
                return error_message


def bulk_import(cell_ids = None, debug = False):
    """
    mostly just calls import_single_file
    """
    if cell_ids is not None:
        neware_files = get_good_neware_files().filter(
            valid_metadata__cell_id__in = cell_ids)

    else:
        neware_files = get_good_neware_files()
    errors = list(map(lambda x: import_single_file(x, debug), neware_files))
    return list(filter(lambda x: x["error"], errors))


def is_monotonically_decreasing(qs):
    mono = True
    for i in range(1, len(qs) - 1):
        if qs[i] > qs[i - 1]:
            mono = False
            break
    return mono


def is_monotonically_increasing(qs, mask = None):
    mono = True
    for i in range(1, len(qs) - 1):
        if (
            qs[i] < qs[i - 1]
            and (mask is None or (mask[i] > .1 and mask[i - 1] > .1))
        ):
            mono = False
            break
    return mono


def average_data(
    data_source_, val_keys, sort_val,
    weight_func = None, weight_exp_func = None, compute_std = False,
):
    if weight_func is not None:
        weights, works = weight_func(data_source_)
    else:
        weights, works = np.ones(len(data_source_)), np.ones(
            len(data_source_), dtype = np.bool,
        )

    weights = weights[works]
    data_source = data_source_[works]
    if len(data_source) == 0:
        return None
    if weight_exp_func is not None:
        weights_exp = weight_exp_func(data_source)
    else:
        weights_exp = np.zeros(len(data_source))

    vals = data_source[sort_val]

    all_ = np.stack(
        [vals, weights, weights_exp] + [data_source[s_v] for s_v in val_keys],
        axis = 1,
    )
    all_ = np.sort(all_, axis = 0)
    if len(all_) >= 15:
        all_ = all_[2:-2]
    elif len(all_) >= 9:
        all_ = all_[1:-1]

    weights = all_[:, 1]
    weights_exp = all_[:, 2]
    vals = all_[:, 3:]
    max_weights_exp = np.max(weights_exp)
    actual_weights = weights * np.exp(
        weights_exp - max_weights_exp)
    if sum(actual_weights) == 0.0:
        actual_weights = None
    avg = np.average(vals, weights = actual_weights, axis = 0)
    if not compute_std:
        return {val_keys[i]: avg[i] for i in range(len(val_keys))}
    else:
        var = np.average(
            np.square(vals - avg), weights = actual_weights, axis = 0,
        )
        if actual_weights is not None:
            actual_weights = (1. / np.sum(
                actual_weights) * actual_weights) + 1e-10
            actual_weights = np.expand_dims(actual_weights, axis = 1)
            var = np.sum(
                actual_weights * np.square(vals - avg), axis = 0,
            ) / (1e-10 + np.abs(
                np.sum(actual_weights, axis = 0) - (
                    np.sum(np.square(actual_weights), axis = 0)
                    / (1e-10 + np.sum(actual_weights, axis = 0))
                )
            ))
        std = np.sqrt(var)
        return {val_keys[i]: (avg[i], std[i]) for i in range(len(val_keys))}


def default_deprecation(cell_id):
    """
    TEMPDOC(sam):
    this decides which of the duplicate files should be deprecated.
    """
    with transaction.atomic():
        files = get_good_neware_files().filter(
            valid_metadata__cell_id = cell_id)
        if files.count() == 0:
            return
        start_cycles = files.order_by(
            "valid_metadata__start_cycle").values_list(
            "valid_metadata__start_cycle", flat = True).distinct()
        for start_cycle in start_cycles:
            files_start = files.filter(
                valid_metadata__start_cycle = start_cycle)
            if files_start.count() <= 1:
                continue
            last_modified_max = files_start.aggregate(Max("last_modified"))[
                "last_modified__max"]

            filesize_max = files_start.aggregate(Max("filesize"))[
                "filesize__max"]

            # the winners are not deprecated
            # the winners are the simultaneously last_modified and largest files
            winners = files_start.filter(last_modified = last_modified_max,
                                         filesize = filesize_max)
            if winners.count() == 0:
                # no winners means no deprecation
                continue
            winner_id = winners[0].id

            files_start.exclude(id = winner_id).update(deprecated = True)


def process_cell_id(cell_id, NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS = 10):
    # TODO(sam): incorporate resting steps properly.
    print("\tPROCESSING CELL ID: {}".format(cell_id))
    with transaction.atomic():
        fs = get_files_for_cell_id(cell_id)
        for f in fs:
            steps = Step.objects.filter(cycle__cycling_file = f).order_by(
                "cycle__cycle_number", "step_number")
            if len(steps) == 0:
                continue
            first_step = steps[0]

            if "Rest" in first_step.step_type:
                first_step.end_current = 0.
                first_step.end_voltage = 0.

            elif "CCCV_" in first_step.step_type:
                sign = +1.
                if "DChg" in first_step.step_type:
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
            elif "CC_" in first_step.step_type:
                sign = +1.
                if "DChg" in first_step.step_type:
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

            first_step.save()

            if len(steps) == 1:
                continue
            for i in range(1, len(steps)):
                step = steps[i]
                if "Rest" in step.step_type:
                    step.end_current = steps[i - 1].end_current
                    step.end_voltage = steps[i - 1].end_voltage

                elif "CCCV_" in step.step_type:
                    sign = +1.
                    if "DChg" in step.step_type:
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

                elif "CC_" in step.step_type:
                    sign = +1.
                    if "DChg" in step.step_type:
                        sign = -1.
                    step.end_current_prev = steps[i - 1].end_current
                    step.end_voltage_prev = steps[i - 1].end_voltage
                    step.constant_current = sign * step.average_current_by_capacity
                    step.end_current = step.constant_current
                    if sign > 0:
                        step.end_voltage = step.maximum_voltage
                    else:
                        step.end_voltage = step.minimum_voltage

                step.save()

        CycleGroup.objects.filter(cell_id = cell_id).delete()

        files = get_good_neware_files().filter(
            valid_metadata__cell_id = cell_id)
        total_capacity =\
            Cycle.objects.filter(cycling_file__database_file__in = files,
                                 valid_cycle = True).aggregate(
                Max("dchg_total_capacity"))["dchg_total_capacity__max"]
        total_capacity = max(1e-10, total_capacity)

        # DISCHARGE
        for polarity in [CHARGE, DISCHARGE]:
            new_data = []
            for cyc in Cycle.objects.filter(
                cycling_file__database_file__in = files,
                valid_cycle = True).order_by("cycle_number"):

                if polarity == DISCHARGE:
                    step = cyc.get_first_discharge_step()
                elif polarity == CHARGE:
                    step = cyc.get_first_charge_step()
                else:
                    raise Exception("unknown polarity {}".format(polarity))

                # if step.end_current_prev is None:
                # print(polarity)
                # print([s.step_type for s in cyc.step_set.order_by("step_number")])
                # print([s.get_v_c_q_t_data() for s in cyc.step_set.order_by("step_number")])
                new_data.append(
                    (
                        cyc.id,  # id
                        math.log(1e-10 + abs(
                            step.constant_current) / total_capacity),
                        # constant current
                        math.log(
                            1e-10 + abs(step.end_current) / total_capacity),
                        # end current
                        math.log(1e-10 + abs(
                            step.end_current_prev) / total_capacity),
                        # end current prev
                        step.end_voltage,
                        step.end_voltage_prev,
                    )
                )

            if len(new_data) > NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS:
                new_data = np.array(
                    new_data, dtype = [
                        ("cycle_id", int),
                        ("constant_rate", "f4"),
                        ("end_rate", "f4"),
                        ("end_rate_prev", "f4"),
                        ("end_voltage", "f4"),
                        ("end_voltage_prev", "f4"),
                    ])

                def separate_data(data_table,
                                  splitting_var = "discharge_c_rate"):
                    rate_step_full = .1 * .5
                    rate_step_cut = .075 * .5
                    min_rate_full = np.min(data_table[splitting_var])
                    min_rate_full = round(
                        max(min_rate_full, math.log(.005)) - rate_step_full,
                        ndigits = 1)

                    max_rate_full = np.max(data_table[splitting_var])
                    max_rate_full = round(
                        min(max_rate_full, math.log(500.)) + rate_step_full,
                        ndigits = 1)

                    number_of_rate_steps = int(
                        (max_rate_full - min_rate_full) / rate_step_full) + 2

                    split_data = {}

                    prev_rate_mask = np.zeros(len(data_table),
                                              dtype = np.bool)
                    prev_avg_rate = 0
                    for i2 in range(number_of_rate_steps):
                        avg_rate = (rate_step_full * i2) + min_rate_full
                        min_rate = avg_rate - rate_step_cut
                        max_rate = avg_rate + rate_step_cut

                        rate_mask = np.logical_and(
                            min_rate <= data_table[splitting_var],
                            data_table[splitting_var] <= max_rate
                        )
                        if not np.any(rate_mask):
                            continue

                        intersection_mask = np.logical_and(rate_mask,
                                                           prev_rate_mask)

                        if np.array_equal(rate_mask,
                                          prev_rate_mask) or np.array_equal(
                            intersection_mask,
                            rate_mask):
                            prev_rate_mask = rate_mask
                            prev_avg_rate = avg_rate
                            continue

                        elif np.array_equal(intersection_mask,
                                            prev_rate_mask) and np.any(
                            prev_rate_mask):
                            if prev_avg_rate in split_data.keys():
                                del split_data[prev_avg_rate]

                        split_data[avg_rate] = rate_mask
                        prev_rate_mask = rate_mask
                        prev_avg_rate = avg_rate

                    sorted_keys = list(split_data.keys())
                    sorted_keys.sort()
                    avg_sorted_keys = {}
                    for sk in sorted_keys:
                        if len(data_table[split_data[sk]][splitting_var]) > 0:
                            avg_sorted_keys[sk] = np.mean(
                                data_table[split_data[sk]][splitting_var])

                    grouped_rates = {}
                    for sk in sorted_keys:
                        found = False
                        for k in grouped_rates.keys():
                            if abs(avg_sorted_keys[k] - avg_sorted_keys[
                                sk]) < 0.13:
                                grouped_rates[k].append(sk)
                                found = True
                                break
                        if not found:
                            grouped_rates[sk] = [sk]

                    split_data2 = {}

                    for k in grouped_rates.keys():
                        split_data2[k] = np.zeros(len(data_table),
                                                  dtype = np.bool)
                        for kk in grouped_rates[k]:
                            split_data2[k] = np.logical_or(split_data[kk],
                                                           split_data2[k])

                    return split_data2

                summary_data = {}

                split_data2 = separate_data(new_data,
                                            splitting_var = "constant_rate")
                for k in split_data2.keys():
                    new_data_2 = new_data[split_data2[k]]
                    if len(new_data_2) > 0:
                        split_data3 = separate_data(new_data_2,
                                                    splitting_var = "end_rate")
                        for k2 in split_data3.keys():
                            new_data_3 = new_data_2[split_data3[k2]]
                            if len(new_data_3) > 0:
                                split_data4 = separate_data(new_data_3,
                                                            splitting_var = "end_rate_prev")
                                for k3 in split_data4.keys():
                                    new_data_4 = new_data_3[split_data4[k3]]
                                    if len(new_data_4):
                                        split_data5 = separate_data(new_data_4,
                                                                    splitting_var = "end_voltage")
                                        for k4 in split_data5.keys():
                                            new_data_5 = new_data_4[
                                                split_data5[k4]]
                                            if len(new_data_5) > 0:
                                                split_data6 = separate_data(
                                                    new_data_5,
                                                    splitting_var = "end_voltage_prev")
                                                for k5 in split_data6.keys():
                                                    new_data_6 = new_data_5[
                                                        split_data6[k5]]
                                                    if len(new_data_6) > 0:
                                                        avg_constant_rate = np.mean(
                                                            new_data_6[
                                                                "constant_rate"])
                                                        avg_end_rate = np.mean(
                                                            new_data_6[
                                                                "end_rate"])
                                                        avg_end_rate_prev = np.mean(
                                                            new_data_6[
                                                                "end_rate_prev"])
                                                        avg_end_voltage = np.mean(
                                                            new_data_6[
                                                                "end_voltage"])
                                                        avg_end_voltage_prev = np.mean(
                                                            new_data_6[
                                                                "end_voltage_prev"])

                                                        summary_data[(
                                                            avg_constant_rate,
                                                            avg_end_rate,
                                                            avg_end_rate_prev,
                                                            avg_end_voltage,
                                                            avg_end_voltage_prev)] = new_data_6

                for k in summary_data.keys():
                    cyc_group = CycleGroup(cell_id = cell_id,
                                           constant_rate = math.exp(k[0]),
                                           end_rate = math.exp(k[1]),
                                           end_rate_prev = math.exp(k[2]),
                                           end_voltage = k[3],
                                           end_voltage_prev = k[4],
                                           polarity = polarity
                                           )

                    cyc_group.save()
                    if polarity == CHARGE:
                        Cycle.objects.filter(
                            id__in = list(summary_data[k]["cycle_id"])).update(
                            charge_group = cyc_group)
                    elif polarity == DISCHARGE:
                        Cycle.objects.filter(
                            id__in = list(summary_data[k]["cycle_id"])).update(
                            discharge_group = cyc_group)


def process_single_file(f, DEBUG = False):
    error_message = {"filename": f.database_file.filename}
    print("\tPROCESSING SINGLE FILE: {}".format(f.database_file.filename))

    def thing_to_try():
        with transaction.atomic():
            if f.process_time <= f.import_time:
                # must process the step data to summarize i
                for cyc in f.cycle_set.filter(processed = False):

                    for step in cyc.step_set.all():
                        dat = step.get_v_c_q_t_data()
                        v_min = np.nanmin(dat[:, 0])
                        v_max = np.nanmax(dat[:, 0])
                        cur_min = np.nanmin(dat[:, 1])
                        cur_max = np.nanmax(dat[:, 1])
                        delta_t = np.max(dat[:, -1]) - np.min(dat[:, -1])
                        capacity = np.max(dat[:, 2])
                        """
                        sum i -> n-1 : 0.5*(vol[i] + vol[i+1]) * (cap[i+1] - cap[i])
                        sum i -> n-1 : (cap[i+1] - cap[i])
                        """
                        if len(dat) > 1:
                            capacity_differences = np.absolute(
                                np.diff(dat[:, 2]))
                            voltage_differences = np.absolute(
                                np.diff(dat[:, 0]))

                            voltage_averages = .5 * (dat[:-1, 0] + dat[1:, 0])
                            current_averages = .5 * (dat[:-1, 1] + dat[1:, 1])
                            if np.nanmax(capacity_differences) > 0:
                                cur_avg_by_cap = np.average(current_averages,
                                                            weights = capacity_differences)
                                vol_avg_by_cap = np.average(voltage_averages,
                                                            weights = capacity_differences)
                            else:
                                cur_avg_by_cap = .5 * (cur_min + cur_max)
                                vol_avg_by_cap = .5 * (v_min + v_max)

                            if np.nanmax(voltage_differences) > 0:
                                cur_avg_by_vol = np.average(current_averages,
                                                            weights = voltage_differences)
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
                    discharge_query = Q(step_type__contains = "DChg")
                    charge_query = ~Q(step_type__contains = "DChg") & Q(
                        step_type__contains = "Chg")
                    discharge_data = np.array(
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
                        dtype = np.dtype(
                            [
                                ("total_capacity", float),
                                ("average_voltage", float),
                                ("minimum_voltage", float),
                                ("maximum_voltage", float),
                                ("average_current_by_capacity", float),
                                ("average_current_by_voltage", float),
                                ("minimum_current", float),
                                ("maximum_current", float),
                                ("duration", float),
                            ]
                        )
                    )
                    charge_data = np.array(
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
                        dtype = np.dtype(
                            [
                                ("total_capacity", float),
                                ("average_voltage", float),
                                ("minimum_voltage", float),
                                ("maximum_voltage", float),
                                ("average_current_by_capacity", float),
                                ("average_current_by_voltage", float),
                                ("minimum_current", float),
                                ("maximum_current", float),
                                ("duration", float),
                            ]
                        )
                    )

                    # charge agregation
                    if len(charge_data) != 0:
                        cyc.chg_total_capacity = np.sum(
                            charge_data["total_capacity"])
                        cyc.chg_duration = np.sum(charge_data["duration"])
                        cyc.chg_minimum_voltage = np.min(
                            charge_data["minimum_voltage"])
                        cyc.chg_maximum_voltage = np.max(
                            charge_data["maximum_voltage"])
                        cyc.chg_minimum_current = np.min(
                            charge_data["minimum_current"])
                        cyc.chg_maximum_current = np.max(
                            charge_data["maximum_current"])
                        cyc.chg_average_voltage = np.average(
                            charge_data["average_voltage"],
                            weights = 1e-6 + charge_data["total_capacity"]
                        )
                        cyc.chg_average_current_by_capacity = np.average(
                            charge_data["average_current_by_capacity"],
                            weights = 1e-6 + charge_data["total_capacity"]
                        )
                        cyc.chg_average_current_by_voltage = np.average(
                            charge_data["average_current_by_voltage"],
                            weights = 1e-6 + charge_data["maximum_voltage"] -
                                      charge_data["minimum_voltage"]
                        )
                    else:
                        cyc.delete()
                        continue

                    if len(discharge_data) != 0:
                        cyc.dchg_total_capacity = np.sum(
                            discharge_data["total_capacity"])
                        cyc.dchg_duration = np.sum(
                            discharge_data["duration"])
                        cyc.dchg_minimum_voltage = np.min(
                            discharge_data["minimum_voltage"])
                        cyc.dchg_maximum_voltage = np.max(
                            discharge_data["maximum_voltage"])
                        cyc.dchg_minimum_current = np.min(
                            discharge_data["minimum_current"])
                        cyc.dchg_maximum_current = np.max(
                            discharge_data["maximum_current"])
                        cyc.dchg_average_voltage = np.average(
                            discharge_data["average_voltage"],
                            weights = 1e-6 + discharge_data["total_capacity"]
                        )
                        cyc.dchg_average_current_by_capacity = np.average(
                            discharge_data["average_current_by_capacity"],
                            weights = 1e-6 + discharge_data["total_capacity"]
                        )
                        cyc.dchg_average_current_by_voltage = np.average(
                            discharge_data["average_current_by_voltage"],
                            weights = 1e-6 + discharge_data["maximum_voltage"] -
                                      discharge_data["minimum_voltage"]
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
            error_message["error"] = True
            error_message["error data"] = e
            error_message["error step"] = "processing raw"
            return error_message

    # after all processing.
    error_message["error"] = False
    return error_message


def detect_point(mus, sigma, x):
    return np.exp(-.5 * (sigma ** -2.) * np.square(x - mus))


def detect_line(mus, sigma, x_min, x_max):
    return 0.5 * (
        special.erf(1. / (sigma * np.sqrt(2.)) * (x_max - mus)) -
        special.erf(1. / (sigma * np.sqrt(2.)) * (x_min - mus))
    )


def detect_sign(signs, x):
    return np.where(
        x * signs > 0.,
        1.,
        0.
    )


def detect_step_cc(V_min, V_max, I, T, sign,
                   voltage_grid,
                   current_grid,
                   temperature_grid,
                   sign_grid):
    sign_dim = detect_sign(
        sign_grid,
        sign
    )

    V_dim = detect_line(
        voltage_grid,
        sigma = (voltage_grid[1] - voltage_grid[0]),
        x_min = V_min,
        x_max = V_max
    )

    I_dim = detect_point(
        current_grid,
        sigma = (current_grid[1] - current_grid[0]),
        x = I,
    )

    T_dim = detect_point(
        temperature_grid,
        sigma = (temperature_grid[1] - temperature_grid[0]),
        x = T,
    )

    total_detect = (
        np.reshape(sign_dim, [2, 1, 1, 1]) *
        np.reshape(V_dim, [1, -1, 1, 1]) *
        np.reshape(I_dim, [1, 1, -1, 1]) *
        np.reshape(T_dim, [1, 1, 1, -1])
    )
    return total_detect


def detect_step_cv(V, I_min, I_max, T, sign,
                   voltage_grid,
                   current_grid,
                   temperature_grid,
                   sign_grid):
    sign_dim = detect_sign(
        sign_grid,
        sign
    )

    V_dim = detect_point(
        voltage_grid,
        sigma = (voltage_grid[1] - voltage_grid[0]),
        x = V,
    )
    I_dim = detect_line(
        current_grid,
        sigma = (current_grid[1] - current_grid[0]),
        x_min = I_min,
        x_max = I_max
    )
    T_dim = detect_point(
        temperature_grid,
        sigma = (temperature_grid[1] - temperature_grid[0]),
        x = T,
    )

    total_detect = (
        np.reshape(sign_dim, [2, 1, 1, 1]) *
        np.reshape(V_dim, [1, -1, 1, 1]) *
        np.reshape(I_dim, [1, 1, -1, 1]) *
        np.reshape(T_dim, [1, 1, 1, -1])
    )
    return total_detect


def get_count_matrix(cyc, voltage_grid_degradation, current_grid,
                     temperature_grid, sign_grid):
    steps = cyc.step_set.order_by("step_number")
    total = None
    if len(steps) == 0:
        return total
    for step in steps:
        # TODO(sam):
        # compute the matrix for a step
        if "CC_DChg" in step.step_type:
            sign = -1.
            V_max = step.end_voltage_prev
            V_min = step.end_voltage
            I = current_to_log_current(step.constant_current)
            T = cyc.get_temperature()

            total_detect = detect_step_cc(V_min, V_max, I, T, sign,
                                          voltage_grid_degradation,
                                          current_grid, temperature_grid,
                                          sign_grid)


        elif "CC_Chg" in step.step_type:
            sign = 1.
            V_min = step.end_voltage_prev
            V_max = step.end_voltage
            I = current_to_log_current(step.constant_current)
            T = cyc.get_temperature()
            total_detect = detect_step_cc(V_min, V_max, I, T, sign,
                                          voltage_grid_degradation,
                                          current_grid,
                                          temperature_grid, sign_grid)


        elif "CCCV_Chg" in step.step_type:
            sign = 1.

            # CC part
            V_min = step.end_voltage_prev
            V_max = step.end_voltage
            I = current_to_log_current(step.constant_current)
            T = cyc.get_temperature()
            total_detect_0 = detect_step_cc(V_min, V_max, I, T, sign,
                                            voltage_grid_degradation,
                                            current_grid,
                                            temperature_grid, sign_grid)

            # CV part
            V = step.end_voltage
            I_max = current_to_log_current(step.constant_current)
            I_min = current_to_log_current(step.end_current)

            total_detect_1 = detect_step_cv(V, I_min, I_max, T, sign,
                                            voltage_grid_degradation,
                                            current_grid,
                                            temperature_grid, sign_grid)

            total_detect = total_detect_0 + total_detect_1

        else:
            continue

        if total is None:
            total = total_detect
        else:
            total += total_detect

    return total


def get_non_redundent_mask(mat):
    """
    this assumes mat is sorted
    :param mat:
    :return:
    """
    mask = np.ones(len(mat), dtype = np.bool)
    for i in range(1, len(mat)):
        if mat[i] == mat[i - 1]:
            mask[i] = False

    return mask


def reshuffle(x, y):
    sorted_ind = np.argsort(x)
    x = x[sorted_ind]
    y = y[sorted_ind]

    redundency_mask = get_non_redundent_mask(x)
    x = x[redundency_mask]
    y = y[redundency_mask]
    return x, y


def resampler(x, y, x_n, flagged = False, log_space = False,
              delta_grace = 0.001):
    x_min = np.min(x)
    x_max = np.max(x)

    if log_space:
        new_x = np.exp(
            np.linspace(np.log(x_min), np.log(x_max), x_n))
    else:
        new_x = np.linspace(x_min, x_max, x_n)

    spline = PchipInterpolator(x, y, extrapolate = True)
    new_y = spline(new_x)
    if flagged:
        print("new_x {}, x {}, y {}".format(
            new_x,
            x,
            y,
        )
        )

        print("spline: {}".format(new_y))

    new_mask = np.where(
        np.logical_and(
            new_x >= (x_min - delta_grace),
            new_x <= (x_max + delta_grace)
        ),
        np.ones(x_n, dtype = np.float32),
        0.0 * np.ones(x_n, dtype = np.float32),
    )

    mask = np.where(
        np.logical_and(
            new_x >= (x_min),
            new_x <= (x_max)
        ),
        np.ones(x_n, dtype = np.float32),
        0.0 * np.ones(x_n, dtype = np.float32),
    )
    if flagged:
        print("x_min {}, x_max {}, delta_grace {}, mask1 {}, mask {}".format(
            x_min,
            x_max,
            delta_grace,
            new_mask,
            mask
        ))

    if not is_monotonically_increasing(new_x, mask = mask):
        new_x[:] = x[:x_n]
        new_y[:] = y[:x_n]
        new_mask[:] = 1.
        if flagged:
            print("was not increasing")
    else:
        if flagged:
            print("was increasing")

    return new_x, new_y, new_mask


def ml_post_process_cycle(cyc, voltage_grid_n, step_type, current_max_n,
                          voltage_grid_min_v, voltage_grid_max_v,
                          flagged = False):
    if flagged:
        print(
            "Starting Verbose ml_post_process_cycle(cyc={}, voltage_grid_n={}, step_type={}, current_max_n={}, voltage_grid_min_v={}, voltage_grid_max_v={}".format(
                cyc,
                voltage_grid_n,
                step_type,
                current_max_n,
                voltage_grid_min_v,
                voltage_grid_max_v
            ))

    if step_type == "dchg":
        steps = cyc.step_set.filter(step_type__contains = "CC_DChg").order_by(
            "cycle__cycle_number", "step_number")

        if len(steps) == 0:
            if flagged:
                print("return because len(steps) == 0")
            return None
        else:
            first_step = steps[0]
            vcqt_curve = first_step.get_v_c_q_t_data()
            curve = vcqt_curve[:, [0, 2]]
            curve[:, 1] = -1 * curve[:, 1]
            curve = np.flip(curve, 0)
            cv_curve = []

    if step_type == "chg":
        steps = cyc.step_set.filter(step_type__contains = "CC_Chg").order_by(
            "cycle__cycle_number", "step_number")
        if len(steps) == 0:
            if flagged:
                print("len(steps) was 0 first time.")
            steps = cyc.step_set.filter(
                step_type__contains = "CCCV_Chg").order_by(
                "cycle__cycle_number", "step_number")
            if len(steps) == 0:
                if flagged:
                    print("return because len(steps) was 0 second time")
                return None
            else:
                # has some CV data
                first_step = steps[0]
                vcqt_curve = first_step.get_v_c_q_t_data()

                if len(vcqt_curve) == 1:
                    if flagged:
                        print("len(vcqt_curve) was 0")
                    curve = vcqt_curve[:, [0, 2]]
                    cv_curve = []
                else:
                    if flagged:
                        print("len(vcqt_curve) was not 0")
                    delta_currents = np.abs(
                        vcqt_curve[1:, 1] - vcqt_curve[:-1, 1])
                    delta_count = 0
                    for d in delta_currents:
                        if d < 5:
                            delta_count += 1
                        else:
                            break

                    if flagged:
                        print('delta_currents = {}, delta_count = {}'.format(
                            delta_currents, delta_count))
                    curve = vcqt_curve[:delta_count + 1, [0, 2]]
                    cv_curve = vcqt_curve[delta_count + 1:, [1, 2]]


        else:
            first_step = steps[0]
            vcqt_curve = first_step.get_v_c_q_t_data()
            curve = vcqt_curve[:, [0, 2]]
            cv_curve = []

    within_bounds = np.logical_and(
        voltage_grid_min_v <= curve[:, 0],
        curve[:, 0] <= voltage_grid_max_v
    )

    curve = curve[within_bounds, :]

    if flagged:
        print(
            "first_step = {}, vcqt_curve = {}, curve = {}, cv_curve = {}".format(
                first_step,
                vcqt_curve,
                curve,
                cv_curve
            ))

    cursor = np.array([-1, len(curve)], dtype = np.int32)
    limits_v = np.array([-10., 10.], dtype = np.float32)
    limits_q = np.array([-1e6, 1e6], dtype = np.float32)
    masks = np.zeros(len(curve), dtype = np.bool)

    never_added_up = True
    never_added_down = True
    if len(curve) < 3:
        print("curve too short: {}".format(curve))
        return None

    while True:
        if cursor[0] + 1 >= cursor[1]:
            break

        can_add_upward = False
        delta_upward = 0.
        can_add_downward = False
        delta_downward = 0.

        # Test upward addition (addition from the left)
        new_index = cursor[0] + 1
        new_v = curve[
            new_index, 0]  # should be more than limits_v[0] and less than limits_v[1]
        new_q = curve[
            new_index, 1]  # should be more than limits_q[0] and less than limits_q[1]

        # should both be positive
        dv1 = new_v - limits_v[0]
        dv2 = limits_v[1] - new_v

        # should both be positive (with some tolerance)
        dq1 = new_q - limits_q[0]
        dq2 = limits_q[1] - new_q

        if dv1 >= 0.0001 and dq1 >= -0.001 and dv2 >= 0.0001 and dq2 >= -0.001:

            delta_upward = dv1 + dq1 / 100.
            if dq1 < 0.0001:
                dq1 = 0.0001
            delta_upward += abs(dv1) / dq1
            if abs(dv1) / dq1 < 10.:
                if never_added_up:
                    dv1 = curve[new_index + 1, 0] - new_v
                    dq1 = curve[new_index + 1, 1] - new_q
                    if dv1 >= -0.001 and dq1 >= -0.001:
                        can_add_upward = True
                else:
                    can_add_upward = True

        # Test downward addition

        new_index = cursor[1] - 1
        new_v = curve[
            new_index, 0]  # should be more than limits_v[0] and less than limits_v[1]
        new_q = curve[
            new_index, 1]  # should be more than limits_q[0] and less than limits_q[1]

        # should both be positive
        dv1 = new_v - limits_v[0]
        dv2 = limits_v[1] - new_v

        # should both be positive (with some tolerance)
        dq1 = new_q - limits_q[0]
        dq2 = limits_q[1] - new_q

        if dv1 >= 0.0001 and dq1 >= -0.001 and dv2 >= 0.0001 and dq2 >= -0.001:

            delta_downward = dv2 + dq2 / 100.

            if dq2 < 0.0001:
                dq2 = 0.0001
            delta_upward += abs(dv2) / dq2
            if abs(dv2) / dq2 < 10.:
                if never_added_down:
                    dv2 = new_v - curve[new_index - 1, 0]
                    dq2 = new_q - curve[new_index - 1, 1]
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
                my_add = "Down"
            else:
                my_add = "Up"

        elif can_add_upward and not can_add_downward:
            my_add = "Up"
        elif not can_add_upward and can_add_downward:
            my_add = "Down"

        if my_add == "Up":
            new_index = cursor[0] + 1
            new_v = curve[new_index, 0]
            new_q = curve[new_index, 1]

            never_added_up = False
            masks[new_index] = True
            limits_v[0] = max(limits_v[0], new_v)
            limits_q[0] = max(limits_q[0], new_q)
            cursor[0] += 1

        elif my_add == "Down":
            new_index = cursor[1] - 1
            new_v = curve[new_index, 0]
            new_q = curve[new_index, 1]

            never_added_down = False
            masks[new_index] = True
            limits_v[1] = min(limits_v[1], new_v)
            limits_q[1] = min(limits_q[1], new_q)

            cursor[1] += -1

    valid_curve = curve[masks]
    invalid_curve = curve[~masks]

    if flagged:
        print("valid_curve {}, invalid_curve {}".format(valid_curve,
                                                        invalid_curve))
    if len(invalid_curve) > 20:
        if flagged:
            print("too many invalids {}. (valids were {})".format(invalid_curve,
                                                                  valid_curve))
        return None

    if len(valid_curve) == 0:
        if flagged:
            print("not enough valids. curve was {}".format(curve))
        return None

    # uniformly sample it

    v, q = reshuffle(
        x = valid_curve[:, 0],
        y = valid_curve[:, 1],
    )

    if flagged:
        print("sorted: v: {}, q: {}".format(v, q))

    if step_type == "dchg":

        last_cc_voltage = v[0]
        last_cc_capacity = q[0]
    else:
        last_cc_voltage = v[-1]
        last_cc_capacity = q[-1]

    if flagged:
        print("last_cc_voltage {}, last_cc_capacity {}".format(last_cc_voltage,
                                                               last_cc_capacity))

    if len(cv_curve) > 0:
        last_cv_capacity = cv_curve[-1, 1]
    else:
        last_cv_capacity = last_cc_capacity

    if flagged:
        print("last_cv_capacity {}".format(last_cv_capacity))

    cv_currents = np.zeros(shape = (current_max_n), dtype = np.float32)
    cv_qs = np.zeros(shape = (current_max_n), dtype = np.float32)
    cv_mask = np.zeros(shape = (current_max_n), dtype = np.float32)

    if len(cv_curve) > 0:
        if current_max_n >= len(cv_curve):
            cv_currents[:len(cv_curve)] = cv_curve[:, 0]
            cv_qs[:len(cv_curve)] = cv_curve[:, 1]
            cv_mask[:len(cv_curve)] = 1.

        else:
            # uniformly sample it
            cv_x, cv_y = reshuffle(
                x = cv_curve[:, 0],
                y = cv_curve[:, 1],
            )

            cv_currents, cv_qs, cv_mask = resampler(
                x = cv_x,
                y = cv_y,
                x_n = current_max_n,
                flagged = flagged,
                log_space = True
            )

    if voltage_grid_n >= len(v):
        if flagged:
            print("len(voltage_grid) >= len(v)")
        voltages = np.zeros(shape = (voltage_grid_n), dtype = np.float32)
        qs = np.zeros(shape = (voltage_grid_n), dtype = np.float32)
        mask = np.zeros(shape = (voltage_grid_n), dtype = np.float32)

        voltages[:len(v)] = v[:]
        qs[:len(v)] = q[:]
        mask[:len(v)] = 1.0
        if flagged:
            print("voltages {}, qs {}, mask {}".format(voltages, qs, mask))

        return {
            "cc_voltages": voltages, "cc_capacities": qs, "cc_masks": mask,
            "cv_currents": cv_currents, "cv_capacities": cv_qs,
            "cv_masks": cv_mask,
            "end_current_prev": first_step.end_current_prev,
            "end_current": first_step.end_current,

            "end_voltage_prev": first_step.end_voltage_prev,
            "end_voltage": first_step.end_voltage,

            "constant_current": first_step.constant_current,
            "last_cc_voltage": last_cc_voltage,
            "last_cc_capacity": last_cc_capacity,
            "last_cv_capacity": last_cv_capacity,
        }
    if flagged:
        print("len(voltage_grid) < len(v)")

    voltages, cc_capacities, cc_masks = resampler(
        x = v,
        y = q,
        x_n = voltage_grid_n,
        flagged = flagged,
        log_space = False
    )

    # TODO(sam): treat and include the CV data as well.
    return {
        "cc_voltages": voltages, "cc_capacities": cc_capacities,
        "cc_masks": cc_masks,
        "cv_currents": cv_currents, "cv_capacities": cv_qs, "cv_masks": cv_mask,
        "end_current_prev": first_step.end_current_prev,
        "end_current": first_step.end_current,

        "end_voltage_prev": first_step.end_voltage_prev,
        "end_voltage": first_step.end_voltage,
        "constant_current": first_step.constant_current,
        "last_cc_voltage": last_cc_voltage,
        "last_cc_capacity": last_cc_capacity,
        "last_cv_capacity": last_cv_capacity,
    }


def bulk_process(
    debug = False, number_of_cycles_before_rate_analysis = 10, cell_ids = None,
):
    """
    TEMPDOC
    first calls process_single_file then, calls process_cell_id
    """
    if cell_ids is None:
        errors = list(
            map(
                lambda x: process_single_file(x, debug),
                CyclingFile.objects.filter(
                    database_file__deprecated = False,
                    process_time__lte = F("import_time"),
                )
            )
        )
        all_current_cell_ids = CyclingFile.objects.filter(
            database_file__deprecated = False
        ).values_list(
            "database_file__valid_metadata__cell_id", flat = True,
        ).distinct()

    else:
        errors = list(
            map(
                lambda x: process_single_file(x, debug),
                CyclingFile.objects.filter(
                    database_file__deprecated = False,
                    database_file__valid_metadata__cell_id__in = cell_ids,
                    process_time__lte = F("import_time"),
                )
            )
        )
        all_current_cell_ids = cell_ids

    for cell_id in all_current_cell_ids:
        process_cell_id(cell_id, number_of_cycles_before_rate_analysis)

    return list(filter(lambda x: x["error"], errors))


def bulk_deprecate(cell_ids = None):
    if cell_ids is None:
        all_current_cell_ids = get_good_neware_files().values_list(
            "valid_metadata__cell_id", flat = True,
        ).distinct()
    else:
        all_current_cell_ids = cell_ids

    for cell_id in all_current_cell_ids:
        print("\tDEFAULT DEPRECATION OF CELL ID:{}".format(cell_id))
        default_deprecation(cell_id)


@background(schedule = 5)
def full_import_cell_ids(cell_ids):
    """
    TEMPDOC(sam):
    this gets called when asking to trigger a reimport
    TODO(sam): does it create CycleGroups?
    """
    # TODO(sam): split into smaller batches?
    with transaction.atomic():
        # TODO what does deprecate do?
        bulk_deprecate(cell_ids)
        # TODO what does import do?
        bulk_import(cell_ids = cell_ids, debug = False)
        # TODO what does process do?
        bulk_process(debug = False, cell_ids = cell_ids)
