from django.core.management.base import BaseCommand
import os
from cell_database.models import *
import csv
def escape_string_to_path(my_string):
    for bad, good in [
        ('\"', '_QUOTE_'),
        ('!', '_EXCLAMATION_'),
        ('#', '_DASH_'),
        ('$', '_DOLLAR_'),
        ('&', '_AMPERSAND_'),
        ("\'", '_HALFQUOTE_'),
        ("(", "_OPENP_"),
        (")", "_CLOSEP_"),
        ("*", "_STAR_"),
        (',', '_COMMA_'),
        (';', '_SEMICOLON_'),
        ('<', '_LESSTHAN_'),
        ('?', '_QUESTION_'),
        ('[', '_OPENB_'),
        (']', '_CLOSEB_'),
        ('=', '_EQUAL_'),
        ('\\', '_SLASH_'),
        ('^', '_HAT_'),
        ('`', '_BACKQUOTE_'),
        ("\{", '_OPENC_'),
        ("\}", '_CLOSEC_'),
        ('|', '_PIPE_'),
        ('~', '_TILDE_'),
        ('.', "_DOT_"),
        (':', "_COLON_"),
        ('/', '_OVER_'),
        # (' ', "_"),

    ]:
        my_string = my_string.replace(bad, good)

    return my_string

from cycling.models import *



#TODO(sam): find better names for the functions that extract data in a plottable format
def compute_dataset(dataset, field_request):
    results = {}
    for wet_cell in dataset.wet_cells.order_by('cell_id'):
        rules = {}
        for filt in DatasetSpecificFilters.objects.filter(dataset=dataset, wet_cell=wet_cell):
            rules[filt.id] = filt.get_rule()
        results[wet_cell.cell_id] = compute_from_database2(wet_cell.cell_id, rules, field_request)
    return results


def get_dataset_labels(dataset):
    results = {}
    dataset_name = dataset.name
    wet_names = {}
    for wet_cell in dataset.wet_cells.order_by('cell_id'):
        wet_name, specified = wet_cell.get_specific_name_details(dataset)
        if not specified:
            wet_name = str(wet_cell.cell_id)
        wet_names[wet_cell.cell_id] = wet_name
        names = {}
        for filt in DatasetSpecificFilters.objects.filter(dataset=dataset, wet_cell=wet_cell):
            names[filt.id] = filt.name
        results[wet_cell.cell_id] = names
    return dataset_name, wet_names, results



def output_dataset_to_csv(data, dataset_name, wet_names, filt_names, csv_format, output_dir):
    for cell_id in data.keys():
        for filt_id in data[cell_id].keys():
            path = os.path.join(
                output_dir,
                escape_string_to_path(dataset_name),
                escape_string_to_path(wet_names[cell_id])
            )

            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, '{}.csv'.format(escape_string_to_path(filt_names[cell_id][filt_id]))), 'w', newline='') as csv_f:
                writer = csv.writer(csv_f)
                header = [h for _, _, h in csv_format]
                writer.writerow(header)
                content = [[f(res[key]) for key, f, _ in csv_format] for res in data[cell_id][filt_id]]
                writer.writerows(content)



def output_dataset_to_csv(dataset, csv_format, field_request, output_dir):
    data = compute_dataset(dataset, field_request)
    dataset_name, wet_names, filt_names = get_dataset_labels(dataset)
    output_dataset_to_csv(data, dataset_name, wet_names, filt_names, csv_format, output_dir)

def output_files(options):
    '''
    Now:
    "Cycle Number",
     "Charge Capacity (mAh)",
     "Discharge Capacity (mAh)",

     "Average Charge Voltage (V)",
     "Average Discharge Voltage (V)",
     "Delta V (V)",

     "Charge Time (hours)",
    "Discharge Time (hours)", \
    "Cumulative Time (hours)",

    Later:
    "Normalized Charge Capacity",
    "Normalized Discharge Capacity", "Zeroed Delta V (V)", "S (V)", "R (V)",
    "Zeroed S (V)", "Zeroed R (V)"
    '''
    csv_format=[
        (Key.N, lambda x: str(int(x)), "Cycle Number"),
        ("total_charge_capacity", lambda x: "{:.2f}".format(x), "Total Charge Capacity (mAh)"),
        ("total_discharge_capacity", lambda x: "{:.2f}".format(x), "Total Discharge Capacity (mAh)"),

        ("avg_charge_voltage", lambda x: "{:.2f}".format(x), "Average Charge Voltage (V)"),
        ("avg_discharge_voltage", lambda x: "{:.2f}".format(x), "Average Discharge Voltage (V)"),
        ("delta_voltage", lambda x: "{:.2f}".format(x), "Delta Voltage (V)"),

        ("charge_time", lambda x: "{:.2f}".format(x), "Charge Time (hours)"),
        ("discharge_time", lambda x: "{:.2f}".format(x), "Discharge Time (hours)"),
        ("cumulative_time", lambda x: str(int(x)), "Cumulative Time (hours)"),

    ]

    field_request = [
        (Key.N, 'f4', "CYCLE_NUMBER", None),
        ("total_charge_capacity", 'f4', "CUSTOM", lambda cyc: cyc.chg_total_capacity),
        ("total_discharge_capacity", 'f4', "CUSTOM", lambda cyc: cyc.dchg_total_capacity),

        ("avg_charge_voltage", 'f4', "CUSTOM", lambda cyc: cyc.chg_average_voltage),
        ("avg_discharge_voltage", 'f4', "CUSTOM", lambda cyc: cyc.dchg_average_voltage),
        ("delta_voltage", 'f4', "CUSTOM", lambda cyc: cyc.chg_average_voltage-cyc.dchg_average_voltage),

        ("charge_time", 'f4', "CUSTOM", lambda cyc: cyc.chg_duration),
        ("discharge_time", 'f4', "CUSTOM", lambda cyc: cyc.dchg_duration),
        ("cumulative_time", 'f4', "CUMULATIVE_TIME", None),

    ]

    for dataset in Dataset.objects.all():
        # output dataset to csv
        output_dataset_to_csv(dataset, csv_format, field_request, options['output_dir'])


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--output_dir', default='')


    def handle(self, *args, **options):
        output_files(options)
