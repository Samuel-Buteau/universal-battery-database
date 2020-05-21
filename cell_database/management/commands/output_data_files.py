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
        for wet_cell in dataset.wet_cells.order_by('cell_id'):
            rules = {}
            file_paths = {}
            for filt in DatasetSpecificFilters.objects.filter(dataset=dataset, wet_cell=wet_cell):
                wet_name, specified = wet_cell.get_specific_name_details(dataset)
                if not specified:
                    wet_name = str(wet_cell.cell_id)
                file_paths[filt.id] = (os.path.join(
                        options['output_dir'],
                        escape_string_to_path(dataset.name),
                        escape_string_to_path(wet_name),
                    ),
                    '{}.csv'.format(escape_string_to_path(filt.name)),
                )
                rules[filt.id] = filt.get_rule()

            results = compute_from_database2(wet_cell.cell_id, rules, field_request)
            for id in results.keys():
                if not os.path.exists(file_paths[id][0]):
                    os.makedirs(file_paths[id][0])
                with open(os.path.join(file_paths[id][0], file_paths[id][1]), 'w', newline='') as csv_f:
                    writer = csv.writer(csv_f)
                    header = [h for _, _, h in csv_format]
                    writer.writerow(header)
                    content = [[f(res[key]) for key, f, _ in csv_format] for res in results[id]]
                    writer.writerows(content)



class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--output_dir', default='')


    def handle(self, *args, **options):
        output_files(options)
