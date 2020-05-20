from django.core.management.base import BaseCommand
import os
from cell_database.models import *

def escape_string_to_path(my_string):
    for bad, good in [
        ('\"', 'QUOTE'),
        ('!', 'EXCLAMATION'),
        ('#', 'DASH'),
        ('$', 'DOLLAR'),
        ('&', 'AMPERSAND'),
        ("\'", 'HALFQUOTE'),
        ("(", "OPENP"),
        (")", "CLOSEP"),
        ("*", "STAR"),
        (',', 'COMMA'),
        (';', 'SEMICOLON'),
        ('<', 'LESSTHAN'),
        ('?', 'QUESTION'),
        ('[', 'OPENB'),
        (']', 'CLOSEB'),
        ('=', 'EQUAL'),
        ('\\', 'SLASH'),
        ('^', 'HAT'),
        ('`', 'BACKQUOTE'),
        ("\{", 'OPENC'),
        ("\}", 'CLOSEC'),
        ('|', 'PIPE'),
        ('~', 'TILDE'),
        ('.', "DOT"),
        (':', "COLON"),
        ('/', '_OVER_')

    ]:
        my_string = my_string.replace(bad, good)

    return my_string

#TODO(sam): for all datasets, for all cells, for all filters,
# print (dataset/cell_name_or_id/filter_name)
def output_files(options):
    for dataset in Dataset.objects.all():
        for wet_cell in dataset.wet_cells.order_by('cell_id'):
            for filt in DatasetSpecificFilters.objects.filter(dataset=dataset, wet_cell=wet_cell):
                wet_name, specified = wet_cell.get_specific_name_details(dataset)
                if not specified:
                    wet_name = str(wet_cell.cell_id)
                print(os.path.join(
                    options['output_dir'],
                    escape_string_to_path(dataset.name),
                    escape_string_to_path(wet_name),
                    escape_string_to_path(filt.name),
                )
                )



class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--output_dir', default='')


    def handle(self, *args, **options):
        output_files(options)
