from django.core.management.base import BaseCommand
import os
from cell_database.models import *

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
        ('/', '_OVER_')

    ]:
        my_string = my_string.replace(bad, good)

    return my_string

from cycling.models import *
#TODO(sam): call compute_from_database2

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
