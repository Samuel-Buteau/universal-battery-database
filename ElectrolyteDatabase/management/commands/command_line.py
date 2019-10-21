from django.core.management.base import BaseCommand
import os
from ElectrolyteDatabase.read_electrolyte_csv import *
from ElectrolyteDatabase.read_dry_cell_csv import *
from ElectrolyteDatabase.models import *
from ElectrolyteDatabase.matplotlib_tutorial import *
from ElectrolyteDatabase.read_dates_table import *

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--mode', choices=[
                                            'read_electrolyte_csv',
                                            'generate_shortstrings',
                                            'read_dry_cells',
                                            'read_sql_cell_info',
                                            'read_electrolytes_and_dry_cells'
                                               ])

    def handle(self, *args, **options):

        if options['mode'] == 'read_electrolyte_csv':

            electrolyte_csv_to_db('electrolyte_excel.csv')


        if options['mode'] == 'generate_shortstrings':

            for electrolyte in Electrolyte.objects.all():

                if electrolyte.shortstring is None:

                    electrolyte.shortstring = electrolyte.generate_shortstring

                    electrolyte.save()

        if options['mode'] == 'read_dry_cells':
            dry_cell_csv_to_db('dry_cells_good.csv')


        if options['mode'] == 'read_sql_cell_info':

            read_sql_cell_table('model-box_id_python.csv','box_id-cell_id-electrolyte_python.csv')



        if options['mode'] == 'read_electrolytes_and_dry_cells':
            electrolyte_csv_to_db('electrolyte_excel.csv')
            dry_cell_csv_to_db('dry_cells_good.csv')
