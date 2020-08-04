from django.core.management.base import BaseCommand
from cell_database.dataset_visualization import *

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--output_dir', default='')

        parser.add_argument('--visuals', dest='visuals', action='store_true')
        parser.add_argument('--no-visuals', dest='visuals', action='store_false')
        parser.set_defaults(visuals=False)

    def handle(self, *args, **options):
        output_files.now(options)
