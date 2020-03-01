from django.core.management.base import BaseCommand

from neware_parser.neware_processing_functions import *


def import_and_process(args):
    if len(args['path_to_filter']) > 0:
        with open(args['path_to_filter'], 'rb') as f:
            barcodes = pickle.load(f)

    else:
        barcodes = None

    # bulk_deprecate()
    # print(bulk_import(
    #     barcodes=barcodes,
    #     DEBUG=args['DEBUG']))
    print(bulk_process(
        DEBUG = args['DEBUG'],
        NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS = args[
            'NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS']))


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--DEBUG', dest = 'DEBUG', action = 'store_true')
        parser.add_argument('--NO_DEBUG', dest = 'DEBUG',
                            action = 'store_false')
        parser.set_defaults(DEBUG = False)
        parser.add_argument('--NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS',
                            type = int, default = 10)
        parser.add_argument('--common_path', default = '')
        parser.add_argument('--path_to_filter', default = '')

    def handle(self, *args, **options):
        import_and_process(options)
