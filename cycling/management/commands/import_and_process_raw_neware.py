from django.core.management.base import BaseCommand
from cycling.neware_processing_functions import *


def import_and_process(args):
    cell_ids = None
    for _ in range(5):
        print()
    print("BULK DEPRECATE")
    bulk_deprecate()
    for _ in range(5):
        print()
    print("BULK IMPORT(PARSE RAW FILE + PUSH RAW DATA)")
    errors = bulk_import(
        cell_ids = cell_ids,
        debug = args["DEBUG"])
    if len(errors) > 0:
        print("ERRORS IN BULK IMPORT: {}".format(errors))
    for _ in range(5):
        print()
    print("BULK PROCESS")
    errors = bulk_process(
        debug = args["DEBUG"],
        number_of_cycles_before_rate_analysis = args[
            "NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS"])
    if len(errors) > 0:
        print("ERRORS IN BULK PROCESS: {}".format(errors))


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("--DEBUG", dest = "DEBUG", action = "store_true")
        parser.add_argument(
            "--NO_DEBUG", dest = "DEBUG", action = "store_false",
        )
        parser.set_defaults(DEBUG = False)
        parser.add_argument(
            "--NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS", type = int, default = 10,
        )

    def handle(self, *args, **options):
        import_and_process(options)
