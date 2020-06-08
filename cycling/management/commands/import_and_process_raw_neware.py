from django.core.management.base import BaseCommand
from cycling.neware_processing_functions import *
from datetime import date
import csv

def import_and_process(args):
    path_to_log = None
    if args['log_dir'] is not None:
        path_to_log = os.path.join(
            args['log_dir'],
        )
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        today = date.today()
        today_string = today.strftime("%y_%m_%d")
        
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
        debug = args["DEBUG"],
        max_filesize= args["max_filesize"]
    )
    if len(errors) > 0:
        print("ERRORS IN BULK IMPORT: {}".format(errors))

        if path_to_log is not None:
            with open(os.path.join(path_to_log, "bulk_import_errors_{}.csv".format(today_string)),"w") as file:
                all_keys = []
                for error in errors:
                    all_keys.append(error.keys())
                all_keys = list(set(all_keys))
                writer = csv.DictWriter(file, fieldnames=all_keys)
                writer.writeheader()
                for error in errors:
                    writer.writerow(
                        {k: str(v) for k, v in error.items()}
                    )


    for _ in range(5):
        print()
    print("BULK PROCESS")
    errors = bulk_process(
        debug = args["DEBUG"],
        number_of_cycles_before_rate_analysis = args[
            "NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS"])
    if len(errors) > 0:
        print("ERRORS IN BULK PROCESS: {}".format(errors))
        if path_to_log is not None:
            with open(os.path.join(path_to_log, "bulk_process_errors_{}.csv".format(today_string)),"w") as file:
                all_keys = []
                for error in errors:
                    all_keys.append(error.keys())
                all_keys = list(set(all_keys))
                writer = csv.DictWriter(file, fieldnames=all_keys)
                writer.writeheader()
                for error in errors:
                    writer.writerow(
                        {k: str(v) for k, v in error.items()}
                    )


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("--DEBUG", dest = "DEBUG", action = "store_true")
        parser.add_argument(
            "--NO_DEBUG", dest = "DEBUG", action = "store_false",
        )
        parser.set_defaults(DEBUG = False)
        parser.add_argument(
            "--NUMBER_OF_CYCLES_BEFORE_RATE_ANALYSIS",
            type = int,
            default = 10,
        )
        parser.add_argument(
            "--max_filesize",
            type = int,
            default = 1000000000,
        )
        parser.add_argument('--log_dir', default='')

    def handle(self, *args, **options):
        import_and_process(options)
