from django.core.management.base import BaseCommand
from FileNameHelper.models import *
import os
import re

class Command(BaseCommand):
    def add_arguments(self, parser):
        print('')
        parser.add_argument('--data_dir', default="/srv/samba/share/DATA")

        parser.add_argument('--allow-small-barcodes', dest='allow_small_barcodes', action='store_true')
        parser.add_argument('--no-allow-small-barcodes', dest='allow_small_barcodes', action='store_false')
        parser.set_defaults(allow_small_barcodes=False)



    def handle(self, *args, **options):
        print('\n\n\n\n\n\nthe program starts')

        path_to_files = options['data_dir']

        count = 0
        max_count = 10
        for root, dirs, filenames in os.walk(path_to_files):
            print('visiting root {}, seeing dirs {}'.format(root, dirs))
            for file in filenames:
                if 'NEWARE308B' in root:
                    count += 1
                    if count > max_count:
                        raise Exception('Done')
                    print('')
                    print('file: ', file, 'root: ', root)
                    list_of_fields = file.split('_')

                    if len(list_of_fields) >3:

                        print('CharID: {}, barcode: {}'.format(list_of_fields[0], list_of_fields[2]))


                        # regular expressions. please explore!

                        matchObj1 = re.match(r'0(\d{4,6})',
                                            list_of_fields[2])

                        if options['allow_small_barcodes']:
                            matchObj2 = re.match(r'(\d{2,4})$',
                                                 list_of_fields[2])

                        else:
                            matchObj2 = re.match(r'(\d{5,5})$',
                                                 list_of_fields[2])


                        if matchObj1:
                            print('matched first pattern')
                            barcode = int(matchObj1.group(1))
                        elif matchObj2:
                            print('matched second pattern')
                            barcode = int(matchObj2.group(1))
                        else:
                            print('Invalid file')
                            continue

                        print('barcode: {}'.format(barcode))

                        matchObj1 = re.match(r'Nw',
                                             list_of_fields[3])

                        matchObj2 = re.match(r'Neware',
                                             list_of_fields[3])

                        if matchObj1:
                            print('matched first pattern')
                        elif matchObj2:
                            print('matched second pattern')
                        else:
                            print('Invalid file')
                            continue

                        matchObj1 = re.match(r'c(\d+)',
                                             list_of_fields[4])


                        if matchObj1:
                            print('matched first pattern')
                            start_cycle = int(matchObj1.group(1))
                        else:
                            print('Invalid file')
                            continue

                        print('start cycle: {}'.format(start_cycle))

                        print('this was a valid file')






'''
path_to_robot = os.path.join('.','Data', 'Robot')
            for root, dirs, filenames in os.walk(path_to_robot):
                for file in filenames:
                    if file.endswith('.asp'):
                        all_filenames.append(os.path.join(root, file))
                        '''


