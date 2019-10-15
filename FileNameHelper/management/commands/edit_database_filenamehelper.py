from django.core.management.base import BaseCommand
from FileNameHelper.models import *
import os
from FileNameHelper.parsing_functions import *
import pytz
halifax_timezone = pytz.timezone("America/Halifax")

models_dict = {
            "ValidMetadata": ValidMetadata,
            "ChargerDriveProfile":ChargerDriveProfile,
            "ExperimentType":ExperimentType,
            'DatabaseFile':DatabaseFile,
            'Category':Category,
            'Token':Token,
            }

def get_drive_profiles():
    drive_profiles = {

        'CXCX':ChargerDriveProfile.objects.get(drive_profile='CXCX'),
        'CXCY':ChargerDriveProfile.objects.get(drive_profile='CXCY'),
        'CX':ChargerDriveProfile.objects.get(drive_profile='CX'),
        'CXCYc':ChargerDriveProfile.objects.get(drive_profile='CXCYc'),
        'CXrc':ChargerDriveProfile.objects.get(drive_profile='CXrc'),
        'CXCYb':ChargerDriveProfile.objects.get(drive_profile='CXCYb'),
        'CXsZZZ':ChargerDriveProfile.objects.get(drive_profile='CXsZZZ'),
        'None':None

    }
    return drive_profiles


def add_files(options):
    path_to_files = options['data_dir']
    file_list = []

    for root, dirs, filenames in os.walk(path_to_files):

        for name in filenames:
            full_path = os.path.join(root, name)
            time_origin = halifax_timezone.localize(
                datetime.datetime.fromtimestamp(os.path.getmtime(full_path)))
            filesize = os.path.getsize(full_path)
            file_list.append({'root': root, 'filename': name, 'last_modified':time_origin, 'filesize':filesize})

            print('ROOT= {}, FILE= {}, TIME= {}, SIZE= {}'.format(root, name, time_origin, filesize))

    already_created_data = []
    not_created_data = []
    for data in file_list:
        if DatabaseFile.objects.filter(filename=data['filename'], root=data['root']).exists():
            my_file = DatabaseFile.objects.get(filename=data['filename'], root=data['root'])
            if  not my_file.last_modified == data['last_modified'] or not my_file.filesize == data['filesize']:
                my_file.last_modified = data['last_modified']
                my_file.filesize = data['filesize']
                already_created_data.append(
                    my_file
                )
        else:
            not_created_data.append(DatabaseFile(
                filename=data['filename'],
                root=data['root'],
                last_modified= data['last_modified'],
                filesize= data['filesize']
            ))


    DatabaseFile.objects.bulk_create(not_created_data, batch_size=50)
    DatabaseFile.objects.bulk_update(already_created_data, ['last_modified', 'filesize'], batch_size=50)


def parse_database_files(options):
    for my_file in (DatabaseFile.objects.all()):
        if my_file.valid_metadata is None:
            res = guess_exp_type(my_file.filename, my_file.root)

            if res:
                meta, valid = deterministic_parser(
                    my_file.filename,
                    res)
                print('before call', meta)
                my_file.set_valid_metadata(valid_metadata=meta)
                print('after call', my_file.valid_metadata)
                my_file.save()
            #else:
            #    print('unknown experiment', ' root: ', my_file.root , ' filename: ',my_file.filename)

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('--mode', choices=['display',
                                               'clear_all',
                                               'add_charger_drive_profile',
                                               'add_experiment_type',
                                               'add_category',
                                               'parse_and_add_files',
                                               'just_add_files',
                                               'just_parse_database_files',
                                               'validate_files',
                                               'print_invalid_files',
                                               'nd_parser',
                                               'output_all_to_excel',
                                               'output_specific_to_excell'


                                               ])
        parser.add_argument('--model', choices=list(models_dict.keys()) + ['*',''],default='')
        parser.add_argument('--visuals', dest='visuals', action='store_true')
        parser.add_argument('--no-visuals', dest='visuals', action='store_false')
        parser.set_defaults(visuals=False)
        parser.add_argument('--data_dir', default="/DB/srv/samba/share/DATA")


    def handle(self, *args, **options):

        if options['mode'] == 'display':

            for element in models_dict[options['model']].objects.all():

                print('{}: {}'.format(options['model'], element))


        if options['mode'] == 'clear_all':
            if options['model'] == '*':
                for key in models_dict.keys():
                    models_dict[key].objects.all().delete()


            else:
                models_dict[options['model']].objects.all().delete()


        if options['mode'] == 'add_category':
            categories = [

                'cycling',
                'rpt',
                'gas',
                'impedance',
                'thermal',
                'storage',
                'electrolyte',
                'electrode',
                'formation',


                ]
            for cat in categories:
                _, _ = Category.objects.get_or_create(name=cat)

            subcategories = [

                'neware',
                'moli',
                'maccor',
                'uhpc',
                'novonix',

                'insitu',
                'eis',
                'symmetric',
                'arc',
                'microcalorimetry',
                'smart',
                'dumb',
                'gcms',
                'ldta',
                'xps',

            ]

            for cat in subcategories:
                _, _ = SubCategory.objects.get_or_create(name=cat)



        if options['mode'] == 'add_charger_drive_profile':

            dummies = [
                {
                    'drive_profile':'CXCX',
                    'test' : 'CCCV cycling',
                    'description' : 'charge at C/X, CV hold, discharge at C/X',
                    'x_active' : True,
                    'y_active' : False,
                    'z_active' : False,
                    'x_name' : 'C-Rate',
                    'y_name' : '',
                    'z_name' : ''
                 },

                {
                    'drive_profile': 'CXCY',
                    'test':'CCCV cycling, different rates',
                    'description':'charge at C/X, CV hold, discharge at C/Y',
                    'x_active':True,
                    'y_active':True,
                    'z_active':False,
                    'x_name' : 'Charge rate',
                    'y_name' : 'Discharge rate',
                    'z_name' : ''
                },

                {
                    'drive_profile' : 'CX',
                    'test': 'CC cycling',
                    'description' : 'charge at C/X, discharge at C/X',
                    'x_active' : True,
                    'y_active' : False,
                    'z_active':False,
                    'x_name' : 'C-Rate',
                    'y_name' : '',
                    'z_name' : ''
                },

                {
                    'drive_profile' : 'CXCYc',
                    'test' : 'CC cycling with different rates',
                    'description' : 'charge at C/X, discharge at C/Y',
                    'x_active' : True,
                    'y_active' : True,
                    'z_active':False,
                    'x_name' : 'Charge Rate',
                    'y_name' : 'Discharge rate',
                    'z_name' : ''

                },

                {

                    'drive_profile' : 'CXrc',
                    'test' : 'rate capability cycling',
                    'description' : 'cycling at initial C/X rate, progressively faster cycling',
                    'x_active' : True,
                    'y_active' : False,
                    'z_active':False,
                    'x_name' : 'Initial C-rate',
                    'y_name' : '',
                    'z_name' : '',

                },

                {
                    'drive_profile' : 'CXCYb',
                    'test' :'Barn cycling',
                    'description' : 'charge at C/X, switch to C/Y, discharge at C/Y, change to C/X',
                    'x_active' : True,
                    'y_active' : True,
                    'z_active' : False,
                    'x_name' : 'First charge rate/second discharge rate',
                    'y_name' : 'Second charge rate/first discharge rate',
                    'z_name' : '',
                },


                {
                  'drive_profile' : 'CXsZZZ',
                  'test':'CC + storage',
                  'description':'charge/discharge at C/X, store at Z V',
                  'x_active':True,
                  'y_active':False,
                  'z_active':True,
                  'x_name':'C-Rate',
                  'y_name':'',
                  'z_name':'storage voltage'

                },


            ]

            for dummy in dummies:
                ChargerDriveProfile.objects.update_or_create(drive_profile=dummy['drive_profile'],
                                                             defaults=dummy )

        if options['mode'] == 'add_experiment_type':

            experiments = \
                [
                    {
                        'category':Category.objects.get(name='cycling'),
                        'subcategory':SubCategory.objects.get(name='neware'),
                        'drive_profile_active':True,
                        'charger_active' : True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger': 'Nw',
                        'shorthand':'CYC'
                    },

                    {
                        'category': Category.objects.get(name='cycling'),
                        'subcategory': SubCategory.objects.get(name='moli'),
                        'drive_profile_active': True,
                        'charger_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger': 'Mo',
                        'shorthand': 'CYC'
                    },

                    {
                        'category': Category.objects.get(name='cycling'),
                        'subcategory': SubCategory.objects.get(name='maccor'),
                        'voltage_active': False,
                        'temperature_active': False,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger': '',
                        'shorthand': 'CYC'

                    },

                    {
                        'category': Category.objects.get(name='cycling'),
                        'subcategory': SubCategory.objects.get(name='uhpc'),
                        'drive_profile_active': True,
                        'charger_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger': 'UHPC',
                        'shorthand': 'CYC'

                    },

                    {
                        'category': Category.objects.get(name='cycling'),
                        'subcategory': SubCategory.objects.get(name='novonix'),
                        'drive_profile_active': True,
                        'charger_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger': 'Nx',
                        'shorthand': 'CYC'
                    },

                    {
                        'category': Category.objects.get(name='rpt'),
                        'subcategory': SubCategory.objects.get(name='neware'),
                        'charger_active': True,
                        'voltage_name': 'storage_potential',
                        'temperature_name': 'storage_temperature',
                        'charger': 'Nw',
                        'shorthand': 'RPT'

                    },



                    {
                        'category': Category.objects.get(name='gas'),
                        'subcategory': SubCategory.objects.get(name='insitu'),
                        'drive_profile_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand':'GasInSitu',
                        'charger':''
                    },

                    {
                        'category': Category.objects.get(name='impedance'),
                        'subcategory': SubCategory.objects.get(name='eis'),
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand':'EIS',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='impedance'),
                        'subcategory': SubCategory.objects.get(name='maccor'),
                        'voltage_active': False,
                        'temperature_active': False,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand':'FRA',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='impedance'),
                        'subcategory': SubCategory.objects.get(name='neware'),
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger':'Nw',
                        'shorthand':'FRA'
                    },

                    {
                        'category': Category.objects.get(name='impedance'),
                        'subcategory': SubCategory.objects.get(name='symmetric'),
                        'AC_active': True,
                        'AC_increment_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand': 'SYM',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='thermal'),
                        'subcategory': SubCategory.objects.get(name='arc'),
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand': 'ARC',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='thermal'),
                        'subcategory': SubCategory.objects.get(name='microcalorimetry'),
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand': 'TAM',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='storage'),
                        'subcategory': SubCategory.objects.get(name='smart'),
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand': 'STOSmart',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='storage'),
                        'subcategory': SubCategory.objects.get(name='dumb'),
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand': 'STODumb',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='electrolyte'),
                        'subcategory': SubCategory.objects.get(name='ldta'),
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand': 'LDTA',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='electrolyte'),
                        'subcategory': SubCategory.objects.get(name='gcms'),
                        'voltage_active': False,
                        'temperature_active': False,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand': 'GCMS',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='electrode'),
                        'subcategory': SubCategory.objects.get(name='xps'),
                        'voltage_active': False,
                        'temperature_active': False,
                        'AC_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand': 'XPS',
                        'charger': ''
                    },

                    {
                        'category': Category.objects.get(name='formation'),
                        'subcategory': SubCategory.objects.get(name='neware'),
                        'start_cycle_active': False,
                        'drive_profile_active': True,
                        'charger_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger': 'Nw',
                        'shorthand': 'FM'

                    },

                    {
                        'category': Category.objects.get(name='formation'),
                        'subcategory': SubCategory.objects.get(name='moli'),
                        'start_cycle_active': False,
                        'drive_profile_active': True,
                        'charger_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger': 'Mo',
                        'shorthand': 'FM'
                    },

                    {
                        'category': Category.objects.get(name='formation'),
                        'subcategory': SubCategory.objects.get(name='uhpc'),
                        'start_cycle_active': False,
                        'drive_profile_active': True,
                        'charger_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger': 'UHPC',
                        'shorthand': 'FM'
                    },

                    {
                        'category': Category.objects.get(name='formation'),
                        'subcategory': SubCategory.objects.get(name='novonix'),
                        'start_cycle_active': False,
                        'drive_profile_active': True,
                        'charger_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'charger': 'Nx',
                        'shorthand': 'FM'
                    },

                    {
                        'category': Category.objects.get(name='formation'),
                        'subcategory': SubCategory.objects.get(name='maccor'),
                        'start_cycle_active': False,
                        'temperature_active': False,
                        'charger_active': True,
                        'voltage_name': 'upper_cutoff_voltage',
                        'temperature_name': 'temperature',
                        'shorthand': 'FM',
                        'charger': ''
                    },

                ]

            for experiment in experiments:
                ExperimentType.objects.update_or_create(subcategory=experiment['subcategory'], category=experiment['category'],
                                                             defaults=experiment)




        if options['mode'] == 'just_add_files':
            add_files(options)

        if options['mode'] == 'just_parse_database_files':
            parse_database_files(options)




        if options['mode'] =='print_invalid_files':

            for data in (DatabaseFile.objects.all()):
                if not data.file_is_valid:
                    print('FILE = {} ROOT = {}'.format(data.filename, data.root))




        if options['mode'] == 'nd_parser':

            count = 0

            for db in DatabaseFile.objects.all():

                nd_parser_results = nd_parser(db.filename)


                if 'student' in nd_parser_results.keys():

                    for element in nd_parser_results['student']:

                        st = Token(student=str(element))

                        st.save()

                        db = DatabaseFile.objects.get(filename=db.filename)

                        db.all_possible_students.add(st)

                        db.save()


                if 'barcode' in nd_parser_results.keys():

                    for element in nd_parser_results['barcode']:

                        ba = Token(barcode=element)

                        ba.save()

                        db = DatabaseFile.objects.get(filename=db.filename)

                        db.all_possible_barcodes.add(ba)

                        db.save()

                if 'date' in nd_parser_results.keys():

                    for element in nd_parser_results['date']:

                        da = Token(date=element)

                        da.save()

                        db = DatabaseFile.objects.get(filename=db.filename)

                        db.all_possible_dates.add(da)

                        db.save()


                if 'voltage' in nd_parser_results.keys():

                    for element in nd_parser_results['voltage']:

                        vo = Token(voltage=element)

                        vo.save()

                        db = DatabaseFile.objects.get(filename=db.filename)

                        db.all_possible_voltages.add(vo)

                        db.save()

                if 'temperature' in nd_parser_results.keys():

                    for element in nd_parser_results['temperature']:

                        tp = Token(temperature=element)

                        tp.save()

                        db = DatabaseFile.objects.get(filename=db.filename)

                        db.all_possible_temperatures.add(tp)

                        db.save()


                if 'start_cycle' in nd_parser_results.keys():

                    for element in nd_parser_results['start_cycle']:

                        sc = Token(temperature=element)

                        sc.save()

                        db = DatabaseFile.objects.get(filename=db.filename)

                        db.all_possible_temperatures.add(sc)

                        db.save()

                count += 1

                print(count)


        if options['mode'] == 'output_all_to_excel':

            if not os.path.exists('/home/ubuntu/site/FileNameHelper/debug'):
                os.mkdir('/home/ubuntu/site/FileNameHelper/debug')

            for exp_type_key in experiment_types.keys():

                for dp_key in drive_profiles.keys():

                    if exp_type_key == 'None':
                        exp_type_id = None

                    else:
                        exp_type_id = experiment_types[exp_type_key].id

                    if dp_key == 'None':
                        dp_id = None

                    else:
                        dp_id = drive_profiles[dp_key].id

                    path = '/home/ubuntu/site/FileNameHelper/debug/debugging_{}_{}.csv'.format(exp_type_key.replace(' ', '_'),dp_key)

                    print(exp_type_key)
                    print(dp_key)

                    output_to_csv(exp_type_id, dp_id, path)


        if options['mode'] == 'output_specific_to_excel':

            if not os.path.exists('/home/ubuntu/site/FileNameHelper/debug'):
                os.mkdir('/home/ubuntu/site/FileNameHelper/debug')

            path = '/home/ubuntu/site/FileNameHelper/debug/debugging_{}_{}.csv'.format(options['exp_type'].replace(' ', '_'), options['dp'])

            output_to_csv(options['exp_type'].id, options['dp'].id, path)



