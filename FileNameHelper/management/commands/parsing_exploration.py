from django.core.management.base import BaseCommand
from contact.models import *
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
        print('\n\n\n\n\n\nThe program starts here:\n\n')

        path_to_files = options['data_dir']

##============================================================================================##
                                # Logic to determine experiment type #
##============================================================================================##

        matches = {}














        ''''#what's this one? A: limit of print
        counter = 0

        #what's this one? A: files in NEWARE path that have 7 or more tokens separated by -,_,.
        proper_neware_file_counter = 0
        proper_uhpc_file_counter = 0
        proper_novonix_file_counter = 0
        proper_insitu_file_counter = 0


        # Cycling
        #what's this one. counts category'''


        '''
        (sam)
        
        I would do 
        category_counts = {
            'cycling':0,
            'gas':0,
            ...
        }
        
       then when you want to count a cycling file, do:
       category_counts['cycling']+=1 
       
        '''

        ## Category count dictionary
        category_counts = {

            'cycling' : 0,
            'gas' : 0,
            'impedance' : 0,
            'thermal' : 0,
            'storage' : 0,
            'electrolyte' : 0,
            'electrode': 0,
            'formation' : 0
        }


        #cycling_count = 0

        '''
        (sam)
        I would do:
        tests_counts = {
            'cycling':{
                'neware':0,
                'moli':0,
                ...
            },
            'gas':{
                ...
            }
        }
        
        
        when you want to increment what used to be absolutely_neware_count:
        tests_counts['cycling']['neware'] +=1
        
        
        then, at the end, 
        for cat in category_counts.keys():
            print(cat, 'testing if sum of tests is cat')
            
            total = 0
            for test in tests_counts[cat].keys():
                total += tests_counts[cat][test]
                print('   test {} is {}'.format(test, tests_counts[cat][test]))
                
                
            print('cat {} is {}. Total of tests is {}'.format(cat, category_counts[cat], total))
        '''


        ## Test Counts

        test_counts = {

            cycling : {'neware' : 0, 'moli': 0, 'uhpc':0, 'novonix':0, 'rpt':0 },
            gas: {'exsitu' : 0, 'insitu' : 0},
            impedance: {'eis' : 0, 'maccor': 0, 'neware': 0, 'symmetric': 0},
            thermal: {'arc': 0, 'microcalorimetry': 0},
            storage: {'smart': 0, 'dumb': 0},
            electrolyte: {'ldta': 0, 'gcms': 0, 'xps': 0},
            electrode: {'xps': 0},
            formation: {'cycler': 0, 'maccor': 0}

        }

        '''
        #counts test and sum of test should equal category.
        absolutely_neware_count = 0
        absolutely_moli_count = 0
        absolutely_uhpc_count = 0
        absolutely_novonix_count = 0
        absolutely_rpt_count = 0

        # Gas
        absolutely_exsitu_count = 0
        absolutely_insitu_count = 0
        #gas_count = 0

        # Impedance
        #imp_count = 0
        absolutely_eis_count = 0
        absolutely_maccorFRA_count = 0
        absolutely_newareFRA_count = 0
        absolutely_symmetric_count = 0


        # Thermal
        #thermal_count = 0
        absolutely_arc_count = 0
        absolutely_microcal_count = 0

        # Storage
        #storage_count = 0
        absolutely_smart_count = 0
        absolutely_dumb_count = 0

        #Electrolyte
        #electrolyte_count = 0
        absolutely_gcms_count = 0
        absolutely_liion_count = 0

        #Electrode
        #electrode_count = 0
        absolutely_xps_count = 0

        #Formation
        #formation_count = 0
        absolutely_cycler_count = 0
        absolutely_maccorfm_count = 0'''


        for root, dirs, filenames in os.walk(path_to_files):
            for file in filenames:



## -------------------------------CYCLING-----------------------------------##


## Cycling Experiments (Not Rpt):
                '''
                after you go through your logic and determine what experiment type the file belongs to, set a variable (exp_type) with the answer.
                
                Then, put the metadata stuff completely after, and just look at the variable you set.
                
                also, this is literally a function def experiment_type_classifier(file, root) --> (category, experiment_type) as strings.
                
                then, the other function is def conditional_metadata_extract(file,root, category, experiment_type) --> a dictionary with metadata in it.
                '''

                '''
                (sam)
                you are adding too many special cases.
                What you want is to ignore the upper/lower case distinction.
                lowercase_file = file.lower()
                cyclingFileMatch = re.search('cyc', lowercase_file) or re.search('cycling', lowercase_file)
                
                or simply:
                cyclingFileMatch = re.search(r'(cyc)|(cycling)', lowercase_file)
                
                
                note that in this case, r'(cyc)|(cycling)' is same as r'(cyc)'
                
                '''


                lowercase_file = file.lower()

                matches['cyclingFileMatch'] = re.search(r'(cyc)|(cycling)', lowercase_file)
                matches['cyclingMatch'] = re.search('CYC', root)



                '''
                (sam) I think you should do it like this:
                matches = {}
                
                
                matches['cycling_file'] = re.search(r'(cyc)|(cycling)', lowercase_file)
                matches['cycling_root'] = re.search('CYC', root)
                matches['neware_root'] = re.search('NEWARE', root)
                ... (do all of them, then go in if statement)                
                '''


                # now dealing with case both evidences are true.
                if matches['cyclingFileMatch'] and matches['cyclingMatch']:
                    category_counts['cycling'] += 1

                    # Neware and RPT

                    # do matches['neware_root'] = re.search('NEWARE', root)
                    matches['neware_root'] = re.search('NEWARE', root)

                    matches['newareFileMatch'] = re.search(r'(neware)|(nw)', lowercase_file)

                    rptFileMatch = re.search('RPT', file) or re.search('rpt', file) or re.search('Rpt', file)


                    if (newareMatch and newareFileMatch):
                        absolutely_neware_count += 1

                    ### ---------   META DATA -------------###

                    if (newareMatch and newareFileMatch):


                        # no reason to put in if statement.
                        fileList = re.split(r'-|_|\.', file)

                       # Counting number of good files -- result was same number as there are Neware files

                        if len(fileList) >= 7:
                            proper_neware_file_counter += 1

                        # Will only extract meta data from 10 files
                        if counter < 10:


                            if len(fileList) >= 7:
                                # it might work for now, but it is not very robust.
                                # this logic belongs to conditional parser, not discriminator.
                                charID = fileList[0]
                                CYC = fileList[1]
                                barcode = fileList[2]
                                experiment = fileList[3]
                                start_cycle = fileList[4]
                                voltage = fileList[5]
                                temp = fileList[6]

                              ## Already Tested -- No need to print every time
                                print('\nFor Neware;')
                                print('CharID: {}\nCYC: {}\nBarcode: {}\nExperiment Type: {}\nStart Cycle: {}\nVoltage: {}\nTemperature: {}\n\n'.format(charID, CYC, barcode, experiment, start_cycle, voltage, temp))

                            counter += 1

                    ### ---------------------------------###

                    # I would probably just have RPTMatch which is the same as newareMatch
                    if (newareMatch and rptFileMatch):
                        absolutely_rpt_count += 1

                    # Moli -- None on count, needs editing

                    moliMatch = re.search('NEWARE', root)
                    moliFileMatch = re.search('MOLI', file) or re.search('MO', file) or re.search('Moli', file) or re.search('Mo', file) or re.search('mo', file) or re.search('moli', file)

                    if (moliMatch and moliFileMatch):
                        absolutely_moli_count += 1


                    # Maccor -- Tackle later, no tag in filename.


                    # UHPC

                    uhpcMatch = re.search('UHPC', root)
                    uhpcFileMatch = re.search('UHPC', file) or re.search('uhpc', file) or re.search('Uhpc', file)

                    if (uhpcMatch and uhpcFileMatch):
                        absolutely_uhpc_count += 1


                    ##---------------------META DATA ------------------------##


                    if (uhpcMatch and uhpcFileMatch):
                        #should be at top, before any if statement.
                        fileList = re.split(r'-|_|\.', file)

                        # Counting number of good files

                        if len(fileList) >= 7:
                            proper_uhpc_file_counter += 1

                        # Will only extract meta data from 10 files
                        if counter < 10:
                            # even if just for testing, wait until after discriminator.
                            if len(fileList) >= 7:
                                charID = fileList[0]
                                CYC = fileList[1]
                                barcode = fileList[2]
                                experiment = fileList[3]
                                start_cycle = fileList[4]
                                voltage = fileList[5]
                                temp = fileList[6]

                                ## No need to print, already tested and it works
                                #print('\nFor UHPC;')
                                #print('CharID: {}\nCYC: {}\nBarcode: {}\nExperiment Type: {}\nStart Cycle: {}\nVoltage: {}\nTemperature: {}\n\n'.format(charID, CYC, barcode, experiment, start_cycle, voltage, temp))

                        counter += 1

                      ## -------------------------------------------------------##



                    # Novonix

                    novonixMatch = re.search('NOVONIX', root)
                    novonixFileMatch = re.search('NOVONIX', file) or re.search('NX', file) or re.search('Novonix', file) or re.search('Nx', file) or re.search('nx', file) or re.search('novonix', file)

                    if (novonixMatch and novonixFileMatch):
                        absolutely_novonix_count += 1

                    ##---------------------META DATA ------------------------##


                    if (novonixMatch and novonixFileMatch):
                        fileList = re.split(r'-|_|\.', file)

                        #counter = 0
                        # Counting number of good files

                        if len(fileList) >= 7:
                            proper_novonix_file_counter += 1

                        # Will only extract meta data from 10 files
                        if counter < 10:

                            if len(fileList) >= 7:
                                charID = fileList[0]
                                CYC = fileList[1]
                                barcode = fileList[2]
                                experiment = fileList[3]
                                start_cycle = fileList[4]
                                voltage = fileList[5]
                                temp = fileList[6]

                                ##Worked, no need to print
                                #print('\nFor UHPC;')
                                #print('CharID: {}\nCYC: {}\nBarcode: {}\nExperiment Type: {}\nStart Cycle: {}\nVoltage: {}\nTemperature: {}\n\n'.format(charID, CYC, barcode, experiment, start_cycle, voltage, temp))

                        counter += 1

                    ## -------------------------------------------------------##

## ------------------------------------GAS--------------------------------------------##


                gasMatch = re.search('GAS', root) or re.search('gas', file) or re.search('Gas', file)


                if gasMatch:
                    gas_count += 1
                    # Ex-Situ -- Should be zero

                    exsituFileMatch = re.search('EXSitu', file) or re.search('exSitu', file) or re.search('ExSitu', file) or re.search('EXSITU', file) or re.search('exSITU', file) or re.search('ExSITU', file)

                    if exsituFileMatch:
                        absolutely_exsitu_count += 1




                    # In-Situ

                    insituFileMatch = re.search('INSitu', file) or re.search('inSitu', file) or re.search('InSitu', file) or re.search('INSITU', file) or re.search('inSITU', file) or re.search('InSITU', file)

                    if insituFileMatch:
                        absolutely_insitu_count += 1

                    ##---------------------META DATA ------------------------##


                    if insituFileMatch:
                        fileList = re.split(r'-|_|\.', file)

                        counter = 0

                        #Counting number of good files
                        if len(fileList) >= 7:
                            proper_insitu_file_counter += 1

                        # Will only extract meta data from 10 files
                        if counter < 10:

                            if len(fileList) >= 6:
                                fileList = re.split(r'-|_|\.', file)
                                charID = fileList[0]
                                GasInSitu = fileList[1]
                                barcode = fileList[2]
                                start_cycle = fileList[3]
                                voltage = fileList[4]
                                temp = fileList[5]

                                ##No need to print, it worked
                                #print('\nFor In-Situ;')
                                #print('CharID: {}\nGasInSitu: {}\nBarcode: {}\nStart Cycle: {}\nTemperature: {}'.format(charID, GasInSitu, barcode, start_cycle, voltage, temp))

                        counter += 1

                    ## -------------------------------------------------------##


## --------------------------------IMPEDANCE--------------------------------------------##

                impedanceMatch = re.search('IMPEDANCE', root)

                if impedanceMatch:
                    imp_count += 1

                    #EIS
                    eisMatch = re.search('EIS', root)
                    eisFileMatch = re.search('EIS', file) or re.search('eis', file) or re.search('Eis', file)

                    if (eisMatch and eisFileMatch):
                        absolutely_eis_count += 1


                    # MACCOR -- Should be zero
                    maccorFRAMatch = re.search('MACCOR', root)
                    maccorFRAFileMatch = re.search('FRA', file) or re.search('fra', file) or re.search('Fra', file)

                    if (maccorFRAMatch and maccorFRAFileMatch):
                        absolutely_maccorFRA_count += 1

                    # NEWARE FRA
                    newareFRAMatch = re.search('NEWARE', root)
                    newareFRAFileMatch = re.search('FRA', file) or re.search('fra', file) or re.search('Fra', file)

                    if (newareFRAMatch and newareFRAFileMatch):
                        absolutely_newareFRA_count += 1

                    # Symmetric/biologic -- should be zero
                    symmetricMatch = re.search('Symmetric', root)
                    symmetricFileMatch = re.search('Sym', file) or re.search('SYM', file) or re.search('sym', file) or re.search('Symmetric', file) or re.search('SYMMETRIC', file) or re.search('symmetric', file)

                    if (symmetricMatch and symmetricFileMatch):
                        absolutely_symmetric_count += 1


## ----------------------------------THERMAL--------------------------------------------------##

                thermalMatch = re.search('THERMAL', root)

                if thermalMatch:

                    # Should be far less than the sum of ARC and Microcalorimetry files since there is a third experiment type, 'TGA'
                    thermal_count += 1

                    # ARC
                    arcMatch = re.search('ARC', root)
                    arcFileMatch = re.search('ARC', file) or re.search('arc', file) or re.search('Arc', file)

                    if (arcMatch and arcFileMatch):
                        absolutely_arc_count += 1


                    # Microcalorimetry
                    microcalMatch = re.search('Microcalorimetry', root)
                    microcalFileMatch = re.search('TAM', file) or re.search('tam', file) or re.search('Tam', file)

                    if (microcalMatch and microcalFileMatch):
                        absolutely_microcal_count += 1


## -------------------------------STORAGE----------------------------------------------------##


                storageMatch = re.search('STORAGE', root)

                if storageMatch:

                    storage_count += 1

                    # Smart
                    smartMatch = re.search('Smart', root)
                    smartFileMatch = re.search('Smart', file) or re.search('smart', file) or re.search('SMART', file)

                    if (smartMatch and smartFileMatch):
                        absolutely_smart_count += 1


                    # Dumb
                    dumbMatch = re.search('Dumb', root)
                    dumbFileMatch = re.search('Dumb', file) or re.search('dumb', file) or re.search('Dumb', file)

                    if (dumbMatch and dumbFileMatch):
                        absolutely_dumb_count += 1


## ------------------------------------ELECTROLYTE--------------------------------------------------##

                electrolyteMatch = re.search('ELECTROLYTE', root)

                if electrolyteMatch:

                    electrolyte_count += 1

                    # GCMS
                    gcmsMatch = re.search('GCMS', root)
                    gcmsFileMatch = re.search('GCMS', file) or re.search('gcms', file) or re.search('Gcms', file)

                    if (gcmsMatch and gcmsFileMatch):
                        absolutely_gcms_count += 1

                    # Li-ion DTA
                    liionMatch = re.search('Li-ion DTA', root)
                    liionFileMatch = re.search('LDTA', file) or re.search('ldta', file) or re.search('Ldta', file)

                    if (liionMatch and liionFileMatch):
                        absolutely_liion_count += 1

##---------------------------------ELECTRODE------------------------------------------------------##

                electrodeMatch = re.search('ELECTRODE', root)

                if electrodeMatch:

                    electrode_count += 1

                    # XPS
                    xpsMatch = re.search('XPS', root)
                    xpsFileMatch = re.search('XPS', file) or re.search('xps', file) or re.search('Xps]', file)

                    if (xpsMatch and xpsFileMatch):
                        absolutely_xps_count += 1

##-------------------------------FORMATION--------------------------------------------------------##

                formationMatch = re.search('FORMATION', root)

                if formationMatch:
                    formation_count += 1

                    # Cycler / Maccor (Formation)
                    # I would do r'(\dc)|(c\d)', and just limit to x number of fileList[:x]
                    cyclerFileMatch = re.search(r'\dC', file) or re.search(r'\dc', file) or re.search(r'C\d', file) or re.search(r'c\d', file)
                    formFileMatch = re.search('FORM', file) or re.search('form', file) or re.search('Form', file) or re.search('FM', file) or re.search('fm', file) or re.search('Fm', file) or re.search('Formation', file) or re.search('FORMATION', file) or re.search('formation', file)

                    if (cyclerFileMatch and formFileMatch):
                        absolutely_cycler_count += 1


                    elif formFileMatch:
                        absolutely_maccorfm_count += 1

##------------------------------------------------------------------------------------------------------------##


        print('Proper Neware file:', proper_neware_file_counter)
        print('Proper UHPC file:', proper_uhpc_file_counter)
        print('Proper Novonix file:', proper_novonix_file_counter)
        print('proper In-Situ file:', proper_insitu_file_counter)
        print('\n\n')

        # Cycling
        print('Cycling files: ', cycling_count)
        print('Absolutely Neware Count: ', absolutely_neware_count)
        print('Absolutely Moli Count: ', absolutely_moli_count)
        print('Absolutely UHPC Count: ', absolutely_uhpc_count)
        print('Absolutely Novonix Count: ', absolutely_novonix_count)
        print('Absolutely RPT Count: ', absolutely_rpt_count)

        # Gas
        print('Gas files:', gas_count)
        print('Absolutely Ex-Situ Count: ', absolutely_exsitu_count)
        print('Absolutely In-Situ Count: ', absolutely_insitu_count)

        # Impedance
        print('Impedance Files: ', imp_count)
        print('Absolutely EIS Count: ', absolutely_eis_count)
        print('Absolutely Maccor FRA Count: ', absolutely_maccorFRA_count)
        print('Absolutely Neware FRA Count: ', absolutely_newareFRA_count)
        print('Absolutely Symmetric Count: ', absolutely_symmetric_count)

        #Thermal
        print('Thermal files: ', thermal_count)
        print('Absolutely ARC Count: ', absolutely_arc_count)
        print('Absolutely Microcalorimetry Count: ', absolutely_microcal_count)

        #Storage
        print('Storage files: ', storage_count)
        print('Absolutely Smart Count: ', absolutely_smart_count)
        print('Absolutely Dumb Count: ', absolutely_dumb_count)

        #Electrolyte
        print('Electrolyte files: ', electrolyte_count)
        print('Absolutely GCMS Count: ', absolutely_gcms_count)
        print('Absolutely Li-ion DTA Count: ', absolutely_liion_count)

        #Electrode
        print('Electrode files: ', electrode_count)
        print('Absolutely XPS Count: ', absolutely_xps_count)

        #Formation
        print('Formation files: ', formation_count)
        print('Absolutely Cycler Count: ', absolutely_cycler_count)
        print("Absolutely Maccor (Formation) Count: ", absolutely_maccorfm_count)
        print('\n\n')


##============================================================================================##
                              # META-DATA EXTRACTOR FUNCTION #
##============================================================================================##



'''def meta_data_extractor(exp_type, filename):

    # Skip for now, havn't solved other logic
    if exp_type == 'Moli':


    if exp_type == 'Maccor':

## WORKED
    if exp_type == 'Neware':
        fileList = re.split(r'-|_|\.', filename)
        charID = fileList[0]
        CYC = fileList[1]
        barcode = fileList[2]
        experiment = fileList[3]
        start_cycle = fileList[4]
        voltage = fileList[5]
        temp = fileList[6]
        print('\nFor Neware;')
        print('CharID: {}\nCYC: {}\nBarcode: {}\nExperiment Type: {}\nStart Cycle: {}\nTemperature: {}'.format(charID, CYC, barcode, experiment, start_cycle, temp))


## WORKED
    if exp_type == 'UHPC':
        fileList = re.split(r'-|_|\.', filename)
        charID = fileList[0]
        CYC = fileList[1]
        barcode = fileList[2]
        experiment = fileList[3]
        start_cycle = fileList[4]
        voltage = fileList[5]
        temp = fileList[6]
        print('\nFor UHPC;')
        print('CharID: {}\nCYC: {}\nBarcode: {}\nExperiment Type: {}\nStart Cycle: {}\nTemperature: {}'.format(charID, CYC, barcode, experiment, start_cycle, temp))

## WORKED
    if exp_type == 'Novonix':
        fileList = re.split(r'-|_|\.', filename)
        charID = fileList[0]
        CYC = fileList[1]
        barcode = fileList[2]
        experiment = fileList[3]
        start_cycle = fileList[4]
        voltage = fileList[5]
        temp = fileList[6]
        print('\nFor Novonix;')
        print('CharID: {}\nCYC: {}\nBarcode: {}\nExperiment Type: {}\nStart Cycle: {}\nTemperature: {}'.format(charID, CYC, barcode, experiment, start_cycle, temp))
    
## NO FILES TO TEST 
    if exp_type == 'RPT':
    
## NO FILES TO TEST    
    if exp_type == 'Ex-Situ':    
    
    if exp_type == 'In-Situ':
        fileList = re.split(r'-|_|\.', filename)
        charID = fileList[0]
        GasInSitu = fileList[1]
        barcode = fileList[2]
        start_cycle = fileList[3]
        voltage = fileList[4]
        temp = fileList[5]
        print('\nFor In-Situ;')
        print('CharID: {}\nGasInSitu: {}\nBarcode: {}\nStart Cycle: {}\nTemperature: {}'.format(charID, CYC, barcode, start_cycle, temp))
    
    
    if exp_type == 'EIS/biologic':
    if exp_type == 'Maccor (Impedance)':
    if exp_type == 'Neware FRA':
    if exp_type == 'Symmetric/biologic':
    if exp_type == 'ARC':
    if exp_type == 'Microcalorimetry':
    if exp_type == 'Smart':
    if exp_type == 'Dumb':
    if exp_type == 'Li-ion DTA':
    if exp_type == 'GCMS':
    if exp_type == 'XPS':
    if exp_type == 'Cycler':
    if exp_type == 'Maccor (Formation)':'''




