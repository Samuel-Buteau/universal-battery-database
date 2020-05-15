from filename_database.models import ExperimentType, ChargerDriveProfile, Category, SubCategory, ValidMetadata
import re
import datetime
import itertools

def guess_exp_type(file, root):
    """
    This function takes a file as input and guesses what experiment type it is.
    :param file:
    :param root:
    :return: the experiment type
    """
    lowercase_file = file.lower()
    fileList = re.split(r'-|_|\.|\s', lowercase_file)
    #We handle cycling, formation and fra, maccor is only exception
    cat_match = {
        'cycling': r'(^cyc$)|(^cycling$)',
        'formation': r'(^form$)|(^fm$)',
        'impedance': r'^fra$',
        'rpt': r'^rpt$',
    }
    cat = None
    broken = False
    for k in cat_match.keys():
        if broken:
            break
        for elem in fileList:
            if re.match(cat_match[k], elem):
                cat = Category.objects.get(name=k)
                broken = True
                break

    if cat is not None:
        # try to match subcategory
        sub_match = {
            'neware':r'(^neware$)|(^nw$)',
            'moli':r'^mo$',
            'uhpc':r'^uhpc$',
            'novonix':r'(^novonix$)|(^nx$)',
        }
        sub = None
        broken = False
        for k in sub_match.keys():
            if broken:
                break
            for elem in fileList[1:]:
                if re.match(sub_match[k], elem):
                    sub = SubCategory.objects.get(name=k)
                    broken = True
                    break

        if sub is None:
            if 'NEWARE' in root:
                sub = SubCategory.objects.get(name='neware')
            else:
                sub = SubCategory.objects.get(name='maccor')

        exp_type = ExperimentType.objects.get(category=cat, subcategory=sub)
        #TODO: make a table in the experiment type to be the valid regexp for file extension.
        if sub.name=='neware':
            if lowercase_file.split('.')[-1] != 'txt':
                return None
        return exp_type



    #handle the rest
    match = [
        ('gas', 'insitu', r'(^insitugas$)|(^insitu$)|(^gasinsitu$)'),
        ('impedance', 'eis', r'^eis$'),
        ('impedance', 'symmetric', r'(^sym$)|(^symmetric$)'),
        ('thermal', 'arc', r'^arc$'),
        ('thermal', 'microcalorimetry', r'^tam$'),
        ('storage', 'smart', r'smart'),
        ('storage', 'dumb', r'dumb'),
        ('electrolyte', 'gcms', r'^gcms$'),
        ('electrolyte', 'ldta', r'^ldta$'),
        ('electrode', 'xps', r'^xps$'),
    ]
    for c, s, p in match:
        for elem in fileList:
            if re.search(p, elem):
                cat = Category.objects.get(name=c)
                sub = SubCategory.objects.get(name=s)
                if cat.name == 'impedance' and sub.name == 'eis':
                    if 'MACCOR' in root:
                        sub = SubCategory.objects.get(name='maccor')
                exp_type = ExperimentType.objects.get(category=cat, subcategory=sub)
                return exp_type

    return None



##============================================================================================##
# META-DATA EXTRACTOR FUNCTION #
##============================================================================================##

def get_date_obj(date_str):
    """
    parse date string

    :param date_str:
    :return:
    """
    mat1 = re.match(r'20(\d{2,2})(\d{2,2})(\d{2,2})', date_str)
    mat2 = re.match(r'(\d{2,2})(\d{2,2})(\d{2,2})', date_str)
    if mat1:
        mat = mat1
    elif mat2:
        mat = mat2
    else:
        return None
    year = 2000 + int(mat.group(1))
    month = int(mat.group(2))
    day = int(mat.group(3))
    try :
        return datetime.date(year,month,day)
    except ValueError:
        return None


# Function Definition
# Takes in name of file and experiment type as arguments

def deterministic_parser(filename, exp_type):
    """
    given a filename and an experiment type,
    parse as much metadata as possible
    and return a valid_metadata object (None means no parsing, valid metadata with gaps in in means partial parsing.)
    :param filename:
    :param exp_type:
    :return:
    """
    lowercase_file = filename.lower()
    fileList = re.split(r'-|_|\.|\s', lowercase_file)
    def get_charID(fileList):
        max_look = min(3, len(fileList)-1)
        for elem in fileList[:max_look]:
            if re.match(r'^[a-z]{2,5}$', elem) and not (
                    re.search(
                        r'(cyc)|(gcms)|(rpt)|(eis)|(fra)|(sym)|(arc)|(tam)|(xps)|(fm)|(mo)|(nw)|(nx)',
                        elem)):
               return elem

        return None

    def get_possible_cell_ids(fileList):
        possible_cell_ids = []
        max_look = min(5, len(fileList) - 1)
        for elem in fileList[:max_look]:
            if (not re.match(r'200[8-9]0[1-9][0-3][0-9]$|'
                             r'200[8-9]1[0-2][0-3][0-9]$|'
                             r'20[1-2][0-9]0[1-9][0-2][0-9]$|'
                             r'20[1-2][0-9]1[0-1][0-2][0-9]$|'
                             r'20[1-2][0-9]0[1-9][0-3][0-1]$|'
                             r'20[1-2][0-9]1[0-1][0-3][0-1]$|'
                             r'0[8-9]0[1-9][0-3][0-9]$|'
                             r'0[8-9]1[0-2][0-3][0-9]$|'
                             r'[1-2][0-9]0[1-9][0-2][0-9]$|'
                             r'[1-2][0-9]1[0-2][0-2][0-9]$|'
                             r'[1-2][0-9]0[1-9][0-3][0-1]$|'
                             r'[1-2][0-9]1[0-2][0-3][0-1]$',
                             elem)) and (re.match(r'^(\d{5,6})$|^(0\d{5,5})$', elem)) and elem.isdigit():
                possible_cell_ids.append( int(elem))
        return possible_cell_ids

    def get_start_cycle(fileList, avoid=None):
        max_look = min(7, len(fileList) - 1)
        for elem in fileList[: max_look]:
            match = re.match(r'^c(\d{1,4})$', elem)
            if match:
                if avoid is not None and avoid == int(match.group(1)):
                    avoid = None
                    continue
                return int(match.group(1))

        return None

    def get_temperature(fileList):
        for elem in fileList:
            match = re.match(r'^(\d{2})c$', elem)
            if match:
                return int(match.group(1))
        return None


    def get_voltage(fileList):
        for elem in fileList:
            match = re.match(r'^(\d{1,3})v$', elem)
            if match:
                str_voltage = match.group(1)
                n = len(str_voltage)
                divider = 10.**(float(n)-1)
                return float(str_voltage)/divider
        return None

    def get_possible_dates(fileList):
        possible_dates = []
        for elem in fileList:
            if re.match(r'^[0-9]{6,8}$', elem):
                date = get_date_obj(elem)
                if date is not None:
                    possible_dates.append(date)
        return possible_dates


    def get_version_number(fileList):
        for field in fileList:
            match = re.match(r'v(\d)', field)
            if match:
                return int(match.group(1))


    def get_ac_increment(fileList):
        for i in range(len(fileList) - 1):
            match1 = re.match(r'^sym$', fileList[i])
            matchA = re.match(r'^a(\d{1,3})$', fileList[i + 1])
            matchC = re.match(r'^c(\d{1,3})$', fileList[i + 1])

            if match1 and matchA:
                return ValidMetadata.ANODE, int(matchA.group(1))
            elif match1 and matchC:
                return ValidMetadata.CATHODE, int(matchC.group(1))
        return None, None

    def get_ac(fileList):
        for i in range(len(fileList) - 1):
            match1 = re.match(r'^xps$', fileList[i])
            matchA = re.match(r'^a$', fileList[i + 1])
            matchC = re.match(r'^c$', fileList[i + 1])

            if match1 and matchA:
                return ValidMetadata.ANODE
            elif match1 and matchC:
                return ValidMetadata.CATHODE
        return None

    drive_profile_match_dict = {
        'cxcy': (r'^c(\d{1,2})c(\d{1,2})$', ChargerDriveProfile.objects.get(drive_profile='CXCY'), True, True),
        'xcyc': (r'^(\d{1,2})c(\d{1,2})c$', ChargerDriveProfile.objects.get(drive_profile='CXCY'), False, False),
        'xccy': (r'^(\d{1,2})cc(\d{1,2})$', ChargerDriveProfile.objects.get(drive_profile='CXCY'), False, True),
        'cxcyc': (r'^c(\d{1,2})c(\d{1,2})c$', ChargerDriveProfile.objects.get(drive_profile='CXCYc'), True, True),
        'xcycc': (r'^(\d{1,2})c(\d{1,2})cc$', ChargerDriveProfile.objects.get(drive_profile='CXCYc'), False, False),
        'xccyc': (r'^(\d{1,2})cc(\d{1,2})c$', ChargerDriveProfile.objects.get(drive_profile='CXCYc'), False, True),
        'cxrc': (r'^c(\d{1,2})rc$', ChargerDriveProfile.objects.get(drive_profile='CXrc'), True),
        'xcrc': (r'^(\d{1,2})crc$', ChargerDriveProfile.objects.get(drive_profile='CXrc'), False),
        'cxcyb': (r'^c(\d{1,2})c(\d{1,2})b$', ChargerDriveProfile.objects.get(drive_profile='CXCYb'), True, True),
        'xcycb': (r'^(\d{1,2})c(\d{1,2})cb$', ChargerDriveProfile.objects.get(drive_profile='CXCYb'), False, False),
        'xccyb': (r'^(\d{1,2})cc(\d{1,2})b$', ChargerDriveProfile.objects.get(drive_profile='CXCYb'), False, True),
        'cxsz': (r'^c(\d{1,2})s(\d{2,3})$', ChargerDriveProfile.objects.get(drive_profile='CXsZZZ'), True),
        'xcsz': (r'^(\d{1,2})cs(\d{2,3})$', ChargerDriveProfile.objects.get(drive_profile='CXsZZZ'), False),
        'cx': (r'^c(\d{1,2})$', ChargerDriveProfile.objects.get(drive_profile='CX'), True),
        'xc': (r'^(\d{1,2})c$', ChargerDriveProfile.objects.get(drive_profile='CX'), False),

    }

    def get_possible_drive_profiles(fileList):
        possible_drive_profiles = []
        if len(fileList) < 4:
            return possible_drive_profiles


        for elem in fileList[3:]:
            if re.match(r'(^0c$)|(^20c$)|(^40c$)|(^55c$)|(^c0$)|(^c1$)', elem):
                continue
            for k in drive_profile_match_dict.keys():
                m = re.match(drive_profile_match_dict[k][0], elem)
                if m:
                    #special cases
                    my_dp = {'drive_profile': drive_profile_match_dict[k][1]}
                    if drive_profile_match_dict[k][2]:
                        my_dp['drive_profile_x_numerator'] = 1
                        my_dp['drive_profile_x_denominator'] = int(m.group(1))
                    else:
                        my_dp['drive_profile_x_numerator'] = int(m.group(1))
                        my_dp['drive_profile_x_denominator'] = 1
                    if ((drive_profile_match_dict[k][1].drive_profile=='CXCY') and
                            (drive_profile_match_dict[k][2] == drive_profile_match_dict[k][3]) and
                            (m.group(1) == m.group(2))):
                        # CXCX
                        my_dp['drive_profile'] = ChargerDriveProfile.objects.get(drive_profile='CXCX')
                    elif drive_profile_match_dict[k][1].drive_profile=='CXsZZZ':
                        # CXsZZZ
                        n = len(m.group(2))
                        my_dp['drive_profile_z'] = float(m.group(2))/(10.**(float(n)-1))
                    else:
                        if len(drive_profile_match_dict[k]) == 4:
                            if drive_profile_match_dict[k][3]:
                                my_dp['drive_profile_y_numerator'] = 1
                                my_dp['drive_profile_y_denominator'] = int(m.group(1))
                            else:
                                my_dp['drive_profile_y_numerator'] = int(m.group(1))
                                my_dp['drive_profile_y_denominator'] = 1

                    possible_drive_profiles.append(my_dp)
                    break
        return possible_drive_profiles


    # TODO: once you have a date, you must prevent cell_id from being that date.
    # TODO: for now, if multiple alternatives show up, take first one and print.

    metadata = ValidMetadata(experiment_type=exp_type)
    valid = True
    charID = get_charID(fileList)
    if charID is None:
        valid = False
    else:
        metadata.charID = charID

    dates = get_possible_dates(fileList)
    if len(dates) == 0:
        valid = False
    elif len(dates) > 1:
        metadata.date = dates[0]
    else:
        metadata.date = dates[0]

    if exp_type.cell_id_active:
        cell_ids = get_possible_cell_ids(fileList)
        if len(cell_ids) == 0:
            valid = False
        else:
            if metadata.date is None:
                if len(cell_ids) > 1:
                    valid = False
                else:
                    metadata.cell_id = cell_ids[0]
            else:
                valid_cell_ids = []
                for cell_id in cell_ids:
                    date_pieces = [metadata.date.year % 100, metadata.date.month, metadata.date.day]
                    all_perms = list(itertools.permutations(date_pieces))
                    cell_id_ok = True
                    for p in all_perms:
                        if cell_id == p[0] + p[1]*100 + p[2]*10000:
                            cell_id_ok = False
                            break

                    if cell_id_ok:
                        valid_cell_ids.append(cell_id)

                if len(valid_cell_ids) > 1 or len(valid_cell_ids) == 0:
                    valid = False
                else:
                    metadata.cell_id = valid_cell_ids[0]

    if exp_type.AC_active and exp_type.AC_increment_active:
        ac, increment = get_ac_increment(fileList)
        if ac is None:
            valid = False
        else:
            metadata.AC = ac
            metadata.AC_increment = increment

    if exp_type.AC_active and not exp_type.AC_increment_active:
        ac = get_ac(fileList)
        if ac is None:
            valid = False
        else:
            metadata.AC = ac



    if exp_type.start_cycle_active:
        avoid = None
        if metadata.AC is not None and metadata.AC == ValidMetadata.CATHODE and metadata.AC_increment is not None:
            avoid = metadata.AC_increment
        start_cycle = get_start_cycle(fileList, avoid)
        if start_cycle is None:
            valid = False
        else:
            metadata.start_cycle = start_cycle


    if exp_type.voltage_active:
        voltage = get_voltage(fileList)
        if voltage is None:
            valid = False
        else:
            metadata.voltage = voltage

    if exp_type.temperature_active:
        temperature = get_temperature(fileList)
        if temperature is None:
            valid = False
        else:
            metadata.temperature = temperature

    if exp_type.drive_profile_active:
        drive_profiles = get_possible_drive_profiles(fileList)
        if not len(drive_profiles) == 0:
            if not exp_type.start_cycle_active or metadata.start_cycle is None:
                dp = drive_profiles[0]
                for key in dp.keys():
                    setattr(metadata, key, dp[key])
            else:
                for dp in drive_profiles:
                    if dp['drive_profile'].test == 'CX' and dp['drive_profile_x_denominator'] == metadata.start_cycle:
                        continue
                    dp = drive_profiles[0]
                    for key in dp.keys():
                        setattr(metadata, key, dp[key])
                    break


    if exp_type.version_number_active:
        version_number = get_version_number(fileList)
        if version_number is None:
            valid = False
        else:
            metadata.version_number = version_number

    print("\t\tEXTRACTED METADATA: {}".format(metadata))
    return metadata, valid











