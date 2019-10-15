from django.db import models
import datetime
import neware_parser.models
# Create your models here.
#---------------- Drive Profile Logic Function ------------------------#
class Category(models.Model):
    name = models.CharField(max_length=100)
    def __str__(self):
        return self.name


class SubCategory(models.Model):
    name = models.CharField(max_length=20)
    def __str__(self):
        return self.name

class ExperimentType(models.Model):
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    subcategory = models.ForeignKey(SubCategory, on_delete=models.CASCADE)
    barcode_active = models.BooleanField(default=True)
    start_cycle_active = models.BooleanField(default=True)
    voltage_active = models.BooleanField(default=True)
    voltage_name = models.CharField(max_length = 20, default = 'upper_cutoff_voltage')
    temperature_active = models.BooleanField(default=True)
    temperature_name = models.CharField(max_length=20, default = 'temperature')
    drive_profile_active = models.BooleanField(default=False)
    AC_active = models.BooleanField(default=False)
    AC_increment_active = models.BooleanField(default=False)
    charger_active = models.BooleanField(default=False)
    version_number_active = models.BooleanField(default=False)
    charger = models.CharField(max_length=5, default = '')
    shorthand = models.CharField(max_length=10, default = '')
    def __str__(self):
        return '{} ({})'.format(self.subcategory.name, self.category.name)


class ChargerDriveProfile(models.Model):
    drive_profile = models.CharField(max_length=20)
    test = models.CharField(max_length=200)
    description = models.CharField(max_length=1000)
    x_name = models.CharField(max_length=20)
    y_name = models.CharField(max_length=20)
    z_name = models.CharField(max_length=20)
    x_active = models.BooleanField(default=True)
    y_active = models.BooleanField(default=False)
    z_active = models.BooleanField(default=False)
    def __str__(self):
        return '{} ({})'.format(self.test, self.drive_profile)

def print_voltage(x):
    print("x = {}".format(x))

    if round((x * 100)) % 10 != 0:
        y = str(int(x * 100))
    elif round((x * 100)) % 10 == 0:
        y = str(int(x * 10))

    if x < 1:
        y = '0' + y

    if x == 0:
        y = "00"

    if x >= 10:
        y = "voltage was invalid: {} volts is too high. Only supports voltages less than 10.".format(x)
    return y

class ValidMetadata(models.Model):
    ANODE = 'A'
    CATHODE = 'C'
    AC_CHOICES = [
        (ANODE, 'Anode'),
        (CATHODE, 'Cathode'),

    ]
    experiment_type = models.ForeignKey(ExperimentType, on_delete=models.CASCADE, null=True)
    charID = models.CharField(max_length=5, null=True)
    barcode = models.IntegerField(null=True)

    start_cycle = models.IntegerField(null=True)
    voltage = models.FloatField(null=True)
    temperature = models.IntegerField(null=True)

    AC = models.CharField(
        max_length=1,
        choices=AC_CHOICES,
        null=True)
    AC_increment = models.IntegerField(null=True)
    version_number = models.IntegerField(null=True)

    drive_profile = models.ForeignKey(ChargerDriveProfile, on_delete=models.CASCADE,null=True)
    drive_profile_x_numerator = models.IntegerField(null=True)
    drive_profile_x_denominator = models.IntegerField(null=True)

    drive_profile_y_numerator = models.IntegerField(null=True)
    drive_profile_y_denominator = models.IntegerField(null=True)
    drive_profile_z = models.FloatField(null=True)
    date = models.DateField(null=True)

    def __str__(self):
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            self.experiment_type,
            self.charID,
            self.barcode,
        self.start_cycle,
        self.voltage,
        self.temperature,
        self.AC,
        self.AC_increment,
        self.version_number,
        self.drive_profile,
        self.drive_profile_x_numerator,
        self.drive_profile_x_denominator,
        self.drive_profile_y_numerator,
        self.drive_profile_y_denominator,
        self.drive_profile_z,
        self.date)

    @property
    def is_valid(self):
        return (( self.charID is not None) and
                (not self.experiment_type.barcode_active or self.barcode is not None) and
                (not self.experiment_type.start_cycle_active or self.start_cycle is not None) and
                (not self.experiment_type.voltage_active or self.voltage is not None) and
                (not self.experiment_type.temperature_active or self.temperature is not None) and
                (not self.experiment_type.AC_active or self.AC is not None) and
                (not self.experiment_type.AC_increment_active or self.AC_increment is not None) and
                (not self.experiment_type.version_number_active or self.version_number is not None) and
                ( self.date is not None))
    @property
    def get_profile(self):
        x_value = ''
        y_value = ''
        z_value = ''

        if self.drive_profile.x_active:
            x_value = self.drive_profile.x_name + '=' + str(self.drive_profile_x_numerator) + '/' + str(
                self.drive_profile_x_denominator)
        if self.drive_profile.y_active:
            y_value = ', ' + self.drive_profile.y_name + '=' + str(self.drive_profile_y_numerator) + '/' + str(
                self.drive_profile_y_denominator)
        if self.drive_profile.z_active:
            z_value = ', ' + self.drive_profile.z_name + '=' + str(self.drive_profile_z)

        return x_value \
               + y_value \
               + z_value

    @property


    # TODO: If statements could still be reduced and charger could be used more, needs more attention

    def get_filename(self):

        filename = ''

        print('Exp type:' + str(self.experiment_type))
        print('shorthand:' + str(self.experiment_type.shorthand))

        if self.experiment_type.subcategory != 'exsitu':
            filename += str(self.charID) + '_' + str(self.experiment_type.shorthand)

            if self.experiment_type.AC_active:
                filename += '_' + str(self.AC)

            if self.experiment_type.barcode_active:
                filename += '_' + str(self.barcode)

            if self.experiment_type.charger != '':
                filename += '_' + str(self.experiment_type.charger)

            if self.experiment_type.start_cycle_active:
                filename += '_c' + str(self.start_cycle)

            if self.experiment_type.voltage_active:

                x = self.voltage
                print("x = {}".format(x))

                if (x * 100) % 10 != 0:
                    y = str(int(x * 100))
                elif (x * 100) % 10 == 0:
                    y = str(int(x * 10))

                if x < 1:
                    y = '0' + y

                if x == 0:
                    y = "00"

                if x >= 10:
                    return "voltage was invalid: {} volts is too high. Only supports voltages less than 10.".format(x)

                filename += '_' + str(y) + 'V'

            if self.experiment_type.temperature_active:
                filename += '_' + str(self.temperature) + 'C'

            if self.experiment_type.drive_profile_active:
                filename += '_' + str(print_drive_profile(xnum=self.drive_profile_x_numerator,
                    xden=self.drive_profile_x_denominator,
                    ynum=self.drive_profile_y_numerator,
                    yden=self.drive_profile_y_denominator,
                    z=self.drive_profile_z,
                    template_name=self.drive_profile.drive_profile))

            filename += '_' + str(self.date.strftime("%y%m%d"))

        elif self.experiment_type.subcategory == 'exsitu':

            filename += 'Ex=situ Gas Checkin_v{}.xls'.format(str(self.version_number))


        return filename

class DatabaseFile(models.Model):
    '''
    Note that valid_metadata is null if filename hasn't been parsed.
    '''
    filename = models.CharField(max_length=200)
    root = models.CharField(max_length=200)
    last_modified = models.DateTimeField(default=datetime.datetime(1970, 1, 1))
    filesize = models.IntegerField(default=0) # in bytes
    valid_metadata = models.OneToOneField(ValidMetadata, on_delete=models.SET_NULL, null=True)
    is_valid = models.BooleanField(default=False)
    deprecated = models.BooleanField(default=False)
    def __str__(self):
        return self.filename
    def set_deprecated(self, new_deprecated):
        if new_deprecated == self.deprecated:
            return
        self.deprecated = new_deprecated
        self.save()
        if not neware_parser.models.CyclingFile.objects.filter(database_file=self).exists():
            return
        if ((self.valid_metadata is None) or (not self.is_valid) or (self.valid_metadata.experiment_type != ExperimentType.objects.get(
                    category=Category.objects.get(name='cycling'),
                    subcategory=SubCategory.objects.get(name='neware')))):
            if not new_deprecated:
                #this means that some invalid data sneaked into the CyclingFiles in deprecated form.
                neware_parser.models.CyclingFile.objects.filter(database_file=self).delete()
            return

        #at this point, we know that the barcode exists.
        neware_parser.models.BarcodeNode.objects.filter(barcode=self.valid_metadata.barcode).update(valid_cache=False)
    def set_valid_metadata(self,
                        valid_metadata = None,
                        experiment_type = None,
                        charID = None,
                        barcode = None,
                        start_cycle = None,
                        voltage = None,
                        temperature = None,
                        AC = None,
                        AC_increment = None,
                        version_number = None,
                        drive_profile=None,
                        drive_profile_x_numerator=None,
                        drive_profile_x_denominator=None,
                        drive_profile_y_numerator=None,
                        drive_profile_y_denominator=None,
                        drive_profile_z=None,
                        date = None):

        if self.valid_metadata is not None:
            print('datafile metadata was', self.valid_metadata)
            if valid_metadata is not None:
                # both exist.
                if ((self.valid_metadata.experiment_type == valid_metadata.experiment_type) and
                    (self.valid_metadata.charID == valid_metadata.charID) and
                    (self.valid_metadata.barcode == valid_metadata.barcode) and
                    (self.valid_metadata.voltage == valid_metadata.voltage) and
                    (self.valid_metadata.temperature == valid_metadata.temperature) and
                    (self.valid_metadata.date == valid_metadata.date) and
                    (self.valid_metadata.version_number == valid_metadata.version_number) and
                    (self.valid_metadata.AC == valid_metadata.AC) and
                    (self.valid_metadata.AC_increment == valid_metadata.AC_increment) and
                    (self.valid_metadata.drive_profile == valid_metadata.drive_profile) and
                    (self.valid_metadata.drive_profile_x_numerator == valid_metadata.drive_profile_x_numerator) and
                    (self.valid_metadata.drive_profile_x_denominator == valid_metadata.drive_profile_x_denominator) and
                    (self.valid_metadata.drive_profile_y_numerator == valid_metadata.drive_profile_y_numerator) and
                    (self.valid_metadata.drive_profile_y_denominator == valid_metadata.drive_profile_y_denominator) and
                    (self.valid_metadata.drive_profile_z == valid_metadata.drive_profile_z)

                ):
                    return

            else:
                if (
                    ((experiment_type is None) or (experiment_type == self.valid_metadata.experiment_type)) and
                    ((charID is None) or (charID == self.valid_metadata.charID)) and
                    ((barcode is None) or (barcode == self.valid_metadata.barcode)) and
                    ((start_cycle is None) or (start_cycle == self.valid_metadata.start_cycle)) and
                    ((voltage is None) or (voltage == self.valid_metadata.voltage)) and
                    ((temperature is None) or (temperature == self.valid_metadata.temperature)) and
                    ((AC is None) or (AC == self.valid_metadata.AC)) and
                    ((AC_increment is None) or (AC_increment == self.valid_metadata.AC_increment)) and
                    ((version_number is None) or (version_number == self.valid_metadata.version_number)) and
                    ((drive_profile is None) or (drive_profile == self.valid_metadata.drive_profile)) and
                    ((drive_profile_x_numerator is None) or (drive_profile_x_numerator == self.valid_metadata.drive_profile_x_numerator)) and
                    ((drive_profile_x_denominator is None) or (drive_profile_x_denominator == self.valid_metadata.drive_profile_x_denominator)) and
                    ((drive_profile_y_numerator is None) or (drive_profile_y_numerator == self.valid_metadata.drive_profile_y_numerator)) and
                    ((drive_profile_y_denominator is None) or (drive_profile_y_denominator == self.valid_metadata.drive_profile_y_denominator)) and
                    ((drive_profile_z is None) or (drive_profile_z == self.valid_metadata.drive_profile_z)) and
                    ((date is None) or (date == self.valid_metadata.date))):

                    return

            cycling_file = None
            if neware_parser.models.CyclingFile.objects.filter(database_file=self).exists():
                cycling_file = neware_parser.models.CyclingFile.objects.get(database_file=self)

            # if there was a cycling file, and if it was not deprecated, invalidate origin barcode.
            if not self.deprecated and cycling_file is not None:
                neware_parser.models.BarcodeNode.objects.filter(barcode=self.valid_metadata.barcode).update(valid_cache=False)

            if valid_metadata is not None:
                # get the new one.
                self.valid_metadata.delete()
                valid_metadata.save()
                self.valid_metadata = valid_metadata
            else:
                if experiment_type is not None:
                    self.valid_metadata.experiment_type = experiment_type
                if charID is not None:
                    self.valid_metadata.charID = charID
                if barcode is not None:
                    self.valid_metadata.barcode = barcode
                if start_cycle is not None:
                    self.valid_metadata.start_cycle = start_cycle
                if voltage is not None:
                    self.valid_metadata.voltage = voltage
                if temperature is not None:
                    self.valid_metadata.temperature = temperature
                if AC is not None:
                    self.valid_metadata.AC = AC
                if AC_increment is not None:
                    self.valid_metadata.AC_increment = AC_increment
                if version_number is not None:
                    self.valid_metadata.version_number = version_number
                if drive_profile is not None:
                    self.valid_metadata.drive_profile = drive_profile
                if drive_profile_x_numerator is not None:
                    self.valid_metadata.drive_profile_x_numerator = drive_profile_x_numerator
                if drive_profile_x_denominator is not None:
                    self.valid_metadata.drive_profile_x_denominator = drive_profile_x_denominator
                if drive_profile_y_numerator is not None:
                    self.valid_metadata.drive_profile_y_numerator = drive_profile_y_numerator
                if drive_profile_y_denominator is not None:
                    self.valid_metadata.drive_profile_y_denominator = drive_profile_y_denominator
                if drive_profile_z is not None:
                    self.valid_metadata.drive_profile_z = drive_profile_z
                if date is not None:
                    self.valid_metadata.date = date

            self.valid_metadata.save()
            self.is_valid = self.valid_metadata.is_valid
            v = (self.is_valid and
                 (self.valid_metadata.experiment_type ==  ExperimentType.objects.get(
                    category=Category.objects.get(name='cycling'),
                    subcategory=SubCategory.objects.get(name='neware')))
                 )
            if cycling_file is not None:
                if not v:
                    cycling_file.delete()
                else:
                    if not self.deprecated:
                        neware_parser.models.BarcodeNode.objects.filter(barcode=self.valid_metadata.barcode).update(
                            valid_cache=False)
            self.save()

        else:
            if valid_metadata is not None:

                valid_metadata.save()
                self.valid_metadata = valid_metadata
                self.is_valid = self.valid_metadata.is_valid
                self.save()
            else:
                return

