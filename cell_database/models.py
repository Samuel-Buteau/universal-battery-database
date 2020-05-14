from django.db import models
from django.db.models import Q, F, Func, Count,Exists, OuterRef
import datetime
import numpy
import re

from enum import Enum


class LotTypes(Enum):
    none = 0
    unknown = 1
    no_lot = 2
    lot = 3


def decode_lot_string(s):

    if s is None or s == '':
        return (None, LotTypes.none)
    if s == '?':
        return (None, LotTypes.unknown)
    if s.endswith('_lot'):
        return (int(s.split('_lot')[0]), LotTypes.lot)
    else:
        return (int(s), LotTypes.no_lot)


def encode_lot_string(ob, lot_type):
    if (lot_type == LotTypes.none) or (lot_type == LotTypes.lot and ob is None) or (
            lot_type == LotTypes.no_lot and ob is None):
        return None

    if lot_type == LotTypes.unknown:
        return '?'
    if lot_type == LotTypes.lot:
        return '{}_lot'.format(ob.id)
    if lot_type == LotTypes.no_lot:
        return '{}'.format(ob.id)


def make_choices(no_lots=None, lots=None, none=False, unknown=False):
    res = []

    if none:
        res.append((encode_lot_string(None, LotTypes.none), '-----'))


    if unknown:
        res.append((encode_lot_string(None, LotTypes.unknown), 'UNKNOWN'))


    if no_lots is not None:
        res = res + [(encode_lot_string(c, LotTypes.no_lot), c.__str__()) for c in no_lots]



    if lots is not None:
        res = res + [(encode_lot_string(c, LotTypes.lot), c.__str__()) for c in lots]

    return sorted(res, key=lambda x: x[1])
    # return res

def unknown_numerical(num_string):
    if num_string is None or num_string == '':
        return None, False
    elif num_string == '?':
        return None, True
    elif num_string.lstrip('-').replace('.','',1).isdigit():
        return float(num_string), False
    else:
        return None, False


def determine_digits(num):
    dig = max(int(-(numpy.floor(numpy.log(num) / numpy.log(10.)) - 1)), 0)
    if (round(num*(10**dig)) % 10) ==0 and dig > 0:
        dig = dig -1

    return dig

def print_digits(num, digits=None):
    if digits is None:
        digits = determine_digits(num)

    format_str = '{:2.' +str(digits) +'f}'
    if digits == 0:
        if str(int(num)) == '1':
            return ''
        return str(int(num))
    else:
        return format_str.format(num)

def print_components(list_of_components, complete=False, prefix=None):
    if len(list_of_components) != 0:
        if complete and len(list_of_components) == 1:
            component_string = list_of_components[0].component_lot.__str__()
        else:
            digit = max([determine_digits(comp.ratio) for comp in list_of_components if comp.ratio is not None], default=0)
            component_string = '+'.join([comp.pretty_print(digits=digit) for comp in list_of_components])
        if prefix is None:
            return component_string
        else:
            return "{}:({})".format(prefix, component_string)
    else:
        return None

def print_lot(lot, type=None):
    if type == 'coating':
        sub = lot.coating.__str__()
    elif type == 'component':
        sub = lot.component.__str__()
    elif type == 'composite':
        sub = lot.composite.__str__()
    elif type == 'dry_cell':
        sub = lot.dry_cell.__str__()
    else:
        raise('Not implemented {}'.format(type))

    if lot.lot_info is None:
        return sub
    else:
        return "{}({})".format(lot.lot_info.__str__(), sub)


def print_unknown(val):
    if val is None:
        ret = '?'
    else:
        ret = val

    return ret

class LotInfo(models.Model):
    notes = models.CharField(max_length=100, null=True, blank=True)


    """
     dates, creator, notes, vendor are either unknown or known 
    """
    creator = models.CharField(max_length=100, null=True, blank=True)
    creator_name = models.BooleanField(default=False, blank=True)
    date = models.DateField(null=True, blank=True, help_text='YYYY-MM-DD')
    date_name = models.BooleanField(default=False, blank=True)
    vendor = models.CharField(max_length=300, null=True, blank=True)
    vendor_name = models.BooleanField(default=False, blank=True)


    def is_valid(self):
        if self.notes is None and not self.creator_name and not self.date_name and not self.vendor_name:
            return False
        if self.notes is None and self.creator is None and self.date is None and self.vendor is None:
            return False

        if self.vendor == '?' or self.creator == '?':
            return False

        if self.vendor_name and self.vendor is None:
            return False

        if self.creator_name and self.creator is None:
            return False

        if self.date_name and self.date is None:
            return False

        return True

    def __str__(self):
        printed_name = ''
        if self.notes is not None:
            printed_name = self.notes

        extras = []
        if self.vendor_name:
            extras.append('VEN:{}'.format(self.vendor))
        if self.creator_name:
            extras.append('BY:{}'.format(self.creator))
        if self.date_name:
            extras.append('DATE:{}'.format(self.date))

        if len(extras) != 0:
            printed_name = '{}[{}]'.format(printed_name, ','.join(extras))

        return printed_name



def define_if_possible(lot, lot_info=None, type =None):
    """
        The objects are made of subobjects and visibility flags.
        we want to make sure that among the objects created, no two (object,visibility) have the same (object projection)
        furthermore, no two objects can have the same string.

        If there is an object clash, return the preexisting object.
        Else: if there is a string clash, set all visibility flags to True.
    """
    print('entered define if possible')
    if lot_info is None or not lot_info.is_valid():
        print('was invalid')
        return None
    if type == 'coating':
        object_equality_query = Q(coating=lot.coating)
    elif type == 'component':
        object_equality_query = Q(component=lot.component)
    elif type == 'composite':
        object_equality_query = Q(composite=lot.composite)
    elif type == 'dry_cell':
        object_equality_query = Q(dry_cell=lot.dry_cell)
    else:
        raise('not yet implemented {}'.format(type))

    object_equality_query = object_equality_query & Q(
        lot_info__notes=lot_info.notes,
        lot_info__creator=lot_info.creator,
        lot_info__date=lot_info.date,
        lot_info__vendor=lot_info.vendor
    )

    string_equality_query = Q(lot_info__notes=lot_info.notes)

    if type == 'coating':
        string_equality_query = string_equality_query & Q(coating=lot.coating)
    elif type == 'component':
        string_equality_query = string_equality_query & Q(component=lot.component)
    elif type == 'composite':
        string_equality_query = string_equality_query & Q(composite=lot.composite)
    elif type == 'dry_cell':
        string_equality_query = string_equality_query & Q(dry_cell=lot.dry_cell)
    else:
        raise ('not yet implemented {}'.format(type))


    if lot_info.creator_name:
        string_equality_query = string_equality_query & Q(lot_info__creator=lot_info.creator,
                                                          lot_info__creator_name=True)
    else:
        string_equality_query = string_equality_query & Q(lot_info__creator_name=False)
    if lot_info.date_name:
        string_equality_query = string_equality_query & Q(lot_info__date=lot_info.date,
                                                          lot_info__date_name=True)
    else:
        string_equality_query = string_equality_query & Q(lot_info__date_name=False)
    if lot_info.vendor_name:
        string_equality_query = string_equality_query & Q(lot_info__vendor=lot_info.vendor,
                                                          lot_info__vendor_name=True)
    else:
        string_equality_query = string_equality_query & Q(lot_info__vendor_name=False)


    if type == 'coating':
        objs = CoatingLot.objects
    elif type == 'component':
        objs = ComponentLot.objects
    elif type == 'composite':
        objs = CompositeLot.objects
    elif type == 'dry_cell':
        objs = DryCellLot.objects
    else:
        raise ('not yet implemented {}'.format(type))

    objs = objs.exclude(lot_info=None)
    set_of_object_equal = objs.filter(object_equality_query)
    string_equals = objs.filter(string_equality_query).exists()

    if (not set_of_object_equal.exists()) and string_equals:
        lot_info.creator_name = True
        lot_info.date_name = True
        lot_info.vendor_name = True

    return helper_return(
        set_of_object_equal=set_of_object_equal,
        my_self=lot,
        lot_info=lot_info
    )





class ElectrodeGeometry(models.Model):
    UNITS_LOADING = ('milligrams per squared centimeters', 'mg/cm^2')
    UNITS_DENSITY = ('grams per cubic centimeters', 'g/cm^3')
    UNITS_POROSITY = 'TODO(sam): I DON"T KNOW THESE UNITS'
    UNITS_THICKNESS = ('micrometers', 'um')
    UNITS_LENGTH = ('millimeters', 'mm')
    UNITS_WIDTH = ('millimeters','mm')
    UNITS_TAB_POSITION_FROM_CORE = 'TODO(sam): I DON"T KNOW THESE UNITS'

    loading = models.FloatField(null=True, blank=True, help_text=UNITS_LOADING[0])
    loading_name = models.BooleanField(default=False, blank=True)

    density = models.FloatField(null=True, blank=True, help_text=UNITS_DENSITY[0])
    density_name = models.BooleanField(default=False, blank=True)
    #porosity = models.FloatField(null=True, blank=True)

    thickness = models.FloatField(null=True, blank=True, help_text=UNITS_THICKNESS[0])
    thickness_name = models.BooleanField(default=False, blank=True)
    #length_single_side = models.FloatField(null=True, blank=True)
    #length_double_side = models.FloatField(null=True, blank=True)
    #width = models.FloatField(null=True, blank=True)
    #tab_position_from_core = models.FloatField(null=True, blank=True)
    #foil_thickness = models.FloatField(null=True, blank=True)

class SeparatorGeometry(models.Model):
    UNITS = ('millimeters','mm')
    thickness = models.FloatField(null=True, blank=True, help_text=UNITS[0])
    thickness_name = models.BooleanField(default=False, blank=True)

    width = models.FloatField(null=True, blank=True, help_text=UNITS[0])
    width_name = models.BooleanField(default=False, blank=True)

    #overhang_in_core = models.FloatField(null=True, blank=True)


class Coating(models.Model):
    notes = models.CharField(max_length=100, null=True, blank=True)
    proprietary = models.BooleanField(default=False, blank=True)
    proprietary_name = models.BooleanField(default=False, blank=True)

    description = models.CharField(max_length=1000, null=True, blank=True)
    description_name = models.BooleanField(default=False, blank=True)

    def __str__(self):
        printed_name = ''
        extras = []
        if self.notes is not None:
            if printed_name == '':
                printed_name = self.notes
            else:
                printed_name = '{} ({})'.format(printed_name,self.notes)

        if self.proprietary_name:
            if self.proprietary:
                secret = "SECRET"
            else:
                secret = "NOT SECRET"
            extras.append(secret)

        if self.description_name:
            if self.description is None:
                desc = ""
            else:
                desc = self.description
            extras.append('DESCRIPTION:{}'.format(desc))


        if len(extras) != 0:
            return "{} [{}]".format(printed_name, ','.join(extras))
        else:
            return printed_name

    def define_if_possible(self, target=None):

        object_equality_query = Q(
            proprietary=self.proprietary,
            notes=self.notes,
            description=self.description,
        )


        string_equality_query = Q(notes=self.notes)

        if self.proprietary_name:
            string_equality_query = string_equality_query & Q(proprietary=self.proprietary,
                                                              proprietary_name=True)
        else:
            string_equality_query = string_equality_query & Q(proprietary_name=False)


        if self.description_name:
            string_equality_query = string_equality_query & Q(description=self.description,
                                                              description_name=True)
        else:
            string_equality_query = string_equality_query & Q(description_name=False)


        target_object = None
        if target is None:

            coating_set = Coating.objects
            target_object = None
        else:
            coating_set = Coating.objects.exclude(id = target)
            if Coating.objects.filter(id=target).exists():
                target_object = Coating.objects.get(id=target)

        set_of_object_equal = coating_set.filter(object_equality_query)
        print('set objects equal: ', set_of_object_equal)
        string_equals = coating_set.filter(string_equality_query).exists()
        print('set of string equal: ', coating_set.filter(string_equality_query))

        if (not set_of_object_equal.exists()) and string_equals:
            self.proprietary_name = True
            self.description_name = True


        if target_object is None:
            my_self = self
        else:
            my_self = target_object
            my_self.notes = self.notes
            my_self.proprietary = self.proprietary
            my_self.proprietary_name = self.proprietary_name
            my_self.description_name = self.description_name
            my_self.description = self.description

        return helper_return(
                set_of_object_equal=set_of_object_equal,
                my_self=my_self,
            )

class CoatingLot(models.Model):
    coating = models.ForeignKey(Coating, on_delete=models.CASCADE, blank=True)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True, blank=True)
    def __str__(self):
        return print_lot(self, type='coating')



SINGLE_CRYSTAL = 'sc'
POLY_CRYSTAL = 'po'
MIXED_CRYSTAL = 'mx'
UNKNOWN_CRYSTAL = 'un'
CRYSTAL_TYPES = [
    (SINGLE_CRYSTAL, 'Single'),
    (POLY_CRYSTAL, 'Poly'),
    (MIXED_CRYSTAL, 'Mixed'),
]




SALT = 'sa'
ADDITIVE = 'ad'
SOLVENT = 'so'
ACTIVE_MATERIAL = 'am'
CONDUCTIVE_ADDITIVE = 'co'
BINDER = 'bi'
SEPARATOR_MATERIAL = 'se'

COMPONENT_TYPES = [
    (SALT, 'salt'),
    (ADDITIVE, 'additive'),
    (SOLVENT, 'solvent'),
    (ACTIVE_MATERIAL, 'active_material'),
    (CONDUCTIVE_ADDITIVE, 'conductive_additive'),
    (BINDER, 'binder'),
    (SEPARATOR_MATERIAL, 'separator_material'),
]

ELECTROLYTE = 'el'
CATHODE = 'ca'
ANODE = 'an'
SEPARATOR = 'se'
COMPOSITE_TYPES = [
    (ELECTROLYTE, 'electrolyte'),
    (CATHODE, 'cathode'),
    (ANODE, 'anode'),
    (SEPARATOR, 'separator'),
]

ELECTRODE = 'ed'
COMPOSITE_TYPES_2 = [
    (ELECTROLYTE, 'electrolyte'),
    (ELECTRODE, 'electrode'),
    (SEPARATOR, 'separator'),
]

class ElectrodeMaterialStochiometry(models.Model):
    LITHIUM = 'Li'
    OXIGEN = 'O'
    CARBON = 'C'
    NICKEL = 'Ni'
    MANGANESE = 'Mn'
    COBALT = 'Co'
    MAGNESIUM = 'Mg'
    ALUMINUM = 'Al'
    IRON = 'Fe'
    PHOSPHORUS= 'P'
    TITANIUM = 'Ti'
    SULFUR = 'S'
    SODIUM = 'Na'
    FLUORINE = 'F'
    CHLORINE = 'Cl'
    COPPER = 'Cu'
    ZINC = 'Zn'
    MOLYBDENUM = 'Mo'
    NIOBIUM = 'Nb'
    SILICON = 'Si'
    PLATINUM = 'Pt'
    ATOMS = [
        (LITHIUM, 'LITHIUM'),
        (OXIGEN, 'OXIGEN'),
        (CARBON, 'CARBON'),
        (NICKEL, 'NICKEL'),
        (MANGANESE, 'MANGANESE'),
        (COBALT, 'COBALT'),
        (MAGNESIUM, 'MAGNESIUM'),
        (ALUMINUM, 'ALUMINUM'),
        (IRON, 'IRON'),
        (PHOSPHORUS, 'PHOSPHORUS'),
        (TITANIUM, 'TITANIUM'),
        (SULFUR, 'SULFUR'),
        (SODIUM, 'SODIUM'),
        (FLUORINE, 'FLUORINE'),
        (CHLORINE, 'CHLORINE'),
        (COPPER, 'COPPER'),
        (ZINC, 'ZINC'),
        (MOLYBDENUM, 'MOLYBDENUM'),
        (NIOBIUM, 'NIOBIUM'),
        (SILICON, 'SILICON'),
        (PLATINUM, 'PLATINUM'),
    ]
    atom = models.CharField(max_length=3, choices=ATOMS, blank=True)
    stochiometry = models.FloatField(blank=True, null=True)

    def pretty_print(self, digits=None):
        my_string = '{}{}'
        if self.stochiometry is not None:
            num = print_digits(self.stochiometry, digits)
        else:
            num = '?'
        return my_string.format(self.atom, num)

    def __str__(self):
        return self.pretty_print()

def helper_return(
        set_of_object_equal=None,
        my_self=None,
        cathode_geometry=None,
        anode_geometry=None,
        separator_geometry=None,
        my_stochiometry_components=None,
        my_ratio_components=None,
        lot_info=None,
        dry_cell_geometry=None,
        anode=None,
        cathode=None,
        separator=None,
):
    if not set_of_object_equal.exists():
        # if string equals, we take care of it before passing it here.

        # we know for a fact that value fields don't exist.
        # so we can create it as it is.
        if cathode_geometry is not None:
            cathode_geometry.save()
            my_self.cathode_geometry = cathode_geometry
        if anode_geometry is not None:
            anode_geometry.save()
            my_self.anode_geometry = anode_geometry

        if dry_cell_geometry is not None:
            dry_cell_geometry.save()
            my_self.geometry = dry_cell_geometry

        if anode is not None:
            anode.save()
            my_self.anode = anode

        if cathode is not None:
            cathode.save()
            my_self.cathode = cathode

        if separator is not None:
            separator.save()
            my_self.separator = separator

        if separator_geometry is not None:
            separator_geometry.save()
            my_self.separator_geometry = separator_geometry

        if lot_info is not None:
            lot_info.save()
            my_self.lot_info = lot_info

        my_self.save()

        if my_ratio_components is not None:
            my_self.components.set(my_ratio_components)
        if my_stochiometry_components is not None:
            my_self.stochiometry.set(my_stochiometry_components)

        return my_self

    else:
        return set_of_object_equal[0]



class Component(models.Model):
    '''
    molecule (salt)
    molecule (solvent)
    molecule (additive)

    electrode material (active_material)
    electrode material (binder)
    electrode material (conductive_additive)

    separator material

    '''
    """
    notes: None or known
    smiles: None or known
    proprietary: known
    composite_type: known
    component_type: known
    coating: known, no, unknown [can be used in name]
    particle_size: known, unknown [can be used in name]
    single_crystal: known, none 
    turbostratic_misalignment: known, unknown [can be used in name]
    preparation_temperature: known, unknown [can be used in name]
    natural: known, unknown [can be used in name]
    
    """
    UNITS_SIZE = ('micrometers','um')
    UNITS_TEMPERATURE = ('celsius','C')
    UNITS_MISALIGNMENT = ('percent', '%')

    notes = models.CharField(max_length=1000, null=True, blank=True)

    smiles = models.CharField(max_length=1000, null=True, blank=True)
    smiles_name = models.BooleanField(default=False, blank=True)

    proprietary = models.BooleanField(default=False, blank=True)
    proprietary_name = models.BooleanField(default=False, blank=True)

    composite_type = models.CharField(max_length=2, choices=COMPOSITE_TYPES_2, blank=True)
    composite_type_name = models.BooleanField(default=False, blank=True)

    component_type = models.CharField(max_length=2, choices=COMPONENT_TYPES, blank=True)
    component_type_name = models.BooleanField(default=False, blank=True)

    coating_lot = models.ForeignKey(CoatingLot, on_delete=models.SET_NULL, null=True, blank=True)
    coating_lot_name = models.BooleanField(default=False, blank=True)

    particle_size = models.FloatField(null=True, blank=True, help_text = UNITS_SIZE[0])
    particle_size_name = models.BooleanField(default=False, blank=True)

    single_crystal = models.CharField(max_length=2, choices=CRYSTAL_TYPES, null=True, blank=True)
    single_crystal_name = models.BooleanField(default=False, blank=True)

    turbostratic_misalignment = models.FloatField(null=True, blank=True, help_text = UNITS_MISALIGNMENT[0])
    turbostratic_misalignment_name = models.BooleanField(default=False, blank=True)

    preparation_temperature = models.FloatField(null=True, blank=True, help_text= UNITS_TEMPERATURE[0])
    preparation_temperature_name = models.BooleanField(default=False, blank=True)

    natural = models.BooleanField(null=True, blank=True)
    natural_name = models.BooleanField(default=False, blank=True)

    stochiometry = models.ManyToManyField(ElectrodeMaterialStochiometry)

    def is_valid(self):
        """
        if proprietary, needs a name.
        if not proprietary and active material, can't have a name.

        all fields which are null must have the corresponding _name field =False

        :return:
        """
        if self.composite_type is None:
            return False
        if self.component_type is None:
            return False

        return True

    def define_if_possible(self, atoms=None, target=None):
        """
        The objects are made of subobjects and visibility flags.
        we want to make sure that among the objects created, no two (object,visibility) have the same (object projection)
        furthermore, no two objects can have the same string.

        If there is an object clash, return the preexisting object.
        Else: if there is a string clash, set all visibility flags to True.
        """
        print('entered define if possible')
        if not self.is_valid():
            print('was invalid')
            return None

        object_equality_query = Q(
            proprietary=self.proprietary,
            smiles=self.smiles,
            composite_type=self.composite_type,
            component_type=self.component_type,
            notes=self.notes,
            coating_lot=self.coating_lot,
            particle_size=self.particle_size,
            single_crystal=self.single_crystal,
            turbostratic_misalignment=self.turbostratic_misalignment,
            preparation_temperature=self.preparation_temperature,
            natural=self.natural
        )


        string_equality_query = Q()
        if self.proprietary_name:
            string_equality_query = string_equality_query & Q(proprietary=self.proprietary,
                                                              proprietary_name=True)
        else:
            string_equality_query = string_equality_query & Q(proprietary_name=False)

        if self.smiles_name:
            string_equality_query = string_equality_query & Q(smiles=self.smiles,
                                                              smiles_name=True)
        else:
            string_equality_query = string_equality_query & Q(smiles_name=False)

        if self.composite_type_name:
            string_equality_query = string_equality_query & Q(composite_type=self.composite_type,
                                                              composite_type_name=True)
        else:
            string_equality_query = string_equality_query & Q(composite_type_name=False)
        if self.component_type_name:
            string_equality_query = string_equality_query & Q(component_type=self.component_type,
                                                              component_type_name=True)
        else:
            string_equality_query = string_equality_query & Q(component_type_name=False)

        string_equality_query = string_equality_query & Q(notes=self.notes)

        if self.coating_lot_name:
            string_equality_query = string_equality_query & Q(coating_lot=self.coating_lot,
                                                              coating_lot_name=True)
        else:
            string_equality_query = string_equality_query & Q(coating_lot_name=False)
        if self.particle_size_name:
            string_equality_query = string_equality_query & Q(particle_size=self.particle_size,
                                                              particle_size_name=True)
        else:
            string_equality_query = string_equality_query & Q(particle_size_name=False)
        if self.single_crystal_name:
            string_equality_query = string_equality_query & Q(single_crystal=self.single_crystal,
                                                              single_crystal_name=True)
        else:
            string_equality_query = string_equality_query & Q(single_crystal_name=False)
        if self.turbostratic_misalignment_name:
            string_equality_query = string_equality_query & Q(turbostratic_misalignment=self.turbostratic_misalignment,
                                                              turbostratic_misalignment_name=True)
        else:
            string_equality_query = string_equality_query & Q(turbostratic_misalignment_name=False)
        if self.preparation_temperature_name:
            string_equality_query = string_equality_query & Q(preparation_temperature=self.preparation_temperature,
                                                              preparation_temperature_name=True)
        else:
            string_equality_query = string_equality_query & Q(preparation_temperature_name=False)
        if self.natural_name:
            string_equality_query = string_equality_query & Q(natural=self.natural,
                                                              natural_name=True)
        else:
            string_equality_query = string_equality_query & Q(natural_name=False)


        if atoms is None:
            atoms = []
        all_ids = list(map(lambda x: x['atom'], atoms))
        if len(set(all_ids)) != len(all_ids):
            print('atoms were not unique')
            return None
        else:
            tolerance = 0.01
            my_stochiometry_components = []
            for atom in atoms:
                actual_atom = atom['atom']
                stochiometry = atom['stochiometry']

                if stochiometry is not None:
                    stochiometry_components = ElectrodeMaterialStochiometry.objects.filter(
                        atom=actual_atom,
                        stochiometry__range=(stochiometry - tolerance, stochiometry + tolerance))
                    if stochiometry_components.exists():
                        selected_stochiometry_component = stochiometry_components.annotate(
                            distance=Func(F('stochiometry') - stochiometry, function='ABS')).order_by('distance')[0]
                    else:
                        selected_stochiometry_component = ElectrodeMaterialStochiometry.objects.create(
                            atom=actual_atom,
                            stochiometry=stochiometry
                        )
                else:
                    stochiometry_components = ElectrodeMaterialStochiometry.objects.filter(
                        atom=actual_atom,
                        stochiometry=None)
                    if stochiometry_components.exists():
                        selected_stochiometry_component = stochiometry_components.order_by('id')[0]
                    else:
                        selected_stochiometry_component = ElectrodeMaterialStochiometry.objects.create(
                            atom=actual_atom,
                            stochiometry=None
                        )

                my_stochiometry_components.append(selected_stochiometry_component)

            print('gathered the following stochiometry:', my_stochiometry_components)

            target_object = None
            if target is None:

                component_set = Component.objects
                target_object = None
            else:
                component_set = Component.objects.exclude(id=target)
                if Component.objects.filter(id=target).exists():
                    target_object = Component.objects.get(id=target)

            set_with_valid_stoc = component_set.annotate(
                count_stochiometry=Count('stochiometry')
            ).filter(count_stochiometry=len(my_stochiometry_components)
                     ).annotate(
                count_valid_stochiometry=Count('stochiometry', filter=Q(stochiometry__in=my_stochiometry_components))
            ).filter(count_valid_stochiometry=len(my_stochiometry_components))



            set_of_object_equal = set_with_valid_stoc.filter(object_equality_query)
            string_equals = set_with_valid_stoc.filter(string_equality_query).exists()

            if (not set_of_object_equal.exists()) and string_equals:
                self.proprietary_name = True
                self.smiles_name = True
                self.composite_type_name = True
                self.component_type_name = True
                self.coating_lot_name = True
                self.particle_size_name = True
                self.single_crystal_name = True
                self.turbostratic_misalignment_name = True
                self.preparation_temperature_name = True
                self.natural_name = True

            if target_object is None:
                my_self = self
            else:
                my_self = target_object
                my_self.notes = self.notes
                my_self.proprietary = self.proprietary
                my_self.proprietary_name = self.proprietary_name
                my_self.smiles = self.smiles
                my_self.smiles_name = self.smiles_name
                my_self.composite_type = self.composite_type
                my_self.composite_type_name = self.composite_type_name
                my_self.component_type = self.component_type
                my_self.component_type_name = self.component_type_name

                my_self.coating_lot_name = self.coating_lot_name

                my_self.particle_size = self.particle_size
                my_self.particle_size_name = self.particle_size_name
                my_self.single_crystal = self.single_crystal
                my_self.single_crystal_name = self.single_crystal_name
                my_self.turbostratic_misalignment = self.turbostratic_misalignment
                my_self.turbostratic_misalignment_name = self.turbostratic_misalignment_name
                my_self.preparation_temperature = self.preparation_temperature
                my_self.preparation_temperature_name = self.preparation_temperature_name
                my_self.natural = self.natural
                my_self.natural_name = self.natural_name


            return helper_return(
                set_of_object_equal=set_of_object_equal,
                my_self=my_self,
                my_stochiometry_components=my_stochiometry_components
            )


    def print_stochiometry(self):
        if self.stochiometry.count() >=1:
            list_of_stochiometry =list(self.stochiometry.order_by('-stochiometry'))
            digit = max([determine_digits(stoc.stochiometry) for stoc in list_of_stochiometry if stoc.stochiometry is not None], default=0)
            return ' '.join([stoc.pretty_print(digits=digit) for stoc in self.stochiometry.order_by('-stochiometry')])
        else:
            return ''

    def __str__(self):
        printed_name = ''
        if self.component_type == ACTIVE_MATERIAL:
            printed_name = self.print_stochiometry()

        if self.notes is not None:
            if printed_name == '':
                printed_name = self.notes
            else:
                printed_name = '{} ({})'.format(printed_name,self.notes)


        extras = []
        if self.proprietary_name:
            if self.proprietary:
                secret = "SECRET"
            else:
                secret = "NOT SECRET"
            extras.append(secret)

        if self.particle_size_name:
            particle_size = '?'
            if self.particle_size is not None:
                particle_size= '{:2.2f}'.format(self.particle_size)
            extras.append('SIZE={}{}'.format(particle_size, self.UNITS_SIZE[1]))

        if self.smiles_name:
            smiles = '?'
            if self.smiles is not None:
                smiles = self.smiles
            extras.append('SMILES={}'.format(smiles))

        if self.single_crystal_name:
            single_crystal = '?'
            if self.single_crystal is not None:
                single_crystal = self.get_single_crystal_display()
            extras.append('CRYSTAL={}'.format(single_crystal))

        if self.natural_name:
            if self.natural is None:
                natural = 'NAT/ART?'
            elif self.natural:
                natural = 'NAT'
            else:
                natural = 'ART'
            extras.append(natural)

        if self.preparation_temperature_name:
            temp = '?'
            if self.preparation_temperature is not None:
                temp = '{:4.0f}'.format(self.preparation_temperature)
            extras.append('TEMP={}{}'.format(temp, self.UNITS_TEMPERATURE[1]))

        if self.turbostratic_misalignment_name:
            turbo = '?'
            if self.turbostratic_misalignment is not None:
                turbo = '{:2.0f}'.format(self.turbostratic_misalignment)

            extras.append('TURBO={}{}'.format(turbo, self.UNITS_MISALIGNMENT[1]))

        if self.coating_lot_name:
            if self.coating_lot is None:
                extras.append('COAT=?')
            else:
                extras.append('COAT={}'.format(self.coating_lot))


        if self.component_type_name:
            extras.append(
                self.get_component_type_display()
            )
        if self.composite_type_name:
            extras.append(
                self.get_composite_type_display()
            )

        if len(extras) != 0:
            return "{}[{}]".format(printed_name, ','.join(extras))
        else:
            return printed_name





class ComponentLot(models.Model):
    component = models.ForeignKey(Component, on_delete=models.CASCADE, blank=True)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return print_lot(self, type='component')


class RatioComponent(models.Model):
    '''
        molecule (salt)
        molecule (solvent)
        molecule (additive)

        electrode material (active_material)
        electrode material (binder)
        electrode material (conductive_additive)

        separator material
    '''

    ratio = models.FloatField(null=True, blank=True)
    component_lot = models.ForeignKey(ComponentLot, on_delete=models.CASCADE, blank=True)

    def pretty_print(self, digits=None):
        my_string = '{}%{}'
        if self.ratio is None:
            pd = '?'
        else:
            pd = print_digits(self.ratio, digits)
            if self.component_lot.component.component_type == SALT:
                my_string = '{}m{}'

            if pd == '':
                pd = '1'
        return my_string.format(pd, self.component_lot.__str__())

    def __str__(self):
        return self.pretty_print(digits=2)

def helper_component_type(x, type=None):
    if type == 'component':
        return x[type].component_type
    if type == 'component_lot':
        return x[type].component.component_type

def helper_component_group(x, type=None):
    if type == "INACTIVE":
        return x != ACTIVE_MATERIAL
    else:
        return x == type

def helper_null_to_zero(val):
    if val is None:
        return 0.
    else:
        return val

def helper_total(fil=None, my_list=None, type_component=None, type_group=None):
    if fil is None:
        fil = lambda x: helper_component_group(helper_component_type(x, type_component), type_group)
    return sum(map(lambda x: helper_null_to_zero(x['ratio']),
         filter(fil, my_list)),0.)


def helper_total_complete(components=None, components_lot=None, type_group=None):
    return (
        helper_total(my_list=components, type_component='component', type_group=type_group) +
        helper_total(my_list=components_lot, type_component='component_lot', type_group=type_group)
    )




class Composite(models.Model):
    '''
        TODO(sam): Add notes to define if possible.
         add notes to everything!!!!

        electrolyte: separate solvent, salt, additive. no name unless proprietary. no other property
        anode & cathode: separate Active materials and inactive materials, then give geometry info
        separator: separator material, then give geometry info
    '''

    proprietary = models.BooleanField(default=False, blank=True)
    proprietary_name = models.BooleanField(default=False, blank=True)

    composite_type = models.CharField(max_length=2, choices=COMPOSITE_TYPES, blank=True)
    composite_type_name = models.BooleanField(default=False, blank=True)

    notes = models.CharField(max_length=1000, null=True, blank=True)


    components = models.ManyToManyField(RatioComponent)

    def is_valid(self):
        """
        if proprietary, needs a name.
        if not proprietary and active material, can't have a name.

        all fields which are null must have the corresponding _name field =False

        :return:
        """
        if self.composite_type is None:
            return False



        #so far, this is valid.
        return True

    def define_if_possible(self, components=None,components_lot=None, target=None):
        """
        The objects are made of subobjects and visibility flags.
        we want to make sure that among the objects created, no two (object,visibility) have the same (object projection)
        furthermore, no two objects can have the same string.

        If there is an object clash, return the preexisting object.
        Else: if there is a string clash, set all visibility flags to True.
        TODO: revamp this with the new abstraction.
        """
        if not self.is_valid():
            return None

        object_equality_query = Q(
            proprietary=self.proprietary,
            composite_type=self.composite_type,
            notes=self.notes,
        )


        string_equality_query = Q()
        if self.composite_type_name:
            string_equality_query = string_equality_query & Q(composite_type=self.composite_type,
                                                              composite_type_name=True)
        else:
            string_equality_query = string_equality_query & Q(composite_type_name=False)



        string_equality_query = string_equality_query & Q(notes=self.notes)

        if self.proprietary_name:
            string_equality_query = string_equality_query & Q(proprietary=self.proprietary,
                                                              proprietary_name=True)
        else:
            string_equality_query = string_equality_query & Q(proprietary_name=False)


        if self.composite_type in [ANODE,CATHODE]:
            string_equality_query = string_equality_query & (Q(composite_type=ANODE)|Q(composite_type=CATHODE))



        elif self.composite_type == SEPARATOR:
            string_equality_query = string_equality_query & Q(composite_type=SEPARATOR)




        if components is None:
            components = []

        if components_lot is None:
            components_lot = []

        all_ids = list(map(lambda x: x['component'].id, components)) + list(
            map(lambda x: x['component_lot'].component.id, components_lot))
        if len(set(all_ids)) == len(all_ids):
            # normalize things.
            if self.composite_type == ELECTROLYTE:
                total_complete = helper_total_complete(
                    components=components,
                    components_lot=components_lot,
                    type_group=SOLVENT
                )
                total_extra = helper_total_complete(
                    components=components,
                    components_lot=components_lot,
                    type_group=ADDITIVE
                )


            if self.composite_type in [ANODE,CATHODE]:
                total_complete = helper_total_complete(
                    components=components,
                    components_lot=components_lot,
                    type_group=ACTIVE_MATERIAL
                )
                total_extra = helper_total_complete(
                    components=components,
                    components_lot=components_lot,
                    type_group="INACTIVE"
                )
            if self.composite_type == SEPARATOR:
                total_complete = helper_total_complete(
                    components=components,
                    components_lot=components_lot,
                    type_group=SEPARATOR_MATERIAL
                )
                total_extra = 0.

            if total_complete == 0.:
                total_complete = 100.


            if total_extra < 100. and total_extra >= 0. and total_complete > 0.:
                # create or get each RatioComponent.

                my_ratio_components = []
                for ms, kind in [(components, 'component'), (components_lot, 'component_lot')]:
                    for component in ms:
                        if kind == 'component':
                            actual_component = component['component']
                        elif kind == 'component_lot':
                            actual_component = component['component_lot'].component

                        if actual_component.component_type in [SOLVENT,ACTIVE_MATERIAL,SEPARATOR_MATERIAL]:
                            if component['ratio'] is None:
                                ratio = None
                            else:
                                ratio = component['ratio'] * 100. / total_complete
                            tolerance = 0.25
                        elif actual_component.component_type in [CONDUCTIVE_ADDITIVE,BINDER,SALT, ADDITIVE]:
                            ratio = component['ratio']
                            tolerance = 0.01

                        if kind == 'component':
                            comp_lot, _ = ComponentLot.objects.get_or_create(
                                lot_info=None,
                                component=component['component']
                            )
                        elif kind == 'component_lot':
                            comp_lot = component['component_lot']

                        if ratio is None:
                            ratio_components = RatioComponent.objects.filter(
                                component_lot=comp_lot,
                                ratio=None)
                            if ratio_components.exists():
                                selected_ratio_component = ratio_components.order_by('id')[0]
                            else:
                                selected_ratio_component = RatioComponent.objects.create(
                                    component_lot=comp_lot,
                                    ratio=None
                                )
                        else:
                            ratio_components = RatioComponent.objects.filter(
                                component_lot=comp_lot,
                                ratio__range=(ratio - tolerance, ratio + tolerance))
                            if ratio_components.exists():
                                selected_ratio_component = ratio_components.annotate(
                                    distance=Func(F('ratio') - ratio, function='ABS')).order_by('distance')[0]
                            else:
                                selected_ratio_component = RatioComponent.objects.create(
                                    component_lot=comp_lot,
                                    ratio=ratio
                                )

                        my_ratio_components.append(selected_ratio_component)



                target_object = None
                if target is None:

                    composite_set = Composite.objects
                    target_object = None
                else:
                    composite_set = Composite.objects.exclude(id=target)
                    if Composite.objects.filter(id=target).exists():
                        target_object = Composite.objects.get(id=target)

                set_with_valid_comp = composite_set.annotate(
                    count_components=Count('components')
                ).filter(count_components=len(my_ratio_components)).annotate(
                    count_valid_components=Count('components', filter=Q(components__in=my_ratio_components))
                ).filter(count_valid_components=len(my_ratio_components))

                set_of_object_equal = set_with_valid_comp.filter(object_equality_query)
                string_equals = set_with_valid_comp.filter(string_equality_query).exists()

                if (not set_of_object_equal.exists()) and string_equals:
                    self.proprietary_name = True
                    self.composite_type_name = True


            if target_object is None:
                my_self = self
            else:
                my_self = target_object
                my_self.notes = self.notes
                my_self.proprietary = self.proprietary
                my_self.proprietary_name = self.proprietary_name
                my_self.composite_type = self.composite_type
                my_self.composite_type_name = self.composite_type_name

            return helper_return(
                    set_of_object_equal=set_of_object_equal,
                    my_self=my_self,
                    my_ratio_components=my_ratio_components
                )

    def __str__(self):
        printed_name = ''

        if self.composite_type == ELECTROLYTE:
            lists_of_lists = [
                print_components(
                    list(self.components.filter(
                        component_lot__component__component_type=SOLVENT).order_by('-ratio')),
                    complete=True, prefix=None
                ),
                print_components(
                    list(self.components.filter(
                        component_lot__component__component_type=SALT).order_by('-ratio')),
                    complete=False, prefix=None
                ),
                print_components(
                    list(self.components.filter(
                        component_lot__component__component_type=ADDITIVE).order_by('-ratio')),
                    complete=False, prefix="ADDITIVES"
                )
            ]
            printed_name = " + ".join([ll for ll in lists_of_lists if ll is not None])


        elif self.composite_type == ANODE or self.composite_type == CATHODE:
            lists_of_lists = [
                print_components(
                    list(self.components.filter(
                        component_lot__component__component_type=ACTIVE_MATERIAL).order_by('-ratio')),
                    complete=True, prefix=None
                ),
                print_components(
                    list(self.components.exclude(
                        component_lot__component__component_type=ACTIVE_MATERIAL).order_by('-ratio')),
                    complete=False, prefix="INACTIVES"
                )
            ]
            printed_name = " + ".join([ll for ll in lists_of_lists if ll is not None])

        elif self.composite_type == SEPARATOR:
            printed_name = print_components(
                    list(self.components.order_by('-ratio')),
                    complete=True, prefix=None
                )

        extras = []
        if self.notes is not None:
            if printed_name == '' or printed_name is None:
                printed_name = self.notes
            else:
                printed_name = '{} ({})'.format(printed_name,self.notes)

        if self.proprietary_name:
            if self.proprietary:
                secret = "SECRET"
            else:
                secret = "NOT SECRET"
            extras.append(secret)



        if self.composite_type_name:
            extras.append(
                self.get_composite_type_display()
            )
        if len(extras) != 0:
            return "{} [{}]".format(printed_name, ','.join(extras))
        else:
            return printed_name


class CompositeLot(models.Model):
    composite = models.ForeignKey(Composite, on_delete=models.CASCADE, blank=True)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return print_lot(self, type='composite')



class DryCellGeometry(models.Model):
    UNITS_LENGTH = 'Millimeters (mm)'
    POUCH = 'po'
    CYLINDER = 'cy'
    STACK = 'st'
    COIN = 'co'
    GEO_TYPES = [(POUCH, 'pouch'), (CYLINDER, 'cylinder'), (STACK, 'stack'), (COIN, 'coin')]

    geometry_category = models.CharField(max_length=2, choices=GEO_TYPES, blank=True)
    geometry_category_name = models.BooleanField(default=False, blank=True)

    width = models.FloatField(null=True, blank=True, help_text = UNITS_LENGTH)
    width_name = models.BooleanField(default=False, blank=True)

    length = models.FloatField(null=True, blank=True, help_text = UNITS_LENGTH)
    length_name = models.BooleanField(default=False, blank=True)

    thickness = models.FloatField(null=True, blank=True, help_text = UNITS_LENGTH)
    thickness_name = models.BooleanField(default=False, blank=True)



class DryCell(models.Model):
    UNITS_SOC = 'Percentage (i.e. 0 to 100)'
    UNITS_ENERGY_ESTIMATE = 'Watt Hour (Wh)'
    UNITS_CAPACITY_ESTIMATE = 'Ampere Hour (Ah)'
    UNITS_MASS_ESTIMATE = 'Grams (g)'
    UNITS_MAX_CHARGE_VOLTAGE = 'Volts (V)'
    UNITS_DCR_ESTIMATE = 'Ohms (\\Omega)'

    notes = models.CharField(max_length=1000, null=True, blank=True)

    proprietary = models.BooleanField(default=False, blank=True)
    proprietary_name = models.BooleanField(default=False, blank=True)

    geometry = models.OneToOneField(DryCellGeometry, on_delete=models.SET_NULL, null=True, blank=True)
    geometry_name = models.BooleanField(default=False, blank=True)

    cathode = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True,
                                related_name='cathode', blank=True)
    cathode_name = models.BooleanField(default=False, blank=True)
    cathode_geometry = models.OneToOneField(ElectrodeGeometry, on_delete=models.SET_NULL, null=True,
                                          related_name='cathode_geometry', blank=True)

    anode = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True, related_name='anode', blank=True)
    anode_name = models.BooleanField(default=False, blank=True)
    anode_geometry = models.OneToOneField(ElectrodeGeometry, on_delete=models.SET_NULL,
                                          null=True, related_name='anode_geometry', blank=True)

    separator = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True, related_name='separator', blank=True)
    separator_name = models.BooleanField(default=False, blank=True)
    separator_geometry = models.OneToOneField(SeparatorGeometry, on_delete=models.SET_NULL, null=True, blank=True)

    def is_valid(self, geometry=None, cathode_geometry=None, anode_geometry=None,separator_geometry=None):
        if geometry is None or cathode_geometry is None or anode_geometry is None or separator_geometry is None:
            return False
        else:
            return True


    def __str__(self):
        printed_name = ''
        extras = []
        if self.notes is not None:
            if printed_name == '':
                printed_name = self.notes
            else:
                printed_name = '{} ({})'.format(printed_name,self.notes)

        if self.proprietary_name:
            if self.proprietary:
                secret = "SECRET"
            else:
                secret = "NOT SECRET"
            extras.append(secret)
        if self.geometry is not None:
            if self.geometry.geometry_category_name:
                if self.geometry.geometry_category is None:
                    extras.append('GEO=?')
                else:
                    extras.append(
                        'GEO={}'.format(
                            self.geometry.get_geometry_category_display()))
            if self.geometry.width_name:
                if self.geometry.width is None:
                    extras.append('WIDTH=?')
                else:
                    extras.append('WIDTH={:2.2f}{}'.format(self.geometry.width,DryCellGeometry.UNITS_LENGTH))
            if self.geometry.length_name:
                if self.geometry.length is None:
                    extras.append('LENGT=?')
                else:
                    extras.append('LENGT={:2.2f}{}'.format(self.geometry.length,DryCellGeometry.UNITS_LENGTH))
            if self.geometry.thickness_name:
                if self.geometry.thickness is None:
                    extras.append('THICK=?')
                else:
                    extras.append('THICK={:2.2f}{}'.format(self.geometry.thickness,DryCellGeometry.UNITS_LENGTH))
        if self.cathode_name:
            if self.cathode is None:
                extras.append('CATH=?')
            else:
                extras.append(
                    'CATH={}'.format(self.cathode.__str__())
                )
        if self.cathode_geometry is not None:
            if self.cathode_geometry.loading_name:
                if self.cathode_geometry.loading is None:
                    extras.append('CATH-LOADING=?')
                else:
                    extras.append('CATH-LOADING={:2.2f}{}'.format(self.cathode_geometry.loading,ElectrodeGeometry.UNITS_LOADING[1]))
            if self.cathode_geometry.density_name:
                if self.cathode_geometry.density is None:
                    extras.append('CATH-DENSITY=?')
                else:
                    extras.append('CATH-DENSITY={:2.2f}{}'.format(self.cathode_geometry.density,ElectrodeGeometry.UNITS_DENSITY[1]))
            if self.cathode_geometry.thickness_name:
                if self.cathode_geometry.thickness is None:
                    extras.append('CATH-THICKNESS=?')
                else:
                    extras.append('CATH-THICKNESS={:2.2f}{}'.format(self.cathode_geometry.thickness,ElectrodeGeometry.UNITS_THICKNESS[1]))



        if self.anode_name:
            if self.anode is None:
                extras.append('ANOD=?')
            else:
                extras.append(
                    'ANOD={}'.format(self.anode.__str__())
                )

        if self.anode_geometry is not None:
            if self.anode_geometry.loading_name:
                if self.anode_geometry.loading is None:
                    extras.append('ANOD-LOADING=?')
                else:
                    extras.append(
                        'ANOD-LOADING={:2.2f}{}'.format(self.anode_geometry.loading, ElectrodeGeometry.UNITS_LOADING[1]))
            if self.anode_geometry.density_name:
                if self.anode_geometry.density is None:
                    extras.append('ANOD-DENSITY=?')
                else:
                    extras.append(
                        'ANOD-DENSITY={:2.2f}{}'.format(self.anode_geometry.density, ElectrodeGeometry.UNITS_DENSITY[1]))
            if self.anode_geometry.thickness_name:
                if self.anode_geometry.thickness is None:
                    extras.append('ANOD-THICKNESS=?')
                else:
                    extras.append('ANOD-THICKNESS={:2.2f}{}'.format(self.anode_geometry.thickness,
                                                               ElectrodeGeometry.UNITS_THICKNESS[1]))

        if self.separator_name:
            if self.separator is None:
                extras.append('SEPA=?')
            else:
                extras.append(
                    'SEPA={}'.format(self.separator.__str__())
                )
        if self.separator_geometry is not None:
            if self.separator_geometry.thickness_name:
                if self.separator_geometry.thickness is None:
                    extras.append('SEPA-THICKNESS=?')
                else:
                    extras.append('SEPA-THICKNESS={:2.2f}{}'.format(self.separator_geometry.thickness,SeparatorGeometry.UNITS[1]))
            if self.separator_geometry.width_name:
                if self.separator_geometry.width is None:
                    extras.append('SEPA-WIDTH=?')
                else:
                    extras.append('SEPA-WIDTH={:2.2f}{}'.format(self.separator_geometry.width, SeparatorGeometry.UNITS[1]))

        if len(extras) != 0:
            return "{} [{}]".format(printed_name, ','.join(extras))
        else:
            return printed_name

    def define_if_possible(self, geometry=None,cathode=None, anode=None,separator=None, cathode_geometry=None,anode_geometry=None,separator_geometry=None,target=None):
        """
        The objects are made of subobjects and visibility flags.
        we want to make sure that among the objects created, no two (object,visibility) have the same (object projection)
        furthermore, no two objects can have the same string.

        If there is an object clash, return the preexisting object.
        Else: if there is a string clash, set all visibility flags to True.
        TODO: revamp this with the new abstraction.
        """
        if not self.is_valid(geometry, cathode_geometry=cathode_geometry, anode_geometry=anode_geometry, separator_geometry=separator_geometry):
            return None

        object_equality_query = Q(
            proprietary=self.proprietary,
            notes=self.notes,
            geometry__geometry_category=geometry.geometry_category,
            geometry__width=geometry.width,
            geometry__length=geometry.length,
            geometry__thickness=geometry.thickness,
            cathode=cathode,
            anode=anode,
            separator=separator,

            cathode_geometry__loading=cathode_geometry.loading,
            cathode_geometry__density=cathode_geometry.density,
            cathode_geometry__thickness=cathode_geometry.thickness,

            anode_geometry__loading=anode_geometry.loading,
            anode_geometry__density=anode_geometry.density,
            anode_geometry__thickness=anode_geometry.thickness,

            separator_geometry__width=separator_geometry.width,
            separator_geometry__thickness=separator_geometry.thickness,

        )


        string_equality_query = Q(notes=self.notes)

        if self.proprietary_name:
            string_equality_query = string_equality_query & Q(proprietary=self.proprietary,
                                                              proprietary_name=True)
        else:
            string_equality_query = string_equality_query & Q(proprietary_name=False)

        if self.anode_name:

            string_equality_query = string_equality_query & Q(anode=anode,
                                                              anode_name=True)
        else:

            string_equality_query = string_equality_query & Q(anode_name=False)

        if anode_geometry.loading_name:
            string_equality_query = string_equality_query & Q(anode_geometry__loading=anode_geometry.loading,
                                                              anode_geometry__loading_name=True)
        else:
            string_equality_query = string_equality_query & Q(anode_geometry__loading_name=False)


        if anode_geometry.density_name:
            string_equality_query = string_equality_query & Q(
                anode_geometry__density=anode_geometry.density,
                anode_geometry__density_name=True)
        else:
            string_equality_query = string_equality_query & Q(anode_geometry__density_name=False)

        if anode_geometry.thickness_name:
            string_equality_query = string_equality_query & Q(anode_geometry__thickness=anode_geometry.thickness,
                                                              anode_geometry__thickness_name=True)
        else:
            string_equality_query = string_equality_query & Q(anode_geometry__thickness_name=False)





        if self.cathode_name:

            string_equality_query = string_equality_query & Q(cathode=cathode,
                                                              cathode_name=True)
        else:

            string_equality_query = string_equality_query & Q(cathode_name=False)

        if cathode_geometry.loading_name:
            string_equality_query = string_equality_query & Q(cathode_geometry__loading=cathode_geometry.loading,
                                                              cathode_geometry__loading_name=True)
        else:
            string_equality_query = string_equality_query & Q(cathode_geometry__loading_name=False)


        if cathode_geometry.density_name:
            string_equality_query = string_equality_query & Q(
                cathode_geometry__density=cathode_geometry.density,
                cathode_geometry__density_name=True)
        else:
            string_equality_query = string_equality_query & Q(cathode_geometry__density_name=False)

        if cathode_geometry.thickness_name:
            string_equality_query = string_equality_query & Q(cathode_geometry__thickness=cathode_geometry.thickness,
                                                              cathode_geometry__thickness_name=True)
        else:
            string_equality_query = string_equality_query & Q(cathode_geometry__thickness_name=False)


        if self.separator_name:
            string_equality_query = string_equality_query & Q(separator=separator,
                                                              separator_name=True)
        else:
            string_equality_query = string_equality_query & Q(separator_name=False)


        if separator_geometry.width_name:
            string_equality_query = string_equality_query & Q(
                separator_geometry__width=separator_geometry.width,
                separator_geometry__width_name=True)
        else:
            string_equality_query = string_equality_query & Q(separator_geometry__width_name=False)

        if separator_geometry.thickness_name:
            string_equality_query = string_equality_query & Q(separator_geometry__thickness=separator_geometry.thickness,
                                                              separator_geometry__thickness_name=True)
        else:
            string_equality_query = string_equality_query & Q(separator_geometry__thickness_name=False)


        if geometry.geometry_category_name:
            string_equality_query = string_equality_query & Q(geometry__geometry_category=geometry.geometry_category,
                                                              geometry__geometry_category_name=True)
        else:
            string_equality_query = string_equality_query & Q(geometry__geometry_category_name=False)

        if geometry.width_name:
            string_equality_query = string_equality_query & Q(geometry__width=geometry.width,
                                                              geometry__width_name=True)
        else:
            string_equality_query = string_equality_query & Q(geometry__width_name=False)

        if geometry.length_name:
            string_equality_query = string_equality_query & Q(geometry__length=geometry.length,
                                                              geometry__length_name=True)
        else:
            string_equality_query = string_equality_query & Q(geometry__length_name=False)

        if geometry.thickness_name:
            string_equality_query = string_equality_query & Q(geometry__thickness=geometry.thickness,
                                                              geometry__thickness_name=True)
        else:
            string_equality_query = string_equality_query & Q(geometry__thickness_name=False)

        target_object = None
        if target is None:

            dry_cell_set = DryCell.objects
            target_object = None
        else:
            dry_cell_set = DryCell.objects.exclude(id = target)
            if DryCell.objects.filter(id=target).exists():
                target_object = DryCell.objects.get(id=target)

        set_of_object_equal = dry_cell_set.filter(object_equality_query)

        string_equals = dry_cell_set.filter(string_equality_query).exists()


        if (not set_of_object_equal.exists()) and string_equals:
            self.proprietary_name = True
            self.anode_name = True
            self.cathode_name = True
            self.separator_name = True
            geometry.geometry_category_name = True
            geometry.width_name = True
            geometry.thickness_name = True
            geometry.length_name = True

            cathode_geometry.loading_name = True
            cathode_geometry.density_name = True
            cathode_geometry.thickness_name = True

            anode_geometry.loading_name = True
            anode_geometry.density_name = True
            anode_geometry.thickness_name = True
            separator_geometry.width_name = True
            separator_geometry.thickness_name = True
        if target_object is None:
            my_self = self
        else:
            my_self = target_object
            my_self.notes = self.notes
            my_self.proprietary = self.proprietary
            my_self.proprietary_name = self.proprietary_name
            my_self.geometry_name = self.geometry_name
            my_self.cathode_name = self.cathode_name
            my_self.anode_name = self.anode_name
            my_self.separator_name = self.separator_name

        return helper_return(
                set_of_object_equal=set_of_object_equal,
                my_self=my_self,
                dry_cell_geometry=geometry,
                anode=anode,
                cathode=cathode,
                separator=separator,
                cathode_geometry=cathode_geometry,
                anode_geometry=anode_geometry,
                separator_geometry=separator_geometry,
            )


class DryCellLot(models.Model):
    dry_cell = models.ForeignKey(DryCell, on_delete=models.CASCADE, blank=True)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True, blank=True)
    def __str__(self):
        return print_lot(self, type='dry_cell')


def get_lot(content, lot, type=None):
    my_lot = None
    if content is not None:
        if type == 'coating':
            my_lot, _ = CoatingLot.objects.get_or_create(
                coating = content,
                lot_info = None
            )
        elif type == 'component':
            my_lot, _ = ComponentLot.objects.get_or_create(
                component=content,
                lot_info=None
            )
        elif type == 'composite':
            my_lot, _ = CompositeLot.objects.get_or_create(
                composite=content,
                lot_info=None
            )
        elif type == 'dry_cell':
            my_lot, _ = DryCellLot.objects.get_or_create(
                dry_cell=content,
                lot_info=None
            )

    elif lot is not None:
        my_lot = lot
    return my_lot



class WetCell(models.Model):
    cell_id = models.IntegerField(primary_key=True, blank=True)
    electrolyte = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True, blank=True)
    dry_cell = models.ForeignKey(DryCellLot, on_delete=models.SET_NULL, null=True, blank=True)
    def __str__(self):
        cell_id_str = 'No Barcode'
        if self.cell_id is not None:
            cell_id_str = '{}'.format(self.cell_id)
        electrolyte_str = '?'
        if self.electrolyte is not None:
            electrolyte_str = '{}'.format(self.electrolyte)
        dry_cell_str = '?'
        if self.dry_cell is not None:
            dry_cell_str = '{}'.format(self.dry_cell)

        return 'BARCODE: {}, ELECTROLYTE: {}, DRY CELL: {}'.format(cell_id_str, electrolyte_str, dry_cell_str)



class Dataset(models.Model):
    name = models.CharField(unique=True, blank=True, max_length=200)
    wet_cells = models.ManyToManyField(WetCell)
    def __str__(self):
        return self.name