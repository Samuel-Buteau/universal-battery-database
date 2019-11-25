from django.db import models
from django.db.models import Q, F, Func, Count,Exists, OuterRef
import datetime
import numpy
import re


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
            digit = max(map(lambda comp: determine_digits(comp.ratio), list_of_components))
            component_string = '+'.join([comp.pretty_print(digits=digit) for comp in list_of_components])
        if prefix is None:
            return component_string
        else:
            return "{}:({})".format(prefix, component_string)
    else:
        return None


class LotInfo(models.Model):
    lot_name = models.CharField(max_length=100, null=True, blank=True)
    creator = models.CharField(max_length=100, null=True, blank=True)
    vendor = models.CharField(max_length=300, null=True, blank=True)


class ElectrodeGeometry(models.Model):
    UNITS_LOADING = 'milligrams per centimeter squared (mg/cm^2)'
    UNITS_DENSITY = 'grams per cubic centimeters (g/cm^3)'
    UNITS_POROSITY = 'TODO(sam): I DON"T KNOW THESE UNITS'
    UNITS_THICKNESS = 'micrometers (\\mu m)'
    UNITS_LENGTH_SINGLE_SIDE = 'millimeters (mm)'
    UNITS_LENGTH_DOUBLE_SIDE = 'millimeters (mm)'
    UNITS_WIDTH = 'millimeters (mm)'
    UNITS_TAB_POSITION_FROM_CORE = 'TODO(sam): I DON"T KNOW THESE UNITS'
    UNITS_FOIL_THICKNESS = 'micrometers (\\mu m)'

    loading = models.FloatField(null=True, blank=True)
    loading_name = models.BooleanField(default=False, blank=True)

    density = models.FloatField(null=True, blank=True)
    density_name = models.BooleanField(default=False, blank=True)
    #porosity = models.FloatField(null=True, blank=True)

    thickness = models.FloatField(null=True, blank=True)
    thickness_name = models.BooleanField(default=False, blank=True)
    #length_single_side = models.FloatField(null=True, blank=True)
    #length_double_side = models.FloatField(null=True, blank=True)
    #width = models.FloatField(null=True, blank=True)
    #tab_position_from_core = models.FloatField(null=True, blank=True)
    #foil_thickness = models.FloatField(null=True, blank=True)

class SeparatorGeometry(models.Model):
    UNITS_BASE_THICKNESS = 'millimeters (mm)'
    UNITS_WIDTH = 'millimeters (mm)'
    UNITS_OVERHANG_IN_CORE = 'millimeters (mm)'

    thickness = models.FloatField(null=True, blank=True)
    thickness_name = models.BooleanField(default=False, blank=True)

    width = models.FloatField(null=True, blank=True)
    width_name = models.BooleanField(default=False, blank=True)

    #overhang_in_core = models.FloatField(null=True, blank=True)


class Coating(models.Model):
    name = models.CharField(max_length=100, null=True, blank=True)
    proprietary = models.BooleanField(default=False, blank=True)
    description = models.CharField(max_length=1000, null=True, blank=True)

    def __str__(self):
        if self.proprietary:
            return "PROPRIETARY:{}".format(self.name)
        else:
            return "{}".format(self.name)


class CoatingLot(models.Model):
    coating = models.ForeignKey(Coating, on_delete=models.CASCADE, blank=True)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True, blank=True)
    def __str__(self):
        if self.lot_info is None:
            return self.coating.__str__()
        else:
            return "{}({})".format(self.lot_info.lot_name, self.coating.__str__())





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
        set_of_candidates=None,
        set_of_valid_duplicates=None,
        my_self=None,
        electrode_geometry=None,
        separator_geometry=None,
        my_stochiometry_components=None,
        my_ratio_components=None):
    if not set_of_candidates.exists():
        # we know for a fact that value fields don't exist.
        # so we can create it as it is.
        if electrode_geometry is not None:
            electrode_geometry.save()
            my_self.electrode_geometry = electrode_geometry

        if separator_geometry is not None:
            separator_geometry.save()
            my_self.separator_geometry = separator_geometry

        my_self.save()

        if my_ratio_components is not None:
            my_self.components.set(my_ratio_components)
        if my_stochiometry_components is not None:
            print(my_stochiometry_components)
            my_self.stochiometry.set(my_stochiometry_components)
            print(my_self.stochiometry)
        return my_self

    else:
        if set_of_valid_duplicates.exists():
            # VF DO, same naming set.
            # don't create anything, just return object.
            return set_of_valid_duplicates[0]
        else:
            # could be VF DONT, not unique wrt
            # could be VF DO, different wrt
            return None



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
    name = models.CharField(max_length=100, null=True, blank=True)
    smiles = models.CharField(max_length=1000, null=True, blank=True)
    proprietary = models.BooleanField(default=False, blank=True)

    composite_type = models.CharField(max_length=2, choices=COMPOSITE_TYPES, blank=True)
    composite_type_name = models.BooleanField(default=False, blank=True)

    component_type = models.CharField(max_length=2, choices=COMPONENT_TYPES, blank=True)
    component_type_name = models.BooleanField(default=False, blank=True)

    notes = models.CharField(max_length=1000, null=True, blank=True)
    notes_name = models.BooleanField(default=False, blank=True)

    coating_lot = models.ForeignKey(CoatingLot, on_delete=models.SET_NULL, null=True, blank=True)
    coating_lot_name = models.BooleanField(default=False, blank=True)

    particle_size = models.FloatField(null=True, blank=True)
    particle_size_name = models.BooleanField(default=False, blank=True)

    single_crystal = models.BooleanField(null=True, blank=True)
    single_crystal_name = models.BooleanField(default=False, blank=True)

    turbostratic_misalignment = models.FloatField(null=True, blank=True)
    turbostratic_misalignment_name = models.BooleanField(default=False, blank=True)

    preparation_temperature = models.FloatField(null=True, blank=True)
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
        if self.proprietary:
            if self.name is None:
                return False
            if self.smiles is not None:
                return False
        elif self.component_type == ACTIVE_MATERIAL and self.name is not None:
                return False
        if self.component_type == ACTIVE_MATERIAL and self.smiles is not None:
            return False
        if self.notes is None and self.notes_name:
            return False
        # a Null coating is a valid thing.
        if self.particle_size is None and self.particle_size_name:
            return False
        if self.single_crystal is None and self.single_crystal_name:
            return False
        if self.turbostratic_misalignment is None and self.turbostratic_misalignment_name:
            return False
        if self.preparation_temperature is None and self.preparation_temperature_name:
            return False
        if self.natural is None and self.natural_name:
            return False

        #so far, this is valid.
        return True

    def define_if_possible(self, atoms=None):
        """
        There are two things defining an object here:
        - value fields
        - naming subset
        :return:
        if value fields DONT exist and not unique w.r.t. naming subset OR
           value fields DO exist and different naming subset:
           return None
        if value fields DONT exist and unique w.r.t. naming subset OR
           value fields DO exist and same naming subset:
           return Instance
        """
        print('entered define if possible')
        if not self.is_valid():
            print('was invalid')
            return None
        wrt_query = Q(name=self.name, proprietary=self.proprietary, smiles=self.smiles)
        if self.composite_type_name:
            wrt_query = wrt_query & Q(composite_type=self.composite_type)
        if self.component_type_name:
            wrt_query = wrt_query & Q(component_type=self.component_type)
        if self.notes_name:
            wrt_query = wrt_query & Q(notes=self.notes)
        if self.coating_lot_name:
            wrt_query = wrt_query & Q(coating_lot=self.coating_lot)
        if self.particle_size_name:
            wrt_query = wrt_query & Q(particle_size=self.particle_size)
        if self.single_crystal_name:
            wrt_query = wrt_query & Q(single_crystal=self.single_crystal)
        if self.turbostratic_misalignment_name:
            wrt_query = wrt_query & Q(turbostratic_misalignment=self.turbostratic_misalignment)
        if self.preparation_temperature_name:
            wrt_query = wrt_query & Q(preparation_temperature=self.preparation_temperature)
        if self.natural_name:
            wrt_query = wrt_query & Q(natural=self.natural)


        duplicate_query = Q(
            name=self.name,
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


        naming_set_query = Q(
            composite_type_name=self.composite_type_name,
            component_type_name=self.component_type_name,
            notes_name=self.notes_name,
            coating_lot_name=self.coating_lot_name,
            particle_size_name=self.particle_size_name,
            single_crystal_name=self.single_crystal_name,
            turbostratic_misalignment_name=self.turbostratic_misalignment_name,
            preparation_temperature_name=self.preparation_temperature_name,
            natural_name=self.natural_name
        )


        if self.proprietary or self.component_type != ACTIVE_MATERIAL:
            print("was proprietary or not active: {}, {}".format(self.proprietary, self.composite_type))
            set_of_candidates = Component.objects.filter(wrt_query)
            set_of_valid_duplicates = Component.objects.filter(duplicate_query & naming_set_query)

            return helper_return(
                set_of_candidates=set_of_candidates,
                set_of_valid_duplicates=set_of_valid_duplicates,
                my_self=self)


        #TODO(sam):
        # we currently assume that if we create a stochiometry,
        # then we will create an active material.
        # if this becomes false, it will be necessary to cleanup after we are done.
        #TODO(sam): maybe we want uniform scaling of stochiometry to mean nothing.
        # In such a case, there will be an issue.
        else:
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
                set_with_valid_stoc = Component.objects.annotate(
                    count_stochiometry=Count('stochiometry')
                ).filter(count_stochiometry=len(my_stochiometry_components)
                         ).annotate(
                    count_valid_stochiometry=Count('stochiometry', filter=Q(stochiometry__in=my_stochiometry_components))
                ).filter(count_valid_stochiometry=len(my_stochiometry_components))

                set_of_candidates = set_with_valid_stoc.filter(wrt_query)
                set_of_valid_duplicates = set_with_valid_stoc.filter(duplicate_query & naming_set_query)

                return helper_return(
                    set_of_candidates=set_of_candidates,
                    set_of_valid_duplicates=set_of_valid_duplicates,
                    my_self=self,
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
        if self.proprietary:
            printed_name = "PROPRIETARY:{}".format(self.name)
        elif self.component_type != ACTIVE_MATERIAL:
            printed_name = self.name
        else:
            printed_name = self.print_stochiometry()

        extras = []
        if self.particle_size_name:
            extras.append('SIZE={:2.2f}'.format(self.particle_size))

        if self.single_crystal_name:
            if self.single_crystal:
                single_crystal = 'SINGLE CRYSTAL'
            else:
                single_crystal = 'POLY'
            extras.append("{}".format(single_crystal))

        if self.natural_name:
            if self.natural:
                natural = 'NAT'
            else:
                natural = 'ART'
            extras.append(natural)


        if self.preparation_temperature_name:
            extras.append('TEMP={:4.0f}'.format(self.preparation_temperature))

        if self.turbostratic_misalignment_name:
            extras.append('TURBO={:3.3f}'.format(self.turbostratic_misalignment))

        if self.notes_name:
            extras.append('NOTES={}'.format(self.notes))

        if self.coating_lot_name:
            if self.coating_lot is None:
                extras.append('NO COAT')
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
        if self.lot_info is None:
            return self.component.__str__()
        else:
            return "{}({})".format(self.lot_info.lot_name, self.component.__str__())



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

def helper_total(fil=None, my_list=None, type_component=None, type_group=None):
    if fil is None:
        fil = lambda x: helper_component_group(helper_component_type(x, type_component), type_group)
    return sum(map(lambda x: x['ratio'],
         filter(fil, my_list)),0.)


def helper_total_complete(components=None, components_lot=None, type_group=None):
    return (
        helper_total(my_list=components, type_component='component', type_group=type_group) +
        helper_total(my_list=components_lot, type_component='component_lot', type_group=type_group)
    )




class Composite(models.Model):
    '''
        TODO(sam): define if possible for electrolyte
        TODO(sam): define if possible for electrode
        electrolyte: separate solvent, salt, additive. no name unless proprietary. no other property
        anode & cathode: separate Active materials and inactive materials, then give geometry info
        separator: separator material, then give geometry info
    '''

    proprietary = models.BooleanField(default=False, blank=True)
    name = models.CharField(max_length=100, null=True, blank=True)

    composite_type = models.CharField(max_length=2, choices=COMPOSITE_TYPES, blank=True)
    composite_type_name = models.BooleanField(default=False, blank=True)

    electrode_geometry = models.OneToOneField(ElectrodeGeometry, on_delete=models.SET_NULL, null=True, blank=True)
    separator_geometry = models.OneToOneField(SeparatorGeometry, on_delete=models.SET_NULL, null=True, blank=True)

    components = models.ManyToManyField(RatioComponent)

    def is_valid(self, electrode_geometry=None,separator_geometry=None):
        """
        if proprietary, needs a name.
        if not proprietary and active material, can't have a name.

        all fields which are null must have the corresponding _name field =False

        :return:
        """
        if self.composite_type is None:
            return False

        if self.proprietary:
            if self.name is None:
                return False

        elif self.name is not None:
            return False

        if self.composite_type in [ANODE,CATHODE]:
            if electrode_geometry is None:
                return False
            if electrode_geometry.loading is None and electrode_geometry.loading_name:
                return False
            if electrode_geometry.density is None and electrode_geometry.density_name:
                return False
            if electrode_geometry.thickness is None and electrode_geometry.thickness_name:
                return False

        if self.composite_type == SEPARATOR:
            if separator_geometry is None:
                return False
            if separator_geometry.thickness is None and separator_geometry.thickness_name:
                return False
            if separator_geometry.width is None and separator_geometry.width_name:
                return False

        #so far, this is valid.
        return True

    def define_if_possible(self, components=None,components_lot=None, electrode_geometry=None,separator_geometry=None):
        """
        There are two things defining an object here:
        - value fields
        - naming subset
        :return:
        if value fields DONT exist and not unique w.r.t. naming subset OR
           value fields DO exist and different naming subset:
           return None
        if value fields DONT exist and unique w.r.t. naming subset OR
           value fields DO exist and same naming subset:
           return Instance
        """
        if not self.is_valid(electrode_geometry=electrode_geometry, separator_geometry=separator_geometry):
            return None
        wrt_query = Q(name=self.name, proprietary=self.proprietary)
        if self.composite_type_name:
            wrt_query = wrt_query & Q(composite_type=self.composite_type)

        if self.composite_type in [ANODE,CATHODE]:
            wrt_query = wrt_query & (Q(composite_type=ANODE)|Q(composite_type=CATHODE))
            if electrode_geometry.loading_name:
                wrt_query = wrt_query & Q(electrode_geometry__loading=electrode_geometry.loading)
            if electrode_geometry.density_name:
                wrt_query = wrt_query & Q(electrode_geometry__density=electrode_geometry.density)
            if electrode_geometry.thickness_name:
                wrt_query = wrt_query & Q(electrode_geometry__thickness=electrode_geometry.thickness)

        elif self.composite_type == SEPARATOR:
            wrt_query = wrt_query & Q(composite_type=SEPARATOR)
            if electrode_geometry.width_name:
                wrt_query = wrt_query & Q(electrode_geometry__width=electrode_geometry.width)
            if electrode_geometry.thickness_name:
                wrt_query = wrt_query & Q(electrode_geometry__thickness=electrode_geometry.thickness)

        duplicate_query = Q(
            name=self.name,
            proprietary=self.proprietary,
            composite_type=self.composite_type,)
        if self.composite_type in [ANODE, CATHODE]:
            duplicate_query = duplicate_query & Q(
                electrode_geometry__loading=electrode_geometry.loading,
                electrode_geometry__density=electrode_geometry.density,
                electrode_geometry__thickness=electrode_geometry.thickness,
            )
        elif self.composite_type == SEPARATOR:
            duplicate_query = duplicate_query & Q(
                electrode_geometry__width=electrode_geometry.width,
                electrode_geometry__thickness=electrode_geometry.thickness,
            )

        naming_set_query = Q(
            composite_type_name=self.composite_type_name,
        )
        if self.composite_type in [ANODE, CATHODE]:
            naming_set_query = naming_set_query & Q(
                electrode_geometry__loading_name=electrode_geometry.loading_name,
                electrode_geometry__density_name=electrode_geometry.density_name,
                electrode_geometry__thickness_name=electrode_geometry.thickness_name,
            )
        elif self.composite_type == SEPARATOR:
            naming_set_query = naming_set_query & Q(
                electrode_geometry__width_name=electrode_geometry.width_name,
                electrode_geometry__thickness_name=electrode_geometry.thickness_name,
            )



        if self.proprietary:
            set_of_candidates = Composite.objects.filter(wrt_query)
            set_of_valid_duplicates = Composite.objects.filter(duplicate_query & naming_set_query)

            return helper_return(
                set_of_candidates=set_of_candidates,
                set_of_valid_duplicates=set_of_valid_duplicates,
                my_self=self,
                electrode_geometry=electrode_geometry,
                separator_geometry=separator_geometry,
            )


        #TODO(sam):
        # we currently assume that if we create a ratio,
        # then we will create a composite.
        # if this becomes false, it will be necessary to cleanup after we are done.
        else:
            #TODO(sam): electrolytes
            # check if there is a problem with the molecules/molecules_lot
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

                if total_extra < 100. and total_extra > 0. and total_complete > 0.:
                    # create or get each RatioComponent.
                    my_ratio_components = []
                    for ms, kind in [(components, 'component'), (components_lot, 'component_lot')]:
                        for component in ms:
                            if kind == 'component':
                                actual_component = component['component']
                            elif kind == 'component_lot':
                                actual_component = component['component_lot'].component

                            if actual_component.component_type in [SOLVENT,ACTIVE_MATERIAL,SEPARATOR_MATERIAL]:
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

                    set_with_valid_comp = Composite.objects.annotate(
                        count_components=Count('components')
                    ).filter(count_components=len(my_ratio_components)).annotate(
                        count_valid_components=Count('components', filter=Q(components__in=my_ratio_components))
                    ).filter(count_valid_components=len(my_ratio_components))

                    set_of_candidates = set_with_valid_comp.filter(wrt_query)
                    set_of_valid_duplicates = set_with_valid_comp.filter(duplicate_query & naming_set_query)

                    return helper_return(
                        set_of_candidates=set_of_candidates,
                        set_of_valid_duplicates=set_of_valid_duplicates,
                        my_self=self,
                        electrode_geometry=electrode_geometry,
                        separator_geometry=separator_geometry,
                        my_ratio_components=my_ratio_components
                    )

    def __str__(self):
        printed_name = ''
        if self.proprietary:
            printed_name = "PROPRIETARY:{}".format(self.name)
            return printed_name

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

            if self.composite_type_name:
                printed_name = "{} [{}]".format(printed_name, self.get_composite_type_display)

            return printed_name

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


            extras = []
            if self.electrode_geometry is not None:
                if self.electrode_geometry.loading_name:
                    extras.append('LOADING={:2.2f}'.format(self.electrode_geometry.loading))
                if self.electrode_geometry.density_name:
                    extras.append('DENSITY={:2.2f}'.format(self.electrode_geometry.density))
                if self.electrode_geometry.thickness_name:
                    extras.append('THICKNESS={:2.2f}'.format(self.electrode_geometry.thickness))
            if self.composite_type_name:
                extras.append(
                    self.get_composite_type_display()
                )
            if len(extras) != 0:
                return "{} [{}]".format(printed_name, ','.join(extras))
            else:
                return printed_name

        elif self.composite_type == SEPARATOR:
            return 'not yet implemented'

class CompositeLot(models.Model):
    composite = models.ForeignKey(Composite, on_delete=models.CASCADE, blank=True)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True, blank=True)
    def __str__(self):
        if self.lot_info is None:
            return self.composite.__str__()
        else:
            return "{}({})".format(self.lot_info.lot_name, self.composite.__str__())




class DryCellGeometry(models.Model):
    UNITS_LENGTH = 'Millimeters (mm)'
    POUCH = 'po'
    CYLINDER = 'cy'
    STACK = 'st'
    COIN = 'co'
    GEO_TYPES = [(POUCH, 'pouch'), (CYLINDER, 'cylinder'), (STACK, 'stack'),(COIN, 'coin')]

    geometry_category = models.CharField(max_length=2, choices=GEO_TYPES, blank=True)
    geometry_category_name = models.BooleanField(default=False, blank=True)

    width = models.FloatField(null=True, blank=True)
    width_name = models.BooleanField(default=False, blank=True)

    length = models.FloatField(null=True, blank=True)
    length_name = models.BooleanField(default=False, blank=True)

    thickness = models.FloatField(null=True, blank=True)
    thickness_name = models.BooleanField(default=False, blank=True)
    #seal_width_side = models.FloatField(null=True, blank=True)
    #seal_width_top = models.FloatField(null=True, blank=True)
    #metal_bag_sheet_thickness = models.FloatField(null=True, blank=True)



class DryCell(models.Model):
    UNITS_SOC = 'Percentage (i.e. 0 to 100)'
    UNITS_ENERGY_ESTIMATE = 'Watt Hour (Wh)'
    UNITS_CAPACITY_ESTIMATE = 'Ampere Hour (Ah)'
    UNITS_MASS_ESTIMATE = 'Grams (g)'
    UNITS_MAX_CHARGE_VOLTAGE = 'Volts (V)'
    UNITS_DCR_ESTIMATE = 'Ohms (\\Omega)'

    name = models.CharField(max_length=300, null=True, blank=True)
    proprietary = models.BooleanField(default=False, blank=True)
    #family = models.CharField(max_length=100,null=True, blank=True)

    version = models.CharField(max_length=100,null=True, blank=True)
    version_name = models.BooleanField(default=False, blank=True)

    description = models.CharField(max_length=10000,null=True, blank=True)
    #marking_on_box = models.CharField(max_length=300, null=True, blank=True)
    #quantity = models.IntegerField(null=True, blank=True)
    #packing_date = models.DateField( null=True, blank=True)
    #ship_date = models.DateField( null=True, blank=True)
    #shipping_soc = models.FloatField(null=True, blank=True)
    #energy_estimate = models.FloatField(null=True, blank=True)
    #capacity_estimate = models.FloatField(null=True, blank=True)
    #mass_estimate = models.FloatField(null=True, blank=True)
    max_charge_voltage = models.FloatField(null=True, blank=True)
    max_charge_voltage_name = models.BooleanField(default=False, blank=True)

    #dcr_estimate = models.FloatField(null=True, blank=True)
    #chemistry_freeze_date_requested = models.DateField(null=True, blank=True)
    geometry = models.OneToOneField(DryCellGeometry, on_delete=models.SET_NULL, null=True, blank=True)
    geometry_name = models.BooleanField(default=False, blank=True)

    #negative_foil_vendor = models.CharField(max_length=100,null=True, blank=True)
    #gasket_vendor = models.CharField(max_length=100,null=True, blank=True)
    #can_vendor = models.CharField(max_length=100,null=True, blank=True)
    #top_cap_vendor = models.CharField(max_length=100,null=True, blank=True)
    #outer_tape_vendor = models.CharField(max_length=100,null=True, blank=True)

    cathode = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True, related_name='cathode', blank=True)
    cathode_name = models.BooleanField(default=False, blank=True)

    anode = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True, related_name='anode', blank=True)
    anode_name = models.BooleanField(default=False, blank=True)

    separator = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True, related_name='separator', blank=True)
    separator_name = models.BooleanField(default=False, blank=True)

class DryCellLot(models.Model):
    dry_cell = models.ForeignKey(DryCell, on_delete=models.CASCADE, blank=True)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True, blank=True)


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


