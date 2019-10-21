from django.db import models
import datetime
import numpy
import re


## ==================================Electrolyte================================== ##
#TODO: this is useless. Find usages and rewrite!!
def test_similarity(elec_dict1, elec_dict2):

    molecule_list1 = []
    for salt in elec_dict1['salts'].keys():
        molecule_list1.append(salt)

    for solvent in elec_dict1['solvents'].keys():
        molecule_list1.append(solvent)

    molecule_list2 = []
    for salt in elec_dict2['salts'].keys():
        molecule_list2.append(salt)
    for solvent in elec_dict2['solvents'].keys():
        molecule_list2.append(solvent)


    if set(molecule_list2) == set(molecule_list1):

        vec1 = []
        vec2 = []

        for molecule in elec_dict1['salts'].keys():
            vec1.append(elec_dict1['salts'][molecule])
            vec2.append(elec_dict2['salts'][molecule])

        vec1 = numpy.array(vec1)
        vec2 = numpy.array(vec2)

        dist_squared = numpy.sum(numpy.square(vec1 - vec2))

        if dist_squared > 0.001 ** 2:
            return False

        vec1 = []
        vec2 = []

        for molecule in elec_dict1['solvents'].keys():
            vec1.append(elec_dict1['solvents'][molecule])
            vec2.append(elec_dict2['solvents'][molecule])

        vec1 = numpy.array(vec1)
        vec2 = numpy.array(vec2)

        if not None in vec1 and not None in vec2:


            vec1 = 1. / (1e-10 + numpy.sum(vec1)) * vec1

            vec2 = 1. / (1e-10 + numpy.sum(vec2)) * vec2

            dist_squared = numpy.sum(numpy.square(vec1 - vec2))

            if dist_squared > 0.001 ** 2:
                return False
            return True
        else:
            return True


    else:
        return False



class Electrolyte(models.Model):
    proprietary = models.BooleanField(default=False)
    proprietary_name = models.CharField(max_length=100, null=True)
    @property
    def get_shortstring(self):
        shortstring = []
        for component in self.component_set.filter(compound_type=ElectrolyteComponent.SALT).order_by('molecule__name'):
            shortstring.append(
                '{:10.3f}m{}'.format(component.molal, component.molecule.name))
        for component in (list(self.component_set.filter(compound_type=ElectrolyteComponent.SOLVENT).order_by('molecule__name')) + list(self.component_set.filter(compound_type=ElectrolyteComponent.ADDITIVE).order_by('molecule__name'))):
            shortstring.append(
                '{:100.3f}%{}'.format(component.weight_percent, component.molecule.name))
        return '+'.join(shortstring)

class ElectrolyteLot(models.Model):
    '''
    if Electrolyte contains everything we know about an electrolyte recipe,
    then ElectrolyteLot is an actual instanciation of that recipe.
    If we wish to specify which recipe was followed but we do not know which lot,
    we simply use a lot with a null lot_name, creation_date and creator, and known_lot==False.
    '''
    electrolyte = models.ForeignKey(Electrolyte, on_delete=models.CASCADE)
    lot_name = models.CharField(max_length=100, null=True)
    known_lot = models.BooleanField(default=True)
    creation_date = models.DatetimeField(null=True)
    creator = models.CharField(max_length=100, null=True)

    def __str__(self):
        return self.name

class ElectrolyteMolecule(models.Model):
    name = models.CharField(max_length=100, null=True)
    smiles = models.CharField(max_length=1000, null=True)
    proprietary = models.BooleanField(default=False)

    can_be_salt = models.BooleanField(default=False)
    can_be_additive = models.BooleanField(default=False)
    can_be_solvent = models.BooleanField(default=False)
    def __str__(self):
        return self.name

class ElectrolyteComponent(models.Model):
    SALT = 'sa'
    ADDITIVE = 'ad'
    SOLVENT = 'so'
    COMPOUND_TYPES = [(SALT, 'salt'), (ADDITIVE, 'additive'), (SOLVENT, 'solvent')]

    electrolyte = models.ForeignKey(Electrolyte, on_delete=models.CASCADE)
    molal = models.FloatField(blank=True,null=True)
    weight_percent = models.FloatField(blank=True,null=True)
    molecule = models.ForeignKey(ElectrolyteMolecule, on_delete=models.CASCADE)
    compound_type = models.CharField(max_length=2, choices=COMPOUND_TYPES)

    @property
    def is_valid(self):
        valid_salt = (
                (self.molal is not None) and
                (self.weight_percent is None) and
                (self.compound_type == ElectrolyteComponent.SALT) and
                (self.molecule.can_be_salt))
        valid_solvent = (
                (self.molal is None) and
                (self.weight_percent is not None) and
                (self.compound_type == ElectrolyteComponent.SOLVENT) and
                (self.molecule.can_be_solvent))

        valid_additive = (
                (self.molal is None) and
                (self.weight_percent is not None) and
                (self.compound_type == ElectrolyteComponent.ADDITIVE) and
                (self.molecule.can_be_additive))

        return (valid_salt or valid_solvent or valid_additive)

    def __str__(self):
        return self.molecule.name


## ==================================Dry Cell================================== ##
## Categories on the excel sheet that have not been addressed:
# - Mechanical Cylindrical

#TODO(sam): units are stored as class strings


# - Build Info (cell to cell confidence)

# Dry Cell


class DryCellGeometry(models.Model):
    POUCH = 'po'
    CYLINDER = 'cy'
    STACK = 'st'
    COIN = 'co'
    GEO_TYPES = [(POUCH, 'pouch'), (CYLINDER, 'cylinder'), (STACK, 'stack'),(COIN, 'coin')]

    UNITS_LENGTH = 'Millimeters (mm)'

    geometry_category = models.CharField(max_length=2, choices=GEO_TYPES)
    cell_width = models.FloatField(null=True)
    cell_length = models.FloatField(null=True)
    cell_thickness = models.FloatField(null=True)
    seal_width_side = models.FloatField(null=True)
    seal_width_top = models.FloatField(null=True)
    metal_bag_sheet_thickness = models.FloatField(null=True)



class DryCell(models.Model):
    UNITS_SOC = 'Percentage (i.e. 0 to 100)'
    UNITS_ENERGY_ESTIMATE = 'Watt Hour (Wh)'
    UNITS_CAPACITY_ESTIMATE = 'Ampere Hour (Ah)'
    UNITS_MASS_ESTIMATE = 'Grams (g)'
    UNITS_MAX_CHARGE_VOLTAGE = 'Volts (V)'
    UNITS_DCR_ESTIMATE = 'Ohms (\\Omega)'

    cell_model = models.CharField(max_length=300)
    family = models.CharField(max_length=100,null=True, blank=True)
    version = models.CharField(max_length=100,null=True, blank=True)
    description = models.CharField(max_length=10000,null=True, blank=True)
    marking_on_box = models.CharField(max_length=300, null=True, blank=True)
    quantity = models.IntegerField(null=True, blank=True)
    packing_date = models.DateField( null=True, blank=True)
    ship_date = models.DateField( null=True, blank=True)
    shipping_soc = models.FloatField(null=True, blank=True)
    energy_estimate = models.FloatField(null=True, blank=True)
    capacity_estimate = models.FloatField(null=True, blank=True)
    mass_estimate = models.FloatField(null=True, blank=True)
    max_charge_voltage = models.FloatField(null=True, blank=True)
    dcr_estimate = models.FloatField(null=True, blank=True)
    chemistry_freeze_date_requested = models.DateField(null=True, blank=True)
    geometry = models.OneToOneField(DryCellGeometry, on_delete=models.SET_NULL, null=True)



class DryCellLot(models.Model):
    '''
    if DryCell contains everything we know about an electrolyte recipe,
    then DryCellLot is an actual instanciation of that recipe.
    If we wish to specify which recipe was followed but we do not know which lot,
    we simply use a lot with a null lot_name, creation_date and creator, and known_lot==False.
    '''
    dry_cell = models.ForeignKey(DryCell, on_delete=models.CASCADE)
    lot_name = models.CharField(max_length=100, null=True)
    known_lot = models.BooleanField(default=True)
    creation_date = models.DatetimeField(null=True)
    creator = models.CharField(max_length=100, null=True)

    def __str__(self):
        return self.name

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

    loading = models.FloatField(null=True)
    density = models.FloatField(null=True)
    porosity = models.FloatField(null=True)
    thickness = models.FloatField(null=True)
    length_single_side = models.FloatField(null=True)
    length_double_side = models.FloatField(null=True)
    width = models.FloatField(null=True)
    tab_position_from_core = models.FloatField(null=True)
    foil_thickness = models.FloatField(null=True)



class Electrode(models.Model):
    CATHODE = '+'
    ANODE = '-'
    ELECTRODE_TYPES = [(CATHODE, 'cathode'), (ANODE, 'anode')]
    electrode_type = models.CharField(max_length=1, choices=ELECTRODE_TYPES)
    proprietary = models.BooleanField(default=False)
    proprietary_name = models.CharField(max_length=100, null=True)
    geometry = models.OneToOneField(ElectrodeGeometry, on_delete=models.SET_NULL, null=True)



class ElectrodeLot(models.Model):
    '''
    if Electrode contains everything we know about an electrolyte recipe,
    then ElectrodeLot is an actual instanciation of that recipe.
    If we wish to specify which recipe was followed but we do not know which lot,
    we simply use a lot with a null lot_name, creation_date and creator, and known_lot==False.
    '''
    electrode = models.ForeignKey(Electrode, on_delete=models.CASCADE)
    lot_name = models.CharField(max_length=100, null=True)
    known_lot = models.BooleanField(default=True)
    creation_date = models.DatetimeField(null=True)
    creator = models.CharField(max_length=100, null=True)

    def __str__(self):
        return self.name

class ElectrodeMaterialCoating(models.Model):
    name = models.CharField(max_length=100, null=True)
    proprietary = models.BooleanField(default=False)
    description = models.CharField(max_length=1000, null=True)

class ElectrodeMaterialAtom(models.Model):
    name = models.CharField(max_length=10)

class ElectrodeMaterial(models.Model):
    name = models.CharField(max_length=100, null=True)
    proprietary = models.BooleanField(default=False)
    notes = models.CharField(max_length=1000, null=True)

    can_be_cathode = models.BooleanField(default=False)
    can_be_anode = models.BooleanField(default=False)

    can_be_active_material = models.BooleanField(default=False)
    can_be_conductive_additive = models.BooleanField(default=False)
    can_be_binder = models.BooleanField(default=False)

    coating = models.ForeignKey(ElectrodeMaterialCoating, on_delete=models.SET_NULL, null=True)
    particle_size = models.FloatField(null=True)
    single_crystal = models.BooleanField(null=True)
    turbostratic_misalignment = models.FloatField(null=True)
    maximum_experienced_temperature = models.FloatField(null=True)
    natural = models.BooleanField(null=True)
    core_shell = models.BooleanField(null=True)
    #TODO(sam): structure = Layered, Spinel, ...
    vendor = models.CharField(null=True)

class ElectrodeMaterialStochiometry(models.Model):
    electrode_material = models.ForeignKey(ElectrodeMaterial, on_delete=models.CASCADE)
    atom = models.ForeignKey(ElectrodeMaterialAtom, on_delete=models.CASCADE)
    stochiometry = models.FloatField()

class ElectrodeMaterialLot(models.Model):
    electrode_material = models.ForeignKey(ElectrodeMaterial, on_delete=models.CASCADE)
    lot_name = models.CharField(max_length=100, null=True)
    known_lot = models.BooleanField(default=True)
    creation_date = models.DatetimeField(null=True)
    creator = models.CharField(max_length=100, null=True)

class ElectrodeComponent(models):
    can_be_active_material = models.BooleanField(default=False)
    can_be_conductive_additive = models.BooleanField(default=False)
    can_be_binder = models.BooleanField(default=False)
    ACTIVE_MATERIAL = 'am'
    CONDUCTIVE_ADDITIVE = 'ad'
    BINDER = 'bi'
    MATERIAL_TYPES = [(ACTIVE_MATERIAL, 'active_material'), (CONDUCTIVE_ADDITIVE, 'conductive_additive'), (BINDER, 'binder')]

    electrode = models.ForeignKey(Electrode, on_delete=models.CASCADE)
    weight_percent = models.FloatField(null=True)
    material_lot = models.ForeignKey(ElectrodeMaterialLot, on_delete=models.CASCADE)
    material_type = models.CharField(max_length=2, choices=MATERIAL_TYPES)







# Separator
#TODO(sam): figure out what the separator is.
class Separator(models.Model):
    separator_notes = models.CharField(max_length=100,null=True)
    separator_base_thickness = models.FloatField(null=True)
    separator_width_mm = models.FloatField(null=True)
    separator_functional_layer = models.CharField(max_length=100,null=True)

    separator_functional_thickness = models.CharField(max_length=100,null=True)
    separator_overhang_in_core_mm = models.FloatField(null=True)

    dry_cell = models.ForeignKey(DryCell,on_delete=models.CASCADE, null=True)


# Wet Cell

class WetCell(models.Model):
    electrolyte = models.ForeignKey(Electrolyte, on_delete=models.CASCADE, null=True)
    dry_cell = models.ForeignKey(DryCell, on_delete=models.CASCADE, null=True)
    cell_id = models.IntegerField(null=True)
    box = models.ForeignKey(Box, on_delete=models.CASCADE, null=True)

    def __str__(self):
        return str(self.cell_id)


# Vendors


class VendorInfo(models.Model):

    cathode_active_1_vendor = models.CharField(max_length=100,null=True)
    cathode_active_2_vendor = models.CharField(max_length=100,null=True)
    cathode_active_3_vendor = models.CharField(max_length=100,null=True)

    cathode_additive_vendor = models.CharField(max_length=100,null=True)

    cathode_binder_1_vendor = models.CharField(max_length=100,null=True)
    cathode_binder_2_vendor = models.CharField(max_length=100,null=True)
    cathode_binder_3_vendor = models.CharField(max_length=100,null=True)

    anode_active_1_vendor = models.CharField(max_length=100, null=True)
    anode_active_2_vendor = models.CharField(max_length=100, null=True)
    anode_active_3_vendor = models.CharField(max_length=100, null=True)
    anode_active_4_vendor = models.CharField(max_length=100, null=True)

    anode_binder_1_vendor = models.CharField(max_length=100,null=True)
    anode_binder_2_vendor = models.CharField(max_length=100,null=True)
    anode_binder_3_vendor = models.CharField(max_length=100,null=True)

    negative_foil_vendor = models.CharField(max_length=100,null=True)
    separator_vendor = models.CharField(max_length=100,null=True)
    separator_coat_vendor = models.CharField(max_length=100,null=True)
    gasket_vendor = models.CharField(max_length=100,null=True)
    can_vendor = models.CharField(max_length=100,null=True)
    top_cap_vendor = models.CharField(max_length=100,null=True)
    outer_tape_vendor = models.CharField(max_length=100,null=True)

    electrolyte_solvent_1_vendor = models.CharField(max_length=100,null=True)
    electrolyte_solvent_2_vendor = models.CharField(max_length=100,null=True)
    electrolyte_solvent_3_vendor = models.CharField(max_length=100,null=True)
    electrolyte_solvent_4_vendor = models.CharField(max_length=100,null=True)
    electrolyte_solvent_5_vendor = models.CharField(max_length=100,null=True)

    dry_cell = models.ForeignKey(DryCell,on_delete=models.CASCADE, null=True)

