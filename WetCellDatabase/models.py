from django.db import models
import datetime
import numpy
import re


'''
TODO(sam):
Composite : shortstring
@property
    def get_shortstring(self):
        shortstring = []
        for component in self.component_set.filter(compound_type=ElectrolyteComponent.SALT).order_by('molecule__name'):
            shortstring.append(
                '{:10.3f}m{}'.format(component.ratio, component.molecule.name))
        for component in (list(self.component_set.filter(compound_type=ElectrolyteComponent.SOLVENT).order_by('molecule__name')) + list(self.component_set.filter(compound_type=ElectrolyteComponent.ADDITIVE).order_by('molecule__name'))):
            shortstring.append(
                '{:100.3f}%{}'.format(component.ratio, component.molecule.name))
        return '+'.join(shortstring)

ComponentRatio: is_valid
@property
    def is_valid(self):
        valid_salt = (
                (self.ratio is not None) and
                (self.compound_type == ElectrolyteComponent.SALT) and
                (self.molecule.can_be_salt))
        valid_solvent = (
                (self.ratio is not None) and
                (self.compound_type == ElectrolyteComponent.SOLVENT) and
                (self.molecule.can_be_solvent))

        valid_additive = (
                (self.ratio is not None) and
                (self.compound_type == ElectrolyteComponent.ADDITIVE) and
                (self.molecule.can_be_additive))

        return (valid_salt or valid_solvent or valid_additive)

Electrolyte Component: shortstring
'''


class LotInfo(models.Model):
    lot_name = models.CharField(max_length=100, null=True)
    creation_date = models.DateField(null=True)
    creator = models.CharField(max_length=100, null=True)
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

    loading = models.FloatField(null=True)
    density = models.FloatField(null=True)
    porosity = models.FloatField(null=True)
    thickness = models.FloatField(null=True)
    length_single_side = models.FloatField(null=True)
    length_double_side = models.FloatField(null=True)
    width = models.FloatField(null=True)
    tab_position_from_core = models.FloatField(null=True)
    foil_thickness = models.FloatField(null=True)

class SeparatorGeometry(models.Model):
    UNITS_BASE_THICKNESS = 'millimeters (mm)'
    UNITS_WIDTH = 'millimeters (mm)'
    UNITS_OVERHANG_IN_CORE = 'millimeters (mm)'
    base_thickness = models.FloatField(null=True)
    width = models.FloatField(null=True)
    overhang_in_core = models.FloatField(null=True)


class Composite(models.Model):
    proprietary = models.BooleanField(default=False)
    proprietary_name = models.CharField(max_length=100, null=True)

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
    composite_type = models.CharField(max_length=2, choices=COMPOSITE_TYPES)
    electrode_geometry = models.OneToOneField(ElectrodeGeometry, on_delete=models.SET_NULL, null=True)
    separator_geometry = models.OneToOneField(SeparatorGeometry, on_delete=models.SET_NULL, null=True)

class CompositeLot(models.Model):
    composite = models.ForeignKey(Composite, on_delete=models.CASCADE)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True)

class Coating(models.Model):
    name = models.CharField(max_length=100, null=True)
    proprietary = models.BooleanField(default=False)
    description = models.CharField(max_length=1000, null=True)

class CoatingLot(models.Model):
    coating = models.ForeignKey(Coating, on_delete=models.CASCADE)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True)


class Component(models.Model):
    name = models.CharField(max_length=100, null=True, blank=True)
    smiles = models.CharField(max_length=1000, null=True, blank=True)
    proprietary = models.BooleanField(default=False)
    can_be_cathode = models.BooleanField(default=False)
    can_be_anode = models.BooleanField(default=False)
    can_be_separator = models.BooleanField(default=False)
    can_be_electrolyte = models.BooleanField(default=False)

    can_be_salt = models.BooleanField(default=False)
    can_be_additive = models.BooleanField(default=False)
    can_be_solvent = models.BooleanField(default=False)
    can_be_active_material = models.BooleanField(default=False)
    can_be_conductive_additive = models.BooleanField(default=False)
    can_be_binder = models.BooleanField(default=False)

    notes = models.CharField(max_length=1000, null=True, blank=True)
    coating_lot = models.ForeignKey(CoatingLot, on_delete=models.SET_NULL, null=True, blank=True)
    particle_size = models.FloatField(null=True, blank=True)
    single_crystal = models.BooleanField(null=True, blank=True)
    turbostratic_misalignment = models.FloatField(null=True, blank=True)
    preparation_temperature = models.FloatField(null=True, blank=True)
    natural = models.BooleanField(null=True, blank=True)
    core_shell = models.BooleanField(null=True, blank=True)

class ComponentLot(models.Model):
    component = models.ForeignKey(Component, on_delete=models.CASCADE)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True)

class RatioComponent(models.Model):
    SALT = 'sa'
    ADDITIVE = 'ad'
    SOLVENT = 'so'
    ACTIVE_MATERIAL = 'am'
    CONDUCTIVE_ADDITIVE = 'co'
    BINDER = 'bi'

    COMPONENT_TYPES = [
        (SALT, 'salt'),
        (ADDITIVE, 'additive'),
        (SOLVENT, 'solvent'),
        (ACTIVE_MATERIAL, 'active_material'),
        (CONDUCTIVE_ADDITIVE, 'conductive_additive'),
        (BINDER, 'binder'),
    ]

    component_type = models.CharField(max_length=2, choices=COMPONENT_TYPES)
    composite = models.ForeignKey(Composite, on_delete=models.CASCADE)
    ratio = models.FloatField(null=True)
    component_lot = models.ForeignKey(ComponentLot, on_delete=models.CASCADE)


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
    electrode_material = models.ForeignKey(Component, on_delete=models.CASCADE)
    atom = models.CharField(max_length=3, choices=ATOMS)
    stochiometry = models.FloatField()


class DryCellGeometry(models.Model):
    UNITS_LENGTH = 'Millimeters (mm)'
    POUCH = 'po'
    CYLINDER = 'cy'
    STACK = 'st'
    COIN = 'co'
    GEO_TYPES = [(POUCH, 'pouch'), (CYLINDER, 'cylinder'), (STACK, 'stack'),(COIN, 'coin')]
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

    negative_foil_vendor = models.CharField(max_length=100,null=True)
    gasket_vendor = models.CharField(max_length=100,null=True)
    can_vendor = models.CharField(max_length=100,null=True)
    top_cap_vendor = models.CharField(max_length=100,null=True)
    outer_tape_vendor = models.CharField(max_length=100,null=True)

    cathode = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True, related_name='cathode')
    anode = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True, related_name='anode')
    separator = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True, related_name='separator')

class DryCellLot(models.Model):
    dry_cell = models.ForeignKey(DryCell, on_delete=models.CASCADE)
    lot_info = models.OneToOneField(LotInfo, on_delete=models.SET_NULL, null=True)

class WetCell(models.Model):
    cell_id = models.IntegerField(primary_key=True)
    electrolyte = models.ForeignKey(CompositeLot, on_delete=models.SET_NULL, null=True)
    dry_cell = models.ForeignKey(DryCellLot, on_delete=models.SET_NULL, null=True)
