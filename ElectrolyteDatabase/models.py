from django.db import models
import datetime
import numpy
import re


## ==================================Electrolyte================================== ##
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
    creation_date = models.DateField(null=True)
    shortstring = models.CharField(max_length=100, null=True)
    proprietary = models.BooleanField(default=False)


    def __str__(self):

        if self.shortstring is None:

            return 'NO SHORTSTRING'

        else:

            return self.shortstring


    @property
    def generate_shortstring(self):

        shortstring = ''

        for component in self.component_set.all():
            if component.molecule.can_be_salt:

                if (float(component.molal) % 1) == 0:
                    num = int(component.molal)
                else:
                    num = round(float(component.molal), 1)

                shortstring += '{}m{}+'.format(num, component.molecule.name)

            if component.molecule.can_be_additive or component.molecule.can_be_solvent:

                if not(component.weight_percent is None) and (float(component.weight_percent) % 1) == 0:
                    num = int(component.weight_percent)

                elif not(component.weight_percent is None):
                    num = round(float(component.weight_percent), 1)
                else:
                    num = None

                shortstring += '{}%{}+'.format(num, component.molecule.name)

        shortstring = re.sub(r'\+$', '', shortstring)

        return shortstring

    @property
    def get_sum(self):

        sum = 0

        for component in self.component_set.all():

            if (not component.molal is None) or (not component.weight_percent is None):

                if component.molecule.can_be_additive or component.molecule.can_be_solvent:

                    sum += float(component.weight_percent)

        return sum

class Molecule(models.Model):

    name = models.CharField(max_length=100, null=True)
    can_be_salt = models.BooleanField(default=False)
    can_be_additive = models.BooleanField(default=False)
    can_be_solvent = models.BooleanField(default=False)


    def __str__(self):
        return self.name


SALT = 'sa'
ADDITIVE = 'ad'
SOLVENT = 'so'
COMPOUND_TYPES = [(SALT,'salt'),(ADDITIVE,'additive'),(SOLVENT,'solvent')]

class Component(models.Model):
    molal = models.FloatField(blank=True,null=True)
    weight_percent = models.FloatField(blank=True,null=True)
    molecule = models.ForeignKey(Molecule, on_delete=models.CASCADE, null=True)
    electrolyte = models.ForeignKey(Electrolyte, on_delete=models.CASCADE, null=True)
    compound_type = models.CharField(max_length=2, choices=COMPOUND_TYPES, null=True)
    notes = models.CharField(max_length=200,null=True)

    def __str__(self):
        return self.molecule.name

class Alias(models.Model):

    #TODO: an argument for the CharField used to be 'unique=True' but I took it out because it wasn't working. Need to figure out.
    name = models.CharField(max_length=100)
    electrolyte = models.ForeignKey(Electrolyte, on_delete=models.CASCADE, null=True)


    def __str__(self):
        return self.name


## ==================================Dry Cell================================== ##
## Categories on the excel sheet that have not been addressed:
# - Mechanical Cylindrical

#TODO(sam): Electrolyte doesn't have to be dealt with.
# We will just have a ForeignKey to an electrolyte (from above).
# We can enter the info manually ourselves if needed in this case because it doesn't belong here.


# - Build Info (cell to cell confidence)

# Dry Cell

class DryCell(models.Model):

    cell_model = models.CharField(max_length=300,null=True, blank=True)
    family = models.CharField(max_length=100,null=True, blank=True)
    version_number = models.CharField(max_length=100,null=True, blank=True)
    description = models.CharField(max_length=100,null=True, blank=True)
    quantity = models.CharField(max_length=100,null=True, blank=True)
    packing_date = models.CharField(max_length=100, null=True, blank=True)
    ship_date = models.CharField(max_length=100,null=True, blank=True)
    marking_on_box = models.CharField(max_length=300,null=True, blank=True)
    shipping_soc = models.CharField(max_length=100,null=True, blank=True)
    energy_estimate_wh = models.FloatField(null=True, blank=True)
    capacity_estimate_ah = models.FloatField(null=True, blank=True)
    mass_estimate_g = models.FloatField(null=True, blank=True)
    max_charge_voltage_v = models.FloatField(null=True, blank=True)
    dcr_estimate = models.CharField(max_length=100,null=True, blank=True)
    chemistry_freeze_date_requested = models.CharField(max_length=100,null=True, blank=True)

    def __str__(self):

        if not self.cell_model is None:

            return self.cell_model

        else:

            return 'NO NAME REGISTERED'


class CellAttribute(models.Model):
    attribute = models.CharField(max_length=100,null=True,blank=True)
    dry_cells = models.ManyToManyField(DryCell, null=True)

    def __str__(self):

        return self.attribute



# Cathode

class CathodeFam(models.Model):
    cathode_family = models.CharField(max_length=100,null=True)

    def __str__(self):
        return self.cathode_family

class AnodeFam(models.Model):
    anode_family = models.CharField(max_length=100,null=True)

    def __str__(self):
        return self.anode_family

class CathodeSpecificMaterials(models.Model):
    cathode_family = models.ForeignKey(CathodeFam,on_delete=models.CASCADE, null=True)
    name = models.CharField(max_length=100,null=True)

    def __str__(self):
        return self.name

class AnodeSpecificMaterials(models.Model):
    anode_family = models.ForeignKey(AnodeFam,on_delete=models.CASCADE, null=True)
    name = models.CharField(max_length=100,null=True)

    def __str__(self):
        return self.name


class CathodeActiveMaterials(models.Model):
    coating = models.CharField(max_length=100, null=True)
    composition = models.CharField(max_length=100,null=True)

    cathode_active_1_notes = models.CharField(max_length=100, null=True)
    cathode_active_2_notes = models.CharField(max_length=100, null=True)
    cathode_active_3_notes = models.CharField(max_length=100, null=True)
    name = models.CharField(max_length=100,null=True)


class CathodeCoating(models.Model):
    name = models.CharField(max_length=100, null=True)

    def __str__(self):

        return self.name

class Cathode(models.Model):

    metal_bag_sheet_structure = models.CharField(max_length=100, null=True)
    positive_electrode_composition_notes = models.CharField(max_length=100, null=True)
    positive_electrode_loading_mg_cm2 = models.FloatField(null=True)
    positive_electrode_density_g_cm3 = models.FloatField(null=True)
    positive_electrode_porosity = models.FloatField(null=True)
    positive_electrode_thickness_um = models.FloatField(null=True)
    positive_electrode_length_single_side = models.FloatField(null=True)
    positive_electrode_length_double_side = models.FloatField(null=True)
    positive_electrode_width = models.FloatField(null=True)
    electrode_tab_position_from_core = models.FloatField(null=True)
    positive_foil_thickness_um = models.FloatField(null=True)
    positive_functional_layer_notes = models.CharField(max_length=100, null=True)
    positive_functional_thickness = models.FloatField(null=True)
    dry_cell = models.ForeignKey(DryCell,on_delete=models.CASCADE, null=True)
    cathode_active_materials = models.ForeignKey(CathodeActiveMaterials,on_delete=models.CASCADE, null=True)
    cathode_specific_materials = models.ForeignKey(CathodeSpecificMaterials,on_delete=models.CASCADE, null=True)
    coating = models.ForeignKey(CathodeCoating, on_delete=models.CASCADE, null=True, blank=True)

class CathodeConductiveAdditive(models.Model):
    notes = models.CharField(max_length=200,null=True)
    cathode = models.ForeignKey(Cathode, on_delete=models.CASCADE,null=True)


class CathodeComponent(models.Model):
    chemical_formula = models.CharField(max_length=3,null=True)
    atom_ratio = models.FloatField(null=True)
    cathode_active_materials = models.ForeignKey(CathodeActiveMaterials,on_delete=models.CASCADE, null=True)

class CathodeBinder(models.Model):
    cathode_binder_1_notes = models.CharField(max_length=100,null=True)
    cathode_binder_2_notes = models.CharField(max_length=100,null=True)
    cathode = models.ForeignKey(Cathode,on_delete=models.CASCADE, null=True)


# Anode

#TODO: I might need some advice regarding how I should describe/parse certain anode active materials.
# Examples include '(natural graphite) BTR918II' and 'carbon coated nano-Si'




class AnodeActiveMaterials(models.Model):
    coating = models.CharField(max_length=100, null=True)
    composition = models.CharField(max_length=100,null=True)
    material_id = models.CharField(max_length=100,null=True)

    anode_active_1_notes = models.CharField(max_length=100, null=True)
    anode_active_2_notes = models.CharField(max_length=100, null=True)
    anode_active_3_notes = models.CharField(max_length=100, null=True)
    anode_active_4_notes = models.CharField(max_length=100, null=True)

class Anode(models.Model):

    negative_electrode_composition_notes = models.CharField(max_length=100,null=True)
    negative_electrode_loading_mg_cm2 = models.FloatField(null=True)
    negative_electrode_density_g_cm3 = models.FloatField(null=True)
    negative_electrode_porosity = models.FloatField(null=True)
    negative_electrode_thickness_um = models.FloatField(null=True)
    negative_electrode_length_single_side = models.FloatField(null=True)
    negative_electrode_length_double_side = models.FloatField(null=True)
    negative_electrode_width = models.FloatField(null=True)
    negative_tab_position_from_core = models.FloatField(null=True)
    negative_foil_thickness_um = models.FloatField(null=True)
    negative_tab_notes = models.CharField(max_length=100,null=True)
    tab_2_notes = models.CharField(max_length=100,null=True)
    negative_functional_layer = models.CharField(max_length=100,null=True)
    negative_functional_thickness = models.FloatField(null=True)
    dry_cell = models.ForeignKey(DryCell, on_delete=models.CASCADE, null=True)
    anode_active_materials = models.ForeignKey(AnodeActiveMaterials, on_delete=models.CASCADE, null=True)
    anode_specific_materials = models.ForeignKey(AnodeSpecificMaterials,on_delete=models.CASCADE, null=True)



class AnodeConductiveAdditive(models.Model):
    notes = models.CharField(max_length=200,null=True)
    anode = models.ForeignKey(Anode,on_delete=models.CASCADE, null=True)


class AnodeComponent(models.Model):
    chemical_formula = models.CharField(max_length=3,null=True)
    atom_ratio = models.FloatField(null=True)
    anode_active_materials = models.ForeignKey(AnodeActiveMaterials,on_delete=models.CASCADE, null=True)

class AnodeType(models.Model):
    name = models.CharField(max_length=50,null=True)
    anode_category = models.CharField(max_length=100, null=True)
    preparation_temperature = models.FloatField(null=True)
    anode = models.ForeignKey(Anode,on_delete=models.CASCADE, null=True)

class AnodeBinder(models.Model):
    anode_binder_1_notes = models.CharField(max_length=200,null=True)
    anode_binder_2_notes = models.CharField(max_length=200,null=True)
    anode_binder_3_notes = models.CharField(max_length=200,null=True)
    anode = models.ForeignKey(Anode,on_delete=models.CASCADE, null=True)




# Build Info

class BuildInfo(models.Model):

    cathode_active_lot = models.IntegerField(null=True)
    anode_active_lot = models.IntegerField(null=True)
    separator_lot = models.IntegerField(null=True)
    cathode_mix_lot = models.IntegerField(null=True)
    anode_mix_lot = models.IntegerField(null=True)
    cell_assembly_lot = models.IntegerField(null=True)
    mix_coat_location = models.CharField(max_length=100, null=True)
    winding_location = models.CharField(max_length=100, null=True)
    assembly_location = models.CharField(max_length=100, null=True)
    other_mechanical_notes = models.CharField(max_length=100, null=True)
    other_electrode_notes = models.CharField(max_length=100, null=True)
    other_process_notes = models.CharField(max_length=100, null=True)
    other_notes = models.CharField(max_length=100, null=True)

    dry_cell = models.ForeignKey(DryCell,on_delete=models.CASCADE, null=True)

# Box

class Box(models.Model):
    box_id_number = models.CharField(max_length=100, null=True)
    cell_model = models.ForeignKey(DryCell,on_delete=models.CASCADE, null=True)

    def __str__(self):

        if not self.cell_model is None:

            return str(self.box_id_number) + " - " + str(self.cell_model.cell_model)

        else:

            return str(self.box_id_number) + " -  NO CELL ASSOCIATED YET"


# Mechanical Pouch

class MechanicalPouch(models.Model):
    outer_taping = models.CharField(max_length=100,null=True)
    cell_width_mm = models.FloatField(null=True)
    cell_length_mm = models.FloatField(null=True)
    cell_thickness_mm = models.FloatField(null=True)
    seal_width_side_mm = models.FloatField(null=True)
    seal_width_top_mm = models.FloatField(null=True)
    cathode_tab_polymer_material = models.CharField(max_length=100,null=True)
    anode_tab_polymer_material = models.CharField(max_length=100,null=True)
    metal_bag_sheet_thickness_mm = models.FloatField(null=True)

    dry_cell = models.ForeignKey(DryCell,on_delete=models.CASCADE, null=True)


class OtherInfo(models.Model):
    jellyroll_centering = models.CharField(max_length=100,null=True)
    ni_tab_rear_tape_material = models.CharField(max_length=100,null=True)
    ni_tab_rear_tape_width_mm = models.FloatField(max_length=100,null=True)
    anode_front_substrate_length = models.FloatField(null=True)
    anode_end_substrate_length = models.FloatField(null=True)
    negative_tab_ultra_sonic_welding_spots = models.CharField(max_length=100,null=True)
    starting_can_height_mm = models.FloatField(null=True)
    positive_tab_laser_welding_spots = models.CharField(max_length=100,null=True)
    alpha = models.CharField(max_length=100,null=True)
    beta = models.CharField(max_length=100,null=True)
    gamma = models.CharField(max_length=100,null=True)

    other_info_cell = models.ForeignKey(DryCell, on_delete=models.CASCADE, null=True)



# Separator

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

