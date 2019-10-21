from django.shortcuts import render, render_to_response
from django.forms import modelformset_factory, formset_factory
from django.db.models import Q
from .forms import *
from .models import ElectrolyteComponent, ElectrolyteMolecule, Alias, SALT, ADDITIVE, SOLVENT, Electrolyte, DryCell, WetCell
from django import forms
from ElectrolyteDatabase.models import DryCell, AnodeBinder, CathodeBinder, Cathode, CathodeConductiveAdditive,AnodeConductiveAdditive, MechanicalPouch, BuildInfo, Box, OtherInfo,\
    Separator, Anode, CathodeActiveMaterials, AnodeActiveMaterials, CathodeComponent, AnodeComponent, AnodeFam, CathodeFam, CathodeSpecificMaterials, AnodeSpecificMaterials, CathodeCoating, CellAttribute
import re


## Get Shortstring Function

def get_shortstring(component_dict):

    shortstring = ''

    if 'salts' in component_dict.keys():
        for salt in component_dict['salts'].keys():
            if (float(component_dict['salts'][salt]) % 1) == 0:
                num = int(component_dict['salts'][salt])
            else:
                num = round(float(component_dict['salts'][salt]),1)

            shortstring += '{}m{}+'.format(num, salt)

    if 'additives' in component_dict.keys():
        for additive in component_dict['additives'].keys():
            if (float(component_dict['additives'][additive]) % 1) == 0:
                num = int(component_dict['additives'][additive])
            else:
                num = round(float(component_dict['additives'][additive]),1)

            shortstring += '{}%{}+'.format(num, additive)

    if 'solvents' in component_dict.keys():
        for solvent in component_dict['solvents'].keys():
            if (float(component_dict['solvents'][solvent]) % 1) == 0:
                num = int(component_dict['solvents'][solvent])
            else:
                num = round(float(component_dict['solvents'][solvent]),1)

            shortstring += '{}%{}+'.format(num, solvent)

    shortstring = re.sub(r'\+$', '', shortstring)

    return shortstring

## ====================================================


def register_cathode_anode_info(request):

    def reload_page():

        actual_cathode_initial = []

        for cathode in Cathode.objects.all():

            if (not cathode.cathode_active_materials.cathode_active_1_notes is None) and (
            not cathode.cathode_specific_materials is None):

                if not cathode.cathode_specific_materials.name is None:
                    my_initial = {
                        'material_notes': cathode.cathode_active_materials.cathode_active_1_notes,
                        'specific_material': cathode.cathode_specific_materials.name
                    }

                    actual_cathode_initial.append(my_initial)

        actual_cathode_specific_materials_formset = ActualCathodeSpecificMaterialsFormSet(
            initial=actual_cathode_initial,
            prefix='actual_cathode_specific_materials_formset')

        actual_anode_initial = []

        for anode in Anode.objects.all():

            if (not anode.anode_active_materials.anode_active_1_notes is None) and (
            not anode.anode_specific_materials is None):

                if not anode.anode_specific_materials.name is None:
                    my_initial = {
                        'material_notes': anode.anode_active_materials.anode_active_1_notes,
                        'specific_material': anode.anode_specific_materials.name
                    }

                    actual_anode_initial.append(my_initial)

        actual_anode_specific_materials_formset = ActualAnodeSpecificMaterialsFormSet(initial=actual_anode_initial,
                                                                                      prefix='actual_anode_specific_materials_form')

        ar['actual_cathode_specific_materials_formset'] = actual_cathode_specific_materials_formset
        ar['actual_anode_specific_materials_formset'] = actual_anode_specific_materials_formset

        list_of_anode_families = []

        register_anode_family_form = RegisterAnodeFamilyForm(prefix='register_anode_family')

        for family in AnodeFam.objects.all():
            list_of_anode_families.append(family.anode_family)

        ar['list_of_anode_families'] = list_of_anode_families

        ar['register_anode_family_form'] = register_anode_family_form

        list_of_anode_specific_material_ratios = []

        register_anode_specific_materials_form = RegisterAnodeSpecificMaterialsForm(
            prefix='register_anode_specific_materials')

        for anode in AnodeSpecificMaterials.objects.all():
            list_of_anode_specific_material_ratios.append(anode)

        ar['list_of_anode_specific_material_ratios'] = list_of_anode_specific_material_ratios

        ar['register_anode_specific_materials_form'] = register_anode_specific_materials_form

        list_of_cathode_families = []

        register_cathode_family_form = RegisterCathodeFamilyForm(prefix='register_cathode_family')

        for family in CathodeFam.objects.all():
            list_of_cathode_families.append(family.cathode_family)

        ar['list_of_cathode_families'] = list_of_cathode_families

        ar['register_cathode_family_form'] = register_cathode_family_form

        list_of_cathode_specific_material_ratios = []

        register_cathode_specific_materials_form = RegisterCathodeSpecificMaterialsForm(
            prefix='register_cathode_specific_materials')

        for cathode in CathodeSpecificMaterials.objects.all():
            list_of_cathode_specific_material_ratios.append(cathode)

        ar['list_of_cathode_specific_material_ratios'] = list_of_cathode_specific_material_ratios

        ar['register_cathode_specific_materials_form'] = register_cathode_specific_materials_form

        return render(request, 'ElectrolyteDatabase/register_cathode_anode_info.html', ar)




    ar = {}

    if request.POST:

        register_anode_family_form = RegisterAnodeFamilyForm(request.POST,prefix='register_anode_family')

        if register_anode_family_form.is_valid():

            if register_anode_family_form.cleaned_data['name'] != '':

                name = register_anode_family_form.cleaned_data['name']

                family_exists = False

                for family in AnodeFam.objects.all():
                    if family.anode_family == name:
                        family_exists = True

                if not family_exists:

                    anode_family = AnodeFam(anode_family = name)

                    anode_family.save()



        register_anode_specific_materials_form = RegisterAnodeSpecificMaterialsForm(request.POST,
                                                                                        prefix='register_anode_specific_materials')

        if register_anode_specific_materials_form.is_valid():

            print(register_anode_specific_materials_form.cleaned_data)

            if (not register_anode_specific_materials_form.cleaned_data['name'] is None) and (not register_anode_specific_materials_form.cleaned_data['anode_family'] is None):

                name = register_anode_specific_materials_form.cleaned_data['name']
                anode_family = register_anode_specific_materials_form.cleaned_data['anode_family']

                material_exists = False

                for material in AnodeSpecificMaterials.objects.all():

                    if material.name == name and material.anode_family == anode_family:
                        material_exists = True

                if not material_exists:
                    anode_specific_materials = AnodeSpecificMaterials(name=name, anode_family=anode_family)

                    anode_specific_materials.save()


        register_cathode_family_form = RegisterCathodeFamilyForm(request.POST, prefix='register_cathode_family')

        if register_cathode_family_form.is_valid():

            if register_cathode_family_form.cleaned_data['name'] != '':

                name = register_cathode_family_form.cleaned_data['name']

                family_exists = False

                for family in CathodeFam.objects.all():
                    if family.cathode_family == name:
                        family_exists = True

                if not family_exists:
                    cathode_family = CathodeFam(cathode_family=name)

                    cathode_family.save()



        register_cathode_specific_materials_form = RegisterCathodeSpecificMaterialsForm(request.POST,
                                                                                    prefix='register_cathode_specific_materials')

        if register_cathode_specific_materials_form.is_valid():

            if (not register_cathode_specific_materials_form.cleaned_data['name'] is None) and (
            not register_cathode_specific_materials_form.cleaned_data['cathode_family'] is None):

                name = register_cathode_specific_materials_form.cleaned_data['name']
                cathode_family = register_cathode_specific_materials_form.cleaned_data['cathode_family']

                material_exists = False

                for material in CathodeSpecificMaterials.objects.all():

                    if material.name == name and material.cathode_family == cathode_family:
                        material_exists = True

                if not material_exists:
                    cathode_specific_materials = CathodeSpecificMaterials(name=name, cathode_family=cathode_family)

                    cathode_specific_materials.save()

        return reload_page()



    else:

        return reload_page()


def register_new_cathode_family(request):

    ar = {}

    if 'register_new_cathode_family' in request.POST:

        register_cathode_family_form = RegisterCathodeFamilyForm(request.POST,prefix='register_cathode_family')

        if register_cathode_family_form.is_valid():

            name = register_cathode_family_form.cleaned_data['name']

            family_exists = False

            for family in CathodeFam.objects.all():
                if family.cathode_family == name:
                    family_exists = True

            if not family_exists:

                cathode_family = CathodeFam(cathode_family = name)

                cathode_family.save()

        return render(request, 'ElectrolyteDatabase/main_page.html')

    else:

        list_of_cathode_families = []

        register_cathode_family_form = RegisterCathodeFamilyForm(prefix='register_cathode_family')

        for family in CathodeFam.objects.all():
            list_of_cathode_families.append(family.cathode_family)

        ar['list_of_cathode_families'] = list_of_cathode_families

        ar['register_cathode_family_form'] = register_cathode_family_form

        if len(list_of_cathode_families) > 0:
            ar['show_existing_family_table'] = True

        return render(request, 'ElectrolyteDatabase/register_new_cathode_family.html', ar)














def register_new_anode_family(request):

    ar = {}

    if 'register_new_anode_family' in request.POST:

        register_anode_family_form = RegisterAnodeFamilyForm(request.POST,prefix='register_anode_family')

        if register_anode_family_form.is_valid():

            print('valid')

            name = register_anode_family_form.cleaned_data['name']

            family_exists = False

            for family in AnodeFam.objects.all():
                if family.anode_family == name:
                    family_exists = True

            if not family_exists:

                anode_family = AnodeFam(anode_family = name)

                anode_family.save()

        return render(request, 'ElectrolyteDatabase/main_page.html')

    else:

        list_of_anode_families = []

        register_anode_family_form = RegisterAnodeFamilyForm(prefix='register_anode_family')

        for family in AnodeFam.objects.all():
            list_of_anode_families.append(family.anode_family)

        ar['list_of_anode_families'] = list_of_anode_families

        ar['register_anode_family_form'] = register_anode_family_form

        if len(list_of_anode_families) > 0:
            ar['show_existing_family_table'] = True

        return render(request, 'ElectrolyteDatabase/register_new_anode_family.html', ar)



def register_new_molecule(request):

    ar = {}

    if request.POST:

        register_molecule_formset = RegisterMoleculeFormSet(request.POST)

        if register_molecule_formset.is_valid():
            print('valid')

            instances = register_molecule_formset.save(commit=False)
            for instance in instances:
                m = ElectrolyteMolecule()

                m.name = instance.name
                m.can_be_solvent = instance.can_be_solvent
                m.can_be_additive = instance.can_be_additive
                m.can_be_salt = instance.can_be_salt

                m.save()

            return render(request, 'ElectrolyteDatabase/main_page.html')

    else:

        registered_molecules_formset = RegisteredMoleculesFormSet(queryset=ElectrolyteMolecule.objects.all())
        register_molecule_formset = RegisterMoleculeFormSet(queryset=ElectrolyteMolecule.objects.none())

        if len(ElectrolyteMolecule.objects.all()) != 0:

            ar['register_molecule_formset'] = register_molecule_formset
            ar['registered_molecules_formset'] = registered_molecules_formset

        else:
            ar['register_molecule_formset'] = register_molecule_formset

        return render(request, 'ElectrolyteDatabase/register_new_molecule.html', ar)


def register_new_electrolyte(request):

    ar = {}

    if request.POST:

        salt_component_formset = SaltComponentFormSet(request.POST, prefix='salt')
        solvent_component_formset = SolventComponentFormSet(request.POST, prefix='solvent')
        alias_name_formset = AliasNameFormSet(request.POST,prefix='alias')


        if 'submit_electrolyte' in request.POST:

            if salt_component_formset.is_valid() and solvent_component_formset.is_valid() and alias_name_formset.is_valid():

                print('all valid')

                component_dict = {
                    'salts': {},
                    'additives': {},
                    'solvents': {}
                }

                salt_instances = salt_component_formset.save(commit=False)
                solvent_instances = solvent_component_formset.save(commit=False)
                alias_instances = alias_name_formset.save(commit=False)

                for instance in salt_instances:
                    print('instance sa')
                    if instance.molal != '':
                        component_dict['salts'][instance.molecule.name] = instance.molal
                for instance in solvent_instances:
                    print('instance so')
                    if instance.weight_percent != '':
                        component_dict['solvents'][instance.molecule.name] = instance.weight_percent

                already_exists = False

                for electrolyte in Electrolyte.objects.all():
                    if electrolyte.shortstring == get_shortstring(component_dict):
                        already_exists = True

                if already_exists:
                    ar[
                        'already_exists_message'] = 'WARNING: This electrolyte already exists in the database. Would you like to add' \
                                                    ' the name you entered as an additional alias for this electrolyte?'
                    ar['alias_name_formset'] = alias_name_formset
                    ar['salt_component_formset'] = salt_component_formset
                    ar['solvent_component_formset'] = solvent_component_formset



                else:

                    elec = Electrolyte()
                    elec.save()
                    elec.shortstring = str(get_shortstring(component_dict))
                    elec.save()

                    for salt in component_dict['salts'].keys():
                        comp = ElectrolyteComponent()
                        comp.save()
                        comp.molecule = ElectrolyteMolecule.objects.get(name=salt)
                        comp.molal = component_dict['salts'][salt]
                        comp.electrolyte = elec
                        comp.compound_type = SALT
                        comp.save()
                        print('comp saved')

                    for solvent in component_dict['solvents'].keys():
                        comp = ElectrolyteComponent()
                        comp.save()
                        comp.molecule = ElectrolyteMolecule.objects.get(name=solvent)
                        comp.weight_percent = component_dict['solvents'][solvent]
                        comp.electrolyte = elec
                        comp.compound_type = SOLVENT
                        comp.save()

                    for instance in alias_instances:
                        a = Alias()
                        a.save()
                        a.name = instance.name
                        a.electrolyte = elec
                        a.save()
                        print('ALIAS SAVED')

            else:
                print('invalid')

        if 'add_alias' in request.POST:

            salt_instances = salt_component_formset.save(commit=False)
            solvent_instances = solvent_component_formset.save(commit=False)
            alias_instances = alias_name_formset.save(commit=False)

            component_dict = {
                'salts': {},
                'additives': {},
                'solvents': {}
            }

            for instance in salt_instances:
                print('instance sa')
                if instance.weight_percent != '':
                    component_dict['salts'][instance.molecule.name] = instance.molal
            for instance in solvent_instances:
                print('instance so')
                if instance.weight_percent != '':
                    component_dict['solvents'][instance.molecule.name] = instance.weight_percent

            name = ''
            for instance in alias_instances:
                name = instance.name

            a = Alias()
            a.save()
            a.name = name
            a.electrolyte = Electrolyte.objects.get(shortstring=str(get_shortstring(component_dict)))
            a.save()


    else:

        salt_component_formset = SaltComponentFormSet(queryset=ElectrolyteComponent.objects.none(), prefix='salt')
        for form in salt_component_formset:
            form.fields['molecule'].queryset = ElectrolyteMolecule.objects.filter(can_be_salt=True)

        solvent_component_formset = SolventComponentFormSet(queryset=ElectrolyteComponent.objects.none(), prefix='solvent')
        for form in solvent_component_formset:
            form.fields['molecule'].queryset = ElectrolyteMolecule.objects.filter(can_be_additive=True)

        alias_name_formset = AliasNameFormSet(queryset=Alias.objects.none(), prefix='alias')

        ar['salt_component_formset'] = salt_component_formset
        ar['solvent_component_formset'] = solvent_component_formset
        ar['alias_name_formset'] = alias_name_formset
        ar['register_new_electrolyte'] = True


    return render(request, 'ElectrolyteDatabase/register_new_electrolyte.html', ar)




def register_new_cathode_coating(request):

    ar = {}

    if 'register_new_cathode_coating' in request.POST:

        register_cathode_coating_form = RegisterCathodeCoatingForm(request.POST,prefix='register_cathode_coating')

        if register_cathode_coating_form.is_valid():

            name = register_cathode_coating_form.cleaned_data['name']

            cathode_coating = CathodeCoating(name = name)

            cathode_coating.save()

        return render(request, 'ElectrolyteDatabase/main_page.html')

    else:

        list_of_cathode_coatings = []

        register_cathode_coating_form = RegisterCathodeCoatingForm(prefix='register_cathode_coating')

        for coating in CathodeCoating.objects.all():
            list_of_cathode_coatings.append(coating.name)

        ar['list_of_cathode_coatings'] = list_of_cathode_coatings

        ar['register_cathode_coating_form'] = register_cathode_coating_form

        if len(list_of_cathode_coatings) > 0:
            ar['show_existing_coating_table'] = True


        return render(request, 'ElectrolyteDatabase/register_new_cathode_coating.html', ar)




def register_new_cell_attribute(request):

    ar = {}

    if request.POST:

        register_attribute_form = RegisterAttributeForm(request.POST,prefix='register_attribute')

        if register_attribute_form.is_valid():

            name = register_attribute_form.cleaned_data['attribute']

            print(name)

            attribute = CellAttribute(attribute = name)

            attribute.save()

        else:
            print(register_attribute_form.errors)

        return render(request, 'ElectrolyteDatabase/main_page.html')



    else:

        list_of_attributes = []

        register_attribute_form = RegisterAttributeForm(prefix='register_attribute')

        for attribute in CellAttribute.objects.all():
            list_of_attributes.append(attribute.attribute)

        ar['list_of_attributes'] = list_of_attributes

        ar['register_attribute_form'] = register_attribute_form

        if len(list_of_attributes) > 0:
            ar['show_existing_attribute_table'] = True

        return render(request, 'ElectrolyteDatabase/register_new_cell_attribute.html',ar)


def register_new_cathode_specific_material_ratio(request):

    ar = {}

    if 'register_new_cathode_specific_material_ratio' in request.POST:

        register_cathode_specific_materials_form = RegisterCathodeSpecificMaterialsForm(request.POST,prefix='register_cathode_specific_materials')

        if register_cathode_specific_materials_form.is_valid():

            name = register_cathode_specific_materials_form.cleaned_data['name']
            cathode_family = register_cathode_specific_materials_form.cleaned_data['cathode_family']

            material_exists = False

            for material in CathodeSpecificMaterials.objects.all():

                if material.name == name and material.cathode_family == cathode_family:

                    material_exists = True

            if not material_exists:

                cathode_specific_materials = CathodeSpecificMaterials(name = name, cathode_family=cathode_family)

                cathode_specific_materials.save()


        return render(request, 'ElectrolyteDatabase/main_page.html')

    else:

        list_of_cathode_specific_material_ratios = []

        register_cathode_specific_materials_form = RegisterCathodeSpecificMaterialsForm(prefix='register_cathode_specific_materials')

        for cathode in CathodeSpecificMaterials.objects.all():
            list_of_cathode_specific_material_ratios.append(cathode)

        ar['list_of_cathode_specific_material_ratios'] = list_of_cathode_specific_material_ratios

        ar['register_cathode_specific_materials_form'] = register_cathode_specific_materials_form

        if len(list_of_cathode_specific_material_ratios) > 0:
            ar['show_existing_table'] = True


        return render(request, 'ElectrolyteDatabase/register_new_cathode_specific_material_ratio.html', ar)


def register_new_anode_specific_material_ratio(request):
    ar = {}

    if 'register_new_anode_specific_material_ratio' in request.POST:

        register_anode_specific_materials_form = RegisterAnodeSpecificMaterialsForm(request.POST,
                                                                                        prefix='register_anode_specific_materials')

        if register_anode_specific_materials_form.is_valid():

            name = register_anode_specific_materials_form.cleaned_data['name']
            anode_family = register_anode_specific_materials_form.cleaned_data['anode_family']

            material_exists = False

            for material in AnodeSpecificMaterials.objects.all():

                if material.name == name and material.anode_family == anode_family:
                    material_exists = True

            if not material_exists:
                anode_specific_materials = AnodeSpecificMaterials(name=name, anode_family=anode_family)

                anode_specific_materials.save()

        return render(request, 'ElectrolyteDatabase/main_page.html')

    else:

        list_of_anode_specific_material_ratios = []

        register_anode_specific_materials_form = RegisterAnodeSpecificMaterialsForm(
            prefix='register_anode_specific_materials')

        for anode in AnodeSpecificMaterials.objects.all():
            list_of_anode_specific_material_ratios.append(anode)

        ar['list_of_anode_specific_material_ratios'] = list_of_anode_specific_material_ratios

        ar['register_anode_specific_materials_form'] = register_anode_specific_materials_form

        if len(list_of_anode_specific_material_ratios) > 0:
            ar['show_existing_table'] = True

        return render(request, 'ElectrolyteDatabase/register_new_anode_specific_material_ratio.html', ar)


def link_box_to_cell(request):



    if request.method == 'POST':

        print(request.POST)

        ar = {}

        if 'link_box_to_cell' in request.POST:

            box_list = []

            for box in Box.objects.all():

                if not box.cell_model is None:
                    box_list.append(box)

            ar['box_list'] = box_list

            link_box_to_cell_form = LinkBoxToCellForm(request.POST)

            ar['link_box_to_cell_form'] = link_box_to_cell_form

            if link_box_to_cell_form.is_valid():

                cell = link_box_to_cell_form.cleaned_data['cell_model']

                box_id = link_box_to_cell_form.cleaned_data['box_id']

                valid_box_ids = []

                for box in Box.objects.all():

                    valid_box_ids.append(int(box.box_id_number))

                if int(box_id) in valid_box_ids:

                    if not ((Box.objects.get(box_id_number=box_id)).cell_model == cell):

                        ar['box_id_already_linked'] = True

                        ar['linked_cell_model'] = (Box.objects.get(box_id_number=box_id)).cell_model

                        ar['new_cell_model'] = cell


                        return render(request, 'ElectrolyteDatabase/link_box_to_cell.html', ar)

                    else:
                        return render(request, 'ElectrolyteDatabase/main_page.html')

                else:

                    box = Box(box_id_number=box_id, cell_model=cell)
                    box.save()





        if 'overwrite_link' in request.POST:

            box_list = []

            for box in Box.objects.all():

                if not box.cell_model is None:
                    box_list.append(box)

            ar['box_list'] = box_list



            link_box_to_cell_form = LinkBoxToCellForm(request.POST)

            if link_box_to_cell_form.is_valid():

                box_id = link_box_to_cell_form.cleaned_data['box_id']

                box = Box.objects.get(box_id_number=box_id)

                box.cell_model = link_box_to_cell_form.cleaned_data['cell_model']

                box.save()

                return render(request, 'ElectrolyteDatabase/main_page.html')




        if 'leave_existing_link' in request.POST:

            box_list = []

            for box in Box.objects.all():

                if not box.cell_model is None:
                    box_list.append(box)

            ar['box_list'] = box_list

            return render(request, 'ElectrolyteDatabase/main_page.html')



    else:

        ar = {}

        box_list = []

        for box in Box.objects.all():

            if not box.cell_model is None:

                box_list.append(box)


        ar['box_list'] = box_list

        print(box_list)


        link_box_to_cell_form = LinkBoxToCellForm

        ar['link_box_to_cell_form'] = link_box_to_cell_form


    return render(request, 'ElectrolyteDatabase/link_box_to_cell.html',ar)


def register_or_modify_dry_cell_models(request):

    ar = {}

    if request.POST:

        show_modify_page = True

        ar['show_modify_page'] = show_modify_page

        if 'choose_cell' in request.POST:

            choose_dry_cell_form = ChooseDryCellForm(request.POST,prefix='choose_dry_cell')

            if choose_dry_cell_form.is_valid():

                print('valid')

                cell = choose_dry_cell_form.cleaned_data['cell_model']
                mechanical_pouch = MechanicalPouch.objects.get(dry_cell=cell)
                other_info = OtherInfo.objects.get(other_info_cell=cell)
                separator = Separator.objects.get(dry_cell=cell)
                build_info = BuildInfo.objects.get(dry_cell=cell)
                anode = Anode.objects.get(dry_cell=cell)
                cathode = Cathode.objects.get(dry_cell=cell)
                cathode_active_materials = cathode.cathode_active_materials
                anode = Anode.objects.get(dry_cell=cell)
                anode_active_materials = anode.anode_active_materials
                vendor_info = VendorInfo.objects.get(dry_cell=cell)
                anode_binder = AnodeBinder.objects.get(anode=anode)
                cathode_binder = CathodeBinder.objects.get(cathode=cathode)


                anode_binder_initial = {

                    'anode_binder_1_notes' : anode_binder.anode_binder_1_notes,
                    'anode_binder_2_notes': anode_binder.anode_binder_2_notes,
                    'anode_binder_3_notes': anode_binder.anode_binder_3_notes,
                }

                cathode_binder_initial = {

                    'cathode_binder_1_notes': cathode_binder.cathode_binder_1_notes,
                    'cathode_binder_2_notes': cathode_binder.cathode_binder_2_notes,

                }


                basics_initial = {

                    'cell_model':cell.cell_model,
                    'family':cell.family,
                    'version_number':cell.version_number,
                    'description':cell.description,
                    'quantity':cell.quantity,
                    'packing_date':cell.packing_date,
                    'ship_date':cell.ship_date,
                    'marking_on_box':cell.marking_on_box,
                    'shipping_soc':cell.shipping_soc,
                    'energy_estimate_wh':cell.energy_estimate_wh,
                    'capacity_estimate_ah':cell.capacity_estimate_ah,
                    'mass_estimate_g':cell.mass_estimate_g,
                    'max_charge_voltage_v':cell.max_charge_voltage_v,
                    'dcr_estimate':cell.dcr_estimate,
                    'chemistry_freeze_date_requested':cell.chemistry_freeze_date_requested

                    }

                mechanical_pouch_initial = {

                    'outer_taping':mechanical_pouch.outer_taping,
                    'cell_width_mm':mechanical_pouch.cell_width_mm,
                    'cell_length_mm':mechanical_pouch.cell_length_mm,
                    'cell_thickness_mm':mechanical_pouch.cell_thickness_mm,
                    'seal_width_side_mm':mechanical_pouch.seal_width_side_mm,
                    'seal_width_top_mm':mechanical_pouch.seal_width_top_mm,
                    'cathode_tab_polymer_material':mechanical_pouch.cathode_tab_polymer_material,
                    'anode_tab_polymer_material':mechanical_pouch.anode_tab_polymer_material,
                    'metal_bag_sheet_thickness_mm':mechanical_pouch.metal_bag_sheet_thickness_mm,


                    }


                other_info_intital = {

                    'jellyroll_centering':other_info.jellyroll_centering,
                    'ni_tab_rear_tape_material':other_info.ni_tab_rear_tape_material,
                    'ni_tab_rear_tape_width_mm':other_info.ni_tab_rear_tape_width_mm,
                    'anode_front_substrate_length':other_info.anode_front_substrate_length,
                    'anode_end_substrate_length':other_info.anode_end_substrate_length,
                    'negative_tab_ultra_sonic_welding_spots':other_info.negative_tab_ultra_sonic_welding_spots,
                    'starting_can_height_mm':other_info.starting_can_height_mm,
                    'positive_tab_laser_welding_spots':other_info.positive_tab_laser_welding_spots,
                    'alpha':other_info.alpha,
                    'beta':other_info.beta,
                    'gamma':other_info.gamma,

                }


                separator_initial = {

                    'separator_notes':separator.separator_notes,
                    'separator_base_thickness':separator.separator_base_thickness,
                    'separator_width_mm':separator.separator_width_mm,
                    'separator_functional_layer':separator.separator_functional_layer,
                    'separator_functional_thickness':separator.separator_functional_thickness,
                    'separator_overhang_in_core_mm':separator.separator_overhang_in_core_mm,
                }

                build_info_initial = {

                    'cathode_active_lot':build_info.cathode_active_lot,
                    'anode_active_lot':build_info.anode_active_lot,
                    'separator_lot':build_info.separator_lot,
                    'cathode_mix_lot':build_info.cathode_mix_lot,
                    'anode_mix_lot':build_info.anode_mix_lot,
                    'cell_assembly_lot':build_info.cell_assembly_lot,
                    'mix_coat_location':build_info.mix_coat_location,
                    'winding_location':build_info.winding_location,
                    'assembly_location':build_info.assembly_location,
                    'other_mechanical_notes':build_info.other_mechanical_notes,
                    'other_electrode_notes':build_info.other_electrode_notes,
                    'other_process_notes':build_info.other_process_notes,
                    'other_notes':build_info.other_notes,
                }


                anode_initial = {

                    'negative_electrode_composition_notes':anode.negative_electrode_composition_notes,
                    'negative_electrode_loading_mg_cm2':anode.negative_electrode_loading_mg_cm2,
                    'negative_electrode_density_g_cm3':anode.negative_electrode_density_g_cm3,
                    'negative_electrode_porosity':anode.negative_electrode_porosity,
                    'negative_electrode_thickness_um':anode.negative_electrode_thickness_um,
                    'negative_electrode_length_single_side':anode.negative_electrode_length_single_side,
                    'negative_electrode_length_double_side':anode.negative_electrode_length_double_side,
                    'negative_electrode_width':anode.negative_electrode_width,
                    'negative_tab_position_from_core':anode.negative_tab_position_from_core,
                    'negative_foil_thickness_um':anode.negative_foil_thickness_um,
                    'negative_tab_notes':anode.negative_tab_notes,
                    'tab_2_notes':anode.tab_2_notes,
                    'negative_functional_layer':anode.negative_functional_layer,
                    'negative_functional_thickness':anode.negative_functional_thickness,
                    'anode_specific_materials':anode.anode_specific_materials,
                }

                anode_active_materials_initial = {

                    'material_id':anode_active_materials.material_id,
                    'anode_active_1_notes':anode_active_materials.anode_active_1_notes,
                    'anode_active_2_notes':anode_active_materials.anode_active_2_notes,
                    'anode_active_3_notes':anode_active_materials.anode_active_3_notes,
                    'anode_active_4_notes':anode_active_materials.anode_active_4_notes,
                }

                vendor_info_initial = {

                    'cathode_active_1_vendor':vendor_info.cathode_active_1_vendor,
                    'cathode_active_2_vendor':vendor_info.cathode_active_2_vendor,
                    'cathode_active_3_vendor':vendor_info.cathode_active_3_vendor,

                    'cathode_additive_vendor':vendor_info.cathode_additive_vendor,

                    'cathode_binder_1_vendor':vendor_info.cathode_binder_1_vendor,
                    'cathode_binder_2_vendor':vendor_info.cathode_binder_2_vendor,
                    'cathode_binder_3_vendor':vendor_info.cathode_binder_3_vendor,

                    'anode_active_1_vendor':vendor_info.anode_active_1_vendor,
                    'anode_active_2_vendor':vendor_info.anode_active_2_vendor,
                    'anode_active_3_vendor':vendor_info.anode_active_3_vendor,
                    'anode_active_4_vendor':vendor_info.anode_active_4_vendor,

                    'anode_binder_1_vendor':vendor_info.anode_binder_1_vendor,
                    'anode_binder_2_vendor':vendor_info.anode_binder_2_vendor,
                    'anode_binder_3_vendor':vendor_info.anode_binder_3_vendor,

                    'negative_foil_vendor':vendor_info.negative_foil_vendor,
                    'separator_vendor':vendor_info.separator_vendor,
                    'separator_coat_vendor':vendor_info.separator_coat_vendor,
                    'gasket_vendor':vendor_info.gasket_vendor,
                    'can_vendor':vendor_info.can_vendor,
                    'top_cap_vendor':vendor_info.top_cap_vendor,
                    'outer_tape_vendor':vendor_info.outer_tape_vendor,

                    'electrolyte_solvent_1_vendor':vendor_info.electrolyte_solvent_1_vendor,
                    'electrolyte_solvent_2_vendor':vendor_info.electrolyte_solvent_2_vendor,
                    'electrolyte_solvent_3_vendor':vendor_info.electrolyte_solvent_3_vendor,
                    'electrolyte_solvent_4_vendor':vendor_info.electrolyte_solvent_4_vendor,
                    'electrolyte_solvent_5_vendor':vendor_info.electrolyte_solvent_5_vendor,

                }

                cathode_initial = {

                    'metal_bag_sheet_structure':cathode.metal_bag_sheet_structure,
                    'positive_electrode_composition_notes':cathode.positive_electrode_composition_notes,
                    'positive_electrode_loading_mg_cm2':cathode.positive_electrode_loading_mg_cm2,
                    'positive_electrode_density_g_cm3':cathode.positive_electrode_density_g_cm3,
                    'positive_electrode_porosity':cathode.positive_electrode_porosity,
                    'positive_electrode_thickness_um':cathode.positive_electrode_thickness_um,
                    'positive_electrode_length_single_side':cathode.positive_electrode_length_single_side,
                    'positive_electrode_length_double_side':cathode.positive_electrode_length_double_side,
                    'positive_electrode_width':cathode.positive_electrode_width,
                    'electrode_tab_position_from_core':cathode.electrode_tab_position_from_core,
                    'positive_foil_thickness_um':cathode.positive_foil_thickness_um,
                    'positive_functional_layer_notes':cathode.positive_functional_layer_notes,
                    'positive_functional_thickness':cathode.positive_functional_thickness,
                    'coating':cathode.coating,
                    'cathode_specific_materials':cathode.cathode_specific_materials
                }

                cathode_active_materials_initial = {

                    'cathode_active_1_notes':cathode_active_materials.cathode_active_1_notes,
                    'cathode_active_2_notes':cathode_active_materials.cathode_active_2_notes,
                    'cathode_active_3_notes':cathode_active_materials.cathode_active_3_notes,

                }

                cathode_components_initial = []

                cathode_component_count = 1

                for cathode_component in CathodeComponent.objects.filter(cathode_active_materials = cathode_active_materials):

                    my_initial = {
                    'chemical_formula'.format(cathode_component_count) : cathode_component.chemical_formula,
                    'atom_ratio'.format(cathode_component) : cathode_component.atom_ratio,
                    'hidden_id' : cathode_component.id
                    }
                    cathode_components_initial.append(my_initial)

                    cathode_component_count += 1

                    if cathode_component_count > 6:
                        break



                anode_components_initial = []

                anode_component_count = 1

                for anode_component in AnodeComponent.objects.filter(anode_active_materials = anode_active_materials):

                    my_initial = {
                    'chemical_formula'.format(anode_component_count) : anode_component.chemical_formula,
                    'atom_ratio'.format(anode_component) : anode_component.atom_ratio,
                    'hidden_id' : anode_component.id
                    }

                    anode_components_initial.append(my_initial)

                    anode_component_count += 1

                    if anode_component_count > 6:

                        break



                anode_conductive_additive_initial = []

                for anode_additive in AnodeConductiveAdditive.objects.filter(anode=anode):

                    my_initial = {
                        'notes':anode_additive.notes
                    }

                    anode_conductive_additive_initial.append(my_initial)


                cathode_conductive_additive_initial = []

                for cathode_additive in CathodeConductiveAdditive.objects.filter(cathode=cathode):
                    my_initial = {
                        'notes': cathode_additive.notes
                    }

                    cathode_conductive_additive_initial.append(my_initial)



                cell_attribute_initial = []

                for attribute in CellAttribute.objects.filter(dry_cells=cell):

                    my_initial = {
                        'attribute':attribute,
                        'hidden_id':attribute.id
                    }

                    cell_attribute_initial.append(my_initial)


                anode_conductive_additive_formset = AnodeConductiveAdditiveFormSet(initial=anode_conductive_additive_initial,prefix='anode_conductive_additive')
                cathode_conductive_additive_formset = CathodeConductiveAdditiveFormSet(initial=cathode_conductive_additive_initial,prefix='cathode_conductive_additive')
                cathode_component_formset = CathodeComponentFormSet(initial=cathode_components_initial,prefix='cathode_components')
                anode_component_formset = AnodeComponentFormSet(initial=anode_components_initial,prefix='anode_components')
                cell_basics_form = CellBasicsForm(initial=basics_initial,prefix='cell_basics')
                cell_mechanical_pouch_form = CellMechanicalPouchForm(initial=mechanical_pouch_initial,prefix='mechanical_pouch')
                cell_other_info_form = CellOtherInfoForm(initial=other_info_intital,prefix='other_info')
                cell_separator_form = CellSeparatorForm(initial=separator_initial,prefix='separator')
                cell_build_info_form = CellBuildInfoForm(initial=build_info_initial,prefix='build_info')
                cell_anode_form = CellAnodeForm(initial=anode_initial,prefix='anode')
                cell_anode_active_materials_form = CellAnodeActiveMaterialsForm(initial=anode_active_materials_initial,prefix='anode_active_materials')
                cell_cathode_form = CellCathodeForm(initial=cathode_initial,prefix='cathode')
                cell_cathode_active_materials_form = CellCathodeActiveMaterialsForm(initial=cathode_active_materials_initial,prefix='cathode_active_materials')
                cell_attribute_formset = CellAttributeFormSet(initial=cell_attribute_initial, prefix='attribute')
                vendor_info_form = VendorInfoForm(initial = vendor_info_initial, prefix='vendor_info')

                cathode_binder_form = CathodeBinderForm(initial=cathode_binder_initial, prefix='cathode_binder')
                anode_binder_form = AnodeBinderForm(initial=anode_binder_initial, prefix='anode_binder')


                ar['cathode_binder_form'] = cathode_binder_form
                ar['anode_binder_form'] = anode_binder_form
                ar['vendor_info_form'] = vendor_info_form
                ar['cathode_conductive_additive_formset'] = cathode_conductive_additive_formset
                ar['anode_conductive_additive_formset'] = anode_conductive_additive_formset
                ar['cell_attribute_formset'] = cell_attribute_formset
                ar['cathode_component_formset'] = cathode_component_formset
                ar['anode_component_formset'] = anode_component_formset
                ar['cell_basics_form'] = cell_basics_form
                ar['cell_mechanical_pouch_form'] = cell_mechanical_pouch_form
                ar['cell_other_info_form'] = cell_other_info_form
                ar['cell_separator_form'] = cell_separator_form
                ar['cell_build_info_form'] = cell_build_info_form
                ar['cell_anode_form'] = cell_anode_form
                ar['cell_anode_active_materials_form'] = cell_anode_active_materials_form
                ar['cell_cathode_form'] = cell_cathode_form
                ar['cell_cathode_active_materials_form'] = cell_cathode_active_materials_form
                ar['choose_dry_cell_form'] = choose_dry_cell_form


        if 'make_changes_to_dry_cell' in request.POST:

            choose_dry_cell_form = ChooseDryCellForm(request.POST, prefix='choose_dry_cell')


            if choose_dry_cell_form.is_valid():


                cell = choose_dry_cell_form.cleaned_data['cell_model']
                mechanical_pouch = MechanicalPouch.objects.get(dry_cell=cell)
                other_info = OtherInfo.objects.get(other_info_cell=cell)
                separator = Separator.objects.get(dry_cell=cell)
                build_info = BuildInfo.objects.get(dry_cell=cell)
                cathode = Cathode.objects.get(dry_cell=cell)
                cathode_active_materials = cathode.cathode_active_materials
                anode = Anode.objects.get(dry_cell=cell)
                anode_active_materials = anode.anode_active_materials
                vendor_info = VendorInfo.objects.get(dry_cell=cell)

                cathode_binder = CathodeBinder.objects.get(cathode=cathode)
                anode_binder = AnodeBinder.objects.get(anode=anode)

                cathode_conductive_additive_formset = CathodeConductiveAdditiveFormSet(request.POST,prefix='cathode_conductive_additive')
                anode_conductive_additive_formset = AnodeConductiveAdditiveFormSet(request.POST,prefix='anode_conductive_additive')


                vendor_info_form = VendorInfoForm(request.POST,instance=vendor_info,prefix='vendor_info')

                vendor_info_form.save()

                if cathode_conductive_additive_formset.is_valid():

                    for cathode_additive in CathodeConductiveAdditive.objects.filter(cathode=cathode):

                        cathode_additive.delete()


                    for form in cathode_conductive_additive_formset:

                        if len(form.cleaned_data) > 0:

                            cathode_conductive_additive = CathodeConductiveAdditive(notes=form.cleaned_data['notes'],cathode=cathode)
                            cathode_conductive_additive.save()

                if anode_conductive_additive_formset.is_valid():

                    for anode_additive in AnodeConductiveAdditive.objects.filter(anode=anode):
                        anode_additive.delete()

                    for form in anode_conductive_additive_formset:

                        if len(form.cleaned_data) > 0:
                            anode_conductive_additive = AnodeConductiveAdditive(notes=form.cleaned_data['notes'],
                                                                                anode=anode)
                            anode_conductive_additive.save()

                cell_basics_form = CellBasicsForm(request.POST, instance=cell, prefix='cell_basics')

                cell_basics_form.save()



                cathode_binder_form = CathodeBinderForm(request.POST, instance=cathode_binder, prefix='cathode_binder')

                cathode_binder_form.save()


                anode_binder_form = AnodeBinderForm(request.POST, instance=anode_binder, prefix='anode_binder')

                anode_binder_form.save()


                cell_mechanical_pouch_form = CellMechanicalPouchForm(request.POST, instance=mechanical_pouch, prefix='mechanical_pouch')

                cell_mechanical_pouch_form.save()


                cell_other_info_form = CellOtherInfoForm(request.POST, instance=other_info, prefix='other_info')

                cell_other_info_form.save()


                cell_separator_form = CellSeparatorForm(request.POST, instance=separator, prefix='separator')

                cell_separator_form.save()


                cell_build_info_form = CellBuildInfoForm(request.POST, instance=build_info, prefix='build_info')

                cell_build_info_form.save()


                cell_anode_form = CellAnodeForm(request.POST, instance=anode, prefix='anode')

                cell_anode_form.save()


                cell_anode_active_materials_form = CellAnodeActiveMaterialsForm(request.POST, instance=anode_active_materials, prefix='anode_active_materials')

                cell_anode_active_materials_form.save()


                cell_cathode_form = CellCathodeForm(request.POST, instance=cathode, prefix='cathode')

                cell_cathode_form.save()


                cell_cathode_active_materials_form = CellCathodeActiveMaterialsForm(request.POST, instance=cathode_active_materials, prefix='cathode_active_materials')

                cell_cathode_active_materials_form.save()


                cathode_component_formset = CathodeComponentFormSet(request.POST, prefix='cathode_components')
                anode_component_formset = AnodeComponentFormSet(request.POST, prefix='anode_components')
                cell_attribute_formset = CellAttributeFormSet(request.POST, prefix='attribute')

                if cell_attribute_formset.is_valid():

                    for form in cell_attribute_formset:

                        if len(form.cleaned_data) > 0:

                            if not form.cleaned_data['hidden_id'] is None:

                                hidden_id = int(form.cleaned_data['hidden_id'])

                                attribute = CellAttribute.objects.get(id=hidden_id)

                                if not attribute == form.cleaned_data['attribute']:

                                    attribute.dry_cells.remove(cell)

                                    new_attribute = form.cleaned_data['attribute']

                                    if not new_attribute is None:

                                        new_attribute.dry_cells.add(cell)

                                        new_attribute.save()


                            else:

                                new_attribute = form.cleaned_data['attribute']

                                new_attribute.dry_cells.add(cell)




                if cathode_component_formset.is_valid():

                    for form in cathode_component_formset:

                        if len(form.cleaned_data) > 0:

                            if not form.cleaned_data['hidden_id'] is None:

                                hidden_id = int(form.cleaned_data['hidden_id'])

                                cathode_component = CathodeComponent.objects.get(id=hidden_id)

                                cathode_component.atom_ratio = form.cleaned_data['atom_ratio']

                                cathode_component.chemical_formula = form.cleaned_data['chemical_formula']

                                cathode_component.save()

                            else:

                                cathode_component = CathodeComponent(cathode_active_materials=cathode_active_materials, atom_ratio=form.cleaned_data['atom_ratio'],
                                                                     chemical_formula=form.cleaned_data['chemical_formula'])
                                cathode_component.save()


                if anode_component_formset.is_valid():

                    for form in anode_component_formset:


                        if len(form.cleaned_data) > 0:

                            if not form.cleaned_data['hidden_id'] is None:

                                hidden_id = int(form.cleaned_data['hidden_id'])

                                anode_component = AnodeComponent.objects.get(id=hidden_id)

                                anode_component.atom_ratio = form.cleaned_data['atom_ratio']

                                anode_component.chemical_formula = form.cleaned_data['chemical_formula']

                                anode_component.save()

                            else:

                                anode_component = AnodeComponent(anode_active_materials=anode_active_materials, atom_ratio=form.cleaned_data['atom_ratio'],
                                                                     chemical_formula=form.cleaned_data['chemical_formula'])
                                anode_component.save()

                return render(request, 'ElectrolyteDatabase/main_page.html')


        if 'register_new_dry_cell' in request.POST:

            cathode_component_formset = CathodeComponentFormSet(prefix='cathode_components')
            anode_component_formset = AnodeComponentFormSet(
                                                            prefix='anode_components')
            vendor_info_form = VendorInfoForm(prefix='vendor_info')
            cathode_conductive_additive_formset = CathodeConductiveAdditiveFormSet(prefix='cathode_conductive_additive')
            anode_conductive_additive_formset = AnodeConductiveAdditiveFormSet(prefix='anode_conductive_additive')
            cell_basics_form = CellBasicsForm( prefix='cell_basics')
            cell_mechanical_pouch_form = CellMechanicalPouchForm(
                                                                 prefix='mechanical_pouch')
            cell_other_info_form = CellOtherInfoForm( prefix='other_info')
            cell_separator_form = CellSeparatorForm( prefix='separator')
            cell_build_info_form = CellBuildInfoForm( prefix='build_info')
            cell_anode_form = CellAnodeForm( prefix='anode')
            cell_anode_active_materials_form = CellAnodeActiveMaterialsForm(
                                                                            prefix='anode_active_materials')
            cell_cathode_form = CellCathodeForm( prefix='cathode')
            cell_cathode_active_materials_form = CellCathodeActiveMaterialsForm(
                 prefix='cathode_active_materials')
            cell_attribute_formset = CellAttributeFormSet( prefix='attribute')

            anode_binder_form = AnodeBinderForm(prefix='anode_binder')
            cathode_binder_form = CathodeBinderForm(prefix='cathode_binder')

            ar['cell_attribute_formset'] = cell_attribute_formset


            ar['cathode_conductive_additive_formset'] = cathode_conductive_additive_formset
            ar['anode_conductive_additive_formset'] = anode_conductive_additive_formset
            ar['vendor_info_form'] = vendor_info_form
            ar['cathode_component_formset'] = cathode_component_formset
            ar['anode_component_formset'] = anode_component_formset
            ar['cell_basics_form'] = cell_basics_form
            ar['cell_mechanical_pouch_form'] = cell_mechanical_pouch_form
            ar['cell_other_info_form'] = cell_other_info_form
            ar['cell_separator_form'] = cell_separator_form
            ar['cell_build_info_form'] = cell_build_info_form
            ar['cell_anode_form'] = cell_anode_form
            ar['cell_anode_active_materials_form'] = cell_anode_active_materials_form
            ar['cell_cathode_form'] = cell_cathode_form
            ar['cell_cathode_active_materials_form'] = cell_cathode_active_materials_form
            ar['new_dry_cell'] = True

            ar['anode_binder_form'] = anode_binder_form
            ar['cathode_binder_form'] = cathode_binder_form




        if 'submit_new_dry_cell' in request.POST:

            dry_cell = DryCell()
            dry_cell.save()
            mechanical_pouch = MechanicalPouch(dry_cell=dry_cell)
            mechanical_pouch.save()
            build_info = BuildInfo(dry_cell=dry_cell)
            build_info.save()
            separator = Separator(dry_cell=dry_cell)
            separator.save()
            anode_active_materials = AnodeActiveMaterials()
            anode_active_materials.save()
            cathode_active_materials = CathodeActiveMaterials()
            cathode_active_materials.save()
            cathode = Cathode(dry_cell=dry_cell,cathode_active_materials=cathode_active_materials)
            cathode.save()
            anode = Anode(dry_cell=dry_cell,anode_active_materials=anode_active_materials)
            anode.save()
            other_info = OtherInfo(other_info_cell=dry_cell)
            other_info.save()
            vendor_info = VendorInfo(dry_cell=dry_cell)
            vendor_info.save()
            cathode_binder = CathodeBinder(cathode=cathode)
            cathode_binder.save()
            anode_binder = AnodeBinder(anode=anode)
            anode_binder.save()


            cathode_conductive_additive_formset = CathodeConductiveAdditiveFormSet(request.POST,prefix='cathode_conductive_additive')

            if cathode_conductive_additive_formset.is_valid():

                for form in cathode_conductive_additive_formset:

                    if len(form.cleaned_data) > 0:
                        cathode_conductive_additive = CathodeConductiveAdditive(notes=form.cleaned_data['notes'],
                                                                                cathode=cathode)
                        cathode_conductive_additive.save()


            anode_conductive_additive_formset = AnodeConductiveAdditiveFormSet(request.POST,prefix='anode_conductive_additive')

            if anode_conductive_additive_formset.is_valid():

                for form in anode_conductive_additive_formset:

                    if len(form.cleaned_data) > 0:
                        anode_conductive_additive = AnodeConductiveAdditive(notes=form.cleaned_data['notes'],
                                                                                anode=anode)
                        anode_conductive_additive.save()

            vendor_info_form = VendorInfoForm(request.POST,instance=vendor_info,prefix='vendor_info')

            vendor_info_form.save()

            cathode_binder_form = CathodeBinderForm(request.POST, instance=cathode_binder, prefix='cathode_binder')

            cathode_binder_form.save()

            anode_binder_form = AnodeBinderForm(request.POST, instance=anode_binder, prefix='anode_binder')

            anode_binder_form.save()

            cell_basics_form = CellBasicsForm(request.POST, instance=dry_cell, prefix='cell_basics')

            cell_basics_form.save()

            cell_mechanical_pouch_form = CellMechanicalPouchForm(request.POST, instance=mechanical_pouch,
                                                                 prefix='mechanical_pouch')
            cell_mechanical_pouch_form.save()

            cell_other_info_form = CellOtherInfoForm(request.POST, instance=other_info, prefix='other_info')

            cell_other_info_form.save()

            cell_separator_form = CellSeparatorForm(request.POST, instance=separator, prefix='separator')

            cell_separator_form.save()

            cell_build_info_form = CellBuildInfoForm(request.POST, instance=build_info, prefix='build_info')

            cell_build_info_form.save()

            cell_anode_form = CellAnodeForm(request.POST, instance=anode, prefix='anode')

            cell_anode_form.save()

            cell_anode_active_materials_form = CellAnodeActiveMaterialsForm(request.POST, instance=anode_active_materials,
                                                                            prefix='anode_active_materials')

            cell_anode_active_materials_form.save()

            cell_cathode_form = CellCathodeForm(request.POST, instance=cathode, prefix='cathode')

            cell_cathode_form.save()

            cell_cathode_active_materials_form = CellCathodeActiveMaterialsForm(request.POST,
                                                                                instance=cathode_active_materials,
                                                                                prefix='cathode_active_materials')

            cell_cathode_active_materials_form.save()

            cathode_component_formset = CathodeComponentFormSet(request.POST, prefix='cathode_components')
            anode_component_formset = AnodeComponentFormSet(request.POST, prefix='anode_components')
            cell_attribute_formset = CellAttributeFormSet(request.POST, prefix='attribute')

            if cell_attribute_formset.is_valid():

                for form in cell_attribute_formset:

                    if len(form.cleaned_data) > 0:

                            new_attribute = form.cleaned_data['attribute']

                            new_attribute.dry_cells.add(dry_cell)


            if cathode_component_formset.is_valid():

                for form in cathode_component_formset:

                    if len(form.cleaned_data) > 0:

                        cathode_component = CathodeComponent(cathode_active_materials=cathode_active_materials,
                                                             atom_ratio=form.cleaned_data['atom_ratio'],
                                                             chemical_formula=form.cleaned_data['chemical_formula'])
                        cathode_component.save()


            if anode_component_formset.is_valid():

                for form in anode_component_formset:

                    if len(form.cleaned_data) > 0:

                        anode_component = AnodeComponent(anode_active_materials=anode_active_materials,
                                                         atom_ratio=form.cleaned_data['atom_ratio'],
                                                         chemical_formula=form.cleaned_data['chemical_formula'])
                        anode_component.save()

            return render(request, 'ElectrolyteDatabase/main_page.html')




    else:

        choose_dry_cell_form = ChooseDryCellForm(prefix='choose_dry_cell')

        ar['choose_dry_cell_form'] = choose_dry_cell_form



    return render(request, 'ElectrolyteDatabase/register_or_modify_dry_cell_models.html', ar)





def specify_and_edit_wet_cells(request):

    ar = {}

    dry_cells_with_no_cathode_ratio = []

    for cathode in Cathode.objects.all():

        if cathode.cathode_specific_materials is None:

            dry_cells_with_no_cathode_ratio.append(cathode.dry_cell.cell_model)

    ar['dry_cells_with_no_cathode_ratio'] = dry_cells_with_no_cathode_ratio


    dry_cells_with_no_anode_ratio = []

    for anode in Anode.objects.all():

        if anode.anode_specific_materials is None:

            dry_cells_with_no_anode_ratio.append(anode.dry_cell.cell_model)

    ar['dry_cells_with_no_anode_ratio'] = dry_cells_with_no_anode_ratio


    if request.POST:

        specify_electrolyte_form = SpecifyElectrolyteForm(request.POST, prefix='electrolyte')
        specify_dry_cell_form = SpecifyDryCellForm(request.POST, prefix='dry_cell')
        salt_component_formset = SaltComponentFormSet(request.POST, prefix='salt')
        solvent_component_formset = SolventComponentFormSet(request.POST, prefix='solvent')
        absent_component_formset = AbsentComponentFormSet(request.POST, prefix='absent')
        box_queryset_formset = BoxQuerysetFormSet(request.POST, prefix='box_queryset')
        electrolyte_queryset_formset = ElectrolyteQuerysetFormSet(request.POST, prefix='electrolyte_queryset')
        cell_list_formset = CellListFormSet(request.POST, prefix='cell_list')
        wet_cell_metadata_edit_formset = WetCellMetadataEditFormSet(request.POST, prefix='wet_cell_metadata')
        cell_id_range_formset = CellIDRangeFormSet(request.POST,prefix='cell_id')
        box_id_formset = BoxIDFormSet(request.POST,prefix='box_id')

        if 'update_box_queryset_formset' in request.POST:

            specific_electrolyte_formset = SpecificElectrolyteFormSet(request.POST, prefix='specific_electrolyte')

            salt_component_formset = SaltComponentFormSet(request.POST, queryset=ElectrolyteComponent.objects.none(), prefix='salt')
            for form in salt_component_formset:
                form.fields['molecule'].queryset = ElectrolyteMolecule.objects.filter(can_be_salt=True)

            solvent_component_formset = SolventComponentFormSet(request.POST, queryset=ElectrolyteComponent.objects.none(), prefix='solvent')
            for form in solvent_component_formset:
                form.fields['molecule'].queryset = ElectrolyteMolecule.objects.filter(can_be_salt=True)

            q = Q()

            ## Expanding/Contracting Table

            if 'cathode_data' in request.POST:

                dry_cell_cathode_data = request.POST['cathode_data']

                if re.match(r'Family', dry_cell_cathode_data):

                    family = CathodeFam.objects.get(cathode_family=re.sub(r'Family-', '', dry_cell_cathode_data))

                    for cathode in Cathode.objects.filter(cathode_specific_materials__cathode_family=family):

                        q = q | Q(cell_model=cathode.dry_cell)

                if re.match(r'Ratio\+Family', dry_cell_cathode_data):

                    family = CathodeFam.objects.get(cathode_family=re.sub(r'Ratio\+Family-|\d.+\d$', '', dry_cell_cathode_data))

                    ratio = re.sub(r'Ratio\+Family-\D+', '', dry_cell_cathode_data)

                    for cathode in Cathode.objects.filter(cathode_specific_materials__cathode_family=family, cathode_specific_materials=ratio):
                        q = q | Q(cell_model=cathode.dry_cell)

                if re.match(r'Example', dry_cell_cathode_data):

                    note = re.sub(r'Example-', '', dry_cell_cathode_data)

                    for cathode in Cathode.objects.filter(cathode_active_materials__cathode_active_1_notes=note):

                        q = q | Q(cell_model=cathode.dry_cell)






            if 'anode_data' in request.POST:

                dry_cell_anode_data = request.POST['anode_data']

                if re.match(r'Family', dry_cell_anode_data):

                    family = AnodeFam.objects.get(anode_family=re.sub(r'Family-', '', dry_cell_anode_data))

                    for anode in Anode.objects.filter(anode_specific_materials__anode_family=family):

                        q = q | Q(cell_model=anode.dry_cell)

                if re.match(r'Ratio\+Family', dry_cell_anode_data):

                    family = AnodeFam.objects.get(anode_family=re.sub(r'Ratio\+Family-|\d.+\d$', '', dry_cell_anode_data))

                    ratio = re.sub(r'Ratio\+Family-\D+', '', dry_cell_anode_data)

                    for anode in Anode.objects.filter(anode_specific_materials__anode_family=family, anode_specific_materials=ratio):
                        q = q | Q(cell_model=anode.dry_cell)

                if re.match(r'Example', dry_cell_anode_data):

                    note = re.sub(r'Example-', '', dry_cell_anode_data)

                    for anode in Anode.objects.filter(anode_active_materials__anode_active_1_notes=note):

                        q = q | Q(cell_model=anode.dry_cell)

            ## Cathode Expanding/Contracting Table

            cathode_data_list = []
            family_list = []
            ratio_list = []
            note_list = []

            for cell in DryCell.objects.all():

                cathode = Cathode.objects.get(dry_cell=cell)

                cathode_active_materials = cathode.cathode_active_materials

                if not cathode.cathode_specific_materials is None:

                    cathode_family = cathode.cathode_specific_materials.cathode_family

                    cathode_ratio = cathode.cathode_specific_materials

                    if not cathode_family in family_list:
                        cathode_data_list.append((cathode_family, []))

                        family_list.append(cathode_family)

                    if not cathode_ratio is None:

                        if not (str(cathode_ratio)) in ratio_list:

                            for fam in cathode_data_list:
                                if fam[0] == cathode_family:
                                    fam[1].append(((str(cathode_ratio)), []))

                            ratio_list.append(str(cathode_ratio))

                    if not cathode_active_materials.cathode_active_1_notes is None:

                        if not cathode_active_materials.cathode_active_1_notes in note_list:

                            for fam in cathode_data_list:

                                if fam[0] == cathode_family:

                                    for ratio in fam[1]:

                                        if ratio[0] == (str(cathode_ratio)):
                                            ratio[1].append(cathode_active_materials.cathode_active_1_notes)

                            note_list.append(cathode_active_materials.cathode_active_1_notes)

            anode_data_list = []
            family_list = []
            ratio_list = []
            note_list = []

            for cell in DryCell.objects.all():

                anode = Anode.objects.get(dry_cell=cell)

                anode_active_materials = anode.anode_active_materials

                if not anode.anode_specific_materials is None:

                    anode_family = anode.anode_specific_materials.anode_family

                    anode_ratio = anode.anode_specific_materials

                    if not anode_family in family_list:
                        anode_data_list.append((anode_family, []))

                        family_list.append(anode_family)

                    if not anode_ratio is None:

                        if not (str(anode_ratio)) in ratio_list:

                            for fam in anode_data_list:
                                if fam[0] == anode_family:
                                    fam[1].append(((str(anode_ratio)), []))

                            ratio_list.append(str(anode_ratio))

                    if not anode_active_materials.anode_active_1_notes is None:

                        if not anode_active_materials.anode_active_1_notes in note_list:

                            for fam in anode_data_list:

                                if fam[0] == anode_family:

                                    for ratio in fam[1]:

                                        if ratio[0] == (str(anode_ratio)):
                                            ratio[1].append(anode_active_materials.anode_active_1_notes)

                            note_list.append(anode_active_materials.anode_active_1_notes)

            if len(cell_list_formset) != 0:
                ar['show_cell_list_formset'] = True

            if specify_dry_cell_form.is_valid():

                if not specify_dry_cell_form.cleaned_data['cell_model'] is None:
                    q = q & Q(cell_model__cell_model=specify_dry_cell_form.cleaned_data['cell_model'])

                if specify_dry_cell_form.cleaned_data['cell_description'] != '':
                    q = q & Q(cell_model__description__icontains=specify_dry_cell_form.cleaned_data['cell_description'])


                if not specify_dry_cell_form.cleaned_data['cell_attribute'] is None:

                    attribute = specify_dry_cell_form.cleaned_data['cell_attribute']

                    for cell in DryCell.objects.all():

                        if not cell in attribute.dry_cells.all():

                            q = q & ~Q(cell_model=cell)

                if not specify_dry_cell_form.cleaned_data['cathode_coating'] is None:

                    cathodes = Cathode.objects.filter(coating=specify_dry_cell_form.cleaned_data['cathode_coating'])

                    cells = []

                    for cathode in cathodes:

                        if not cathode.dry_cell in cells:

                            cells.append(cathode.dry_cell)

                    for cell in DryCell.objects.all():

                        if cell not in cells:

                            q = q & ~Q(cell_model=cell)

            if box_id_formset.is_valid():

                for form in box_id_formset:

                    if 'box_id' in form.cleaned_data:

                        if not form.cleaned_data['box_id'] is None:
                            q = q | Q(box_id_number=form.cleaned_data['box_id'])

            q = q & ~Q(box_id_number=None)

            box_queryset = Box.objects.filter(q)

            ignore_queryset = []
            good_box_queryset = []

            if box_queryset_formset.is_valid():

                for form in box_queryset_formset:

                    if 'ignore' in form.cleaned_data:

                        if form.cleaned_data['ignore']:

                            ignore_queryset.append(Box.objects.get(box_id_number=form.cleaned_data['box_id']))

                        else:
                            good_box_queryset.append(Box.objects.get(box_id_number=form.cleaned_data['box_id']))

            initial = []

            if q == Q() and len(ignore_queryset) == 0:
                box_queryset = Box.objects.all()

            for box in box_queryset:

                if box in ignore_queryset:
                    my_initial = {
                        'box_id': box.box_id_number,
                        'cell_model': box.cell_model.cell_model,
                        'ignore': True
                    }
                    initial.append(my_initial)

                else:
                    my_initial = {
                        'box_id': box.box_id_number,
                        'cell_model': box.cell_model.cell_model,
                        'ignore': False
                    }
                    initial.append(my_initial)

            box_queryset_formset = BoxQuerysetFormSet(initial=initial, prefix='box_queryset')



            ## Forms
            ar['specific_electrolyte_formset'] = specific_electrolyte_formset
            ar['cathode_data_list'] = cathode_data_list
            ar['anode_data_list'] = anode_data_list
            ar['salt_component_formset'] = salt_component_formset
            ar['solvent_component_formset'] = solvent_component_formset
            ar['absent_component_formset'] = absent_component_formset
            ar['specify_electrolyte_form'] = specify_electrolyte_form
            ar['specify_dry_cell_form'] = specify_dry_cell_form
            ar['box_id_formset'] = box_id_formset
            ar['cell_id_range_formset'] = cell_id_range_formset

            ## Returned Lists
            ar['box_queryset_formset'] = box_queryset_formset
            ar['electrolyte_queryset_formset'] = electrolyte_queryset_formset
            ar['cell_list_formset'] = cell_list_formset

            if len(anode_data_list) > 0:
                ar['show_anode_data_list'] = True

            if len(electrolyte_queryset_formset) > 0:
                ar['show_electrolyte_queryset_formset'] = True

            if len(box_queryset_formset) > 0:
                ar['show_box_queryset_formset'] = True

            if len(cell_list_formset) != 0:
                ar['show_cell_list_formset'] = True


        if 'update_electrolyte_queryset_formset' in request.POST:

            specific_electrolyte_formset = SpecificElectrolyteFormSet(request.POST, prefix='specific_electrolyte')

            salt_component_formset = SaltComponentFormSet(request.POST, queryset=ElectrolyteComponent.objects.none(), prefix='salt')
            for form in salt_component_formset:
                form.fields['molecule'].queryset = ElectrolyteMolecule.objects.filter(can_be_salt=True)

            solvent_component_formset = SolventComponentFormSet(request.POST, queryset=ElectrolyteComponent.objects.none(), prefix='solvent')
            for form in salt_component_formset:
                form.fields['molecule'].queryset = ElectrolyteMolecule.objects.filter(can_be_salt=True)

            q = Q()

            if specific_electrolyte_formset.is_valid():

                for form in specific_electrolyte_formset:

                    if form.cleaned_data['electrolyte'] != '----':

                        shortstring = re.sub(r'\s-\s.+', '', form.cleaned_data['electrolyte'])
                        alias = re.sub(r'.+\s-\s', '', form.cleaned_data['electrolyte'])

                        q = q | Q(shortstring=shortstring) | Q(id=(Alias.objects.get(name=alias)).id)


            if specify_electrolyte_form.is_valid():

                if specify_electrolyte_form.cleaned_data['electrolyte_alias'] != '':

                    description_matches = []

                    for alias in Alias.objects.all():

                        if re.search(specify_electrolyte_form.cleaned_data['electrolyte_alias'], alias.name):
                            description_matches.append(alias)

                    if len(description_matches) == 0:
                        q = q & ~Q()

                    else:
                        for alias in Alias.objects.all():

                            if not alias in description_matches:
                                q = q & ~Q(id=alias.electrolyte.id)

            salt_instances = salt_component_formset.save(commit=False)
            solvent_instances = solvent_component_formset.save(commit=False)
            absent_instances = absent_component_formset.save(commit=False)


            if not specify_electrolyte_form.cleaned_data['use_ratio']:

                for instance in salt_instances:
                    if (not instance.molecule is None) and (not instance.molal is None):
                        electrolyte_ids = ElectrolyteComponent.objects.filter(molecule=instance.molecule,
                                                                              molal=float(instance.molal)).values_list('electrolyte_id',
                                                                                                            flat=True)
                        q = q & Q(id__in=electrolyte_ids)

                    elif not instance.molecule is None:
                        electrolyte_ids = ElectrolyteComponent.objects.filter(molecule=instance.molecule).values_list('electrolyte_id',
                                                                                                                      flat=True)
                        q = q & Q(id__in=electrolyte_ids)

                for instance in solvent_instances:
                    if (not instance.molecule is None) and (not instance.weight_percent is None):
                        electrolyte_ids = ElectrolyteComponent.objects.filter(molecule=instance.molecule,
                                                                              weight_percent=float(instance.weight_percent)).values_list(
                            'electrolyte_id', flat=True)
                        q = q & Q(id__in=electrolyte_ids)

                    elif not instance.molecule is None:
                        electrolyte_ids = ElectrolyteComponent.objects.filter(molecule=instance.molecule).values_list('electrolyte_id',
                                                                                                                      flat=True)
                        q = q & Q(id__in=electrolyte_ids)

                for instance in absent_instances:
                    if not instance.molecule is None:
                        electrolyte_ids = ElectrolyteComponent.objects.filter(molecule=instance.molecule).values_list('electrolyte_id',
                                                                                                                      flat=True)
                        q = q & ~Q(id__in=electrolyte_ids)


            else:


                ratio_dict = {}

                x = Q()

                for instance in solvent_instances:

                    if (instance.weight_percent != '') and (not instance.molecule is None):
                        ratio_dict[instance.molecule.name] = instance.weight_percent

                for key in ratio_dict.keys():
                    electrolyte_ids = ElectrolyteComponent.objects.filter(molecule=ElectrolyteMolecule.objects.get(name=key)).values_list(
                        'electrolyte_id', flat=True)
                    x = x & Q(id__in=electrolyte_ids)

                possible_electrolytes = Electrolyte.objects.filter(x)

                set_list = []
                sum1 = 0
                for molecule in ratio_dict.keys():
                    sum1 += float(ratio_dict[molecule])

                for molecule in ratio_dict.keys():
                    set_list.append(round((float(ratio_dict[molecule]) / sum1), 3))

                search_set = set(set_list)

                for electrolyte in possible_electrolytes:

                    electrolyte_sum = 0

                    electrolyte_set_list = []

                    for component in electrolyte.component_set.all():
                        if component.molecule.name in ratio_dict.keys():
                            electrolyte_sum += float(component.weight_percent)

                    for component in electrolyte.component_set.all():

                        if component.molecule.name in ratio_dict.keys():

                            electrolyte_set_list.append(
                                round((float(component.weight_percent / electrolyte_sum)), 3))

                            electrolyte_set = set(electrolyte_set_list)

                            if electrolyte_set == search_set:
                                q = q & Q(shortstring=electrolyte.shortstring)

            ## Cathode Expanding/Contracting Table

            cathode_data_list = []
            family_list = []
            ratio_list = []
            note_list = []

            for cell in DryCell.objects.all():

                cathode = Cathode.objects.get(dry_cell=cell)

                cathode_active_materials = cathode.cathode_active_materials

                if not cathode.cathode_specific_materials is None:

                    cathode_family = cathode.cathode_specific_materials.cathode_family

                    cathode_ratio = cathode.cathode_specific_materials

                    if not cathode_family in family_list:
                        cathode_data_list.append((cathode_family, []))

                        family_list.append(cathode_family)

                    if not cathode_ratio is None:

                        if not (str(cathode_ratio)) in ratio_list:

                            for fam in cathode_data_list:
                                if fam[0] == cathode_family:
                                    fam[1].append(((str(cathode_ratio)), []))

                            ratio_list.append(str(cathode_ratio))

                    if not cathode_active_materials.cathode_active_1_notes is None:

                        if not cathode_active_materials.cathode_active_1_notes in note_list:

                            for fam in cathode_data_list:

                                if fam[0] == cathode_family:

                                    for ratio in fam[1]:

                                        if ratio[0] == (str(cathode_ratio)):
                                            ratio[1].append(cathode_active_materials.cathode_active_1_notes)

                            note_list.append(cathode_active_materials.cathode_active_1_notes)

            anode_data_list = []
            family_list = []
            ratio_list = []
            note_list = []

            for cell in DryCell.objects.all():

                anode = Anode.objects.get(dry_cell=cell)

                anode_active_materials = anode.anode_active_materials

                if not anode.anode_specific_materials is None:

                    anode_family = anode.anode_specific_materials.anode_family

                    anode_ratio = anode.anode_specific_materials

                    if not anode_family in family_list:
                        anode_data_list.append((anode_family, []))

                        family_list.append(anode_family)

                    if not anode_ratio is None:

                        if not (str(anode_ratio)) in ratio_list:

                            for fam in anode_data_list:
                                if fam[0] == anode_family:
                                    fam[1].append(((str(anode_ratio)), []))

                            ratio_list.append(str(anode_ratio))

                    if not anode_active_materials.anode_active_1_notes is None:

                        if not anode_active_materials.anode_active_1_notes in note_list:

                            for fam in anode_data_list:

                                if fam[0] == anode_family:

                                    for ratio in fam[1]:

                                        if ratio[0] == (str(anode_ratio)):
                                            ratio[1].append(anode_active_materials.anode_active_1_notes)

                            note_list.append(anode_active_materials.anode_active_1_notes)

            electrolyte_queryset = Electrolyte.objects.filter(q)

            ignore_queryset = []
            good_electrolyte_queryset = []

            if electrolyte_queryset_formset.is_valid():

                for form in electrolyte_queryset_formset:

                    if form.is_valid():
                        if 'ignore' in form.cleaned_data:

                            if form.cleaned_data['ignore']:
                                ignore_queryset.append(
                                    Electrolyte.objects.get(shortstring=form.cleaned_data['shortstring']))

                            elif form.cleaned_data['shortstring'] != '':
                                good_electrolyte_queryset.append(
                                    Electrolyte.objects.get(shortstring=form.cleaned_data['shortstring']))

            initial = []

            if q == Q() and len(ignore_queryset) == 0:
                electrolyte_queryset = Electrolyte.objects.all()

            for electrolyte in electrolyte_queryset:

                if electrolyte in ignore_queryset:
                    my_initial = {
                        'shortstring': electrolyte.shortstring,
                        'ignore': True
                    }
                    initial.append(my_initial)

                else:
                    my_initial = {
                        'shortstring': electrolyte.shortstring,
                        'ignore': False
                    }
                    initial.append(my_initial)

            electrolyte_queryset_formset = ElectrolyteQuerysetFormSet(initial=initial, prefix='electrolyte_queryset')


            ## Forms
            ar['specific_electrolyte_formset'] = specific_electrolyte_formset
            ar['cathode_data_list'] = cathode_data_list
            ar['anode_data_list'] = anode_data_list
            ar['salt_component_formset'] = salt_component_formset
            ar['solvent_component_formset'] = solvent_component_formset
            ar['absent_component_formset'] = absent_component_formset
            ar['specify_electrolyte_form'] = specify_electrolyte_form
            ar['specify_dry_cell_form'] = specify_dry_cell_form
            ar['box_id_formset'] = box_id_formset
            ar['cell_id_range_formset'] = cell_id_range_formset

            ## Returned Lists
            ar['box_queryset_formset'] = box_queryset_formset
            ar['electrolyte_queryset_formset'] = electrolyte_queryset_formset
            ar['cell_list_formset'] = cell_list_formset

            if len(electrolyte_queryset_formset) > 0:
                ar['show_electrolyte_queryset_formset'] = True

            if len(box_queryset_formset) > 0:
                ar['show_box_queryset_formset'] = True

            if len(cell_list_formset) != 0:
                ar['show_cell_list_formset'] = True


            if len(anode_data_list) > 0:
                ar['show_anode_data_list'] = True


        if 'edit_cell_queryset_metadata' in request.POST:


            q = Q()

            shortstring_list = []

            if electrolyte_queryset_formset.is_valid():

                for form in electrolyte_queryset_formset:

                    if not form.cleaned_data['ignore']:
                        shortstring_list.append(form.cleaned_data['shortstring'])

                for electrolyte in Electrolyte.objects.all():

                    if (not electrolyte.shortstring in shortstring_list) and len(electrolyte_queryset_formset) > 0:
                        q = q & ~Q(electrolyte=electrolyte)

            box_id_list = []

            if box_queryset_formset.is_valid():

                for form in box_queryset_formset:

                    if not form.cleaned_data['ignore']:
                        box_id_list.append(form.cleaned_data['box_id'])

                for box in Box.objects.all():

                    if (not box.box_id_number in box_id_list) and len(box_queryset_formset) > 0:
                        q = q & ~Q(box=box)

            if cell_id_range_formset.is_valid():

                for form in cell_id_range_formset:

                    if 'range_max' in form.cleaned_data.keys() or 'range_min' in form.cleaned_data.keys():

                        if (not form.cleaned_data['range_min'] is None) and (
                        not form.cleaned_data['range_max'] is None):
                            q = q & Q(cell_id__range=(
                                form.cleaned_data['range_min'], form.cleaned_data['range_max']))

                        if (not form.cleaned_data['range_min'] is None) and (form.cleaned_data['range_max'] is None):
                            q = q & Q(cell_id=(
                                form.cleaned_data['range_min']
                            ))

                        if (form.cleaned_data['range_min'] is None) and (not form.cleaned_data['range_max'] is None):
                            q = q & Q(cell_id=(
                                form.cleaned_data['range_max']
                            ))

            cell_list = WetCell.objects.filter(q)

            initial = []

            for cell in cell_list:

                my_initial = {}

                my_initial['cell_id'] = cell.cell_id

                my_initial['electrolyte'] = cell.electrolyte

                if not cell.box is None:

                    my_initial['box_id'] = cell.box.box_id_number

                if not cell.box is None:
                    my_initial['box_id'] = cell.box.box_id_number

                if not cell.dry_cell is None:

                    dry_cell = cell.dry_cell

                    box = cell.box

                    electrolyte = cell.electrolyte

                    my_initial['cell_model'] = dry_cell.cell_model

                    my_initial['box'] = box

                    my_initial['electrolyte'] = electrolyte

                my_initial['exclude'] = True

                initial.append(my_initial)

            wet_cell_metadata_edit_formset = WetCellMetadataEditFormSet(initial=initial, prefix='wet_cell_metadata')

            ar['wet_cell_metadata_edit_formset'] = wet_cell_metadata_edit_formset


        if 'submit_wet_cell_edit' in request.POST:

            wet_cell_metadata_edit_formset = WetCellMetadataEditFormSet(request.POST, prefix='wet_cell_metadata')

            if wet_cell_metadata_edit_formset.is_valid():

                for form in wet_cell_metadata_edit_formset:

                    if not form.cleaned_data['exclude']:
                        wet_cell = WetCell.objects.get(cell_id=form.cleaned_data['cell_id'])

                        wet_cell.box = form.cleaned_data['box']

                        wet_cell.electrolyte = form.cleaned_data['electrolyte']

                        wet_cell.save()

            return render(request, 'ElectrolyteDatabase/main_page.html')

    else:

        specific_electrolyte_formset = SpecificElectrolyteFormSet(prefix='specific_electrolyte')


        salt_component_formset = SaltComponentFormSet(queryset=ElectrolyteComponent.objects.none(), prefix='salt')
        for form in salt_component_formset:
            form.fields['molecule'].queryset = ElectrolyteMolecule.objects.filter(can_be_salt=True)

        solvent_component_formset = SolventComponentFormSet(queryset=ElectrolyteComponent.objects.none(), prefix='solvent')
        for form in salt_component_formset:
            form.fields['molecule'].queryset = ElectrolyteMolecule.objects.filter(can_be_salt=True)

        absent_component_formset = AbsentComponentFormSet(queryset=ElectrolyteComponent.objects.none(), prefix='absent')

        ## Returned Formset Lists

        box_queryset_formset = BoxQuerysetFormSet(prefix='box_queryset')
        electrolyte_queryset_formset = ElectrolyteQuerysetFormSet(prefix='electrolyte_queryset')
        cell_list_formset = CellListFormSet(prefix='cell_list')

        specify_dry_cell_form = SpecifyDryCellForm(prefix='dry_cell')

        box_id_formset = BoxIDFormSet(prefix='box_id')
        cell_id_range_formset = CellIDRangeFormSet(prefix='cell_id')
        specify_electrolyte_form = SpecifyElectrolyteForm(prefix='electrolyte')


        ## Cathode Expanding/Contracting Table

        cathode_data_list = []
        family_list = []
        ratio_list = []
        note_list = []

        for cell in DryCell.objects.all():

            cathode = Cathode.objects.get(dry_cell=cell)

            cathode_active_materials = cathode.cathode_active_materials

            if not cathode.cathode_specific_materials is None:

                cathode_family = cathode.cathode_specific_materials.cathode_family

                cathode_ratio = cathode.cathode_specific_materials

                if not cathode_family in family_list:
                    cathode_data_list.append((cathode_family, []))

                    family_list.append(cathode_family)

                if not cathode_ratio is None:

                    if not (str(cathode_ratio)) in ratio_list:

                        for fam in cathode_data_list:
                            if fam[0] == cathode_family:
                                fam[1].append(((str(cathode_ratio)), []))

                        ratio_list.append(str(cathode_ratio))

                if not cathode_active_materials.cathode_active_1_notes is None:

                    if not cathode_active_materials.cathode_active_1_notes in note_list:

                        for fam in cathode_data_list:

                            if fam[0] == cathode_family:

                                for ratio in fam[1]:

                                    if ratio[0] == (str(cathode_ratio)):
                                        ratio[1].append(cathode_active_materials.cathode_active_1_notes)

                        note_list.append(cathode_active_materials.cathode_active_1_notes)

        anode_data_list = []
        family_list = []
        ratio_list = []
        note_list = []

        for cell in DryCell.objects.all():

            anode = Anode.objects.get(dry_cell=cell)

            anode_active_materials = anode.anode_active_materials

            if not anode.anode_specific_materials is None:

                anode_family = anode.anode_specific_materials.anode_family

                anode_ratio = anode.anode_specific_materials

                if not anode_family in family_list:
                    anode_data_list.append((anode_family, []))

                    family_list.append(anode_family)

                if not anode_ratio is None:

                    if not (str(anode_ratio)) in ratio_list:

                        for fam in anode_data_list:
                            if fam[0] == anode_family:
                                fam[1].append(((str(anode_ratio)), []))

                        ratio_list.append(str(anode_ratio))

                if not anode_active_materials.anode_active_1_notes is None:

                    if not anode_active_materials.anode_active_1_notes in note_list:

                        for fam in anode_data_list:

                            if fam[0] == anode_family:

                                for ratio in fam[1]:

                                    if ratio[0] == (str(anode_ratio)):
                                        ratio[1].append(anode_active_materials.anode_active_1_notes)

                        note_list.append(anode_active_materials.anode_active_1_notes)


        ## Forms
        ar['cathode_data_list'] = cathode_data_list
        ar['anode_data_list'] = anode_data_list
        ar['salt_component_formset'] = salt_component_formset
        ar['solvent_component_formset'] = solvent_component_formset
        ar['absent_component_formset'] = absent_component_formset
        ar['specify_electrolyte_form'] = specify_electrolyte_form
        ar['specify_dry_cell_form'] = specify_dry_cell_form
        ar['box_id_formset'] = box_id_formset
        ar['cell_id_range_formset'] = cell_id_range_formset
        ar['specific_electrolyte_formset'] = specific_electrolyte_formset

        ## Returned Lists
        ar['box_queryset_formset'] = box_queryset_formset
        ar['electrolyte_queryset_formset'] = electrolyte_queryset_formset
        ar['cell_list_formset'] = cell_list_formset

        if len(anode_data_list) > 0:
            ar['show_anode_data_list'] = True




    return render(request, 'ElectrolyteDatabase/specify_and_edit_wet_cells.html', ar)




def main_page(request):

    return render(request, 'ElectrolyteDatabase/main_page.html')




