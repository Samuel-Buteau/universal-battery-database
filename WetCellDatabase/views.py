from django.shortcuts import render, render_to_response
from django.forms import modelformset_factory, formset_factory
from django.db.models import Q, F, Func, Count,Exists, OuterRef
from .forms import *
from .models import *
from django import forms
import re


#TODO(sam): condense the code!!!
'''
DONE - remove all the fields that I don't currently care about.
DONE - good name generation
- good uniqueness check.
- streamline the various definitions into much simpler and unique flows.
- make the processing specific to machine learning optional.
- bake a tensor instead of having to use the database all the time.

'''

#TODO(sam): name proposal
'''
basically, if proprietary, name is specified. 
Otherwise, name is generated based on data itself.
make sure the *visible* data is unique. 
(i.e., given the set of visible data, there should not exist another element 
which coincides with the candidate on this set.)

In order not to drown in tons of text, allow opt-out of various fields.

SIZE=
SINGLE/POLY
P=
T=
NAT/ART
CORE-SHELL
(Notes)
COAT=

Li1 Ni.5 Mn.3 Co.2 O2

'''

def define_page(request):
    ActiveMaterialCompositionFormset = formset_factory(
        ElectrodeMaterialStochiometryForm,
        extra=10
    )

    ElectrolyteCompositionFormset = formset_factory(
        ElectrolyteCompositionForm,
        extra=10
    )

    ElectrodeCompositionFormset = formset_factory(
        ElectrodeCompositionForm,
        extra=10
    )


    SeparatorCompositionFormset = formset_factory(
        SeparatorCompositionForm,
        extra=10
    )


    active_material_composition_formset = ActiveMaterialCompositionFormset(prefix='active-material-composition-formset')
    electrolyte_composition_formset = ElectrolyteCompositionFormset(prefix='electrolyte-composition-formset')
    electrode_composition_formset = ElectrodeCompositionFormset(prefix='electrode-composition-formset')
    separator_composition_formset = SeparatorCompositionFormset(prefix='separator-composition-formset')

    ar = {}
    ar['active_material_composition_formset'] = active_material_composition_formset
    ar['electrolyte_composition_formset'] = electrolyte_composition_formset
    ar['electrode_composition_formset'] = electrode_composition_formset
    ar['separator_composition_formset'] = separator_composition_formset

    ar['define_molecule_form'] = ElectrolyteMoleculeForm(prefix='electrolyte-molecule-form')
    ar['define_molecule_lot_form'] = ElectrolyteMoleculeLotForm(prefix='electrolyte-molecule-lot-form')

    ar['define_coating_form'] = CoatingForm(prefix='coating-form')
    ar['define_coating_lot_form'] = CoatingLotForm(prefix='coating-lot-form')

    ar['define_inactive_form'] = ElectrodeInactiveForm(prefix='electrode-inactive-form')
    ar['define_inactive_lot_form'] = ElectrodeInactiveLotForm(prefix='electrode-inactive-lot-form')

    ar['define_active_material_form'] = ElectrodeActiveMaterialForm(prefix='electrode-active-material-form')
    ar['define_active_material_lot_form'] = ElectrodeActiveMaterialLotForm(prefix='electrode-active-material-lot-form')


    ar['define_separator_material_form'] = SeparatorMaterialForm(prefix='separator-material-form')
    ar['define_separator_material_lot_form'] = SeparatorMaterialLotForm(prefix='separator-material-lot-form')

    ar['define_electrolyte_form'] = ElectrolyteForm(prefix='electrolyte-form')
    ar['define_electrolyte_lot_form'] = ElectrolyteLotForm(prefix='electrolyte-lot-form')

    ar['define_electrode_form'] = ElectrodeForm(prefix='electrode-form')
    ar['define_electrode_lot_form'] = ElectrodeLotForm(prefix='electrode-lot-form')
    ar['define_electrode_geometry_form'] = ElectrodeGeometryForm(prefix='electrode-geometry-form')

    ar['define_separator_form'] = SeparatorForm(prefix='separator-form')
    ar['define_separator_lot_form'] = SeparatorLotForm(prefix='separator-lot-form')
    ar['define_separator_geometry_form'] = SeparatorGeometryForm(prefix='separator-geometry-form')

    ar['define_dry_cell_form'] = DryCellForm()
    ar['define_dry_cell_lot_form'] = DryCellLotForm()
    ar['define_dry_cell_geometry_form'] = DryCellGeometryForm()

    ar['define_wet_cell_form'] = WetCellForm()

    def define_simple(post, content=None):
        if content=='electrolyte':
            define_electrolyte_form = ElectrolyteForm(request.POST, prefix='electrolyte-form')
            if define_electrolyte_form.is_valid():
                ar['define_electrolyte_form'] = define_electrolyte_form

                if 'proprietary' in define_electrolyte_form.cleaned_data.keys() and \
                        define_electrolyte_form.cleaned_data['proprietary']:
                    if 'name' in define_electrolyte_form.cleaned_data.keys() and \
                            define_electrolyte_form.cleaned_data['name'] is not None:
                        print('This is a proprietary electrolyte called: ',
                              define_electrolyte_form.cleaned_data['name'])
                        name = define_electrolyte_form.cleaned_data['name']
                        my_electrolyte, _ = Composite.objects.get_or_create(
                                composite_type=ELECTROLYTE,
                                proprietary=True,
                                name=name
                            )

                        return True, my_electrolyte
                    else:
                        print('name is missing.')
                else:
                    print('This is an electrolyte defined by components.')
                    electrolyte_composition_formset = ElectrolyteCompositionFormset(request.POST,
                                                                                    prefix='electrolyte-composition-formset')
                    print(electrolyte_composition_formset)
                    molecules = []
                    molecules_lot = []
                    for form in electrolyte_composition_formset:

                        validation_step = form.is_valid()
                        if validation_step:
                            print(form.cleaned_data)
                            if not 'molecule' in form.cleaned_data.keys() or not 'molecule_lot' in form.cleaned_data.keys():
                                continue
                            if form.cleaned_data['ratio'] is  None or form.cleaned_data['ratio'] <= 0:
                                continue
                            if form.cleaned_data['molecule'] is not None:
                                molecules.append(
                                    {
                                        'molecule': form.cleaned_data['molecule'],
                                        'ratio': form.cleaned_data['ratio']
                                    }
                                )
                            elif form.cleaned_data['molecule_lot'] is not None:
                                molecules_lot.append(
                                    {
                                        'molecule_lot': form.cleaned_data['molecule_lot'],
                                        'ratio': form.cleaned_data['ratio']
                                    }
                                )

                    print(molecules)
                    print(molecules_lot)

                    # check if there is a problem with the molecules/molecules_lot
                    all_ids = list(map(lambda x: x['molecule'].id, molecules)) + list(
                        map(lambda x: x['molecule_lot'].component.id, molecules_lot))
                    if len(set(all_ids)) == len(all_ids):
                        # normalize things.
                        total_solvent = (sum(map(lambda x: x['ratio'],
                                                 filter(lambda x: x['molecule'].component_type == SOLVENT, molecules)),
                                             0.) +
                                         sum(map(lambda x: x['ratio'],
                                                 filter(lambda x: x['molecule_lot'].component.component_type == SOLVENT,
                                                        molecules_lot)), 0.))

                        total_additive = (sum(map(lambda x: x['ratio'],
                                                  filter(lambda x: x['molecule'].component_type == ADDITIVE,
                                                         molecules)),
                                              0.) +
                                          sum(map(lambda x: x['ratio'],
                                                  filter(
                                                      lambda x: x['molecule_lot'].component.component_type == ADDITIVE,
                                                      molecules_lot)), 0.))

                        print('total solvent: ', total_solvent)
                        print('total additive: ', total_additive)
                        if total_additive < 100. and total_solvent > 0.:
                            # create or get each RatioComponent.
                            my_ratio_components = []
                            for ms, kind in [(molecules, 'molecule'), (molecules_lot, 'molecule_lot')]:
                                for molecule in ms:
                                    if kind == 'molecule':
                                        actual_molecule = molecule['molecule']
                                    elif kind == 'molecule_lot':
                                        actual_molecule = molecule['molecule_lot'].component

                                    if actual_molecule.component_type == SOLVENT:
                                        ratio = molecule['ratio'] * 100. / total_solvent
                                        tolerance = 0.25
                                    elif actual_molecule.component_type == ADDITIVE:
                                        ratio = molecule['ratio']
                                        tolerance = 0.01
                                    elif actual_molecule.component_type == SALT:
                                        ratio = molecule['ratio']
                                        tolerance = 0.1

                                    if kind == 'molecule':
                                        comp_lot, _ = ComponentLot.objects.get_or_create(
                                            lot_info=None,
                                            component=molecule['molecule']
                                        )
                                    elif kind == 'molecule_lot':
                                        comp_lot = molecule['molecule_lot']

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

                            # For every component, filter electrolytes which don't have it.

                            query = Composite.objects.annotate(
                                count_components=Count('components')
                            ).filter(count_components=len(my_ratio_components)).annotate(
                                count_valid_components=Count('components', filter=Q(components__in=my_ratio_components))
                            ).filter(count_valid_components=len(my_ratio_components))
                            if not query.exists():
                                my_electrolyte = Composite(composite_type=ELECTROLYTE)
                                my_electrolyte.save()
                                my_electrolyte.components.set(my_ratio_components)
                                print('my electrolyte: ', my_electrolyte)
                            elif query.count() ==1:
                                my_electrolyte = query[0]
                            else:
                                return False, None
                            ar['electrolyte_composition_formset'] = electrolyte_composition_formset
                            return True, my_electrolyte
            return False, None


        if content=='electrode':
            define_electrode_form = ElectrodeForm(request.POST, prefix='electrode-form')
            define_electrode_geometry_form = ElectrodeGeometryForm(request.POST, prefix='electrode-geometry-form')
            if define_electrode_form.is_valid():
                ar['define_electrode_form'] = define_electrode_form
                if define_electrode_geometry_form.is_valid():
                    ar['define_electrode_geometry_form'] = define_electrode_geometry_form

                    if 'name' in define_electrode_form.cleaned_data.keys() and \
                            define_electrode_form.cleaned_data['name'] is not None:
                        name = define_electrode_form.cleaned_data['name']

                        if 'proprietary' in define_electrode_form.cleaned_data.keys() and \
                                define_electrode_form.cleaned_data['proprietary']:

                            if not Composite.objects.filter(
                                    composite_type=define_electrode_form.cleaned_data['composite_type'],
                                    proprietary=True,
                                    name=name
                                ).exists():

                                electrode_geometry = ElectrodeGeometry.objects.create(
                                    loading= define_electrode_geometry_form.cleaned_data['loading'],
                                    porosity=define_electrode_geometry_form.cleaned_data['porosity'],
                                    thickness=define_electrode_geometry_form.cleaned_data['thickness'],
                                    length_single_side=define_electrode_geometry_form.cleaned_data['length_single_side'],
                                    length_double_side=define_electrode_geometry_form.cleaned_data['length_double_side'],
                                    width=define_electrode_geometry_form.cleaned_data['width'],
                                    tab_position_from_core=define_electrode_geometry_form.cleaned_data['tab_position_from_core'],
                                    foil_thickness=define_electrode_geometry_form.cleaned_data['foil_thickness'],
                                )
                                my_electrode = Composite.objects.create(
                                        composite_type=define_electrode_form.cleaned_data['composite_type'],
                                        proprietary=True,
                                        name=name,
                                        electrode_geometry=electrode_geometry,
                                    )
                            else:
                                my_electrode = Composite.objects.filter(
                                    composite_type=define_electrode_form.cleaned_data['composite_type'],
                                    proprietary=True,
                                    name=name
                                )[0]

                            return True, my_electrode

                        else:
                            electrode_composition_formset = ElectrodeCompositionFormset(request.POST,
                                                                                            prefix='electrode-composition-formset')
                            print(electrode_composition_formset)
                            composite_type = define_electrode_form.cleaned_data['composite_type']
                            materials = []
                            materials_lot = []

                            for form in electrode_composition_formset:

                                validation_step = form.is_valid()
                                if validation_step:
                                    print(form.cleaned_data)
                                    if not 'material' in form.cleaned_data.keys() or not 'material_lot' in form.cleaned_data.keys():
                                        continue
                                    if form.cleaned_data['ratio'] is  None or form.cleaned_data['ratio'] <= 0:
                                        continue

                                    if form.cleaned_data['material'] is not None and form.cleaned_data['material'].composite_type==composite_type:
                                        materials.append(
                                            {
                                                'material': form.cleaned_data['material'],
                                                'ratio': form.cleaned_data['ratio']
                                            }
                                        )
                                    elif form.cleaned_data['material_lot'] is not None and form.cleaned_data['material_lot'].composite.composite_type==composite_type:
                                        materials_lot.append(
                                            {
                                                'material_lot': form.cleaned_data['material_lot'],
                                                'ratio': form.cleaned_data['ratio']
                                            }
                                        )

                            print(materials)
                            print(materials_lot)

                            # check if there is a problem with the materials/materials_lot
                            all_ids = list(map(lambda x: x['material'].id, materials)) + list(
                                map(lambda x: x['material_lot'].component.id, materials_lot))
                            if len(set(all_ids)) == len(all_ids):
                                # normalize things.
                                total_active = (sum(map(lambda x: x['ratio'],
                                                         filter(lambda x: x['material'].component_type == ACTIVE_MATERIAL, materials)),
                                                     0.) +
                                                 sum(map(lambda x: x['ratio'],
                                                         filter(lambda x: x['material_lot'].component.component_type == ACTIVE_MATERIAL,
                                                                materials_lot)), 0.))

                                total_inactive = (sum(map(lambda x: x['ratio'],
                                                          filter(lambda x: x['material'].component_type != ACTIVE_MATERIAL,
                                                                 materials)),
                                                      0.) +
                                                  sum(map(lambda x: x['ratio'],
                                                          filter(
                                                              lambda x: x['material_lot'].component.component_type != ACTIVE_MATERIAL,
                                                              materials_lot)), 0.))

                                print('total solvent: ', total_active)
                                print('total additive: ', total_inactive)
                                if total_inactive < 100. and total_active > 0.:
                                    # create or get each RatioComponent.
                                    my_ratio_components = []
                                    for ms, kind in [(materials, 'material'), (materials_lot, 'material_lot')]:
                                        for material in ms:
                                            if kind == 'material':
                                                actual_material = material['material']
                                            elif kind == 'material_lot':
                                                actual_material = material['material_lot'].component

                                            if actual_material.component_type == ACTIVE_MATERIAL:
                                                ratio = material['ratio'] * 100. / total_active
                                                tolerance = 0.25
                                            elif actual_material.component_type != ACTIVE_MATERIAL:
                                                ratio = material['ratio']
                                                tolerance = 0.01


                                            if kind == 'material':
                                                comp_lot, _ = ComponentLot.objects.get_or_create(
                                                    lot_info=None,
                                                    component=material['material']
                                                )
                                            elif kind == 'material_lot':
                                                comp_lot = material['material_lot']

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

                                    # For every component, filter electrodes which don't have it.
                                    if not Composite.objects.filter(
                                            composite_type=define_electrode_form.cleaned_data['composite_type'],
                                            proprietary=False,
                                            name=name
                                    ).exists():

                                        electrode_geometry = ElectrodeGeometry.objects.create(
                                            loading=define_electrode_geometry_form.cleaned_data['loading'],
                                            porosity=define_electrode_geometry_form.cleaned_data['porosity'],
                                            thickness=define_electrode_geometry_form.cleaned_data['thickness'],
                                            length_single_side=define_electrode_geometry_form.cleaned_data[
                                                'length_single_side'],
                                            length_double_side=define_electrode_geometry_form.cleaned_data[
                                                'length_double_side'],
                                            width=define_electrode_geometry_form.cleaned_data['width'],
                                            tab_position_from_core=define_electrode_geometry_form.cleaned_data[
                                                'tab_position_from_core'],
                                            foil_thickness=define_electrode_geometry_form.cleaned_data[
                                                'foil_thickness'],
                                        )
                                        my_electrode = Composite.objects.create(
                                            composite_type=define_electrode_form.cleaned_data['composite_type'],
                                            proprietary=False,
                                            name=name,
                                            electrode_geometry=electrode_geometry,
                                        )
                                        my_electrode.components.set(my_ratio_components)

                                    else:
                                        my_electrode = Composite.objects.filter(
                                            composite_type=define_electrode_form.cleaned_data['composite_type'],
                                            proprietary=False,
                                            name=name
                                        )[0]


                                    ar['electrode_composition_formset'] = electrode_composition_formset
                                    return True, my_electrode
            return False, None

        if content=='active_material':
            define_active_material_form = ElectrodeActiveMaterialForm(request.POST, prefix='electrode-active-material-form')
            if define_active_material_form.is_valid():
                ar['define_active_material_form'] = define_active_material_form

                if 'name' in define_active_material_form.cleaned_data.keys() and \
                        define_active_material_form.cleaned_data['name'] is not None:
                    name = define_active_material_form.cleaned_data['name']
                    print(name)
                    if 'proprietary' in define_active_material_form.cleaned_data.keys() and \
                            define_active_material_form.cleaned_data['proprietary']:
                        my_active_material, _ = Component.objects.get_or_create(
                            composite_type=define_active_material_form.cleaned_data['composite_type'],
                            component_type=ACTIVE_MATERIAL,
                            proprietary=True,
                            name=name,
                        )
                        print(my_active_material)
                        return True, my_active_material

                    else:
                        active_material_composition_formset = ActiveMaterialCompositionFormset(request.POST,
                                                                                        prefix='active-material-composition-formset')
                        print(active_material_composition_formset)
                        atoms = []

                        for form in active_material_composition_formset:

                            validation_step = form.is_valid()
                            if validation_step:
                                print(form.cleaned_data)
                                if not 'atom' in form.cleaned_data.keys():
                                    continue
                                if form.cleaned_data['stochiometry'] is None or form.cleaned_data['stochiometry'] <= 0:
                                    continue
                                if form.cleaned_data['atom'] is not None :
                                    atoms.append(
                                        {
                                            'atom': form.cleaned_data['atom'],
                                            'stochiometry': form.cleaned_data['stochiometry']
                                        }
                                    )
                        print(atoms)
                        # check if there is a problem with the atoms/atoms_lot
                        all_ids = list(map(lambda x: x['atom'], atoms))
                        if len(set(all_ids)) == len(all_ids):
                            # normalize things.

                            # create or get each RatioComponent.
                            tolerance = 0.01
                            my_stochiometry_components = []
                            for atom in atoms:
                                actual_atom = atom['atom']
                                stochiometry = atom['stochiometry']

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

                                my_stochiometry_components.append(selected_stochiometry_component)

                            # For every component, filter active_atoms which don't have it.
                            if not Component.objects.filter(
                                    composite_type=define_active_material_form.cleaned_data['composite_type'],
                                    proprietary=True,
                                    name=name,
                                    component_type=ACTIVE_MATERIAL
                            ).exists():
                                coating_lot = get_lot(
                                    define_active_material_form.cleaned_data['coating'],
                                    define_active_material_form.cleaned_data['coating_lot'],
                                    type='coating'
                                )
                                my_active_material = Component.objects.create(
                                    composite_type=define_active_material_form.cleaned_data['composite_type'],
                                    proprietary=True,
                                    name=name,
                                    component_type=ACTIVE_MATERIAL,
                                    particle_size=define_active_material_form.cleaned_data['particle_size'],
                                    single_crystal=define_active_material_form.cleaned_data['single_crystal'],
                                    turbostratic_misalignment=define_active_material_form.cleaned_data['turbostratic_misalignment'],
                                    preparation_temperature=define_active_material_form.cleaned_data['preparation_temperature'],
                                    natural=define_active_material_form.cleaned_data['natural'],
                                    core_shell=define_active_material_form.cleaned_data['core_shell'],
                                    notes=define_active_material_form.cleaned_data['notes'],
                                    coating_lot=coating_lot,
                                )
                                my_active_material.stochiometry.set(my_stochiometry_components)

                            else:
                                my_active_material = Component.objects.filter(
                                    composite_type=define_active_material_form.cleaned_data['composite_type'],
                                    proprietary=True,
                                    name=name,
                                    component_type=ACTIVE_MATERIAL,
                                )[0]


                            ar['active_material_composition_formset'] = active_material_composition_formset
                            return True, my_active_material
            return False, None

        if content=='separator':
            define_separator_form = SeparatorForm(request.POST, prefix='separator-form')
            define_separator_geometry_form = SeparatorGeometryForm(request.POST, prefix='separator-geometry-form')
            if define_separator_form.is_valid():
                ar['define_separator_form'] = define_separator_form
                if define_separator_geometry_form.is_valid():
                    ar['define_separator_geometry_form'] = define_separator_geometry_form

                    if 'name' in define_separator_form.cleaned_data.keys() and \
                            define_separator_form.cleaned_data['name'] is not None:
                        name = define_separator_form.cleaned_data['name']

                        if 'proprietary' in define_separator_form.cleaned_data.keys() and \
                                define_separator_form.cleaned_data['proprietary']:

                            if not Composite.objects.filter(
                                    composite_type=SEPARATOR,
                                    proprietary=True,
                                    name=name
                                ).exists():

                                separator_geometry = SeparatorGeometry.objects.create(
                                    base_thickness= define_separator_geometry_form.cleaned_data['base_thickness'],
                                    width=define_separator_geometry_form.cleaned_data['width'],
                                    overhang_in_core=define_separator_geometry_form.cleaned_data['overhang_in_core'],
                                )
                                my_separator = Composite.objects.create(
                                        composite_type=SEPARATOR,
                                        proprietary=True,
                                        name=name,
                                        separator_geometry=separator_geometry,
                                    )
                            else:
                                my_separator = Composite.objects.filter(
                                    composite_type=SEPARATOR,
                                    proprietary=True,
                                    name=name
                                )[0]

                            return True, my_separator

                        else:
                            separator_composition_formset = SeparatorCompositionFormset(request.POST,
                                                                                            prefix='separator-composition-formset')
                            print(separator_composition_formset)
                            materials = []
                            materials_lot = []
                            composite_type = SEPARATOR
                            for form in separator_composition_formset:

                                validation_step = form.is_valid()
                                if validation_step:
                                    print(form.cleaned_data)
                                    if not 'material' in form.cleaned_data.keys() or not 'material_lot' in form.cleaned_data.keys():
                                        continue
                                    if form.cleaned_data['ratio'] is  None or form.cleaned_data['ratio'] <= 0:
                                        continue

                                    if form.cleaned_data['material'] is not None and form.cleaned_data['material'].composite_type==composite_type:
                                        materials.append(
                                            {
                                                'material': form.cleaned_data['material'],
                                                'ratio': form.cleaned_data['ratio']
                                            }
                                        )
                                    elif form.cleaned_data['material_lot'] is not None and form.cleaned_data['material_lot'].composite.composite_type==composite_type:
                                        materials_lot.append(
                                            {
                                                'material_lot': form.cleaned_data['material_lot'],
                                                'ratio': form.cleaned_data['ratio']
                                            }
                                        )

                            print(materials)
                            print(materials_lot)

                            # check if there is a problem with the materials/materials_lot
                            all_ids = list(map(lambda x: x['material'].id, materials)) + list(
                                map(lambda x: x['material_lot'].component.id, materials_lot))
                            if len(set(all_ids)) == len(all_ids):
                                # normalize things.
                                total = (sum(map(lambda x: x['ratio'],
                                                         filter(lambda x: x['material'].component_type == SEPARATOR_MATERIAL, materials)),
                                                     0.) +
                                                 sum(map(lambda x: x['ratio'],
                                                         filter(lambda x: x['material_lot'].component.component_type == SEPARATOR_MATERIAL,
                                                                materials_lot)), 0.))

                                if total > 0.:
                                    # create or get each RatioComponent.
                                    my_ratio_components = []
                                    for ms, kind in [(materials, 'material'), (materials_lot, 'material_lot')]:
                                        for material in ms:

                                            ratio = material['ratio'] * 100. / total
                                            tolerance = 0.25


                                            if kind == 'material':
                                                comp_lot, _ = ComponentLot.objects.get_or_create(
                                                    lot_info=None,
                                                    component=material['material']
                                                )
                                            elif kind == 'material_lot':
                                                comp_lot = material['material_lot']

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

                                    # For every component, filter separators which don't have it.
                                    if not Composite.objects.filter(
                                            composite_type=SEPARATOR,
                                            proprietary=False,
                                            name=name
                                    ).exists():

                                        separator_geometry = SeparatorGeometry.objects.create(
                                            base_thickness=define_separator_geometry_form.cleaned_data[
                                                'base_thickness'],
                                            width=define_separator_geometry_form.cleaned_data['width'],
                                            overhang_in_core=define_separator_geometry_form.cleaned_data[
                                                'overhang_in_core'],
                                        )
                                        my_separator = Composite.objects.create(
                                            composite_type=SEPARATOR,
                                            proprietary=False,
                                            name=name,
                                            separator_geometry=separator_geometry,
                                        )
                                        my_separator.components.set(my_ratio_components)

                                    else:
                                        my_separator = Composite.objects.filter(
                                            composite_type=SEPARATOR,
                                            proprietary=False,
                                            name=name
                                        )[0]

                                    ar['separator_composition_formset'] = separator_composition_formset
                                    return True, my_separator
            return False, None

        if content=='molecule':
            simple_form = ElectrolyteMoleculeForm(post, prefix='electrolyte-molecule-form')
        elif content=='coating':
            simple_form = CoatingForm(post, prefix='coating-form')
        elif content=='inactive':
            simple_form = ElectrodeInactiveForm(request.POST, prefix='electrode-inactive-form')
        elif content == 'separator_material':
            simple_form = SeparatorMaterialForm(request.POST, prefix='separator-material-form')

        simple_form_string = 'define_{}_form'.format(content)

        if simple_form.is_valid():
            ar[simple_form_string] = simple_form
            if simple_form.cleaned_data['name'] is not None:
                if content=='molecule':
                    my_content, _ = Component.objects.get_or_create(
                        name=simple_form.cleaned_data['name'],
                        composite_type=ELECTROLYTE,
                        component_type=simple_form.cleaned_data['component_type'],
                        defaults = {
                            'smiles':simple_form.cleaned_data['smiles'],
                            'proprietary':simple_form.cleaned_data['proprietary'],
                     }
                    )
                elif content=='coating':
                    my_content, _ = Coating.objects.get_or_create(
                        name=simple_form.cleaned_data['name'],
                        defaults={
                            'description':simple_form.cleaned_data['description'],
                            'proprietary':simple_form.cleaned_data['proprietary'],
                        }
                    )
                elif content=='inactive':
                    coating_lot = get_lot(simple_form.cleaned_data['coating'], simple_form.cleaned_data['coating_lot'], type='coating')
                    my_content, _ = Component.objects.get_or_create(
                        name=simple_form.cleaned_data['name'],
                        component_type=simple_form.cleaned_data['component_type'],
                        composite_type= simple_form.cleaned_data['composite_type'],
                        defaults={
                            'smiles': simple_form.cleaned_data['smiles'],
                            'proprietary': simple_form.cleaned_data['proprietary'],
                            'particle_size': simple_form.cleaned_data['particle_size'],
                            'preparation_temperature': simple_form.cleaned_data['preparation_temperature'],
                            'notes': simple_form.cleaned_data['notes'],
                            'coating_lot': coating_lot,
                        }
                    )
                elif content == 'separator_material':
                    coating_lot = get_lot(simple_form.cleaned_data['coating'],
                                                  simple_form.cleaned_data['coating_lot'], type='coating')
                    my_content, _ = Component.objects.get_or_create(
                        name=simple_form.cleaned_data['name'],
                        component_type=SEPARATOR_MATERIAL,
                        composite_type=SEPARATOR,
                        defaults={
                            'smiles': simple_form.cleaned_data['smiles'],
                            'proprietary': simple_form.cleaned_data['proprietary'],
                            'particle_size': simple_form.cleaned_data['particle_size'],
                            'preparation_temperature': simple_form.cleaned_data['preparation_temperature'],
                            'notes': simple_form.cleaned_data['notes'],
                            'coating_lot': coating_lot,
                        }
                    )


                print(my_content)
                return True, my_content
        print('failed ,', simple_form.cleaned_data)
        return False, None


    def define_lot(post, content=None):
        define_lot_form_string = 'define_{}_lot_form'.format(content)
        predefined_string = 'predefined_{}'.format(content)
        if content == 'molecule':
            define_lot_form = ElectrolyteMoleculeLotForm(
                post,
                prefix='electrolyte-molecule-lot-form'
            )
        elif content == 'coating':
            define_lot_form = CoatingLotForm(
                post,
                prefix='coating-lot-form'
            )
        elif content == 'inactive':
            define_lot_form = ElectrodeInactiveLotForm(
                post,
                prefix='electrode-inactive-lot-form'
            )


        elif content == 'separator_material':
            define_lot_form = SeparatorMaterialLotForm(
                post,
                prefix='separator-material-lot-form'
            )
        elif content == 'electrolyte':
            define_lot_form = ElectrolyteLotForm(
                post,
                prefix='electrolyte-lot-form'
            )
        elif content == 'electrode':
            define_lot_form = ElectrodeLotForm(
                post,
                prefix='electrode-lot-form'
            )
        elif content == 'active_material':
            define_lot_form = ElectrodeActiveMaterialLotForm(
                post,
                prefix='electrode-active-material-lot-form'
            )
        elif content == 'separator':
            define_lot_form = SeparatorLotForm(
                post,
                prefix='separator-lot-form'
            )


        if define_lot_form.is_valid():
            ar[define_lot_form_string] = define_lot_form
            if define_lot_form.cleaned_data[predefined_string] is not None:
                my_content = define_lot_form.cleaned_data[predefined_string]
                valid_content = True
            else:
                valid_content, my_content = define_simple(post, content=content)

            if define_lot_form.cleaned_data['lot_name'] is not None and valid_content:
                if content == 'molecule':
                    base_query = ComponentLot.objects.filter(component__composite_type=ELECTROLYTE)
                elif content == 'coating':
                    base_query = CoatingLot.objects
                elif content == 'inactive':
                    base_query = ComponentLot.objects.filter(Q(composite__composite_type=CONDUCTIVE_ADDITIVE)|Q(composite__composite_type=BINDER))
                elif content == 'separator_material':
                    base_query = ComponentLot.objects.filter(component__component_type=SEPARATOR_MATERIAL,
                                                             component__composite_type=SEPARATOR)
                elif content == 'electrolyte':
                    base_query = CompositeLot.objects.filter(composite__composite_type=ELECTROLYTE)

                elif content == 'electrode':
                    base_query = CompositeLot.objects.filter(Q(composite__composite_type=ANODE)|Q(composite__composite_type=CATHODE))
                elif content == 'active_material':
                    base_query = ComponentLot.objects.filter(component__component_type=ACTIVE_MATERIAL)
                elif content == 'separator':
                    base_query = CompositeLot.objects.filter(composite__composite_type=SEPARATOR)

                if not base_query.exclude(lot_info=None).filter(
                        lot_info__lot_name=define_lot_form.cleaned_data['lot_name']).exists():
                    print('reached the end')
                    lot_info = LotInfo(
                        lot_name=define_lot_form.cleaned_data['lot_name'],
                        creation_date=define_lot_form.cleaned_data['creation_date'],
                        creator=define_lot_form.cleaned_data['creator'],
                        vendor=define_lot_form.cleaned_data['vendor'],
                    )
                    lot_info.save()
                    if content == 'molecule':
                        ComponentLot.objects.create(
                            component=my_content,
                            lot_info=lot_info
                        )
                    elif content == 'coating':
                        CoatingLot.objects.create(
                            coating=my_content,
                            lot_info=lot_info
                        )
                    if content == 'inactive':
                        ComponentLot.objects.create(
                            component=my_content,
                            lot_info=lot_info
                        )
                    if content == 'separator_material':
                        ComponentLot.objects.create(
                            component=my_content,
                            lot_info=lot_info
                        )

                    if content == 'electrolyte':
                        CompositeLot.objects.create(
                            composite=my_content,
                            lot_info=lot_info
                        )
                    if content == 'electrode':
                        CompositeLot.objects.create(
                            composite=my_content,
                            lot_info=lot_info
                        )
                    if content == 'active_material':
                        ComponentLot.objects.create(
                            component=my_content,
                            lot_info=lot_info
                        )
                    if content == 'separator':
                        CompositeLot.objects.create(
                            composite=my_content,
                            lot_info=lot_info
                        )

    if request.method == 'POST':
        if ('define_molecule' in request.POST) or ('define_molecule_lot' in request.POST):
            if 'define_molecule' in request.POST:
                define_simple(request.POST, content='molecule')
            if 'define_molecule_lot' in request.POST:
                define_lot(request.POST, content='molecule')

        elif ('define_coating' in request.POST) or ('define_coating_lot' in request.POST):
            if 'define_coating' in request.POST:
                define_simple(request.POST, content='coating')
            if 'define_coating_lot' in request.POST:
                define_lot(request.POST, content='coating')

        elif ('define_inactive' in request.POST) or ('define_inactive_lot' in request.POST):
            if 'define_inactive' in request.POST:
                define_simple(request.POST, content='inactive')
            if 'define_inactive_lot' in request.POST:
                define_lot(request.POST, content='inactive')

        elif ('define_electrolyte' in request.POST) or ('define_electrolyte_lot' in request.POST):
            if 'define_electrolyte' in request.POST:
                define_simple(request.POST, content='electrolyte')
            if 'define_electrolyte_lot' in request.POST:
                define_lot(request.POST, content='electrolyte')

        elif ('define_active_material' in request.POST) or ('define_active_material_lot' in request.POST):
            print('got it')
            if 'define_active_material' in request.POST:
                define_simple(request.POST, content='active_material')
            if 'define_active_material_lot' in request.POST:
                define_lot(request.POST, content='active_material')

        elif ('define_separator_material' in request.POST) or ('define_separator_material_lot' in request.POST):
            if 'define_separator_material' in request.POST:
                define_simple(request.POST, content='separator_material')
            if 'define_separator_material_lot'  in request.POST:
                define_lot(request.POST, content='separator_material')

        elif ('define_electrode' in request.POST) or ('define_electrode_lot' in request.POST):
            if 'define_electrode' in request.POST:
                define_simple(request.POST, content='electrode')
            if 'define_electrode_lot'  in request.POST:
                define_lot(request.POST, content='electrode')


        elif ('define_separator' in request.POST) or ('define_separator_lot' in request.POST):
            if 'define_separator' in request.POST:
                define_simple(request.POST, content='separator')
            if 'define_separator_lot'  in request.POST:
                define_lot(request.POST, content='separator')

        elif ('define_dry_cell' in request.POST) or ('define_dry_cell_lot' in request.POST):
            define_dry_cell_form = DryCellForm(request.POST)
            if define_dry_cell_form.is_valid():
                print(define_dry_cell_form.cleaned_data)
                ar['define_dry_cell_form'] = define_dry_cell_form
            define_dry_cell_geometry_form = DryCellGeometry(request.POST)
            if define_dry_cell_geometry_form.is_valid():
                print(define_dry_cell_geometry_form.cleaned_data)
                ar['define_dry_cell_geometry_form'] = define_dry_cell_geometry_form

            if 'define_dry_cell_lot' in request.POST:
                define_dry_cell_lot_form = DryCellLotForm(request.POST)
                if define_dry_cell_lot_form.is_valid():
                    print(define_dry_cell_lot_form.cleaned_data)
                    ar['define_dry_cell_lot_form'] = define_dry_cell_lot_form

        elif ('define_wet_cell' in request.POST) :
            define_wet_cell_form = WetCellForm(request.POST)
            if define_wet_cell_form.is_valid():
                print(define_wet_cell_form.cleaned_data)
                ar['define_wet_cell_form'] = define_wet_cell_form


    return render(request, 'WetCellDatabase/define_page.html', ar)



def search_page(request):
    ElectrolyteCompositionFormset = formset_factory(
        SearchElectrolyteComponentForm,
        extra=10
    )


    ElectrolytePreviewFormset = formset_factory(
        ElectrolytePreviewForm,
        extra=0
    )

    electrolyte_composition_formset = ElectrolyteCompositionFormset(prefix='electrolyte_composition')
    electrolyte_preview_formset = ElectrolytePreviewFormset( prefix='electrolyte_preview')
    ar = {}
    ar['electrolyte_composition_formset'] = electrolyte_composition_formset
    ar['electrolyte_preview_formset'] = electrolyte_preview_formset
    ar['electrolyte_form'] = SearchElectrolyteForm()

    if request.method == 'POST':
        electrolyte_composition_formset = ElectrolyteCompositionFormset(request.POST,prefix='electrolyte_composition')
        electrolyte_composition_formset_is_valid = electrolyte_composition_formset.is_valid()
        if electrolyte_composition_formset_is_valid:
            print('valid1')
            ar['electrolyte_composition_formset'] = electrolyte_composition_formset

        search_electrolyte_form = SearchElectrolyteForm(request.POST)
        search_electrolyte_form_is_valid = search_electrolyte_form.is_valid()
        if search_electrolyte_form_is_valid:
            print('valid2')
            ar['electrolyte_form'] = search_electrolyte_form

        electrolyte_preview_formset = ElectrolytePreviewFormset(request.POST,prefix='electrolyte_preview')
        if electrolyte_preview_formset.is_valid():
            print(
                'valid3'
            )
            for form in electrolyte_preview_formset:
                if not form.is_valid():
                    continue
                print(form.cleaned_data)
            ar['electrolyte_preview_formset'] = electrolyte_preview_formset

        if 'preview_electrolyte' in request.POST:

            if electrolyte_composition_formset_is_valid and search_electrolyte_form_is_valid:
                complete_salt = search_electrolyte_form.cleaned_data['complete_salt']
                complete_solvent = search_electrolyte_form.cleaned_data['complete_solvent']
                complete_additive = search_electrolyte_form.cleaned_data['complete_additive']
                relative_tolerance = search_electrolyte_form.cleaned_data['relative_tolerance']
                proprietary_flag = search_electrolyte_form.cleaned_data['proprietary_flag']
                proprietary_search = search_electrolyte_form.cleaned_data['proprietary_search']
                if proprietary_flag:
                    print('search for proprietary flag')
                    q = Q(composite_type=ELECTROLYTE,
                          proprietary=True)
                    if proprietary_search is not None and len(proprietary_search) > 0:
                        q = q & Q(name__icontains=proprietary_search)

                    total_query = Composite.objects.filter(q)
                else:
                    all_data = []
                    for form in electrolyte_composition_formset:
                        if form.is_valid:
                            if 'molecule' in form.cleaned_data.keys() and 'molecule_lot' in form.cleaned_data.keys() and (form.cleaned_data['molecule'] is not None or form.cleaned_data['molecule_lot'] is not None):
                              all_data.append(form.cleaned_data)

                    # TODO(sam): search electrolyte database
                    prohibited = filter(
                        lambda x: x['must_type'] == SearchElectrolyteComponentForm.PROHIBITED,
                        all_data
                    )

                    prohibited_lots = []
                    prohibited_molecules = []
                    for pro in prohibited:
                        if pro['molecule'] is not None:
                            prohibited_molecules.append(pro['molecule'])
                        elif pro['molecule_lot'] is not None:
                            prohibited_lots.append(pro['molecule_lot'])

                    q = Q(composite_type=ELECTROLYTE, proprietary=False)
                    # prohibit molecules

                    q = q & ~Q(components__component_lot__in=prohibited_lots)
                    q = q & ~Q(components__component_lot__component__in=prohibited_molecules)

                    allowed_molecules = []
                    allowed_molecules_lot = []
                    not_prohibited = filter(
                        lambda x: x['must_type'] != SearchElectrolyteComponentForm.PROHIBITED,
                        all_data
                    )
                    for pro in not_prohibited:
                        if pro['molecule'] is not None:
                            allowed_molecules.append(pro['molecule'])
                        elif pro['molecule_lot'] is not None:
                            allowed_molecules_lot.append(pro['molecule_lot'])

                    total_query = Composite.objects.filter(q)
                    completes = []
                    if complete_solvent:
                        completes.append(SOLVENT)
                    if complete_salt:
                        completes.append(SALT)
                    if complete_additive:
                        completes.append(ADDITIVE)

                    if len(completes) !=0:
                        total_query = total_query.annotate(
                            has_extra = Exists(RatioComponent.objects.filter(Q(composite=OuterRef('pk'))&
                                                          Q(component_lot__component__component_type__in=completes)&
                                                          ~Q(component_lot__component__in=allowed_molecules) &
                                                          ~Q(component_lot__in=allowed_molecules_lot)
                                                         ))
                        ).filter(has_extra=False)

                    mandatory_molecules = []
                    mandatory_molecules_lot = []
                    mandatory = filter(
                        lambda x: x['must_type'] == SearchElectrolyteComponentForm.MANDATORY,
                        all_data
                    )
                    for pro in mandatory:
                        if pro['molecule'] is not None:
                            mandatory_molecules.append(pro)
                        elif pro['molecule_lot'] is not None:
                            mandatory_molecules_lot.append(pro)

                    for mol in mandatory_molecules:
                        if mol['ratio'] is None:
                            total_query = total_query.annotate(
                                checked_molecule=Exists(RatioComponent.objects.filter(
                                    composite=OuterRef('pk'),
                                    component_lot__component=mol['molecule']))
                            ).filter(checked_molecule=True)
                        else:
                            tolerance = relative_tolerance/100. * mol['ratio']
                            if mol['tolerance'] is not None:
                                tolerance = mol['tolerance']

                            total_query = total_query.annotate(
                                checked_molecule=Exists(RatioComponent.objects.filter(
                                    composite=OuterRef('pk'),
                                    component_lot__component=mol['molecule'],
                                    ratio__range=(mol['ratio']-tolerance, mol['ratio']+tolerance)))
                            ).filter(checked_molecule=True)

                    for mol in mandatory_molecules_lot:
                        if mol['ratio'] is None:
                            total_query = total_query.annotate(
                                checked_molecule=Exists(RatioComponent.objects.filter(
                                    composite=OuterRef('pk'),
                                    component_lot=mol['molecule_lot']))
                            ).filter(checked_molecule=True)
                        else:
                            tolerance = relative_tolerance / 100. * mol['ratio']
                            if mol['tolerance'] is not None:
                                tolerance = mol['tolerance']

                            total_query = total_query.annotate(
                                checked_molecule=Exists(RatioComponent.objects.filter(
                                    composite=OuterRef('pk'),
                                    component_lot=mol['molecule_lot'],
                                    ratio__range=(mol['ratio'] - tolerance, mol['ratio'] + tolerance)))
                            ).filter(checked_molecule=True)

                initial = []
                for electrolyte in total_query[:100]:
                    my_initial = {
                        "electrolyte": electrolyte.__str__(),
                        "electrolyte_id": electrolyte.id,
                        "exclude": True,
                    }

                    initial.append(my_initial)
                electrolyte_preview_formset = ElectrolytePreviewFormset(initial=initial,prefix='electrolyte_preview')
                ar['electrolyte_preview_formset'] = electrolyte_preview_formset


    return render(request, 'WetCellDatabase/search_page.html', ar)

