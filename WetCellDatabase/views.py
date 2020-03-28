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
DONE - good uniqueness check.(library)
DONE - good uniqueness check (view)
DONE - test uniqueness check
DONE - streamline the various definitions into much simpler and unique flows.
- make the processing specific to machine learning optional.
- bake a tensor instead of having to use the database all the time.
'''





def define_page(request, mode=None):
    ar = {'mode':mode}



    def define_simple(post, content=None):
        if content in ['electrolyte', 'electrode', 'separator']:
            electrode_geometry = None
            separator_geometry = None
            components = None
            components_lot = None

            if content == 'electrolyte':
                ElectrolyteCompositionFormset = formset_factory(
                    ElectrolyteCompositionForm,
                    extra=10
                )
                simple_form = ElectrolyteForm(request.POST, prefix='electrolyte-form')
                component_string = 'molecule'
            elif content == 'electrode':
                ElectrodeCompositionFormset = formset_factory(
                    ElectrodeCompositionForm,
                    extra=10
                )
                simple_form = ElectrodeForm(request.POST, prefix='electrode-form')
                component_string = 'material'
                define_electrode_geometry_form = ElectrodeGeometryForm(request.POST, prefix='electrode-geometry-form')
            elif content == 'separator':
                SeparatorCompositionFormset = formset_factory(
                    SeparatorCompositionForm,
                    extra=10
                )
                simple_form = SeparatorForm(request.POST, prefix='separator-form')
                component_string = 'material'
                define_separator_geometry_form = SeparatorGeometryForm(request.POST, prefix='separator-geometry-form')


            simple_form_string = 'define_{}_form'.format(content)
            composition_formset_string = '{}_composition_formset'.format(content)

            if not simple_form.is_valid():
                return None
            else:
                ar[simple_form_string] = simple_form
                if content == 'electrode':
                    if not define_electrode_geometry_form.is_valid():
                        return None
                    else:
                        ar['define_electrode_geometry_form'] = define_electrode_geometry_form
                        electrode_geometry = ElectrodeGeometry(
                            loading=define_electrode_geometry_form.cleaned_data['loading'],
                            loading_name=define_electrode_geometry_form.cleaned_data['loading_name'],
                            density=define_electrode_geometry_form.cleaned_data['density'],
                            density_name=define_electrode_geometry_form.cleaned_data['density_name'],
                            thickness=define_electrode_geometry_form.cleaned_data['thickness'],
                            thickness_name=define_electrode_geometry_form.cleaned_data['thickness_name'],

                        )
                if content == 'separator':
                    if not define_separator_geometry_form.is_valid():
                        return None
                    else:
                        ar['define_separator_geometry_form'] = define_separator_geometry_form
                        separator_geometry = SeparatorGeometry(
                            thickness=define_separator_geometry_form.cleaned_data['thickness'],
                            thickness_name=define_separator_geometry_form.cleaned_data['thickness_name'],
                            width=define_separator_geometry_form.cleaned_data['width'],
                            width_name=define_separator_geometry_form.cleaned_data['width_name'],

                        )


                my_composite = Composite(
                    proprietary=simple_form.cleaned_data['proprietary'],
                    proprietary_name=simple_form.cleaned_data['proprietary_name'],
                    notes=simple_form.cleaned_data['notes'],
                )
                if content == 'electrolyte':
                    my_composite.composite_type = ELECTROLYTE
                elif content == 'electrode':
                    my_composite.composite_type = simple_form.cleaned_data['composite_type']
                    my_composite.composite_type_name = simple_form.cleaned_data['composite_type_name']

                if content == 'separator':
                    my_composite.composite_type = SEPARATOR

                if not my_composite.proprietary:
                    if content == 'electrolyte':
                        composition_formset = ElectrolyteCompositionFormset(request.POST,
                                                                prefix='electrolyte-composition-formset')
                    if content == 'electrode':
                        composition_formset = ElectrodeCompositionFormset(request.POST,
                                                                prefix='electrode-composition-formset')
                    if content == 'separator':
                        composition_formset = SeparatorCompositionFormset(request.POST,
                                                                prefix='separator-composition-formset')

                    ar[composition_formset_string] = composition_formset

                    components = []
                    components_lot = []
                    for form in composition_formset:
                        validation_step = form.is_valid()
                        if validation_step:
                            print(form.cleaned_data)
                            if not component_string in form.cleaned_data.keys():
                                continue
                            mol_s = form.cleaned_data[component_string]
                            my_id, lot_type = decode_lot_string(mol_s)
                            if lot_type == LotTypes.none:
                                continue
                            value, unknown = unknown_numerical(form.cleaned_data['ratio'])
                            if not unknown and value is None:
                                continue
                            h = {'ratio': value}
                            if lot_type == LotTypes.lot:
                                h['component_lot']= ComponentLot.objects.get(id=my_id)
                                components_lot.append(h)
                            if lot_type == LotTypes.no_lot:

                                h['component']= Component.objects.get(id=my_id)
                                components.append(h)

            return my_composite.define_if_possible(
                components=components,
                components_lot=components_lot,
                electrode_geometry=electrode_geometry,
                separator_geometry=separator_geometry,
            )

        if content in ['active_material','molecule','inactive','separator_material']:
            atoms = None
            if content == 'active_material':
                ActiveMaterialCompositionFormset = formset_factory(
                    ElectrodeMaterialStochiometryForm,
                    extra=10
                )
                simple_form = ElectrodeActiveMaterialForm(request.POST, prefix='electrode-active-material-form')
            elif content == 'molecule':
                simple_form = ElectrolyteMoleculeForm(post, prefix='electrolyte-molecule-form')
            elif content == 'inactive':
                simple_form = ElectrodeInactiveForm(request.POST, prefix='electrode-inactive-form')
            elif content == 'separator_material':
                simple_form = SeparatorMaterialForm(request.POST, prefix='separator-material-form')

            simple_form_string = 'define_{}_form'.format(content)

            if not simple_form.is_valid():
                return None
            else:
                ar[simple_form_string] = simple_form
                if content in ['active_material', 'inactive']:
                    composite_type = simple_form.cleaned_data['composite_type']
                    composite_type_name = simple_form.cleaned_data['composite_type_name']
                if content == 'molecule':
                    composite_type = ELECTROLYTE
                    composite_type_name = False
                if content == 'separator_material':
                    composite_type = SEPARATOR
                    composite_type_name = False

                if content == 'active_material':
                    component_type = ACTIVE_MATERIAL
                    component_type_name = False
                if content in ['inactive','molecule']:
                    component_type = simple_form.cleaned_data['component_type']
                    component_type_name = simple_form.cleaned_data['component_type_name']
                if content == 'separator_material':
                    component_type = SEPARATOR_MATERIAL
                    component_type_name = False



                my_component = Component(
                    proprietary = simple_form.cleaned_data['proprietary'],
                    proprietary_name=simple_form.cleaned_data['proprietary_name'],
                    notes=simple_form.cleaned_data['notes'],
                    component_type=component_type,
                    component_type_name=component_type_name,
                    composite_type=composite_type,
                    composite_type_name=composite_type_name,
                )

                if content != 'molecule':
                    my_id, lot_type = decode_lot_string(
                        simple_form.cleaned_data['coating']
                    )

                    my_component.coating_lot = None

                    if lot_type == LotTypes.unknown:
                        my_component.coating_lot = None
                    else:
                        if lot_type == LotTypes.no_lot:
                            my_component.coating_lot = get_lot(
                                Coating.objects.get(id=my_id),
                                None,
                                type='coating'
                            )
                        elif lot_type == LotTypes.lot:
                            my_component.coating_lot = CoatingLot.objects.get(id=my_id)

                    my_component.particle_size = simple_form.cleaned_data['particle_size']
                    my_component.particle_size_name = simple_form.cleaned_data['particle_size_name']
                    my_component.preparation_temperature = simple_form.cleaned_data['preparation_temperature']
                    my_component.preparation_temperature_name = simple_form.cleaned_data[
                                                       'preparation_temperature_name']
                    my_component.coating_lot_name = simple_form.cleaned_data['coating_lot_name']


                if content == 'active_material':
                    my_component.single_crystal = simple_form.cleaned_data['single_crystal']
                    my_component.single_crystal_name = simple_form.cleaned_data['single_crystal_name']
                    my_component.turbostratic_misalignment = simple_form.cleaned_data['turbostratic_misalignment']
                    my_component.turbostratic_misalignment_name = simple_form.cleaned_data[
                        'turbostratic_misalignment_name']
                    my_component.natural = simple_form.cleaned_data['natural']
                    my_component.natural_name = simple_form.cleaned_data['natural_name']

                if content!='active_material':
                    my_component.smiles = simple_form.cleaned_data['smiles']
                    my_component.smiles_name = simple_form.cleaned_data['smiles_name']

                if not my_component.proprietary and content=='active_material':
                    active_material_composition_formset = ActiveMaterialCompositionFormset(request.POST,
                                                                                           prefix='active-material-composition-formset')
                    atoms = []
                    for form in active_material_composition_formset:
                        validation_step = form.is_valid()
                        if validation_step:
                            print(form.cleaned_data)
                            if not 'atom' in form.cleaned_data.keys():
                                continue
                            if form.cleaned_data['atom'] is not None:
                                value, unknown = unknown_numerical(form.cleaned_data['stochiometry'])
                                if not unknown and value is None:
                                    continue
                                else:
                                    atoms.append(
                                        {
                                            'atom': form.cleaned_data['atom'],
                                            'stochiometry': value
                                        }
                                    )
                    print(atoms)
                return my_component.define_if_possible(
                    atoms=atoms,
                )

        if content == 'coating':
            simple_form = CoatingForm(post, prefix='coating-form')

            simple_form_string = 'define_{}_form'.format(content)

            if not simple_form.is_valid():
                return None
            else:
                ar[simple_form_string] = simple_form
                if simple_form.cleaned_data['name'] is not None:
                    my_content, _ = Coating.objects.get_or_create(
                        name=simple_form.cleaned_data['name'],
                        defaults={
                            'description': simple_form.cleaned_data['description'],
                            'proprietary': simple_form.cleaned_data['proprietary'],
                        }
                    )
                    return my_content

        if content == 'dry_cell':
            simple_form = DryCellForm(
                request.POST,
                prefix='dry-cell-form'
            )

            simple_form_string = 'define_{}_form'.format(content)
            define_dry_cell_geometry_form = DryCellGeometryForm(request.POST, prefix='dry-cell-geometry-form')

            if not define_dry_cell_geometry_form.is_valid():
                return None
            else:
                ar['define_dry_cell_geometry_form'] = define_dry_cell_geometry_form
                dry_cell_geometry = DryCellGeometry(
                    geometry_category=define_dry_cell_geometry_form.cleaned_data['geometry_category'],
                    geometry_category_name=define_dry_cell_geometry_form.cleaned_data['geometry_category_name'],
                    width=define_dry_cell_geometry_form.cleaned_data['width'],
                    width_name=define_dry_cell_geometry_form.cleaned_data['width_name'],
                    length=define_dry_cell_geometry_form.cleaned_data['length'],
                    length_name=define_dry_cell_geometry_form.cleaned_data['length_name'],
                    thickness=define_dry_cell_geometry_form.cleaned_data['thickness'],
                    thickness_name=define_dry_cell_geometry_form.cleaned_data['thickness_name'],

                )
            if not simple_form.is_valid():
                return None
            else:
                ar[simple_form_string] = simple_form
                my_dry_cell = DryCell(
                    proprietary=simple_form.cleaned_data['proprietary'],
                    proprietary_name=simple_form.cleaned_data['proprietary_name'],
                    notes=simple_form.cleaned_data['notes'],
                    anode_name = simple_form.cleaned_data['anode_name'],
                    cathode_name=simple_form.cleaned_data['cathode_name'],
                    separator_name=simple_form.cleaned_data['separator_name'],

                )

                #Cathode
                my_id, lot_type = decode_lot_string(
                    simple_form.cleaned_data['cathode']
                )
                cathode = None
                if lot_type == LotTypes.no_lot:
                    cathode = get_lot(
                        Composite.objects.get(id=my_id),
                        None,
                        type='composite'
                    )
                elif lot_type == LotTypes.lot:
                    cathode = CompositeLot.objects.get(id=my_id)

                #Anode
                my_id, lot_type = decode_lot_string(
                    simple_form.cleaned_data['anode']
                )
                anode = None
                if lot_type == LotTypes.no_lot:
                    anode = get_lot(
                        Composite.objects.get(id=my_id),
                        None,
                        type='composite'
                    )
                elif lot_type == LotTypes.lot:
                    anode = CompositeLot.objects.get(id=my_id)

                #Separator
                my_id, lot_type = decode_lot_string(
                    simple_form.cleaned_data['separator']
                )
                separator = None
                if lot_type == LotTypes.no_lot:
                    separator = get_lot(
                        Composite.objects.get(id=my_id),
                        None,
                        type='composite'
                    )
                elif lot_type == LotTypes.lot:
                    separator = CompositeLot.objects.get(id=my_id)



                return my_dry_cell.define_if_possible(
                    geometry=dry_cell_geometry,
                    cathode=cathode,
                    anode=anode,
                    separator=separator,
                )


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

        elif content == 'dry_cell':
            define_lot_form = DryCellLotForm(
                post,
                prefix='dry-cell-lot-form'
            )


        if define_lot_form.is_valid():
            ar[define_lot_form_string] = define_lot_form
            if define_lot_form.cleaned_data[predefined_string] is not None:
                my_content = define_lot_form.cleaned_data[predefined_string]

            else:
                my_content = define_simple(post, content=content)

            if my_content is not None:
                lot_info = LotInfo(
                    notes=define_lot_form.cleaned_data['notes'],
                    creator=define_lot_form.cleaned_data['creator'],
                    creator_name=define_lot_form.cleaned_data['creator_name'],

                    date=define_lot_form.cleaned_data['date'],
                    date_name=define_lot_form.cleaned_data['date_name'],

                    vendor=define_lot_form.cleaned_data['vendor'],
                    vendor_name=define_lot_form.cleaned_data['vendor_name'],

                )
                if content == 'molecule' or content == 'separator_material' or content == 'inactive' or content == 'active_material':
                    type = 'component'
                    lot = ComponentLot(component = my_content)
                elif content == 'coating':
                    type = 'coating'
                    lot = CoatingLot(coating = my_content)
                elif content == 'electrolyte' or content == 'electrode' or content == 'separator':
                    type = 'composite'
                    lot = CompositeLot(composite = my_content)
                elif content == 'dry_cell':
                    type = 'dry_cell'
                    lot = DryCellLot(composite=my_content)

                else:
                    raise('not yet implemented {}'.format(content))


                define_if_possible(lot, lot_info=lot_info, type=type)

    if request.method == 'POST':
        for m,context in [
            ('molecule','electrolyte'),
            ('coating','electrode'),
            ('inactive','electrode'),
            ('electrolyte','electrolyte'),
            ('active_material','electrode'),
            ('separator_material','separator'),
            ('electrode','electrode'),
            ('separator','separator'),
            ('dry_cell' , 'dry_cell')
        ]:
            if context == mode:
                if ('define_{}'.format(m) in request.POST) or ('define_{}_lot'.format(m) in request.POST):
                    if 'define_{}'.format(m) in request.POST:
                        print(define_simple(request.POST, content=m))
                    if 'define_{}_lot'.format(m) in request.POST:
                        define_lot(request.POST, content=m)


        if mode == 'wet_cell':
            if ('define_wet_cell' in request.POST) :
                define_wet_cell_form = WetCellForm(request.POST)
                if define_wet_cell_form.is_valid():
                    print(define_wet_cell_form.cleaned_data)
                    ar['define_wet_cell_form'] = define_wet_cell_form

    def conditional_register(name, content):
        if name not in ar.keys():
            ar[name] = content

    if mode == 'electrode':
        ActiveMaterialCompositionFormset = formset_factory(
            ElectrodeMaterialStochiometryForm,
            extra=10
        )

        ElectrodeCompositionFormset = formset_factory(
            ElectrodeCompositionForm,
            extra=10
        )

    if mode == 'electrolyte':
        ElectrolyteCompositionFormset = formset_factory(
            ElectrolyteCompositionForm,
            extra=10
        )

    if mode == 'separator':
        SeparatorCompositionFormset = formset_factory(
            SeparatorCompositionForm,
            extra=10
        )

    if mode=='electrode':
        conditional_register(
            'active_material_composition_formset',
            ActiveMaterialCompositionFormset(
                prefix='active-material-composition-formset')
        )

        conditional_register(
            'electrode_composition_formset',
            ElectrodeCompositionFormset(prefix='electrode-composition-formset')
        )

        conditional_register(
            'define_coating_form',
            CoatingForm(prefix='coating-form')
        )


        conditional_register(
            'define_coating_lot_form',
            CoatingLotForm(prefix='coating-lot-form')
        )

        conditional_register(
            'define_inactive_form',
            ElectrodeInactiveForm(prefix='electrode-inactive-form')
        )

        conditional_register(
            'define_inactive_lot_form',
            ElectrodeInactiveLotForm(prefix='electrode-inactive-lot-form')
        )

        conditional_register(
            'define_active_material_form',
            ElectrodeActiveMaterialForm(prefix='electrode-active-material-form')
        )

        conditional_register(
            'define_active_material_lot_form',
            ElectrodeActiveMaterialLotForm(
                prefix='electrode-active-material-lot-form')
        )

        conditional_register(
            'define_electrode_form',
            ElectrodeForm(prefix='electrode-form')
        )

        conditional_register(
            'define_electrode_lot_form',
            ElectrodeLotForm(prefix='electrode-lot-form')
        )

        conditional_register(
            'define_electrode_geometry_form',
            ElectrodeGeometryForm(prefix='electrode-geometry-form')
        )

    if mode =='electrolyte':

        conditional_register(
            'electrolyte_composition_formset',
            ElectrolyteCompositionFormset(prefix='electrolyte-composition-formset')
        )

        conditional_register(
            'define_molecule_form',
            ElectrolyteMoleculeForm(prefix='electrolyte-molecule-form')
        )

        conditional_register(
            'define_molecule_lot_form',
            ElectrolyteMoleculeLotForm(prefix='electrolyte-molecule-lot-form')
        )

        conditional_register(
            'define_electrolyte_form',
            ElectrolyteForm(prefix='electrolyte-form')
        )
        conditional_register(
            'define_electrolyte_lot_form',
            ElectrolyteLotForm(prefix='electrolyte-lot-form')
        )

    if mode == 'separator':

        conditional_register(
            'separator_composition_formset',
            SeparatorCompositionFormset(prefix='separator-composition-formset')
        )
        conditional_register(
            'define_separator_material_form',
            SeparatorMaterialForm(prefix='separator-material-form')
        )
        conditional_register(
            'define_separator_material_lot_form',
            SeparatorMaterialLotForm(prefix='separator-material-lot-form')
        )
        conditional_register(
            'define_separator_form',
            SeparatorForm(prefix='separator-form')
        )
        conditional_register(
            'define_separator_lot_form',
            SeparatorLotForm(prefix='separator-lot-form')
        )
        conditional_register(
            'define_separator_geometry_form',
            SeparatorGeometryForm(prefix='separator-geometry-form')
        )

    if mode == 'dry_cell':
        conditional_register(
            'define_dry_cell_form',
            DryCellForm()
        )
        conditional_register(
            'define_dry_cell_lot_form',
            DryCellLotForm()
        )
        conditional_register(
            'define_dry_cell_geometry_form',
            DryCellGeometryForm()
        )

    if mode == 'wet_cell':
        conditional_register(
            'define_wet_cell_form',
            WetCellForm()
        )


    return render(request, 'WetCellDatabase/define_page.html', ar)

def define_wet_cell_bulk(request, predefined=None):
    ar = {'predefined': predefined}
    if predefined == 'True':
        predefined = True
    elif predefined == 'False':
        predefined = False

    messages = []
    if not predefined:
        ar['bulk_parameters_form'] = ElectrolyteBulkParametersForm(prefix='bulk-parameters-form')
    else:
        ar['bulk_parameters_form'] = WetCellParametersForm(prefix='bulk-parameters-form')
    if request.method == 'POST':
        if ('bulk_parameters_update' in request.POST) or ('bulk_process_entries' in request.POST) :
            if not predefined:
                bulk_parameters_form = ElectrolyteBulkParametersForm(request.POST, prefix='bulk-parameters-form')
            else:
                bulk_parameters_form = WetCellParametersForm(request.POST,prefix='bulk-parameters-form')
            if bulk_parameters_form.is_valid():
                ar['bulk_parameters_form']=bulk_parameters_form
                override_existing = bulk_parameters_form.cleaned_data['override_existing']
                if not predefined:
                    BulkEntriesFormset = formset_factory(
                        ElectrolyteBulkSingleEntryForm,
                        extra=0
                    )
                else:
                    BulkEntriesFormset = formset_factory(
                        WetCellForm,
                        extra=0
                    )
                if ('bulk_parameters_update' in request.POST):
                    initial = {}
                    if not predefined:
                        for i in range(10):
                            initial['value_{}'.format(i)] = bulk_parameters_form.cleaned_data['value_{}'.format(i)]
                        for s in ['proprietary', 'proprietary_name','notes','dry_cell']:
                            initial[s] = bulk_parameters_form.cleaned_data[s]
                    else:
                        initial['dry_cell']=bulk_parameters_form.cleaned_data['dry_cell']
                        initial['electrolyte'] = bulk_parameters_form.cleaned_data['electrolyte']

                    start_barcode = bulk_parameters_form.cleaned_data['start_barcode']
                    end_barcode = bulk_parameters_form.cleaned_data['end_barcode']


                    if start_barcode is not None and end_barcode is None:
                        end_barcode = start_barcode

                    if start_barcode is not None and end_barcode is not None:

                        initials = []
                        for bc in range(start_barcode, end_barcode + 1):
                            init = {'barcode':bc}

                            for k in initial.keys():
                                init[k] = initial[k]
                            initials.append(init)

                        bulk_entries_formset = BulkEntriesFormset(
                            initial = initials,
                            prefix='bulk-entries-formset')
                        ar['bulk_entries_formset'] = bulk_entries_formset

                if ('bulk_process_entries' in request.POST) :
                    bulk_entries_formset = BulkEntriesFormset(
                        request.POST,
                        prefix='bulk-entries-formset')
                    ar['bulk_entries_formset'] = bulk_entries_formset

                    if not predefined:
                        header = []
                        for i in range(10):
                            mol_s = bulk_parameters_form.cleaned_data['molecule_{}'.format(i)]
                            my_id, lot_type = decode_lot_string(mol_s)
                            if lot_type == LotTypes.none:
                                continue
                            if lot_type == LotTypes.lot :
                                header.append({'index': i, 'molecule_lot': ComponentLot.objects.get(id=my_id)})
                                continue
                            if lot_type == LotTypes.no_lot:
                                header.append({'index': i, 'molecule': Component.objects.get(id=my_id)})
                                continue

                    for entry in bulk_entries_formset:
                        if entry.is_valid():
                            if not predefined:
                                my_composite = Composite(
                                    proprietary = entry.cleaned_data['proprietary'],
                                    proprietary_name=entry.cleaned_data['proprietary_name'],
                                    notes=entry.cleaned_data['notes'],
                                    composite_type = ELECTROLYTE)

                                if not my_composite.proprietary:

                                    components = []
                                    components_lot = []
                                    for h in header:
                                        value, unknown = unknown_numerical(entry.cleaned_data['value_{}'.format(h['index'])])
                                        if not unknown and value is None:
                                            continue
                                        else:
                                            if 'molecule' in h.keys():
                                                print('recognised molecule')
                                                components.append(
                                                    {
                                                        'component': h['molecule'],
                                                        'ratio': value
                                                    }
                                                )
                                            elif 'molecule_lot' in h.keys():
                                                print('recognised lot')
                                                components_lot.append(
                                                    {
                                                        'component_lot':h['molecule_lot'],
                                                        'ratio': value
                                                    }
                                                )

                                my_electrolyte = my_composite.define_if_possible(
                                    components=components,
                                    components_lot=components_lot,

                                )

                                my_electrolyte_lot = get_lot(
                                    my_electrolyte,
                                    None,
                                    'composite'
                                )
                            else:
                                electrolyte_s = entry.cleaned_data['electrolyte']
                                my_id, lot_type = decode_lot_string(electrolyte_s)
                                my_electrolyte_lot = None
                                if lot_type == LotTypes.none:
                                    my_electrolyte_lot = None
                                if lot_type == LotTypes.lot:
                                    my_electrolyte_lot = CompositeLot.objects.get(id=my_id)
                                if lot_type == LotTypes.no_lot:
                                    my_electrolyte_lot = get_lot(
                                        Composite.objects.get(id=my_id),
                                        None,
                                        type='composite'
                                    )

                            dry_cell_s = entry.cleaned_data['dry_cell']
                            my_id, lot_type = decode_lot_string(dry_cell_s)
                            my_dry_cell = None
                            if lot_type == LotTypes.none:
                                my_dry_cell = None
                            if lot_type == LotTypes.lot:
                                my_dry_cell = DryCellLot.objects.get(id=my_id)
                            if lot_type == LotTypes.no_lot:
                                my_dry_cell = get_lot(
                                    DryCell.objects.get(id=my_id),
                                    None,
                                    type='dry_cell'
                                )

                            my_cell_id = entry.cleaned_data['barcode']

                            print('!!!!creating a wet cell!!!!')
                            print(my_cell_id)
                            print(my_electrolyte_lot)
                            print(my_dry_cell)


                            if not WetCell.objects.filter(cell_id=my_cell_id).exists():
                                my_wet_cell= WetCell.objects.create(
                                    cell_id = my_cell_id,
                                    electrolyte=my_electrolyte_lot,
                                    dry_cell = my_dry_cell
                                )
                            else:
                                my_wet_cell = WetCell.objects.get(
                                    cell_id=my_cell_id)
                                if override_existing:
                                    my_wet_cell.electrolyte = my_electrolyte_lot
                                    my_wet_cell.dry_cell = my_dry_cell
                                    my_wet_cell.save()

                            messages.append(my_wet_cell.__str__())
                        else:
                            print('!!!!something was invalid!!!!')
                            print(entry)
                            messages.append(None)

                    ar['messages'] = messages

    if not predefined:
        return render(request, 'WetCellDatabase/define_wet_cell_bulk.html', ar)
    else:
        return render(request, 'WetCellDatabase/define_wet_cell_bulk.html', ar)




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

