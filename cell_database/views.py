from django.shortcuts import render, render_to_response
from django.forms import modelformset_factory, formset_factory
from django.db.models import Q, F, Func, Count,Exists, OuterRef
from .forms import *
from .models import *
from plot import plot_cycling_direct
from django import forms
import re



def conditional_register(ar, name, content):
    if name not in ar.keys():
        ar[name] = content



def define_page(request, mode=None):
    ar = {'mode':mode}
    def define_simple(post, content=None):
        if content in ['electrolyte', 'electrode', 'separator']:
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

            elif content == 'separator':
                SeparatorCompositionFormset = formset_factory(
                    SeparatorCompositionForm,
                    extra=10
                )
                simple_form = SeparatorForm(request.POST, prefix='separator-form')
                component_string = 'material'


            simple_form_string = 'define_{}_form'.format(content)
            composition_formset_string = '{}_composition_formset'.format(content)

            if not simple_form.is_valid():
                return None
            else:
                ar[simple_form_string] = simple_form




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

            my_target = simple_form.cleaned_data['override_target']

            if my_target is not None:
                my_target = my_target.id

            return my_composite.define_if_possible(
                components=components,
                components_lot=components_lot,
                target=my_target
            )

        if content in ['material','molecule','separator_material']:
            atoms = None
            if content == 'material':
                ActiveMaterialCompositionFormset = formset_factory(
                    ElectrodeMaterialStochiometryForm,
                    extra=10
                )
                simple_form = ElectrodeMaterialForm(request.POST, prefix='electrode-material-form')
            elif content == 'molecule':
                simple_form = ElectrolyteMoleculeForm(post, prefix='electrolyte-molecule-form')
            elif content == 'separator_material':
                simple_form = SeparatorMaterialForm(request.POST, prefix='separator-material-form')

            simple_form_string = 'define_{}_form'.format(content)

            if not simple_form.is_valid():
                return None
            else:
                ar[simple_form_string] = simple_form
                if content in ['material']:
                    composite_type = simple_form.cleaned_data['composite_type']
                    composite_type_name = simple_form.cleaned_data['composite_type_name']
                if content == 'molecule':
                    composite_type = ELECTROLYTE
                    composite_type_name = False
                if content == 'separator_material':
                    composite_type = SEPARATOR
                    composite_type_name = False

                if content in ['material','molecule']:
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


                if content == 'material':
                    my_component.single_crystal = simple_form.cleaned_data['single_crystal']
                    my_component.single_crystal_name = simple_form.cleaned_data['single_crystal_name']
                    my_component.turbostratic_misalignment = simple_form.cleaned_data['turbostratic_misalignment']
                    my_component.turbostratic_misalignment_name = simple_form.cleaned_data[
                        'turbostratic_misalignment_name']
                    my_component.natural = simple_form.cleaned_data['natural']
                    my_component.natural_name = simple_form.cleaned_data['natural_name']


                my_component.smiles = simple_form.cleaned_data['smiles']
                my_component.smiles_name = simple_form.cleaned_data['smiles_name']

                if not my_component.proprietary and content=='material':
                    active_material_composition_formset = ActiveMaterialCompositionFormset(
                        request.POST,
                        prefix='active-material-composition-formset'
                    )
                    atoms = []
                    for form in active_material_composition_formset:
                        validation_step = form.is_valid()
                        if validation_step:

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


                my_target = simple_form.cleaned_data['override_target']

                if my_target is not None:
                    my_target = my_target.id
                return my_component.define_if_possible(
                    atoms=atoms,
                    target=my_target
                )

        if content == 'coating':
            simple_form = CoatingForm(post, prefix='coating-form')

            simple_form_string = 'define_{}_form'.format(content)

            if not simple_form.is_valid():
                return None
            else:
                ar[simple_form_string] = simple_form

                my_coating = Coating(
                    proprietary=simple_form.cleaned_data['proprietary'],
                    proprietary_name=simple_form.cleaned_data['proprietary_name'],
                    notes=simple_form.cleaned_data['notes'],
                    description=simple_form.cleaned_data['description'],
                    description_name=simple_form.cleaned_data['description_name'],
                )


                my_target = simple_form.cleaned_data['override_target']

                if my_target is not None:
                    my_target = my_target.id

                return my_coating.define_if_possible(
                    target=my_target

                )

        if content == 'dry_cell':
            cathode_geometry = None
            anode_geometry = None
            separator_geometry = None
            define_cathode_geometry_form = ElectrodeGeometryForm(request.POST, prefix='cathode-geometry-form')
            define_anode_geometry_form = ElectrodeGeometryForm(request.POST, prefix='anode-geometry-form')
            define_separator_geometry_form = SeparatorGeometryForm(request.POST, prefix='separator-geometry-form')

            simple_form = DryCellForm(
                request.POST,
                prefix='dry-cell-form'
            )

            simple_form_string = 'define_{}_form'.format(content)
            define_dry_cell_geometry_form = DryCellGeometryForm(request.POST, prefix='dry-cell-geometry-form')

            if not define_dry_cell_geometry_form.is_valid() or not define_cathode_geometry_form.is_valid() or not define_anode_geometry_form.is_valid() or not define_separator_geometry_form.is_valid():
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

                ar['define_cathode_geometry_form'] = define_cathode_geometry_form
                cathode_geometry = ElectrodeGeometry(
                    loading=define_cathode_geometry_form.cleaned_data['loading'],
                    loading_name=define_cathode_geometry_form.cleaned_data['loading_name'],
                    density=define_cathode_geometry_form.cleaned_data['density'],
                    density_name=define_cathode_geometry_form.cleaned_data['density_name'],
                    thickness=define_cathode_geometry_form.cleaned_data['thickness'],
                    thickness_name=define_cathode_geometry_form.cleaned_data['thickness_name'],

                )

                ar['define_anode_geometry_form'] = define_anode_geometry_form
                anode_geometry = ElectrodeGeometry(
                    loading=define_anode_geometry_form.cleaned_data['loading'],
                    loading_name=define_anode_geometry_form.cleaned_data['loading_name'],
                    density=define_anode_geometry_form.cleaned_data['density'],
                    density_name=define_anode_geometry_form.cleaned_data['density_name'],
                    thickness=define_anode_geometry_form.cleaned_data['thickness'],
                    thickness_name=define_anode_geometry_form.cleaned_data['thickness_name'],

                )

                ar['define_separator_geometry_form'] = define_separator_geometry_form
                separator_geometry = SeparatorGeometry(
                    thickness=define_separator_geometry_form.cleaned_data['thickness'],
                    thickness_name=define_separator_geometry_form.cleaned_data['thickness_name'],
                    width=define_separator_geometry_form.cleaned_data['width'],
                    width_name=define_separator_geometry_form.cleaned_data['width_name'],

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

                my_target = simple_form.cleaned_data['override_target']

                if my_target is not None:
                    my_target = my_target.id

                return my_dry_cell.define_if_possible(
                    geometry=dry_cell_geometry,
                    cathode=cathode,
                    anode=anode,
                    separator=separator,
                    cathode_geometry=cathode_geometry,
                    anode_geometry=anode_geometry,
                    separator_geometry=separator_geometry,
                    target = my_target

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
        elif content == 'material':
            define_lot_form = ElectrodeMaterialLotForm(
                post,
                prefix='electrode-material-lot-form'
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
                if content == 'molecule' or content == 'separator_material' or content == 'material':
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
                    lot = DryCellLot(dry_cell=my_content)

                else:
                    raise('not yet implemented {}'.format(content))


                define_if_possible(lot, lot_info=lot_info, type=type)

    if request.method == 'POST':
        for m,context in [
            ('molecule','electrolyte'),
            ('coating','electrode'),
            ('electrolyte','electrolyte'),
            ('material','electrode'),
            ('separator_material','separator'),
            ('electrode','electrode'),
            ('separator','separator'),
            ('dry_cell' , 'dry_cell')
        ]:
            if context == mode:
                if ('define_{}'.format(m) in request.POST) or ('define_{}_lot'.format(m) in request.POST):
                    if 'define_{}'.format(m) in request.POST:
                        define_simple(request.POST, content=m)
                    if 'define_{}_lot'.format(m) in request.POST:
                        define_lot(request.POST, content=m)



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
        conditional_register(ar,
            'active_material_composition_formset',
            ActiveMaterialCompositionFormset(
                prefix='active-material-composition-formset')
        )

        conditional_register(ar,
            'electrode_composition_formset',
            ElectrodeCompositionFormset(prefix='electrode-composition-formset')
        )

        conditional_register(ar,
            'define_coating_form',
            CoatingForm(prefix='coating-form')
        )


        conditional_register(ar,
            'define_coating_lot_form',
            CoatingLotForm(prefix='coating-lot-form')
        )


        conditional_register(ar,
            'define_material_form',
            ElectrodeMaterialForm(prefix='electrode-material-form')
        )

        conditional_register(ar,
            'define_material_lot_form',
            ElectrodeMaterialLotForm(
                prefix='electrode-material-lot-form')
        )

        conditional_register(ar,
            'define_electrode_form',
            ElectrodeForm(prefix='electrode-form')
        )

        conditional_register(ar,
            'define_electrode_lot_form',
            ElectrodeLotForm(prefix='electrode-lot-form')
        )



    if mode =='electrolyte':

        conditional_register(ar,
            'electrolyte_composition_formset',
            ElectrolyteCompositionFormset(prefix='electrolyte-composition-formset')
        )

        conditional_register(ar,
            'define_molecule_form',
            ElectrolyteMoleculeForm(prefix='electrolyte-molecule-form')
        )

        conditional_register(ar,
            'define_molecule_lot_form',
            ElectrolyteMoleculeLotForm(prefix='electrolyte-molecule-lot-form')
        )

        conditional_register(ar,
            'define_electrolyte_form',
            ElectrolyteForm(prefix='electrolyte-form')
        )
        conditional_register(ar,
            'define_electrolyte_lot_form',
            ElectrolyteLotForm(prefix='electrolyte-lot-form')
        )

    if mode == 'separator':

        conditional_register(ar,
            'separator_composition_formset',
            SeparatorCompositionFormset(prefix='separator-composition-formset')
        )
        conditional_register(ar,
            'define_separator_material_form',
            SeparatorMaterialForm(prefix='separator-material-form')
        )
        conditional_register(ar,
            'define_separator_material_lot_form',
            SeparatorMaterialLotForm(prefix='separator-material-lot-form')
        )
        conditional_register(ar,
            'define_separator_form',
            SeparatorForm(prefix='separator-form')
        )
        conditional_register(ar,
            'define_separator_lot_form',
            SeparatorLotForm(prefix='separator-lot-form')
        )


    if mode == 'dry_cell':
        conditional_register(ar,
            'define_dry_cell_form',
            DryCellForm(prefix='dry-cell-form')
        )
        conditional_register(ar,
            'define_dry_cell_lot_form',
            DryCellLotForm(prefix='dry-cell-lot-form')
        )
        conditional_register(ar,
            'define_dry_cell_geometry_form',
            DryCellGeometryForm(prefix='dry-cell-geometry-form')
        )
        conditional_register(ar,
            'define_separator_geometry_form',
            SeparatorGeometryForm(prefix='separator-geometry-form')
        )

        conditional_register(ar,
            'define_cathode_geometry_form',
            ElectrodeGeometryForm(prefix='cathode-geometry-form')
        )
        conditional_register(ar,
            'define_anode_geometry_form',
            ElectrodeGeometryForm(prefix='anode-geometry-form')
        )
    return render(request, 'cell_database/define_page.html', ar)

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
                        # initial['dry_cell']=bulk_parameters_form.cleaned_data['dry_cell']
                        initial['electrolyte'] = bulk_parameters_form.cleaned_data['electrolyte']

                    start_cell_id = bulk_parameters_form.cleaned_data['start_cell_id']
                    end_cell_id = bulk_parameters_form.cleaned_data['end_cell_id']
                    number_of_cell_ids = bulk_parameters_form.cleaned_data['number_of_cell_ids']

                    if start_cell_id is not None and end_cell_id is None and number_of_cell_ids is None:
                        end_cell_id = start_cell_id
                    if start_cell_id is not None and end_cell_id is None and number_of_cell_ids is not None:
                        end_cell_id = start_cell_id + number_of_cell_ids - 1

                    if start_cell_id is None and number_of_cell_ids is not None:
                        start_cell_id = 0
                        end_cell_id = start_cell_id + number_of_cell_ids - 1

                    if start_cell_id is not None and end_cell_id is not None:

                        initials = []
                        for bc in range(start_cell_id, end_cell_id + 1):
                            init = {'cell_id':bc}

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

                                                components.append(
                                                    {
                                                        'component': h['molecule'],
                                                        'ratio': value
                                                    }
                                                )
                                            elif 'molecule_lot' in h.keys():

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

                            if not predefined:
                                dry_cell_s = bulk_parameters_form.cleaned_data['dry_cell']
                            else:
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

                            my_cell_id = entry.cleaned_data['cell_id']





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


    return render(request, 'cell_database/define_wet_cell_bulk.html', ar)


def get_preview_electrolytes(search_electrolyte_form, electrolyte_composition_formset):
    complete_salt = search_electrolyte_form.cleaned_data['complete_salt']
    complete_solvent = search_electrolyte_form.cleaned_data['complete_solvent']
    complete_additive = search_electrolyte_form.cleaned_data['complete_additive']
    relative_tolerance = search_electrolyte_form.cleaned_data['relative_tolerance']
    proprietary_flag = search_electrolyte_form.cleaned_data['proprietary_flag']
    notes = search_electrolyte_form.cleaned_data['notes']
    q = Q(composite_type=ELECTROLYTE)
    if notes is not None and len(notes) > 0:
        q = q & Q(notes__icontains=notes)

    if proprietary_flag:
        q = q & Q(proprietary=True)

        total_query = Composite.objects.filter(q)
    else:
        all_data = []
        for form in electrolyte_composition_formset:
            if form.is_valid:
                if 'molecule' in form.cleaned_data.keys() and form.cleaned_data['molecule'] is not None and \
                        form.cleaned_data['molecule'] != '':
                    all_data.append(form.cleaned_data)

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

        if len(completes) != 0:
            total_query = total_query.annotate(
                has_extra=Exists(RatioComponent.objects.filter(
                    Q(composite=OuterRef('pk')) &
                    Q(component_lot__component__component_type__in=completes) &
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
                tolerance = relative_tolerance / 100. * mol['ratio']
                if mol['tolerance'] is not None:
                    tolerance = mol['tolerance']

                total_query = total_query.annotate(
                    checked_molecule=Exists(RatioComponent.objects.filter(
                        composite=OuterRef('pk'),
                        component_lot__component=mol['molecule'],
                        ratio__range=(mol['ratio'] - tolerance, mol['ratio'] + tolerance)))
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

    return total_query

def get_preview_dry_cells( search_dry_cell, dry_cell_scalars):
    dry_cell_notes = ""
    dry_cell_proprietary = False
    geometry_category = []
    cathode_id = []
    anode_id = []
    separator_id = []

    relative_tolerance = 5.

    if search_dry_cell.is_valid():
        dry_cell_notes = search_dry_cell.cleaned_data["notes"]
        dry_cell_proprietary = search_dry_cell.cleaned_data["proprietary"]
        # TODO(sam): deal with geometry category.
        geometry_category = search_dry_cell.cleaned_data.get("geometry_category")
        geometry_category_exclude_missing = search_dry_cell.cleaned_data["geometry_category_exclude_missing"]

        cathode_strings = search_dry_cell.cleaned_data.get("cathode")
        cathode_exclude_missing = search_dry_cell.cleaned_data["cathode_exclude_missing"]

        anode_strings = search_dry_cell.cleaned_data.get("anode")
        anode_exclude_missing = search_dry_cell.cleaned_data["anode_exclude_missing"]

        separator_strings = search_dry_cell.cleaned_data.get("separator")
        separator_exclude_missing = search_dry_cell.cleaned_data["separator_exclude_missing"]

        def extract_ids(things):
            all_id = []
            for thing_s in things:
                my_id, lot_type = decode_lot_string(thing_s)
                if lot_type == LotTypes.no_lot:
                    all_id.append(my_id)
            return all_id

        cathode_id = extract_ids(cathode_strings)
        anode_id = extract_ids(anode_strings)
        separator_id = extract_ids(separator_strings)

    dry_cell_scalar_dict = {}
    for dry_cell_scalar in dry_cell_scalars:
        if dry_cell_scalar.is_valid():
            if "name" in dry_cell_scalar.cleaned_data.keys() and dry_cell_scalar.cleaned_data["name"] != "":
                name = dry_cell_scalar.cleaned_data["name"]
                if "scalar" in dry_cell_scalar.cleaned_data.keys() and dry_cell_scalar.cleaned_data[
                    "scalar"] is not None:
                    scalar = dry_cell_scalar.cleaned_data["scalar"]
                    if "tolerance" in dry_cell_scalar.cleaned_data.keys() and dry_cell_scalar.cleaned_data[
                        "tolerance"] is not None:
                        tolerance = dry_cell_scalar.cleaned_data["tolerance"]
                    else:
                        tolerance = scalar * relative_tolerance / 100.

                    exclude_missing = False
                    if "exclude_missing" in dry_cell_scalar.cleaned_data.keys():
                        exclude_missing = dry_cell_scalar.cleaned_data["exclude_missing"]
                    dry_cell_scalar_dict[name] = (scalar, tolerance, exclude_missing)

    q = Q()
    if dry_cell_notes is not None and len(dry_cell_notes) > 0:
        q = q & Q(notes__icontains=dry_cell_notes)

    q = q & Q(proprietary=dry_cell_proprietary)

    todos = [
        (
            geometry_category,
            geometry_category_exclude_missing,
            Q(geometry__geometry_category=None),
            Q(geometry__geometry_category__in=geometry_category),
        ),
        (
            cathode_id,
            cathode_exclude_missing,
            Q(cathode__composite=None),
            Q(cathode__composite__id__in=cathode_id),
        ),
        (
            anode_id,
            anode_exclude_missing,
            Q(anode__composite=None),
            Q(anode__composite__id__in=anode_id),
        ),
        (
            separator_id,
            separator_exclude_missing,
            Q(separator__composite=None),
            Q(separator__composite__id__in=separator_id),
        ),
    ]
    for list_of_things, exclude_missing, none_q, contained_q in todos:
        if len(list_of_things) > 0:
            if exclude_missing:
                q = q & (~none_q & contained_q)
            else:
                q = q & (none_q | contained_q)
        elif exclude_missing:
            q = q & ~none_q

    todos = [
        (
            "Cathode Loading",
            lambda: Q(cathode_geometry__loading=None),
            lambda scalar, tolerance: Q(cathode_geometry__loading__range=(scalar - tolerance, scalar + tolerance))
        ),
        (
            "Cathode Density",
            lambda: Q(cathode_geometry__density=None),
            lambda scalar, tolerance: Q(
                cathode_geometry__density__range=(scalar - tolerance, scalar + tolerance))
        ),
        (
            "Cathode Thickness",
            lambda: Q(cathode_geometry__thickness=None),
            lambda scalar, tolerance: Q(
                cathode_geometry__thickness__range=(scalar - tolerance, scalar + tolerance))
        ),

        (
            "Anode Loading",
            lambda: Q(anode_geometry__loading=None),
            lambda scalar, tolerance: Q(
                anode_geometry__loading__range=(scalar - tolerance, scalar + tolerance))
        ),
        (
            "Anode Density",
            lambda: Q(anode_geometry__density=None),
            lambda scalar, tolerance: Q(
                anode_geometry__density__range=(scalar - tolerance, scalar + tolerance))
        ),
        (
            "Anode Thickness",
            lambda: Q(anode_geometry__thickness=None),
            lambda scalar, tolerance: Q(
                anode_geometry__thickness__range=(scalar - tolerance, scalar + tolerance))
        ),

        (
            "Separator Thickness",
            lambda: Q(separator_geometry__thickness=None),
            lambda scalar, tolerance: Q(
                separator_geometry__thickness__range=(scalar - tolerance, scalar + tolerance))
        ),

        (
            "Separator Width",
            lambda: Q(separator_geometry__width=None),
            lambda scalar, tolerance: Q(
                separator_geometry__width__range=(scalar - tolerance, scalar + tolerance))
        ),
        (
            "Cell Width",
            lambda: Q(geometry__width=None),
            lambda scalar, tolerance: Q(
                geometry__width__range=(scalar - tolerance, scalar + tolerance))
        ),
        (
            "Cell Length",
            lambda: Q(geometry__length=None),
            lambda scalar, tolerance: Q(
                geometry__length__range=(scalar - tolerance, scalar + tolerance))
        ),
        (
            "Cell Thickness",
            lambda: Q(geometry__thickness=None),
            lambda scalar, tolerance: Q(
                geometry__thickness__range=(scalar - tolerance, scalar + tolerance))
        ),

    ]

    for name, get_none_q, get_range_q in todos:
        if name in dry_cell_scalar_dict.keys():
            scalar, tolerance, exclude_missing = dry_cell_scalar_dict[name]
            none_q = get_none_q()
            range_q = get_range_q(scalar, tolerance)

            if exclude_missing:
                q = q & (~none_q & range_q)
            else:
                q = q & (none_q | range_q)

    total_query = DryCell.objects.filter(q)

    return total_query

#TODO(sam): share across applications
def focus_on_page(search_form, n, number_per_page = 10):
    pn = search_form.cleaned_data["page_number"]
    if pn is None:
        pn = 1
        search_form.set_page_number(pn)
    max_page = int(n / number_per_page)
    if (n % number_per_page) != 0:
        max_page += 1

    if pn > max_page:
        pn = max_page
        search_form.set_page_number(pn)

    if pn < 1:
        pn = 1
        search_form.set_page_number(pn)

    return (pn - 1) * number_per_page, min(n, (pn) * number_per_page), max_page, pn


def search_page(request):
    ElectrolyteCompositionFormset = formset_factory(
        SearchElectrolyteComponentForm,
        extra=10
    )
    DryCellScalarsFormset = formset_factory(
        SearchGenericNamedScalarForm,
        extra=0
    )
    WetCellPreviewFormset = formset_factory(
        WetCellPreviewForm,
        extra=0
    )

    def set_scalars(ar):
        dry_cell_scalars_initial = [
            {
                'name': "Cathode Loading",
            },
            {
                'name': "Cathode Density",
            },
            {
                'name': "Cathode Thickness",
            },
            {
                'name': "Anode Loading",
            },
            {
                'name': "Anode Density",
            },
            {
                'name': "Anode Thickness",
            },
            {
                'name': "Separator Thickness",
            },
            {
                'name': "Separator Width",
            },
            {
                'name': "Cell Width",
            },
            {
                'name': "Cell Length",
            },
            {
                'name': "Cell Thickness",
            },
        ]
        ar['dry_cell_scalars'] = DryCellScalarsFormset(
            initial=dry_cell_scalars_initial,
            prefix='dry_cell_scalars'
        )

    def get_dry_cell_forms(ar, post):
        search_dry_cell = SearchDryCellForm(post, prefix="search_dry_cell")
        dry_cell_scalars = DryCellScalarsFormset(
            post,
            prefix='dry_cell_scalars'
        )
        dry_cell_scalars_is_valid = dry_cell_scalars.is_valid()
        search_dry_cell_is_valid = search_dry_cell.is_valid()
        if not dry_cell_scalars_is_valid:
            set_scalars(ar)
        else:
            ar["dry_cell_scalars"] = dry_cell_scalars

        if search_dry_cell_is_valid:
            ar["search_dry_cell"] = search_dry_cell


        return search_dry_cell, dry_cell_scalars, dry_cell_scalars_is_valid and search_dry_cell_is_valid

    def get_electrolyte_forms(ar, post):
        electrolyte_composition_formset = ElectrolyteCompositionFormset(post, prefix='electrolyte-composition')
        electrolyte_composition_formset_is_valid = electrolyte_composition_formset.is_valid()
        if electrolyte_composition_formset_is_valid:
            ar['electrolyte_composition_formset'] = electrolyte_composition_formset

        search_electrolyte_form = SearchElectrolyteForm(post, prefix='search-electrolyte')
        search_electrolyte_form_is_valid = search_electrolyte_form.is_valid()
        if search_electrolyte_form_is_valid:
            ar['electrolyte_form'] = search_electrolyte_form

        return electrolyte_composition_formset, search_electrolyte_form, electrolyte_composition_formset_is_valid and search_electrolyte_form_is_valid

    ar = {}
    if request.method == 'GET':
        set_scalars(ar)

    if request.method == 'POST':

        # electrolyte
        electrolyte_composition_formset, search_electrolyte_form, proceed_electrolyte = get_electrolyte_forms(ar, request.POST)
        # dry cell
        search_dry_cell, dry_cell_scalars, proceed_dry_cell = get_dry_cell_forms(ar, request.POST)

        if 'preview_dry_cell' in request.POST:
            if proceed_dry_cell:
                total_query = get_preview_dry_cells(search_dry_cell, dry_cell_scalars)
                max_n = 25
                preview_dry_cells =  [dry_cell.__str__() for dry_cell in total_query[:max_n]]
                if total_query.count() > max_n:
                    preview_dry_cells.append("... (more than {} found) ...".format(max_n))
                ar['preview_dry_cells'] = preview_dry_cells

        if 'preview_electrolyte' in request.POST:
            if proceed_electrolyte:
                total_query = get_preview_electrolytes(search_electrolyte_form, electrolyte_composition_formset)

                max_n = 25
                preview_electrolytes =  [electrolyte.__str__() for electrolyte in total_query[:max_n]]
                if total_query.count() > max_n:
                    preview_electrolytes.append("... (more than {} found) ...".format(max_n))
                ar['preview_electrolytes'] = preview_electrolytes
        if 'preview_wet_cell' in request.POST:
            if proceed_electrolyte and proceed_dry_cell:

                electrolyte_query = get_preview_electrolytes(
                    search_electrolyte_form,
                    electrolyte_composition_formset
                )

                dry_cell_query = get_preview_dry_cells(
                    search_dry_cell,
                    dry_cell_scalars
                )

                wet_cell_query = WetCell.objects.filter(
                    electrolyte__composite__in=electrolyte_query,
                    dry_cell__dry_cell__in=dry_cell_query
                )
                search_wet_cell_form = SearchWetCellForm(request.POST, prefix='search-wet-cell-form')
                if search_wet_cell_form.is_valid():
                    min_i, max_i, max_page_number, page_number = focus_on_page(search_wet_cell_form, wet_cell_query.count(),number_per_page=20)
                    ar["max_page_number"] = max_page_number
                    ar["page_number"] = page_number
                    ar['search_wet_cell_form']=search_wet_cell_form
                    initial = []
                    for wet_cell in wet_cell_query[min_i:max_i]:
                        my_initial = {
                            "wet_cell": wet_cell.__str__(),
                            "cell_id": wet_cell.cell_id,
                            "exclude": True,
                        }

                        initial.append(my_initial)

                    wet_cell_preview_formset = WetCellPreviewFormset(
                        initial=initial,
                        prefix='wet-cell-preview-formset')
                    ar['wet_cell_preview_formset'] = wet_cell_preview_formset
                    ar['dataset_form'] = DatasetForm(prefix='dataset-form')
                    # preview_wet_cells = [wet_cell.__str__() for wet_cell in wet_cell_query[min_i:max_i]]
                    # ar['preview_wet_cells'] = preview_wet_cells

        if 'register_wet_cells_to_dataset' in request.POST:
            wet_cell_preview_formset = WetCellPreviewFormset(request.POST, prefix='wet-cell-preview-formset')
            dataset_form =DatasetForm(request.POST, prefix='dataset-form')
            if dataset_form.is_valid():

                dataset = dataset_form.cleaned_data["dataset"]
                if dataset is not None:
                    if wet_cell_preview_formset.is_valid():

                        for form in wet_cell_preview_formset:
                            if not form.is_valid():
                                continue
                            if "exclude" not in form.cleaned_data.keys():
                                continue
                            if form.cleaned_data["exclude"]:
                                continue

                            cell_id = form.cleaned_data["cell_id"]

                            if WetCell.objects.filter(cell_id = cell_id).exists():
                                wet_cell = WetCell.objects.get(cell_id = cell_id)

                                dataset.wet_cells.add(wet_cell)


                        ar['wet_cell_preview_formset'] = wet_cell_preview_formset
                ar["dataset_form"] = dataset_form
    # electrolyte
    conditional_register(ar,
                         'electrolyte_composition_formset',
                         ElectrolyteCompositionFormset(prefix='electrolyte-composition')
                         )
    conditional_register(ar,
                         'electrolyte_form',
                         SearchElectrolyteForm(prefix='search-electrolyte')
                         )

    # dry cell
    conditional_register(ar,
                         "search_dry_cell",
                         SearchDryCellForm(prefix="search_dry_cell")
                         )
    conditional_register(ar,
                         'search_wet_cell_form',
                         SearchWetCellForm(prefix='search_wet_cell_form')
                         )

    conditional_register(ar,
                         'wet_cell_preview_formset',
                         WetCellPreviewFormset(prefix='wet_cell_preview_formset')
                         )
    return render(request, 'cell_database/search_page.html', ar)



def delete_page(request):
    ar = {}
    if request.method == 'POST':
        form = DeleteForm(request.POST)
        if form.is_valid():
            todos = [
                (
                    'delete_molecules',
                    ComponentLot,
                    Component,
                    lambda my_id: Q(component__id=my_id)
                ),

                (
                    'delete_electrolytes',
                    CompositeLot,
                    Composite,
                    lambda my_id: Q(composite__id=my_id)
                ),
                (
                    'delete_coatings',
                    CoatingLot,
                    Coating,
                    lambda my_id: Q(coating__id=my_id)
                ),
                (
                    'delete_materials',
                    ComponentLot,
                    Component,
                    lambda my_id: Q(component__id=my_id)
                ),
                (
                    'delete_anodes',
                    CompositeLot,
                    Composite,
                    lambda my_id: Q(composite__id=my_id)
                ),
                (
                    'delete_cathodes',
                    CompositeLot,
                    Composite,
                    lambda my_id: Q(composite__id=my_id)
                ),
                (
                    'delete_separator_materials',
                    ComponentLot,
                    Component,
                    lambda my_id: Q(component__id=my_id)
                ),
                (
                    'delete_separators',
                    CompositeLot,
                    Composite,
                    lambda my_id: Q(composite__id=my_id)
                ),

                (
                    'delete_dry_cells',
                    DryCellLot,
                    DryCell,
                    lambda my_id: Q(dry_cell__id=my_id)
                ),
            ]
            for name, cat_lot, cat, f in todos:
                things = form.cleaned_data.get(name)
                for thing_s in things:
                    my_id, lot_type = decode_lot_string(thing_s)
                    # my_ = None
                    if lot_type == LotTypes.lot:
                        cat_lot.objects.filter(id=my_id).delete()
                    if lot_type == LotTypes.no_lot:
                        cat_lot.objects.filter(f(my_id)).delete()
                        cat.objects.filter(id=my_id).delete()

    else:
        form = DeleteForm

    ar['form'] =form
    return render(request, 'cell_database/delete_page.html', ar)


def view_dataset(request, pk):
    dataset = Dataset.objects.get(id=pk)

    wet_cells = dataset.wet_cells.order_by('cell_id')
    zipped_data = zip(wet_cells, [wet_cell.__str__() for wet_cell in wet_cells])
    ar = {}
    ar["zipped_data"] = zipped_data
    ar["dataset"] = dataset

    if request.method == 'POST':
        if 'plot_cells' in request.POST:
            visuals_form = DatasetVisualsForm(request.POST, prefix='visuals-form')
            if visuals_form.is_valid():
                number_per_page = visuals_form.cleaned_data["per_page"]
                rows = visuals_form.cleaned_data["rows"]
                min_i, max_i, max_page_number, page_number = focus_on_page(visuals_form, len(wet_cells), number_per_page=number_per_page)
                datas = [(wet_cell.cell_id, plot_cycling_direct(wet_cell.cell_id, path_to_plots=None, figsize=[5., 4.])) for wet_cell in wet_cells[min_i:max_i]]
                split_datas = [datas[i:min(len(datas), i + rows)] for i in range(0, len(datas), rows)]

                ar["visual_data"] = split_datas
                ar["visuals_form"] = visuals_form
                ar["page_number"] = page_number
                ar["max_page_number"] = max_page_number

    conditional_register(ar,
                         "visuals_form",
                         DatasetVisualsForm(prefix='visuals-form')
                         )
    return render(request, 'cell_database/view_dataset.html',ar)



def dataset_overview(request):
    ar = {}
    if request.method == 'GET':
        ar['create_dataset_form'] = CreateDatasetForm(prefix='create-dataset-form')
    if request.method == 'POST':
        create_dataset_form = CreateDatasetForm(request.POST, prefix='create-dataset-form')
        success = False
        name = ''
        if create_dataset_form.is_valid():
            name = create_dataset_form.cleaned_data['name']
            success = not Dataset.objects.filter(name=name).exists()
        if success:
            Dataset.objects.create(name=name)
            ar['message'] = "SUCCESS: created dataset with name {}".format(name)
        else:
            ar['message'] = "FAIL: dataset with name {} already existed".format(name)
    datasets = Dataset.objects.order_by('name')
    counts = [dataset.wet_cells.count() for dataset in datasets]
    zipped_data = zip(datasets, counts)
    ar['zipped_data'] = zipped_data
    return render(request, 'cell_database/dataset_overview.html', ar)