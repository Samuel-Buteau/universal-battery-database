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
        if content in ['electrolyte', 'electrode', 'separator']:
            electrode_geometry = None
            separator_geometry = None
            components = None
            components_lot = None

            if content == 'electrolyte':
                simple_form = ElectrolyteForm(request.POST, prefix='electrolyte-form')
                component_string = 'molecule'
            elif content == 'electrode':
                simple_form = ElectrodeForm(request.POST, prefix='electrode-form')
                component_string = 'material'
                define_electrode_geometry_form = ElectrodeGeometryForm(request.POST, prefix='electrode-geometry-form')
            elif content == 'separator':
                simple_form = SeparatorForm(request.POST, prefix='separator-form')
                component_string = 'material'
                define_separator_geometry_form = SeparatorGeometryForm(request.POST, prefix='separator-geometry-form')


            simple_form_string = 'define_{}_form'.format(content)
            composition_formset_string = '{}_composition_formset'.format(content)
            component_lot_string = '{}_lot'.format(component_string)

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
                    name=simple_form.cleaned_data['name']
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
                            if not component_string in form.cleaned_data.keys() or not component_lot_string in form.cleaned_data.keys():
                                continue
                            if form.cleaned_data['ratio'] is None or form.cleaned_data['ratio'] <= 0:
                                continue
                            if form.cleaned_data[component_string] is not None and form.cleaned_data[component_string].composite_type==my_composite.composite_type:
                                components.append(
                                    {
                                        'component': form.cleaned_data[component_string],
                                        'ratio': form.cleaned_data['ratio']
                                    }
                                )
                            elif form.cleaned_data[component_lot_string] is not None and form.cleaned_data[component_lot_string].composite.composite_type==my_composite.composite_type:
                                components_lot.append(
                                    {
                                        'component_lot': form.cleaned_data[component_lot_string],
                                        'ratio': form.cleaned_data['ratio']
                                    }
                                )

            return my_composite.define_if_possible(
                components=components,
                components_lot=components_lot,
                electrode_geometry=electrode_geometry,
                separator_geometry=separator_geometry,
            )

        if content in ['active_material','molecule','inactive','separator_material']:
            atoms = None
            if content == 'active_material':
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
                    name = simple_form.cleaned_data['name'],
                    component_type=component_type,
                    component_type_name=component_type_name,
                    composite_type=composite_type,
                    composite_type_name=composite_type_name,
                )

                if content != 'molecule':
                    my_component.coating_lot = get_lot(
                        simple_form.cleaned_data['coating'],
                        simple_form.cleaned_data['coating_lot'],
                        type='coating'
                    )
                    my_component.particle_size = simple_form.cleaned_data['particle_size']
                    my_component.particle_size_name = simple_form.cleaned_data['particle_size_name']
                    my_component.preparation_temperature = simple_form.cleaned_data['preparation_temperature']
                    my_component.preparation_temperature_name = simple_form.cleaned_data[
                                                       'preparation_temperature_name']
                    my_component.notes = simple_form.cleaned_data['notes']
                    my_component.notes_name = simple_form.cleaned_data['notes_name']
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

            else:
                my_content = define_simple(post, content=content)

            if define_lot_form.cleaned_data['lot_name'] is not None and my_content is not None:
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
                    lot_info = LotInfo(
                        lot_name=define_lot_form.cleaned_data['lot_name'],
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
        for m in ['molecule', 'coating', 'inactive', 'electrolyte', 'active_material', 'separator_material','electrode','separator']:
            if ('define_{}'.format(m) in request.POST) or ('define_{}_lot'.format(m) in request.POST):
                if 'define_{}'.format(m) in request.POST:
                    print(define_simple(request.POST, content=m))
                if 'define_{}_lot'.format(m) in request.POST:
                    define_lot(request.POST, content=m)

        if ('define_dry_cell' in request.POST) or ('define_dry_cell_lot' in request.POST):
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

