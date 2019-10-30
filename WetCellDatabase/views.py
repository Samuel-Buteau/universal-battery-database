from django.shortcuts import render, render_to_response
from django.forms import modelformset_factory, formset_factory
from django.db.models import Q
from .forms import *
from .models import *
from django import forms
import re


def define_page(request):
    ActiveMaterialCompositionFormset = modelformset_factory(
        ElectrodeMaterialStochiometry,
        exclude=['electrode_material'],
        extra=10
    )

    ElectrolyteCompositionFormset = modelformset_factory(
        RatioComponent,
        formset=BaseElectrolyteCompositionFormSet,
        fields=['ratio', 'component_type'],
        extra=10
    )

    ElectrodeCompositionFormset = modelformset_factory(
        RatioComponent,
        formset=BaseElectrodeCompositionFormSet,
        fields=['ratio', 'component_type'],
        extra=10
    )

    SeparatorCompositionFormset = modelformset_factory(
        RatioComponent,
        formset=BaseSeparatorCompositionFormSet,
        fields=['ratio', 'component_type'],
        extra=10
    )


    active_material_composition_formset = ActiveMaterialCompositionFormset()
    electrolyte_composition_formset = ElectrolyteCompositionFormset()
    electrode_composition_formset = ElectrodeCompositionFormset()
    separator_composition_formset = SeparatorCompositionFormset()

    ar = {}
    ar['active_material_composition_formset'] = active_material_composition_formset
    ar['electrolyte_composition_formset'] = electrolyte_composition_formset
    ar['electrode_composition_formset'] = electrode_composition_formset
    ar['separator_composition_formset'] = separator_composition_formset

    ar['define_molecule_form'] = ElectrolyteMoleculeForm(prefix='electrolyte-molecule-form')
    ar['define_molecule_lot_form'] = ElectrolyteMoleculeLotForm(prefix='electrolyte-molecule-lot-form')

    ar['define_coating_form'] = CoatingForm()
    ar['define_coating_lot_form'] = CoatingLotForm()

    ar['define_conductive_additive_form'] = ElectrodeConductiveAdditiveForm()
    ar['define_conductive_additive_lot_form'] = ElectrodeConductiveAdditiveLotForm()

    ar['define_active_material_form'] = ElectrodeActiveMaterialForm()
    ar['define_active_material_lot_form'] = ElectrodeActiveMaterialLotForm()

    ar['define_binder_form'] = ElectrodeBinderForm()
    ar['define_binder_lot_form'] = ElectrodeBinderLotForm()

    ar['define_separator_material_form'] = SeparatorMaterialForm()
    ar['define_separator_material_lot_form'] = SeparatorMaterialLotForm()

    ar['define_electrolyte_form'] = ElectrolyteForm()
    ar['define_electrolyte_lot_form'] = ElectrolyteLotForm()

    ar['define_electrode_form'] = ElectrodeForm()
    ar['define_electrode_lot_form'] = ElectrodeLotForm()
    ar['define_electrode_geometry_form'] = ElectrodeGeometryForm()

    ar['define_separator_form'] = SeparatorForm()
    ar['define_separator_lot_form'] = SeparatorLotForm()
    ar['define_separator_geometry_form'] = SeparatorGeometryForm()

    ar['define_dry_cell_form'] = DryCellForm()
    ar['define_dry_cell_lot_form'] = DryCellLotForm()
    ar['define_dry_cell_geometry_form'] = DryCellGeometryForm()

    ar['define_wet_cell_form'] = WetCellForm()

    if request.method == 'POST':
        if ('define_molecule' in request.POST) or ('define_molecule_lot' in request.POST):
            define_molecule_form = ElectrolyteMoleculeForm(request.POST, prefix='electrolyte-molecule-form')
            if define_molecule_form.is_valid():
                print(define_molecule_form.cleaned_data)
                if 'define_molecule' in request.POST:
                    if define_molecule_form.cleaned_data['name'] is not None:
                        # TODO(sam):handle already existing molecule
                        if not Component.objects.filter(can_be_electrolyte=True, name=define_molecule_form.cleaned_data['name']).exists():
                            Component.objects.create(
                                name = define_molecule_form.cleaned_data['name'],
                                smiles = define_molecule_form.cleaned_data['smiles'],
                                proprietary=define_molecule_form.cleaned_data['proprietary'],
                                can_be_electrolyte = True,
                                can_be_salt = define_molecule_form.cleaned_data['can_be_salt'],
                                can_be_solvent=define_molecule_form.cleaned_data['can_be_solvent'],
                                can_be_additive=define_molecule_form.cleaned_data['can_be_additive'],
                            )
                        ar['define_molecule_form'] = define_molecule_form

            if 'define_molecule_lot' in request.POST:
                define_molecule_lot_form = ElectrolyteMoleculeLotForm(request.POST, prefix='electrolyte-molecule-lot-form')
                if define_molecule_lot_form.is_valid():
                    if define_molecule_lot_form.cleaned_data['predefined_molecule'] is not None:
                        if define_molecule_lot_form.cleaned_data['lot_name'] is not None:
                            #TODO(sam): handle already existing lot
                            if not ComponentLot.objects.filter(component__can_be_electrolyte=True).exclude(lot_info=None).filter(lot_info__lot_name=define_molecule_lot_form.cleaned_data['lot_name']).exists():
                                lot_info = LotInfo(
                                    lot_name = define_molecule_lot_form.cleaned_data['lot_name'],
                                    creation_date=define_molecule_lot_form.cleaned_data['creation_date'],
                                    creator=define_molecule_lot_form.cleaned_data['creator'],
                                    vendor=define_molecule_lot_form.cleaned_data['vendor'],

                                )
                                lot_info.save()
                                ComponentLot.objects.create(
                                    component=define_molecule_lot_form.cleaned_data['predefined_molecule'],
                                    lot_info = lot_info
                                )
                    else:
                        #TODO(sam): handle the case where we create on the fly.
                        raise("not yet implemented")
                    print(define_molecule_lot_form.cleaned_data)
                    ar['define_molecule_lot_form'] = define_molecule_lot_form


        if ('define_coating' in request.POST) or ('define_coating_lot' in request.POST):
            define_coating_form = CoatingForm(request.POST)
            if define_coating_form.is_valid():
                print(define_coating_form.cleaned_data)
                ar['define_coating_form'] = define_coating_form
            if 'define_coating_lot' in request.POST:
                define_coating_lot_form = CoatingLotForm(request.POST)
                if define_coating_lot_form.is_valid():
                    print(define_coating_lot_form.cleaned_data)
                    ar['define_coating_lot_form'] = define_coating_lot_form


        elif ('define_conductive_additive' in request.POST) or ('define_conductive_additive_lot' in request.POST):
            define_conductive_additive_form = ElectrodeConductiveAdditiveForm(request.POST)
            if define_conductive_additive_form.is_valid():
                print(define_conductive_additive_form.cleaned_data)
                ar['define_conductive_additive_form'] = define_conductive_additive_form

            if 'define_conductive_additive_lot' in request.POST:
                define_conductive_additive_lot_form = ElectrodeConductiveAdditiveLotForm(request.POST)
                if define_conductive_additive_lot_form.is_valid():
                    print(define_conductive_additive_lot_form.cleaned_data)
                    ar['define_conductive_additive_lot_form'] = define_conductive_additive_lot_form

        elif ('define_binder' in request.POST) or ('define_binder_lot' in request.POST):
            define_binder_form = ElectrodeBinderForm(request.POST)
            if define_binder_form.is_valid():
                print(define_binder_form.cleaned_data)
                ar['define_binder_form'] = define_binder_form

            if 'define_binder_lot' in request.POST:
                define_binder_lot_form = ElectrodeBinderLotForm(request.POST)
                if define_binder_lot_form.is_valid():
                    print(define_binder_lot_form.cleaned_data)
                    ar['define_binder_lot_form'] = define_binder_lot_form


        elif ('define_active_material' in request.POST) or ('define_active_material_lot' in request.POST):
            define_active_material_form = ElectrodeActiveMaterialForm(request.POST)
            if define_active_material_form.is_valid():
                print(define_active_material_form.cleaned_data)
                ar['define_active_material_form'] = define_active_material_form

            electrolyte_composition_formset = ElectrolyteCompositionFormset(request.POST)
            for form in active_material_composition_formset:
                validation_step = form.is_valid()
                if validation_step:
                    print(form.cleaned_data)

            ar['active_material_composition_formset'] = active_material_composition_formset

            if 'define_active_material_lot' in request.POST:
                define_active_material_lot_form = ElectrodeActiveMaterialLotForm(request.POST)
                if define_active_material_lot_form.is_valid():
                    print(define_active_material_lot_form.cleaned_data)
                    ar['define_active_material_lot_form'] = define_active_material_lot_form


        elif ('define_separator_material' in request.POST) or ('define_separator_material_lot' in request.POST):
            define_separator_material_form = SeparatorMaterialForm(request.POST)
            if define_separator_material_form.is_valid():
                print(define_separator_material_form.cleaned_data)
                ar['define_separator_material_form'] = define_separator_material_form

            if 'define_separator_material_lot' in request.POST:
                define_separator_material_lot_form = SeparatorMaterialLotForm(request.POST)
                if define_separator_material_lot_form.is_valid():
                    print(define_separator_material_lot_form.cleaned_data)
                    ar['define_separator_material_lot_form'] = define_separator_material_lot_form

        elif ('define_electrolyte' in request.POST) or ('define_electrolyte_lot' in request.POST):
            define_electrolyte_form = ElectrolyteForm(request.POST)
            if define_electrolyte_form.is_valid():
                print(define_electrolyte_form.cleaned_data)
                ar['define_electrolyte_form'] = define_electrolyte_form

            electrolyte_composition_formset = ElectrolyteCompositionFormset(request.POST)
            for form in electrolyte_composition_formset:
                validation_step = form.is_valid()
                if validation_step:
                    print(form.cleaned_data)

            ar['electrolyte_composition_formset'] = electrolyte_composition_formset

            if 'define_electrolyte_lot' in request.POST:
                define_electrolyte_lot_form = ElectrolyteLotForm(request.POST)
                if define_electrolyte_lot_form.is_valid():
                    print(define_electrolyte_lot_form.cleaned_data)
                    ar['define_electrolyte_lot_form'] = define_electrolyte_lot_form


        elif ('define_electrode' in request.POST) or ('define_electrode_lot' in request.POST):
            define_electrode_form = ElectrodeForm(request.POST)
            if define_electrode_form.is_valid():
                print(define_electrode_form.cleaned_data)
                ar['define_electrode_form'] = define_electrode_form

            define_electrode_geometry_form = ElectrodeGeometry(request.POST)
            if define_electrode_geometry_form.is_valid():
                print(define_electrode_geometry_form.cleaned_data)
                ar['define_electrode_geometry_form'] = define_electrode_geometry_form

            electrode_composition_formset = ElectrodeCompositionFormset(request.POST)
            for form in electrode_composition_formset:
                validation_step = form.is_valid()
                if validation_step:
                    print(form.cleaned_data)

            ar['electrode_composition_formset'] = electrode_composition_formset
            if 'define_electrode_lot' in request.POST:
                define_electrode_lot_form = ElectrodeLotForm(request.POST)
                if define_electrode_lot_form.is_valid():
                    print(define_electrode_lot_form.cleaned_data)
                    ar['define_electrode_lot_form'] = define_electrode_lot_form


        elif ('define_separator' in request.POST) or ('define_separator_lot' in request.POST):
            define_separator_form = SeparatorForm(request.POST)
            if define_separator_form.is_valid():
                print(define_separator_form.cleaned_data)
                ar['define_separator_form'] = define_separator_form

            define_separator_geometry_form = SeparatorGeometry(request.POST)
            if define_separator_geometry_form.is_valid():
                print(define_separator_geometry_form.cleaned_data)
                ar['define_separator_geometry_form'] = define_separator_geometry_form

            separator_composition_formset = SeparatorCompositionFormset(request.POST)
            for form in separator_composition_formset:
                validation_step = form.is_valid()
                if validation_step:
                    print(form.cleaned_data)

            ar['separator_composition_formset'] = separator_composition_formset

            if 'define_separator_lot' in request.POST:
                define_separator_lot_form = SeparatorLotForm(request.POST)
                if define_separator_lot_form.is_valid():
                    print(define_separator_lot_form.cleaned_data)
                    ar['define_separator_lot_form'] = define_separator_lot_form

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
                if not complete_salt or not complete_additive or not complete_solvent:
                    raise('error!!!')

                all_data = []
                for form in electrolyte_composition_formset:
                    if form.is_valid:
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

                q = Q(composite_type=Composite.ELECTROLYTE)
                # prohibit molecules

                q = q & ~Q(ratiocomponent__component_lot__in=prohibited_lots)
                q = q & ~Q(ratiocomponent__component_lot__component__in=prohibited_molecules)

                total_query = Composite.objects.filter(q)

                initial = []
                for electrolyte in total_query[:1]:
                    my_initial = {
                        "electrolyte": electrolyte.shortstring,
                        "electrolyte_id": electrolyte.id,
                        "exclude": True,
                    }

                    initial.append(my_initial)
                electrolyte_preview_formset = ElectrolytePreviewFormset(initial=initial,prefix='electrolyte_preview')
                ar['electrolyte_preview_formset'] = electrolyte_preview_formset


    return render(request, 'WetCellDatabase/search_page.html', ar)

