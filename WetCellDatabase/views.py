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

    ar['define_molecule_form'] = ElectrolyteMoleculeForm()
    ar['define_molecule_lot_form'] = ElectrolyteMoleculeLotForm()

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
            define_molecule_form = ElectrolyteMoleculeForm(request.POST)
            if define_molecule_form.is_valid():
                print(define_molecule_form.cleaned_data)
                ar['define_molecule_form'] = define_molecule_form

            if 'define_molecule_lot' in request.POST:
                define_molecule_lot_form = ElectrolyteMoleculeLotForm(request.POST)
                if define_molecule_lot_form.is_valid():
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


    return render(request, 'WetCellDatabase/form_interface.html', ar)
