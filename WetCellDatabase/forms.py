from django import forms
from django.forms import formset_factory, modelformset_factory, ModelForm, Form
from .models import *
from django.forms import BaseModelFormSet
from django.db.models import Q





class ElectrolyteMoleculeForm(ModelForm):
    component_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [SALT, SOLVENT, ADDITIVE],
        COMPONENT_TYPES))
    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary', 'component_type_name']


class ElectrolyteMoleculeLotForm(ModelForm):
    predefined_molecule = forms.ModelChoiceField(queryset=Component.objects.filter(composite_type=ELECTROLYTE), required=False)
    class Meta:
        model = LotInfo
        exclude = []



class ElectrodeActiveMaterialForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all(), required=False)
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None), required=False)
    composite_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [ANODE, CATHODE],
        COMPOSITE_TYPES))
    class Meta:
        model = Component
        fields = ['name', 'proprietary',
                  'particle_size',
                  'particle_size_name',

                  'single_crystal',
                  'single_crystal_name',

                  'turbostratic_misalignment',
                  'turbostratic_misalignment_name',

                  'preparation_temperature',
                  'preparation_temperature_name',

                  'natural',
                  'natural_name',

                  'notes',
                  'notes_name',

                  'composite_type_name',
                  
                  'coating_lot_name',

                  ]

class ElectrodeActiveMaterialLotForm(ModelForm):
    predefined_active_material = forms.ModelChoiceField(queryset=Component.objects.filter(component_type=ACTIVE_MATERIAL), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class ElectrodeInactiveForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all(), required=False)
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None), required=False)
    composite_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [ANODE, CATHODE],
        COMPOSITE_TYPES))
    component_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [CONDUCTIVE_ADDITIVE, BINDER],
        COMPONENT_TYPES))

    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary',

                  'particle_size',
                  'particle_size_name',

                  'preparation_temperature',
                  'preparation_temperature_name',

                  'notes',
                  'notes_name',

                  'composite_type_name',
                  'component_type_name',

                  'coating_lot_name',

                  ]

class ElectrodeInactiveLotForm(ModelForm):
    predefined_conductive_additive = forms.ModelChoiceField(queryset=Component.objects.filter(Q(component_type=CONDUCTIVE_ADDITIVE)|Q(component_type=BINDER)), required=False)
    class Meta:
        model = LotInfo
        exclude = []



class SeparatorMaterialForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all(), required=False)
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None), required=False)
    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary',
                   'particle_size', 'preparation_temperature',
                   'notes'
                  ]

class SeparatorMaterialLotForm(ModelForm):
    predefined_separator_material = forms.ModelChoiceField(queryset=Component.objects.filter(composite_type=SEPARATOR), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class CoatingForm(ModelForm):
    class Meta:
        model = Coating
        exclude = []

class CoatingLotForm(ModelForm):
    predefined_coating = forms.ModelChoiceField(queryset=Coating.objects.all(), required=False)
    class Meta:
        model = LotInfo
        exclude = []

class ElectrolyteForm(ModelForm):
    class Meta:
        model = Composite
        fields = ['proprietary', 'name']

class ElectrolyteLotForm(ModelForm):
    predefined_electrolyte = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=ELECTROLYTE), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class ElectrolyteCompositionForm(Form):
    molecule = forms.ModelChoiceField(queryset=Component.objects.filter(composite_type=ELECTROLYTE), required=False)
    molecule_lot = forms.ModelChoiceField(queryset=ComponentLot.objects.filter(component__composite_type=ELECTROLYTE).exclude(lot_info=None), required=False)
    ratio = forms.FloatField(required=False)

class ElectrodeForm(ModelForm):
    composite_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [ANODE, CATHODE],
        COMPOSITE_TYPES))

    class Meta:
        model = Composite
        fields = ['proprietary', 'name']


class ElectrodeLotForm(ModelForm):
    predefined_electrode = forms.ModelChoiceField(queryset=Composite.objects.filter(Q(composite_type=CATHODE)|Q(composite_type=ANODE)), required=False)
    class Meta:
        model = LotInfo
        exclude = []

class ElectrodeCompositionForm(Form):
    material = forms.ModelChoiceField(queryset=Component.objects.filter(Q(composite_type=ANODE)|Q(composite_type=CATHODE)), required=False)
    material_lot = forms.ModelChoiceField(queryset=ComponentLot.objects.filter(Q(component__composite_type=ANODE)|Q(component__composite_type=CATHODE)).exclude(lot_info=None), required=False)
    ratio = forms.FloatField(required=False)


class ElectrodeGeometryForm(ModelForm):
    class Meta:
        model = ElectrodeGeometry
        exclude = []


class ElectrodeMaterialStochiometryForm(Form):
    atom = forms.ChoiceField(choices=ElectrodeMaterialStochiometry.ATOMS + [(None,'-------')], required=False)
    stochiometry = forms.CharField(required=False)



class SeparatorForm(ModelForm):
    class Meta:
        model = Composite
        fields = ['proprietary', 'name']

class SeparatorLotForm(ModelForm):
    predefined_separator = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=SEPARATOR), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class SeparatorGeometryForm(ModelForm):
    class Meta:
        model = SeparatorGeometry
        exclude = []


class SeparatorCompositionForm(Form):
    material = forms.ModelChoiceField(queryset=Component.objects.filter(composite_type=SEPARATOR), required=False)
    material_lot = forms.ModelChoiceField(queryset=ComponentLot.objects.filter(component__composite_type=SEPARATOR).exclude(lot_info=None), required=False)
    ratio = forms.FloatField(required=False)




class DryCellForm(ModelForm):
    anode = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=ANODE), required=False)
    anode_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=ANODE), required=False)

    cathode = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=CATHODE), required=False)
    cathode_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=CATHODE), required=False)

    separator = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=SEPARATOR), required=False)
    separator_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=SEPARATOR), required=False)

    class Meta:
        model = DryCell
        exclude = ['geometry', 'anode', 'cathode', 'separator']

class DryCellLotForm(ModelForm):
    predefined_drycell = forms.ModelChoiceField(queryset=DryCell.objects.all(), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class DryCellGeometryForm(ModelForm):
    class Meta:
        model = DryCellGeometry
        exclude = []


class WetCellForm(ModelForm):
    dry_cell = forms.ModelChoiceField(queryset=DryCell.objects.all(), required=False)
    dry_cell_lot = forms.ModelChoiceField(queryset=DryCellLot.objects.all(), required=False)

    electrolyte = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=ELECTROLYTE), required=False)
    electrolyte_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=ELECTROLYTE), required=False)

    class Meta:
        model = WetCell
        fields = ['cell_id']



class SearchElectrolyteForm(Form):
    complete_salt = forms.BooleanField(initial=True, required=False)
    complete_solvent = forms.BooleanField(initial=True, required=False)
    complete_additive = forms.BooleanField(initial=True, required=False)
    relative_tolerance = forms.FloatField(initial=5., help_text='the default tolerance in percentage.')
    proprietary_flag = forms.BooleanField(initial=False, required=False)
    proprietary_search = forms.CharField(required=False)

class SearchElectrolyteComponentForm(Form):
    MANDATORY = 'ma'
    PROHIBITED = 'pr'
    ALLOWED = 'al'
    MUST_TYPES = [
        (MANDATORY, 'mandatory'),
        (PROHIBITED, 'prohibited'),
        (ALLOWED, 'allowed'),
    ]
    molecule = forms.ModelChoiceField(queryset=Component.objects.filter(composite_type=ELECTROLYTE),required=False)
    molecule_lot = forms.ModelChoiceField(
        queryset=ComponentLot.objects.filter(component__composite_type=ELECTROLYTE).exclude(lot_info=None),required=False)
    must_type = forms.ChoiceField(choices=MUST_TYPES, initial = MANDATORY)
    ratio = forms.FloatField(required=False)
    tolerance = forms.FloatField(required=False)


class ElectrolytePreviewForm(Form):
    electrolyte = forms.CharField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 200}), required=True)
    electrolyte_id = forms.IntegerField(widget=forms.HiddenInput(), required=True)
    exclude = forms.BooleanField(required=True)