from django import forms
from django.forms import formset_factory, modelformset_factory, ModelForm, Form
from .models import *
from django.forms import BaseModelFormSet
from django.db.models import Q

def choices_molecule():
    return make_choices(
            none=True,
            no_lots=Component.objects.filter(composite_type=ELECTROLYTE),
            lots=ComponentLot.objects.filter(component__composite_type=ELECTROLYTE).exclude(lot_info=None)
        )

def coating_choices():
    return make_choices(
            no_lots= Coating.objects.all(),
            lots = CoatingLot.objects.exclude(lot_info=None),
            none=True,
            unknown=True,
        )

def material_choices():
    return make_choices(
            no_lots= Component.objects.filter(Q(composite_type=ANODE) | Q(composite_type=CATHODE)),
            lots= ComponentLot.objects.filter(
            Q(component__composite_type=ANODE) | Q(component__composite_type=CATHODE)).exclude(lot_info=None),
            none=True,
        )

def separator_material_choices():
    return make_choices(
            no_lots= Component.objects.filter(composite_type=SEPARATOR),
            lots= ComponentLot.objects.filter(component__composite_type=SEPARATOR).exclude(lot_info=None),
            none=True,
        )

def composite_choices(composite_type=None):
    return make_choices(
        no_lots=Composite.objects.filter(composite_type=composite_type),
        lots=CompositeLot.objects.filter(composite__composite_type=composite_type),
        none=True,
    )

def dry_cell_choices():
    return make_choices(
            none=True,
            no_lots=DryCell.objects.all(),
            lots=DryCellLot.objects.exclude(lot_info=None)
        )


class ElectrolyteMoleculeForm(ModelForm):
    component_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [SALT, SOLVENT, ADDITIVE],
        COMPONENT_TYPES))
    class Meta:
        model = Component
        fields = ['notes','smiles', 'proprietary','smiles_name', 'proprietary_name', 'component_type_name']


class ElectrolyteMoleculeLotForm(ModelForm):
    predefined_molecule = forms.ModelChoiceField(queryset=Component.objects.filter(composite_type=ELECTROLYTE), required=False)
    class Meta:
        model = LotInfo
        exclude = []



class ElectrodeActiveMaterialForm(ModelForm):
    coating = forms.ChoiceField(choices=coating_choices, required=False)
    composite_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [ANODE, CATHODE],
        COMPOSITE_TYPES))
    class Meta:
        model = Component
        fields = ['proprietary',
                  'proprietary_name',
                  'notes',
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
                  'composite_type_name',
                  'coating_lot_name',

                  ]

class ElectrodeActiveMaterialLotForm(ModelForm):
    predefined_active_material = forms.ModelChoiceField(queryset=Component.objects.filter(component_type=ACTIVE_MATERIAL), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class ElectrodeInactiveForm(ModelForm):
    coating = forms.ChoiceField(choices=coating_choices, required=False)
    composite_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [ANODE, CATHODE],
        COMPOSITE_TYPES))
    component_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [CONDUCTIVE_ADDITIVE, BINDER],
        COMPONENT_TYPES))
    class Meta:
        model = Component
        fields = ['smiles',
                  'smiles_name',
                  'proprietary',
                  'proprietary_name',
                  'particle_size',
                  'particle_size_name',
                  'preparation_temperature',
                  'preparation_temperature_name',
                  'notes',
                  'composite_type_name',
                  'component_type_name',
                  'coating_lot_name',
                  ]

class ElectrodeInactiveLotForm(ModelForm):
    predefined_inactive_material = forms.ModelChoiceField(queryset=Component.objects.filter(Q(component_type=CONDUCTIVE_ADDITIVE)|Q(component_type=BINDER)), required=False)
    class Meta:
        model = LotInfo
        exclude = []



class SeparatorMaterialForm(ModelForm):
    coating = forms.ChoiceField(choices=coating_choices, required=False)
    class Meta:
        model = Component
        fields = [
            'smiles',
            'proprietary',
            'smiles_name',
            'proprietary_name',
            'particle_size',
            'particle_size_name',
            'preparation_temperature',
            'preparation_temperature_name',
            'notes',
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
        fields = ['proprietary', 'proprietary_name',
                  'notes']

class ElectrolyteLotForm(ModelForm):
    predefined_electrolyte = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=ELECTROLYTE), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class ElectrolyteCompositionForm(Form):

    molecule = forms.ChoiceField(choices= choices_molecule(), required=False)
    ratio = forms.CharField(required=False)

class ElectrodeForm(ModelForm):
    composite_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [ANODE, CATHODE],
        COMPOSITE_TYPES))

    class Meta:
        model = Composite
        fields = ['proprietary', 'proprietary_name', 'notes', 'composite_type_name']


class ElectrodeLotForm(ModelForm):
    predefined_electrode = forms.ModelChoiceField(queryset=Composite.objects.filter(Q(composite_type=CATHODE)|Q(composite_type=ANODE)), required=False)
    class Meta:
        model = LotInfo
        exclude = []

class ElectrodeCompositionForm(Form):
    material = forms.ChoiceField(choices= material_choices, required=False)
    ratio = forms.CharField(required=False)


class ElectrodeGeometryForm(ModelForm):
    class Meta:
        model = ElectrodeGeometry
        exclude = []


class ElectrodeMaterialStochiometryForm(Form):
    atom = forms.ChoiceField(choices=ElectrodeMaterialStochiometry.ATOMS + [(None,'----')], required=False)
    stochiometry = forms.CharField(required=False)



class SeparatorForm(ModelForm):
    class Meta:
        model = Composite
        fields = ['proprietary','proprietary_name', 'notes', ]

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
    material = forms.ChoiceField(choices=separator_material_choices, required=False)
    ratio = forms.CharField(required=False)




class DryCellForm(ModelForm):
    anode = forms.ChoiceField(choices=composite_choices(composite_type=ANODE), required=False)
    cathode = forms.ChoiceField(choices=composite_choices(composite_type=CATHODE), required=False)
    separator = forms.ChoiceField(choices=composite_choices(composite_type=SEPARATOR), required=False)

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
    dry_cell = forms.ChoiceField(choices=dry_cell_choices, required=False)
    electrolyte = forms.ChoiceField(choices=composite_choices(composite_type=ELECTROLYTE), required=False)

    class Meta:
        model = WetCell
        fields = ['cell_id']


def initialize_mini_electrolyte(self, value=False, molecule=False, number=10):
    if value:

        self.fields['dry_cell'] = forms.ChoiceField(
            choices = dry_cell_choices, required=False)

    if molecule:
        c_molecule = choices_molecule()
    for i in range(number):
        if molecule:
            self.fields['molecule_{}'.format(i)] = forms.ChoiceField(choices = c_molecule, required=False)
        if value:
            self.fields['value_{}'.format(i)] = forms.CharField(
                required=False,
                max_length=7,
                widget=forms.TextInput(attrs={'size':7})
            )


class ElectrolyteBulkParametersForm(ElectrolyteForm):
    start_barcode = forms.IntegerField(required=False)
    end_barcode = forms.IntegerField(required=False)

    def __init__(self, *args, **kwargs):
        super(ElectrolyteBulkParametersForm, self).__init__(*args, **kwargs)
        initialize_mini_electrolyte(self, value=True, molecule=True)

    def get_molecule_fields(self):
        for i in range(10):
            yield self['molecule_{}'.format(i)]
    def get_value_fields(self):
        for i in range(10):
            yield self['value_{}'.format(i)]


class ElectrolyteBulkSingleEntryForm(ElectrolyteForm):
    barcode = forms.IntegerField(required=False)

    def __init__(self, *args, **kwargs):
        super(ElectrolyteBulkSingleEntryForm, self).__init__(*args, **kwargs)
        initialize_mini_electrolyte(self, value=True)

    def get_value_fields(self):
        for i in range(10):
            yield self['value_{}'.format(i)]



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
    molecule = forms.ChoiceField(choices=choices_molecule(),required=False)
    must_type = forms.ChoiceField(choices=MUST_TYPES, initial = MANDATORY)
    ratio = forms.FloatField(required=False)
    tolerance = forms.FloatField(required=False)


class ElectrolytePreviewForm(Form):
    electrolyte = forms.CharField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 200}), required=True)
    electrolyte_id = forms.IntegerField(widget=forms.HiddenInput(), required=True)
    exclude = forms.BooleanField(required=True)