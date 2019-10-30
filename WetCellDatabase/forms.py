from django import forms
from django.forms import formset_factory, modelformset_factory, ModelForm, Form
from .models import *
from django.forms import BaseModelFormSet
from django.db.models import Q


class MoleculeChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        return "{}".format(obj.name)


class MoleculeLotChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        return "{} ({})".format(obj.lot_info.lot_name, obj.component.name)



class ElectrolyteMoleculeForm(ModelForm):
    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary', 'can_be_salt', 'can_be_solvent', 'can_be_additive']


class ElectrolyteMoleculeLotForm(ModelForm):
    predefined_molecule = MoleculeChoiceField(queryset=Component.objects.filter(can_be_electrolyte=True), required=False)
    class Meta:
        model = LotInfo
        exclude = []



class ElectrodeActiveMaterialForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all(), required=False)
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None), required=False)

    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary',
                  'can_be_cathode','can_be_anode',
                   'particle_size',
                  'single_crystal',
                  'turbostratic_misalignment',
                  'preparation_temperature',
                  'natural',
                  'core_shell',
                   'notes'
                  ]

class ElectrodeActiveMaterialLotForm(ModelForm):
    predefined_active_material = forms.ModelChoiceField(queryset=Component.objects.filter(can_be_active_material=True), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class ElectrodeConductiveAdditiveForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all(), required=False)
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None), required=False)

    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary',
                  'can_be_cathode','can_be_anode',
                   'particle_size', 'preparation_temperature',
                   'notes'
                  ]

class ElectrodeConductiveAdditiveLotForm(ModelForm):
    predefined_conductive_additive = forms.ModelChoiceField(queryset=Component.objects.filter(can_be_conductive_additive=True), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class ElectrodeBinderForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all(), required=False)
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None), required=False)

    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary',
                  'can_be_cathode','can_be_anode',
                   'particle_size', 'preparation_temperature',
                   'notes'
                  ]

class ElectrodeBinderLotForm(ModelForm):
    predefined_binder = forms.ModelChoiceField(queryset=Component.objects.filter(can_be_binder=True), required=False)
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
    predefined_separator_material = forms.ModelChoiceField(queryset=Component.objects.filter(can_be_separator=True), required=False)
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
        fields = ['proprietary', 'proprietary_name']

class ElectrolyteLotForm(ModelForm):
    predefined_electrolyte = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.ELECTROLYTE), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class BaseElectrolyteCompositionFormSet(BaseModelFormSet):
    def add_fields(self, form, index):
        super().add_fields(form, index)
        form.fields["molecule"] = MoleculeChoiceField(queryset=Component.objects.filter(can_be_electrolyte=True), required=False)
        form.fields["molecule_lot"] = MoleculeLotChoiceField(queryset=ComponentLot.objects.filter(component__can_be_electrolyte=True).exclude(lot_info=None), required=False)
        form.fields["component_type"] = forms.ChoiceField(choices=filter(
            lambda x: x[0] in [RatioComponent.SALT,RatioComponent.SOLVENT,RatioComponent.ADDITIVE],
            RatioComponent.COMPONENT_TYPES)
            , required=False
        )

class ElectrodeForm(ModelForm):
    class Meta:
        model = Composite
        fields = ['proprietary', 'proprietary_name']


class ElectrodeLotForm(ModelForm):
    predefined_electrode = forms.ModelChoiceField(queryset=Composite.objects.filter(Q(composite_type=Composite.CATHODE)|Q(composite_type=Composite.ANODE)), required=False)
    class Meta:
        model = LotInfo
        exclude = []

class ElectrodeGeometryForm(ModelForm):
    class Meta:
        model = ElectrodeGeometry
        exclude = []


class BaseElectrodeCompositionFormSet(BaseModelFormSet):
    def add_fields(self, form, index):
        super().add_fields(form, index)
        form.fields["material"] =forms.ModelChoiceField(queryset=Component.objects.filter(Q(can_be_cathode=True)|Q(can_be_anode=True)), required=False)
        form.fields["material_lot"] = forms.ModelChoiceField(queryset=ComponentLot.objects.filter(Q(component__can_be_cathode=True)|Q(component__can_be_anode=True)).exclude(lot_info=None), required=False)
        form.fields["component_type"] = forms.ChoiceField(choices=filter(
            lambda x: x[0] in [RatioComponent.ACTIVE_MATERIAL,RatioComponent.CONDUCTIVE_ADDITIVE,RatioComponent.BINDER],
            RatioComponent.COMPONENT_TYPES)
            , required=False
        )



class SeparatorForm(ModelForm):
    class Meta:
        model = Composite
        fields = ['proprietary', 'proprietary_name']

class SeparatorLotForm(ModelForm):
    predefined_separator = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.SEPARATOR), required=False)
    class Meta:
        model = LotInfo
        exclude = []


class SeparatorGeometryForm(ModelForm):
    class Meta:
        model = SeparatorGeometry
        exclude = []


class BaseSeparatorCompositionFormSet(BaseModelFormSet):
    def add_fields(self, form, index):
        super().add_fields(form, index)
        form.fields["material"] =forms.ModelChoiceField(queryset=Component.objects.filter(can_be_separator=True), required=False)
        form.fields["material_lot"] = forms.ModelChoiceField(queryset=ComponentLot.objects.filter(component__can_be_separator=True).exclude(lot_info=None), required=False)



class DryCellForm(ModelForm):
    anode = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.ANODE), required=False)
    anode_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=Composite.ANODE), required=False)

    cathode = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.CATHODE), required=False)
    cathode_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=Composite.CATHODE), required=False)

    separator = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.SEPARATOR), required=False)
    separator_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=Composite.SEPARATOR), required=False)

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

    electrolyte = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.ELECTROLYTE), required=False)
    electrolyte_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=Composite.ELECTROLYTE), required=False)

    class Meta:
        model = WetCell
        fields = ['cell_id']



class SearchElectrolyteForm(Form):
    complete_salt = forms.BooleanField(initial=True)
    complete_solvent = forms.BooleanField(initial=True)
    complete_additive = forms.BooleanField(initial=True)
    relative_tolerance = forms.FloatField(initial=5., help_text='the default tolerance in percentage.')

class SearchElectrolyteComponentForm(Form):
    MANDATORY = 'ma'
    PROHIBITED = 'pr'
    ALLOWED = 'al'
    MUST_TYPES = [
        (MANDATORY, 'mandatory'),
        (PROHIBITED, 'prohibited'),
        (ALLOWED, 'allowed'),
    ]
    molecule = MoleculeChoiceField(queryset=Component.objects.filter(can_be_electrolyte=True),required=False)
    molecule_lot = MoleculeLotChoiceField(
        queryset=ComponentLot.objects.filter(component__can_be_electrolyte=True).exclude(lot_info=None),required=False)
    component_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [RatioComponent.SALT, RatioComponent.SOLVENT, RatioComponent.ADDITIVE],
        RatioComponent.COMPONENT_TYPES)
    )
    must_type = forms.ChoiceField(choices=MUST_TYPES, initial = MANDATORY)
    ratio = forms.FloatField(required=False)
    tolerance = forms.FloatField(required=False)


class ElectrolytePreviewForm(Form):
    electrolyte = forms.CharField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 200}), required=True)
    electrolyte_id = forms.IntegerField(widget=forms.HiddenInput(), required=True)
    exclude = forms.BooleanField(required=True)