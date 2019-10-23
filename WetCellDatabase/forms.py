from django import forms
from django.forms import formset_factory, modelformset_factory, ModelForm
from .models import *
from django.forms import BaseModelFormSet
from django.db.models import Q

class ElectrolyteMoleculeForm(ModelForm):
    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary', 'can_be_salt', 'can_be_solvent', 'can_be_additive']


class ElectrolyteMoleculeLotForm(ModelForm):
    predefined_molecule = forms.ModelChoiceField(queryset=Component.objects.filter(can_be_electrolyte=True))
    class Meta:
        model = LotInfo
        exclude = []



class ElectrodeActiveMaterialForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all())
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None))

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
    predefined_active_material = forms.ModelChoiceField(queryset=Component.objects.filter(can_be_active_material=True))
    class Meta:
        model = LotInfo
        exclude = []


class ElectrodeConductiveAdditiveForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all())
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None))

    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary',
                  'can_be_cathode','can_be_anode',
                   'particle_size', 'preparation_temperature',
                   'notes'
                  ]

class ElectrodeConductiveAdditiveLotForm(ModelForm):
    predefined_conductive_additive = forms.ModelChoiceField(queryset=Component.objects.filter(can_be_conductive_additive=True))
    class Meta:
        model = LotInfo
        exclude = []


class ElectrodeBinderForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all())
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None))

    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary',
                  'can_be_cathode','can_be_anode',
                   'particle_size', 'preparation_temperature',
                   'notes'
                  ]

class ElectrodeBinderLotForm(ModelForm):
    predefined_binder = forms.ModelChoiceField(queryset=Component.objects.filter(can_be_binder=True))
    class Meta:
        model = LotInfo
        exclude = []


class SeparatorMaterialForm(ModelForm):
    coating = forms.ModelChoiceField(queryset=Coating.objects.all())
    coating_lot = forms.ModelChoiceField(queryset=CoatingLot.objects.exclude(lot_info=None))

    class Meta:
        model = Component
        fields = ['name', 'smiles', 'proprietary',
                   'particle_size', 'preparation_temperature',
                   'notes'
                  ]

class SeparatorMaterialLotForm(ModelForm):
    predefined_separator_material = forms.ModelChoiceField(queryset=Component.objects.filter(can_be_separator=True))
    class Meta:
        model = LotInfo
        exclude = []


class CoatingForm(ModelForm):
    class Meta:
        model = Coating
        exclude = []

class CoatingLotForm(ModelForm):
    predefined_coating = forms.ModelChoiceField(queryset=Coating.objects.all())
    class Meta:
        model = LotInfo
        exclude = []

class ElectrolyteForm(ModelForm):
    class Meta:
        model = Composite
        fields = ['proprietary', 'proprietary_name']

class ElectrolyteLotForm(ModelForm):
    predefined_electrolyte = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.ELECTROLYTE))
    class Meta:
        model = LotInfo
        exclude = []


class BaseElectrolyteCompositionFormSet(BaseModelFormSet):
    def add_fields(self, form, index):
        super().add_fields(form, index)
        form.fields["molecule"] =forms.ModelChoiceField(queryset=Component.objects.filter(can_be_electrolyte=True))
        form.fields["molecule_lot"] = forms.ModelChoiceField(queryset=ComponentLot.objects.filter(component__can_be_electrolyte=True).exclude(lot_info=None))
        form.fields["component_type"] = forms.ChoiceField(choices=filter(
            lambda x: x[0] in [RatioComponent.SALT,RatioComponent.SOLVENT,RatioComponent.ADDITIVE],
            RatioComponent.COMPONENT_TYPES)
        )

class ElectrodeForm(ModelForm):
    class Meta:
        model = Composite
        fields = ['proprietary', 'proprietary_name']


class ElectrodeLotForm(ModelForm):
    predefined_electrode = forms.ModelChoiceField(queryset=Composite.objects.filter(Q(composite_type=Composite.CATHODE)|Q(composite_type=Composite.ANODE)))
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
        form.fields["material"] =forms.ModelChoiceField(queryset=Component.objects.filter(Q(can_be_cathode=True)|Q(can_be_anode=True)))
        form.fields["material_lot"] = forms.ModelChoiceField(queryset=ComponentLot.objects.filter(Q(component__can_be_cathode=True)|Q(component__can_be_anode=True)).exclude(lot_info=None))
        form.fields["component_type"] = forms.ChoiceField(choices=filter(
            lambda x: x[0] in [RatioComponent.ACTIVE_MATERIAL,RatioComponent.CONDUCTIVE_ADDITIVE,RatioComponent.BINDER],
            RatioComponent.COMPONENT_TYPES)
        )



class SeparatorForm(ModelForm):
    class Meta:
        model = Composite
        fields = ['proprietary', 'proprietary_name']

class SeparatorLotForm(ModelForm):
    predefined_separator = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.SEPARATOR))
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
        form.fields["material"] =forms.ModelChoiceField(queryset=Component.objects.filter(can_be_separator=True))
        form.fields["material_lot"] = forms.ModelChoiceField(queryset=ComponentLot.objects.filter(component__can_be_separator=True).exclude(lot_info=None))



class DryCellForm(ModelForm):
    anode = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.ANODE))
    anode_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=Composite.ANODE))

    cathode = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.CATHODE))
    cathode_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=Composite.CATHODE))

    separator = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.SEPARATOR))
    separator_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=Composite.SEPARATOR))

    class Meta:
        model = DryCell
        exclude = ['geometry', 'anode', 'cathode', 'separator']

class DryCellLotForm(ModelForm):
    predefined_drycell = forms.ModelChoiceField(queryset=DryCell.objects.all())
    class Meta:
        model = LotInfo
        exclude = []


class DryCellGeometryForm(ModelForm):
    class Meta:
        model = DryCellGeometry
        exclude = []


class WetCellForm(ModelForm):

    dry_cell = forms.ModelChoiceField(queryset=DryCell.objects.all())
    dry_cell_lot = forms.ModelChoiceField(
        queryset=DryCellLot.objects.all())

    electrolyte = forms.ModelChoiceField(queryset=Composite.objects.filter(composite_type=Composite.ELECTROLYTE))
    electrolyte_lot = forms.ModelChoiceField(queryset=CompositeLot.objects.filter(composite__composite_type=Composite.ELECTROLYTE))

    class Meta:
        model = WetCell
        fields = ['cell_id']
