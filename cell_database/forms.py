from django import forms
from django.forms import formset_factory, modelformset_factory, ModelForm, Form
from .models import *
from django.forms import BaseModelFormSet
from django.db.models import Q

def molecule_choices(none=True):
    return make_choices(
            none=none,
            no_lots=Component.objects.filter(composite_type=ELECTROLYTE),
            lots=ComponentLot.objects.filter(component__composite_type=ELECTROLYTE).exclude(lot_info__isnull=True)
        )

def coating_choices(unknown=True):
    return make_choices(
            no_lots= Coating.objects.all(),
            lots = CoatingLot.objects.exclude(lot_info__isnull=True),
            unknown=unknown,
        )

def material_choices(none=True):
    return make_choices(
            no_lots= Component.objects.filter(composite_type=ELECTRODE ),
            lots= ComponentLot.objects.filter(component__composite_type=ELECTRODE).exclude(lot_info__isnull=True),
            none=none,
        )

def separator_material_choices(none=True):
    return make_choices(
            no_lots= Component.objects.filter(composite_type=SEPARATOR),
            lots= ComponentLot.objects.filter(component__composite_type=SEPARATOR).exclude(lot_info__isnull=True),
            none=none,
        )

def composite_choices(composite_type=None, none=True):
    return make_choices(
        no_lots=Composite.objects.filter(composite_type=composite_type),
        lots=CompositeLot.objects.filter(composite__composite_type=composite_type).exclude(lot_info__isnull=True),
        none=none,
    )

def dry_cell_choices(none=True):
    return make_choices(
            none=none,
            no_lots=DryCell.objects.all(),
            lots=DryCellLot.objects.exclude(lot_info__isnull=True)
        )


class DeleteForm(forms.Form):
    delete_molecules = forms.MultipleChoiceField(
        choices=lambda: molecule_choices(
            none=False
        ),
        widget=forms.CheckboxSelectMultiple(),
        required=False
    )

    delete_electrolytes = forms.MultipleChoiceField(
        choices=lambda : composite_choices(
            composite_type=ELECTROLYTE,
            none=False
        ),
        widget=forms.CheckboxSelectMultiple(),
        required = False
    )

    delete_coatings = forms.MultipleChoiceField(
        choices=lambda: coating_choices(
            unknown=False
        ),
        widget=forms.CheckboxSelectMultiple(),
        required=False
    )

    delete_materials = forms.MultipleChoiceField(
        choices=lambda : material_choices(
            none=False
        ),
        widget=forms.CheckboxSelectMultiple(),
        required = False
    )

    delete_anodes = forms.MultipleChoiceField(
        choices=lambda: composite_choices(
            composite_type=ANODE,
            none=False
        ),
        widget=forms.CheckboxSelectMultiple(),
        required=False
    )
    delete_cathodes = forms.MultipleChoiceField(
        choices=lambda: composite_choices(
            composite_type=CATHODE,
            none = False
        ),
        widget = forms.CheckboxSelectMultiple(),
        required = False
    )

    delete_separator_materials = forms.MultipleChoiceField(
        choices=lambda : separator_material_choices(
            none=False
        ),
        widget=forms.CheckboxSelectMultiple(),
        required = False
    )

    delete_separators = forms.MultipleChoiceField(
        choices=lambda: composite_choices(
            composite_type=SEPARATOR,
            none=False
        ),
        widget=forms.CheckboxSelectMultiple(),
        required=False
    )

    delete_dry_cells = forms.MultipleChoiceField(
        choices=lambda : dry_cell_choices(
            none=False
        ),
        widget=forms.CheckboxSelectMultiple(),
        required = False
    )

# Electrolyte Molecule
class ElectrolyteMoleculeForm(ModelForm):
    component_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [SALT, SOLVENT, ADDITIVE],
        COMPONENT_TYPES))
    override_target = forms.ModelChoiceField(
        queryset=Component.objects.filter(composite_type=ELECTROLYTE),
        required=False
    )

    class Meta:
        model = Component
        fields = ['notes','smiles', 'proprietary','smiles_name', 'proprietary_name', 'component_type_name']


class ElectrolyteMoleculeLotForm(ModelForm):
    predefined_molecule = forms.ModelChoiceField(queryset=Component.objects.filter(composite_type=ELECTROLYTE), required=False)
    class Meta:
        model = LotInfo
        exclude = []


# ActiveMaterial
class ElectrodeMaterialForm(ModelForm):
    coating = forms.ChoiceField(choices=coating_choices, required=False)
    composite_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [ELECTRODE],
        COMPOSITE_TYPES_2))
    component_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [CONDUCTIVE_ADDITIVE, BINDER, ACTIVE_MATERIAL],
        COMPONENT_TYPES))

    override_target = forms.ModelChoiceField(
        queryset=Component.objects.filter( Q(component_type=CONDUCTIVE_ADDITIVE) |
                                           Q(component_type=BINDER) |
                                           Q(component_type=ACTIVE_MATERIAL)),
        required=False
    )

    class Meta:
        model = Component
        fields = ['proprietary',
                  'proprietary_name',
                  'notes',
                  'smiles',
                  'smiles_name',
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
                  'component_type_name',
                  'coating_lot_name',

                  ]

class ElectrodeMaterialLotForm(ModelForm):
    predefined_active_material = forms.ModelChoiceField(queryset=Component.objects.filter(
        Q(component_type=CONDUCTIVE_ADDITIVE) |
        Q(component_type=BINDER) |
        Q(component_type=ACTIVE_MATERIAL)), required=False)
    class Meta:
        model = LotInfo
        exclude = []


#Separator Material
class SeparatorMaterialForm(ModelForm):
    coating = forms.ChoiceField(
        choices=coating_choices,
        required=False
    )

    override_target = forms.ModelChoiceField(
        queryset=Component.objects.filter(composite_type=SEPARATOR),
        required=False
    )

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
            'coating_lot_name'
        ]

class SeparatorMaterialLotForm(ModelForm):
    predefined_separator_material = forms.ModelChoiceField(
        queryset=Component.objects.filter(composite_type=SEPARATOR),
        required=False
    )
    class Meta:
        model = LotInfo
        exclude = []

# Coating
class CoatingForm(ModelForm):
    override_target = forms.ModelChoiceField(
        queryset=Coating.objects.all(),
        required=False
    )

    class Meta:
        model = Coating
        exclude = []

class CoatingLotForm(ModelForm):
    predefined_coating = forms.ModelChoiceField(
        queryset=Coating.objects.all(),
        required=False
    )
    class Meta:
        model = LotInfo
        exclude = []


# Electrolyte
class ElectrolyteForm(ModelForm):
    override_target = forms.ModelChoiceField(
        queryset=Composite.objects.filter(composite_type=ELECTROLYTE),
        required=False
    )

    class Meta:
        model = Composite
        fields = ['proprietary', 'proprietary_name',
                  'notes']


class ElectrolyteLotForm(ModelForm):
    predefined_electrolyte = forms.ModelChoiceField(
        queryset=Composite.objects.filter(composite_type=ELECTROLYTE),
        required=False
    )

    class Meta:
        model = LotInfo
        exclude = []


class ElectrolyteCompositionForm(Form):
    molecule = forms.ChoiceField(choices=molecule_choices, required=False)
    ratio = forms.CharField(required=False)


#Electrode

class ElectrodeForm(ModelForm):
    composite_type = forms.ChoiceField(choices=filter(
        lambda x: x[0] in [ANODE, CATHODE],
        COMPOSITE_TYPES))

    override_target = forms.ModelChoiceField(
        queryset=Composite.objects.filter(
            Q(composite_type=CATHODE) | Q(composite_type=ANODE)
        ),
        required=False
    )

    class Meta:
        model = Composite
        fields = ['proprietary', 'proprietary_name', 'notes', 'composite_type_name']


class ElectrodeLotForm(ModelForm):
    predefined_electrode = forms.ModelChoiceField(
        queryset=Composite.objects.filter(
            Q(composite_type=CATHODE)|Q(composite_type=ANODE)
        ),
        required=False
    )
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
    atom = forms.ChoiceField(
        choices=ElectrodeMaterialStochiometry.ATOMS + [(None,'----')],
        required=False
    )
    stochiometry = forms.CharField(required=False)



#Separator

class SeparatorForm(ModelForm):
    override_target = forms.ModelChoiceField(
        queryset=Composite.objects.filter(composite_type=SEPARATOR),
        required=False
    )

    class Meta:
        model = Composite
        fields = ['proprietary','proprietary_name', 'notes', ]

class SeparatorLotForm(ModelForm):
    predefined_separator = forms.ModelChoiceField(
        queryset=Composite.objects.filter(composite_type=SEPARATOR),
        required=False
    )
    class Meta:
        model = LotInfo
        exclude = []


class SeparatorGeometryForm(ModelForm):
    class Meta:
        model = SeparatorGeometry
        exclude = []


class SeparatorCompositionForm(Form):
    material = forms.ChoiceField(
        choices=separator_material_choices,
        required=False
    )
    ratio = forms.CharField(required=False)




class DryCellForm(ModelForm):
    anode = forms.ChoiceField(
        choices=lambda : composite_choices(composite_type=ANODE),
        required=False
    )
    cathode = forms.ChoiceField(
        choices=lambda : composite_choices(composite_type=CATHODE),
        required=False
    )
    separator = forms.ChoiceField(
        choices=lambda : composite_choices(composite_type=SEPARATOR),
        required=False
    )
    override_target = forms.ModelChoiceField(
        queryset=DryCell.objects.all(),
        required=False
    )

    class Meta:
        model = DryCell
        exclude = ['geometry', 'anode', 'cathode', 'separator']

class DryCellLotForm(ModelForm):
    predefined_dry_cell = forms.ModelChoiceField(
        queryset=DryCell.objects.all(),
        required=False
    )
    class Meta:
        model = LotInfo
        exclude = []


class DryCellGeometryForm(ModelForm):
    class Meta:
        model = DryCellGeometry
        exclude = []


class WetCellForm(Form):
    cell_id = forms.IntegerField(required=False)
    dry_cell = forms.ChoiceField(
        choices=dry_cell_choices,
        required=False
    )
    electrolyte = forms.ChoiceField(
        choices=lambda : composite_choices(composite_type=ELECTROLYTE),
        required=False
    )



class WetCellParametersForm(WetCellForm):
    start_cell_id = forms.IntegerField(required=False)
    end_cell_id = forms.IntegerField(required=False)
    number_of_cell_ids = forms.IntegerField(required=False)

    override_existing = forms.BooleanField(required=False)

    def __init__(self, *args, **kwargs):
        super(WetCellParametersForm, self).__init__(*args, **kwargs)


def initialize_mini_electrolyte(self, value=False, molecule=False, number=10, dry_cell=True):
    if dry_cell:
        self.fields['dry_cell'] = forms.ChoiceField(
            choices = dry_cell_choices, required=False)

    if molecule:
        c_molecule = molecule_choices
    for i in range(number):
        if molecule:
            self.fields['molecule_{}'.format(i)] = forms.ChoiceField(
                choices = c_molecule,
                required=False
            )
        if value:
            self.fields['value_{}'.format(i)] = forms.CharField(
                required=False,
                max_length=7,
                widget=forms.TextInput(attrs={'size':7})
            )


class ElectrolyteBulkParametersForm(ElectrolyteForm):
    start_cell_id = forms.IntegerField(required=False)
    end_cell_id = forms.IntegerField(required=False)
    number_of_cell_ids = forms.IntegerField(required=False)
    override_existing = forms.BooleanField(required=False)

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
    cell_id = forms.IntegerField(required=False)

    def __init__(self, *args, **kwargs):
        super(ElectrolyteBulkSingleEntryForm, self).__init__(*args, **kwargs)
        initialize_mini_electrolyte(self, value=True, dry_cell=False)

    def get_value_fields(self):
        for i in range(10):
            yield self['value_{}'.format(i)]



class SearchElectrolyteForm(Form):
    complete_salt = forms.BooleanField(initial=False, required=False)
    complete_solvent = forms.BooleanField(initial=False, required=False)
    complete_additive = forms.BooleanField(initial=False, required=False)
    relative_tolerance = forms.FloatField(initial=5., help_text='the default tolerance in percentage.')
    proprietary_flag = forms.BooleanField(initial=False, required=False)
    notes = forms.CharField(required=False)


class SearchElectrolyteComponentForm(Form):
    MANDATORY = 'ma'
    PROHIBITED = 'pr'
    ALLOWED = 'al'
    MUST_TYPES = [
        (MANDATORY, 'mandatory'),
        (PROHIBITED, 'prohibited'),
        (ALLOWED, 'allowed'),
    ]

    molecule = forms.ChoiceField(choices=molecule_choices, required=False)
    must_type = forms.ChoiceField(choices=MUST_TYPES, initial = MANDATORY)
    ratio = forms.FloatField(required=False)
    tolerance = forms.FloatField(required=False)


class ElectrolytePreviewForm(Form):
    electrolyte = forms.CharField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 200}), required=True)
    electrolyte_id = forms.IntegerField(widget=forms.HiddenInput(), required=True)
    exclude = forms.BooleanField(required=True)

class SearchGenericNamedScalarForm(Form):
    name = forms.CharField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 20}), required=False)
    scalar = forms.FloatField(required=False)
    tolerance = forms.FloatField(required=False)
    exclude_missing = forms.BooleanField(initial=False,required=False)

class SearchDryCellForm(Form):
    notes = forms.CharField(required=False)
    relative_tolerance = forms.FloatField(initial=5., help_text='the default tolerance in percentage.')
    proprietary = forms.BooleanField(initial=False, required=False)
    geometry_category = forms.MultipleChoiceField(choices=DryCellGeometry.GEO_TYPES, required=False)
    geometry_category_exclude_missing = forms.BooleanField(initial=False, required=False)
    cathode = forms.MultipleChoiceField(
        choices=lambda : make_choices(
            no_lots=Composite.objects.filter(composite_type=CATHODE),
            lots=[],
            none=False,
        ),
        required=False,
        widget=forms.SelectMultiple(attrs = {'size':10})
    )
    cathode_exclude_missing = forms.BooleanField(initial=False, required=False)
    anode = forms.MultipleChoiceField(
        choices=lambda: make_choices(
            no_lots=Composite.objects.filter(composite_type=ANODE),
            lots=[],
            none=False,
        ),
        required=False
    )
    anode_exclude_missing = forms.BooleanField(initial=False, required=False)
    separator = forms.MultipleChoiceField(
        choices=lambda: make_choices(
            no_lots=Composite.objects.filter(composite_type=SEPARATOR),
            lots=[],
            none=False,
        ),
        required=False
    )
    separator_exclude_missing = forms.BooleanField(initial=False, required=False)

class SearchWetCellForm(Form):
    page_number = forms.IntegerField(required=False)
    def set_page_number(self, page_number):
        data = self.data.copy()
        data["page_number"] = page_number
        self.data = data

class DatasetForm(Form):
    dataset = forms.ModelChoiceField(queryset=Dataset.objects.all(), required=False)

class WetCellPreviewForm(Form):
    wet_cell = forms.CharField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 200}), required=False)
    cell_id = forms.IntegerField(widget=forms.HiddenInput(), required=False)
    exclude = forms.BooleanField(required=False)

class CreateDatasetForm(Form):
    name = forms.CharField(required=False)


#TODO(sam): create a superclass that has all of this stuff instead of duplicating
class DatasetVisualsForm(Form):
    page_number = forms.IntegerField(initial=1, required=False)
    per_page = forms.IntegerField(initial=25, required=False)
    rows = forms.IntegerField(initial=5, required=False)

    def set_page_number(self, page_number):
        data = self.data.copy()
        data["page_number"] = page_number
        self.data = data