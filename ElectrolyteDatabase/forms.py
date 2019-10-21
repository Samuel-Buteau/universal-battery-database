from django import forms
from django.forms import formset_factory, modelformset_factory, ModelForm
from .models import Molecule, Electrolyte, DryCell, Box, Component, Alias, MechanicalPouch, OtherInfo, Separator, BuildInfo, \
Cathode, CathodeActiveMaterials, Anode, AnodeActiveMaterials, AnodeFam, CathodeFam, CathodeCoating, CathodeSpecificMaterials, AnodeSpecificMaterials, \
    CellAttribute, CathodeConductiveAdditive, AnodeConductiveAdditive, VendorInfo, AnodeBinder, CathodeBinder
from django.db.models import Q

## Cell ID Range

class CellIDRangeForm(forms.Form):
    range_min = forms.IntegerField(widget=forms.TextInput(attrs={'placeholder': 'Range minimum'}), required=False)
    range_max = forms.IntegerField(widget=forms.TextInput(attrs={'placeholder': 'Range maximum'}), required=False)

CellIDRangeFormSet = formset_factory(CellIDRangeForm,extra=10)

## Specify Cell and Electrolyte

class SpecifyDryCellForm(forms.Form):
    cell_model = forms.ModelChoiceField(DryCell.objects.all(), required=False)
    cell_description = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Cell Description'}),
                                       required=False)
    cathode_coating = forms.ModelChoiceField(CathodeCoating.objects.all(),required=False)
    cell_attribute = forms.ModelChoiceField(CellAttribute.objects.all(),required=False)


class SpecifyElectrolyteForm(forms.Form):
    electrolyte_alias = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Electrolyte Alias Keywords'}), required=False)
    use_ratio = forms.BooleanField(required=False)


## Box ID Dynamic Forms

class BoxIDForm(forms.Form):
    box_id = forms.IntegerField(widget=forms.TextInput(attrs={'size': 10,'placeholder': 'Box ID'}),
                                   required=False)

BoxIDFormSet = formset_factory(BoxIDForm,extra=10)

## Returned Querysets

class CellListForm(forms.Form):
    cell_id = forms.IntegerField(required=False, widget=forms.TextInput(attrs={'size': 40,'readonly':'readonly'}))
    ignore = forms.BooleanField(required=False)

class BoxQuerysetForm(forms.Form):
    box_id = forms.IntegerField(required=False, widget=forms.TextInput(attrs={'size': 10,'readonly':'readonly'}))
    cell_model = forms.CharField(required=False, widget=forms.TextInput(attrs={'size': 30,'readonly':'readonly'}))
    ignore = forms.BooleanField(required=False)

class ElectrolyteQuerysetForm(forms.Form):
    shortstring = forms.CharField(required=False, widget=forms.TextInput(attrs={'size': 43,'readonly':'readonly'}))
    ignore = forms.BooleanField(required=False)



## Actual Anode with Specific materials

class ActualAnodeSpecificMaterialsForm(forms.Form):

    material_notes = forms.CharField(max_length=100,widget=forms.TextInput(attrs={'size': 30,'readonly':'readonly'}))
    specific_material = forms.CharField(max_length=100,widget=forms.TextInput(attrs={'size': 10,'readonly':'readonly'}))

ActualAnodeSpecificMaterialsFormSet = formset_factory(ActualAnodeSpecificMaterialsForm,extra=0)



## Actual Cathode with Specific materials

class ActualCathodeSpecificMaterialsForm(forms.Form):
    material_notes = forms.CharField(max_length=100,widget=forms.TextInput(attrs={'size': 30,'readonly':'readonly'}))
    specific_material = forms.CharField(max_length=100,widget=forms.TextInput(attrs={'size': 10,'readonly':'readonly'}))

ActualCathodeSpecificMaterialsFormSet = formset_factory(ActualCathodeSpecificMaterialsForm,extra=0)

## Electrolytes

electrolyte_number = 1

ELECTROLYTE_CHOICES = [('0', '----')]

for electrolyte in Electrolyte.objects.all():

    name = 'NO ALIASES'

    for alias in Alias.objects.all():

        if alias.electrolyte == electrolyte:

            name = alias.name

            break

    ELECTROLYTE_CHOICES.append((str(electrolyte_number), '{} - {}'.format(electrolyte.shortstring,name)))

    electrolyte_number += 1


class SpecificElectrolyteForm(forms.Form):

    electrolyte = forms.ChoiceField(choices=ELECTROLYTE_CHOICES, initial='0')

SpecificElectrolyteFormSet = formset_factory(SpecificElectrolyteForm,extra=5)


## Wet Cell Metadata

class WetCellMetadataEditForm(forms.Form):

    cell_id = forms.IntegerField(widget=forms.TextInput(attrs={'readonly':'readonly'}),required=False)
    box = forms.ModelChoiceField(Box.objects.all(),required=False)
    electrolyte = forms.ModelChoiceField(Electrolyte.objects.all(),required=False)
    exclude = forms.BooleanField(required=False)



## Registering New Molecules

RegisteredMoleculesFormSet = modelformset_factory(Molecule, fields=('name','can_be_salt','can_be_additive', 'can_be_solvent', ), extra=0, widgets={
    'name': forms.TextInput(attrs={'readonly': 'readonly'}),
    'can_be_salt':forms.CheckboxInput(attrs={'readonly':'readonly'}),
    'can_be_additive': forms.CheckboxInput(attrs={'readonly': 'readonly'}),
    'can_be_solvent': forms.CheckboxInput(attrs={'readonly': 'readonly'}),
})
RegisterMoleculeFormSet = modelformset_factory(Molecule, exclude=tuple(['vendor']), can_delete=False,extra=1)


## Search By Ratio

RatioSearchFormSet = modelformset_factory(Component, exclude=tuple(['molal', 'electrolyte', 'compound_type']),can_delete=False, extra=6)

## Alias Name Form

AliasNameFormSet = modelformset_factory(Alias,exclude=tuple(['electrolyte']),extra=1)

## Cell Component Formsets

SaltComponentFormSet = modelformset_factory(Component,exclude=tuple(['weight_percent','notes', 'electrolyte', 'compound_type']),extra=3)
SolventComponentFormSet = modelformset_factory(Component,exclude=tuple(['molal','notes', 'electrolyte', 'compound_type']),extra=6)
AbsentComponentFormSet = modelformset_factory(Component,exclude=tuple(['molal', 'notes', 'weight_percent','electrolyte', 'compound_type']),can_delete=False, extra=3,)


## Search Querysets

BoxQuerysetFormSet = formset_factory(BoxQuerysetForm, extra=0)

ElectrolyteQuerysetFormSet = formset_factory(ElectrolyteQuerysetForm, extra=0)

CellListFormSet = formset_factory(CellListForm, extra=0)


## Edit Wet Cell Metadata Formset

WetCellMetadataEditFormSet = formset_factory(WetCellMetadataEditForm, extra=0)


## Register Cathode Specific Materials

class RegisterCathodeSpecificMaterialsForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(RegisterCathodeSpecificMaterialsForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():
            self.fields[field].required = False


    class Meta:
        model = CathodeSpecificMaterials
        exclude = []


## Register New Attribute

class RegisterAttributeForm(ModelForm):

    class Meta:
        model = CellAttribute
        exclude = tuple(['dry_cells'])


## Register Anode Specific Materials

class RegisterAnodeSpecificMaterialsForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(RegisterAnodeSpecificMaterialsForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():
            self.fields[field].required = False


    class Meta:
        model = AnodeSpecificMaterials
        exclude = []


class RegisterCathodeCoatingForm(forms.Form):
    name = forms.CharField(max_length=100,required=False)

## Register Cathode Families

class RegisterCathodeFamilyForm(forms.Form):
    name = forms.CharField(max_length=100, required=False)

    def __init__(self, *args, **kwargs):
        super(RegisterCathodeFamilyForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():
            self.fields[field].required = False



class RegisterAnodeFamilyForm(forms.Form):
    name = forms.CharField(max_length=100,required=False)

    def __init__(self, *args, **kwargs):
        super(RegisterAnodeFamilyForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():
            self.fields[field].required = False



## Dry Cell CSV Model Forms


class ChooseDryCellForm(forms.Form):

    cell_model = forms.ModelChoiceField(DryCell.objects.all())

    def __init__(self, *args, **kwargs):
        super(ChooseDryCellForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False



class CellBasicsForm(ModelForm):


    class Meta:
        model = DryCell
        exclude = []


    def __init__(self, *args, **kwargs):
        super(CellBasicsForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False



class CellMechanicalPouchForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(CellMechanicalPouchForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False

    class Meta:
        model = MechanicalPouch
        exclude = ['dry_cell']

class CellOtherInfoForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(CellOtherInfoForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False


    class Meta:
        model = OtherInfo
        exclude = ['other_info_cell']


class CellSeparatorForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(CellSeparatorForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False

    class Meta:
        model = Separator
        exclude = ['dry_cell']

class CellBuildInfoForm(ModelForm):


    def __init__(self, *args, **kwargs):
        super(CellBuildInfoForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False


    class Meta:
        model = BuildInfo
        exclude = ['dry_cell']


class CellCathodeForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(CellCathodeForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False

    class Meta:
        model = Cathode
        exclude = ['dry_cell','cathode_active_materials']


class CathodeConductiveAdditiveForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(CathodeConductiveAdditiveForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False

    class Meta:
        model = CathodeConductiveAdditive
        exclude = ['cathode']

CathodeConductiveAdditiveFormSet = formset_factory(CathodeConductiveAdditiveForm, extra=3)


class AnodeConductiveAdditiveForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(AnodeConductiveAdditiveForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False

    class Meta:
        model = AnodeConductiveAdditive
        exclude = ['anode']

AnodeConductiveAdditiveFormSet = formset_factory(AnodeConductiveAdditiveForm,extra=3)



class CellCathodeActiveMaterialsForm(ModelForm):


    def __init__(self, *args, **kwargs):
        super(CellCathodeActiveMaterialsForm, self).__init__(*args, **kwargs)

        self.fields['cathode_active_1_notes'].widget.attrs.update(size='27')
        self.fields['cathode_active_2_notes'].widget.attrs.update(size='27')
        self.fields['cathode_active_3_notes'].widget.attrs.update(size='27')


        for field in self.fields.keys():

            self.fields[field].required = False

    class Meta:
        model = CathodeActiveMaterials
        exclude = tuple(['cathode','name','coating','composition'])



class VendorInfoForm(ModelForm):


    def __init__(self, *args, **kwargs):
        super(VendorInfoForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False

    class Meta:
        model = VendorInfo
        exclude = tuple(['dry_cell'])



class CathodeBinderForm(ModelForm):


    def __init__(self, *args, **kwargs):
        super(CathodeBinderForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False

    class Meta:
        model = CathodeBinder
        exclude = tuple(['cathode'])


class AnodeBinderForm(ModelForm):


    def __init__(self, *args, **kwargs):
        super(AnodeBinderForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False

    class Meta:
        model = AnodeBinder
        exclude = tuple(['anode'])



class CellAttributeForm(forms.Form):


    attribute = forms.ModelChoiceField(CellAttribute.objects.all(), required=False)

    hidden_id = forms.IntegerField(required=False, widget=forms.HiddenInput())

CellAttributeFormSet = formset_factory(CellAttributeForm, extra=3)



class LinkBoxToCellForm(forms.Form):

    box_id = forms.IntegerField(required=False, widget=forms.TextInput(attrs={'placeholder': 'Box ID'}),)
    cell_model = forms.ModelChoiceField(DryCell.objects.all(),required=False)


class CellAnodeForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(CellAnodeForm, self).__init__(*args, **kwargs)

        for field in self.fields.keys():

            self.fields[field].required = False


    class Meta:
        model = Anode
        exclude = ['dry_cell','anode_active_materials']


class CellAnodeActiveMaterialsForm(ModelForm):

    def __init__(self, *args, **kwargs):
        super(CellAnodeActiveMaterialsForm, self).__init__(*args, **kwargs)

        self.fields['anode_active_1_notes'].widget.attrs.update(size='27')
        self.fields['anode_active_2_notes'].widget.attrs.update(size='27')
        self.fields['anode_active_3_notes'].widget.attrs.update(size='27')
        self.fields['anode_active_4_notes'].widget.attrs.update(size='27')

        for field in self.fields.keys():

            self.fields[field].required = False


    class Meta:
        model = AnodeActiveMaterials
        exclude = tuple(['composition','coating','anode'])



class AnodeComponentForm(forms.Form):

    hidden_id = forms.IntegerField(required=False,widget=forms.HiddenInput())

    chemical_formula = forms.CharField(max_length=100, required=False)

    atom_ratio = forms.FloatField(required=False)


class CathodeComponentForm(forms.Form):

    hidden_id = forms.IntegerField(required=False,widget=forms.HiddenInput())

    chemical_formula = forms.CharField(max_length=100, required=False)

    atom_ratio = forms.FloatField(required=False)


CathodeComponentFormSet = formset_factory(CathodeComponentForm, extra=6)

AnodeComponentFormSet = formset_factory(AnodeComponentForm, extra=6)

