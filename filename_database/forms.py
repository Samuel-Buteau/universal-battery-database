from django import forms
from filename_database.models import *
from datetime import date
from django.forms import BaseModelFormSet

class ParserFieldForm(forms.Form):

    experiment_type_search = forms.BooleanField(required=False)
    experiment_type = forms.ModelChoiceField(queryset=ExperimentType.objects.all(), required = False)

    student_search = forms.BooleanField(required=False)
    student = forms.CharField(required=False)

    cell_id_search = forms.BooleanField(required=False)
    cell_id = forms.IntegerField(required=False)

    start_cycle_search = forms.BooleanField(required=False)
    start_cycle = forms.IntegerField(required=False)

    voltage_search = forms.BooleanField(required=False)
    voltage = forms.FloatField(required=False)

    temperature_search = forms.BooleanField(required=False)
    temperature = forms.IntegerField(required=False)

    charger_search = forms.BooleanField(required = False)
    charger = forms.CharField(max_length=5, required = False)

    AC_search = forms.BooleanField(required=False)
    AC = forms.IntegerField(required=False)

    version_number_search = forms.BooleanField(required=False)
    version_number = forms.IntegerField(required=False)

    drive_profile_search = forms.BooleanField(required=False)
    drive_profile = forms.ModelChoiceField(queryset=ChargerDriveProfile.objects.all(), required = False)

    drive_profile_x_numerator_search = forms.BooleanField(required=False)
    drive_profile_x_numerator = forms.IntegerField(required=False)

    drive_profile_x_denominator_search = forms.BooleanField(required=False)
    drive_profile_x_denominator = forms.IntegerField(required=False)

    drive_profile_y_numerator_search = forms.BooleanField(required=False)
    drive_profile_y_numerator = forms.IntegerField(required=False)

    drive_profile_y_denominator_search = forms.BooleanField(required=False)
    drive_profile_y_denominator = forms.IntegerField(required=False)

    drive_profile_z_search = forms.BooleanField(required=False)
    drive_profile_z = forms.FloatField(required=False)

    test_date_search = forms.BooleanField(required=False)
    test_date = forms.DateField(widget=forms.SelectDateWidget(empty_label="--------", years=range(2000,date.today().year + 1)), required=False)



class SearchForm(forms.Form):

    filename1_search = forms.BooleanField(required=False)
    filename1 = forms.CharField(required=False)

    filename2_search = forms.BooleanField(required=False)
    filename2 = forms.CharField(required=False)

    filename3_search = forms.BooleanField(required=False)
    filename3 = forms.CharField(required=False)

    root1_search = forms.BooleanField(required=False)
    root1 = forms.CharField(required=False)

    root2_search = forms.BooleanField(required=False)
    root2 = forms.CharField(required=False)

    root3_search = forms.BooleanField(required=False)
    root3 = forms.CharField(required=False)

    show_valid = forms.BooleanField(required=False)
    show_invalid = forms.BooleanField(required=False)

    show_deprecated = forms.BooleanField(required=False)
    show_nondeprecated = forms.BooleanField(required=False)
    experiment_type = forms.ModelChoiceField(queryset=ExperimentType.objects.all(), required=True, empty_label=None)
    show_wrong_experiment_type = forms.BooleanField(required=False)
    page_number = forms.IntegerField(required=False)
    def set_page_number(self, page_number):
        data = self.data.copy()
        data['page_number'] = page_number
        self.data = data




class BaseExperimentTypeCorrectionFormSet(BaseModelFormSet):
    def add_fields(self, form, index):
        super().add_fields(form, index)
        form.fields["filename"] =forms.CharField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 100}),
                                          required=False)
        form.fields["filename_id"] = forms.IntegerField(required=False,widget=forms.HiddenInput())
        form.fields["exclude"] = forms.BooleanField(required=False)
        form.fields["deprecate"] = forms.BooleanField(required=False)

        form.fields["last_modified"] = forms.DateTimeField(widget=forms.DateTimeInput(attrs={'readonly': 'readonly'}),
                                          required=False)

        form.fields["filesize"] = forms.IntegerField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 5}),
                                          required=False)

        form.fields["old_experiment_type"] = forms.CharField(
            widget=forms.TextInput(attrs={'readonly':'readonly', 'size':50}),
            required=False)
        form.fields["reparse"] = forms.BooleanField(required=False)


class BaseMetadataCorrectionFormSet(BaseModelFormSet):
    def add_fields(self, form, index):
        super().add_fields(form, index)
        form.fields["filename"] =forms.CharField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 100}),
                                          required=False)
        form.fields["filename_id"] = forms.IntegerField(required=False,widget=forms.HiddenInput())
        form.fields["exclude"] = forms.BooleanField(required=False)
        form.fields["deprecate"] = forms.BooleanField(required=False)

        form.fields["last_modified"] = forms.DateTimeField(widget=forms.DateTimeInput(attrs={'readonly': 'readonly'}),
                                          required=False)

        form.fields["filesize"] = forms.IntegerField(widget=forms.TextInput(attrs={'readonly': 'readonly', 'size': 5}),
                                          required=False)


