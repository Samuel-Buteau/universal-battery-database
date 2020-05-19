from django import forms
from filename_database.models import *
from datetime import date
from django.forms import BaseModelFormSet, formset_factory
from cell_database.models import Dataset


class SearchForm(forms.Form):
    filename1_search = forms.BooleanField(required = False)
    filename1 = forms.CharField(required = False)

    filename2_search = forms.BooleanField(required = False)
    filename2 = forms.CharField(required = False)

    filename3_search = forms.BooleanField(required = False)
    filename3 = forms.CharField(required = False)

    root1_search = forms.BooleanField(required = False)
    root1 = forms.CharField(required = False)

    root2_search = forms.BooleanField(required = False)
    root2 = forms.CharField(required = False)

    root3_search = forms.BooleanField(required = False)
    root3 = forms.CharField(required = False)

    charID_search = forms.BooleanField(required = False)
    charID_exact = forms.CharField(required = False)

    cell_id_search = forms.BooleanField(required = False)
    cell_id_exact = forms.IntegerField(required = False)
    cell_id_minimum = forms.IntegerField(required = False)
    cell_id_maximum = forms.IntegerField(required = False)

    voltage_search = forms.BooleanField(required = False)
    voltage_exact = forms.FloatField(required = False)
    voltage_minimum = forms.FloatField(required = False)
    voltage_maximum = forms.FloatField(required = False)

    temperature_search = forms.BooleanField(required = False)
    temperature_exact = forms.IntegerField(required = False)
    temperature_minimum = forms.IntegerField(required = False)
    temperature_maximum = forms.IntegerField(required = False)

    date_search = forms.BooleanField(required = False)
    date_exact = forms.DateField(
        widget = forms.SelectDateWidget(
            empty_label = "--------",
            years = range(2000, date.today().year + 1),
        ),
        required = False,
    )
    date_minimum = forms.DateField(
        widget = forms.SelectDateWidget(
            empty_label = "--------",
            years = range(2000, date.today().year + 1),
        ),
        required = False,
    )
    date_maximum = forms.DateField(
        widget = forms.SelectDateWidget(
            empty_label = "--------",
            years = range(2000, date.today().year + 1),
        ),
        required = False,
    )

    page_number = forms.IntegerField(required = False)
    show_visuals = forms.BooleanField(required = False)

    dataset = forms.ModelChoiceField(
        queryset = Dataset.objects.all(), required = False,
    )

    def set_page_number(self, page_number):
        data = self.data.copy()
        data["page_number"] = page_number
        self.data = data


class CellIDOverviewForm(forms.Form):
    cell_id = forms.IntegerField(
        widget = forms.TextInput(attrs = {"readonly": "readonly", "size": 5}),
        required = False
    )
    exclude = forms.BooleanField(required = False)

    number_of_active = forms.IntegerField(
        widget = forms.TextInput(attrs = {"readonly": "readonly", "size": 5}),
        required = False,
    )
    number_of_deprecated = forms.IntegerField(
        widget = forms.TextInput(attrs = {"readonly": "readonly", "size": 5}),
        required = False,
    )
    number_of_needs_importing = forms.IntegerField(
        widget = forms.TextInput(attrs = {"readonly": "readonly", "size": 5}),
        required = False,
    )
    first_active_file = forms.CharField(
        widget = forms.TextInput(attrs = {"readonly": "readonly", "size": 100}),
        required = False,
    )


CellIDOverviewFormset = formset_factory(CellIDOverviewForm, extra = 0)
