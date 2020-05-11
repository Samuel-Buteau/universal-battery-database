from django.shortcuts import render
from django.forms import modelformset_factory, formset_factory
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from .forms import *
from django.db.models import Q
from django import forms
from filename_database.parsing_functions import *
import datetime

def get_exclusions(experiment_type=None):

    exclude = ['drive_profile', 'experiment_type','drive_profile_x_numerator', 'drive_profile_x_denominator', 'drive_profile_y_numerator',
                    'drive_profile_y_denominator', 'drive_profile_z']


    if experiment_type is None or not experiment_type.cell_id_active:
        exclude.append('cell_id')

    if experiment_type is None or not experiment_type.start_cycle_active:
        exclude.append('start_cycle')

    if experiment_type is None or not experiment_type.voltage_active:
        exclude.append('voltage')

    if experiment_type is None or not experiment_type.temperature_active:
        exclude.append('temperature')

    if experiment_type is None or not experiment_type.AC_active:
        exclude.append('AC')
    if experiment_type is None or not experiment_type.AC_increment_active:
        exclude.append('AC_increment')

    if experiment_type is None or not experiment_type.version_number_active:
        exclude.append('version_number')

    return exclude


def get_headings(experiment_type=None, exclude=None, has_file_linked=True):
    headings = []
    possible_fields = [
        'charID',
        'cell_id',
        'start_cycle',
        'voltage',
        'temperature',
        'AC',
        'AC_increment',
        'version_number',
        'date',
    ]
    if has_file_linked:
        possible_fields += ['last_modified','filesize']

    for field in possible_fields:
        if field not in exclude:
            if field == 'voltage':
                field = experiment_type.voltage_name
            if field == 'temperature':
                field = experiment_type.temperature_name
            f = field
            headings.append(f)
    return headings


def main_page(request):
    ar ={

    }
    if request.method == 'POST':
        search_form = SearchForm(request.POST)
        if search_form.is_valid():
            ar['search_form'] = search_form
            experiment_type = search_form.cleaned_data['experiment_type']
            exclude = get_exclusions(experiment_type)
            headings = get_headings(experiment_type, exclude)
            headings = [ h.replace('_', ' ') for h in headings]
            MetadataEditForm = modelformset_factory(
                ValidMetadata,
                formset=BaseMetadataCorrectionFormSet,
                exclude=exclude,
                widgets={
                    'date': forms.SelectDateWidget(empty_label="Nothing", years=range(2000, datetime.date.today().year + 1)),
                    'charID':forms.TextInput(attrs={'size': 5})}
            )

            ExperimentTypeEditForm = modelformset_factory(
                ValidMetadata,
                formset=BaseExperimentTypeCorrectionFormSet,
                exclude=get_exclusions()+["experiment_type", 'date', 'charID'],
            )

            ValidMetadataForm = modelformset_factory(
                ValidMetadata,
                exclude=get_exclusions(experiment_type) + ["experiment_type"],
                widgets={
                    'date': forms.SelectDateWidget(empty_label="Nothing",
                                                   years=range(2000, datetime.date.today().year + 1)),
                    'charID': forms.TextInput(attrs={'size': 5})},
                extra=1
            )

            ar['search_form'] = search_form

            if 'change_exp_type' in request.POST:
                valid_metadata_formset = ValidMetadataForm()
                ar['metadata_edit_formset'] = valid_metadata_formset
                ar['make_changes'] = 'show_filename'
                headings = get_headings(experiment_type, get_exclusions(experiment_type), has_file_linked=False)
                headings = [h.replace('_', ' ') for h in headings]
                print(headings)
                ar['headings'] = headings

            elif 'search_and_fix_metadata' in request.POST:
                q = Q()
                if search_form.cleaned_data['show_valid'] and not search_form.cleaned_data['show_invalid']:
                    q = Q(is_valid=True)
                elif not search_form.cleaned_data['show_valid'] and search_form.cleaned_data['show_invalid']:
                    q = Q(is_valid=False)

                if search_form.cleaned_data['show_deprecated'] and not search_form.cleaned_data[
                    'show_nondeprecated']:
                    q = q & Q(deprecated=True)
                elif not search_form.cleaned_data['show_deprecated'] and search_form.cleaned_data[
                        'show_nondeprecated']:
                    q = q & Q(deprecated=False)
                if search_form.cleaned_data['filename1_search']:
                    q = q & Q(filename__icontains=search_form.cleaned_data['filename1'])
                if search_form.cleaned_data['filename2_search']:
                    q = q & Q(filename__icontains=search_form.cleaned_data['filename2'])
                if search_form.cleaned_data['filename3_search']:
                    q = q & Q(filename__icontains=search_form.cleaned_data['filename3'])

                if search_form.cleaned_data['root1_search']:
                    q = q & Q(root__icontains=search_form.cleaned_data['root1'])
                if search_form.cleaned_data['root2_search']:
                    q = q & Q(root__icontains=search_form.cleaned_data['root2'])
                if search_form.cleaned_data['root3_search']:
                    q = q & Q(root__icontains=search_form.cleaned_data['root3'])

                if search_form.cleaned_data['show_wrong_experiment_type']:
                    q = q & ~Q(valid_metadata__experiment_type=experiment_type)
                    total_query = DatabaseFile.objects.filter(q)
                else:
                    q = q & Q(valid_metadata__experiment_type=experiment_type)
                    total_query = DatabaseFile.objects.filter(q)
                initial = []
                pn = search_form.cleaned_data['page_number']
                if pn is None:
                    pn = 1
                    search_form.set_page_number(pn)
                n = total_query.count()
                max_page = int(n / 15)
                if (n % 15) != 0:
                    max_page += 1

                if pn > max_page:
                    pn = max_page
                    search_form.set_page_number(pn)

                if pn < 1:
                    pn = 1
                    search_form.set_page_number(pn)

                if search_form.cleaned_data['show_wrong_experiment_type']:
                    for db_filename in total_query[(pn - 1) * 15:min(n, (pn) * 15)]:
                        my_initial = {
                            "filename": db_filename.filename,
                            "filename_id": db_filename.id,
                            "exclude": True,
                            'deprecate': db_filename.deprecated,
                            "reparse": True,
                            "last_modified": db_filename.last_modified,
                            "filesize": '{}'.format(int(db_filename.filesize / 1024))
                        }
                        if db_filename.valid_metadata is None:
                            my_initial["old_experiment_type"] = ''
                        else:
                            my_initial["old_experiment_type"] = db_filename.valid_metadata.experiment_type.__str__()
                        print(my_initial)

                        initial.append(my_initial)

                    ExperimentTypeEditForm = modelformset_factory(
                        ValidMetadata,
                        formset=BaseExperimentTypeCorrectionFormSet,
                        exclude=get_exclusions()+["experiment_type", 'date', 'charID'],
                        extra=len(initial)
                    )

                    experiment_type_edit_formset = ExperimentTypeEditForm(queryset=DatabaseFile.objects.none(),
                                                             initial=initial)

                    ar['metadata_edit_formset'] = experiment_type_edit_formset
                    ar['make_changes'] = 'make_changes_to_experiment_type'
                    headings = get_headings(None, get_exclusions()+['date', 'charID'])
                    headings = [h.replace('_', ' ') for h in headings] + ['old_experiment_type', 'reparse']
                    ar['headings'] = headings
                else:
                    for db_filename in total_query[(pn - 1) * 15:min(n, (pn) * 15)]:
                        my_initial = {
                            "filename": db_filename.filename,
                            "filename_id": db_filename.id,
                            "exclude": True,
                            'deprecate': db_filename.deprecated,
                            "experiment_type": experiment_type,
                            "charID": db_filename.valid_metadata.charID,
                            "cell_id": db_filename.valid_metadata.cell_id,
                            "start_cycle": db_filename.valid_metadata.start_cycle,
                            "voltage": db_filename.valid_metadata.voltage,
                            "temperature": db_filename.valid_metadata.temperature,
                            "AC": db_filename.valid_metadata.AC,
                            "AC_increment": db_filename.valid_metadata.AC_increment,
                            "version_number": db_filename.valid_metadata.version_number,
                            "date": db_filename.valid_metadata.date,
                            "last_modified": db_filename.last_modified,
                            "filesize": '{}'.format(int(db_filename.filesize / 1024))
                        }
                        initial.append(my_initial)

                    # Defining it again so that 'extra' works

                    MetadataEditForm = modelformset_factory(
                        ValidMetadata,
                        formset=BaseMetadataCorrectionFormSet,
                        exclude=exclude , extra=len(initial),
                        widgets={
                            'date': forms.SelectDateWidget(empty_label="Nothing",
                                                           years=range(2000, datetime.date.today().year + 1)),
                            'charID': forms.TextInput(attrs={'size': 5}),
                            'cell_id': forms.NumberInput(attrs={'style': 'width:8ch'}),
                            'start_cycle': forms.NumberInput(attrs={'style': 'width:8ch'}),
                            'temperature': forms.NumberInput(attrs={'style': 'width:8ch'}),
                            'voltage': forms.NumberInput(attrs={'style': 'width:6ch'}), }
                    )
                    metadata_edit_formset = MetadataEditForm(queryset=DatabaseFile.objects.none(),
                                                             initial=initial)
                    ar['metadata_edit_formset'] = metadata_edit_formset
                    ar['make_changes'] = 'make_changes_to_metadata'
                    ar['headings'] = headings

                ar['search_form'] = search_form

                ar['page_number'] = pn
                ar['max_page_number'] = max_page
            elif 'make_changes_to_metadata' in request.POST or 'make_changes_to_experiment_type' in request.POST:
                if 'make_changes_to_metadata' in request.POST:
                    metadata_edit_formset = MetadataEditForm(request.POST)
                elif 'make_changes_to_experiment_type' in request.POST:
                    metadata_edit_formset = ExperimentTypeEditForm(request.POST)

                for form in metadata_edit_formset:
                    validation_step = form.is_valid()
                    if not DatabaseFile.objects.filter(id=form.cleaned_data['filename_id']).exists():
                        print('invalid id', form.cleaned_data['filename_id'])
                        continue
                    file = DatabaseFile.objects.get(id=form.cleaned_data['filename_id'])

                    to_be_excluded = form.cleaned_data['exclude']
                    to_be_deprecated = form.cleaned_data['deprecate']

                    if to_be_excluded:
                        print('exclude')
                        continue

                    if to_be_deprecated:
                        file.deprecated = True
                        file.save()
                        print('deprecated')
                        continue

                    if validation_step:
                        if not to_be_deprecated:
                            file.deprecated = False

                        if 'make_changes_to_metadata' in request.POST:
                            charID = None
                            cell_id = None
                            voltage = None
                            temperature = None
                            AC = None
                            AC_increment = None
                            version_number = None
                            date = None
                            if 'charID' in form.cleaned_data.keys():
                                charID = form.cleaned_data['charID']
                            if 'cell_id' in form.cleaned_data.keys():
                                cell_id = form.cleaned_data['cell_id']
                            if 'start_cycle' in form.cleaned_data.keys():
                                start_cycle = form.cleaned_data['start_cycle']
                            if 'voltage' in form.cleaned_data.keys():
                                voltage = form.cleaned_data['voltage']
                            if 'temperature' in form.cleaned_data.keys():
                                temperature = form.cleaned_data['temperature']
                            if 'AC' in form.cleaned_data.keys():
                                AC = form.cleaned_data['AC']
                            if 'AC_increment' in form.cleaned_data.keys():
                                AC_increment = form.cleaned_data['AC_increment']
                            if 'version_number' in form.cleaned_data.keys():
                                version_number = form.cleaned_data['version_number']
                            if 'date' in form.cleaned_data.keys():
                                date = form.cleaned_data['date']

                            file.set_valid_metadata(
                                experiment_type=experiment_type,
                                charID=charID,
                                cell_id=cell_id,
                                start_cycle=start_cycle,
                                voltage=voltage,
                                temperature=temperature,
                                AC=AC,
                                AC_increment=AC_increment,
                                version_number=version_number,
                                date = date
                            )
                            file.save()

                        elif 'make_changes_to_experiment_type' in request.POST:
                            to_be_reparsed = form.cleaned_data['reparse']
                            to_be_reparsed = to_be_reparsed or file.valid_metadata is None
                            if not to_be_reparsed:
                                file.set_valid_metadata(
                                    experiment_type=experiment_type
                                )
                                file.save()
                            else:
                                meta, valid = deterministic_parser(file.filename, experiment_type)
                                file.set_valid_metadata(valid_metadata=meta)
                                file.save()

            elif 'show_filename' in request.POST:
                metadata_edit_formset = ValidMetadataForm(request.POST)
                filenames = []
                for form in metadata_edit_formset:

                    if form.is_valid():
                        charID = None
                        cell_id = None
                        voltage = None
                        temperature = None
                        AC = None
                        AC_increment = None
                        version_number = None
                        date = None
                        if 'charID' in form.cleaned_data.keys():
                            charID = form.cleaned_data['charID']
                        if 'cell_id' in form.cleaned_data.keys():
                            cell_id = form.cleaned_data['cell_id']
                        if 'start_cycle' in form.cleaned_data.keys():
                            start_cycle = form.cleaned_data['start_cycle']
                        if 'voltage' in form.cleaned_data.keys():
                            voltage = form.cleaned_data['voltage']
                        if 'temperature' in form.cleaned_data.keys():
                            temperature = form.cleaned_data['temperature']
                        if 'AC' in form.cleaned_data.keys():
                            AC = form.cleaned_data['AC']
                        if 'AC_increment' in form.cleaned_data.keys():
                            AC_increment = form.cleaned_data['AC_increment']
                        if 'version_number' in form.cleaned_data.keys():
                            version_number = form.cleaned_data['version_number']
                        if 'date' in form.cleaned_data.keys():
                            date = form.cleaned_data['date']


                        filenames.append(
                            ValidMetadata(
                            experiment_type=experiment_type,
                            charID=charID,
                            cell_id=cell_id,
                            start_cycle=start_cycle,
                            voltage=voltage,
                            temperature=temperature,
                            AC=AC,
                            AC_increment=AC_increment,
                            version_number=version_number,
                            date=date).get_filename)
                if len(filenames) != 0:
                    ar['filename_to_show'] = filenames
                ar['search_form'] = search_form
    else:
        ar['search_form'] = SearchForm()
    return render(request, 'filename_database/form_interface.html', ar)





