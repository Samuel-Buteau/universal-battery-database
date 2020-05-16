from django.http import HttpResponseRedirect, HttpResponse

from django.urls import reverse

from django.shortcuts import render

from django import forms
import datetime

from django.utils import timezone
from cycling.neware_processing_functions import full_import_cell_ids
from .models import *
from django.db.models import Max, Min
import math
from .forms import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io import BytesIO
import base64
from django.db.models import Q, F
from plot import plot_cycling_direct

import numpy as np


def split_interval(initial, n, M, i):
    i = min(n - 1, i)
    d = int(n / M)
    r = n % M
    new_initial = initial + max(i - r, 0) * d + min(i, r) * (d + 1)
    if i < r:
        new_n = d + 1
    else:
        new_n = d

    return new_initial, new_n


def get_all_intervals(initial, n, M):
    return [split_interval(initial, n, M, i) for i in range(min(M, n))]


colors = ["k", "r", "b", "g", "c", "m", "o"]
number_of_options = 26


def view_cell_id(request, cell_id, cursor):
    list_all_options = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    ]

    cell_id = int(cell_id)
    ar = {"cell_id": cell_id}

    if len(cursor) == 0:
        chosen = []
    else:
        list_of_choices = cursor.split("_")

        chosen = [
            list_all_options.index(o) for o in list_of_choices
            if o and o in list_all_options
        ]

    if request.method == "POST" and "back" in request.POST:
        if len(chosen) == 0:
            return HttpResponseRedirect(
                reverse("view_cell_id", args = (cell_id, cursor))
            )

        new_cursor = "_".join([list_all_options[c] for c in chosen[:-1]])

        return HttpResponseRedirect(
            reverse("view_cell_id", args = (cell_id, new_cursor))
        )

    if request.method == "POST" and "start" in request.POST:
        return HttpResponseRedirect(
            reverse("view_cell_id", args = (cell_id, ""))
        )

    files_cell_id = get_files_for_cell_id(cell_id)

    if len(files_cell_id) != 0:
        c_curves = set([])
        for f in files_cell_id:
            offset_cycle = f.database_file.valid_metadata.start_cycle
            c_curves.update(
                set([
                    offset_cycle + cyc
                    for cyc in Cycle.objects.filter(
                        cycling_file = f
                    ).order_by("cycle_number").values_list(
                        "cycle_number", flat = True,
                    )
                ])
            )

        cycle_list = np.sort(np.array(list(c_curves)))
        interval = (0, len(cycle_list))

        """
        we must get to the point that
        we have the whole interval,
        the list of sub intervals, and the ability to choose a sub interval.
        if at any point we go out of bounds, redirect to the prefix of the cursor
        """

        print("interval", interval)
        for count, choice in enumerate(chosen):
            if choice >= interval[1]:
                new_cursor = "_".join(
                    [list_all_options[c] for c in chosen[:count]]
                )
                return HttpResponseRedirect(
                    reverse("view_cell_id", args = (cell_id, new_cursor))
                )
            interval = split_interval(
                interval[0], interval[1], M = number_of_options, i = choice,
            )
            print("interval", interval)
        max_possible_options = min(number_of_options, interval[1])
        print("max_possible_options", max_possible_options)

        class ChoiceForm(forms.Form):
            option = forms.ChoiceField(
                choices = [
                    (o, o) for o in list_all_options[:max_possible_options]
                ]
            )

        if request.method == "POST" and "zoom" in request.POST:
            if interval[1] == 1:
                return HttpResponseRedirect(
                    reverse("view_cell_id", args = (cell_id, cursor))
                )

            my_form = ChoiceForm(request.POST)
            if my_form.is_valid():

                option = my_form.cleaned_data["option"]
                if option in list_all_options[:max_possible_options]:
                    new_cursor = "_".join(
                        [list_all_options[c] for c in chosen] + [option]
                    )
                else:
                    return HttpResponseRedirect(
                        reverse("view_cell_id", args = (cell_id, cursor))
                    )

            return HttpResponseRedirect(
                reverse("view_cell_id", args = (cell_id, new_cursor))
            )

        if (
            request.method == "POST"
            and ("curse" in request.POST or "bless" in request.POST)
        ):
            doing = {"curse": [True, False], "bless": [False, True]}
            if "curse" in request.POST:
                doing = doing["curse"]
            elif "bless" in request.POST:
                doing = doing["bless"]

            my_form = ChoiceForm(request.POST)
            if my_form.is_valid():

                option = my_form.cleaned_data["option"]
                if option not in list_all_options[:max_possible_options]:
                    return HttpResponseRedirect(
                        reverse("view_cell_id", args = (cell_id, cursor)))

                option_index = list_all_options.index(option)
                selected_interval = split_interval(
                    interval[0], interval[1],
                    M = max_possible_options, i = option_index,
                )
                lowest_cycle = cycle_list[selected_interval[0]]
                largest_cycle = cycle_list[
                    selected_interval[0] + selected_interval[1] - 1]

            # TODO: add option to choose file.
            for f in files_cell_id:
                offset_cycle = f.database_file.valid_metadata.start_cycle
                Cycle.objects.filter(
                    cycling_file = f,
                    cycle_number__range = (
                        lowest_cycle - offset_cycle,
                        largest_cycle - offset_cycle
                    ),
                    valid_cycle = doing[0]
                ).update(valid_cycle = doing[1])

            return HttpResponseRedirect(
                reverse("view_cell_id", args = (cell_id, cursor)),
            )

        all_intervals = get_all_intervals(
            interval[0], interval[1], M = max_possible_options,
        )
        print("all_intervals", all_intervals)
        vertical_barriers = []
        for index_set_i in range(len(all_intervals) - 1):
            max_low_index = all_intervals[index_set_i + 1][0] - 1
            min_up_index = all_intervals[index_set_i + 1][0]
            vertical_barriers.append(
                .5 * (cycle_list[min_up_index] + cycle_list[max_low_index])
            )

        lowest_cycle = cycle_list[interval[0]]
        largest_cycle = cycle_list[interval[0] + interval[1] - 1]

        image_base64 = plot_cycling_direct(
            cell_id,
            lower_cycle = lowest_cycle,
            upper_cycle = largest_cycle,
            show_invalid = True,
            vertical_barriers = vertical_barriers,
            list_all_options = list_all_options[:max_possible_options],
            figsize = [14., 6.],
        )

        my_form = ChoiceForm()

        ar["image_base64"] = image_base64
        ar["my_form"] = my_form

    ar["cursor"] = cursor

    active_files = CyclingFile.objects.filter(
        database_file__deprecated = False,
        database_file__valid_metadata__cell_id = cell_id,
        database_file__last_modified__lte = F("import_time"),
    )
    if active_files.exists():
        ar["active_files"] = [
            (
                f.database_file.filename,
                f.database_file.valid_metadata.start_cycle,
                "{}-{}-{}".format(
                    f.database_file.last_modified.year,
                    f.database_file.last_modified.month,
                    f.database_file.last_modified.day,
                ),
                int(f.database_file.filesize / 1024),
            )
            for f in active_files.order_by(
                "database_file__valid_metadata__start_cycle"
            )
        ]

    deprecated_files = CyclingFile.objects.filter(
        database_file__deprecated = True,
        database_file__valid_metadata__cell_id = cell_id)
    if deprecated_files.exists():
        ar["deprecated_files"] = [
            (
                f.database_file.filename,
                f.database_file.valid_metadata.start_cycle,
                "{}-{}-{}".format(
                    f.database_file.last_modified.year,
                    f.database_file.last_modified.month,
                    f.database_file.last_modified.day,
                ),
                int(f.database_file.filesize / 1024),
            )
            for f in deprecated_files.order_by(
                "database_file__valid_metadata__start_cycle"
            )
        ]

    needs_importing_files = []
    needs_importing_files1 = CyclingFile.objects.filter(
        database_file__deprecated = False,
        database_file__valid_metadata__cell_id = cell_id,
        database_file__last_modified__gt = F("import_time"))
    if needs_importing_files1.exists():
        needs_importing_files += [
            (
                f.database_file.filename,
                f.database_file.valid_metadata.start_cycle,
                "{}-{}-{}".format(
                    f.database_file.last_modified.year,
                    f.database_file.last_modified.month,
                    f.database_file.last_modified.day,
                ),
                int(f.database_file.filesize / 1024),
                "{}-{}-{}".format(
                    f.import_time.year,
                    f.import_time.month,
                    f.import_time.day,
                ),
            )
            for f in needs_importing_files1.order_by(
                "database_file__valid_metadata__start_cycle"
            )
        ]

    exp_type = filename_database.models.ExperimentType.objects.get(
        category = filename_database.models.Category.objects.get(
            name = "cycling"
        ),
        subcategory = filename_database.models.SubCategory.objects.get(
            name = "neware"
        )
    )
    needs_importing_files2 = DatabaseFile.objects.filter(
        is_valid = True,
        deprecated = False
    ).exclude(valid_metadata = None).filter(
        valid_metadata__experiment_type = exp_type,
        valid_metadata__cell_id = cell_id
    ).exclude(id__in = CyclingFile.objects.filter(
        database_file__valid_metadata__cell_id = cell_id
    ).values_list("database_file_id", flat = True))
    if needs_importing_files2.exists():
        needs_importing_files += [
            (
                f.filename,
                f.valid_metadata.start_cycle,
                "{}-{}-{}".format(
                    f.last_modified.year,
                    f.last_modified.month,
                    f.last_modified.day,
                ),
                int(f.filesize / 1024),
                "Never",
            )
            for f in needs_importing_files2.order_by(
                "valid_metadata__start_cycle"
            )
        ]

    if len(needs_importing_files) != 0:
        ar["needs_importing_files"] = needs_importing_files

    return render(
        request, "cycling/view_cell_id.html", ar,
    )


def index(request):
    return render(
        request, "cycling/index.html",
    )


def main_page(request):
    ar = {}

    if request.method == "POST":

        search_form = SearchForm(request.POST)

        if search_form.is_valid():

            ar["search_form"] = search_form

            if "search_validated_cycling_data" in request.POST:

                exp_type = filename_database.models.ExperimentType.objects.get(
                    category = filename_database.models.Category.objects.get(
                        name = "cycling"
                    ),
                    subcategory
                    = filename_database.models.SubCategory.objects.get(
                        name = "neware"
                    )
                )

                q = Q(is_valid = True) & ~Q(valid_metadata = None) & Q(
                    valid_metadata__experiment_type = exp_type
                )
                dataset = search_form.cleaned_data["dataset"]
                if dataset is not None:
                    cell_ids = [
                        wet_cell.cell_id for wet_cell
                        in dataset.wet_cells.order_by("cell_id")
                    ]
                    q = q & Q(valid_metadata__cell_id__in = cell_ids)

                if search_form.cleaned_data["filename1_search"]:
                    q = q & Q(
                        filename__icontains
                        = search_form.cleaned_data["filename1"]
                    )

                if search_form.cleaned_data["filename2_search"]:
                    q = q & Q(
                        filename__icontains
                        = search_form.cleaned_data["filename2"]
                    )

                if search_form.cleaned_data["filename3_search"]:
                    q = q & Q(
                        filename__icontains
                        = search_form.cleaned_data["filename3"]
                    )

                if search_form.cleaned_data["root1_search"]:
                    q = q & Q(
                        root__icontains = search_form.cleaned_data["root1"]
                    )
                if search_form.cleaned_data["root2_search"]:
                    q = q & Q(
                        root__icontains = search_form.cleaned_data["root2"]
                    )
                if search_form.cleaned_data["root3_search"]:
                    q = q & Q(
                        root__icontains = search_form.cleaned_data["root3"]
                    )

                if search_form.cleaned_data["charID_search"]:
                    q = q & Q(
                        valid_metadata__charID
                        = search_form.cleaned_data["charID_exact"]
                    )

                if search_form.cleaned_data["cell_id_search"]:
                    if search_form.cleaned_data["cell_id_exact"] is not None:
                        q = q & Q(
                            valid_metadata__cell_id
                            = search_form.cleaned_data["cell_id_exact"]
                        )
                    else:
                        if (
                            search_form.cleaned_data["cell_id_minimum"]
                            is not None
                            and search_form.cleaned_data["cell_id_maximum"]
                            is not None
                        ):
                            q = q & Q(
                                valid_metadata__cell_id__range = (
                                    search_form.cleaned_data["cell_id_minimum"],
                                    search_form.cleaned_data["cell_id_maximum"],
                                )
                            )
                        elif (
                            search_form.cleaned_data["cell_id_minimum"]
                            is None
                            and search_form.cleaned_data["cell_id_maximum"]
                            is not None
                        ):
                            q = q & Q(
                                valid_metadata__cell_id__lte
                                = search_form.cleaned_data["cell_id_maximum"]
                            )
                        elif (
                            search_form.cleaned_data["cell_id_minimum"]
                            is not None
                            and search_form.cleaned_data["cell_id_maximum"]
                            is None
                        ):
                            q = q & Q(
                                valid_metadata__cell_id__gte
                                = search_form.cleaned_data["cell_id_minimum"]
                            )

                if search_form.cleaned_data["voltage_search"]:
                    if search_form.cleaned_data["voltage_exact"] is not None:
                        q = q & Q(
                            valid_metadata__voltage = search_form.cleaned_data[
                                "voltage_exact"]
                        )
                    else:
                        if (
                            search_form.cleaned_data["voltage_minimum"]
                            is not None
                            and search_form.cleaned_data["voltage_maximum"]
                            is not None
                        ):
                            q = q & Q(
                                valid_metadata__voltage__range = (
                                    search_form.cleaned_data["voltage_minimum"],
                                    search_form.cleaned_data["voltage_maximum"],
                                )
                            )
                        elif (
                            search_form.cleaned_data["voltage_minimum"]
                            is None
                            and search_form.cleaned_data["voltage_maximum"]
                            is not None
                        ):
                            q = q & Q(
                                valid_metadata__voltage__lte
                                = search_form.cleaned_data["voltage_maximum"]
                            )
                        elif (
                            search_form.cleaned_data["voltage_minimum"]
                            is not None
                            and search_form.cleaned_data["voltage_maximum"]
                            is None
                        ):
                            q = q & Q(
                                valid_metadata__voltage__gte
                                = search_form.cleaned_data["voltage_minimum"]
                            )

                if search_form.cleaned_data["temperature_search"]:
                    if (
                        search_form.cleaned_data["temperature_exact"]
                        is not None
                    ):
                        q = q & Q(
                            valid_metadata__temperature
                            = search_form.cleaned_data["temperature_exact"]
                        )
                    else:
                        if (
                            search_form.cleaned_data["temperature_minimum"]
                            is not None
                            and search_form.cleaned_data["temperature_maximum"]
                            is not None
                        ):
                            q = q & Q(
                                valid_metadata__temperature__range = (
                                    search_form.cleaned_data[
                                        "temperature_minimum"
                                    ],
                                    search_form.cleaned_data[
                                        "temperature_maximum"
                                    ],
                                )
                            )
                        elif (
                            search_form.cleaned_data["temperature_minimum"]
                            is None
                            and search_form.cleaned_data["temperature_maximum"]
                            is not None
                        ):
                            q = q & Q(
                                valid_metadata__temperature__lte
                                = search_form.cleaned_data[
                                    "temperature_maximum"
                                ]
                            )
                        elif (
                            search_form.cleaned_data["temperature_minimum"]
                            is not None
                            and search_form.cleaned_data["temperature_maximum"]
                            is None
                        ):
                            q = q & Q(
                                valid_metadata__temperature__gte
                                = search_form.cleaned_data[
                                    "temperature_minimum"
                                ]
                            )

                if search_form.cleaned_data["date_search"]:
                    if search_form.cleaned_data["date_exact"] is not None:
                        q = q & Q(
                            valid_metadata__date
                            = search_form.cleaned_data["date_exact"]
                        )
                    else:
                        if (
                            search_form.cleaned_data["date_minimum"] is not None
                            and
                            search_form.cleaned_data["date_maximum"] is not None
                        ):
                            q = q & Q(valid_metadata__date__range = (
                                search_form.cleaned_data["date_minimum"],
                                search_form.cleaned_data["date_maximum"]))
                        elif (
                            search_form.cleaned_data["date_minimum"] is None
                            and
                            search_form.cleaned_data["date_maximum"] is not None
                        ):
                            q = q & Q(
                                valid_metadata__date__lte
                                = search_form.cleaned_data["date_maximum"]
                            )
                        elif (
                            search_form.cleaned_data["date_minimum"] is not None
                            and search_form.cleaned_data["date_maximum"] is None
                        ):
                            q = q & Q(
                                valid_metadata__date__gte
                                = search_form.cleaned_data["date_minimum"]
                            )

                total_query = DatabaseFile.objects.filter(q).order_by(
                    "valid_metadata__cell_id").values_list(
                    "valid_metadata__cell_id", flat = True
                ).distinct()

                number_per_page = 10
                initial = []
                pn = search_form.cleaned_data["page_number"]
                if pn is None:
                    pn = 1
                    search_form.set_page_number(pn)
                n = total_query.count()
                max_page = int(n / number_per_page)
                if (n % number_per_page) != 0:
                    max_page += 1

                if pn > max_page:
                    pn = max_page
                    search_form.set_page_number(pn)

                if pn < 1:
                    pn = 1
                    search_form.set_page_number(pn)

                for cell_id in total_query[
                    (pn - 1) * number_per_page:min(n, pn * number_per_page)
                ]:
                    """
                    cell_id
                    exclude
                    number_of_active
                    number_of_deprecated
                    number_of_needs_importing
                    import_soon
                    """

                    my_initial = {
                        "cell_id": cell_id,
                        "exclude": True,
                        "number_of_active": CyclingFile.objects.filter(
                            database_file__deprecated = False,
                            database_file__valid_metadata__cell_id = cell_id,
                            database_file__last_modified__lte = F("import_time")
                        ).count(),
                        "number_of_deprecated": CyclingFile.objects.filter(
                            database_file__deprecated = True,
                            database_file__valid_metadata__cell_id = cell_id
                        ).count(),
                        "number_of_needs_importing":
                            CyclingFile.objects.filter(
                                database_file__deprecated = False,
                                database_file__valid_metadata__cell_id
                                = cell_id,
                                database_file__last_modified__gt
                                = F("import_time"),
                            ).count() + DatabaseFile.objects.filter(
                                is_valid = True, deprecated = False,
                            ).exclude(valid_metadata = None).filter(
                                valid_metadata__experiment_type = exp_type,
                                valid_metadata__cell_id = cell_id,
                            ).exclude(
                                id__in = CyclingFile.objects.filter(
                                    database_file__valid_metadata__cell_id
                                    = cell_id
                                ).values_list(
                                    "database_file_id", flat = True,
                                )
                            ).count(),
                        "first_active_file": ""
                    }
                    if CyclingFile.objects.filter(
                        database_file__deprecated = False,
                        database_file__valid_metadata__cell_id = cell_id,
                        database_file__last_modified__lte = F("import_time"),
                    ).exists():
                        my_initial["first_active_file"] = (
                            CyclingFile.objects.filter(
                                database_file__deprecated = False,
                                database_file__valid_metadata__cell_id
                                = cell_id,
                                database_file__last_modified__lte
                                = F("import_time"),
                            )[0]
                        ).database_file.filename

                    initial.append(my_initial)

                cell_id_overview_formset = CellIDOverviewFormset(
                    initial = initial
                )

                if search_form.cleaned_data["show_visuals"]:
                    datas = []
                    for cell_id in total_query[
                        (pn - 1) * number_per_page:min(n, pn * number_per_page)
                    ]:
                        image_base64 = plot_cycling_direct(
                            cell_id, path_to_plots = None, figsize = [5., 4.],
                        )
                        datas.append((cell_id, image_base64))

                    n = 5

                    split_datas = [
                        datas[i:min(len(datas), i + n)] for i
                        in range(0, len(datas), n)
                    ]
                    ar["visual_data"] = split_datas

                ar["cell_id_overview_formset"] = cell_id_overview_formset
                ar["search_form"] = search_form
                ar["page_number"] = pn
                ar["max_page_number"] = max_page

            elif "trigger_reimport" in request.POST:
                cell_id_overview_formset = CellIDOverviewFormset(request.POST)
                collected_cell_ids = []
                for form in cell_id_overview_formset:
                    validation_step = form.is_valid()
                    to_be_excluded = form.cleaned_data["exclude"]

                    if to_be_excluded:
                        print("exclude")
                        continue

                    if validation_step:
                        collected_cell_ids.append(form.cleaned_data["cell_id"])

                full_import_cell_ids(collected_cell_ids)
                ar["search_form"] = search_form

    else:
        ar["search_form"] = SearchForm()

    return render(request, "cycling/form_interface.html", ar)


StepHeaderValue = [
    "Step Name", "Capacity (mAh)", "Average Voltage (V)", "Min Voltage (V)",
    "Max Voltage (V)", "Average Current by Capacity (mA)",
    "Average Current by Voltage (mA)", "Min Current (mA)", "Max Current (mA)",
    "Time (hours)", "Cumulative Time (hours)"
]
StepHeaderKey = ["Cycle Number", "Step Number"]
CycleHeaderValue = [
    "Charge Capacity (mAh)", "Discharge Capacity (mAh)",
    "Charge Average Voltage (V)", "Discharge Average Voltage (V)",
    "Charge Min Voltage (V)", "Discharge Min Voltage (V)",
    "Charge Max Voltage (V)", "Discharge Max Voltage (V)",
    "Charge Average Current by Capacity (mA)",
    "Discharge Average Current by Capacity (mA)",
    "Charge Average Current by Voltage (mA)",
    "Discharge Average Current by Voltage (mA)",
    "Charge Min Current (mA)", "Discharge Min Current (mA)",
    "Charge Max Current (mA)", "Discharge Max Current (mA)",
    "Charge Time (hours)", "Discharge Time (hours)",
    "Charge Cumulative Time (hours)", "Discharge Cumulative Time (hours)"
]
CycleHeaderKey = ["Cycle Number"]


def convert_to_csv2(headers, np_content):
    contents = ""
    for header in headers:
        contents += ",".join(header) + "\n"
    for np_content_i in np_content:
        contents += ",".join(np_content_i) + "\n"
    return contents


# TODO(harvey): rename function
def ExportStep(my_step_data):
    my_content = []
    for cycle in my_step_data.keys():
        for step in my_step_data[cycle].keys():
            my_content.append(
                [str(cycle), str(step)] + [my_step_data[cycle][step][0]]
                + [str(field) for field in my_step_data[cycle][step][1]]
            )

    my_content = np.array(my_content)

    # cap, v_avg, v_min, v_max, cur_avg_by_cap,
    # cur_avg_by_vol, cur_min,cur_max, time
    my_content = convert_to_csv2([StepHeaderKey + StepHeaderValue], my_content)
    return my_content


# TODO(harvey): rename function
def ExportCycle(my_cycle_data):
    my_content = []
    for cycle in my_cycle_data.keys():
        my_content.append(
            [str(cycle)] + [str(field) for field in my_cycle_data[cycle]]
        )

    my_content = np.array(my_content)

    # cap, v_avg, v_min, v_max, cur_avg_by_cap,
    # cur_avg_by_vol, cur_min,cur_max, time
    my_content = convert_to_csv2(
        [CycleHeaderKey + CycleHeaderValue], my_content,
    )
    return my_content


lab_header = [
    "Cycle Number", "Charge Capacity (mAh)", "Discharge Capacity (mAh)",
    "Delta V (V)", "Average Charge Voltage (V)",
    "Average Discharge Voltage (V)", "Charge Time (hours)",
    "Discharge Time (hours)", "Charge Cumulative Time (hours)",
    "Discharge Cumulative Time (hours)", "Normalized Charge Capacity",
    "Normalized Discharge Capacity", "Zeroed Delta V (V)", "S (V)", "R (V)",
    "Zeroed S (V)", "Zeroed R (V)"
]

first_header_line = []
cc_header = ["First C/20", "Second C/20", "C/2", "C", "2C", "3C"]
for header_i in cc_header:
    first_header_line.extend([header_i] + (14 + 2) * [""])
second_header_line = 6 * lab_header


# TODO(harvey): rename function
def ExportRateMaps(my_rate_maps):
    my_content = np.array(
        [
            [str(my_rate_maps_i_i) for my_rate_maps_i_i in my_rate_maps_i]
            for my_rate_maps_i in my_rate_maps
        ]
    )

    my_content = convert_to_csv2(
        [first_header_line, second_header_line], my_content,
    )
    return my_content


# TODO(harvey): rename function
def ExportRobyPattern(my_rate_maps):
    my_content = np.array(
        [
            [str(my_rate_maps_i_i) for my_rate_maps_i_i in my_rate_maps_i]
            for my_rate_maps_i in my_rate_maps
        ]
    )

    my_content = convert_to_csv2([lab_header], my_content)
    return my_content


# TODO(harvey): rename function
def ExportSeparateRate(my_rate_maps):
    my_content = np.array(
        [
            [str(my_rate_maps_i_i) for my_rate_maps_i_i in my_rate_maps_i]
            for my_rate_maps_i in my_rate_maps
        ]
    )

    my_content = convert_to_csv2([lab_header], my_content)
    return my_content
