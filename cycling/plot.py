from plot import *
from cycling.models import *
from django.db.models import Q, Max, Min
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_file_legends_and_vertical(
    ax, cell_id, lower_cycle = None, upper_cycle = None, show_invalid = False,
    vertical_barriers = None, list_all_options = None, leg1 = None,
):
    files_cell_id = CyclingFile.objects.filter(
        database_file__deprecated = False,
        database_file__valid_metadata__cell_id = cell_id,
    ).order_by("database_file__last_modified")

    file_leg = []
    if len(files_cell_id) >= 1:
        for f_i, f in enumerate(files_cell_id):
            offset_cycle = f.database_file.valid_metadata.start_cycle
            if show_invalid:
                min_cycle = offset_cycle + Cycle.objects.filter(
                    cycling_file = f
                ).aggregate(Min("cycle_number"))["cycle_number__min"]
                max_cycle = offset_cycle + Cycle.objects.filter(
                    cycling_file = f
                ).aggregate(Max("cycle_number"))["cycle_number__max"]

            else:
                min_cycle = offset_cycle + Cycle.objects.filter(
                    cycling_file = f, valid_cycle = True,
                ).aggregate(Min("cycle_number"))["cycle_number__min"]
                max_cycle = offset_cycle + Cycle.objects.filter(
                    cycling_file = f, valid_cycle = True,
                ).aggregate(Max("cycle_number"))["cycle_number__max"]

            if lower_cycle is not None:
                if min_cycle < lower_cycle:
                    min_cycle = lower_cycle - .5
                if min_cycle > upper_cycle:
                    continue

            if upper_cycle is not None:
                if max_cycle > upper_cycle:
                    max_cycle = upper_cycle + .5
                if max_cycle < lower_cycle:
                    continue

            bla = plt.axvspan(
                min_cycle, max_cycle, ymin = .05 * (1 + f_i),
                ymax = .05 * (2 + f_i),
                facecolor = plot_constants.COLORS[f_i],
                alpha = 0.1
            )
            file_leg.append(
                (
                    bla,
                    "File {} Last Modif: {}-{}-{}. Size: {}KB".format(
                        f_i,
                        f.database_file.last_modified.year,
                        f.database_file.last_modified.month,
                        f.database_file.last_modified.day,
                        int(f.database_file.filesize / 1024),
                    ),
                )
            )

    if vertical_barriers is not None:
        for index_set_i in range(len(vertical_barriers) + 1):
            col = ["1.", ".1"][index_set_i % 2]
            if index_set_i == 0 and len(vertical_barriers) > 0:
                min_x, max_x = (lower_cycle - 0.5, vertical_barriers[0])
            elif index_set_i == 0 and len(vertical_barriers) == 0:
                min_x, max_x = (lower_cycle - 0.5, upper_cycle + 0.5)
            elif index_set_i == len(vertical_barriers):
                min_x, max_x = (vertical_barriers[-1], upper_cycle + 0.5)
            else:
                min_x, max_x = (
                    vertical_barriers[index_set_i - 1],
                    vertical_barriers[index_set_i],
                )
            print(min_x, max_x)
            ax.axvspan(min_x, max_x, facecolor = col, alpha = 0.1)
            plt.text(
                0.9 * min_x + .1 * max_x,
                .99 * ax.get_ylim()[0] + .01 * ax.get_ylim()[1],
                list_all_options[index_set_i],
                size = 18,
            )

        for index_set_i in range(len(list_all_options) - 1):
            plt.axvline(
                x = vertical_barriers[index_set_i],
                color = "k", linestyle = "--",
            )

    ax.tick_params(
        direction = "in", length = 7, width = 2, labelsize = 11,
        bottom = True, top = True, left = True, right = True,
    )

    if len(file_leg) > 0:
        if list_all_options is None:
            loc = "lower left"
        else:
            loc = "upper left"
        ax.legend([x[0] for x in file_leg], [x[1] for x in file_leg], loc = loc)
        ax.add_artist(leg1)




def compute_from_database(
    cell_id, lower_cycle = None, upper_cycle = None, valid = True,
):
    rules = {}
    for cycle_group in get_discharge_groups_from_cell_id(cell_id):
        rules[(
            cycle_group.constant_rate, cycle_group.end_rate_prev,
            cycle_group.end_rate, cycle_group.end_voltage,
            cycle_group.end_voltage_prev, cycle_group.polarity,
        )]={
            DISCHARGE:{'direct_id':cycle_group.id},
        }

    field_request = [
        (Key.N, 'f4', "CYCLE_NUMBER", None),
        ("last_cc_capacity", 'f4', "CUSTOM", lambda cyc: -cyc.dchg_total_capacity),
    ]
    return compute_from_database2(cell_id, rules, field_request, lower_cycle, upper_cycle, valid)


def generic_field(tp, f, cyc, offset_cycle=None, cell_first_time=None, reference_delta_v=None, reference_capacity=None, typical_capacity=None):
    if tp == "CUSTOM":
        return f(cyc)
    elif tp == "CYCLE_NUMBER":
        return float(cyc.cycle_number + offset_cycle)
    elif tp == "CUMULATIVE_TIME":
        return (cyc.get_start_time() - cell_first_time).total_seconds() / (60. * 60.)
    elif tp == "NORMALIZED_DELTA_V":
        return cyc.get_delta_v()/reference_delta_v
    elif tp == "NORMALIZED_CAPACITY":
        return f(cyc)/reference_capacity
    elif tp == "NORMALIZED_TYPICAL_CAPACITY":
        return f(cyc)/typical_capacity




def compute_from_database2(
    cell_id, rules, field_request, lower_cycle = None, upper_cycle = None, valid = True,
):
    """


    Args:
        cell_id:
        rules: a dictionary where the key can be anything. the value is a dictionary which can have
           DISCHARGE: a dictionary which contains
                 'no_group':True/False,
                 'direct_id': the id of a perfect match
                 'constant_rate': (None/val,None/val)
                 'end_rate': ..
                 'end_rate_prev': ..
                 'end_voltage': ..
                 'end_voltage_prev': ..
           CHARGE: a dictionary which contains
                 'no_group':True/False,
                 'constant_rate': (None/val,None/val)
                 'end_rate': ..
                 'end_rate_prev': ..
                 'end_voltage': ..
                 'end_voltage_prev': ..
        field_request: a list of tuples containing:
             (
                key (the key to use in dtype),
                return_type (the type used in dtype)
                function_type (either "CUSTOM" or a known type)
                a function in case type was custom (the function takes cyc as input)
             )
        lower_cycle:
        upper_cycle:
        valid:

    Returns:

    """
    files_cell_id = CyclingFile.objects.filter(
        database_file__deprecated = False,
        database_file__valid_metadata__cell_id = cell_id,
    ).order_by("database_file__last_modified")

    cell_first_time = get_cell_id_first_time(cell_id)
    reference_delta_v, typical_q = get_reference_delta_v_and_typical_capacity(cell_id)
    total_capacity = 1.
    if CellGlobals.objects.filter(cell_id=cell_id).exists():
        total_capacity = CellGlobals.objects.get(cell_id=cell_id).theoretical_capacity

    groups = {}
    for rule_id in rules.keys():
        q_curves = []
        for f in files_cell_id:
            failed = False
            offset_cycle = f.database_file.valid_metadata.start_cycle
            filters = Q(valid_cycle = valid) & Q(cycling_file = f)
            if not (lower_cycle is None and upper_cycle is None):
                filters = filters & Q(
                    cycle_number__range = (
                        lower_cycle - offset_cycle, upper_cycle - offset_cycle,
                    ),
                )

            rule = rules[rule_id]

            for polarity, group_is_none, group_id_f, group_in_f in [
                (
                    DISCHARGE,
                    Q(discharge_group=None),
                    lambda x: Q(discharge_group__id=x),
                    lambda x: Q(discharge_group__in=x),
                ),
                (
                    CHARGE,
                    Q(charge_group=None),
                    lambda x: Q(charge_group__id=x),
                    lambda x: Q(charge_group__in=x),
                ),
            ]:
                if polarity in rule.keys():
                    if 'no_group' in rule[polarity].keys() and rule[polarity]['no_group']:
                        filters = group_is_none & filters
                    elif 'direct_id' in rule[polarity].keys():
                        filters = group_id_f(rule[polarity]['direct_id']) & filters
                    else:
                        group_filter = Q(cell_id=cell_id, polarity=polarity)
                        for lab, leq_f, geq_f, range_f in [
                            (
                                'constant_rate',
                                lambda x: Q(constant_current__leq=x*total_capacity),
                                lambda x: Q(constant_current__geq=x*total_capacity),
                                lambda x: Q(constant_current__range=(x[0]*total_capacity, x[1]*total_capacity)),
                            ),
                            (
                                'end_rate',
                                lambda x: Q(end_current__leq=x*total_capacity),
                                lambda x: Q(end_current__geq=x*total_capacity),
                                lambda x: Q(end_current__range=(x[0]*total_capacity, x[1]*total_capacity)),
                            ),
                            (
                                'end_rate_prev',
                                lambda x: Q(end_current_prev__leq=x*total_capacity),
                                lambda x: Q(end_current_prev__geq=x*total_capacity),
                                lambda x: Q(end_current_prev__range=(x[0]*total_capacity, x[1]*total_capacity)),
                            ),
                            (
                                'end_voltage',
                                lambda x: Q(end_voltage__leq=x),
                                lambda x: Q(end_voltage__geq=x),
                                lambda x: Q(end_voltage__range=x),
                            ),
                            (
                                'end_voltage_prev',
                                lambda x: Q(end_voltage_prev__leq=x),
                                lambda x: Q(end_voltage_prev__geq=x),
                                lambda x: Q(end_voltage_prev__range=x),
                            ),

                        ]:
                            if lab in rule[polarity].keys():
                                if rule[polarity][lab][0] is None:
                                    group_filter = group_filter & leq_f(rule[polarity][lab][1])
                                elif rule[polarity][lab][1] is None:
                                    group_filter = group_filter & geq_f(rule[polarity][lab][0])
                                else:
                                    group_filter = group_filter & range_f(rule[polarity][lab])

                        good_groups = CycleGroup.objects.filter(group_filter)
                        if good_groups.exists():
                            filters = group_in_f(good_groups) & filters
                        else:
                            failed = True
                            break
            if not failed:
                cycles = Cycle.objects.filter(filters)
                if cycles.exists():
                    q_curves += list([
                        tuple(
                            [
                                generic_field(tp, f, cyc, offset_cycle, cell_first_time, reference_delta_v, total_capacity, typical_q)
                                for _, _, tp, f in field_request
                            ]
                        )
                        for cyc in cycles.order_by("cycle_number")
                        ]
                    )

        if len(q_curves) > 0:
            groups[rule_id] = np.array(
                q_curves,
                dtype = [
                    (key, return_type)
                    for key, return_type, _, _ in field_request
                ],
            )

    return groups





def data_engine_database(
    target: str, data, typ, generic_map, mode, max_cyc_n,
    lower_cycle = None, upper_cycle = None,
):
    """ TODO(harvey)
    Args: TODO(harvey)
        source: Specifies the source of the data to be plot: "model",
            "database", or "compiled".
        target: Plot type - "generic_vs_capacity" or "generic_vs_cycle".
    Returns: TODO(harvey)
    """

    if typ != "dchg" or mode != "cc":
        return None, None, None
    cell_id, valid = data
    generic = compute_from_database(
        cell_id, lower_cycle, upper_cycle, valid,
    )
    list_of_keys = get_list_of_keys(generic.keys(), typ)

    return generic, list_of_keys #TODO returned generic_map before



def plot_cycling_direct(
    cell_id, path_to_plots = None, lower_cycle = None, upper_cycle = None,
    show_invalid = False, vertical_barriers = None, list_all_options = None,
    figsize = None, dpi=300,
):
    if show_invalid:
        data_streams = [
            ('database', (cell_id, True), 'scatter_valid', 100),
            ('database', (cell_id, False), 'scatter_invalid', 100),
        ]
    else:
        data_streams = [
            ('database', (cell_id, True), 'scatter_valid', 100),
        ]

    if path_to_plots is None:
        return plot_engine_direct(
            data_streams = data_streams,
            target = "generic_vs_cycle",
            todos = [("dchg", "cc")],
            fit_args = {'path_to_plots': path_to_plots},

            lower_cycle = lower_cycle,
            upper_cycle = upper_cycle,
            vertical_barriers = vertical_barriers,
            list_all_options = list_all_options,
            show_invalid = show_invalid,
            figsize = figsize,
            known_data_engines={
                "database": data_engine_database,
            },
            send_to_file=False,
            dpi=dpi,
            make_file_legends_and_vertical=make_file_legends_and_vertical,

        )
    else:
        plot_engine_direct(
            data_streams = data_streams,
            target = "generic_vs_cycle",
            todos = [("dchg", "cc")],
            fit_args = {'path_to_plots': path_to_plots},
            filename = "Initial_{}.png".format(cell_id),
            lower_cycle = lower_cycle,
            upper_cycle = upper_cycle,
            vertical_barriers = vertical_barriers,
            list_all_options = list_all_options,
            show_invalid = show_invalid,
            figsize = figsize,
            known_data_engines={
                "database": data_engine_database,
            },
            send_to_file=True,
            dpi=dpi,
            make_file_legends_and_vertical=make_file_legends_and_vertical,

        )
