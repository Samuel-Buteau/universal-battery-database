import numpy
from django.core.management.base import BaseCommand

from cycling.neware_processing_functions import *
from cell_database.models import *
from Key import Key


def get_dry_cell_metadata(cell_id):
    """ TODO(harvey)
    Args:
        cell_id: TODO(harvey)
    Returns: TODO(harvey)
    """
    wet_cells = WetCell.objects.filter(cell_id = cell_id)
    if wet_cells.exists():
        wet_cell = wet_cells[0]
        dry_cell = wet_cell.dry_cell
        if dry_cell is None:
            return None, None, None
        dry_cell = dry_cell.dry_cell
        if dry_cell is None:
            return None, None, None

        meta = {}  # metadata
        dry_cell_id = dry_cell.id
        dry_cell_str = str(dry_cell)
        if dry_cell.proprietary:
            return dry_cell_id, {}, dry_cell_str

        if dry_cell.cathode_geometry is not None:
            if dry_cell.cathode_geometry.loading is not None:
                meta["cathode_loading"] = dry_cell.cathode_geometry.loading
            if dry_cell.cathode_geometry.density is not None:
                meta["cathode_density"] = dry_cell.cathode_geometry.density
            if dry_cell.cathode_geometry.thickness is not None:
                meta["cathode_thickness"]\
                    = dry_cell.cathode_geometry.thickness / 1000.

        if dry_cell.anode_geometry is not None:
            if dry_cell.anode_geometry.loading is not None:
                meta["anode_loading"] = dry_cell.anode_geometry.loading
            if dry_cell.anode_geometry.density is not None:
                meta["anode_density"] = dry_cell.anode_geometry.density
            if dry_cell.anode_geometry.thickness is not None:
                meta["anode_thickness"]\
                    = dry_cell.anode_geometry.thickness / 1000.

        return dry_cell_id, meta, dry_cell_str

    return None, None, None


def get_cathod_id(cell_id):
    """ TODO(harvey)
    Args:
        cell_id: TODO(harvey)
    Returns: TODO(harvey)
    """
    wet_cells = WetCell.objects.filter(cell_id = cell_id)
    if wet_cells.exists():
        wet_cell = wet_cells[0]
        dry_cell = wet_cell.dry_cell
        if dry_cell is None:
            return None, None
        dry_cell = dry_cell.dry_cell
        if dry_cell is None:
            return None, None
        cathode = dry_cell.cathode
        if cathode is None:
            return None, None
        else:
            return cathode.id, str(cathode)

    return None, None


def get_anode_id(cell_id):
    """ TODO(harvey)
    Args:
        cell_id: TODO(harvey)
    Returns: TODO(harvey)
    """
    wet_cells = WetCell.objects.filter(cell_id = cell_id)
    if wet_cells.exists():
        wet_cell = wet_cells[0]
        dry_cell = wet_cell.dry_cell
        if dry_cell is None:
            return None, None
        dry_cell = dry_cell.dry_cell
        if dry_cell is None:
            return None, None
        anode = dry_cell.anode
        if anode is None:
            return None, None
        else:
            return anode.id, str(anode)

    return None, None


def get_electrolyte_id_from_cell_id(cell_id):
    """ TODO(harvey)
    Args:
        cell_id: TODO(harvey)
    Returns: TODO(harvey)
    """
    wet_cells = WetCell.objects.filter(cell_id = cell_id)
    if wet_cells.exists():
        wet_cell = wet_cells[0]
        electrolyte = wet_cell.electrolyte
        if electrolyte is None:
            return None, None
        else:
            return electrolyte.id, str(electrolyte)

    return None, None


def get_component_from_electrolyte(electrolyte):
    """ TODO(harvey)
    Args:
        electrolyte: TODO(harvey)
    Returns: TODO(harvey)
    """
    weight_dict = {"solvent": {}, "salt": {}, "additive": {}}
    name_dict = {"solvent": {}, "salt": {}, "additive": {}}
    electrolyte_lots = CompositeLot.objects.filter(id = electrolyte)
    if electrolyte_lots.exists():
        electrolyte_lot = electrolyte_lots[0]
        if electrolyte_lot.composite is not None:
            my_electrolyte = electrolyte_lot.composite
            my_components = my_electrolyte.components

            for component in my_components.order_by("id"):
                for st, s in [
                    ("solvent", SOLVENT), ("salt", SALT), ("additive", ADDITIVE)
                ]:
                    if component.component_lot.component.component_type == s:
                        name_dict[st][component.component_lot.id]\
                            = str(component.component_lot)
                        weight_dict[st][component.component_lot.id]\
                            = [component.ratio]

    return weight_dict, name_dict


def make_my_cell_ids(options):
    """ TODO(harvey)
    Args:
        options: TODO(harvey)
    Returns: TODO(harvey)
    """
    my_cell_ids = CyclingFile.objects.filter(
        database_file__deprecated = False,
        database_file__is_valid = True
    ).exclude(
        database_file__valid_metadata = None
    ).order_by(
        "database_file__valid_metadata__cell_id"
    ).values_list(
        "database_file__valid_metadata__cell_id",
        flat = True
    ).distinct()

    used_cell_ids = []
    for b in my_cell_ids:
        if (
            CycleGroup.objects.filter(cell_id = b).exists()
        ):
            used_cell_ids.append(b)

    if len(options[Key.CELL_IDS]) == 0:
        return used_cell_ids
    else:
        return list(
            set(used_cell_ids).intersection(set(options[Key.CELL_IDS]))
        )


# TODO (harvey): reformat docstring
def initial_processing(cell_ids, options, flags):
    """

    Returns:
        Two dictionaries with the following sets of keys
        { Key.Q_MAX, Key.ALL_DATA, Key.V_GRID, Key.I_GRID, Key.TEMP_GRID,
          Key.SIGN_GRID, Key.CELL_TO_POS, Key.CELL_TO_NEG, Key.CELL_TO_LYTE,
          Key.CELL_TO_DRY, Key.DRY_TO_META, Key.CELL_TO_LAT, Key.LYTE_TO_LAT,
          Key.LYTE_TO_SOL, Key.LYTE_TO_SALT, Key.LYTE_TO_ADD },
        { Key.NAME_POS, Key.NAME_NEG, Key.NAME_LYTE, Key.NAME_MOL,
          Key.NAME_DRY }
    """

    all_data = {}

    voltage_grid_degradation = make_voltage_grid(
        options[Key.V_MIN_GRID],
        options[Key.V_MAX_GRID],
        int(options[Key.V_N_GRID] / 4),
        cell_ids
    )

    current_grid = make_current_grid(
        options[Key.I_MIN_GRID],
        options[Key.I_MAX_GRID],
        options[Key.I_N_GRID],
        cell_ids
    )

    temperature_grid = make_temperature_grid(
        options[Key.TEMP_GRID_MIN_V],
        options[Key.TEMP_GRID_MAX_V],
        options[Key.TEMP_GRID_N],
        cell_ids
    )
    sign_grid = make_sign_grid()
    """
    - cycles are grouped by their charge rates and discharge rates.
    - a cycle group contains many cycles
    - things are split up this way to sample each group equally
    - each cell_id corresponds to a single cell
    """
    for cell_id in cell_ids:
        """
        - dictionary indexed by charging and discharging rate (i.e. cycle group)
        - contains structured arrays of
            - cycle_number
            - capacity_vector: a vector where each element is a
              capacity associated with a given voltage
              [(voltage_grid[i], capacity_vector[i])
              is a voltage-capacity pair]
            - vq_curve_mask: a vector where each element is a weight
              corresponding to a voltage-capacity pair
              [this allows us to express the fact that sometimes a given
              voltage was not measured, so the capacity is meaningless.
              (mask of 0)]
        """

        files = get_files_for_cell_id(cell_id)

        all_mats = []
        for cyc in Cycle.objects.filter(cycling_file__in = files).order_by(
            Key.N):
            count_matrix = get_count_matrix(
                cyc, voltage_grid_degradation, current_grid, temperature_grid,
                sign_grid,
            )
            true_cycle = cyc.get_offset_cycle()
            # for each cycle, call COUNT_MATRIX,
            # and get (true_cyc, COUNT_MATRIX) list
            if count_matrix is None:
                continue
            all_mats.append((true_cycle, count_matrix))

        all_mats = numpy.array(
            all_mats,
            dtype = [
                (Key.N, "f4"),
                (
                    Key.COUNT_MATRIX, "f4",
                    (
                        len(sign_grid), len(voltage_grid_degradation),
                        len(current_grid), len(temperature_grid),
                    ),
                ),
            ]
        )

        min_cycle = numpy.min(all_mats[Key.N])
        max_cycle = numpy.max(all_mats[Key.N])
        delta_cycle = (max_cycle - min_cycle) / float(options[Key.REF_CYC_N])

        reference_cycles = numpy.linspace(
            min_cycle + delta_cycle, max_cycle, float(options[Key.REF_CYC_N]),
        )

        all_reference_mats = []
        # then for reference cycle,
        # mask all cycles < reference cycle compute the average.
        for reference_cycle in reference_cycles:
            prev_matrices = all_mats[Key.COUNT_MATRIX][
                all_mats[Key.N] <= reference_cycle
                ]
            avg_matrices = numpy.average(prev_matrices)
            all_reference_mats.append((reference_cycle, avg_matrices))
            # each step points to the nearest reference cycle

        all_reference_mats = numpy.array(
            all_reference_mats,
            dtype = [
                (Key.N, "f4"),
                (
                    Key.COUNT_MATRIX, "f4",
                    (
                        len(sign_grid), len(voltage_grid_degradation),
                        len(current_grid), len(temperature_grid),
                    ),
                ),
            ]
        )

        cyc_grp_dict = {}
        for typ in [CHARGE, DISCHARGE]:
            if typ == DISCHARGE:
                sign = -1.
            else:
                sign = 1.

            groups = CycleGroup.objects.filter(
                cell_id = cell_id, polarity = typ
            ).order_by("constant_rate")

            for cyc_group in groups:
                result = []

                future_key = (
                    cyc_group.constant_rate, cyc_group.end_rate_prev,
                    cyc_group.end_rate, cyc_group.end_voltage,
                    cyc_group.end_voltage_prev, typ,
                )

                if any([
                    abs(cyc_group.end_rate_prev) < 1e-5,
                    abs(cyc_group.constant_rate) < 1e-5,
                    abs(cyc_group.end_rate) < 1e-5,
                    options["voltage_grid_min_v"] > cyc_group.end_voltage,
                    options["voltage_grid_max_v"] < cyc_group.end_voltage,
                    options["voltage_grid_min_v"] > cyc_group.end_voltage_prev,
                    options["voltage_grid_max_v"] < cyc_group.end_voltage_prev,
                ]):
                    continue

                for cyc in cyc_group.cycle_set.order_by(Key.N):
                    if cyc.valid_cycle:
                        offset_cycle = cyc.get_offset_cycle()
                        # Check if flagged.
                        flagged = False
                        for flag_type in flags.keys():
                            list_of_flags = flags[flag_type]
                            if len(list_of_flags) == 0:
                                continue
                            list_of_flags = [
                                fs for fs in list_of_flags
                                if fs["cell_id"] == cell_id
                            ]
                            if len(list_of_flags) == 0:
                                continue

                            list_of_flags = [
                                fs for fs in list_of_flags
                                if fs[Key.CYC] == offset_cycle
                            ]
                            if len(list_of_flags) == 0:
                                continue
                            list_of_flags = [
                                fs for fs in list_of_flags
                                if fs["group"] == future_key
                            ]
                            if len(list_of_flags) == 0:
                                continue

                            flagged = True
                            print("Flags were triggered:")
                            print(list_of_flags)
                            break

                        post_process_results = ml_post_process_cycle(
                            cyc, options[Key.V_N_GRID], typ,
                            current_max_n = options[Key.I_MAX],
                            voltage_grid_min_v = options["voltage_grid_min_v"],
                            voltage_grid_max_v = options["voltage_grid_max_v"],
                            flagged = flagged,
                        )

                        if post_process_results is None:
                            continue

                        result.append((
                            cyc.get_offset_cycle(),
                            post_process_results[Key.V_CC],
                            post_process_results[Key.Q_CC],
                            post_process_results[Key.MASK_CC],
                            sign * post_process_results[Key.I_CV],
                            post_process_results[Key.Q_CV],
                            post_process_results[Key.MASK_CV],
                            sign * post_process_results[Key.I_CC],
                            -sign * post_process_results[Key.I_PREV_END],
                            sign * post_process_results[Key.I_END],
                            post_process_results[Key.V_PREV_END],
                            post_process_results[Key.V_END],
                            post_process_results[Key.V_CC_LAST],
                            post_process_results[Key.Q_CC_LAST],
                            post_process_results[Key.Q_CV_LAST],
                            cyc.get_temperature(),
                        ))

                res = numpy.array(
                    result,
                    dtype = [
                        (Key.N, "f4"),

                        (Key.V_CC_VEC, "f4", options[Key.V_N_GRID]),
                        (Key.Q_CC_VEC, "f4", options[Key.V_N_GRID]),
                        (Key.MASK_CC_VEC, "f4", options[Key.V_N_GRID]),

                        (Key.I_CV_VEC, "f4", options[Key.I_MAX]),
                        (Key.Q_CV_VEC, "f4", options[Key.I_MAX]),
                        (Key.MASK_CV_VEC, "f4", options[Key.I_MAX]),

                        (Key.I_CC, "f4"),
                        (Key.I_PREV_END, "f4"),
                        (Key.I_END, "f4"),
                        (Key.V_PREV_END, "f4"),
                        (Key.V_END, "f4"),
                        (Key.V_CC_LAST, "f4"),
                        (Key.Q_CC_LAST, "f4"),
                        (Key.Q_CV_LAST, "f4"),
                        (Key.TEMP, "f4"),
                    ]
                )

                if len(res) > 0:
                    cyc_grp_dict[(
                        cyc_group.constant_rate, cyc_group.end_rate_prev,
                        cyc_group.end_rate, cyc_group.end_voltage,
                        cyc_group.end_voltage_prev, typ,
                    )] = {
                        Key.MAIN: res,
                        Key.I_CC_AVG: numpy.average(res[Key.I_CC]),
                        Key.I_PREV_END_AVG: numpy.average(res[Key.I_PREV_END]),
                        Key.I_END_AVG: numpy.average(res[Key.I_END]),
                        Key.V_PREV_END_AVG: numpy.average(res[Key.V_PREV_END]),
                        Key.V_END_AVG: numpy.average(res[Key.V_END]),
                        Key.V_CC_LAST_AVG: numpy.average(res[Key.V_CC_LAST]),
                    }

        all_data[cell_id] = {
            Key.CYC_GRP_DICT: cyc_grp_dict,
            Key.REF_ALL_MATS: all_reference_mats,
        }

    """
    "cell_id_list": 1D array of cell_ids
    "pos_id_list": 1D array of positive electrode ids
    "neg_id_list": 1D array of negative electrode ids
    "electrolyte_id_list": 1D array of electrolyte ids
    Key.CELL_TO_POS: a dictionary indexed by cell_id yielding a positive
        electrode id.
    Key.CELL_TO_NEG: a dictionary indexed by cell_id yielding a positive
        electrode id.
    Key.CELL_TO_ELE: a dictionary indexed by cell_id yielding a
        positive electrode id.
    """

    # cell ID to cathode ID
    cell_to_cath_id = {}
    # cell ID to anode ID
    cell_to_an_id = {}
    # cell ID to electrolyte ID
    cell_to_lyte_id = {}
    # cell ID to dry cell ID
    cell_to_dry_cell_id = {}
    # dry cell ID to metadata
    dry_cell_to_meta = {}

    cell_id_to_latent = {}
    electrolyte_id_to_latent = {}
    electrolyte_id_to_solvent_id_weight = {}
    electrolyte_id_to_salt_id_weight = {}
    electrolyte_id_to_additive_id_weight = {}

    pos_to_pos_name = {}
    neg_to_neg_name = {}
    electrolyte_to_electrolyte_name = {}
    dry_cell_to_dry_cell_name = {}
    molecule_to_molecule_name = {}

    for cell_id in cell_ids:
        cathode, cathode_name = get_cathod_id(cell_id)
        anode, anode_name = get_anode_id(cell_id)
        electrolyte, electrolyte_name = get_electrolyte_id_from_cell_id(cell_id)
        dry_cell_id, dry_cell_meta, dry_cell_name\
            = get_dry_cell_metadata(cell_id)

        if (
            cathode is None or anode is None
            or electrolyte is None or dry_cell_id is None
        ):
            cell_id_to_latent[cell_id] = 1.
        else:
            pos_to_pos_name[cathode] = cathode_name
            neg_to_neg_name[anode] = anode_name
            electrolyte_to_electrolyte_name[electrolyte] = electrolyte_name
            dry_cell_to_dry_cell_name[dry_cell_id] = dry_cell_name

            cell_id_to_latent[cell_id] = 0.
            cell_to_cath_id[cell_id] = cathode
            cell_to_an_id[cell_id] = anode
            cell_to_dry_cell_id[cell_id] = dry_cell_id
            dry_cell_to_meta[dry_cell_id] = dry_cell_meta

            cell_to_lyte_id[cell_id] = electrolyte

            component_weight, component_name\
                = get_component_from_electrolyte(electrolyte)

            if (
                any([
                    s[0] is None for s in component_weight["solvent"].values()
                ]) or
                any([
                    s[0] is None for s in component_weight["salt"].values()
                ]) or
                any([
                    s[0] is None for s in component_weight["additive"].values()
                ])
            ):
                electrolyte_id_to_latent[electrolyte] = 1.
            else:
                for k in component_name["solvent"].keys():
                    molecule_to_molecule_name[k] = component_name["solvent"][k]
                for k in component_name["salt"].keys():
                    molecule_to_molecule_name[k] = component_name["salt"][k]
                for k in component_name["additive"].keys():
                    molecule_to_molecule_name[k] = component_name["additive"][k]

                electrolyte_id_to_latent[electrolyte] = 0.
                electrolyte_id_to_solvent_id_weight[electrolyte] = [
                    (sid, component_weight["solvent"][sid][0]) for sid in
                    component_weight["solvent"].keys()
                ]
                electrolyte_id_to_salt_id_weight[electrolyte] = [
                    (sid, component_weight["salt"][sid][0]) for sid in
                    component_weight["salt"].keys()
                ]
                electrolyte_id_to_additive_id_weight[electrolyte] = [
                    (sid, component_weight["additive"][sid][0]) for sid in
                    component_weight["additive"].keys()
                ]

    max_cap = 0.
    for cell_id in all_data.keys():

        cyc_grp_dict = all_data[cell_id][Key.CYC_GRP_DICT]
        # find largest cap measured for this cell (max over all cycle groups)
        for k in cyc_grp_dict.keys():
            if len(cyc_grp_dict[k][Key.MAIN][Key.Q_CC_LAST]) > 0:
                max_cap = max(
                    max_cap,
                    max(abs(cyc_grp_dict[k][Key.MAIN][Key.Q_CC_LAST]))
                )

    return {
               Key.Q_MAX: max_cap,
               Key.ALL_DATA: all_data,
               Key.V_GRID: voltage_grid_degradation,
               Key.I_GRID: current_grid,
               Key.TEMP_GRID: temperature_grid,
               Key.SIGN_GRID: sign_grid,
               Key.CELL_TO_POS: cell_to_cath_id,
               Key.CELL_TO_NEG: cell_to_an_id,
               Key.CELL_TO_LYTE: cell_to_lyte_id,
               Key.CELL_TO_DRY: cell_to_dry_cell_id,
               Key.DRY_TO_META: dry_cell_to_meta,
               Key.CELL_TO_LAT: cell_id_to_latent,
               Key.LYTE_TO_LAT: electrolyte_id_to_latent,
               Key.LYTE_TO_SOL: electrolyte_id_to_solvent_id_weight,
               Key.LYTE_TO_SALT: electrolyte_id_to_salt_id_weight,
               Key.LYTE_TO_ADD: electrolyte_id_to_additive_id_weight,
           }, {
               Key.NAME_POS: pos_to_pos_name,
               Key.NAME_NEG: neg_to_neg_name,
               Key.NAME_LYTE: electrolyte_to_electrolyte_name,
               Key.NAME_MOL: molecule_to_molecule_name,
               Key.NAME_DRY: dry_cell_to_dry_cell_name,
           }


def compile_dataset(options):
    if not os.path.exists(options[Key.PATH_DATASET]):
        os.mkdir(options[Key.PATH_DATASET])
    my_cell_ids = make_my_cell_ids(options)

    flags = {}
    if options["path_to_flags"] != '':
        flag_filename = os.path.join(options["path_to_flags"], "FLAGS.file")
        if os.path.exists(flag_filename):
            with open(flag_filename, 'rb') as file:
                flags = pickle.load(file)

    pick, pick_names = initial_processing(my_cell_ids, options, flags)
    with open(
        os.path.join(
            options[Key.PATH_DATASET],
            "dataset_ver_{}.file".format(options[Key.DATA_VERSION])
        ),
        "wb"
    ) as f:
        pickle.dump(pick, f, pickle.HIGHEST_PROTOCOL)

    with open(
        os.path.join(
            options[Key.PATH_DATASET],
            "dataset_ver_{}_names.file".format(options[Key.DATA_VERSION])
        ),
        "wb"
    ) as f:
        pickle.dump(pick_names, f, pickle.HIGHEST_PROTOCOL)


class Command(BaseCommand):

    def add_arguments(self, parser):
        required_args = [
            "--path_to_dataset",
            "--dataset_version",
        ]
        float_args = {
            "--voltage_grid_min_v": 2.5,
            "--voltage_grid_max_v": 5.0,
            "--current_grid_min_v": 1.,
            "--current_grid_max_v": 1000.,
            "--temperature_grid_min_v": 20.,
            "--temperature_grid_max_v": 60.,
        }
        int_args = {
            "--reference_cycles_n": 10,
            "--voltage_grid_n_samples": 32,
            "--current_grid_n_samples": 8,
            "--current_max_n": 8,
            "--temperature_grid_n_samples": 3,
        }

        for arg in required_args:
            parser.add_argument(arg, required = True)
        for arg in float_args:
            parser.add_argument(arg, type = float, default = float_args[arg])
        for arg in int_args:
            parser.add_argument(arg, type = int, default = int_args[arg])

        # default_cell_ids = [
        #     57706, 57707, 57710, 57711, 57714, 57715, 64260, 64268, 81602,
        #     81603, 81604, 81605, 81606, 81607, 81608, 81609, 81610, 81611,
        #     81612, 81613, 81614, 81615, 81616, 81617, 81618, 81619, 81620,
        #     81621, 81622, 81623, 81624, 81625, 81626, 81627, 81712, 81713,
        #     82300, 82301, 82302, 82303, 82304, 82305, 82306, 82307, 82308,
        #     82309, 82310, 82311, 82406, 82407, 82410, 82411, 82769, 82770,
        #     82771, 82775, 82776, 82777, 82779, 82992, 82993, 83010, 83011,
        #     83012, 83013, 83014, 83015, 83016, 83083, 83092, 83101, 83106,
        #     83107, 83220, 83221, 83222, 83223, 83224, 83225, 83226, 83227,
        #     83228, 83229, 83230, 83231, 83232, 83233, 83234, 83235, 83236,
        #     83237, 83239, 83240, 83241, 83242, 83243, 83310, 83311, 83312,
        #     83317, 83318, 83593, 83594, 83595, 83596, 83741, 83742, 83743,
        #     83744, 83745, 83746, 83747, 83748,
        # ]

        default_cell_ids = []
        parser.add_argument(
            "--wanted_cell_ids", type = int, nargs = "+",
            default = default_cell_ids,

        )
        parser.add_argument(
            "--path_to_flags", default = ""
        )

    def handle(self, *args, **options):
        compile_dataset(options)
