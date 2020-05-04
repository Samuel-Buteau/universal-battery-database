class Key:
    """ Define dictionary keys """

    N = "cycle_number"
    N_SAMPLE = "n_sample"

    # TODO: Why are there keys with the same values except "_vector"?
    #       Are they the same values just different types?
    #           If so, maybe create a new class to store both values
    #           (don't be obsessed with primitive values)
    #       Are they different values?
    #           If so, give them different and more descriptive names

    V_CC = "cc_voltages"
    V_CC_VEC = "cc_voltage_vector"
    V_CC_LAST = "last_cc_voltage"
    V_CC_LAST_AVG = "avg_last_cc_voltage"
    V_END = "end_voltage"
    V_PREV_END = "end_voltage_prev"
    V_PREV_END_AVG = "avg_end_voltage_prev"
    V_END_AVG = "avg_end_voltage"
    V_MIN_GRID = "voltage_grid_min_v"
    V_MAX_GRID = "voltage_grid_max_v"
    V_N_GRID = "voltage_grid_n_samples"

    Q_CC = "cc_capacities"
    Q_CC_VEC = "cc_capacity_vector"
    Q_CV = "cv_capacities"
    Q_CV_VEC = "cv_capacity_vector"
    Q_CC_LAST = "last_cc_capacity"
    Q_CV_LAST = "last_cv_capacity"

    I_CC = "constant_current"
    I_CC_AVG = "avg_constant_current"
    I_CV = "cv_currents"
    I_CV_VEC = "cv_current_vector"
    I_END = "end_current"
    I_END_AVG = "avg_end_current"
    I_PREV = "end_current_prev"
    I_PREV_END_AVG = "avg_end_current_prev"
    I_MAX = "current_max_n"
    I_MIN_GRID = "current_grid_min_v"
    I_MAX_GRID = "current_grid_max_v"
    I_N_GRID = "current_grid_n_samples"

    TEMP = "temperature"
    TEMP_GRID_MIN_V = "temperature_grid_min_v"
    TEMP_GRID_MAX_V = "temperature_grid_max_v"
    TEMP_GRID_N = "temperature_grid_n_samples"

    MASK_CC = "cc_masks"
    MASK_CC_VEC = "cc_mask_vector"
    MASK_CV = "cv_masks"
    MASK_CV_VEC = "cv_mask_vector"
    COUNT_MATRIX = "count_matrix"
    SVIT_GRID = "svit_grid"

    WIDTH = "width"
    DEPTH = "depth"
    LRN_RATE = "learning_rate"
    BATCH = "batch_size"
    PATH_DATASET = "path_to_dataset"
    PATH_PLOTS = "path_to_plots"
    PATH_V_CURVES = "path_v_curves"
    PATH_V_CURVES_META = "path_v_curves_meta"
    DATA_VERSION = "dataset_version"
    BARCODES = "wanted_barcodes"
    STOP = "stop_count"
    VIS = "visualize_fit_every"
    PRINT_LOSS = "print_loss_every"
    MIN_LAT = "min_latent"
    GLB_NORM_CLIP = "global_norm_clip"

    # Begin: fit_args-related keys =============================================

    FIT_ARGS = "fit_args"

    # Begin: keys from complie_dataset.Command.add_arguments ===================

    MIN_V_GRID = "voltage_grid_min_v"
    MAX_V_GRID = "voltage_grid_max_v"

    # End: keys from complie_dataset.Command.add_arguments =====================

    """ Structured array with dtype:
        [
            (Key.N, "f4"),
            (Key.V_CC, "f4", len(voltage_grid)),
            (Key.Q_CC, "f4", len(voltage_grid)),
            (Key.MASK_CC, "f4", len(voltage_grid)),
            (Key.I_CV, "f4", fit_args[Key.I_MAX]),
            (Key.Q_CV, "f4", fit_args[Key.I_MAX]),
            (Key.MASK_CV, "f4", fit_args[Key.I_MAX]),
            (Key.I_CC, "f4"),
            (Key.I_PREV, "f4"),
            (Key.I_END, "f4"),
            (Key.V_PREV_END, "f4"),
            (Key.V_END, "f4"),
            (Key.V_CC_LAST, "f4"),
            (Key.Q_CC_LAST, "f4"),
            (Key.Q_CV_LAST, "f4"),
            (Key.TEMP, "f4"),
        ]
    """
    MAIN = "main_data"
    DATASET = "my_data"

    """
    Groups of steps indexed by group averages of
        ( end_current_prev, constant_current, end_current,
          end_voltage_prev, end_voltage, sign )
        Each group is a dictionary indexed by:
        [ Key.MAIN, Key.I_CC_AVG, Key.I_PREV_END_AVG, Key.Q_END_AVG,
          Key.V_PREV_END_AVG, Key.V_END_AVG, Key.V_CC_LAST_AVG ]
    """
    CYC_GRP_DICT = "cyc_grp_dict"

    """ Structured array with dtype:
        [
            (N, "f4"),
            (
                COUNT_MATRIX, "f4",
                (
                    len(sign_grid), len(voltage_grid_degradation),
                    len(current_grid), len(temperature_grid),
                )
            ),
        ]
    """
    REF_ALL_MATS = "all_reference_mats"
    REF_CYC = "reference_cycles_n"

    # Begin: keys inside dataset ===============================================

    """ (dict) Indexed by barcode, each yields a dictionary with keys:
        [ Key.ALL_REF_MATS, Key.CYC_GRP_DICT ]
    """
    ALL_DATA = "all_data"

    # (number) The maximum capacity across the dataset.
    MAX_Q = "max_cap"

    """ (1d array) Measured voltages """
    GRID_V = "voltage_grid"
    GRID_I = "current_grid"
    """ (1d array) Temperatures """
    GRID_T = "temperature_grid"
    """ (1d array) Signs """
    GRID_SIGN = "sign_grid"

    # (dict) Indexed by barcode; yields a positive electrode id.
    CELL_TO_POS = "cell_id_to_pos_id"
    # (dic) Indexed by barcode; yields a negative electrode id.
    CELL_TO_NEG = "cell_id_to_neg_id"
    # (dic) Indexed by barcode; yields an electrolyte id.
    CELL_TO_LYTE = "cell_id_to_electrolyte_id"
    # (dic) Indexed by barcode; yields
    #   1 if the cell is latent, 0 if made of known pos, neg, electrolyte.
    CELL_TO_LAT = "cell_id_to_latent"
    CELL_TO_DRY = "cell_to_dry"

    DRY_TO_META = "dry_to_meta"

    LYTE_TO_SOL = "electrolyte_id_to_solvent_id_weight"
    LYTE_TO_SALT = "electrolyte_id_to_salt_id_weight"
    LYTE_TO_ADD = "electrolyte_id_to_additive_id_weight"
    LYTE_TO_LAT = "electrolyte_id_to_latent"

    # Begin: keys for dataset names ============================================

    NAME_DRY = "dry_to_dry_name"
    NAME_POS = "pos_to_pos_name"
    NAME_LYTE = "electrolyte_to_electrolyte_name"
    NAME_NEG = "neg_to_neg_name"
    NAME_MOL = "molecule_to_molecule_name"

    # End: keys for dataset names ==============================================

    STRAT = "strategy"
    TENSORS = "compiled_tensors"
    MODEL = "degradation_model"
    TRAIN_DS = "train_ds"
    CYC_M = "cycle_m"
    CYC_V = "cycle_v"
    OPT = "optimizer"

    # TODO (harvey): replace the following class of keys
    class Coeff:
        Q = "coeff_q"
        Q_CV = "coeff_cv_capacity"
        Q_CC = "coeff_cc_capacity"
        V_CC = "coeff_cc_voltage"
        SCALE = "coeff_scale"
        R = "coeff_r"
        SHIFT = "coeff_shift"
        CELL = "coeff_cell"
        RECIP = "coeff_reciprocal"
        PROJ = "coeff_projection"
        OOB = "coeff_out_of_bounds"

    # TODO (harvey): replace the following class of keys
    class Loss:
        Q = "q_loss"
        Q_CC = "cc_capacity_loss"
        Q_CV = "cv_capacity_loss"
        V_CC = "cc_voltage_loss"
        V_CV = "cv_voltage_loss"
        SCALE = "scale_loss"
        R = "r_loss"
        SHIFT = "shift_loss"
        CELL = "cell_loss"
        RECIP = "reciprocal_loss"
        PROJ = "projection_loss"
        OOB = "out_of_bounds_loss"
