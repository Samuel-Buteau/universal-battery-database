class Key:
    """ Define dictionary keys used throughout the project."""

    # TODO: Why are there keys with the same values except "_vector"?
    #       Are they the same values just different types?
    #           If so, maybe create a new class to store both values
    #           (don't be obsessed with primitive values)
    #       Are they different values?
    #           If so, give them different and more descriptive names

    # TODO(harvey): What's the difference between N and CYC?
    N = "cycle_number"
    CYC = "cycle"
    CELL_FEAT = "features_cell"

    REF_CYC = "reference_cycle"
    NEIGH_DATA = "neighborhood_data"

    COUNT_BATCH = "batch_count"
    COUNT_V = "voltage_count"
    COUNT_I = "current_count"

    V = "v"
    V_CC = "cc_voltages"
    V_CC_VEC = "cc_voltage_vector"
    V_CC_LAST = "last_cc_voltage"
    V_CC_LAST_AVG = "avg_last_cc_voltage"
    V_END = "end_voltage"
    V_PREV_END = "end_voltage_prev"
    V_PREV_END_AVG = "avg_end_voltage_prev"
    V_END_AVG = "avg_end_voltage"
    V_TENSOR = "voltage_tensor"

    Q_CC = "cc_capacities"
    Q_CC_VEC = "cc_capacity_vector"
    Q_CV = "cv_capacities"
    Q_CV_VEC = "cv_capacity_vector"
    Q_CC_LAST = "last_cc_capacity"
    Q_CV_LAST = "last_cv_capacity"

    I = "current"
    I_CC = "constant_current"
    I_CC_AVG = "avg_constant_current"
    I_CV = "cv_current"
    I_CVS = "cv_currents"
    I_CV_VEC = "cv_current_vector"
    I_END = "end_current"
    I_END_AVG = "avg_end_current"
    I_PREV_END = "end_current_prev"
    I_PREV_END_AVG = "avg_end_current_prev"
    I_TENSOR = "current_tensor"

    T = "temperature"

    STRESS = "encoded_stress"

    MASK_CC = "cc_masks"
    MASK_CC_VEC = "cc_mask_vector"
    MASK_CV = "cv_masks"
    MASK_CV_VEC = "cv_mask_vector"
    COUNT_MATRIX = "count_matrix"
    """ Multi-grid of (S, V, I, T) """
    SVIT_GRID = "svit_grid"

    PATH_V_CURVES = "path_v_curves"
    PATH_V_CURVES_META = "path_v_curves_meta"
    CELL_IDS = "wanted_cell_ids"

    INDICES = "indices"

    # Begin: Keys in options (ml_smoothing and compile_dataset =================

    OPTIONS = "options"

    PATH_DATASET = "path_to_dataset"
    PATH_PLOTS = "path_to_plots"
    DATA_VERSION = "dataset_version"

    GLB_NORM_CLIP = "global_norm_clip"
    LRN_RATE = "learning_rate"
    MIN_LAT = "min_latent"
    COEFF_FEAT_CELL_DER = "coeff_d_features_cell"
    COEFF_FEAT_CELL_DER2 = "coeff_d2_features_cell"
    COEFF_CELL = "coeff_cell"
    COEFF_CELL_OUT = "coeff_cell_output"
    COEFF_CELL_IN = "coeff_cell_input"
    COEFF_CELL_DER = "coeff_cell_derivative"
    COEFF_CELL_EQ = "coeff_cell_eq"
    COEFF_LYTE = "coeff_electrolyte"
    COEFF_LYTE_OUT = "coeff_electrolyte_output"
    COEFF_LYTE_IN = "coeff_electrolyte_input"
    COEFF_LYTE_DER = "coeff_electrolyte_derivative"
    COEFF_LYTE_EQ = "coeff_electrolyte_eq"
    COEFF_Q_CV = "coeff_cv_capacity"
    COEFF_Q_CC = "coeff_cc_capacity"
    COEFF_Q = "coeff_q"
    COEFF_Q_GEQ = "coeff_q_geq"
    COEFF_Q_LEQ = "coeff_q_leq"
    COEFF_Q_V_MONO = "coeff_q_v_mono"
    COEFF_Q_DER3_V = "coeff_q_d3_v"
    COEFF_Q_DER3_I = "coeff_q_d3_current"
    COEFF_Q_DER3_N = "coeff_q_d3_cycle"
    COEFF_Q_DER_I = "coeff_q_d_current"
    COEFF_Q_DER_N = "coeff_q_d_cycle"

    MIN_V_GRID = "voltage_grid_min_v"
    MAX_V_GRID = "voltage_grid_max_v"
    I_MIN_V_GRID = "current_grid_min_v"
    I_MAX_V_GRID = "current_grid_max_v"
    T_MIN_V_GRID = "temperature_grid_min_v"
    T_MAX_V_GRID = "temperature_grid_max_v"

    N_SAMPLE = "n_sample"
    WIDTH = "width"
    DEPTH = "depth"
    BATCH = "batch_size"

    PRINT_LOSS = "print_loss_every"
    VIS_FIT = "visualize_fit_every"
    VIS_VQ = "visualize_vq_every"
    STOP = "stop_count"
    CELL_ID_SHOW = "cell_id_show"

    REF_CYC_N = "reference_cycles_n"
    V_N_GRID = "voltage_grid_n_samples"
    I_N_GRID = "current_grid_n_samples"
    I_MAX = "current_max_n"
    TEMP_GRID_N = "temperature_grid_n_samples"

    # End: Keys in options (ml_smoothing and compile_dataset ===================

    # Begin: Keys in derivatives ===============================================

    D_V = "d_v"
    D3_V = "d3_v"
    D_I = "d_current"
    D3_I = "d3_current"
    D_CYC = "d_cycle"
    D3_CYC = "d3_cycle"
    D_CELL_FEAT = "d_features_cell"
    D2_CELL_FEAT = "d2_features_cell"

    # End: Keys in derivatives =================================================

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
    # TODO(harvey): change the key
    DATASET = "dataset"

    """ Groups of steps indexed by group averages of
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

    # Begin: keys inside dataset ===============================================

    """ (dict) Indexed by cell ID, each yields a dictionary with keys:
        [ Key.ALL_REF_MATS, Key.CYC_GRP_DICT ]
    """
    ALL_DATA = "all_data"

    """ (number) The maximum capacity across the dataset """
    Q_MAX = "max_cap"

    """ (1d array) Measured voltages """
    V_GRID = "voltage_grid"
    I_GRID = "current_grid"
    """ (1d array) Temperatures """
    TEMP_GRID = "temperature_grid"
    """ (1d array) Signs """
    SIGN_GRID = "sign_grid"

    """ (dict) Indexed by cell ID; yields a positive electrode id """
    CELL_TO_POS = "cell_id_to_pos_id"
    """ (dic) Indexed by cell ID; yields a negative electrode id """
    CELL_TO_NEG = "cell_id_to_neg_id"
    """ (dic) Indexed by cell ID; yields an electrolyte id """
    CELL_TO_LYTE = "cell_id_to_electrolyte_id"
    """ (dic) Indexed by cell ID; yields
        1 if the cell is latent, 0 if made of known pos, neg, electrolyte.
    """
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
    NAME_NEG = "neg_to_neg_name"
    NAME_LYTE = "electrolyte_to_electrolyte_name"
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

    class Pred:
        I_CC = "pred_cc_capacity"
        I_CV = "pred_cv_capacity"
