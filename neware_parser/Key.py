class Key:
    """ Define dictionary keys """

    N = "cycle_number"
    N_SAMPLE = "n_sample"

    # TODO: Why are there keys with the same values except "_vector"?
    #       Are they the same values just different types?
    #           If so, maybe create a new class to store both values
    #           (don't be obsessed with primitive values)
    #       Are they differnt values?
    #           If so, give them different and more descriptive names

    V_CC = "cc_voltages"
    V_CC_VEC = "cc_voltage_vector"
    V_CC_LAST = "last_cc_voltage"
    V_CC_LAST_AVG = "avg_last_cc_voltage"
    V_END = "end_voltage"
    V_PREV_END = "end_voltage_prev"
    V_PREV_END_AVG = "avg_end_voltage_prev"
    V_END_AVG = "avg_end_voltage"
    V_GRID = "voltage_grid"

    Q_CC = "cc_capacities"
    Q_CC_VEC = "cc_capacity_vector"
    Q_CV = "cv_capacities"
    Q_CV_VEC = "cv_capacity_vector"
    Q_END_AVG = "avg_end_current"
    Q_CC_LAST = "last_cc_capacity"
    Q_CV_LAST = "last_cv_capacity"
    Q_MAX = "max_cap"

    I_CC = "constant_current"
    I_CC_AVG = "avg_constant_current"
    I_CV = "cv_currents"
    I_CV_VEC = "cv_current_vector"
    I_END = "end_current"
    I_PREV = "end_current_prev"
    I_PREV_END_AVG = "avg_end_current_prev"
    I_GRID = "current_grid"
    I_MAX = "current_max_n"

    TEMP = "temperature"

    MASK_CC = "cc_masks"
    MASK_CC_VEC = "cc_mask_vector"
    MASK_CV = "cv_masks"
    MASK_CV_VEC = "cv_mask_vector"
    COUNT_MATRIX = "count_matrix"
    SIGN_GRID = "sign_grid"
    TEMP_GRID = "temperature_grid"

    WIDTH = "width"
    DEPTH = "depth"
    LRN_RATE = "learning_rate"
    BATCH = "batch_size"
    PATH_DATASET = "path_to_dataset"
    PATH_PLOTS = "path_to_plots"
    DATA_VERSION = "dataset_version"
    BARCODES = "wanted_barcodes"
    STOP = "stop_count"
    VIS = "visualize_fit_every"
    PRINT_LOSS = "print_loss_every"
    MIN_LAT = "min_latent"
    GLB_NORM_CLIP = "global_norm_clip"

    MAIN = "main_data"
    MY_DATA = "my_data"
    ALL_DATA = "all_data"
    CYC_GRP_DICT = "cyc_grp_dict"

    REF_ALL_MATS = "all_reference_mats"
    REF_CYC = "reference_cycles_n"

    CELL_TO_POS = "cell_id_to_pos_id"
    CELL_TO_NEG = "cell_id_to_neg_id"
    CELL_TO_ELE = "cell_id_to_electrolyte_id"
    CELL_TO_LAT = "cell_id_to_latent"
    ELE_TO_SOL = "electrolyte_id_to_solvent_id_weight"
    ELE_TO_SALT = "electrolyte_id_to_salt_id_weight"
    ELE_TO_ADD = "electrolyte_id_to_additive_id_weight"
    ELE_TO_LAT = "electrolyte_id_to_latent"

    # TODO (harvey): replace the following block of keys
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
