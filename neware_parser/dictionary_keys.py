# TODO (harvey): This is probably not the most elegant solution,
#                but at least things can be grouped logically now.
#                Could use another refactor.


class Key:
    """ Provide dictionary keys """

    N = "cycle_number"
    V_CC = "cc_voltage_vector"
    V_END = "end_voltage"
    V_PREV_END = "end_voltage_prev"
    Q_CC = "cc_capacity_vector"
    Q_CV = "cv_capacity_vector"
    Q_CC_LAST = "last_cc_capacity"
    Q_CV_LAST = "last_cv_capacity"
    I_CC = "constant_current"
    I_CV = "cv_current_vector"
    I_PREV = "end_current_prev"
    MASK_CC = "cc_mask_vector"
    MASK_CV = "cv_mask_vector"

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
    N_SAMPLE = "n_sample"
    MIN_LAT = "min_latent"
    GLB_NORM_CLIP = "global_norm_clip"

    MAIN = "main_data"
    Q_END_AVG = "avg_end_current"
    V_PREV_END_AVG = "avg_end_voltage_prev"
    V_END_AVG = "avg_end_voltage"
    I_CC_AVG = "avg_constant_current"
    I_PREV_END_AVG = "avg_end_current_prev"

    ALL_DATA = "all_data"
    MAX_CAP = "max_cap"
    Q_GRID = "current_grid"
    CELL_TO_POS = "cell_id_to_pos_id"
    CELL_TO_NEG = "cell_id_to_neg_id"
    CELL_TO_ELE = "cell_id_to_electrolyte_id"
    CELL_TO_LAT = "cell_id_to_latent"
    ELE_TO_SOL = "electrolyte_id_to_solvent_id_weight"
    ELE_TO_SALT = "electrolyte_id_to_salt_id_weight"
    ELE_TO_ADD = "electrolyte_id_to_additive_id_weight"
    ELE_TO_LAT = "electrolyte_id_to_latent"

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

    class Init:
        STRAT = "strategy"
        TENSORS = "compiled_tensors"
        MODEL = "degradation_model"
        TRAIN_DS = "train_ds"
        CYC_M = "cycle_m"
        CYC_V = "cycle_v"
        OPT = "optimizer"
        MY_DATA = "my_data"


# params keys

V_GRID = "voltage_grid"

COUNT_MATRIX = "count_matrix"

SIGN_GRID = "sign_grid"
TEMP_GRID = "temperature_grid"
