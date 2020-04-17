# TODO (harvey): This is probably not the most elegant solution,
#                but at least things can be grouped logically now.
#                Could use another refactor.

class Key:
    class Main:
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

    class Fit:
        # fig args keys
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


# params keys
TENSORS = "compiled_tensors"
MODEL = "degradation_model"

Q_END_AVG = "avg_end_current"
Q_GRID = "current_grid"

I_CC_AVG = "avg_constant_current"
I_PREV_END_AVG = "avg_end_current_prev"

V_END_AVG = "avg_end_voltage"
V_PREV_END_AVG = "avg_end_voltage_prev"
V_GRID = "voltage_grid"

COUNT_MATRIX = "count_matrix"

SIGN_GRID = "sign_grid"
TEMP_GRID = "temperature_grid"

# data keys
ALL_DATA = "all_data"

# my_data keys
CELL_TO_POS = "cell_id_to_pos_id"
CELL_TO_NEG = "cell_id_to_neg_id"
CELL_TO_ELE = "cell_id_to_electrolyte_id"
CELL_TO_LAT = "cell_id_to_latent"

ELE_TO_SOL = "electrolyte_id_to_solvent_id_weight"
ELE_TO_SALT = "electrolyte_id_to_salt_id_weight"
ELE_TO_ADD = "electrolyte_id_to_additive_id_weight"
ELE_TO_LAT = "electrolyte_id_to_latent"

# loss keys
Q_LOSS = "q_loss"
Q_CC_LOSS = "cc_capacity_loss"
Q_CV_LOSS = "cv_capacity_loss"

V_CC_LOSS = "cc_voltage_loss"
V_CV_LOSS = "cv_voltage_loss"

SCALE_LOSS = "scale_loss"
R_LOSS = "r_loss"
SHIFT_LOSS = "shift_loss"
CELL_LOSS = "cell_loss"
RECIP_LOSS = "reciprocal_loss"

PROJ_LOSS = "projection_loss"
OOB_LOSS = "out_of_bounds_loss"

COEFF_Q = "coeff_q"
COEFF_Q_CV = "coeff_cv_capacity"
COEFF_Q_CC = "coeff_cc_capacity"
COEFF_V_CC = "coeff_cc_voltage"
COEFF_SCALE = "coeff_scale"
COEFF_R = "coeff_r"
COEFF_SHIFT = "coeff_shift"
COEFF_CELL = "coeff_cell"
COEFF_RECIP = "coeff_reciprocal"
COEFF_PROJ = "coeff_projection"
COEFF_OOB = "coeff_out_of_bounds"
