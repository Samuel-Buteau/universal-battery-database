import numpy
import tensorflow as tf
from django.core.management.base import BaseCommand

from neware_parser.models import *
from neware_parser.DegradationModel import DegradationModel
from neware_parser.plot import *

'''
Shortened Variable Names:
    vol -   voltage
    cap -   capacity
    dchg -  discharge
    neigh - neighbourhood
    der -   derivative
    pred -  predicted
    meas -  measured
    eval -  evaluation
    eq -    equillibrium
    res -   result
'''

#TODO(sam): For each barcode, needs a multigrid of (S, V, I, T) (current needs to be adjusted)
#TODO(sam): Each cycle must have an index mapping to the nearest reference cycle.
#TODO(sam): to evaluate a cycle, there must be the multigrid, the reference cycle scaled by cycle number, the cell features, pasted together and ran through a neural net.

NEIGH_INT_MIN_CYC_INDEX = 0
NEIGH_INT_MAX_CYC_INDEX = 1
NEIGH_INT_RATE_INDEX = 2
NEIGH_INT_BARCODE_INDEX = 3
NEIGH_INT_ABSOLUTE_INDEX = 4
NEIGH_INT_VALID_CYC_INDEX = 5
NEIGH_FLOAT_DELTA = 0
NEIGH_FLOAT_CHG_RATE = 1
NEIGH_FLOAT_DCHG_RATE = 2


#TODO(sam): these huge tensors would be much easier to understand with ragged tensors.
# right now, I am just flattening everything.

def numpy_acc(my_dict, my_key, my_dat):
    if my_key in my_dict.keys():
        my_dict[my_key] = numpy.concatenate(
            (my_dict[my_key], my_dat)
        )
    else:
        my_dict[my_key] = my_dat

    return my_dict


# ==== Begin: initial processing ===============================================

def initial_processing(my_data, barcodes, fit_args):
    """
    my_data has the following structure:
        my_data: a dictionary indexed by various data:
            - 'voltage_grid': 1D array of voltages
            - 'current_grid': 1D array of log currents
            - 'temperature_grid': 1D array of temperatures
            - 'sign_grid': 1D array of signs
            - 'all_data': a dictionary indexed by barcode. Each barcode yields:
                - 'all_reference_mats': structured array with dtype =
                    [
                        (
                            'cycle_number',
                            'f4'
                        ),
                        (
                            'count_matrix',
                            'f4',
                            (
                                len(sign_grid),
                                len(voltage_grid_degradation),
                                len(current_grid),
                                len(temperature_grid)
                            )
                        ),
                    ]

                - 'cyc_grp_dict': we know how this works.
                    basically groups of steps indexed by group averages of
                    (
                        end_current_prev,
                        constant_current,
                        end_current,
                        end_voltage_prev,
                        end_voltage,
                        sign
                    )

                    each group is a dictinary indexed by various quantities:
                        - 'main_data':  a numpy structured array with dtype:
                            [
                                ('cycle_number', 'f4'),
                                ('cc_voltage_vector', 'f4', len(voltage_grid)),
                                ('cc_capacity_vector', 'f4', len(voltage_grid)),
                                ('cc_mask_vector', 'f4', len(voltage_grid)),
                                ('cv_current_vector', 'f4', fit_args['current_max_n']),
                                ('cv_capacity_vector', 'f4', fit_args['current_max_n']),
                                ('cv_mask_vector', 'f4', fit_args['current_max_n']),
                                ('constant_current', 'f4'),
                                ('end_current_prev', 'f4'),
                                ('end_current', 'f4'),
                                ('end_voltage_prev', 'f4'),
                                ('end_voltage', 'f4'),
                                ('last_cc_voltage', 'f4'),
                                ('last_cc_capacity', 'f4'),
                                ('last_cv_capacity', 'f4'),
                                ('temperature', 'f4'),
                            ]

                        -     'avg_constant_current'
                        -     'avg_end_current_prev'
                        -     'avg_end_current'
                        -     'avg_end_voltage_prev'
                        -     'avg_end_voltage'
                        -     'avg_last_cc_voltage'





    """

    compiled_data = {}
    number_of_compiled_cycles = 0

    for barcode_count, barcode in enumerate(barcodes):




        max_cap = 0.
        cyc_grp_dict = my_data[barcode]
        # find largest cap measured for this cell (max over all cycle groups)
        for k in cyc_grp_dict.keys():
            max_cap = max(
                max_cap, max(abs(cyc_grp_dict[k]['main_data']['last_cc_capacity'])))




        for k_count, k in enumerate(cyc_grp_dict.keys()):

            # normalize capacity_vector with max_cap
            my_data[barcode][k]['main_data']['cc_capacity_vector'] = (
                1. / max_cap * cyc_grp_dict[k]['main_data']['cc_capacity_vector']
            )

            my_data[barcode][k]['main_data']['cv_capacity_vector'] = (
                1. / max_cap * cyc_grp_dict[k]['main_data']['cv_capacity_vector']
            )

            my_data[barcode][k]['main_data']['cv_current_vector'] = (
                1. / max_cap * cyc_grp_dict[k]['main_data']['cv_current_vector']
            )

            my_data[barcode][k]['main_data']['last_cc_capacity'] = (
                1. / max_cap * cyc_grp_dict[k]['main_data']['last_cc_capacity']
            )

            my_data[barcode][k]['main_data']['last_cv_capacity'] = (
                1. / max_cap * cyc_grp_dict[k]['main_data']['last_cv_capacity']
            )

            my_data[barcode][k]['main_data']['constant_current'] = (
                1. / max_cap * cyc_grp_dict[k]['main_data']['constant_current']
            )
            my_data[barcode][k]['main_data']['end_current_prev'] = (
                1. / max_cap * cyc_grp_dict[k]['main_data']['end_current_prev']
            )

            my_data[barcode][k]['avg_constant_current'] = (
                1. / max_cap * cyc_grp_dict[k]['avg_constant_current']
            )

            my_data[barcode][k]['avg_end_current'] = (
                1. / max_cap * cyc_grp_dict[k]['avg_end_current']
            )

            my_data[barcode][k]['avg_end_current_prev'] = (
                1. / max_cap * cyc_grp_dict[k]['avg_end_current_prev']
            )


            # range of cycles which exist for this cycle group
            min_cyc = min(cyc_grp_dict[k]['main_data']['cycle_number'])
            max_cyc = max(cyc_grp_dict[k]['main_data']['cycle_number'])

            '''
            - now create neighborhoods, which contains the cycles,
              grouped by proximity
            - want to sample neighborhoods equally
            - neighborhoods have a central cycle and a delta on each side
            - to a first approximation, we want a delta_cyc = 300, but we have
              to vary this near the beginning of data and near the end.
            '''

            # gives an absolute scale
            total_delta = max_cyc - min_cyc

            # the baseline, at least 200, but up to total_delta/5
            delta_cyc = max(200, int(float(total_delta) / 5.))

            # the centers of neighborhoods we will try to create
            all_neigh_center_cycles = list(filter(
                lambda x: x > min_cyc - 100,
                range(20, int(max_cyc + 50), 40))
            )


            # check all tentative neighborhood centers and
            # commit the ones that contain good data to the dataset
            neighborhood_data = []

            valid_cycles = 0
            for cyc in all_neigh_center_cycles:
                # max_cyc and min_cyc are the limits of existing cycles.

                # at least 200, but can extend up to the limit
                # starting from the current neighborhood center
                delta_up = max(max_cyc - cyc, 200)

                # same thing going down
                delta_down = max(cyc - min_cyc, 200)

                # the max symetric interval that fits into the
                # [cyc - delta_down, cyc + delta_up] interval is
                # [cyc - delta_actual, cyc + delta_actual]
                delta_actual = min(delta_up, delta_down)

                # choose the largest interval that fits both
                # [cyc - delta_actual, cyc + delta_actual] and
                # [cyc - delta_cyc, cyc + delta_cyc]
                combined_delta = min(delta_actual, delta_cyc)

                below_cyc = cyc - combined_delta
                above_cyc = cyc + combined_delta

                # numpy array of True and False; same length as cyc_grp_dict[k]
                # False when cycle_number falls outside out of
                # [below_cyc, above_cyc] interval
                mask = numpy.logical_and(
                    below_cyc <= cyc_grp_dict[k]['main_data']['cycle_number'],
                    cyc_grp_dict[k]['main_data']['cycle_number'] <= above_cyc
                )

                # the indecies for the cyc_grp_dict[k] array which correspond
                # to a True mask
                all_valid_indecies = numpy.arange(len(mask))[mask]

                # if there are less than 2 valid cycles, skip that neighborhood
                if len(all_valid_indecies) < 2:
                    continue

                '''
                at this point, we know that this neighborhood
                will be added to the dataset.
                '''

                min_cyc_index = all_valid_indecies[0]
                max_cyc_index = all_valid_indecies[-1]



                valid_cycles += 1

                '''
                this commits the neighborhood to the dataset

                - record the info about the center of the neighborhood
                  (cycle number, voltage, rate of charge, rate of discharge)
                - record the relative index (within the cycle group)
                  of the min cycle, max cycle
                - record a voltage index, and a cycle group index,
                  and a cell index
                - record the absolute index into the table of cycles
                  (len(cycles_full)).
                - keep a slot empty for later
                
 
                '''
                #TODO(sam): here, figure out where the reference cycle is, and note the index

                neighborhood_data.append(
                    [
                        min_cyc_index,
                        max_cyc_index,
                        k_count,
                        barcode_count,
                        number_of_compiled_cycles,
                        0 # a weight based on prevalance. Set later
                    ]
                )

            if valid_cycles != 0:
                neighborhood_data = numpy.array(
                    neighborhood_data, dtype = numpy.int32
                )

                # the empty slot becomes the count of added neighborhoods, which
                # are used to counterbalance the bias toward longer cycle life
                neighborhood_data[:, NEIGH_INT_VALID_CYC_INDEX] = valid_cycles

                numpy_acc(compiled_data, 'neighborhood_data', neighborhood_data)


            number_of_compiled_cycles += len(cyc_grp_dict[k]['main_data']['cycle_number'])

            numpy_acc(compiled_data, 'cycle', cyc_grp_dict[k]['main_data']['cycle_number'])
            numpy_acc(compiled_data, 'cc_voltage_vector', cyc_grp_dict[k]['main_data']['cc_voltage_vector'])
            numpy_acc(compiled_data, 'cc_capacity_vector', cyc_grp_dict[k]['main_data']['cc_capacity_vector'])
            numpy_acc(compiled_data, 'cc_mask_vector', cyc_grp_dict[k]['main_data']['cc_mask_vector'])
            numpy_acc(compiled_data, 'cv_current_vector', cyc_grp_dict[k]['main_data']['cv_current_vector'])
            numpy_acc(compiled_data, 'cv_capacity_vector', cyc_grp_dict[k]['main_data']['cv_capacity_vector'])
            numpy_acc(compiled_data, 'cv_mask_vector', cyc_grp_dict[k]['main_data']['cv_mask_vector'])
            numpy_acc(compiled_data, 'constant_current', cyc_grp_dict[k]['main_data']['constant_current'])
            numpy_acc(compiled_data, 'end_current_prev', cyc_grp_dict[k]['main_data']['end_current_prev'])
            numpy_acc(compiled_data, 'end_voltage_prev', cyc_grp_dict[k]['main_data']['end_voltage_prev'])
            numpy_acc(compiled_data, 'end_voltage', cyc_grp_dict[k]['main_data']['end_voltage'])



    neighborhood_data = tf.constant(compiled_data['neighborhood_data'])

    # cycles go from 0 to 6000, but nn prefers normally distributed variables
    # so cycle numbers is normalized with mean and variance
    cycle_tensor = tf.constant(compiled_data['cycle'])
    cycle_m, cycle_v = tf.nn.moments(cycle_tensor, axes = [0])
    cycle_m = cycle_m.numpy()
    cycle_v = cycle_v.numpy()
    cycle_tensor = (cycle_tensor - cycle_m) / tf.sqrt(cycle_v)


    cc_voltage_tensor = tf.constant(compiled_data['cc_voltage_vector'])
    cc_capacity_tensor = tf.constant(compiled_data['cc_capacity_vector'])
    cc_mask_tensor = tf.constant(compiled_data['cc_mask_vector'])
    cv_capacity_tensor = tf.constant(compiled_data['cv_capacity_vector'])
    cv_current_tensor = tf.constant(compiled_data['cv_current_vector'])
    cv_mask_tensor = tf.constant(compiled_data['cv_mask_vector'])

    constant_current_tensor = tf.constant(compiled_data['constant_current'])
    end_current_prev_tensor = tf.constant(compiled_data['end_current_prev'])
    end_voltage_prev_tensor = tf.constant(compiled_data['end_voltage_prev'])
    end_voltage_tensor = tf.constant(compiled_data['end_voltage'])






    batch_size = fit_args['batch_size']
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        train_ds_ = tf.data.Dataset.from_tensor_slices(
            neighborhood_data
        ).repeat(2).shuffle(100000).batch(batch_size)

        train_ds = mirrored_strategy.experimental_distribute_dataset(
            train_ds_)

        degradation_model = DegradationModel(
            num_keys = len(barcodes),
            width = fit_args['width'],
            depth = fit_args['depth']
        )

        optimizer = tf.keras.optimizers.Adam()

    return {
        "mirrored_strategy":       mirrored_strategy,
        "degradation_model":       degradation_model,

        "cycle_tensor":            cycle_tensor,
        "constant_current_tensor": constant_current_tensor,
        "end_current_prev_tensor": end_current_prev_tensor,
        "end_voltage_prev_tensor": end_voltage_prev_tensor,
        "end_voltage_tensor":      end_voltage_tensor,

        "train_ds":                train_ds,
        "cycle_m":                 cycle_m,
        "cycle_v":                 cycle_v,

        'cc_voltage_tensor':       cc_voltage_tensor,
        'cc_capacity_tensor':      cc_capacity_tensor,
        'cc_mask_tensor':          cc_mask_tensor,
        'cv_capacity_tensor':      cv_capacity_tensor,
        'cv_current_tensor':       cv_current_tensor,
        'cv_mask_tensor':          cv_mask_tensor,

        "optimizer":               optimizer,
        "all_data":                my_data
    }


# === End: initial processing ==================================================

# === Begin: train =============================================================

def train_and_evaluate(init_returns, barcodes, fit_args):
    mirrored_strategy = init_returns["mirrored_strategy"]

    EPOCHS = 100000
    count = 0

    template = 'Epoch {}, Count {}'

    with mirrored_strategy.scope():
        for epoch in range(EPOCHS):
            for neighborhood in init_returns["train_ds"]:
                count += 1

                train_step_params = {
                    "neighborhood":               neighborhood,

                    "cycle_tensor":           init_returns["cycle_tensor"],
                    "constant_current_tensor": init_returns[
                                                   "constant_current_tensor"],
                    "end_current_prev_tensor": init_returns[
                                                   "end_current_prev_tensor"],
                    "end_voltage_prev_tensor": init_returns[
                                                   "end_voltage_prev_tensor"],
                    "end_voltage_tensor":      init_returns[
                                                   "end_voltage_tensor"],

                    "degradation_model":       init_returns[
                                                   "degradation_model"],
                    "optimizer":               init_returns["optimizer"],

                    'cc_voltage_tensor':       init_returns[
                                                   "cc_voltage_tensor"],
                    'cc_capacity_tensor':      init_returns[
                                                   "cc_capacity_tensor"],
                    'cc_mask_tensor':          init_returns["cc_mask_tensor"],
                    'cv_capacity_tensor':      init_returns[
                                                   "cv_capacity_tensor"],
                    'cv_current_tensor':       init_returns[
                                                   "cv_current_tensor"],
                    'cv_mask_tensor':          init_returns["cv_mask_tensor"],

                }

                dist_train_step(mirrored_strategy, train_step_params, fit_args)

                if count != 0:
                    if (count % fit_args['print_loss_every']) == 0:
                        print(template.format(epoch + 1, count, ))

                    plot_params = {
                        "barcodes": barcodes,
                        "count":    count,
                        "fit_args": fit_args,
                    }

                    if (count % fit_args['visualize_fit_every']) == 0:
                        plot_capacity(plot_params, init_returns)
                        plot_vq(plot_params, init_returns)

                if count == fit_args['stop_count']:
                    return


# === End: train ===============================================================

# === Begin: train step ========================================================

def train_step(params, fit_args):

    neighborhood = params["neighborhood"]

    cycle_tensor = params["cycle_tensor"]
    constant_current_tensor = params["constant_current_tensor"]
    end_current_prev_tensor = params["end_current_prev_tensor"]
    end_voltage_prev_tensor = params["end_voltage_prev_tensor"]
    end_voltage_tensor = params["end_voltage_tensor"]

    degradation_model = params["degradation_model"]
    optimizer = params["optimizer"]

    cc_voltage_tensor = params["cc_voltage_tensor"]
    cc_capacity_tensor = params["cc_capacity_tensor"]
    cc_mask_tensor = params["cc_mask_tensor"]
    cv_capacity_tensor = params["cv_capacity_tensor"]
    cv_current_tensor = params["cv_current_tensor"]
    cv_mask_tensor = params["cv_mask_tensor"]

    # need to split the range
    batch_size2 = neighborhood.shape[0]

    '''
    if you have the minimum cycle and maximum cycle for a neighborhood,
    you can sample cycle from this neighborhood by sampling real numbers
    x from [0,1] and computing min_cyc*(1.-x) + max_cyc*x,
    but here this computation is done in index space,
    then cycle numbers and vq curves are gathered
    '''

    cycle_indecies_lerp = tf.random.uniform(
        [batch_size2], minval = 0., maxval = 1., dtype = tf.float32)
    cycle_indecies = tf.cast(
        (1. - cycle_indecies_lerp) * tf.cast(
            neighborhood[:, NEIGH_INT_MIN_CYC_INDEX]
            + neighborhood[:, NEIGH_INT_ABSOLUTE_INDEX],
            tf.float32
        ) + (cycle_indecies_lerp) * tf.cast(
            neighborhood[:, NEIGH_INT_MAX_CYC_INDEX]
            + neighborhood[:, NEIGH_INT_ABSOLUTE_INDEX],
            tf.float32
        ),
        tf.int32
    )

    cycle = tf.gather(
        cycle_tensor,
        indices = cycle_indecies, axis = 0
    )
    constant_current = tf.gather(
        constant_current_tensor,
        indices = cycle_indecies, axis = 0
    )
    end_current_prev = tf.gather(
        end_current_prev_tensor,
        indices = cycle_indecies, axis = 0
    )
    end_voltage_prev = tf.gather(
        end_voltage_prev_tensor,
        indices = cycle_indecies, axis = 0
    )

    end_voltage = tf.gather(
        end_voltage_tensor,
        indices = cycle_indecies, axis = 0
    )

    cc_capacity = tf.gather(cc_capacity_tensor, indices = cycle_indecies)
    cc_voltage = tf.gather(cc_voltage_tensor, indices = cycle_indecies)
    cc_mask = tf.gather(cc_mask_tensor, indices = cycle_indecies)
    cc_mask_2 = tf.tile(
        tf.reshape(
            1. / (tf.cast(neighborhood[:, NEIGH_INT_VALID_CYC_INDEX], tf.float32)),
            [batch_size2, 1]
        ),
        [1, cc_voltage.shape[1]]
    )

    cv_capacity = tf.gather(cv_capacity_tensor, indices = cycle_indecies)
    cv_current = tf.gather(cv_current_tensor, indices = cycle_indecies)
    cv_mask = tf.gather(cv_mask_tensor, indices = cycle_indecies)
    cv_mask_2 = tf.tile(
        tf.reshape(
            1. / (tf.cast(neighborhood[:, NEIGH_INT_VALID_CYC_INDEX], tf.float32)),
            [batch_size2, 1]
        ),
        [1, cv_current.shape[1]]
    )

    cell_indecies = neighborhood[:, NEIGH_INT_BARCODE_INDEX]

    with tf.GradientTape() as tape:
        train_results = degradation_model(
            (
                tf.expand_dims(cycle, axis = 1),
                tf.expand_dims(constant_current, axis = 1),
                tf.expand_dims(end_current_prev, axis = 1),
                tf.expand_dims(end_voltage_prev, axis = 1),
                tf.expand_dims(end_voltage, axis = 1),
                cell_indecies,
                cc_voltage,
                cv_current,
            ),
            training = True
        )

        pred_cc_capacity = train_results["pred_cc_capacity"]
        pred_cv_capacity = train_results["pred_cv_capacity"]

        cc_capacity_loss = (
            tf.reduce_mean(
                cc_mask_2 * cc_mask * tf.square(cc_capacity - pred_cc_capacity))
            / (1e-10 + tf.reduce_mean(cc_mask_2 * cc_mask))
        )
        cv_capacity_loss = (
            tf.reduce_mean(
                cv_mask_2 * cv_mask * tf.square(cv_capacity - pred_cv_capacity))
            / (1e-10 + tf.reduce_mean(cv_mask_2 * cv_mask))
        )

        loss = (
            cc_capacity_loss + 0.05 * cv_capacity_loss
            + train_results["soc_loss"]
            + train_results["theo_cap_loss"]
            + train_results["r_loss"]
            + train_results["shift_loss"]
            + fit_args['kl_coeff'] * train_results["kl_loss"]
        )

    gradients = tape.gradient(loss, degradation_model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, degradation_model.trainable_variables)
    )


# === End : train step =========================================================

@tf.function
def dist_train_step(mirrored_strategy, train_step_params, fit_args):
    mirrored_strategy.experimental_run_v2(
        train_step, args = (train_step_params, fit_args)
    )


def ml_smoothing(fit_args):
    if not os.path.exists(fit_args['path_to_plots']):
        os.mkdir(fit_args['path_to_plots'])

    dataset_path = os.path.join(
        fit_args['path_to_dataset'],
        'dataset_ver_{}.file'.format(fit_args['dataset_version'])
    )

    if not os.path.exists(dataset_path):
        print("Path \"" + dataset_path + "\" does not exist.")
        return

    with open(dataset_path, 'rb') as f:
        my_data = pickle.load(f)

    barcodes = list(my_data.keys())

    if len(fit_args['wanted_barcodes']) != 0:
        barcodes = list(
            set(barcodes).intersection(set(fit_args['wanted_barcodes'])))

    if len(barcodes) == 0:
        return

    train_and_evaluate(
        initial_processing(my_data, barcodes, fit_args), barcodes, fit_args)


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path_to_dataset', required = True)
        parser.add_argument('--dataset_version', required = True)
        parser.add_argument('--path_to_plots', required = True)
        parser.add_argument('--kl_coeff', type = float, default = 0.00001)
        parser.add_argument('--mono_coeff', type = float, default = .005)
        parser.add_argument('--smooth_coeff', type = float, default = .05)
        parser.add_argument('--const_f_coeff', type = float, default = .0)
        parser.add_argument('--smooth_f_coeff', type = float, default = .01)
        parser.add_argument('--depth', type = int, default = 3)
        parser.add_argument('--width', type = int, default = 32)
        parser.add_argument('--batch_size', type = int, default = 2 * 16)

        vis = 10000
        parser.add_argument('--print_loss_every', type = int, default = vis)
        parser.add_argument('--visualize_fit_every', type = int, default = vis)
        parser.add_argument('--visualize_vq_every', type = int, default = vis)

        parser.add_argument('--stop_count', type = int, default = 100000)
        parser.add_argument(
            '--wanted_barcodes', type = int, nargs = '+',
            default = [83220, 83083]
        )

    def handle(self, *args, **options):
        ml_smoothing(options)
