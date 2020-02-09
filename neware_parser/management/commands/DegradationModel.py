import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer

from .colour_print import Print

def feedforward_nn_parameters(depth, width):
    initial = Dense(
        width,
        activation = 'elu',
        use_bias = True,
        bias_initializer = 'zeros'
    )

    bulk = [
        Dense(
            width,
            activation = 'elu',
            use_bias = True,
            bias_initializer = 'zeros'
        ) for _ in range(depth)
    ]

    final = Dense(
        1,
        activation = None,
        use_bias = True,
        bias_initializer = 'zeros',
        kernel_initializer = 'zeros'
    )
    return {'initial': initial, 'bulk': bulk, 'final': final}

class DegradationModel(Model):

    def __init__(self, num_keys, depth, width):
        super(DegradationModel, self).__init__()

        self.nn_cap = feedforward_nn_parameters(depth, width)
        self.nn_eq_vol = feedforward_nn_parameters(depth, width)
        self.nn_r = feedforward_nn_parameters(depth, width)

        self.nn_theoretical_cap = feedforward_nn_parameters(depth, width)
        self.nn_soc = feedforward_nn_parameters(depth, width)
        self.nn_soc_0 = feedforward_nn_parameters(depth, width)
        self.nn_init_soc = feedforward_nn_parameters(depth, width)

        self.nn_soc_part2 = feedforward_nn_parameters(depth, width)
        self.nn_eq_voltage_0 = feedforward_nn_parameters(depth, width)

        self.nn_shift = feedforward_nn_parameters(depth, width)
        self.nn_soc_0_part3 = feedforward_nn_parameters(depth, width)
        self.nn_soc_1_part3 = feedforward_nn_parameters(depth, width)

        self.dictionary = DictionaryLayer(num_features=width, num_keys=num_keys)

        self.width = width
        self.num_keys = num_keys

    # Begin: nn application functions ==========================================

    def norm_cycle(self, params, scalar = True):
        cycles = "cycles"
        norm_constant = "norm_constant"
        if not scalar:
            norm_constant += "_flat"
            cycles += "_flat"
        return params[cycles] * (1e-10 + tf.exp(-params[norm_constant]))

    ''' Part 4 ------------------------------------------------------------- '''

    ''' Part 3 ------------------------------------------------------------- '''

    def dchg_cap_part3(self, params):
        theoretical_cap = self.add_volt_dep(
            self.theoretical_cap(params),
            params
        )

        shift = self.shift(params)

        soc_0 = self.add_volt_dep(
            self.soc_part3(params, shift, self.eq_voltage_0(params)),
            params
        )
        # "whereever you are in the voltage curve"
        soc_1 = self.soc_part3(
            params,
            self.add_volt_dep(shift, params),
            self.eq_voltage_1(params),
            scalar = False
        )

        return -theoretical_cap * (soc_1 - soc_0)

    def shift(self, params):
        dependencies = (
            self.norm_cycle(params),
            params["chg_rate"],
            params["dchg_rate"],
            params["cell_feat"]
        )
        return self.nn_call(self.nn_shift, dependencies)

    def soc_part3(self, params, shift, voltage, scalar = True):
        cell_feat = "cell_feat"
        if not scalar:
            cell_feat = "cell_feat_flat"

        dependencies = (
            shift,
            voltage,
            params[cell_feat]
        )
        return tf.exp(self.nn_call(self.nn_soc_0_part3, dependencies))

    ''' Part 2 ------------------------------------------------------------- '''

    def dchg_cap_part2(self, params):
        theoretical_cap = self.add_volt_dep(
            self.theoretical_cap(params),
            params
        )
        soc_0 = self.add_volt_dep(
            self.soc(params, self.eq_voltage_0(params), scalar = True),
            params
        )
        soc_1 = self.soc(params, self.eq_voltage_1(params), scalar = False)

        return -theoretical_cap * (soc_1 - soc_0)

    # eq_voltage_0(cycle, cell_feat)
    def eq_voltage_0(self, params):
        dependencies = (
            self.norm_cycle(params),
            params["chg_rate"],
            params["cell_feat"]
        )
        return tf.exp(self.nn_call(self.nn_eq_voltage_0, dependencies))

    ''' Part 1 ------------------------------------------------------------- '''

    # dchg_cap_part1 = -theoretical_cap * (soc_1 - soc_0)
    def dchg_cap_part1(self, params):
        theoretical_cap = self.add_volt_dep(
            self.theoretical_cap(params), # scalar
            params
        )
        soc_0 = self.add_volt_dep(self.soc_0_part1(params), params)
        soc_1 = self.soc(params, self.eq_voltage_1(params), scalar = False)

        return -theoretical_cap * (soc_1 - soc_0)

    # eq_voltage_1 = voltage + dchg_rate * R
    def eq_voltage_1(self, params):
        r_flat = self.add_volt_dep(self.r(params), params)

        return params["voltage_flat"] + params["dchg_rate_flat"] * r_flat

    # theoretical_cap(cycles, dchg_rate, cell_feat)
    def theoretical_cap(self, params):
        dependencies = (
            self.norm_cycle(params),
            params["dchg_rate"],
            params["cell_feat"]
        )
        return tf.exp(self.nn_call(self.nn_theoretical_cap, dependencies))

    # sco_1 = soc(voltage = eq_voltage_1, cell_feat)
    def soc(self, params, voltage, scalar = True):
        cell_feat = "cell_feat"
        if not scalar:
            cell_feat = "cell_feat_flat"

        dependencies = (
            voltage,
            params[cell_feat]
        )
        return self.nn_call(self.nn_soc, dependencies)

    # soc_0(cycles, dchg_rate, cell_feat)
    def soc_0_part1(self, params):
        dependencies = (
            self.norm_cycle(params),
            params["dchg_rate"],
            params["cell_feat"]
        )
        return tf.exp(self.nn_call(self.nn_soc_0, dependencies))

    ''' End ---------------------------------------------------------------- '''

    def max_dchg_vol(self, params):
        eq_vol = self.eq_vol(params)
        r = self.r(params)

        return eq_vol - (params["dchg_rate"] * r)

    # Unstructured variables ---------------------------------------------------

    def nn_call(self, nn_func, dependencies):
        centers = nn_func['initial'](
            tf.concat(dependencies, axis=1)
        )
        for d in nn_func['bulk']:
            centers = d(centers)
        return nn_func['final'](centers)

    def cap(self, params):
        dependencies = (
            self.norm_cycle(params, scalar = False),
            params["chg_rate_flat"],
            params["dchg_rate_flat"],
            params["voltage_flat"],
            params["cell_feat_flat"]
        )
        return self.nn_call(self.nn_cap, dependencies)

    def eq_vol(self, params):
        dependencies = (
            self.norm_cycle(params),
            params["chg_rate"],
            params["cell_feat"]
        )
        return self.nn_call(self.nn_eq_vol, dependencies)

    def r(self, params):
        dependencies = (
            self.norm_cycle(params),
            params["cell_feat"]
        )
        return tf.exp(self.nn_call(self.nn_r, dependencies))

    # End: nn application functions ============================================

    def create_derivatives(self, params, nn, scalar = True):
        derivatives = {}

        cycles = "cycles"
        chg_rate = "chg_rate"
        dchg_rate = "dchg_rate"
        cell_feat = "cell_feat"
        if not scalar:
            cycles = "cycles_flat"
            chg_rate = "chg_rate_flat"
            dchg_rate = "dchg_rate_flat"
            cell_feat = "cell_feat_flat"


        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch(params[cycles])
            tape3.watch(params[chg_rate])
            tape3.watch(params[dchg_rate])
            tape3.watch(params[cell_feat])
            if not scalar:
                tape3.watch(params["voltage_flat"])

            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(params[cycles])
                tape2.watch(params[chg_rate])
                tape2.watch(params[dchg_rate])
                tape2.watch(params[cell_feat])
                if not scalar:
                    tape2.watch(params["voltage_flat"])

                res = tf.reshape(nn(params), [-1, 1])

            derivatives['dCyc'] = tape2.batch_jacobian(
                source=params[cycles],
                target=res
            )[:, 0, :]
            derivatives['d_chg_rate'] = tape2.batch_jacobian(
                source=params[chg_rate],
                target=res
            )[:, 0, :]
            derivatives['d_dchg_rate'] = tape2.batch_jacobian(
                source=params[dchg_rate],
                target=res
            )[:, 0, :]
            derivatives['dFeatures'] = tape2.batch_jacobian(
                source=params[cell_feat],
                target=res
            )[:, 0, :]
            if not scalar:
                derivatives['dVol'] = tape2.batch_jacobian(
                    source=params["voltage_flat"],
                    target=res
                )[:, 0, :]

            del tape2

        derivatives['d2Cyc'] = tape3.batch_jacobian(
            source=params[cycles],
            target=derivatives['dCyc']
        )[:, 0, :]
        derivatives['d2_chg_rate'] = tape3.batch_jacobian(
            source=params[chg_rate],
            target=derivatives['d_chg_rate']
        )[:, 0, :]
        derivatives['d2_dchg_rate'] = tape3.batch_jacobian(
            source=params[dchg_rate],
            target=derivatives['d_dchg_rate']
        )[:, 0, :]
        derivatives['d2Features'] = tape3.batch_jacobian(
            source=params[cell_feat],
            target=derivatives['dFeatures']
        )
        if not scalar:
            derivatives['d2Vol'] = tape3.batch_jacobian(
                source=params["voltage_flat"],
                target=derivatives['dVol']
            )[:, 0, :]

        del tape3
        return res, derivatives

    # add voltage dependence ([cyc] -> [cyc, vol])
    def add_volt_dep(self, thing, params, dim = 1):
        return tf.reshape(
            tf.tile(
                tf.expand_dims(
                    thing,
                    axis=1
                ),
                [1, params["voltage_count"],1]
            ),
            [params["batch_count"] * params["voltage_count"], dim]
        )

    # add cycle dependence ([vol] -> [cyc, vol])
    def add_cyc_dep(self, thing, params, dim = 1):
        return tf.reshape(
            tf.tile(
                tf.expand_dims(
                    thing,
                    axis=0
                ),
                [params["batch_count"], 1,1]
            ),
            [params["batch_count"] * params["voltage_count"], dim]
        )

    def call(self, x, training=False):

        centers = x[0]  # batch of [cyc, k[0], k[1]]; dim: [batch, 3]
        indecies = x[1]  # batch of index; dim: [batch]
        meas_cycles = x[2]  # batch of cycles; dim: [batch]
        voltage_vector = x[3] # dim: [voltages]
        #Print.colour(Print.RED, voltage_vector)
        # TODO get another input `shift_vector`
        #tf.constant([32_things_in_here?])

        features, mean, log_sig = self.dictionary(indecies, training=training)
        cycles = centers[:, 0:1] # matrix; dim: [batch, 1]
        rates = centers[:, 1:] # dim: [batch, 2]

        # duplicate cycles and others for all the voltages
        # dimensions are now [batch, voltages, features]
        batch_count = cycles.shape[0]
        voltage_count = voltage_vector.shape[0]
        count_dict = {
            "batch_count": batch_count,
            "voltage_count": voltage_count
        }

        params = {
            "batch_count": batch_count,
            "voltage_count": voltage_count,
            "cycles_flat": self.add_volt_dep(
                cycles,
                count_dict
            ),
            "chg_rate_flat": self.add_volt_dep(
                rates[:, 0:1],
                count_dict
            ),
            "dchg_rate_flat": self.add_volt_dep(
                rates[:, 1:2],
                count_dict
            ),
            "cell_feat_flat": self.add_volt_dep(
                features[:, 1:],
                count_dict,
                dim = self.width - 1
            ),
            "voltage_flat": self.add_cyc_dep(
                tf.expand_dims(voltage_vector, axis = 1),
                count_dict
            ),
            "norm_constant_flat": self.add_volt_dep(
                features[:, 0:1],
                count_dict
            ),

            "norm_constant": features[:, 0:1],
            "cycles": cycles,
            "chg_rate": rates[:, 0:1],
            "dchg_rate": rates[:, 1:2],
            "cell_feat": features[:, 1:]
        }

        if training:

            var_cyc = tf.expand_dims(meas_cycles, axis=1) - cycles
            var_cyc_squared = tf.square(var_cyc)

            ''' discharge capacity '''
            cap, cap_der = self.create_derivatives(
                params,
                self.dchg_cap_part3,
                scalar = False
            )
            cap = tf.reshape(cap, [-1, voltage_count])

            pred_cap = (
                cap + var_cyc * tf.reshape(
                    cap_der['dCyc'], [-1, voltage_count]
                ) + var_cyc_squared * tf.reshape(
                    cap_der['d2Cyc'], [-1, voltage_count]
                )
            )

            ''' discharge max voltage '''
            max_dchg_vol, max_dchg_vol_der = self.create_derivatives(
                params,
                self.max_dchg_vol
            )
            max_dchg_vol = tf.reshape(max_dchg_vol, [-1])

            '''resistance derivatives '''
            r, r_der = self.create_derivatives(params, self.r)
            r = tf.reshape(r, [-1])

            '''eq_vol derivatives '''
            eq_vol, eq_vol_der = self.create_derivatives(params, self.eq_vol)
            eq_vol = tf.reshape(eq_vol, [-1])

            pred_max_dchg_vol = (
                max_dchg_vol + tf.reshape(max_dchg_vol_der['dCyc'], [-1])
                * tf.reshape(var_cyc, [-1])
                + tf.reshape(max_dchg_vol_der['d2Cyc'], [-1])
                * tf.reshape(var_cyc_squared, [-1])
            )

            return {
                "pred_cap": pred_cap,
                "pred_max_dchg_vol": tf.reshape(pred_max_dchg_vol, [-1]),
                "pred_eq_vol": tf.reshape(eq_vol, [-1]),
                "pred_r": tf.reshape(r, [-1]),
                "mean": mean,
                "log_sig": log_sig,
                "cap_der": cap_der,
                "max_dchg_vol_der": max_dchg_vol_der,
                "r_der": r_der,
                "eq_vol_der": eq_vol_der,
            }

        else:
            pred_cap = tf.reshape(
                self.dchg_cap_part3(params),
                [-1, voltage_vector.shape[0]]
            )
            pred_max_dchg_vol = self.max_dchg_vol(params)
            pred_eq_vol = self.eq_vol(params)
            pred_r = self.r(params)

            shift = self.shift(params)
            pred_eq_voltage_0 = self.eq_voltage_0(params)

            return {
                "pred_cap": pred_cap,
                "pred_max_dchg_vol": pred_max_dchg_vol,
                "pred_eq_vol": pred_eq_vol,
                "pred_r": pred_r,

                "shift": shift,
                "pred_eq_voltage_0": pred_eq_voltage_0
            }

# stores cell features
# key: index
# value: feature (matrix)
class DictionaryLayer(Layer):

    def __init__(self, num_features, num_keys):
        super(DictionaryLayer, self).__init__()
        self.num_features = num_features
        self.num_keys = num_keys
        self.kernel = self.add_weight(
            "kernel", shape=[self.num_keys, self.num_features * 2])

    def call(self, input, training=True):
        eps = tf.random.normal(
            shape=[self.num_keys, self.num_features])
        mean = self.kernel[:, :self.num_features]
        log_sig = self.kernel[:, self.num_features:]

        if not training:
            features = mean
        else:
            features = mean + tf.exp(log_sig / 2.) * eps

        # tf.gather: "fetching in the dictionary"
        fetched_features = tf.gather(features, input, axis=0)
        fetched_mean = tf.gather(mean, input, axis=0)
        fetched_log_sig = tf.gather(log_sig, input, axis=0)

        return fetched_features, fetched_mean, fetched_log_sig
