import tensorflow as tf
from tensorflow.keras.layers import Layer

main_activation = "relu"


class StressToEncodedLayer(Layer):
    def __init__(self, n_channels):
        super(StressToEncodedLayer, self).__init__()
        self.n_channels = n_channels
        self.input_kernel = self.add_weight(
            "input_kernel", shape = [1, 1, 1, 4 + 1, self.n_channels],
        )

        self.v_i_kernel_1 = self.add_weight(
            "v_i_kernel_1", shape = [3, 3, 1, self.n_channels, self.n_channels],
        )

        self.v_i_kernel_2 = self.add_weight(
            "v_i_kernel_2", shape = [3, 3, 1, self.n_channels, self.n_channels],
        )

        self.t_kernel = self.add_weight(
            "t_kernel", shape = [1, 1, 3, self.n_channels, self.n_channels],
        )

        self.output_kernel = self.add_weight(
            "output_kernel",
            shape = [1, 1, 1, self.n_channels, self.n_channels],
        )

    def __call__(self, input, training = True):
        # tensor; dim: [batch, n_sign, n_voltage, n_current, n_temperature, 4]
        svit_grid = input[0]

        # tensor; dim: [batch, n_sign, n_voltage, n_current, n_temperature, 1]
        count_matrix = input[1]

        count_matrix_0 = count_matrix[:, 0, :, :, :, :]
        count_matrix_1 = count_matrix[:, 1, :, :, :, :]

        svit_grid_0 = svit_grid[:, 0, :, :, :, :]
        svit_grid_1 = svit_grid[:, 1, :, :, :, :]

        val_0 = tf.concat((svit_grid_0, count_matrix_0), axis = -1)
        val_1 = tf.concat((svit_grid_1, count_matrix_1), axis = -1)

        filters = [
            (self.input_kernel, "none"),
            (0, "branch"),
            (self.v_i_kernel_1, main_activation),
            (self.v_i_kernel_2, "none"),
            (0, "combine"),
            (self.t_kernel, main_activation),
            (self.output_kernel, "none")
        ]

        for fil, activ in filters:
            if activ == "branch":
                val_0_save = val_0
                val_1_save = val_1
                continue
            if activ == "combine":
                val_0 = tf.nn.relu(val_0 + val_0_save)
                val_1 = tf.nn.relu(val_1 + val_1_save)
                continue
            val_0 = tf.nn.convolution(
                input = val_0, filters = fil, padding = "SAME"
            )
            val_1 = tf.nn.convolution(
                input = val_1, filters = fil, padding = "SAME"
            )

            if activ is "relu":
                val_0 = tf.nn.relu(val_0)
                val_1 = tf.nn.relu(val_1)
            elif activ is "elu":
                val_0 = tf.nn.elu(val_0)
                val_1 = tf.nn.elu(val_1)

        # each entry is scaled by its count.
        val_0 = val_0 * count_matrix_0
        val_1 = val_1 * count_matrix_1

        # then we take the average over all the grid.
        val_0 = tf.reduce_mean(val_0, axis = [1, 2, 3], keepdims = False)
        val_1 = tf.reduce_mean(val_1, axis = [1, 2, 3], keepdims = False)

        return val_0 + val_1
