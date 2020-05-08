import tensorflow as tf
from tensorflow.keras.layers import Layer

from machine_learning.incentives import *


# stores cell features
# key: index
# value: feature (matrix)
class PrimitiveDictionaryLayer(Layer):

    def __init__(self, num_feats, id_dict):
        super(PrimitiveDictionaryLayer, self).__init__()
        self.num_features = num_feats
        self.num_keys = 1 + max(id_dict.values())
        self.id_dict = id_dict
        self.kernel = self.add_weight(
            "kernel", shape = [self.num_keys, self.num_features]
        )
        self.sample_epsilon = 0.05

    def get_main_ker(self):
        return self.kernel.numpy()

    def __call__(self, input, training = True, sample = False):
        fetched_features = tf.gather(self.kernel, input, axis = 0)
        if training:
            features_loss = .1 * incentive_magnitude(
                fetched_features,
                Target.Small,
                Level.Proportional
            )
            features_loss = tf.reduce_mean(
                features_loss,
                axis = 1,
                keepdims = True
            )

        else:
            features_loss = None

        if sample:
            eps = tf.random.normal(
                shape = [input.shape[0], self.num_features]
            )
            fetched_features = fetched_features + self.sample_epsilon * eps

        return fetched_features, features_loss
