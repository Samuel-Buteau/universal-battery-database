import tensorflow as tf


def gather0(tensor, indices):
    """ Wrapper for `tf.gather` with `axis = 0`"""
    return tf.gather(tensor, indices = indices, axis = 0)


# TODO(harvey): use this function for cleanup
def tile_then_reshape(tensor, tile, reshape):
    """ Wrapper for `tf.tile` and `tf.reshape.

    Args:
        tensor: tensor to be shaped
        tile: shape of `tf.tile`
        reshape: shape of `tf.reshape`
    """
    return tf.tile(tf.reshape(tensor, reshape), tile)
