import tensorflow as tf


def iterable_to_dataset(iterable):
    return tf.data.Dataset.from_generator(
        lambda x: iter(iterable),

    )