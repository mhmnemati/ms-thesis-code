import keras
import tensorflow as tf


def block_1(conv_filters, conv_kernel_size, conv_padding, conv_strides, pool_padding, pool_size, pool_stride):
    return keras.Sequential([
        keras.layers.ZeroPadding2D(padding=conv_padding),
        keras.layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.ZeroPadding2D(padding=pool_padding),
        keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_stride),
        keras.layers.Dropout(0.5)
    ])


def sub_block_2(filters, kernel_size, padding, strides):
    return keras.Sequential([
        keras.layers.ZeroPadding2D(padding=padding),
        keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU()
    ])


def block_2(conv_filters, conv_kernel_size, conv_padding, conv_strides, pool_padding, pool_size, pool_strides):
    return keras.Sequential([
        sub_block_2(conv_filters, conv_kernel_size, conv_padding, conv_strides),
        sub_block_2(conv_filters, conv_kernel_size, conv_padding, conv_strides),
        sub_block_2(conv_filters, conv_kernel_size, conv_padding, conv_strides),
        keras.layers.ZeroPadding2D(padding=pool_padding),
        keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides),
        keras.layers.Flatten()
    ])


class DeepSleepNet(keras.Model):
    def __init__(self, num_class):
        super().__init__()

        self.conv_small = tf.keras.Sequential([
            block_1(
                conv_filters=64,
                conv_kernel_size=(1, 400),
                conv_padding=(0, 175),
                conv_strides=(1, 50),
                pool_padding=0,
                pool_size=(1, 4),
                pool_stride=(1, 4)
            ),
            block_2(
                conv_filters=128,
                conv_kernel_size=(1, 6),
                conv_padding=1,
                conv_strides=1,
                pool_padding=(0, 1),
                pool_size=(1, 2),
                pool_strides=(1, 2)
            )
        ])

        self.conv_large = tf.keras.Sequential([
            block_1(
                conv_filters=64,
                conv_kernel_size=(1, 50),
                conv_padding=(0, 22),
                conv_strides=(1, 6),
                pool_padding=(0, 2),
                pool_size=(1, 8),
                pool_stride=(1, 8)
            ),
            block_2(
                conv_filters=128,
                conv_kernel_size=(1, 8),
                conv_padding=1,
                conv_strides=1,
                pool_padding=(0, 1),
                pool_size=(1, 4),
                pool_strides=(1, 4)
            )
        ])

        self.dropout = keras.layers.Dropout(0.5)

        self.dense = keras.layers.Dense(1024)
        self.lstm = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
            keras.layers.Dropout(0.5),
            keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=False)),
            keras.layers.Dropout(0.5),
        ])

        self.final = keras.Sequential([
            keras.layers.Add(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_class, activation="softmax"),
        ])

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)

        x1 = self.conv_small(x)
        x2 = self.conv_large(x)

        x = tf.concat([x1, x2], axis=-1)
        x = self.dropout(x)

        x1 = self.dense(x)
        x2 = self.lstm(tf.expand_dims(x, axis=-1))

        return self.final([x1, x2])
