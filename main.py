from data import SleepEDFX
from models import DeepSleepNet
from callbacks import callbacks

import tensorflow as tf

validset = SleepEDFX(split="valid", window_secs=30)

# model = DeepSleepNet()
# model.build()
# model.fit(trainset, epochs=10, batch_size=4, validation_data=validset, callbacks=callbacks)

tf.data.Dataset.from_generator(
    lambda x: iter(SleepEDFX(split="train", window_secs=30)),
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
