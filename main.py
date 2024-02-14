import data.sleep_edfx
import tensorflow as tf
import tensorflow_datasets as tfds

from models import DeepSleepNet
from callbacks import callbacks


train_set, test_set = tfds.load("sleep_edfx", split=["train[:5%]", "test[:5%]"], as_supervised=True)


def process(data, label):
    return data, label


AUTOTUNE = tf.data.AUTOTUNE

train_set = train_set.map(process, num_parallel_calls=AUTOTUNE).cache().batch(8).prefetch(AUTOTUNE)
test_set = test_set.map(process, num_parallel_calls=AUTOTUNE).cache().batch(8).prefetch(AUTOTUNE)

model = DeepSleepNet(num_class=6)

model.build((None, 2, 3000))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_set, epochs=10,
    batch_size=8,
    validation_data=test_set,
    validation_batch_size=8,
    callbacks=callbacks,
)
