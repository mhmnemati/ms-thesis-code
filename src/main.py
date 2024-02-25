import tensorflow as tf
import tensorflow_datasets as tfds

from models import DeepSleepNet
from callbacks import callbacks

# tf.keras.mixed_precision.set_global_policy("mixed_float16")

train_set, test_set = tfds.load("sleep_edfx/20", split=["train", "test"], as_supervised=True)

BATCH = 8
AUTOTUNE = tf.data.AUTOTUNE
train_set = train_set.cache().batch(BATCH).prefetch(AUTOTUNE)
test_set = test_set.cache().batch(BATCH).prefetch(AUTOTUNE)

model = DeepSleepNet(num_class=6)

model.build((None, 2, 3000))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_set, epochs=10, validation_data=test_set, callbacks=callbacks)
