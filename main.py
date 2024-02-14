import data.sleep_edfx
import tensorflow as tf
import tensorflow_datasets as tfds

from models import DeepSleepNet
from callbacks import callbacks

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
except ValueError:
    tpu = None

if tpu:
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
else:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

train_set, test_set = tfds.load("sleep_edfx", split=["train[:5%]", "test[:5%]"], as_supervised=True)

BATCH = 16
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
model.fit(
    train_set, epochs=10,
    batch_size=BATCH,
    validation_data=test_set,
    validation_batch_size=BATCH,
    callbacks=callbacks,
)
