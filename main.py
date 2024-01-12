import keras
import tensorflow as tf

from data.SleepEDFX import SleepEDFX
from models.DeepSleepNet import DeepSleepNet

trainset = SleepEDFX(split="train")
validset = SleepEDFX(split="valid")

model = DeepSleepNet()
model.build((None, 7, 3000))

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(trainset, epochs=10, batch_size=2, validation_data=validset)
