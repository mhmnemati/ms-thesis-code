from data import SleepEDFX
from models import DeepSleepNet
from callbacks import callbacks

trainset = SleepEDFX(split="train", window_secs=30).batch(4)
validset = SleepEDFX(split="valid", window_secs=30).batch(4)

model = DeepSleepNet()
model.build()
model.fit(trainset, epochs=10, batch_size=4, validation_data=validset, callbacks=callbacks)
