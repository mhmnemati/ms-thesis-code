import lightning as L

from data import CHBMITDataset
from models import Brain2Vec, EEGInception

model = Brain2Vec()
trainer = L.Trainer(max_epochs=5)
trainer.fit(model, Brain2Vec.loader(CHBMITDataset, train=True), Brain2Vec.loader(CHBMITDataset, train=False))

# model = EEGInception()
# trainer = L.Trainer(max_epochs=5)
# trainer.fit(model, EEGInception.loader(CHBMITDataset, train=True), EEGInception.loader(CHBMITDataset, train=False))
