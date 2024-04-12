import lightning as L

from data import CHBMITDataset
from models import Brain2Vec, EEGInception

train_set = CHBMITDataset(train=True, transform=Brain2Vec.transform)
test_set = CHBMITDataset(train=False, transform=Brain2Vec.transform)

model = Brain2Vec()
trainer = L.Trainer(max_epochs=5)
trainer.fit(model, Brain2Vec.loader(train_set), Brain2Vec.loader(test_set))
