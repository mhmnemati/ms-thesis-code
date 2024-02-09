import mne
import wfdb
import glob
import random
import pathlib
import warnings
import numpy as np

warnings.filterwarnings("ignore")

path = pathlib.Path(__file__).parent


class CHBMIT:
    def __init__(self, root=f"{path}/downloads/chb_mit", seed=1, split="train"):
        self.label2id = {
            "Normal": 0,
            "Seizure": 1
        }

        self.id2label = {idx: label for label, idx in reversed(self.label2id.items())}

        records = sorted(glob.glob(f"{root}/**/*.edf", recursive=True))
        random.Random(seed).shuffle(records)

        split = {
            "train": (int(len(records) * 0.0), int(len(records) * 0.6)),
            "valid": (int(len(records) * 0.6), int(len(records) * 0.8)),
            "test": (int(len(records) * 0.8), int(len(records) * 1.0)),
        }[split]

        self.records = records[split[0]:split[1]]
        self.annotations = sorted(glob.glob(f"{root}/**/*.edf.seizures", recursive=True))

    def __iter__(self):
        for record in self.records:
            raw = mne.io.read_raw_edf(record, verbose=False)

            seconds = int(raw.n_times / raw.info["sfreq"])
            labels = np.zeros(seconds)

            if f"{record}.seizures" in self.annotations:
                seizure = wfdb.io.rdann(record, extension="seizures")
                start = int(seizure.sample[0] / raw.info["sfreq"])
                finish = int(seizure.sample[1] / raw.info["sfreq"])
                labels[start:finish] = self.label2id["Seizure"]

            tmin = 0
            tmax = seconds

            data = raw.get_data(tmin=tmin, tmax=tmax)
            labels = labels[tmin:tmax]
            channels = raw.info["ch_names"]
            frequency = int(raw.info["sfreq"])

            yield data, labels, channels, frequency
