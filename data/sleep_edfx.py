import mne
import glob
import random
import pathlib
import warnings
import numpy as np

warnings.filterwarnings("ignore")

path = pathlib.Path(__file__).parent


class SleepEDFX:
    def __init__(self, root=f"{path}/downloads/sleep_edfx", seed=1, split="train", crop_wake_mins=30):
        self.label2id = {
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 4,
            "Sleep stage R": 5,
            "Sleep stage ?": 0,
            "Movement time": 0
        }

        self.id2label = {idx: label for label, idx in reversed(self.label2id.items())}

        records = list(zip(
            sorted(glob.glob(f"{root}/**/*-PSG.edf", recursive=True)),
            sorted(glob.glob(f"{root}/**/*-Hypnogram.edf", recursive=True))
        ))
        random.Random(seed).shuffle(records)

        split = {
            "train": (int(len(records) * 0.0), int(len(records) * 0.6)),
            "valid": (int(len(records) * 0.6), int(len(records) * 0.8)),
            "test": (int(len(records) * 0.8), int(len(records) * 1.0)),
        }[split]

        self.records = records[split[0]:split[1]]
        self.crop_wake_mins = crop_wake_mins

    def __iter__(self):
        for record in self.records:
            raw = mne.io.read_raw_edf(record[0], verbose=False)
            annotation = mne.read_annotations(record[1])

            seconds = int(raw.n_times / raw.info["sfreq"])
            labels = np.zeros(seconds)

            for item in annotation:
                onset = int(item["onset"])
                duration = int(item["duration"])
                labels[onset:onset+duration] = self.label2id[item["description"]]

            non_zeros = np.nonzero(labels)
            tmin = max(int(raw.times[0]), np.min(non_zeros) - self.crop_wake_mins * 60)
            tmax = min(int(raw.times[-1]), np.max(non_zeros) + self.crop_wake_mins * 60)

            data = raw.get_data(tmin=tmin, tmax=tmax)
            labels = labels[tmin:tmax]
            channels = raw.info["ch_names"]
            frequency = int(raw.info["sfreq"])

            yield data, labels, channels, frequency
