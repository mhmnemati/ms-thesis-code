import mne
import glob
import pathlib
import warnings
import numpy as np

warnings.filterwarnings("ignore")

path = pathlib.Path(__file__).parent


class SleepEDFX:
    def __init__(self, root=f"{path}/downloads/sleep_edfx", window_secs=1, crop_wake_mins=30):
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

        self.records = list(zip(
            sorted(glob.glob(f"{root}/**/*-PSG.edf", recursive=True)),
            sorted(glob.glob(f"{root}/**/*-Hypnogram.edf", recursive=True))
        ))

        self.window_secs = window_secs
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
            tmax = tmin + int((tmax - tmin) / self.window_secs) * self.window_secs

            data = raw.get_data(tmin=tmin, tmax=tmax).T.reshape(-1, self.window_secs * int(raw.info["sfreq"]), raw.info["nchan"])
            labels = labels[tmin:tmax].reshape(-1, self.window_secs).max(-1)

            yield data, labels

            # data = raw.get_data(tmin=tmin, tmax=tmax)
            # sfreq =

            # yield

            # data = .T.reshape(-1, self.window_secs * int(raw.info["sfreq"]), raw.info["nchan"])
            # labels = labels[tmin:tmax].reshape(-1, self.window_secs).max(-1)

            # yield data, labels
