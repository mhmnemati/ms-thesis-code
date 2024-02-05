import mne
import wfdb
import glob
import pathlib
import warnings
import numpy as np

warnings.filterwarnings("ignore")

path = pathlib.Path(__file__).parent


class CHBMIT:
    def __init__(self, root=f"{path}/downloads/chb_mit"):
        self.label2id = {
            "Normal": 0,
            "Seizure": 1
        }

        self.id2label = {idx: label for label, idx in reversed(self.label2id.items())}

        self.records = sorted(glob.glob(f"{root}/**/*.edf", recursive=True))
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

            data = raw.get_data().reshape(len(raw.info["ch_names"]), -1, int(raw.info["sfreq"]))

            features = {
                val: data[key] for key, val in enumerate(raw.info["ch_names"])
            }

            yield features, labels

    def montage(self):
        pass
