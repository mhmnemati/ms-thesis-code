import mne
import wfdb
import glob
import random
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

warnings.filterwarnings("ignore")


class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    sfreq = 100
    window = 30
    labels = [
        "Normal",
        "Seizure"
    ]

    def _info(self):
        return self.dataset_info_from_configs(
            homepage="https://physionet.org/content/chbmit/1.0.0/",
            supervised_keys=("data", "label"),
            features=tfds.features.FeaturesDict({
                "data": tfds.features.Tensor(shape=(7, self.sfreq * self.window), dtype=tf.float16),
                "label": tfds.features.ClassLabel(names=self.labels),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.download_and_extract("https://physionet.org/static/published-projects/chbmit/chb-mit-scalp-eeg-database-1.0.0.zip")

        records = sorted(glob.glob(f"{path}/**/*.edf", recursive=True))
        annotations = glob.glob(f"{path}/**/*.edf.seizures", recursive=True)
        random.shuffle(records)

        return {
            "train": self._generate_examples(records[slice(int(len(records) * 0.0), int(len(records) * 0.8))], annotations),
            "test": self._generate_examples(records[slice(int(len(records) * 0.8), int(len(records) * 1.0))], annotations),
        }

    def _generate_examples(self, records, annotations):
        for record in records:
            labels, tmin, tmax = self._get_labels(record, annotations)

            raw = mne.io.read_raw_edf(record, infer_types=True)
            # TODO: resample raw to self.sfreq

            picks = mne.pick_types(raw.info, eeg=True)
            data = raw.get_data(picks=picks, tmin=tmin, tmax=tmax)

            for low in range(0, len(labels), self.window):
                key = f'{record[0].split("/")[-1]}_{low}'
                high = low + self.window
                if high > self.window:
                    break

                yield key, {
                    "data": data[:, low*self.sfreq:high*self.sfreq],
                    "label": labels[low:high].max(-1)
                }

    def _get_labels(self, record, annotations):
        raw = mne.io.read_raw_edf(record, preload=False)

        seconds = int(raw.n_times / self.sfreq)
        labels = np.zeros(seconds)

        if f"{record}.seizures" in annotations:
            seizure = wfdb.io.rdann(record, extension="seizures")
            start = int(seizure.sample[0] / raw.info["sfreq"])
            finish = int(seizure.sample[1] / raw.info["sfreq"])
            labels[start:finish] = self.labels[1]

        tmin = 0
        tmax = seconds

        return labels[tmin:tmax], tmin, tmax
