import mne
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
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
        "Sleep stage ?",
        "Movement time"
    ]

    def _info(self):
        return self.dataset_info_from_configs(
            homepage="https://www.physionet.org/content/sleep-edfx/1.0.0/",
            supervised_keys=("data", "label"),
            features=tfds.features.FeaturesDict({
                "data": tfds.features.Tensor(shape=(7, self.sfreq * self.window), dtype=tf.float16),
                "label": tfds.features.ClassLabel(names=self.labels),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.download_and_extract("https://www.physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip")

        records = list(zip(
            sorted(glob.glob(f"{path}/**/*-PSG.edf", recursive=True)),
            sorted(glob.glob(f"{path}/**/*-Hypnogram.edf", recursive=True))
        ))
        random.shuffle(records)

        return {
            "train": self._generate_examples(records[slice(int(len(records) * 0.0), int(len(records) * 0.8))]),
            "test": self._generate_examples(records[slice(int(len(records) * 0.8), int(len(records) * 1.0))]),
        }

    def _generate_examples(self, records):
        for record in records:
            labels, tmin, tmax = self._get_labels(record)

            raw = mne.io.read_raw_edf(record[0], infer_types=True, exclude=["Event marker"])
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

    def _get_labels(self, record, crop_wake_mins=30):
        raw = mne.io.read_raw_edf(record[0], preload=False)
        annots = mne.read_annotations(record[1])

        seconds = int(raw.n_times / self.sfreq)
        labels = np.zeros(seconds)

        for item in annots:
            onset = int(item["onset"])
            duration = int(item["duration"])
            labels[onset:onset+duration] = (
                self.labels.index(item["description"])
                if item["description"] in self.labels else
                0
            )

        non_zeros = np.nonzero(labels)
        tmin = max(int(raw.times[0]), np.min(non_zeros) - crop_wake_mins * 60)
        tmax = min(int(raw.times[-1]), np.max(non_zeros) + crop_wake_mins * 60)

        return labels[tmin:tmax], tmin, tmax
