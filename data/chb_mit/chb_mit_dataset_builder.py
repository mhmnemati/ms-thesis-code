import mne
import wfdb
import glob
import random
import warnings
import dataclasses
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

warnings.filterwarnings("ignore")


@dataclasses.dataclass
class Config(tfds.core.BuilderConfig):
    pass


class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    BUILDER_CONFIGS = [
        Config(name="all")
    ]

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
                "data": tfds.features.Tensor(shape=(None, self.sfreq * self.window), dtype=tf.float16),
                "label": tfds.features.ClassLabel(names=self.labels),
                "sources": tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
                "targets": tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.download_and_extract("https://physionet.org/static/published-projects/chbmit/chb-mit-scalp-eeg-database-1.0.0.zip")

        records = sorted(glob.glob(f"{path}/**/*.edf", recursive=True))
        annotations = glob.glob(f"{path}/**/*.edf.seizures", recursive=True)
        random.shuffle(records)

        montage = mne.channels.make_standard_montage("standard_1020")
        positions = {
            key.upper(): val for key, val in
            montage.get_positions()["ch_pos"].items()
        }

        return {
            "train": self._generate_examples(records[slice(int(len(records) * 0.0), int(len(records) * 0.8))], annotations, positions),
            "test": self._generate_examples(records[slice(int(len(records) * 0.8), int(len(records) * 1.0))], annotations, positions),
        }

    def _generate_examples(self, records, annotations, positions):
        for record in records:
            raw = mne.io.read_raw_edf(record, preload=False, include=[
                "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"
            ])

            if len(raw.ch_names) <= 0:
                continue

            labels, tmin, tmax = self._get_labels(raw, annotations)
            sources, targets, picks = self._get_montage(raw, positions)

            if labels is None:
                continue

            # TODO: resample raw to self.sfreq
            data = raw.get_data(picks=picks, tmin=tmin, tmax=tmax).astype(np.float16)

            for low in range(0, len(labels), self.window):
                key = f'{record.split("/")[-1]}_{low}'

                high = low + self.window
                if high > self.window:
                    break

                yield key, {
                    "data": data[:, low*self.sfreq:high*self.sfreq],
                    "label": labels[low:high].max(-1),
                    "sources": sources,
                    "targets": targets,
                }

    def _get_labels(self, raw, annotations, crop_wake_mins=30):
        seconds = int(raw.n_times / self.sfreq)
        labels = np.zeros(seconds)

        if f"{raw.filenames[0]}.seizures" in annotations:
            seizure = wfdb.io.rdann(raw.filenames[0], extension="seizures")
            start = int(seizure.sample[0] / raw.info["sfreq"])
            finish = int(seizure.sample[1] / raw.info["sfreq"])
            labels[start:finish] = 1

        non_zeros = np.nonzero(labels)

        if len(non_zeros[0]) <= 0:
            return None, None, None

        tmin = max(int(raw.times[0]), np.min(non_zeros) - crop_wake_mins * 60)
        tmax = min(int(raw.times[-1]), np.max(non_zeros) + crop_wake_mins * 60)

        return labels[tmin:tmax], tmin, tmax

    def _get_montage(self, raw, positions):
        picks = mne.pick_types(raw.info, eeg=True)

        print(raw.ch_names)

        sources = np.zeros((len(picks), 3), dtype=np.float32)
        targets = np.zeros((len(picks), 3), dtype=np.float32)
        for idx, pick in enumerate(picks):
            channel = raw.info["ch_names"][pick]
            electrodes = channel.upper().split("-")
            print(raw.filenames[0])
            print(channel)
            sources[idx] = positions[electrodes[0]]
            targets[idx] = positions[electrodes[1]]

        return sources, targets, picks
