import os
import mne
import glob
import random
import numpy as np

from .zip import ZipDataset


sleep_edf_20 = [
    "SC4001E0-PSG.edf",
    "SC4002E0-PSG.edf",
    "SC4011E0-PSG.edf",
    "SC4012E0-PSG.edf",
    "SC4021E0-PSG.edf",
    "SC4022E0-PSG.edf",
    "SC4031E0-PSG.edf",
    "SC4032E0-PSG.edf",
    "SC4041E0-PSG.edf",
    "SC4042E0-PSG.edf",
    "SC4051E0-PSG.edf",
    "SC4052E0-PSG.edf",
    "SC4061E0-PSG.edf",
    "SC4062E0-PSG.edf",
    "SC4071E0-PSG.edf",
    "SC4072E0-PSG.edf",
    "SC4081E0-PSG.edf",
    "SC4082E0-PSG.edf",
    "SC4091E0-PSG.edf",
    "SC4092E0-PSG.edf",
    "SC4101E0-PSG.edf",
    "SC4102E0-PSG.edf",
    "SC4111E0-PSG.edf",
    "SC4112E0-PSG.edf",
    "SC4121E0-PSG.edf",
    "SC4122E0-PSG.edf",
    "SC4131E0-PSG.edf",
    "SC4141E0-PSG.edf",
    "SC4142E0-PSG.edf",
    "SC4151E0-PSG.edf",
    "SC4152E0-PSG.edf",
    "SC4161E0-PSG.edf",
    "SC4162E0-PSG.edf",
    "SC4171E0-PSG.edf",
    "SC4172E0-PSG.edf",
    "SC4181E0-PSG.edf",
    "SC4182E0-PSG.edf",
    "SC4191E0-PSG.edf",
    "SC4192E0-PSG.edf",
]


class SleepEDFXDataset(ZipDataset):
    sfreq = 100
    window = 30
    overlap = 0
    labels = [
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R"
    ]

    def __init__(self, root=f"{os.path.expanduser('~')}/pytorch_datasets/sleep_edfx", train=True, transform=None):
        super().__init__(
            root=root,
            split="train" if train else "test",
            transform=transform,
            generators=self._split_generators,
            url="https://www.physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip"
        )

    def _split_generators(self, path):
        records = list(zip(
            sorted(glob.glob(f"{path}/**/*-PSG.edf", recursive=True)),
            sorted(glob.glob(f"{path}/**/*-Hypnogram.edf", recursive=True))
        ))
        records = list(filter(lambda x: x[0].split("/")[-1] in sleep_edf_20, records))
        random.shuffle(records)

        montage = mne.channels.make_standard_montage("standard_1020")
        positions = {
            key.upper(): val for key, val in
            montage.get_positions()["ch_pos"].items()
        }

        return {
            "train": self._get_items(records[slice(int(len(records) * 0.0), int(len(records) * 0.8))], positions),
            "test": self._get_items(records[slice(int(len(records) * 0.8), int(len(records) * 1.0))], positions),
        }

    def _get_items(self, records, positions):
        for record in records:
            raw = mne.io.read_raw_edf(record[0], infer_types=True, exclude=["Event marker", "Marker"])
            annotations = mne.read_annotations(record[1])

            labels, tmin, tmax = self._get_labels(raw, annotations)
            sources, targets, picks = self._get_montage(raw, positions)

            # TODO: resample raw to self.sfreq
            data = raw.get_data(tmin=tmin, tmax=tmax, picks=picks).astype(np.float32)

            for low in range(0, len(labels), self.window - self.overlap):
                high = low + self.window
                if high > len(labels):
                    break

                yield {
                    "data": data[:, low*self.sfreq:high*self.sfreq],
                    "label": labels[low:high].max(),
                    "sources": sources,
                    "targets": targets,
                }

    def _get_labels(self, raw, annotations, crop_wake_mins=30):
        seconds = int(raw.n_times / self.sfreq)
        labels = np.zeros(seconds, dtype=np.int64)

        for item in annotations:
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

    def _get_montage(self, raw, positions):
        picks = mne.pick_types(raw.info, eeg=True)

        sources = np.zeros((len(picks), 3), dtype=np.float32)
        targets = np.zeros((len(picks), 3), dtype=np.float32)
        for idx, pick in enumerate(picks):
            channel = raw.info["ch_names"][pick]
            electrodes = channel.upper().split("-")
            sources[idx] = positions[electrodes[0]]
            targets[idx] = positions[electrodes[1]]

        return sources, targets, picks
