import mne
import glob
import numpy as np

from base import build


class Generator:
    url = "https://www.physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip"
    seed = 100
    name = "sleep_edfx"
    label2id = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }

    hparams = [
        {"window": 30, "overlap": 0},
        # {"window": 30, "overlap": 1},
        # {"window": 30, "overlap": 5},
    ]

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

    def __init__(self, window=1, overlap=0):
        self.window = window
        self.overlap = overlap

    def __call__(self, path):
        records = list(zip(
            sorted(glob.glob(f"{path}/**/*-PSG.edf", recursive=True)),
            sorted(glob.glob(f"{path}/**/*-Hypnogram.edf", recursive=True))
        ))
        records = list(filter(lambda x: x[0].split("/")[-1] in self.sleep_edf_20, records))

        montage = mne.channels.make_standard_montage("standard_1020")
        positions = {
            key.upper(): val for key, val in
            montage.get_positions()["ch_pos"].items()
        }

        return {
            "train": self.get_items(records, positions),
        }

    def get_items(self, records, positions):
        patients = {}
        for record in records:
            name = record[0].split("/")[-1][:5]
            if name not in patients:
                patients[name] = []

            patients[name].append(record)

        for patient, records in patients.items():
            for idx, record in enumerate(records):
                raw = mne.io.read_raw_edf(record[0], infer_types=True, exclude=["Event marker", "Marker"])
                annotation = mne.read_annotations(record[1])

                labels, tmin, tmax = self.get_labels(raw, annotation)
                ch_names, picks = self.get_montage(raw, positions)

                data = raw.get_data(tmin=tmin, tmax=tmax, picks=picks).astype(np.float32)

                for low in range(0, len(labels), self.window - self.overlap):
                    high = low + self.window
                    sfreq = raw.info["sfreq"]
                    if high >= len(labels):
                        break

                    yield f"{patient}", {
                        "data": data[:, int(low*sfreq):int(high*sfreq)],
                        "labels": labels[low:high],
                        "ch_names": ch_names,
                    }

    def get_labels(self, raw, annotation, crop_wake_mins=30):
        seconds = int(raw.n_times / raw.info["sfreq"])
        labels = np.zeros(seconds, dtype=np.int64)

        for item in annotation:
            onset = int(item["onset"])
            duration = int(item["duration"])
            labels[onset:onset+duration] = (
                self.label2id[item["description"]]
                if item["description"] in self.label2id else 0
            )

        non_zeros = np.nonzero(labels)
        tmin = max(int(raw.times[0]), np.min(non_zeros) - crop_wake_mins * 60)
        tmax = min(int(raw.times[-1]), np.max(non_zeros) + crop_wake_mins * 60)

        return labels[tmin:tmax], tmin, tmax

    def get_montage(self, raw, positions):
        picks_eeg = list(mne.pick_types(raw.info, eeg=True))
        picks_eog = list(mne.pick_types(raw.info, eog=True))
        picks_emg = list(mne.pick_types(raw.info, emg=True))

        names_eeg = [f"EEG {raw.info['ch_names'][p]}" for p in picks_eeg]
        names_eog = [f"EOG {raw.info['ch_names'][p]}" for p in picks_eog]
        names_emg = [f"EMG {raw.info['ch_names'][p]}" for p in picks_emg]

        return (names_eeg + names_eog + names_emg), (picks_eeg + picks_eog + picks_emg)


build(Generator)
