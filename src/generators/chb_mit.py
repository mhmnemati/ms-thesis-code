import mne
import wfdb
import glob
import numpy as np

from base import build


class Generator:
    url = "https://physionet.org/static/published-projects/chbmit/chb-mit-scalp-eeg-database-1.0.0.zip"
    name = "chb_mit"
    labels = [
        "normal",
        "seizure"
    ]

    random_seed = 100

    hparams = [
        {"window": 1},
    ]

    def __init__(self, window=1):
        self.window = window

    def __call__(self, path):
        records = sorted(glob.glob(f"{path}/**/*.edf", recursive=True))
        annotations = glob.glob(f"{path}/**/*.edf.seizures", recursive=True)

        montage = mne.channels.make_standard_montage("standard_1020")
        positions = {
            key.upper(): val for key, val in
            montage.get_positions()["ch_pos"].items()
        }

        return {
            "train": self.get_items(records, annotations, positions),
        }

    def get_items(self, records, annotations, positions):
        patients = {}
        for record in records:
            name = record.split("/")[-2]
            if name not in patients:
                patients[name] = []

            patients[name].append(record)

        for patient, records in patients.items():
            normals = []
            seizures = []
            for idx, record in enumerate(records):
                raw = mne.io.read_raw_edf(record, infer_types=True, include=[
                    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"
                ])
                if len(raw.ch_names) <= 0:
                    continue

                labels, _, _ = self.get_labels(raw, annotations)

                for label_idx in np.where(labels == 0)[0]:
                    normals.append([idx, label_idx])
                for label_idx in np.where(labels == 1)[0]:
                    seizures.append([idx, label_idx])

            normals = np.array(normals)
            seizures = np.array(seizures)

            np.random.seed(self.random_seed)
            selects = np.random.choice(normals.shape[0], seizures.shape[0] * 4)

            normals = normals[selects]

            normal_selections = {}
            for idx, label_idx in normals:
                if idx not in normal_selections:
                    normal_selections[idx] = []

                normal_selections[idx].append(label_idx)

            seizure_selections = {}
            for idx, label_idx in seizures:
                if idx not in seizure_selections:
                    seizure_selections[idx] = []

                seizure_selections[idx].append(label_idx)

            for idx, record in enumerate(records):
                raw = mne.io.read_raw_edf(record, infer_types=True, include=[
                    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"
                ])
                if len(raw.ch_names) <= 0:
                    continue

                labels, tmin, tmax = self.get_labels(raw, annotations)
                sources, targets, picks = self.get_montage(raw, positions)

                # TODO: resample raw to self.sfreq
                data = raw.get_data(tmin=tmin, tmax=tmax, picks=picks).astype(np.float32)

                for low in normal_selections[idx]:
                    high = low + self.window
                    sfreq = raw.info["sfreq"]

                    _labels = labels[int(low):int(high)]

                    yield f"{patient}/{self.labels[_labels.max()]}", {
                        "data": data[:, int(low*sfreq):int(high*sfreq)],
                        "labels": _labels,
                        "sources": sources,
                        "targets": targets,
                    }

                for low in seizure_selections[idx]:
                    high = low + self.window
                    sfreq = raw.info["sfreq"]

                    _labels = labels[int(low):int(high)]

                    yield f"{patient}/{self.labels[_labels.max()]}", {
                        "data": data[:, int(low*sfreq):int(high*sfreq)],
                        "labels": _labels,
                        "sources": sources,
                        "targets": targets,
                    }

    def get_labels(self, raw, annotations):
        seconds = int(raw.n_times / raw.info["sfreq"])
        labels = np.zeros(seconds, dtype=np.int64)

        if f"{raw.filenames[0]}.seizures" in annotations:
            seizure = wfdb.io.rdann(raw.filenames[0], extension="seizures")
            start = int(seizure.sample[0] / raw.info["sfreq"])
            finish = int(seizure.sample[1] / raw.info["sfreq"])
            labels[start:finish] = 1

        tmin = int(raw.times[0])
        tmax = int(raw.times[-1]) + 1

        return labels[tmin:tmax], tmin, tmax

    def get_montage(self, raw, positions):
        picks = mne.pick_types(raw.info, eeg=True)

        sources = []
        targets = []
        for idx, pick in enumerate(picks):
            channel = raw.info["ch_names"][pick]
            electrodes = channel.upper().split("-")
            sources.append(electrodes[0])
            targets.append(electrodes[1])

        return sources, targets, picks


build(Generator)
