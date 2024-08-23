import os
import mne
import glob
import numpy as np

from base import build


class Generator:
    url = "https://sleeptight.isr.uc.pt/"
    seed = 100
    name = "isruc"
    label2id = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "5": 4,
    }

    hparams = [
        {"window": 30, "overlap": 0},
        # {"window": 30, "overlap": 1},
        # {"window": 30, "overlap": 5},
    ]

    def __init__(self, window=1, overlap=0):
        self.window = window
        self.overlap = overlap

    def __call__(self, path):
        records = sorted(glob.glob(f"{path}/**/*.edf", recursive=True))

        montage = mne.channels.make_standard_montage("standard_1020")
        positions = {
            key.upper(): val for key, val in
            montage.get_positions()["ch_pos"].items()
        }

        return {
            "train": self.get_items(records, positions),
        }

    def get_items(self, records, positions):
        for idx, record in enumerate(records):
            raw = mne.io.read_raw_edf(record, infer_types=True, include=[
                "F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1",
                "F3-M2", "C3-M2", "O1-M2", "F4-M1", "C4-M1", "O2-M1",
            ])
            if len(raw.ch_names) <= 0:
                continue

            labels, tmin, tmax = self.get_labels(raw)
            sources, targets, ch_names, picks = self.get_montage(raw, positions)

            data = raw.get_data(tmin=tmin, tmax=tmax, picks=picks).astype(np.float32)

            for low in range(0, len(labels), self.window - self.overlap):
                high = low + self.window
                sfreq = raw.info["sfreq"]
                if high >= len(labels):
                    break

                yield f"{idx}", {
                    "data": data[:, int(low*sfreq):int(high*sfreq)],
                    "labels": labels[low:high],
                    "sources": sources,
                    "targets": targets,
                    "ch_names": ch_names,
                }

    def get_labels(self, raw):
        seconds = int(raw.n_times / raw.info["sfreq"])
        labels = np.zeros(seconds, dtype=np.int64)

        with open(raw.filenames[0].replace(".edf", "_1.txt")) as f:
            annotations = f.read().split("\n")[:-1]
            for idx, annotation in enumerate(annotations):
                labels[(idx*30):((idx+1)*30)] = (
                self.label2id[annotation]
                if annotation in self.label2id else 0
            )

        tmin = int(raw.times[0])
        tmax = int(raw.times[-1]) + 1

        return labels[tmin:tmax], tmin, tmax

    def get_montage(self, raw, positions):
        picks = mne.pick_types(raw.info, eeg=True)

        sources = np.zeros((len(picks), 3), dtype=np.float32)
        targets = np.zeros((len(picks), 3), dtype=np.float32)
        for idx, pick in enumerate(picks):
            channel = raw.info["ch_names"][pick]
            electrodes = channel.upper().split("-")
            sources[idx] = positions[electrodes[0]]
            targets[idx] = positions[electrodes[1]]

        ch_names = [f"EEG {raw.info['ch_names'][p]}" for p in picks]

        return sources, targets, ch_names, picks


build(Generator)
