import os
import mne
import wfdb
import glob
import random
import numpy as np
import torch as pt

import warnings
warnings.filterwarnings("ignore")


def get_labels(raw, annotations):
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


def get_montage(raw, positions):
    picks = mne.pick_types(raw.info, eeg=True)

    sources = []
    targets = []
    for idx, pick in enumerate(picks):
        channel = raw.info["ch_names"][pick]
        electrodes = channel.upper().split("-")
        sources.append(electrodes[0])
        targets.append(electrodes[1])

    return sources, targets, picks


def get_seizures_duration(records, annotations):
    patient_seizures = 0
    for record in records:
        raw = mne.io.read_raw_edf(record, infer_types=True, verbose=False, include=[
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"
        ])

        if f"{raw.filenames[0]}.seizures" in annotations:
            seizure = wfdb.io.rdann(raw.filenames[0], extension="seizures")
            start = int(seizure.sample[0] / raw.info["sfreq"])
            finish = int(seizure.sample[1] / raw.info["sfreq"])
            patient_seizures += (finish - start)

    return patient_seizures


root = "/home/mhnemati/pytorch_datasets"
path = f"{root}/chb_mit_raw"
patient_paths = glob.glob(f"{path}/*")

os.makedirs(f"{root}/chb_mit_transformed", exist_ok=True)
for patient_path in patient_paths:
    records = sorted(glob.glob(f"{patient_path}/*.edf", recursive=True))
    annotations = glob.glob(f"{patient_path}/*.edf.seizures", recursive=True)

    patient_seizures = get_seizures_duration(records, annotations)

    window = 1
    overlap = 0

    patient_name = os.path.basename(patient_path)
    os.makedirs(f"{root}/chb_mit_transformed/{patient_name}/normal", exist_ok=True)
    os.makedirs(f"{root}/chb_mit_transformed/{patient_name}/seizure", exist_ok=True)
    x = 0
    y = 0
    for record in records:
        raw = mne.io.read_raw_edf(record, infer_types=True, include=[
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"
        ])

        labels, tmin, tmax = get_labels(raw, annotations)
        sources, targets, picks = get_montage(raw, None)

        data = raw.get_data(tmin=tmin, tmax=tmax, picks=picks).astype(np.float32)

        low = 0
        while True:
            high = low + window
            sfreq = raw.info["sfreq"]
            if high >= len(labels):
                break

            item = {
                "data": data[:, int(low*sfreq):int(high*sfreq)],
                "label": max(labels[int(low)], labels[int(high)]),
                "sources": sources,
                "targets": targets,
            }

            if patient_seizures < 400 and item["label"] == 1:
                low += 1/2
            else:
                low += 1

            if item["label"] == 0:
                x += 1
                pt.save(item, f"{root}/chb_mit_transformed/{patient_name}/normal/{x}.pt")
            else:
                y += 1
                pt.save(item, f"{root}/chb_mit_transformed/{patient_name}/seizure/{y}.pt")
