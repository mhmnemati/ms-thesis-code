import mne
import wfdb
import glob
import random
import numpy as np
import scipy as sp
import torch as pt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from .base import BaseDataset


class Generator:
    sfreq = 100
    window = 1
    overlap = 0
    labels = [
        "Normal",
        "Seizure"
    ]

    def __call__(self, path):
        records = sorted(glob.glob(f"{path}/**/*.edf", recursive=True))
        annotations = glob.glob(f"{path}/**/*.edf.seizures", recursive=True)
        records = list(filter(lambda x: f"{x}.seizures" in annotations, records))
        random.shuffle(records)

        montage = mne.channels.make_standard_montage("standard_1020")
        positions = {
            key.upper(): val for key, val in
            montage.get_positions()["ch_pos"].items()
        }

        return {
            "train": self._get_items(records[slice(int(len(records) * 0.0), int(len(records) * 0.8))], annotations, positions),
            "valid": self._get_items(records[slice(int(len(records) * 0.8), int(len(records) * 1.0))], annotations, positions)
        }

    def _get_items(self, records, annotations, positions):
        for record in records:
            raw = mne.io.read_raw_edf(record, infer_types=True, include=[
                "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8"
            ])
            if len(raw.ch_names) <= 0:
                continue

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

    def _get_labels(self, raw, annotations):
        seconds = int(raw.n_times / self.sfreq)
        labels = np.zeros(seconds, dtype=np.int64)

        if f"{raw.filenames[0]}.seizures" in annotations:
            seizure = wfdb.io.rdann(raw.filenames[0], extension="seizures")
            start = int(seizure.sample[0] / raw.info["sfreq"])
            finish = int(seizure.sample[1] / raw.info["sfreq"])
            labels[start:finish] = 1

        non_zeros = np.nonzero(labels)
        tmin = max(int(raw.times[0]), np.min(non_zeros) - 10 * 60)
        tmax = min(int(raw.times[-1]), np.max(non_zeros) + 10 * 60)

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


class CHBMITDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(
            name="chb_mit",
            url="https://physionet.org/static/published-projects/chbmit/chb-mit-scalp-eeg-database-1.0.0.zip",
            generator=Generator(),
            transform=self.transform,
            data_loader=DataLoader,
            batch_size=kwargs["batch_size"],
        )

    def transform(self, item):
        data = item["data"]
        sources = item["sources"]
        targets = item["targets"]

        electrodes = np.unique(np.concatenate([sources, targets]), axis=0)

        node_features = np.zeros((electrodes.shape[0], data.shape[1]), dtype=np.float32)
        for i in range(len(data)):
            # TODO: use other transformations (wavelet, fourier, hilbert, ...)
            power = data[i] ** 2

            source_idx = np.argwhere((electrodes == sources[i]).all(1)).item()
            target_idx = np.argwhere((electrodes == targets[i]).all(1)).item()

            node_features[source_idx] += power / 2
            node_features[target_idx] += power / 2

        adjecancy_matrix = np.zeros((electrodes.shape[0], electrodes.shape[0]), dtype=np.float64)
        for i in range(electrodes.shape[0]):
            for j in range(electrodes.shape[0]):
                # TODO: construct graph edges methods (constant, clustering, dynamic, ...)
                distance = np.linalg.norm(electrodes[j] - electrodes[i])
                adjecancy_matrix[i, j] = 1 if distance < 0.1 else 0

        edge_index = from_scipy_sparse_matrix(sp.sparse.csr_matrix(adjecancy_matrix))[0]
        y = pt.tensor(item["label"])

        return Data(x=pt.from_numpy(node_features), edge_index=edge_index, y=y)

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("CHBMIT")
        parser.add_argument("--batch_size", type=int, default=8)
        return parent_parser
