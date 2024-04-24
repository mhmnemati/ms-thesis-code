import mne
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

    def __call__(self, path):
        records = list(zip(
            sorted(glob.glob(f"{path}/**/*-PSG.edf", recursive=True)),
            sorted(glob.glob(f"{path}/**/*-Hypnogram.edf", recursive=True))
        ))
        records = list(filter(lambda x: x[0].split("/")[-1] in self.sleep_edf_20, records))
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


class SleepEDFXDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(
            name="sleep_edfx",
            url="https://www.physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip",
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
