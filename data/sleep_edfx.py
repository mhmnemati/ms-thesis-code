import mne
import glob
import pathlib
import warnings
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")


class SleepEDFX(tf.data.Dataset):
    def __new__(cls, root=f"{pathlib.Path(__file__).parent}/downloads/sleep_edfx", split="train"):
        labels = [
            "Sleep stage W",
            "Sleep stage 1",
            "Sleep stage 2",
            "Sleep stage 3",
            "Sleep stage 4",
            "Sleep stage R",
            "Sleep stage ?"
        ]

        cls.label2id = {val: idx for idx, val in enumerate(labels)}
        cls.id2label = {idx: val for idx, val in enumerate(labels)}

        records = list(zip(
            sorted(glob.glob(f"{root}/**/*-PSG.edf", recursive=True)),
            sorted(glob.glob(f"{root}/**/*-Hypnogram.edf", recursive=True))
        ))

        cls.splits = {
            "train": records[int(len(records) * 0.0):int(len(records) * 0.6)],
            "valid": records[int(len(records) * 0.6):int(len(records) * 0.8)],
            "test": records[int(len(records) * 0.8):int(len(records) * 1.0)],
        }

        return tf.data.Dataset.from_generator(
            cls._generator, args=(split,),
            output_signature=(
                tf.TensorSpec(shape=(7, 3000), dtype=tf.int32),
                tf.TensorSpec(shape=(1), dtype=tf.int32)
            )
        )

    def _generator(split):
        cls = SleepEDFX

        for record in cls.splits[split.decode("ASCII")]:
            raw = mne.io.read_raw_edf(record[0], verbose=False)
            annotation = mne.read_annotations(record[1])

            seconds = int(raw.n_times / raw.info["sfreq"])
            labels = np.zeros(seconds)

            for item in annotation:
                onset = int(item["onset"])
                duration = int(item["duration"])
                labels[onset:onset+duration] = cls.label2id[item["description"]]

            crop_wake_mins = 30
            non_zeros = np.nonzero(labels)
            tmin = max(raw.times[0], np.min(non_zeros) - crop_wake_mins * 60)
            tmax = min(raw.times[-1], np.max(non_zeros) + crop_wake_mins * 60)

            for second in range(tmin, tmax, 30):
                data = raw.get_data(tmin=second, tmax=second+30)

                X = tf.convert_to_tensor(data)
                y = tf.constant([labels[second+15]])

                yield X, y
