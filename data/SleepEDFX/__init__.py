import mne
import glob
import pathlib
import warnings
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")

root = pathlib.Path(__file__).parent


class SleepEDFX(tf.data.Dataset):
    def __new__(cls, root=f"{root}/downloads", split="train"):
        return tf.data.Dataset.from_generator(
            cls._generator, args=(root, split),
            output_signature=(
                tf.TensorSpec(shape=(7, 3000), dtype=tf.int32),
                tf.TensorSpec(shape=(1), dtype=tf.int32)
            )
        ).batch(2)

    def _generator(root, split):
        root = root.decode("ASCII")
        split = split.decode("ASCII")

        records = sorted(glob.glob(f"{root}/**/*-PSG.edf", recursive=True))
        annotations = sorted(glob.glob(f"{root}/**/*-Hypnogram.edf", recursive=True))

        label2id = {
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 4,
            "Sleep stage R": 5,
            "Sleep stage ?": 6
        }

        id2label = {val: key for key, val in label2id.items()}

        for idx in range(len(records)):
            record = mne.io.read_raw_edf(records[idx], verbose=False)
            annotation = mne.read_annotations(annotations[idx])

            seconds = int(record.n_times / record.info["sfreq"])

            labels = np.zeros(seconds)
            for annot in annotation:
                onset = int(annot["onset"])
                duration = int(annot["duration"])
                labels[onset:onset+duration] = label2id[annot["description"]]

            for second in range(0, seconds, 30):
                data = record.get_data(start=0, stop=3000)

                X = tf.convert_to_tensor(data)
                y = tf.constant([1])

                yield X, y
