import mne
import wfdb
import glob
import pathlib
# import numpy as np
# import tensorflow as tf

path = pathlib.Path(__file__).parent


class CHBMIT:
    def __init__(self):
        pass

    def __next__(self):
        pass


def main():
    records = glob.glob(f"{path}/downloads/**/*.edf", recursive=True)
    seizures = glob.glob(f"{path}/downloads/**/*.edf.seizures", recursive=True)

    for record in records:
        raw = mne.io.read_raw_edf(record)

        if f"{record}.seizures" in seizures:
            seizure = wfdb.io.rdann(record, extension="seizures")
            start = seizure.sample[0] / raw.info["sfreq"]
            finish = seizure.sample[1] / raw.info["sfreq"]
            raw.set_annotations(mne.Annotations(onset=start, duration=(finish - start), description="seizure"))

        raw.plot(duration=5, n_channels=30)
        input()


# def preprocess():
#     pass

# def postprocess():
#     pass

# def reverse_postprocess():
#     pass

# # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# # def batch(iterable, n=1):
# #     l = len(iterable)
# #     for ndx in range(0, l, n):
# #         yield iterable[ndx:min(ndx + n, l)]


# raw = mne.io.read_raw_edf("./SC4001E0-PSG.edf")

# seizure = wfdb.io.rdann("./chb03_01.edf", extension="seizures")
# start = seizure.sample[0] / raw.info["sfreq"]
# finish = seizure.sample[1] / raw.info["sfreq"]
# raw.set_annotations(mne.Annotations(onset=start, duration=(finish - start), description="seizure"))

# raw.set_annotations(mne.read_annotations("./SC4001EC-Hypnogram.edf"))

# raw.plot(duration=5, n_channels=30)

# input()


# def generator(subset=0, batch_size=20):
#     yield dict(inputs), (
#         tf.constant([tags_map("ner_tags", idx, inputs.word_ids(idx - idxs[0]), ner_label2id) for idx in idxs]),
#         tf.constant([tags_map("pos_tags", idx, inputs.word_ids(idx - idxs[0]), pos_label2id) for idx in idxs])
#     )

# trainset = tf.data.Dataset.from_generator(generator, args=(0, 64), output_signature=(
#     {
#         "input_ids": tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
#         "token_type_ids": tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
#         "attention_mask": tf.TensorSpec(shape=(None, 128), dtype=tf.int32)
#     },
#     (
#         tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
#         tf.TensorSpec(shape=(None, 128), dtype=tf.int32)
#     )
# ))

# validset = tf.data.Dataset.from_generator(generator, args=(1, 64), output_signature=(
#     {
#         "input_ids": tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
#         "token_type_ids": tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
#         "attention_mask": tf.TensorSpec(shape=(None, 128), dtype=tf.int32)
#     },
#     (
#         tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
#         tf.TensorSpec(shape=(None, 128), dtype=tf.int32)
#     )
# ))

# testset = tf.data.Dataset.from_generator(generator, args=(2, 64), output_signature={
#     "input_ids": tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
#     "token_type_ids": tf.TensorSpec(shape=(None, 128), dtype=tf.int32),
#     "attention_mask": tf.TensorSpec(shape=(None, 128), dtype=tf.int32)
# })
