import glob
import tensorflow as tf


def write(path, iterator):
    for i, (X, Y) in enumerate(iterator):
        with tf.io.TFRecordWriter(f"{path}/data_{i}.tfrecords") as writer:
            for index in range(len(Y)):
                feature = {
                    key: tf.train.Feature(float_list=tf.train.FloatList(value=val[index]))
                    for key, val in X.items()
                }
                feature["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[Y[index]]))

                writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())


def read(path, columns):
    schema = {
        column: tf.io.VarLenFeature(tf.float32) for column in columns
    }
    schema["label"] = tf.io.FixedLenFeature([1], tf.int64)

    return tf.data.TFRecordDataset(
        filenames=glob.glob(f"{path}/*.tfrecords"),
        num_parallel_reads=4
    ).map(lambda x: tf.io.parse_single_example(x, schema))
