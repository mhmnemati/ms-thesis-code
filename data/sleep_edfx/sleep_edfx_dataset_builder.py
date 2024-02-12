"""sleep_edfx dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sleep_edfx dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'wave': tfds.features.Tensor(shape=(7, 3000), dtype=tf.float16),
                'label': tfds.features.ClassLabel(names=['no', 'yes']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('wave', 'label'),  # Set to `None` to disable
            homepage='https://www.physionet.org/content/sleep-edfx/1.0.0/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract('https://www.physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip')

        return {
            'train': self._generate_examples(path / 'train_imgs'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(sleep_edfx): Yields (key, example) tuples from the dataset
        for f in path.glob('*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes',
            }
