"""chb_mit dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for chb_mit dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'wave': tfds.features.Tensor(dtype=float, shape=(7, 3000)),
                'label': tfds.features.ClassLabel(names=['no', 'yes']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('wave', 'label'),  # Set to `None` to disable
            homepage='https://physionet.org/content/chbmit/1.0.0/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        with open("RECORDS") as f:
            records = filter(lambda x: bool(x), f.readlines())
        with open("SEIZURES") as f:
            seizures = filter(lambda x: bool(x), f.readlines())

        for record in records:
            path = dl_manager.download(f'https://physionet.org/files/chbmit/1.0.0/{record}?download')

        path = dl_manager.download_and_extract('')

        # TODO(chb_mit): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train_imgs'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(chb_mit): Yields (key, example) tuples from the dataset
        for f in path.glob('*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes',
            }
