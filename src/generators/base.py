import os
import itertools
import functools

from torch import save


def build(Generator):
    keys, values = zip(*Generator.hparams.items())

    for kwargs in [dict(zip(keys, v)) for v in itertools.product(*values)]:
        generator = Generator(**kwargs)
        root = os.path.expanduser(f"~/pytorch_datasets/{generator.name}")

        hparams = functools.reduce(lambda x, y: f"{x}_{y[0]}_{y[1]}", kwargs.items(), "")
        os.makedirs(f"{root}{hparams}", exist_ok=True)

        for key, records in generator(f"{root}_raw").items():
            path = f"{root}{hparams}/{key}"
            os.makedirs(path, exist_ok=True)

            for idx, item in enumerate(records):
                save(item, f"{path}/{idx}.pt")
