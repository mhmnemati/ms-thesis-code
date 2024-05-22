import os
import functools

from torch import save


def build(Generator):
    for kwargs in Generator.hparams:
        generator = Generator(**kwargs)
        root = os.path.expanduser(f"~/pytorch_datasets/{generator.name}")

        hparams = functools.reduce(lambda x, y: f"{x}_{y[0]}_{y[1]}", kwargs.items(), "")
        os.makedirs(f"{root}{hparams}", exist_ok=True)

        for key, records in generator(f"{root}_raw").items():
            path = f"{root}{hparams}/{key}"
            os.makedirs(path, exist_ok=True)

            counters = {}
            for subpath, item in records:
                if subpath not in counters:
                    os.makedirs(f"{path}/{subpath}", exist_ok=True)
                    counters[subpath] = 0

                counters[subpath] += 1

                save(item, f"{path}/{subpath}/{counters[subpath]}.pt")
