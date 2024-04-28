import os
import argparse
import itertools
import functools

from torch import save

from generators import CHBMIT, SleepEDFX

generator_classes = {
    "chb_mit": CHBMIT,
    "sleep_edfx": SleepEDFX,
}

parser = argparse.ArgumentParser(description="Build generator.")
parser.add_argument("-g", "--generator", type=str, default=list(generator_classes.keys())[0], choices=list(generator_classes.keys()))

args = parser.parse_args()

Generator = generator_classes[args.generator]
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
