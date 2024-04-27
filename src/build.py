import argparse

from models import Brain2Vec, EEGInception
from data import CHBMITGraph2Vec, CHBMITGraph2Seq

parser = argparse.ArgumentParser(description="Build data.")
parser.add_argument("-d", "--name", type=str, default=list(data_classes.keys())[0], choices=list(data_classes.keys()))
