import mne
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("webagg")

data = "/root/pytorch_datasets/sleep_edfx_raw/sleep-cassette/SC4002E0-PSG.edf"
annots = "/root/pytorch_datasets/sleep_edfx_raw/sleep-cassette/SC4002EC-Hypnogram.edf"

raw = mne.io.read_raw_edf(data, include=["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"])
annots = mne.read_annotations(annots)
raw.set_annotations(annots)


rg = 5000
begin = int(26070 + 50)
raw.crop(begin, begin + rg)
raw.plot(duration=500)

input()
