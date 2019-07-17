from torchvision import datasets
from librosa.core import load
import numpy as np

from utils.audio import trim_or_pad


def load_wave_file(filepath, sample_rate=16000, max_length=16000):
    aud, sr = load(filepath, sr=sample_rate)
    return trim_or_pad(aud, max_length)


class SC09Dataset(datasets.DatasetFolder):
    def __init__(self, data_dir, extensions=['.wav'], transform=None, target_transform=None):
        super().__init__(
            data_dir, load_wave_file,
            extensions=extensions,
            transform=transform,
            target_transform=target_transform
        )

    def __getitem__(self, index):
        data = super().__getitem__(index)
        return {
            "data_input": data[0].astype(np.float32),
            "data_target": data[1]
        }
