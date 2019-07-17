import numpy as np


def trim_or_pad(aud: np.ndarray, max_length: int):
    trimmed_or_padded_audio = np.zeros((1, max_length))
    trimmed_or_padded_audio[:, :len(aud)] = aud
    return trimmed_or_padded_audio.astype(np.float32)
