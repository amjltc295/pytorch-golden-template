from torchvision import datasets


def load_wave_file(filepath):
    return []


class SC09Dataset(datasets.DatasetFolder):
    def __init__(self, data_dir, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        super().__init__(
            data_dir, load_wave_file,
            extensions=extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )

    def __getitem__(self, index):
        data = super().__getitem__(index)
        return {
            "data_input": data[0],
            "data_target": data[1]
        }
