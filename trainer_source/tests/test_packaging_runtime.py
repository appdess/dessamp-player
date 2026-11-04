import importlib.resources

import torch

import nam
from nam.models.base import BaseNet


def test_package_import_and_loudness_resource_are_available():
    assert nam.__version__

    resource = importlib.resources.files("nam.models._resources").joinpath(
        "loudness_input.wav"
    )
    assert resource.is_file()

    signal = BaseNet._metadata_loudness_x()
    assert signal.shape == (48_000,)
    assert torch.isfinite(signal).all()
