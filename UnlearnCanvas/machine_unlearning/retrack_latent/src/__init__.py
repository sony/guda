"""
ReTrack-style Unlearning for UnlearnCanvas

This package implements ReTrack-style machine unlearning for Stable Diffusion
on the UnlearnCanvas dataset with three core functionalities:

1. All-class learning (train_allclass.py)
2. LOGO learning (train_logo.py)
3. ReTrack unlearning (train_retrack.py)

Plus evaluation and calibration utilities.
"""

__version__ = "0.1.0"
__author__ = "ReTrack-UnlearnCanvas Team"

from . import utils
from . import dataset_pairing

__all__ = [
    "utils",
    "dataset_pairing",
]
