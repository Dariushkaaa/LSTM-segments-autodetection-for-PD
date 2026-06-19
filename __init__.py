"""IMU Gait Segmentation Package."""
__version__ = "1.0.0"

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .model import AttentionLSTM
from .dataset import SegDataset
from .config import CFG

__all__ = ["AttentionLSTM", "SegDataset", "CFG", "DEVICE"]
