from pathlib import Path
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    """Base class for Point Cloud datasets."""

    def __init__(self, name: str, data_path: Path, labels: Dict[int, str]) -> None:
        """
        Initialize parameters for a dataset.
        """
        self.name = name
        self.data_path = data_path
        self.num_classes = len(labels)
        self.index_to_label = labels
        self.label_to_index = {v: k for k, v in labels.items()}
        self.label_values = list(labels.keys())
        self.label_indices = list(labels.values())

    def __len__(self) -> int:
        """
        Return the length of data.
        """
        return 0

    def __getitem__(self, index) -> Any:
        """
        Return the item at a given index.
        """
        return 0
