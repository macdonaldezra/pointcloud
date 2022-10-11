import random
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from pointcloud.config import DATA_PATH
from pointcloud.processors.base import PointCloudDataset
from pointcloud.processors.sensat.preprocessing import get_sensat_model_inputs
from pointcloud.utils.files import distribute_indices, get_files
from pointcloud.utils.io import read_ply_file

LABELS = {
    0: "Ground",
    1: "High Vegetation",
    2: "Buildings",
    3: "Walls",
    4: "Bridge",
    5: "Parking",
    6: "Rail",
    7: "Traffic Roads",
    8: "Street Furniture",
    9: "Cars",
    10: "Footpath",
    11: "Bikes",
    12: "Water",
}

TEST_FILES = [
    "birmingham_block_2",
    "birmingham_block_8",
    "cambridge_block_15",
    "cambridge_block_16",
    "cambridge_block_22",
    "cambridge_block_27",
]

VALIDATION_FILES = [
    "birmingham_block_1",
    "birmingham_block_5",
    "cambridge_block_10",
    "cambridge_block_7",
]


class SensatDataSet(PointCloudDataset):
    """
    PyTorch dataset for SensatUrban dataset.
    """

    def __init__(
        self,
        transform: Optional[
            Callable[
                [np.ndarray, np.ndarray, np.ndarray],
                list[np.ndarray, np.ndarray, np.ndarray],
            ]
        ] = None,
        data_partition: str = "train",
        max_points: int = 80000,
        data_path: Path = DATA_PATH / "sensat_urban",
        labels: Dict[int, str] = LABELS,
        shuffle_indices: bool = False,
        include_labels: bool = True,
        distribute_files: bool = True,
        cache_size: int = 10,
    ) -> None:
        """
        Initialize SensatUrban Pointcloud dataset.
        """
        PointCloudDataset.__init__(
            self, name="SensatUrban", data_path=data_path, labels=labels
        )

        self.data_partition = data_partition
        self.max_points = max_points
        self.data_path = data_path
        self.shuffle_indices = shuffle_indices
        self.include_labels = include_labels
        self.transform = transform
        self.distribute_files = distribute_files

        self.num_clouds = 0
        self.input_files = []
        self.file_indices = []
        self.cache_size = cache_size
        self.point_cache = OrderedDict()

        # load files into the input_files list
        self.load_data_files()
        if self.distribute_files:
            print(
                f"Number of iterations required before (somewhat) evenly sampling all files: {len(self.file_indices)}"
            )

        if self.shuffle_indices:
            random.shuffle(self.file_indices)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read in pre-processed voxel-sampled .ply file contents and return randomly sampled
        pointcloud data, feature, and label points.
        """
        if self.distribute_files:
            index = self.file_indices[index % len(self.file_indices)]
        else:
            index = index % len(self.input_files)

        points, features, labels = self.get_pointcloud(self.input_files[index])
        inputs, labels = get_sensat_model_inputs(
            points,
            features,
            labels,
            transform=self.transform,
            shuffle_indices=self.shuffle_indices,
            max_points=self.max_points,
        )

        return inputs, labels

    def __len__(self) -> int:
        if self.distribute_files:
            return len(self.file_indices)
        else:
            return len(self.input_files)

    def load_data_files(self) -> None:
        """
        Load all data into this dataset class.
        """
        if "train" in self.data_partition:
            self.input_files = get_files(
                data_path=self.data_path,
                exclude_files=TEST_FILES + VALIDATION_FILES,
                pattern="*_sample.ply",
            )
        elif "test" in self.data_partition:
            self.input_files = get_files(
                data_path=self.data_path,
                include_files=TEST_FILES,
                pattern="*_sample.ply",
            )
        elif "validation" in self.data_partition:
            self.input_files = get_files(
                data_path=self.data_path,
                include_files=VALIDATION_FILES,
                pattern="*_sample.ply",
            )
        else:
            # Load all files from data path that have pattern *_sample.ply
            self.input_files = list(self.data_path.glob("*_sample.ply"))

        self.file_indices = distribute_indices(self.input_files)

    def get_pointcloud(
        self, filename: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Check if a pointcloud file's contents are in the cache, if not
        read them from a file and store them in the file cache.
        """
        if filename in self.point_cache:
            points, colors, labels = self.point_cache[filename]
            return points, colors, labels

        points, colors, labels = read_ply_file(filename)
        if len(self.point_cache.keys()) == self.cache_size:
            self.point_cache.popitem(last=False)
            self.point_cache[filename] = (points.copy(), colors.copy(), labels.copy())

        return (points, colors, labels)
