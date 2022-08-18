import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from pointcloud.config import DATA_PATH
from pointcloud.processors.base import PointCloudDataset
from pointcloud.processors.sensat.preprocessing import get_sensat_model_inputs
from pointcloud.utils.files import get_files

from ...utils.io import read_ply_file

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


class SensatDataLoader(PointCloudDataset):
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
        max_points: int = 10000,
        data_path: Path = DATA_PATH / "sensat_urban",
        labels: Dict[int, str] = LABELS,
        shuffle_indices: bool = False,
        include_labels: bool = True,
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

        self.num_clouds = 0
        self.input_files = []

        # load files into the input_files list
        self.load_data_files()

        # Commented out code at this point is making use of another point selection
        # method that can be used for collecting a sample of pointcloud points.
        #
        # self.potentials = []
        # self.min_potentials = []
        # self.argmin_potentials = []
        # for tree in self.pot_trees:
        #     self.potentials.append(
        #         torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)
        #     )
        #     min_index = int(torch.argmin(self.potentials[-1]))
        #     self.argmin_potentials.append(min_index)
        #     self.min_potentials.append(float(self.potentials[-1][min_index]))

        # self.argmin_potentials = torch.from_numpy(
        #     np.array(self.argmin_potentials, dtype=np.int64)
        # )
        # self.min_potentials = torch.from_numpy(
        #     np.array(self.min_potentials, dtype=np.float64)
        # )
        # self.epoch_index = 0
        # self.epoch_indices = None

    def __getitem__(self, index) -> Any:
        """
        Read in pre-processed voxel-sampled file from a pointcloud file in pointcloud dataset
        and get a set of points within the same given area from that pointcloud
        """
        points, features, labels = read_ply_file(
            self.input_files[index % len(self.input_files)]
        )
        points, features, labels = get_sensat_model_inputs(points, features, labels)

        return points, features, labels

    def load_data_files(self) -> None:
        """
        Load all data into this dataset class.

        TODO: Sensat .ply files have very different sizes. We need to seriously think
        about at least splitting up larger files into smaller files in order to ensure that
        the datasets are appropriately sampled during training and validation.
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

    # def get_potential_item(self, batch_index: int):

    #     batch_size = 0

    #     all_points = []
    #     all_features = []
    #     all_labels = []
    #     all_input_indices = []
    #     point_indices = []
    #     cloud_indices = []

    #     while batch_size < self.batch_limit:

    #         cloud_index = int(torch.argmin(self.min_potentials))
    #         point_index = int(self.argmin_potentials[cloud_index])

    #         # Get potential points from tree structure
    #         potential_points = np.array(self.pot_trees[point_index, :].data, copy=False)
    #         center_point = potential_points[point_index, :].reshape(1, -1)

    #         # Get indices of points in the input region
    #         # TODO: Make radius of the input sphere (value for input_radius) configurable.
    #         input_radius = 1.0
    #         potential_indices, distances = self.pot_trees[cloud_index].query_radius(
    #             center_point, r=input_radius, return_distance=True
    #         )

    #         dist_squared = np.square(distances[0])
    #         potential_indices = potential_indices[0]

    #         tukey_loss = np.square(1 - dist_squared / np.square(input_radius))
    #         tukey_loss[dist_squared > np.square(input_radius)] = 0
    #         self.potentials[cloud_index][potential_indices] += tukey_loss

    #         min_index = torch.argmin(self.potentials[cloud_index])
    #         self.min_potentials[[cloud_index]] = self.potentials[cloud_index][min_index]
    #         self.argmin_potentials[[cloud_index]] = min_index

    #         points = np.array(self.input_trees[cloud_index].data, copy=False)
    #         input_indices = self.input_trees[cloud_index].query_radius(
    #             center_point, r=input_radius
    #         )[0]
    #         n = input_indices.shape[0]

    #         if n < 2:
    #             # don't add this pointcloud to the batch as it represents an empty sphere
    #             continue

    #         input_points = (points[input_indices] - center_point).astype(np.float32)
    #         input_colors = self.input_colors[cloud_index][input_indices]
    #         if self.training:
    #             input_labels = self.input_labels[cloud_index][input_indices]
    #             input_labels = np.array([self.label_to_index[l] for l in input_labels])
    #         else:
    #             input_labels = np.zeros(input_points.shape[0])

    #         input_features = np.hstack(
    #             (input_colors, input_points[:, 2:] + center_point[:, 2:])
    #         ).astype(np.float32)

    #         # Add this iterations points to lists
    #         all_points.append(input_points)
    #         all_features.append(input_features)
    #         all_labels.append(input_labels)
    #         all_input_indices.append(input_indices)
    #         point_indices.append(point_index)
    #         cloud_indices.append(cloud_index)

    #         batch_size += n

    #     # Concatenate each component of the batch
    #     stacked_points = np.concatenate(all_points, axis=0)
    #     features = np.concatenate(all_features, axis=0)
    #     labels = np.concatenate(all_labels, axis=0)
    #     input_inds = np.concatenate(all_input_indices, axis=0)
    #     stack_lengths = np.array([p.shape[0] for p in all_points], dtype=np.int32)

    #     stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
    #     np.hstack((stacked_features, features))

    #     # Get segmentation inputs
    #     # input_list += [cloud_indices, point_indices, input_indices]

    #     return [cloud]
