import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from pointcloud.config import DATA_PATH
from pointcloud.processors.base import PointCloudDataset
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

# TODO: Create a list of files for validation files so we can split out train files
# more explicitly.


class SensatDataset(PointCloudDataset):
    """
    PyTorch dataset for SensatUrban dataset.
    """

    def __init__(
        self,
        train: bool = False,
        batch_limit: int = 10000,
        data_path: Path = DATA_PATH / "sensat_urban",
        labels: Dict[int, str] = LABELS,
    ) -> None:
        """
        Initialize SensatUrban Pointcloud dataset.
        """
        PointCloudDataset.__init__(
            self, name="SensatUrban", data_path=data_path, labels=labels
        )

        self.training = train
        self.batch_limit = batch_limit
        self.data_path = data_path

        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_projection = []
        self.validation_labels = []

        # Load data into input_trees, pot_trees, input_labels, and input_colors

        # Use potentials
        self.potentials = []
        self.min_potentials = []
        self.argmin_potentials = []
        for tree in self.pot_trees:
            self.potentials.append(
                torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)
            )
            min_index = int(torch.argmin(self.potentials[-1]))
            self.argmin_potentials.append(min_index)
            self.min_potentials.append(float(self.potentials[-1][min_index]))

        self.argmin_potentials = torch.from_numpy(
            np.array(self.argmin_potentials, dtype=np.int64)
        )
        self.min_potentials = torch.from_numpy(
            np.array(self.min_potentials, dtype=np.float64)
        )
        self.epoch_index = 0
        self.epoch_indices = None

    def __getitem__(self, index) -> Any:
        return self.get_potential_item(index)

    def load_all_data(self) -> None:
        """
        Load all data into this dataset class.
        """
        if self.training:
            tree_files = get_files(exclude_files=TEST_FILES, pattern="*_KDTree.pkl")
            point_files = get_files(exclude_files=TEST_FILES)
        else:
            tree_files = get_files(exclude_files=TEST_FILES, pattern="*_KDTree.pkl")
            point_files = get_files(exclude_files=TEST_FILES)

        for file in point_files:
            # Load data from sub-sampled .ply files
            _, colors, labels = read_ply_file(file)
            self.input_colors.append(colors)
            self.input_labels.append(labels)

        for file in tree_files:
            # Load pointclouds already loaded into KDTree files
            with open(file, "rb") as tree_file:
                tree = pickle.load(tree_file)
                self.input_trees.append(tree)

    def get_potential_item(self, batch_index: int):

        batch_size = 0

        all_points = []
        all_features = []
        all_labels = []
        all_input_indices = []
        point_indices = []
        cloud_indices = []

        while batch_size < self.batch_limit:

            cloud_index = int(torch.argmin(self.min_potentials))
            point_index = int(self.argmin_potentials[cloud_index])

            # Get potential points from tree structure
            potential_points = np.array(self.pot_trees[point_index, :].data, copy=False)
            center_point = potential_points[point_index, :].reshape(1, -1)

            # Get indices of points in the input region
            # TODO: Make radius of the input sphere (value for input_radius) configurable.
            input_radius = 1.0
            potential_indices, distances = self.pot_trees[cloud_index].query_radius(
                center_point, r=input_radius, return_distance=True
            )

            dist_squared = np.square(distances[0])
            potential_indices = potential_indices[0]

            tukey_loss = np.square(1 - dist_squared / np.square(input_radius))
            tukey_loss[dist_squared > np.square(input_radius)] = 0
            self.potentials[cloud_index][potential_indices] += tukey_loss

            min_index = torch.argmin(self.potentials[cloud_index])
            self.min_potentials[[cloud_index]] = self.potentials[cloud_index][min_index]
            self.argmin_potentials[[cloud_index]] = min_index

            points = np.array(self.input_trees[cloud_index].data, copy=False)
            input_indices = self.input_trees[cloud_index].query_radius(
                center_point, r=input_radius
            )[0]
            n = input_indices.shape[0]

            if n < 2:
                # don't add this pointcloud to the batch as it represents an empty sphere
                continue

            input_points = (points[input_indices] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_index][input_indices]
            if self.training:
                input_labels = self.input_labels[cloud_index][input_indices]
                input_labels = np.array([self.label_to_index[l] for l in input_labels])
            else:
                input_labels = np.zeros(input_points.shape[0])

            input_features = np.hstack(
                (input_colors, input_points[:, 2:] + center_point[:, 2:])
            ).astype(np.float32)

            # Add this iterations points to lists
            all_points.append(input_points)
            all_features.append(input_features)
            all_labels.append(input_labels)
            all_input_indices.append(input_indices)
            point_indices.append(point_index)
            cloud_indices.append(cloud_index)

            batch_size += n

        # Concatenate each component of the batch
        stacked_points = np.concatenate(all_points, axis=0)
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        input_inds = np.concatenate(all_input_indices, axis=0)
        stack_lengths = np.array([p.shape[0] for p in all_points], dtype=np.int32)

        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        np.hstack((stacked_features, features))

        # Get segmentation inputs
        # input_list += [cloud_indices, point_indices, input_indices]

        return [cloud]
