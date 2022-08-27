import typing as T

import numpy as np
import open3d as o3
import torch
from PIL import Image
from pointcloud.config import DATA_PATH


class ShapeNetDataSet(torch.utils.data.Dataset):
    """
    Preprocess ShapeNet dataset
    """

    def __init__(
        self,
        n_points: int = 2500,
        classification: bool = False,
        class_choice: T.Optional[T.List[str]] = None,
        train: bool = True,
        image: bool = False,
        split: float = 0.9,
    ) -> None:
        self.n_points = n_points
        self.data_path = DATA_PATH / "shapenetcore_partanno_segmentation_benchmark_v0"
        self.categories = self.read_categories(
            self.data_path / "synsetoffset2category.txt"
        )
        self.classification = classification
        self.train = train
        self.image = image
        self.metadata = {}
        self.filepaths = []
        self.labels = {}
        self.num_segmentation_classes = 0

        if class_choice:
            self.categories = {
                k: v for k, v in self.categories.items() if k in class_choice
            }

        # For each item in a category, assign the point, segmentation and image path.
        for item in self.categories:
            self.metadata[item] = []
            point_path = self.data_path / str(self.categories[item]) / "points"
            label_path = self.data_path / str(self.categories[item]) / "points_label"
            image_path = self.data_path / str(self.categories[item]) / "seg_img"

            files = sorted(point_path.glob("*"))
            if train:
                files = files[: int(len(files) * split)]
            else:
                files = files[int(len(files) * split) :]

            for file in files:
                self.metadata[item].append(
                    (
                        point_path / (file.stem + ".pts"),
                        label_path / (file.stem + ".seg"),
                        image_path / (file.stem + ".png"),
                    )
                )

        # Create a variable where you have (item, points, segmentation points, and segmentation image)
        for item in self.categories:
            for file in self.metadata[item]:
                self.filepaths.append((item, file[0], file[1], file[2]))

        self.labels = dict(zip(sorted(self.categories), range(len(self.categories))))

        if not self.classification:
            for i in range(len(self.filepaths) // 50):
                self.num_segmentation_classes = max(
                    self.num_segmentation_classes,
                    len(np.unique(np.loadtxt(self.filepaths[i][-2]).astype(np.uint8))),
                )

    def __getitem__(self, index: int):
        """
        Pick a specific element from the dataset.
        """
        instance = self.filepaths[index]
        class_label = self.labels[instance[0]]

        # Read in the Point Cloud to a numpy array
        point_set = np.asarray(
            o3.io.read_point_cloud(instance[1].as_posix(), format="xyz").points,
            dtype=np.float32,
        )
        # Read in Segmentation data
        segmentation = np.loadtxt(instance[2]).astype(np.int64)

        image = Image.open(instance[3])
        choice = np.random.choice(len(segmentation), self.n_points, replace=True)

        # resample
        point_set = point_set[choice, :]
        segmentation = segmentation[choice]
        point_set = torch.from_numpy(point_set)
        segmentation = torch.from_numpy(segmentation)
        class_label = torch.from_numpy(np.array([class_label]).astype(np.int64))

        if self.classification:
            if self.image:
                return point_set, class_label, image
            else:
                return point_set, class_label
        else:
            if self.image:
                return point_set, segmentation, image
            else:
                return point_set, segmentation

    def __len__(self):
        return len(self.filepaths)

    @staticmethod
    def read_categories(filepath: str) -> T.Dict[str, int]:
        categories = {}
        with open(filepath, "r") as infile:
            for line in infile:
                pair = line.strip().split()
                categories[pair[0]] = pair[1]

        return categories
