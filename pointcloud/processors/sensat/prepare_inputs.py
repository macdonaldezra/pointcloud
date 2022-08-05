import pickle
import time
from pathlib import Path

import numpy as np
from pointcloud.utils.io import read_ply_file, write_ply_file
from pointcloud.utils.subsampling import grid_subsampling
from sklearn.neighbors import KDTree


def prepare_sensat_pointcloud_data(
    data_path: Path, grid_size: float = 0.1, leaf_size: int = 50
) -> None:
    """
    Read in pointcloud training and testing pointcloud files and perform the following:
        1. Sub-sample each pointcloud.
        2. Normalize colors for each file.
        3. Create and save a KDTree for each sub-sampled pointcloud.
        4. Write out the sub-sampled pointcloud to a file in the output directory.

    Args:
        data_path (Path): A path to pointcloud data files with th
        grid_size (float, optional): The size of the voxel grid to be used in grid sub-sampling task. Defaults to 0.1.
        leaf_size (int, optional): The size of the leafs used to store each subset of pointclouds in the KDTree. Defaults to 50.

    Raises:
        ValueError: When the output directory already exists, ie. we have already computed a grid subsampling for that dataset; or
            when the data directory doesn't contain test or train subdirectories.
    """
    output_path = data_path / f"grid_{grid_size}"
    if output_path.exists():
        raise ValueError(
            f"Directory with voxel grid of size: {grid_size} already exists. Either delete that directory or use those files."
        )

    output_path.mkdir()
    test_path = data_path / "test"
    train_path = data_path / "train"
    if not test_path.exists() or not train_path.exists():
        raise ValueError(
            f"A train and test directory was not found in {data_path.as_posix()}."
        )

    print("\nPreparing ply files\n")
    t0 = time.time()
    test_files = list(test_path.glob("*.ply"))
    train_files = list(train_path.glob("*.ply"))
    all_files = train_files + test_files

    for filepath in all_files:
        print(f"Starting to process {filepath.stem}")
        if filepath in test_files:
            points, colors = read_ply_file(filepath, include_labels=False)
            sub_points, sub_colors = grid_subsampling(
                points, features=colors, sampleDl=grid_size
            )
            sub_labels = np.zeros(len(sub_points), dtype=np.uint8)
        else:
            points, colors, labels = read_ply_file(filepath)
            sub_points, sub_colors, sub_labels = grid_subsampling(
                points, features=colors, labels=labels, sampleDl=grid_size
            )

            # If there are more than one column for sub_labels, then make it so there's only one.
            if sub_labels.shape[1] > 1:
                sub_labels = np.squeeze(sub_labels)

        # Normalize colors if they haven't been normalized between 0 and 1 already.
        if np.max(sub_colors[:, 0]) > 1:
            sub_colors = sub_colors / 255

        search_tree = KDTree(sub_points, leaf_size=leaf_size)
        with open(output_path / f"{filepath.stem}_KDTree.pkl", "wb") as outfile:
            pickle.dump(search_tree, outfile)

        projected_indices = np.squeeze(
            search_tree.query(sub_points, return_distance=False)
        )
        with open(output_path / f"{filepath.stem}_proj.pkl", "wb") as outfile:
            pickle.dump([projected_indices, sub_labels], outfile)

        write_ply_file(
            output_path / f"{filepath.stem}_sample.ply",
            [sub_points, sub_colors, sub_labels],
            ["x", "y", "z", "red", "green", "blue", "class"],
        )

    print("Done in {:.1f}s".format(time.time() - t0))