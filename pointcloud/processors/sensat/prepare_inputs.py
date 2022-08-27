import pickle
import time
from pathlib import Path

import numpy as np
from pointcloud.utils.cli import parse_input_preprocessing_args
from pointcloud.utils.io import read_ply_file, write_ply_file
from pointcloud.utils.subsampling import grid_subsampling
from sklearn.neighbors import KDTree


def prepare_sensat_pointcloud_data(
    data_path: Path,
    grid_size: float,
    leaf_size: int,
    create_kd_tree: bool,
    create_proj: bool,
) -> None:
    """
    Read in pointcloud training and testing pointcloud files and perform the following:
        1. Sub-sample each pointcloud.
        2. Normalize colors for each file.
        3. Create and save a KDTree for each sub-sampled pointcloud.
        4. Write out the sub-sampled pointcloud to a file in the output directory.

    Args:
        data_path (Path): A path to pointcloud data files with th
        grid_size (float): The size of the voxel grid to be used in grid sub-sampling task. Defaults to 0.2.
        leaf_size (int): The size of the leafs used to store each subset of pointclouds in the KDTree. Defaults to 50.
        create_kd_tree (bool): Indicates whether or not to output a KDTree for PointClouds.
        create_proj (bool): Indicates whether or not to output a projection of PointCloud points.

    Raises:
        ValueError: When the data directory doesn't contain test or train subdirectories.
    """
    output_path = data_path / f"grid_{grid_size}"
    if output_path.exists():
        print(
            f"Directory with voxel grid of size: {grid_size} already exists. Only preparing ply files "
            + "for files that do not exist in that directory."
        )

    output_path.mkdir(exist_ok=True)
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
        if (output_path / f"{filepath.stem}_sample.ply").exists():
            print(
                f"Skipping processing {filepath.stem} as a downsampled version of this file already exists."
            )
            continue

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

            # If there are more than one column for sub_ labels, then make it so there's only one.
            if sub_labels.shape[1] > 1:
                sub_labels = np.squeeze(sub_labels)

        if create_kd_tree:
            search_tree = KDTree(sub_points, leaf_size=leaf_size)
            with open(output_path / f"{filepath.stem}_KDTree.pkl", "wb") as outfile:
                pickle.dump(search_tree, outfile)

        if create_proj:
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


if __name__ == "__main__":
    args = parse_input_preprocessing_args()
    prepare_sensat_pointcloud_data(
        args.data_directory, args.grid_size, args.leaf_size, args.kd_tree, args.proj
    )
