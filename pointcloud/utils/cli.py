import argparse
from pathlib import Path
from typing import NamedTuple

from pointcloud.config import DATA_PATH


def validate_filepath(path: Path) -> None:
    if not isinstance(path, Path):
        raise TypeError(f"{path} is not a valid filepath.")
    if not path.exists():
        raise FileNotFoundError(f"Path {path.as_posix()} does not exist.")


def parse_input_preprocessing_args() -> NamedTuple:
    """
    Parse, validate and return arguments provided on the command line that are
    designed to be fed as input into a function that pre-processes pointcloud data
    for model input.

    Returns:
        NamedTuple: Args to be used as input for pre-processing pointcloud data.
    """
    parser = argparse.ArgumentParser(
        description="Parse command line args for pre-processing pointcloud data files."
    )
    parser.add_argument(
        "-g",
        "--grid-size",
        type=float,
        default=0.2,
        help="The size of the voxel grid to be used in outputted sub-sampled pointclouds.",
    )
    parser.add_argument(
        "-d",
        "--data-directory",
        type=Path,
        default=DATA_PATH / "sensat_urban",
        help="The data directory to read .ply files from.",
    )
    parser.add_argument(
        "-l",
        "--leaf-size",
        type=int,
        default=50,
        help="Integer to indicate the size of each KD-Tree leaf.",
    )
    parser.add_argument(
        "-k",
        "--kd-tree",
        type=bool,
        default=False,
        help="Boolean to indicate whether or not to generate KD-Tree for pointcloud points.",
    )
    parser.add_argument(
        "-p",
        "--proj",
        type=bool,
        default=False,
        help="Boolean to indicate whether or not to generate projection for pointcloud points.",
    )

    args = parser.parse_args()
    validate_filepath(args.data_directory)

    return args


def parse_train_args() -> NamedTuple:
    """
    Parse, validate and return args for training a model.
    """
    parser = argparse.ArgumentParser(
        description="Parse command line args for training a segmentation model."
    )
    parser.add_argument(
        "-d",
        "--data-directory",
        type=Path,
        default=DATA_PATH / "sensat_urban" / "grid_0.2",
        help="The data directory to read sampled .ply files from.",
    )

    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        default=DATA_PATH / "output",
        help="The directory to output files to",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="The size of training batches",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model for",
    )

    args = parser.parse_args()
    validate_filepath(args.data_directory)
    if isinstance(args.output_directory, Path):
        args.output_directory.mkdir(exist_ok=True, parents=True)

    validate_filepath(args.output_directory)

    return args
