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
        description="Parse command line args for training a segmentation model."
    )
    parser.add_argument("-g", "--grid-size", type=float, default=0.2)
    parser.add_argument(
        "-d", "--data-directory", type=Path, default=DATA_PATH / "sensat_urban"
    )
    parser.add_argument("-l", "--leaf-size", type=int, default=50)

    args = parser.parse_args()
    validate_filepath(args.data_directory)

    return args
