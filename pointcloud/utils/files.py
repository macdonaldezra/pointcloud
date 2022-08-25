from pathlib import Path
from typing import List, Optional


def get_files(
    data_path: Path,
    exclude_files: Optional[List[str]] = None,
    include_files: Optional[List[str]] = None,
    pattern: str = "*.ply",
) -> List[Path]:
    """
    Return a list of files either excluding a set of files, including only a set of files, or returning
    all files with a given filename pattern.
    """
    all_files = list(data_path.glob(pattern))

    if include_files:
        file_list = []
        for include in include_files:
            for file in all_files:
                if include in file.name:
                    file_list.append(file)

        return file_list

    if exclude_files:
        for exclude in exclude_files:
            for file in all_files:
                if exclude in file.name:
                    all_files.remove(file)

    return all_files


def distribute_indices(files: List[Path]) -> List[int]:
    """
    Given a list of files, select the minimum sized file, and return
    a list that contains indices for other files that occurs with the
    same frequency.
    """
    file_sizes = [file.stat().st_size for file in files]

    min_size = min(file_sizes)
    indices = []
    for index, size in enumerate(file_sizes):
        count = round(size / min_size)
        for i in range(count):
            indices.append(index)

    return indices
