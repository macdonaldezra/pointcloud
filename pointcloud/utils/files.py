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
