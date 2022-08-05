import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Define PLY types
POLY_DTYPES = dict(
    [
        (b"int8", "i1"),
        (b"char", "i1"),
        (b"uint8", "u1"),
        (b"uchar", "u1"),
        (b"int16", "i2"),
        (b"short", "i2"),
        (b"uint16", "u2"),
        (b"ushort", "u2"),
        (b"int32", "i4"),
        (b"int", "i4"),
        (b"uint32", "u4"),
        (b"uint", "u4"),
        (b"float32", "f4"),
        (b"float", "f4"),
        (b"float64", "f8"),
        (b"double", "f8"),
    ]
)

VALID_FILE_FORMATS = {
    "ascii": "",
    "binary_big_endian": ">",
    "binary_little_endian": "<",
}


def parse_header(poly_file, ext) -> Tuple[int, List[str]]:
    """
    Parse .ply file metadata from header.
    """
    line = []
    properties = []
    num_points = None

    while b"end_header" not in line and line != b"":
        line = poly_file.readline()

        if b"element" in line:
            line = line.split()
            num_points = int(line[2])

        elif b"property" in line:
            line = line.split()
            properties.append((line[2].decode(), ext + POLY_DTYPES[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    """
    Determine properties of .ply file based on the files header.
    """
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        # Find point element
        if b"element vertex" in line:
            current_element = "vertex"
            line = line.split()
            num_points = int(line[2])

        elif b"element face" in line:
            current_element = "face"
            line = line.split()
            num_faces = int(line[2])

        elif b"property" in line:
            if current_element == "vertex":
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + POLY_DTYPES[line[1]]))
            elif current_element == "vertex":
                if not line.startswith("property list uchar int"):
                    raise ValueError("Unsupported faces property : " + line)

    return num_points, num_faces, vertex_properties


def parse_ply_file(filepath: Path, triangular_mesh: bool = False) -> np.ndarray:
    """
    Read a .ply file and return an array that contains data stored in the file.
    """
    with open(filepath, "rb") as poly_file:
        # Check if the file start with ply
        if b"ply" not in poly_file.readline():
            raise ValueError("The file does not start with the word 'ply'")

        # get binary_little/big or ascii
        fmt = poly_file.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError("The file is not binary")

        # get extension for building the numpy dtypes
        ext = VALID_FILE_FORMATS[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(poly_file, ext)

            # Get point data
            vertex_data = np.fromfile(poly_file, dtype=properties, count=num_points)

            # Get face data
            face_properties = [
                ("k", ext + "u1"),
                ("v1", ext + "i4"),
                ("v2", ext + "i4"),
                ("v3", ext + "i4"),
            ]
            faces_data = np.fromfile(poly_file, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data["v1"], faces_data["v2"], faces_data["v3"])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(poly_file, ext)

            # Get data
            data = np.fromfile(poly_file, dtype=properties, count=num_points)

    return data


def read_ply_file(filepath: Path, include_labels: bool = True) -> List[np.ndarray]:
    """
    Read a .ply file and return the points, color, and labels.

    Args:
        filepath (Path): The path to the .ply file.
        include_labels (bool): A boolean indicating whether or not to read labels from a file. Note that
            test files for the SensatUrban dataset are not included.

    Returns:
        Tuple[np.ndarray]: A list containing the points, color, and labels for a given pointcloud file.
    """
    data = parse_ply_file(filepath)
    points = np.vstack((data["x"], data["y"], data["z"])).T
    colors = np.vstack((data["red"], data["green"], data["blue"])).T
    if not include_labels:
        return [points.astype(np.float32), colors.astype(np.uint8)]

    return (
        points.astype(np.float32),
        colors.astype(np.uint8),
        data["class"].astype(np.uint8),
    )


def set_header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append("element vertex %d" % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append("property %s %s" % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply_file(
    filepath: Path, points: List[np.ndarray], headers: List[str]
) -> None:
    """
    Write a .ply file with with the provided points and headers.

    Args:
        filepath (Path): A path to the output file.
        points (List[np.ndarray]): A collection of points to write.
        headers (List[str]): headers that correspond to each column of points.
    """
    filename = filepath.as_posix()
    # Format list input to the right form
    field_list = (
        list(points)
        if (type(points) == list or type(points) == tuple)
        else list((points,))
    )
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print("fields have more than 2 dimensions")
            return False

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print("wrong field dimensions")
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(headers):
        print("wrong number of field names")
        return False

    # Add extension if not there
    if not filename.endswith(".ply"):
        filename += ".ply"

    # open in text mode to write the header
    with open(filename, "w") as plyfile:

        # First magical word
        header = ["ply"]

        # Encoding format
        header.append("format binary_" + sys.byteorder + "_endian 1.0")

        # Points properties description
        header.extend(set_header_properties(field_list, headers))

        # End of header
        header.append("end_header")

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, "ab") as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(headers[i], field.dtype.str)]
                i += 1

        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[headers[i]] = field
                i += 1

        data.tofile(plyfile)

    return True
