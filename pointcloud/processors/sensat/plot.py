import numpy as np
import open3d as o3

LABEL_COLORS = [
    [85, 107, 47],  # ground -> OliveDrab
    [0, 255, 0],  # tree -> Green
    [255, 165, 0],  # building -> orange
    [41, 49, 101],  # Walls ->  darkblue
    [0, 0, 0],  # Bridge -> black
    [0, 0, 255],  # parking -> blue
    [255, 0, 255],  # rail -> Magenta
    [200, 200, 200],  # traffic Roads ->  grey
    [89, 47, 95],  # Street Furniture  ->  DimGray
    [255, 0, 0],  # cars -> red
    [255, 255, 0],  # Footpath  ->  deeppink
    [0, 255, 255],  # bikes -> cyan
    [0, 191, 255],  # water ->  skyblue
]


def draw_pointcloud(points: np.ndarray, point_colors: np.ndarray) -> None:
    """
    Draw a PointCloud given a set of points and colors for each set of points.

    Args:
        points (np.ndarray): A nx3 dimensional array of pointcloud points.
        point_colors (np.ndarray): A nx3 dimensional array containing RGB color coordinates.
    """
    pointcloud = o3.geometry.PointCloud()
    pointcloud.points = o3.utility.Vector3dVector(points)
    # Scale pointcloud colors if they haven't been
    if np.max(point_colors) > 1:
        pointcloud.colors = o3.utility.Vector3dVector(point_colors / 255.0)
    else:
        pointcloud.colors = o3.utility.Vector3dVector(point_colors)

    o3.geometry.PointCloud.estimate_normals(pointcloud)
    o3.visualization.draw_geometries([pointcloud], width=1000, height=1000)


def draw_segmented_pointcloud(
    points: np.ndarray, point_labels: np.ndarray
) -> np.ndarray:
    """
    Given a set of pointcloud points and a set of point labels, color the labelled pointcloud
    points with a corresponding color from the list of label colors.

    Args:
        points (np.ndarray): A nx3 dimensional array of pointcloud points.
        point_colors (np.ndarray): A nx3 dimensional array containing RGB color coordinates.

    Returns:
        np.ndarray: An ndarray with the points and corresponding label colorings, with the points
            set in the first 3 columns and the colors in the latter 3 columns.
    """

    found_labels = np.unique(point_labels)
    label_colors = np.zeros((point_labels.shape[0], 3))
    for label in found_labels:
        label_indices = np.argwhere(point_labels == label)[:, 0]
        if label <= -1:
            colors = [0, 0, 0]
        else:
            colors = LABEL_COLORS[label]

        label_colors[label_indices] = colors

    draw_pointcloud(points, label_colors)

    return np.concatenate([points, label_colors], axis=-1)


def draw_voxelgrid(
    points: np.ndarray, point_colors: np.ndarray, voxel_size: float = 0.1
) -> None:
    """
    Given a set of points and point colorings, render a voxel grid with the provided
    voxel size.

    Args:
        points (np.ndarray): A nx3 dimensional array of pointcloud points.
        point_colors (np.ndarray): A nx3 dimensional array containing RGB color coordinates.
        voxel_size (float, optional): A float that indicates the volume for a given voxel size. Defaults to 0.1.
    """
    pointcloud = o3.geometry.PointCloud()
    pointcloud.points = o3.utility.Vector3dVector(points)
    pointcloud.colors = o3.utility.Vector3dVector(point_colors)
    o3.geometry.PointCloud.estimate_normals(pointcloud)

    voxel_grid = o3.geometry.VoxelGrid.create_from_point_cloud(
        pointcloud, voxel_size=0.1
    )
    o3.visualization.draw_geometries([voxel_grid])
