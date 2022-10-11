import typing as T

import numpy as np
from pointcloud.processors.sensat.preprocessing import sample_points


def test_point_sample() -> None:
    """
    Test that the point sampling function is capable of ensuring that there are
    always the specified number of points returned from the function.
    """
    points = np.full((100, 3), 1)
    colors = points.copy()
    labels = np.full((100, 1), -1)
    max_points = 340

    points, colors, labels = sample_points(points, colors, labels, max_points)
    assert points.shape[0] == max_points
    assert labels.shape[0] == max_points
    assert len(points.shape) == 2

    points = np.full((400, 3), 1)
    colors = points.copy()
    labels = np.full((400, 1), -1)

    points, colors, labels = sample_points(points, colors, labels, max_points)
    assert points.shape[0] == max_points
    assert labels.shape[0] == max_points
    assert len(points.shape) == 2
