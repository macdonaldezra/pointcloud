from typing import Callable, Optional, Tuple

import numpy as np
import torch


def sample_points(
    points: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    num_points: int,
    shuffle_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample points until we reach num_points number of points in the set of indices.

    If the number of sampled indices is not enough to reach num_points, then continue choosing
    a random point from the pointcloud, and appending the neighbouring set of points to the current
    set of pointcloud indices. This sampling technique will always return the specified
    number of points.

    Note: This function doesn't work when being called by the below function. :-(
    """
    start_index = np.random.randint(labels.shape[0])
    sorted_indices = np.argsort(np.sum(np.square(points - points[start_index]), 1))
    if sorted_indices.shape[0] < num_points:
        indices = sorted_indices
        while indices.shape[0] < num_points:
            start_index = np.random.randint(labels.shape[0])
            sorted_indices = np.argsort(
                np.sum(np.square(points - points[start_index]), 1)
            )
            if sorted_indices.shape[0] + indices.shape[0] > num_points:
                last_index = num_points - indices.shape[0]
                indices = np.concatenate([indices, sorted_indices[:last_index]])
            else:
                indices = np.concatenate([indices, sorted_indices])
    else:
        indices = sorted_indices[:num_points]

    if shuffle_indices:
        indices = np.arange(labels.shape[0])
        np.random.shuffle(indices)
        points, features, labels = points[indices], features[indices], labels[indices]

    points[:] = points[indices]
    features[:] = features[indices]
    labels[:] = labels[indices]

    return (points, features, labels)


def get_sensat_model_inputs(
    points: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    transform: Optional[
        Callable[
            [np.ndarray, np.ndarray, np.ndarray],
            list[np.ndarray, np.ndarray, np.ndarray],
        ]
    ] = None,
    shuffle_indices: bool = False,
    max_points: int = 80000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return a set of sampled pointcloud points, features, and labels as PyTorch tensors.

    Args:
        points (np.ndarray): An nx3 numpy array containing pointcloud coordinates.
        features (np.ndarray): An nxd array containing pointcloud coordinate features.
            Please Note: features should already be normalized between 0 and 1 at this point.
        labels (np.ndarray): An nxd array containing pointcloud coordinate labels.
        transform: A callable function that is applied to points, features, and labels
        training: A boolean indicating whether or not a model is being trained with these inputs.
        shuffle_indices: A boolean indicating whether or not pointcloud indices should be shuffled
            for a given set of points.
        max_points: The maximum number of points to be sample for a given model. Default is 80,000

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing points and features concatenated, and
            labels.
    """
    start_index = np.random.randint(labels.shape[0])
    sorted_indices = np.argsort(np.sum(np.square(points - points[start_index]), 1))
    if sorted_indices.shape[0] < max_points:
        indices = sorted_indices
        while indices.shape[0] < max_points:
            start_index = np.random.randint(labels.shape[0])
            sorted_indices = np.argsort(
                np.sum(np.square(points - points[start_index]), 1)
            )
            if sorted_indices.shape[0] + indices.shape[0] > max_points:
                last_index = max_points - indices.shape[0]
                indices = np.concatenate([indices, sorted_indices[:last_index]])
            else:
                indices = np.concatenate([indices, sorted_indices])
    else:
        indices = sorted_indices[:max_points]

    if shuffle_indices:
        np.random.shuffle(indices)

    points, features, labels = points[indices], features[indices], labels[indices]

    assert (
        points.shape[0] == max_points
    ), f"Number of returned points: {points.shape} != {max_points}"

    if transform:
        points, features, labels = transform(points, features, labels)

    if np.max(features) > 1:
        # Default normalization to divide all additional features by 255 as
        # you would for RGB colors
        features = torch.FloatTensor(features) / 255.0
    else:
        features = torch.FloatTensor(features)

    points = torch.FloatTensor(points)
    labels = torch.LongTensor(labels)

    return (torch.cat((points, features), dim=-1), labels)
