from typing import Callable, Tuple

import numpy as np
import torch


def get_sensat_model_inputs(
    points: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    transform: Callable[
        [np.ndarray, np.ndarray, np.ndarray], list[np.ndarray, np.ndarray, np.ndarray]
    ],
    training: bool = True,
    shuffle_indices: bool = False,
    max_points: int = 80000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing points, features and labels.
    """
    if transform:
        points, features, labels = transform(points, features, labels)

    if max_points and labels.shape[0] > max_points:
        if training:
            start_index = np.random.randint(labels.shape[0])
        else:
            start_index = labels.shape[0] // 2

        indices = np.argsort(np.sum(np.square(points - points[start_index]), 1))[
            :max_points
        ]
        points, features, labels = points[indices], features[indices], labels[indices]

    if shuffle_indices:
        indices = np.arange(labels.shape[0])
        np.random.shuffle(indices)
        points, features, labels = points[indices], features[indices], labels[indices]

    min_point = np.min(points, 0)
    points -= min_point
    if np.max(features) > 1:
        # default normalization to divide all additional features by 255 as
        # you would for RGB colors
        labels /= 255

    return (
        torch.FloatTensor(points),
        torch.FloatTensor(features),
        torch.LongTensor(labels),
    )
