import typing as T

import numpy as np
import torch


class MetricCounter(object):
    """
    Class that computes and stores basic counter state values.
    """

    def __init__(self) -> None:
        self.current = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        self.current = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value: T.Any, n: int = 1) -> None:
        """
        Update the current value, sum, count, and average from the value parameter.
        """
        self.current = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def compute_intersection_and_union(
    output: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: T.Optional[int] = None,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute and return the intersection and union of an output and set of labels.
    """
    output = output.view(-1)
    labels = labels.view(-1)
    if ignore_index:
        output[labels == ignore_index] = ignore_index

    intersection = output[output == labels]
    intersection, _ = torch.histc(
        intersection, bins=num_classes, min=0, max=num_classes - 1
    )
    labels_area, _ = torch.histc(labels, bins=num_classes, min=0, max=num_classes - 1)
    output_area, _ = torch.histc(output, bins=num_classes, min=0, max=num_classes - 1)

    union = labels_area + output_area - intersection

    return intersection, union, labels_area
