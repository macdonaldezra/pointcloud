import logging
import typing as T

import numpy as np
import torch
from pointcloud.models.pointtransformer import PointTransformerSeg
from pointcloud.processors.sensat.dataset import LABELS, SensatDataSet
from tensorboardX import SummaryWriter


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


def get_logger() -> logging.RootLogger:
    """
    Return a generic logger to be called by the main process training a given model.
    """
    logger_name = "Train Logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    format = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)

    return logger


def train(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_function: T.Any,
    optimizer: T.Any,
    logger: logging.RootLogger,
    tensorboard_writer: SummaryWriter,
    current_epoch: int,
    epochs: int,
    num_classes: int,
    ignore_index: T.Optional[int] = None,
    debug: bool = True,
    print_frequency: int = 5,
) -> T.Tuple[float, float, float, float]:
    """
    Train the provided model for an epoch.
    """
    model.train()
    loss_counter = MetricCounter()
    intersection_counter = MetricCounter()
    union_counter = MetricCounter()
    label_counter = MetricCounter()

    for index, (points, features, labels, offset) in enumerate(data_loader):
        points, features, labels, offset = (
            points.cuda(non_blocking=True),
            features.cuda(non_blocking=True),
            labels.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        output = model([points, features, offset])
        if labels.shape[-1] == 1:
            labels = labels[:, 0]

        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        n = points.size(0)

        if debug:
            assert output.dim() in [1, 2, 3], f"Actual output dimension: {output.dim()}"
            assert (
                output.shape == labels.shape
            ), f"Output shape of {output.shape} != Provided label shape of {labels.shape}"

        intersection, union, labels = compute_intersection_and_union(
            output,
            labels,
            num_classes,
            ignore_index,
        )
        intersection, union, labels = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            labels.cpu().numpy(),
        )
        intersection_counter.update(intersection)
        union_counter.update(union)
        label_counter.update(labels)
        current_iter = current_epoch * len(data_loader) + index + 1

        accuracy = sum(intersection_counter.value) / (sum(labels.value) + 1e-10)
        loss_counter.update(loss.item(), n)

        if (index + 1) % print_frequency == 0:
            logger.info(
                f"Epoch: [{current_epoch}/{epochs}] Loss: {loss_counter.value} Accuracy: {accuracy}"
            )

        # Update TensorBoard Writer values
        tensorboard_writer.add_scalar(
            "loss_train_batch", loss_counter.value, current_iter
        )
        tensorboard_writer.add_scalar(
            "mean_intersection_over_union_train_batch",
            np.mean(intersection / (union + 1e-10)),
            current_iter,
        )
        tensorboard_writer.add_scalar(
            "mean_accuracy_train_batch",
            np.mean(intersection / (labels + 1e-10)),
            current_iter,
        )
        tensorboard_writer.add_scalar("accuracy_train_batch", accuracy, current_iter)

    iou = intersection_counter.sum / (union_counter.sum + 1e-10)
    accuracy = intersection_counter.sum / (label_counter.sum + 1e-10)
    average_iou = np.mean(iou)
    average_accuracy = np.mean(accuracy)
    total_accuracy = sum(intersection_counter.sum) / sum(label_counter.sum + 1e-10)

    logger.info(
        f"Train result at epoch [{current_epoch}/{epochs}:"
        + " Average IoU = {average_iou:.4f}, Average Accuracy: {average_accuracy:.4f}"
        + "All Accuracy: {total_accuracy:.4f}"
    )

    return (loss_counter.average, average_iou, average_accuracy, total_accuracy)


def validate(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_function: T.Any,
    optimizer: T.Any,
    logger: logging.RootLogger,
    tensorboard_writer: SummaryWriter,
    current_epoch: int,
    epochs: int,
    num_classes: int,
    ignore_index: T.Optional[int] = None,
    debug: bool = True,
    print_frequency: int = 5,
) -> T.Tuple[float, float, float, float]:
    """
    Test a model for a given dataset.
    """

    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    model.eval()
    loss_counter = MetricCounter()
    intersection_counter = MetricCounter()
    union_counter = MetricCounter()
    label_counter = MetricCounter()

    for index, (points, features, labels, offset) in enumerate(data_loader):
        points, features, labels, offset = (
            points.cuda(non_blocking=True),
            features.cuda(non_blocking=True),
            labels.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        output = model([points, features, offset])
        if labels.shape[-1] == 1:
            labels = labels[:, 0]

        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        n = points.size(0)

        if debug:
            assert output.dim() in [1, 2, 3], f"Actual output dimension: {output.dim()}"
            assert (
                output.shape == labels.shape
            ), f"Output shape of {output.shape} != Provided label shape of {labels.shape}"

        intersection, union, labels = compute_intersection_and_union(
            output,
            labels,
            num_classes,
            ignore_index,
        )
        intersection, union, labels = (
            intersection.cpu().numpy(),
            union.cpu().numpy(),
            labels.cpu().numpy(),
        )
        intersection_counter.update(intersection)
        union_counter.update(union)
        label_counter.update(labels)
        current_iter = current_epoch * len(data_loader) + index + 1

        accuracy = sum(intersection_counter.value) / (sum(labels.value) + 1e-10)
        loss_counter.update(loss.item(), n)

        if (index + 1) % print_frequency == 0:
            logger.info(
                f"Epoch: [{current_epoch}/{epochs}] Loss: {loss_counter.value} Accuracy: {accuracy}"
            )

        # Update TensorBoard Writer values
        tensorboard_writer.add_scalar(
            "loss_train_batch", loss_counter.value, current_iter
        )
        tensorboard_writer.add_scalar(
            "mean_intersection_over_union_train_batch",
            np.mean(intersection / (union + 1e-10)),
            current_iter,
        )
        tensorboard_writer.add_scalar(
            "mean_accuracy_train_batch",
            np.mean(intersection / (labels + 1e-10)),
            current_iter,
        )
        tensorboard_writer.add_scalar("accuracy_train_batch", accuracy, current_iter)

    iou = intersection_counter.sum / (union_counter.sum + 1e-10)
    accuracy = intersection_counter.sum / (label_counter.sum + 1e-10)
    average_iou = np.mean(iou)
    average_accuracy = np.mean(accuracy)
    total_accuracy = sum(intersection_counter.sum) / sum(label_counter.sum + 1e-10)

    logger.info(
        f"Validation result at epoch [{current_epoch}/{epochs}: "
        + f"Average IoU = {average_iou:.4f}, Average Accuracy: {average_accuracy:.4f} "
        + f"All Accuracy: {total_accuracy:.4f}"
    )
    for i in range(num_classes):
        logger.info(
            f"Class: {LABELS[index]} Result: IOU = {iou[index]}, Accuracy = {accuracy[index]}"
        )
    logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    return (loss_counter.average, average_iou, average_accuracy, total_accuracy)
