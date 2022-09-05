import logging
import random
import shutil
import typing as T
from pathlib import Path

import numpy as np
import pointcloud.utils.transforms as transformers
import torch
from pointcloud.config import DATA_PATH
from pointcloud.models.point_transformer import SimplePointTransformerSeg
from pointcloud.processors.sensat.dataset import LABELS, SensatDataSet
from pointcloud.utils.logging import get_logger
from pointcloud.utils.metrics import MetricCounter, compute_intersection_and_union
from tensorboardX import SummaryWriter
from tqdm import tqdm


def get_transformers(choose: int = 3) -> transformers.DataTransformer:
    """
    Load all of the transforms and return the data transformer class.
    """
    return transformers.DataTransformer(
        transforms=[
            transformers.RandomPointRotation(),
            transformers.RandomPointScale(),
            transformers.RandomPointShift(),
            transformers.RandomPointFlip(),
            transformers.RandomPointJitter(),
            transformers.RandomlyDropColor(),
            transformers.RandomlyShiftBrightness(),
            transformers.ChromaticColorContrast(),
            transformers.ChromaticColorTranslation(),
            transformers.ChromaticColorJitter(),
            transformers.HueSaturationTranslation(),
        ],
        choose=choose,
    )


def collate_fn(
    batch: torch.Tensor,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Overrides the default collate_fn provided by the dataloader. For more
    on this function, see here:
        https://pytorch.org/docs/stable/data.html#working-with-collate-fn
    """
    points, features, labels = list(zip(*batch))
    offset, count = [], 0
    for item in points:
        count += item.shape[0]
        offset.append(count)

    return (
        torch.cat(points),
        torch.cat(features),
        torch.cat(labels),
    )


def manual_seed(seed: int) -> None:
    """
    Seed random, numpy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    for index, (inputs, labels) in tqdm(
        enumerate(data_loader), total=len(data_loader), smoothing=0.9
    ):
        inputs, labels = (
            inputs.cuda(non_blocking=True),
            labels.cuda(non_blocking=True),
        )
        logger.info(f"Shapes: points - {inputs.size()}, labels - {labels.size()}")
        output = model(inputs)
        if labels.shape[-1] == 1:
            labels = labels[:, 0]

        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        n = inputs.size(0)

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

    for index, (inputs, labels) in tqdm(
        enumerate(data_loader), total=len(data_loader), smoothing=0.9
    ):
        inputs, labels = (
            inputs.cuda(non_blocking=True),
            labels.cuda(non_blocking=True),
        )
        output = model(inputs)
        if labels.shape[-1] == 1:
            labels = labels[:, 0]

        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.max(1)[1]
        n = inputs.size(0)

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


def main(
    feature_dim: int,
    num_classes: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    epochs: int,
    tensorboard_path: Path,
    batch_size: int,
    max_points: int,
    save_frequency: int,
    data_path: Path,
    save_path: Path,
    weights_path: T.Optional[Path] = None,
    checkpoint_path: T.Optional[Path] = None,
    include_validation: bool = True,
) -> None:
    """
    The main training loop that performs train and validation steps to train point_transformer.

    Currently this directory and routine is only configured to train on the Sensat Urban dataset.
    """
    logger = get_logger()
    if not torch.cuda.is_available():
        logger.info(f"CUDA must be available for model to run.")

    loss_function = torch.nn.CrossEntropyLoss()
    model = SimplePointTransformerSeg(
        input_dim=feature_dim, num_classes=num_classes, num_neighbours=16
    )
    model = torch.nn.DataParallel(model.cuda())
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1
    )

    tensorboard_writer = SummaryWriter(tensorboard_path.as_posix())
    start_epoch = 0
    best_iou = 0

    if weights_path and weights_path.is_file():
        checkpoint = torch.load(weights_path.as_posix())
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Loaded weights from {weights_path.as_posix()}")

    # Load model from checkpoint if checkpoint filepath is provided
    if checkpoint_path and checkpoint_path.exists():
        logging.info(f"Loading checkpoint from: {checkpoint_path.as_posix()}")

        checkpoint = torch.load(
            checkpoint.as_posix(), map_location=lambda storage, loc: storage.cuda()
        )
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_iou = checkpoint["best_iou"]

        logger.info(
            f"Loaded checkpoint from: {checkpoint_path.as_posix()} at epoch {start_epoch}"
        )
    elif checkpoint_path:
        logger.info(f"No checkpoint found at {checkpoint_path.as_posix()}")

    # Load training and potentially validation data loaders
    training_data = SensatDataSet(
        data_partition="train",
        data_path=data_path,
        transform=get_transformers(),
        shuffle_indices=True,
        max_points=max_points,
    )
    logger.info(f"Count of training data samples: {len(training_data)}")

    training_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        # collate_fn=collate_fn,
    )

    if include_validation:
        validation_data = SensatDataSet(
            data_partition="validation", data_path=data_path, max_points=max_points
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=4,
            drop_last=True,
            pin_memory=True,
            # collate_fn=collate_fn,
        )
    else:
        validation_loader = None

    for epoch in range(start_epoch, epochs):
        loss_train, average_IoU_train, average_acc_train, all_acc_train = train(
            data_loader=training_loader,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            logger=logger,
            tensorboard_writer=tensorboard_writer,
            current_epoch=epoch,
            epochs=epochs,
            num_classes=num_classes,
        )

        epoch_log = epoch + 1
        tensorboard_writer.add_scalar("loss_train", loss_train, epoch_log)
        tensorboard_writer.add_scalar("average_IoU_train", average_IoU_train, epoch_log)
        tensorboard_writer.add_scalar("average_acc_train", average_acc_train, epoch_log)
        tensorboard_writer.add_scalar("all_acc_train", all_acc_train, epoch_log)

        if include_validation:
            loss_val, average_IoU_val, average_acc_val, all_acc_val = validate(
                data_loader=validation_loader,
                model=model,
                loss_function=loss_function,
                optimizer=optimizer,
                logger=logger,
                tensorboard_writer=tensorboard_writer,
                current_epoch=epoch,
                epochs=epochs,
                num_classes=num_classes,
            )

            tensorboard_writer.add_scalar("loss_val", loss_val, epoch_log)
            tensorboard_writer.add_scalar("average_IoU_val", average_IoU_val, epoch_log)
            tensorboard_writer.add_scalar("average_acc_val", average_acc_val, epoch_log)
            tensorboard_writer.add_scalar("all_acc_val", all_acc_val, epoch_log)

            is_best = average_IoU_val > best_iou
            best_iou = max(best_iou, average_IoU_val)

        if epoch % save_frequency == 0:
            filepath = save_path / "model_latest.pth"
            logger.info(f"Saving model checkpoint to: {filepath}")
            torch.save(
                {
                    "epoch": epoch_log,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_iou": best_iou,
                    "is_best": is_best,
                },
                filepath.as_posix(),
            )

            if is_best:
                logger.info(
                    f"Best validation average Intersection/Union updated to: {best_iou:.4f}"
                )
                shutil.copyfile(filepath, save_path / "model_best.pth")

    tensorboard_writer.close()
    logger.info(f"Training Complete!\nBest Intersection/Union Recorded: {best_iou:.4f}")


if __name__ == "__main__":
    # TODO: Move these configurations to a Pydantic object and add functionality
    # to load configurations from YAML files.
    tensorboard_path = DATA_PATH / "tensorboard"
    tensorboard_path.mkdir(exist_ok=True)
    save_path = DATA_PATH / "model"
    save_path.mkdir(exist_ok=True)
    manual_seed(32)

    main(
        feature_dim=6,
        num_classes=len(LABELS),
        learning_rate=0.5,
        momentum=0.9,
        weight_decay=0.0001,
        epochs=5,
        tensorboard_path=tensorboard_path,
        batch_size=16,
        max_points=80000,
        save_frequency=1,
        data_path=DATA_PATH / "sensat_urban" / "grid_0.2",
        save_path=save_path,
    )
