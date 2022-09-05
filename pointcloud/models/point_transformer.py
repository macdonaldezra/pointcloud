from copy import deepcopy
from typing import Tuple

import pointcloud.utils.pointops as pointops
import torch
import torch.nn as nn
from lib.pointops.functions import pointops
from pointcloud.utils.logging import get_logger

LOGGER = get_logger()

EPSILON = 1e-8


class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_neigbours: int) -> None:
        super().__init__()

        self.query = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.positional_encoding = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, out_channels, kernel_size=1),
        )

        self.attention = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.key_group = pointops.QueryAndGroup(nsample=num_neigbours, return_idx=True)
        self.value_group = pointops.QueryAndGroup(nsample=num_neigbours, use_xyz=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        points, features = inputs
        query = self.query(features)
        key = self.key(features)
        value = self.value(features)

        n_k, _, n_idx = self.key_group(xyz=points, features=key)
        n_v, _ = self.value_group(xyz=points, features=value, idx=n_idx.int())

        # Compute relative position encoding
        n_r = self.positional_encoding(n_k[:, 0:3, :, :])
        n_v = n_v + n_r

        # Compute self-attention
        attention = self.attention(query.unsqueeze(-1) - n_k[:, 3:, :, :] + n_r)
        attention = self.softmax(attention)
        y = torch.sum(n_v * attention, dim=-1, keepdim=False)

        return (points, y)


class PointTransformerBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_neighbours: int = 16
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.linear1 = nn.Conv1d(
            in_channels, self.out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.transformer = PointTransformerLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            num_neigbours=num_neighbours,
        )
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.linear2 = nn.Conv1d(
            self.out_channels, self.out_channels, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        points, features = inputs

        y = self.relu(self.bn1(self.linear1(features)))
        y = self.relu(self.bn(self.transformer([points, y])[1]))
        y = self.bn2(self.linear2(y))
        y += features
        y = self.relu(y)
        return [points, y]


class TransitionDown(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        num_neighbours: int = 16,
    ):
        assert stride > 1
        super().__init__()
        self.out_channels = in_channels if out_channels is None else out_channels

        self.stride = stride
        self.grouper = pointops.QueryAndGroup(nsample=num_neighbours, use_xyz=True)
        self.mlp = nn.Sequential(
            nn.Conv2d(3 + in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.max_pool = nn.MaxPool2d((1, num_neighbours))

    def forward(self, inputs):
        # points, p: (B, N, 3)
        # in_features, x: (B, C_in, N)
        points, features = inputs

        # furthest point sampling and neighbor search
        M = features.shape[-1] // self.stride
        p1_trans = points.transpose(1, 2).contiguous()  # (B, 3, N)
        p_out = (
            pointops.gathering(p1_trans, pointops.furthestsampling(points, M))
            .transpose(1, 2)
            .contiguous()
        )
        n_x, _ = self.grouper(
            xyz=points, new_xyz=p_out, features=features
        )  # (B, 3 + C_in, M, K)

        # mlp and local max pooling
        n_y = self.mlp(n_x)  # (B, C_out, M, K)
        y = self.max_pool(n_y).squeeze(-1)  # (B, C_out, M)
        return [p_out, y]


class TransitionUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.skip_channels = out_channels

        self.linear1 = nn.Sequential(
            nn.Conv1d(in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.linear2 = nn.Sequential(
            nn.Conv1d(self.skip_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, p1x1: torch.Tensor, p2x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # in_points, p1: (B, N, 3)
        # in_features, x1: (B, C_in, N)
        # skip_points, p2: (B, M, 3)
        # skip_features, x2: (B, C_skip, M)
        p1, x1 = p1x1
        p2, x2 = p2x2

        # Three nearest neighbor upsampling
        dist, idx = pointops.nearestneighbor(p2, p1)
        dist_recip = 1.0 / (dist + EPSILON)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        up_x1 = pointops.interpolation(self.linear1(x1), idx, weight)

        # aggregation
        y = self.linear2(x2) + up_x1  # (B, C_out, M)
        return (p2, y)


class SimplePointTransformerSeg(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_neighbours: int):
        super().__init__()
        hidden_channels = input_dim * 4

        # encoder
        self.in_mlp = nn.Sequential(
            nn.Conv1d(3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.block1 = PointTransformerBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_neighbours=num_neighbours,
        )
        self.down = TransitionDown(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_neighbours=num_neighbours,
        )
        self.block2 = PointTransformerBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_neighbours=num_neighbours,
        )

        # decoder
        self.up = TransitionUp(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
        )
        self.block3 = PointTransformerBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_neighbours=num_neighbours,
        )
        self.out_mlp = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, num_classes, kernel_size=1),
        )

    def _split_inputs(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        points = inputs[..., 0:3].contiguous()
        features = (
            inputs[..., 3:].transpose(1, 2).contiguous()
            if inputs.size(-1) > 3
            else deepcopy(points)
        )
        return (points, features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # stride == 1
        points, features = self._split_inputs(inputs)
        features = self.in_mlp(features)
        p1x1 = self.block1([points, features])

        # stride == 4
        p4x4 = self.down(p1x1)
        p4x4 = self.block2(p4x4)

        # stride == 1
        p1y = self.up(p4x4, p1x1)
        p1y = self.block3(p1y)
        y = self.out_mlp(p1y[1])

        return y
