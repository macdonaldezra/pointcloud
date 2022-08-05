import numpy as np
import torch


class TNet(torch.nn.Module):
    """
    T-Net Model
    """

    def __init__(self, num_points: int = 2500) -> None:
        super(TNet, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.max_pooling = torch.nn.MaxPool1d(kernel_size=num_points)
        self.relu = torch.nn.ReLU()

        self.linear1 = torch.nn.Linear(in_features=1024, out_features=512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 9)

        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=64)
        self.batch_norm2 = torch.nn.BatchNorm1d(128)
        self.batch_norm3 = torch.nn.BatchNorm1d(1024)
        self.batch_norm4 = torch.nn.BatchNorm1d(512)
        self.batch_norm5 = torch.nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))

        x = self.max_pooling(x)
        x = x.view(-1, 1024)

        x = self.relu(self.batch_norm4(self.linear1(x)))
        x = self.relu(self.batch_norm5(self.linear2(x)))
        x = self.linear3(x)

        iden = (
            torch.autograd.Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batch_size, 1)
        )

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        return x


class TNetFeatureTransformer(torch.nn.Module):
    """
    T-Net for Feature Transform.
    """

    def __init__(self, num_points: int = 2500, global_feature: bool = True) -> None:
        super(TNetFeatureTransformer, self).__init__()
        self.num_points = num_points
        self.global_feature = global_feature

        self.tnet = TNet(num_points=num_points)

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.batch_norm1 = torch.nn.BatchNorm1d(64)
        self.batch_norm2 = torch.nn.BatchNorm1d(128)
        self.batch_norm3 = torch.nn.BatchNorm1d(1024)

        self.max_pooling = torch.nn.MaxPool1d(num_points)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]
        transformer = self.tnet(x)

        x = x.transpose(2, 1)
        x = torch.bmm(x, transformer)

        x = x.transpose(2, 1)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        point_feature = x

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.max_pooling(self.batch_norm3(self.conv3(x)))

        x = x.view(-1, 1024)
        if self.global_feature:
            return x, transformer
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, point_feature], 1), transformer


class PointNetClassifier(torch.nn.Module):
    """
    Network for Classification: 512, 156, K.
    """

    def __init__(self, num_points: int = 2500, k: int = 2) -> None:
        super(PointNetClassifier, self).__init__()
        self.num_points = num_points

        self.feature_tnet = TNetFeatureTransformer(num_points)

        self.linear1 = torch.nn.Linear(1024, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, k)

        self.batch_norm1 = torch.nn.BatchNorm1d(512)
        self.batch_norm2 = torch.nn.BatchNorm1d(256)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x, transformer = self.feature_tnet(x)

        x = self.relu(self.batch_norm1(self.linear1(x)))
        x = self.relu(self.batch_norm2(self.linear2(x)))

        x = self.linear3(x)
        return self.softmax(x), transformer


class PointNetSegmentor(torch.nn.Module):
    """
    Network for segmentation.
    """

    def __init__(self, num_points: int = 2500, k: int = 2) -> None:
        super(PointNetSegmentor, self).__init__()
        self.num_points = num_points
        self.k = k

        self.feature_tnet = TNetFeatureTransformer(num_points, global_feature=False)

        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)

        self.batch_norm1 = torch.nn.BatchNorm1d(512)
        self.batch_norm2 = torch.nn.BatchNorm1d(256)
        self.batch_norm3 = torch.nn.BatchNorm1d(128)

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x, transformer = self.feature_tnet(x)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))

        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = self.log_softmax(x.view(-1, self.k))
        x = x.view(batchsize, self.num_points, self.k)

        return x, transformer
