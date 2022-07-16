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
        x = self.view(-1, 1024)

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
