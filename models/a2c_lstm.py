import torch.nn as nn


class A2CNetwork(nn.Module):
    """
      A2C Network
    """

    def __init__(self, output_dim):
        super(A2CNetwork, self).__init__()

        self.convBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=4,
                stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=2)
        )

        self.lstm = nn.LSTMCell(1920, 512)

        self.actor = nn.Linear(512, output_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x, hx, cx):
        features = self.convBlock(x)
        features = features.reshape(features.size(0), -1)
        hx, cx = self.lstm(features, (hx, cx))

        policy = self.actor(hx)
        value = self.critic(hx)

        return policy, value, hx, cx
