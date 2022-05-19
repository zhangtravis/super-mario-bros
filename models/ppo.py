import torch.nn as nn

class PPO(nn.Module):
  def __init__(self, output_dim):
    super(PPO, self).__init__()

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

    self.linear = nn.Linear(1920, 512)

    self.actor = nn.Linear(512, output_dim)
    self.critic = nn.Linear(512, 1)

  def forward(self, x):
    features = self.convBlock(x)
    features = features.view(features.size(0), -1)
    features = self.linear(features)

    policy = self.actor(features)
    value = self.critic(features)

    return policy, value
