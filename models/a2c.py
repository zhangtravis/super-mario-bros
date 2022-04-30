import torch
import torch.nn as nn

def initialize_uniformly(layer, init_w=3e-3):
    """
      Initialize the weights and bias in [-init_w, init_w]

      :param layer: Linear layer in the model
      :type layer: nn.Linear
      :param init_w: Bounds for weight initialization
      :type init_w: float
      :return: None
    """
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

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
          kernel_size=8, 
          stride=4),
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
      nn.ReLU(inplace=True)
    )

    self.linear = nn.Linear(5184, 512)

    self.actor = nn.Linear(512, output_dim)
    self.critic = nn.Linear(512, 1)

  def forward(self, x):
    features = self.convBlock(x)
    features = features.view(features.size(0), -1)
    features = self.linear(features)

    policy = self.actor(features)
    value = self.critic(features)

    return policy, value
