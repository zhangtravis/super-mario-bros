from models.a2c import A2CNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

class Agent():
  def __init__(self, output_dim, env, gamma, load_path=None, use_cuda=False, render=False):
    self.env = env
    self.use_cuda = use_cuda
    self.device = torch.device('cuda' if use_cuda else 'cpu')
    self.model = A2CNetwork(output_dim).to(self.device)

    if load_path != None:
      self.load_model(load_path)

    self.optimizer = optim.Adam(self.model.parameters(), lr=3e-2)

    # Actions & Rewards buffer
    self.saved_actions = []
    self.rewards = []

    self.gamma = gamma
    self.render = render

  def get_action(self, state):
    policy, value = self.model(state)
    action_prob = F.softmax(policy, dim=-1)

    cat = Categorical(action_prob)

    action = cat.sample()

    self.saved_actions.append((cat.log_prob(action), value))
    return action.item()

  def backprop(self):
    R = 0
    policy_losses = []
    value_losses = []
    td_targets = []

    for reward in self.rewards:
      R = reward + self.gamma * R
      td_targets.insert(0, R)

    td_targets = torch.tensor(td_targets)
    td_targets = (td_targets - td_targets.mean()) / (td_targets.std())

    for (log_prob, value), R in zip(self.saved_actions, td_targets):
      advantage = R - value.item()

      # calculate actor (policy) loss 
      policy_losses.append(-log_prob * advantage)

      # calculate critic (value) loss using L1 smooth loss
      value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    self.optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    self.optimizer.step()

    self.rewards = []
    self.saved_actions = []

  def save_model(self, model_path):
    torch.save(self.model.state_dict(), model_path)

  def load_model(self, model_path):
    self.model.load_state_dict(torch.load(model_path)).to(self.device)

  def train(self, num_episodes):
    running_reward = 1

    print('Training...')
    for ep in range(num_episodes):
      state = self.env.reset()
      ep_reward = 0

      print(f'Episode {ep}: ')
      for _ in tqdm(range(1, 10000)):
        state = (torch.from_numpy(state.copy()).float() / 255.).unsqueeze(0).to(self.device)
        state = torch.permute(state, (0, 3, 1, 2))
        action = self.get_action(state)
        state, reward, done, _ = self.env.step(action)

        if self.render:
          self.env.render()

        ep_reward += reward
        if done:
          break

      # update cumulative reward
      running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

      self.backprop()

      print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            ep, ep_reward, running_reward))

      self.save_model('model_weights/a2c.model')
  
