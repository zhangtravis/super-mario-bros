from models.ppo import PPO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

class Agent():
  def __init__(self, output_dim, env, gamma, tau, lr, eps=0.2, local_steps=512, num_epochs=10, 
  batch_size=16, load_path=None, use_cuda=False, render=False):
    self.env = env
    self.use_cuda = use_cuda
    self.device = torch.device('cuda' if use_cuda else 'cpu')
    self.model = PPO(output_dim).to(self.device)
    self.entropy = None
    self.num_local_steps = local_steps
    self.tau = tau
    self.eps = eps

    self.num_epochs = num_epochs
    self.batch_size = batch_size

    if load_path != None:
      self.load_model(load_path)

    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

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

    self.entropy = cat.entropy()

    self.saved_actions.append((cat.log_prob(action), value[0]))
    return action.item(), policy

  def backprop(self):
    R = 0
    policy_losses = []
    value_losses = []
    td_targets = []
    mse = nn.MSELoss()

    for reward in self.rewards:
      R = reward + self.gamma * R
      td_targets.insert(0, R)

    td_targets = torch.tensor(td_targets)

    for (log_prob, value), R in zip(self.saved_actions, td_targets):
      advantage = R - value.item()

      # calculate actor (policy) loss 
      policy_losses.append(-log_prob * advantage)

      # calculate critic (value) loss using L1 smooth loss
      value_losses.append(mse(value, torch.tensor([R])))

    # reset gradients
    self.optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() - 0.02 * self.entropy

    # perform backprop
    loss.backward()
    self.optimizer.step()

    self.rewards = []
    self.saved_actions = []

  def save_model(self, model_path):
    torch.save(self.model.state_dict(), model_path)

  def load_model(self, model_path):
    self.model.load_state_dict(torch.load(model_path))
    self.model = self.model.to(self.device)

  def train(self, num_episodes):
    running_reward = 1

    print('Training...')
    for ep in range(num_episodes):
      state = self.env.reset()
      ep_reward = 0

      old_log_policies = []
      states = []
      actions = []
      values = []
      rewards = []
      dones = []

      print(f'Episode {ep}: ')

      for _ in tqdm(range(self.num_local_steps)):
        state = (torch.from_numpy(state.copy()).float() / 255.).to(self.device)
        state = torch.permute(state.unsqueeze(0), (0, 3, 1, 2))
        states.append(state)
        policy, value = self.model(state)
        values.append(value.squeeze())
        policy = F.softmax(policy, dim=1)
        old_m = Categorical(policy)
        action = old_m.sample()
        actions.append(action)
        
        old_log_policy = old_m.log_prob(action)
        old_log_policies.append(old_log_policy)

        # print(action)
        state, reward, done, _ = self.env.step(action.item())
        rewards.append(reward)
        dones.append(done)

      state = (torch.from_numpy(state.copy()).float() / 255.).to(self.device)
      state = torch.permute(state.unsqueeze(0), (0, 3, 1, 2))
      _, next_value, = self.model(state)
      next_value = next_value.squeeze()
      old_log_policies = torch.cat(old_log_policies).detach()
      # print(actions)
      actions = torch.cat(actions)
      # print(values)
      values = torch.stack(values).detach()
      # print(values.shape)
      states = torch.cat(states)
      gae = 0
      R = []

      for value, reward, done in list(zip(values, rewards, dones))[::-1]:
        gae = gae * self.gamma * self.tau
        gae = gae + reward + self.gamma * next_value.detach() * (1 - done) - value.detach()
        next_value = value
        R.append(gae + value)
      R = R[::-1]
      R = torch.stack(R).detach()
      # print(R.shape)
      advantages = R - values
      print('Loop through data...')
      for i in tqdm(range(self.num_epochs)):
          indice = torch.randperm(self.num_local_steps)
          for j in range(self.batch_size):
              batch_indices = indice[
                              int(j * (self.num_local_steps / self.batch_size)): int((j + 1) * (
                                      self.num_local_steps / self.batch_size))]
              # print(states.shape)
              # print(states[batch_indices].shape)
              logits, value = self.model(states[batch_indices])
              new_policy = F.softmax(logits, dim=1)
              new_m = Categorical(new_policy)
              new_log_policy = new_m.log_prob(actions[batch_indices])
              ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
              # print(ratio.shape)
              # print(advantages[batch_indices].shape)
              actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                  torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) *
                                                  advantages[
                                                      batch_indices]))
              # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
              critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
              entropy_loss = torch.mean(new_m.entropy())
              total_loss = actor_loss + critic_loss - 0.02 * entropy_loss
              self.optimizer.zero_grad()
              total_loss.backward()
              torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
              self.optimizer.step()
      print("Episode: {}. Total loss: {}".format(ep, total_loss))

      self.save_model('model_weights/ppo/ppo.model')

  def play(self):
    running_reward = 1
    ep = 0
    print('Playing...')
    while True:
      state = self.env.reset()
      ep_reward = 0

      # print(f'Episode {ep}: ')
      for _ in tqdm(range(1, 10000)):
        with torch.no_grad():
          state = (torch.from_numpy(state.copy()).float() / 255.).unsqueeze(0).to(self.device)
          state = torch.permute(state, (0, 3, 1, 2))
          action = self.get_action(state)
          state, reward, done, _ = self.env.step(action)
          self.rewards.append(reward)

          self.env.render()

          ep_reward += reward
          if done:
            break

      # update cumulative reward
      running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

      print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            ep, ep_reward, running_reward))
      ep += 1


  
