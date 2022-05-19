from models.a2c_lstm import A2CNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Agent():
    def __init__(self, output_dim, env, gamma, eps, lr, load_path=None, use_cuda=False, render=False):
        self.env = env
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = A2CNetwork(output_dim).to(self.device)
        self.entropy = None
        self.learning_rate = lr

        if load_path != None:
            print(f'Loading in weights from {load_path}')
            self.load_model(load_path)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        self.writer = SummaryWriter()

        # Actions & Rewards buffer
        self.saved_actions = []
        self.rewards = []

        self.gamma = gamma
        self.eps = eps
        self.render = render

    def get_action(self, state, hx, cx):
        policy, value, hx, cx = self.model(state, hx, cx)
        action_prob = F.softmax(policy, dim=-1)

        cat = Categorical(action_prob)

        action = cat.sample()

        self.entropy = cat.entropy()

        self.saved_actions.append((cat.log_prob(action), value[0]))
        return action.item(), hx, cx

    def backprop(self):
        R = 0
        policy_losses = []
        value_losses = []
        td_targets = []
        mse = nn.MSELoss()

        for reward in self.rewards:
            R = reward + self.gamma * R
            td_targets.insert(0, R)

        td_targets = torch.tensor(td_targets).to(self.device)
        td_targets = (td_targets - td_targets.mean()) / \
            (td_targets.std() + self.eps)

        for (log_prob, value), R in zip(self.saved_actions, td_targets):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            # value_losses.append(F.smooth_l1_loss(value.float(), torch.tensor([R]).float()))
            value_losses.append(mse(value.float(), torch.tensor([R]).float().to(self.device)))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum() - 0.02 * self.entropy

        # perform backprop
        loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.saved_actions = []

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)

    def train(self, num_episodes):
        running_reward = 1
        done = True
        state = self.env.reset()

        print('Training...')
        for ep in range(num_episodes):
            if done:
                state = self.env.reset()
                hx = torch.zeros((1, 512), dtype=torch.float).to(self.device)
                cx = torch.zeros((1, 512), dtype=torch.float).to(self.device)
            else:
                hx = hx.detach()
                cx = cx.detach()
            ep_reward = 0

            print(f'Episode {ep}: ')
            for _ in tqdm(range(1, 4500)):
                state = (torch.from_numpy(state.copy()).float() /
                         255.).unsqueeze(0).to(self.device)
                state = torch.permute(state, (0, 3, 1, 2))
                action, hx, cx = self.get_action(state, hx, cx)
                state, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)

                if self.render:
                    self.env.render()

                ep_reward += reward
                if done:
                    break

            # update cumulative reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            print('Backpropagating...')
            self.backprop()

            self.writer.add_scalar('Reward/train', running_reward, ep)
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  ep, ep_reward, running_reward))
            state = (torch.from_numpy(state.copy()).float() / 255.).unsqueeze(0).to(self.device)
            state = torch.permute(state, (0, 3, 1, 2))
            state = torch.permute(state, (0, 2, 3, 1))
            state = state.cpu().squeeze().numpy()

            if((ep+1) % 100 == 0):
                self.save_model(
                    f'model_weights/lstm_extra_conv/a2c_ep_{ep}.model')

    def play(self):
        running_reward = 1
        ep = 0
        done = True
        print('Playing...')
        while True:
            if done:
                state = self.env.reset()
                hx = torch.zeros((1, 512), dtype=torch.float).to(self.device)
                cx = torch.zeros((1, 512), dtype=torch.float).to(self.device)
            else:
                hx = hx.detach()
                cx = cx.detach()
            ep_reward = 0

            print(f'Episode {ep}: ')
            for _ in tqdm(range(1, 10000)):
                with torch.no_grad():
                    state = (torch.from_numpy(state.copy()).float() /
                         255.).unsqueeze(0).to(self.device)
                    state = torch.permute(state, (0, 3, 1, 2))
                    action, hx, cx = self.get_action(state, hx, cx)
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
                state = (torch.from_numpy(state.copy()).float() / 255.).unsqueeze(0).to(self.device)
                state = torch.permute(state, (0, 3, 1, 2))
                state = torch.permute(state, (0, 2, 3, 1))
                state = state.cpu().squeeze().numpy()
