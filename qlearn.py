from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym
from gym.spaces import Box
from gym import Wrapper
import itertools
import numpy as np
from collections import defaultdict
import time
import pickle

# Implementation of Greedy Epsilon Policy and Q value adapted from: https://www.geeksforgeeks.org/q-learning-in-python/


class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.current_y = 40
        self.world = world
        self.stage = stage
        self.life = 3

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # score = info['score'] - self.curr_score
        # if score == 200:
        #     score -= 200.
        # elif score == 50:
        #     score -= 48.
        # elif score == 100:
        #     score -= 95.
        # else:
        #     score /= 40.
        # reward += score
        self.curr_score = info["score"]
        if info['life'] < self.life:
            reward -= 30
        self.life = info['life']
        # if info['status'] == 'tall':
        #     reward += 5
        reward += (info['y_pos'] - self.current_y)/2
        reward += (info['x_pos'] - self.current_x)/2
        self.current_y = info['y_pos']
        self.current_x = info["x_pos"]
        return state, reward, done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        return self.env.reset()


def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    def policyFunction(state):

        Action_probabilities = np.ones(num_actions,
                                       dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction


def defaultaction():
    return np.zeros(env.action_space.n)


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            prior_Q = pickle.load(f)
            print(len(list(prior_Q.keys())))
            return defaultdict(defaultaction, prior_Q)
    except:
        pass


def save_dict(filename, d):
    with open(filename, 'wb') as f:
        pickle.dump(d, f)


def qLearning(env, num_episodes, discount_factor=0.3, alpha=0.6, epsilon=0.1, Q=None):

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    if not Q:
        Q = defaultdict(defaultaction)
    rewards = []
    # Create an empty variable
    empty_list = []
    # Open the pickle file in 'wb' so that you can write and dump the empty variable
    openfile = open('rewards.pkl', 'wb')
    pickle.dump(empty_list, openfile)
    openfile.close()
    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)

    # For every episode
    stage_complete = False

    for ith_episode in range(num_episodes):
        # Reset the environment and pick the first action
        if stage_complete:
            break
        state = env.reset()
        str_state = state.tostring()
        for t in itertools.count():

            # get probabilities of all actions from current state
            action_probabilities = policy(str_state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            if t % 1000 == 0:
                f = open("rewards.pkl", "wb")
                pickle.dump(rewards, f)
                f.close()
                print(t)

            str_next_state = next_state.tostring()
            env.render()

            # TD Update
            best_next_action = np.argmax(Q[str_next_state])
            td_target = reward + discount_factor * \
                Q[str_next_state][best_next_action]
            td_delta = td_target - Q[str_state][action]
            Q[str_state][action] += alpha * td_delta
            if info['flag_get']:
                print('STAGE COMPLETE')
                stage_complete = True
            # done is True if episode terminated
            if done:
                break
            state = next_state

    return Q


env = gym_super_mario_bros.make('SuperMarioBros-v3')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
# env = CustomReward(env, world=0, stage=0)
before = time.time()
# prior_Q = load_dict('Q.pkl')
prior_Q = None
Q = qLearning(env, 5, Q=prior_Q)
print(len(list(Q.keys())))
# save_dict('Q.pkl', dict(Q))
print(time.time() - before)
