from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import matplotlib.pyplot as plt
import pickle

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
rewards = []
done = True
rewards = []
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    print(state.shape)
    rewards.append(reward)
    env.render()
env.close()
f = open("rewards.pkl", "wb")
pickle.dump(rewards, f)
f.close()
