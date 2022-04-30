from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from agents.a2c_agent import Agent

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    a2c_agent = Agent(len(SIMPLE_MOVEMENT), env, 0.01)
    a2c_agent.train(100)
