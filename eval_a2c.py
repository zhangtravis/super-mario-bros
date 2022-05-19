from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agents.a2c_agent import Agent
from agents.a2c_lstm_agent import Agent as LSTMAgent
import argparse

parser = argparse.ArgumentParser(description="Training A2C for Super Mario")
parser.add_argument("--lstm", action='store_true', default=False, help="determines whether to use lstm or not")
args = parser.parse_args()

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    if(args.lstm):
        a2c_agent = LSTMAgent(len(SIMPLE_MOVEMENT), env, 0.99, 1e-30, 0.00025, load_path='model_weights/lstm_extra_conv/a2c_ep_3999.model')
    else:
        a2c_agent = Agent(len(SIMPLE_MOVEMENT), env, 0.99, 1e-30, 0.00025, load_path='model_weights/a2c_extra_conv/a2c_ep_2899.model')
    a2c_agent.play()
