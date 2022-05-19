from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from agents.ppo_agent import Agent

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    ppo_agent = Agent(len(SIMPLE_MOVEMENT), env, 0.9, 1.0, 1e-4, load_path='model_weights/ppo/ppo_25199.model', use_cuda=False)
    ppo_agent.play()
