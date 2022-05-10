import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import pyplot as plt
import tensorflow as tf
import multiprocessing
import os 
import time
from model.model import TrainAndLoggingCallback

# 1. Create the base environment
env = gym_tetris.make('TetrisA-v3')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

def run_action():
    done = True
    for step in range(100000):
        if done:
            env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
    env.close()

def print_help():
    print('exit - saves the best model and ends the application\n'
    'test - trains the NN model (will take a while)\n'
    'run - runs the enviroment so that you can see the model in action\n\n')

CHECKPOINT_DIR = './model/modelLocation/train/'
LOG_DIR = './model/modelLocation/logs/'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) 

def learn(num):
    model.learn(total_timesteps=num , callback=callback)

def main():
    while True:
        answer = input("Enter your desired function, or type help for the description of each funciton: ")
        if answer.strip().lower() == 'exit':
            break
        elif answer.strip().lower() == 'help':
            print_help()
        elif answer.strip().lower() == 'test':
            num = input("Enter the ammount of run you wish to pass. The minimum is 10000: ")
            try:
                num = int(num)
            except(TypeError):
                print("Invalid input, number of runs will be 10000")
                num = 10000
            learn(num)
        elif answer.strip().lower() == 'run':
            run_action()
        else:
            print_help()
main()

model.save('LatestModel')