import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib import pyplot as plt
import tensorflow as tf
from model.model import TrainAndLoggingCallback


CHECKPOINT_DIR = './model/modelLocation/train/'
LOG_DIR = './model/modelLocation/logs/'

# 1. Create the base environment
env = gym_tetris.make('TetrisA-v3')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the framesP
env = VecFrameStack(env, 4, channels_order='last')

with open('currentModel.txt', 'r') as f:
    current_model = f.read()

try:
    model = PPO.load('./model/modelLocation/train/{}', current_model)
except:
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
           n_steps=512) 

def run_action():
    with tf.device('/GPU:0'):
        state = env.reset()
        model.predict(state)
        while True:
            action, _state = model.predict(state)
            state, reward, done, info = env.step(action)
            env.render()
            if done:
                env.close()
                exit()


def print_help():
    print('exit - saves the best model and ends the application\n'
    'test - trains the NN model (will take a while)\n'
    'run - runs the enviroment so that you can see the model in action\n\n')

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

def learn(num):
    with tf.device('/GPU:0'):
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
            except:
                print("Invalid input, number of runs will be 10000")
                num = 10000
            if num < 10000:
                print("input is too low, defaulting to 10000")
                num = 10000
            learn(num)
        elif answer.strip().lower() == 'run':
            run_action()
        else:
            print_help()
main()
