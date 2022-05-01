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
    for step in range(1000):
        if done:
            env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
    env.close()

processes = []

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) 

model.learn(total_timesteps=1000000, callback=callback)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    for _ in range(50):
        p = multiprocessing.Process(target=model.learn, args=(1000000, callback))
        p.start()
    for process in processes:
        p.join()

model.save('thisisatestmodel')