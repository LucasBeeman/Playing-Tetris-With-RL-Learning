ECHO OFF
conda activate base
pip install opencv-python tensorflow stable-baselines[mpi] nes-py gym gym-tetris
PAUSE