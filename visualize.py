import gymnasium as gym
import ale_py
# pip install ale-py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import random
from collections import namedtuple, deque
import time
gym.register_envs(ale_py)

env = gym.make('ALE/Skiing-v5', render_mode="human")
while True:
    env.render()
    env.step(0)