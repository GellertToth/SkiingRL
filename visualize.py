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

ACTION_REPETITION = 1

class DoubleConv(nn.Module):
    def __init__(self, channels, kernel, pool_kernel):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel)
        self.pool_kernel = pool_kernel

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, self.pool_kernel)
        return x

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        """self.double_conv1 = DoubleConv(channels=(3, 4, 8), kernel=(5,5), pool_kernel=(3,3))
        self.double_conv2 = DoubleConv(channels=(8, 8, 16), kernel=(5,5), pool_kernel=(3,3))
        """
        self.layer1 = nn.Linear(6, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 3)     
      

    def forward(self, x): 
        """x = F.max_pool2d(x, (3,3))
        x = self.double_conv1(x)
        x = self.double_conv2(x)
        x = torch.flatten(x, x.dim()-3)
        x = torch.cat((x, f), x.dim()-1)"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def truncate_picture(observation):
    return observation[:, 40:180, 10:-10]

def find_pole_middle(observation):
    for i in range(observation.shape[1]):
        where = (observation[0, i, :] < 100) & (observation[1, i, :] < 100)
        blue_pixels = sum(where)
        if blue_pixels == 10:
            return (i, sum(np.where(where)[0])/10)
    return None

def find_player_pos(observation):
    for i in range(observation.shape[1]):
        where = (observation[1, i, :] < 100) & (observation[2, i, :] < 100)
        red_pixels = sum(where)
        if red_pixels > 0:
            return sum(np.where(where)[0])/red_pixels
    return None

def find_all_trees(observation):
    trees = []
    continous = False
    for i in range(observation.shape[1]):
        where = (observation[0, i, :] < 100) & (observation[2, i, :] < 100)
        green_pixels = sum(where)
        if green_pixels > 0 and not continous:
            continous = True
            trees.append((i, sum(np.where(where)[0])/green_pixels))
        if green_pixels == 0:
            continous = False
    return trees


def get_distance_travelled(old, new):
    old_pole = find_pole_middle(old)
    new_pole = find_pole_middle(new)
    if new_pole is None or old_pole is None:
        return 0
    return max(old_pole[0] - new_pole[0], 0)

def extract_extra_features(observation, old_observation, turn_delta):
    pole = find_pole_middle(observation)
    player = find_player_pos(observation)
    if pole is not None:
        y_delta = pole[0] / observation.shape[0]
        x_delta = (pole[1]-player)/observation.shape[1]
    else:
        y_delta, x_delta = 1, 0
    trees = find_all_trees(observation)
    trees = [(y/observation.shape[0], (x-player)/observation.shape[1]) for (x,y) in trees]
    if len(trees) == 0:
        trees.append((1, 1))
    dist = get_distance_travelled(old_observation, observation)
    return [y_delta, x_delta, trees[0][0], trees[0][1], 1/(dist+1), turn_delta/MAX_TURN]

def get_reward(old_observation, current_observation):
    reward = get_distance_travelled(old_observation, current_observation)

    player_pos = find_player_pos(current_observation)
    new_pole = find_pole_middle(current_observation)
    if new_pole is not None and new_pole[0] < 5 and abs(new_pole[1] - player_pos) < 12:
        reward += (2**8)
    return reward

MAX_TURN = 9

def play_game(target_net, MAX_ITER=1000):
    env = gym.make('ALE/Skiing-v5', render_mode="human")
    obs, info = env.reset()
    obs = np.swapaxes(np.swapaxes(obs, 1, 2), 0, 1)
    obs = truncate_picture(obs)
    turn_delta = 0
    last_features = extract_extra_features(obs, obs, turn_delta)
    done = False
    game_start = time.time()
    action_sum, action_cnt = 0, 0
    reward_sum = 0
    last_rewards = [10**9 for _ in range(10)]
    while not done and action_cnt < MAX_ITER:
        env.render()
        s = time.time()
        if turn_delta == MAX_TURN:
            action = 1
        elif turn_delta == -MAX_TURN:
            action = 2
        elif random.random() < EPSILON:
            # torch.tensor(obs, dtype=torch.float32), 
            res = target_net(torch.tensor(last_features, dtype=torch.float32)).detach().numpy()
            action = np.argmax(res)
        else:
            action = int(random.random()*A)
        diff = time.time()-s
        action_sum += (diff)
        action_cnt += 1
        for _ in range(ACTION_REPETITION):
            observation, reward, done, trunc, info = env.step(action)
            if action == 2:
                turn_delta = min(turn_delta + 1, MAX_TURN)
            elif action == 1:
                turn_delta = max(turn_delta - 1, -MAX_TURN)
            if done:
                break
        observation = np.swapaxes(np.swapaxes(observation, 1, 2), 0, 1)
        observation = truncate_picture(observation)
        features = extract_extra_features(observation, obs, turn_delta)
        reward = get_reward(obs, observation)
        print(turn_delta, features, reward)
        last_rewards.append(reward)
        reward_sum += reward
        obs = observation
        last_features = features

    print(f"Game done in {time.time()-game_start}s, avg it length {action_sum/action_cnt}, avg reward {reward_sum/action_cnt}")

MODEL_PATH = "./"
EPSILON = 0.8
A = 3
target_net = DQN()
target_net.load_state_dict(torch.load(MODEL_PATH+"target", weights_only=True, map_location=torch.device('cpu')))
play_game(target_net)