# %%
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
import matplotlib.pyplot as plt
import time
gym.register_envs(ale_py)

# %%
class Memory():
    def __init__(self, capacity, file=None):
        if file is not None:
            try:
                with open(file, "rb") as f:
                    d = pickle.load(f)
                    self.memory = d
            except Exception as e:
                self.memory = deque([], maxlen=capacity)
                print(e)
        else:
            self.memory = deque([], maxlen=capacity)

    def push(self, el):
        self.memory.append(el)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def to_list(self):
        return list(self.memory)

# %%
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
        """
        self.double_conv1 = DoubleConv(channels=(3, 4, 8), kernel=(5,5), pool_kernel=(3,3))
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
        x = torch.cat((x, f), x.dim()-1)
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def optimize_model(optimizer, memory, policy_net, target_net):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # state_batch = torch.tensor(np.array([state for (state, _, _, _, _, _) in transitions]), dtype=torch.float32, device="cuda")
    feature_batch = torch.tensor(np.array([f for (_, f, _, _, _, _) in transitions]), dtype=torch.float32, device="cuda")

    action_batch = torch.tensor([[action] for (_, _, action, _, _, _) in transitions], device="cuda")
    reward_batch =  torch.tensor([reward for (_, _, _, reward, _, _) in transitions], device="cuda")
    # next_state_batch = torch.tensor(np.array([s2 for (_, _, _, _, s2, _) in transitions]), dtype=torch.float32, device="cuda")
    next_feature_batch = torch.tensor(np.array([f for (_, _, _, _, _, f) in transitions]), dtype=torch.float32, device="cuda")



    state_qvalues = policy_net(feature_batch)
    state_action_values = state_qvalues.squeeze(1).gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        next_qvalues = target_net(next_feature_batch).squeeze(1)
        next_state_values = next_qvalues.max(axis=1).values
        next_state_values = torch.tensor(next_state_values)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()

    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)

# %%
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

# %%
def get_distance_travelled(old, new):
    old_pole = find_pole_middle(old)
    new_pole = find_pole_middle(new)
    if new_pole is None or old_pole is None:
        return 0
    dist = max(old_pole[0] - new_pole[0], 0)
    if dist > 50:
        dist = 0
    return dist

def extract_extra_features(observation, old_observation, turn_delta):
    pole = find_pole_middle(observation)
    player = find_player_pos(observation)
    if pole is not None and player is not None:
        y_delta = pole[0] / observation.shape[1]
        x_delta = (pole[1]-player)/observation.shape[1]
    else:
        y_delta, x_delta = 1, 0
    trees = find_all_trees(observation)
    trees = [(y/observation.shape[1], (x-player)/observation.shape[1]) for (x,y) in trees]
    if len(trees) == 0:
        trees.append((1, 1))
    dist = get_distance_travelled(old_observation, observation)
    return [y_delta, x_delta, trees[0][0], trees[0][1], 1/(dist+1), turn_delta/MAX_TURN]

def get_reward(old_observation, current_observation):
    reward = get_distance_travelled(old_observation, current_observation)
    if reward == 0:
        reward -= 30
    player_pos = find_player_pos(current_observation)
    new_pole = find_pole_middle(current_observation)
    if new_pole is not None and new_pole[0] < 5 and abs(new_pole[1] - player_pos) < 12:
        reward += (2**9)
    return reward

MAX_TURN = 9

# %%
def play_game(optimizer, memory, policy_net, target_net, env, MAX_ITER=1000):
    game_start = time.time()
    obs, info = env.reset()
    obs = np.swapaxes(np.swapaxes(obs, 1, 2), 0, 1)
    obs = truncate_picture(obs)
    turn_delta = 0
    last_features = extract_extra_features(obs, obs, turn_delta)
    done = False
    action_sum, action_cnt = 0, 0
    reward_sum = 0
    while not done and action_cnt < MAX_ITER:
        s = time.time()
        optimize_model(optimizer, memory, policy_net, target_net)
        if turn_delta == MAX_TURN:
            action = 1
        elif turn_delta == -MAX_TURN:
            action = 2
        elif random.random() < EPSILON:
            # torch.tensor(obs, dtype=torch.float32, device="cuda"), 
            res = target_net(torch.tensor(last_features, dtype=torch.float32, device="cuda")).cpu().detach().numpy()
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
        reward_sum += reward
        memory.push((obs/255.0, last_features, action, reward, observation/255.0, features))
        obs = observation
        last_features = features

    print(f"Game done in {time.time()-game_start}s, avg it length {action_sum/action_cnt}, avg reward {reward_sum/action_cnt}")

    


# %%
BATCH_SIZE = 128
GAMMA = 0.95
TAU = 0.005
LR = 0.0005
EPSILON = 0.8
A = 3
MODEL_PATH = "./"
MEMORY_PATH = "./memory"
ACTION_REPETITION = 1


target_net = DQN()
target_net.load_state_dict(torch.load(MODEL_PATH+"target", weights_only=True, map_location=torch.device('cuda')))
target_net.to("cuda")
policy_net = DQN()
policy_net.load_state_dict(target_net.state_dict())
policy_net.load_state_dict(torch.load(MODEL_PATH+"policy", weights_only=True, map_location=torch.device('cuda')))
policy_net.to("cuda")


memory = Memory(7*(10**4), file=MEMORY_PATH)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# %%
env = gym.make('ALE/Skiing-v5')
cnt = 0
while True:
    try:
        play_game(optimizer, memory, policy_net, target_net, env)
    except Exception as e:
        pass
    torch.save(target_net.state_dict(), MODEL_PATH+"target")
    torch.save(policy_net.state_dict(), MODEL_PATH+"policy")
    cnt += 1
    if cnt % 1000 == 0:
        with open(MEMORY_PATH, 'wb') as f:
            pickle.dump(memory.memory, f)




