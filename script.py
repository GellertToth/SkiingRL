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
        self.double_conv1 = DoubleConv(channels=(3, 4, 8), kernel=(5,5), pool_kernel=(2,2))
        self.double_conv2 = DoubleConv(channels=(8, 8, 16), kernel=(5,5), pool_kernel=(2,2))
        self.double_conv3 = DoubleConv(channels=(16, 16, 32), kernel=(5,5), pool_kernel=(2,2))
        self.double_conv4 = DoubleConv(channels=(32, 32, 64), kernel=(5,5), pool_kernel=(3,3))


        self.layer1 = nn.Linear(192, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 3)        

    def forward(self, x):
        x = self.double_conv1(x)
        x = self.double_conv2(x)
        x = self.double_conv3(x)
        x = self.double_conv4(x)
        x = torch.flatten(x, x.dim()-3)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def optimize_model(optimizer, memory, policy_net, target_net):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    state_batch = torch.tensor(np.array([state for (state, _, _, _) in transitions]), dtype=torch.float32)
    action_batch = torch.tensor([[action] for (_, action, _, _,) in transitions])
    reward_batch =  torch.tensor([reward for (_, _, reward, _) in transitions])
    next_state_batch = torch.tensor(np.array([s2 for (_, _, _, s2) in transitions]), dtype=torch.float32)


    state_qvalues = policy_net(state_batch)
    state_action_values = state_qvalues.squeeze(1).gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        next_qvalues = target_net(next_state_batch).squeeze(1)
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

def play_game(optimizer, memory, policy_net, target_net, MAX_ITER=1000):
    env = gym.make('ALE/Skiing-v5')
    obs, info = env.reset()
    obs = np.swapaxes(np.swapaxes(obs, 1, 2), 0, 1)
    done = False
    game_start = time.time()
    action_sum, action_cnt = 0, 0
    while not done and action_cnt < MAX_ITER:
        s = time.time()
        optimize_model(optimizer, memory, policy_net, target_net)
        if random.random() < EPSILON:
            res = target_net(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
            action = np.argmax(res)
        else:
            action = int(random.random()*A)
        action_sum += (time.time()-s)
        action_cnt += 1
        observation, reward, done, trunc, info = env.step(action)
        observation = np.swapaxes(np.swapaxes(observation, 1, 2), 0, 1)
        memory.push((obs, action, reward, observation))
        obs = observation

    print(f"Game done in {time.time()-game_start}s, avg it length {action_sum/action_cnt}")

BATCH_SIZE = 128
GAMMA = 0.95
TAU = 0.005
LR = 0.0005
EPSILON = 0.8
A = 3
MODEL_PATH = "./"
MEMORY_PATH = "./memory"


target_net = DQN()
# target_net.load_state_dict(torch.load(MODEL_PATH+"target", weights_only=True, map_location=torch.device('cpu')))
policy_net = DQN()
policy_net.load_state_dict(target_net.state_dict())
# policy_net.load_state_dict(torch.load(MODEL_PATH+"policy", weights_only=True, map_location=torch.device('cpu')))



memory = Memory(10**5, file=MEMORY_PATH)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

while True:
    play_game(optimizer, memory, policy_net, target_net)
    torch.save(target_net.state_dict(), MODEL_PATH+"target")
    torch.save(policy_net.state_dict(), MODEL_PATH+"policy")
    with open(MEMORY_PATH, 'wb') as f:
        pickle.dump(memory.memory, f)