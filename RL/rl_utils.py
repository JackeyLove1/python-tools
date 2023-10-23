import random

from tqdm import tqdm
import numpy as np
import torch
import collections
from typing import Dict, List, Any, Optional, Tuple

from dataclasses import dataclass
from transformers import set_seed

set_seed(0)


@dataclass
class Args:
    dtype = torch.float
    capacity = 10000
    batch_size = 100
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class Unit:
    def __init__(self, state: List[float], action: int, reward: int, next_state: List[float], done: bool):
        self.state = torch.tensor(state, dtype=Args.dtype)
        self.action = action
        self.reward = reward
        self.next_state = torch.tensor(next_state, dtype=Args.dtype)
        self.done = done


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=Args.capacity)

    def add(self, unit: Unit):
        self.buffer.append(unit)

    def sample(self, batch_size=Args.batch_size):
        assert len(self.buffer) >= batch_size
        units = random.sample(self.buffer, batch_size)
        return units

    def size(self):
        return len(self.buffer)

    def __len__(self):
        return self.size()

def train_on_policy_agent():
    pass