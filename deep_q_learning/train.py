from collections import namedtuple, deque
from torch import Tensor, tensor
import torch.nn as nn
import torch
import random
from deep_shuffling.deep_q_learning.env import DataPreparer, PlaylistGame, RewardFunction

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda")
class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ShuffleNet(nn.Module):
    def __init__(self, n_tracks: int, n_features: int, n_embed: int):
        super().__init__()
        # B, n_tracks, n_features, 1
        self.linear_2d = nn.Sequential(
            nn.Conv2d(in_channels=1,
                       out_channels=n_embed,
                       kernel_size=(1,1),
                       stride=1,
                       padding=0,
                       device=device),
        # B, n_tracks, n_features, n_embed
            nn.Conv2d(in_channels=n_embed,
                      out_channels=n_embed,
                      kernel_size=(1,n_features),
                      stride=1,
                      padding=0,
                      device=device),
        # B, n_tracks, 1, n_embed
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=n_tracks*n_embed,
                      out_features=n_tracks*2, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=n_tracks*2, out_features=n_tracks, bias=False),
        )

    def forward(self, state: Tensor) -> Tensor:
        # state (B, n_tracks, n_features)
        state = torch.unsqueeze(state, dim=-1)
        # state (B, n_tracks, n_features, 1)
        values = self.linear_2d(state)
        values = torch.squeeze(values)
        # values (B, n_tracks)
        return values

    def pick_best_action(self, values: Tensor):
        # values (B, n_tracks)
        # best action are the indicies of the two largest output elements
        sorted_indicies = torch.argsort(input=values, dim=-1, descending=True)
        action = sorted_indicies[:,:2]
        # action (B, 2)
        return action

    def epsilon_greedy(self, state: Tensor):
        




class TrainingSession:
    def __init__(self, memory_size: int = 10000):
        self.replay_memory = ReplayMemory(capacity=memory_size)
    def train(self,
              iterations: int,
              max_number_of_moves: int,
              discount_factor: float = 0.95,
              learning_rate: float = 0.001,
              batch_size: int = 128,
              EPS_START: float = 0.9,
              EPS_END: float = 0.05,
              EPS_DECAY: float = 1000
              ):


