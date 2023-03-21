import torch
from torch import Tensor, tensor
import torch.nn as nn
import numpy as np
from dataset import create_playlist, PlaylistDataset
import os
# state space: Playlist (n*d)
device = torch.device("cpu")
maximum_playlist_length = 4
n_tracks = 4
feature_types = [PlaylistDataset.constant]
n_features_per_type = 1


class DataPreparer:
    def __init__(self,
                 feature_types: list[str],
                 n_features_per_type: int,
                 maximum_playlist_length: int,
                 n_tracks: int,
                 n_artists: int = None):
        self.feature_types = feature_types
        self.n_features_per_type = n_features_per_type
        self.maximum_playlist_length = maximum_playlist_length
        self.n_tracks = n_tracks
        self.n_artists = n_artists

    def random_playlist_tensor(self) -> Tensor:
        playlist: dict = create_playlist(maximum_playlist_length=self.maximum_playlist_length,
                                         n_tracks=self.n_tracks,
                                         n_artists=self.n_artists)
        playlist_tensor: Tensor = torch.stack(
            tensors=[playlist[feature_type][:, :n_features_per_type] for feature_type in feature_types])
        return playlist_tensor


# only works for constant features atm
class RewardFunction:
    def __init__(self):
        self.avg_pooling = torch.nn.AvgPool2d(kernel_size=(3, 1))

    def get_state_score(self, state: Tensor) -> Tensor:
        avg_feats = torch.sum(state, dim=-2)
        shifted_features = torch.roll(state, -1, -2)
        pooling = self.avg_pooling(state)
        noise_squared_diff = (state[:, :-1, :] - shifted_features[:, :-1, :]) ** 2
        noise_loss = torch.sum(noise_squared_diff) ** 0.5
        pooling_squared_diff = (pooling - avg_feats) ** 2
        global_level_loss = torch.sum(pooling_squared_diff)
        state_score = -(noise_loss + global_level_loss)
        return state_score

    def get_reward(self, state: Tensor, action: Tensor, new_state: Tensor):
        current_state_score = self.get_state_score(state)
        new_state_score = self.get_state_score(new_state)
        reward = new_state_score - current_state_score
        reward = torch.reshape(reward, shape=(1, 1))
        return reward


class PlaylistGame:
    def __init__(self, dp: DataPreparer, reward_function: RewardFunction):
        self.dp = dp
        self.reward_function = reward_function

    def get_initial_state(self) -> Tensor:
        state = self.dp.random_playlist_tensor()
        return state

    def get_new_state_reward(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):  # new_state, reward
        # an action is swapping of two tracks, requiring a tuple of integers (i,j),
        # meaning that track at i will swap place with track at j. This can be easily be described by a permutation matrix
        assert action.shape == (1, 2)
        assert state.shape == (1, self.dp.n_tracks, self.dp.n_features_per_type)
        i, j = action[0, :]
        permutation_matrix = torch.eye(self.dp.n_tracks, device=device)
        permutation_matrix[i, i] = 0
        permutation_matrix[i, j] = 1
        permutation_matrix[j, j] = 0
        permutation_matrix[j, i] = 1
        state_batchless = state[0, :, :]
        new_state_batchless = permutation_matrix @ state_batchless
        new_state = torch.unsqueeze(new_state_batchless, dim=0)
        assert new_state.shape == (1, self.dp.n_tracks, self.dp.n_features_per_type)
        reward = self.get_reward(state=state, action=action, new_state=new_state)
        return new_state, reward

    def get_reward(self, state: Tensor, action: Tensor, new_state: Tensor):
        reward = self.reward_function.get_reward(state=state,
                                                 action=action,
                                                 new_state=new_state)

        return reward


if __name__ == "__main__":

    dp = DataPreparer(feature_types=feature_types,
                      n_features_per_type=n_features_per_type,
                      maximum_playlist_length=maximum_playlist_length,
                      n_tracks=n_tracks)

    rf = RewardFunction()

    random_playlist_tensor = dp.random_playlist_tensor()
    print(random_playlist_tensor)

    playlist_game = PlaylistGame(dp=dp, reward_function=rf)

    initial_state = playlist_game.get_initial_state()
    print(initial_state)

    test_action = tensor([[0, 3]], device=device)
    new_state, reward = playlist_game.get_new_state_reward(state=initial_state, action=test_action)
    print(new_state)
