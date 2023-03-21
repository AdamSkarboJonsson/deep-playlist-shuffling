import numpy as np
from dataclasses import dataclass
import numpy.random as npr
import torch
from torch.utils.data import Dataset, DataLoader

energy_range = (1,10)
modern_range = (1,10)

device = torch.device('cpu')
torch.cuda.manual_seed(1337)
torch.random.manual_seed(1337)
npr.seed(1337)


class PlaylistDataset(Dataset):
    constant = "constant"
    must_vary = "must_vary"

    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return str(self.data)



def create_track_dict(artist_one_hot, energy = torch.tensor(0, device=device), modern = torch.tensor(0, device=device)):
    return {PlaylistDataset.constant: torch.stack([energy, modern]),
            PlaylistDataset.must_vary: artist_one_hot}


def create_pad_track(n_artists: int):
    return create_track_dict(artist_one_hot=torch.zeros(n_artists, device=device))


def create_track(n_artists: int, artist_id: int) -> dict[str, torch.tensor]:
    energy = torch.tensor(energy_range[0]+npr.random()*(energy_range[-1]-energy_range[0]), device=device)
    modern = torch.tensor(modern_range[0]+npr.random()*(modern_range[-1]-modern_range[0]), device=device)
    artist_one_hot = torch.zeros(n_artists, device=device)
    artist_one_hot[artist_id] = 1
    track = create_track_dict(artist_one_hot=artist_one_hot,energy=energy, modern=modern)
    return track


def create_artist_dist(n_tracks: int, n_artists: int) -> list[int]:
    artist_dist = [1]*n_artists
    tracks_left = n_tracks-n_artists
    for i in range(tracks_left):
        idx = npr.randint(low=0, high=n_artists)
        artist_dist[idx] += 1
    assert sum(artist_dist) == n_tracks
    return artist_dist


def create_playlist(maximum_playlist_length: int, n_tracks: int=512, n_artists: int=None) -> dict:
    if n_artists is None:
        n_artists = n_tracks//2
    assert n_tracks >= n_artists
    playlist = []
    artist_dist = create_artist_dist(n_tracks=n_tracks,
                                     n_artists=n_artists)
    # fill playlist with tracks
    for artist_id, n_tracks_by_artist in enumerate(artist_dist):
        playlist += [create_track(n_artists=n_artists, artist_id=artist_id) for _ in range(n_tracks_by_artist)]
    assert len(playlist) == n_tracks
    len_playlist = len(playlist)
    # pad if necessary
    diff = maximum_playlist_length-len_playlist
    playlist += [create_pad_track(n_artists=n_artists) for _ in range(diff)]
    # create mask
    mask = [True]*maximum_playlist_length
    mask[:n_tracks] = [False]*n_tracks
    mask = torch.tensor(mask, device=device)
    playlist_dict = {}
    for field in [PlaylistDataset.constant, PlaylistDataset.must_vary]:
        d = {}
        l = [t[field] for t in playlist]
        d[field] = torch.stack(l)
        playlist_dict.update(d)
    playlist_dict["n"] = torch.tensor(n_tracks, device=device)
    playlist_dict["mask"] = mask
    return playlist_dict

def create_playlist_dataset(n_playlists: int = 1, maximum_playlist_length=1024) -> PlaylistDataset:
    if n_playlists == 1:
        return PlaylistDataset(data=[create_playlist(maximum_playlist_length=maximum_playlist_length)])
    else:
        raise NotImplementedError("Does not support more than 1 playlist yet lmao")


if __name__ == "__main__":
    playlist_dataset = create_playlist_dataset()
    print(playlist_dataset)


