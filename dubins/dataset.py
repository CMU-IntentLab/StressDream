"""Dataset loader for Dubins car trajectories with continuous safety margin."""

import pickle
import h5py
import numpy as np
import torch
from typing import Dict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class DubinsDataset(Dataset):
    def __init__(self, dataset_path, sequence_length=10, split="train",
                 num_train_trajs=4800, obs_config=None):
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.split = split
        self.obs_config = obs_config or {'x': 0.0, 'y': 0.0, 'r': 0.25}

        dataset_path = Path(dataset_path)
        if dataset_path.suffix.lower() in ('.hdf5', '.h5'):
            self._load_hdf5()
        else:
            self._load_pickle()

        total_trajs = len(self.demos)
        val_size = max(1, int(0.05 * total_trajs))
        train_size = total_trajs - val_size
        self.demos = self.demos[:train_size] if split == "train" else self.demos[train_size:]

        self.valid_trajs = []
        self.traj_lengths = []
        for traj_idx, demo in enumerate(self.demos):
            traj_len = demo['length'] if self.is_hdf5 else len(demo['obs']['image'])
            if traj_len >= sequence_length:
                self.valid_trajs.append(traj_idx)
                self.traj_lengths.append(traj_len)

        print(f"Dataset ({split}): {len(self.valid_trajs)}/{len(self.demos)} valid trajectories")
        self.total_sequences = sum(t - sequence_length + 1 for t in self.traj_lengths)

    def _load_hdf5(self):
        self.is_hdf5 = True
        self.hdf5_file = h5py.File(self.dataset_path, 'r')
        traj_names = sorted(self.hdf5_file['images'].keys())
        self.demos = []
        for name in traj_names:
            meta = self.hdf5_file['metadata'][name].attrs
            self.demos.append({
                'length': meta['length'], 'image_shape': meta['image_shape'],
                'action_dim': meta['action_dim'], 'state_dim': meta['state_dim'],
                'traj_name': name,
            })
        print(f"Loaded HDF5 dataset with {len(self.demos)} trajectories")

    def _load_pickle(self):
        self.is_hdf5 = False
        self.hdf5_file = None
        with open(self.dataset_path, 'rb') as f:
            self.demos = pickle.load(f)
        print(f"Loaded pickle dataset with {len(self.demos)} trajectories")

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        traj_idx = np.random.choice(len(self.valid_trajs))
        actual_traj_idx = self.valid_trajs[traj_idx]
        demo = self.demos[actual_traj_idx]
        traj_len = self.traj_lengths[traj_idx]
        start_idx = np.random.randint(0, traj_len - self.sequence_length + 1)
        end_idx = start_idx + self.sequence_length

        if self.is_hdf5:
            name = demo['traj_name']
            images = self.hdf5_file['images'][name][start_idx:end_idx].astype(np.float32)
            images = images / 255.0 * 2 - 1
            actions = self.hdf5_file['actions'][name][start_idx:end_idx].astype(np.float32)
            states = self.hdf5_file['states'][name][start_idx:end_idx].astype(np.float32)
            dones = self.hdf5_file['dones'][name][start_idx:end_idx].astype(np.float32)
        else:
            images = np.array(demo['obs']['image'][start_idx:end_idx], dtype=np.float32)
            images = images.transpose(0, 3, 1, 2) / 255.0 * 2 - 1
            actions = np.array(demo['actions'][start_idx:end_idx], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[:, None]
            states = np.array(demo['obs'].get('priv_state', [[0, 0, 0]] * self.sequence_length)[start_idx:end_idx], dtype=np.float32)
            dones = np.array(demo['dones'][start_idx:end_idx], dtype=np.float32)

        # Continuous margin: obs_r^2 - dist^2 (positive = inside obstacle)
        ox = self.obs_config['x']
        oy = self.obs_config['y']
        r = self.obs_config['r']
        margins = np.array([
            r ** 2 - (states[i, 0] - ox) ** 2 - (states[i, 1] - oy) ** 2
            for i in range(self.sequence_length)
        ], dtype=np.float32)

        return {
            'images': torch.from_numpy(images),
            'actions': torch.from_numpy(actions),
            'states': torch.from_numpy(states),
            'dones': torch.from_numpy(dones),
            'margins': torch.from_numpy(margins),
        }

    def __del__(self):
        if hasattr(self, 'hdf5_file') and self.hdf5_file is not None:
            self.hdf5_file.close()


def create_dataloader(dataset_path, sequence_length=10, batch_size=32, split="train",
                      num_train_trajs=4800, num_workers=4, shuffle=True,
                      obs_config=None, pin_memory=True, prefetch_factor=2):
    dataset = DubinsDataset(
        dataset_path=dataset_path,
        sequence_length=sequence_length,
        split=split,
        num_train_trajs=num_train_trajs,
        obs_config=obs_config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
    )
