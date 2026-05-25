"""
Generate training dataset for the Dubins car world model.

Generates random trajectories with "naughty" action reversals for diversity
and saves them as an HDF5 file.

Usage:
    python dubins/generate_dataset.py --save_path dubins/checkpoints/train_data \
        --num_trajs 4000 --traj_length 100
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pathlib
import random
import h5py
import numpy as np
import torch
from tqdm import tqdm

from dubins.env import DubinsConfig, get_frame, get_init_state


def gen_one_trajectory(config, traj_length, naughty_prob=0.4):
    """
    Generate one trajectory with "naughty" action reversals.

    Args:
        config: DubinsConfig
        traj_length: Max number of steps
        naughty_prob: Probability of reversing each action

    Returns:
        (actions, states, images, dones) as lists
    """
    state = get_init_state(config)
    images, states, actions, dones = [], [], [], []

    u_max = config.turnRate
    dt = config.dt
    v = config.speed

    for t in range(traj_length):
        ac = torch.rand(1) * 2 * u_max - u_max
        actual_ac = -ac if random.random() < naughty_prob else ac

        states_next = torch.zeros(3)
        states_next[0] = state[0] + v * dt * torch.cos(state[2])
        states_next[1] = state[1] + v * dt * torch.sin(state[2])
        states_next[2] = state[2] + dt * actual_ac

        states.append(state.numpy())
        actions.append(ac)
        images.append(get_frame(state, config))

        out_of_bounds = (abs(state[0]) > config.x_max - config.buffer or
                         abs(state[1]) > config.y_max - config.buffer)
        done = 1 if (t == traj_length - 1 or out_of_bounds) else 0
        dones.append(done)

        state = states_next
        if done:
            break

    return actions, states, images, dones


def generate_dataset(args):
    config = DubinsConfig()

    output_path = pathlib.Path(args.save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdf5_path = f"{args.save_path}_{args.num_trajs}_{config.size[0]}.hdf5"

    print(f"Generating {args.num_trajs} trajectories -> {hdf5_path}")

    with h5py.File(hdf5_path, 'w') as hf:
        images_group = hf.create_group('images')
        actions_group = hf.create_group('actions')
        states_group = hf.create_group('states')
        dones_group = hf.create_group('dones')

        hf.attrs['num_trajectories'] = args.num_trajs
        hf.attrs['dataset_format'] = 'dubins_car_trajectories'
        hf.attrs['image_size'] = config.size
        hf.attrs['data_length'] = args.traj_length

        for i in tqdm(range(args.num_trajs), desc="Generating"):
            acs, state_gt, img_obs, done_flags = gen_one_trajectory(
                config, args.traj_length, args.naughty_prob
            )

            name = f'traj_{i:06d}'

            imgs = np.array(img_obs, dtype=np.float32)
            if imgs.ndim == 4 and imgs.shape[-1] == 3:
                imgs = imgs.transpose(0, 3, 1, 2)
            images_group.create_dataset(name, data=imgs, compression='gzip')

            acs_arr = np.array([a.numpy() if torch.is_tensor(a) else a for a in acs], dtype=np.float32)
            if acs_arr.ndim == 1:
                acs_arr = acs_arr[:, None]
            actions_group.create_dataset(name, data=acs_arr, compression='gzip')

            states_group.create_dataset(name, data=np.array(state_gt, dtype=np.float32), compression='gzip')
            dones_group.create_dataset(name, data=np.array(done_flags, dtype=np.float32), compression='gzip')

            meta = hf.create_group(f'metadata/{name}')
            meta.attrs['length'] = len(state_gt)
            meta.attrs['image_shape'] = imgs.shape[1:]
            meta.attrs['action_dim'] = acs_arr.shape[1]
            meta.attrs['state_dim'] = 3

    print(f"\nSaved {args.num_trajs} trajectories to {hdf5_path}")
    print(f"File size: {os.path.getsize(hdf5_path) / (1024 * 1024):.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Dubins car training dataset")
    parser.add_argument('--save_path', type=str, default='dubins/checkpoints/train_data',
                        help='Output path prefix (without extension)')
    parser.add_argument('--num_trajs', type=int, default=4000)
    parser.add_argument('--traj_length', type=int, default=100)
    parser.add_argument('--naughty_prob', type=float, default=0.4,
                        help='Probability of reversing each action')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    generate_dataset(args)
