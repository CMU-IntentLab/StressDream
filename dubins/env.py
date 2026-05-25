"""Dubins car environment: dynamics, rendering, and helpers."""

import io
import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


class DubinsConfig:
    turnRate = 1.25
    speed = 1.0
    dt = 0.05
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    obs_x, obs_y, obs_r = 0.0, 0.0, 0.25
    buffer = 0.1
    size = (128, 128)


def dubins_step(state, action, config=None):
    """
    Dubins car dynamics: state = [x, y, theta], action = angular velocity.
    Returns next state as torch tensor.
    """
    if config is None:
        config = DubinsConfig()
    ns = torch.zeros(3)
    ns[0] = state[0] + config.speed * config.dt * torch.cos(state[2])
    ns[1] = state[1] + config.speed * config.dt * torch.sin(state[2])
    ns[2] = state[2] + config.dt * action
    return ns


def check_failure(state, config=None):
    """Returns True if state is inside the obstacle."""
    if config is None:
        config = DubinsConfig()
    dist = np.linalg.norm(np.array([state[0].item(), state[1].item()]) - np.array([config.obs_x, config.obs_y]))
    return dist < config.obs_r


def get_frame(state, config=None):
    """Render Dubins car state to a (H, W, 3) uint8 numpy image."""
    if config is None:
        config = DubinsConfig()
    fig, ax = plt.subplots()
    plt.xlim([config.x_min, config.x_max])
    plt.ylim([config.y_min, config.y_max])
    plt.axis('off')
    fig.set_size_inches(1, 1)

    circle = patches.Circle([config.obs_x, config.obs_y], config.obs_r,
                             edgecolor="#b3b3b3", facecolor="#b3b3b3", linewidth=2)
    ax.add_patch(circle)

    dt, v = config.dt, config.speed
    plt.quiver(state[0], state[1],
               dt * v * torch.cos(state[2]), dt * v * torch.sin(state[2]),
               angles='xy', scale_units='xy', minlength=0,
               width=0.1, scale=0.15, color='black', zorder=3)
    plt.scatter(state[0], state[1], s=20, color='black', zorder=3)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=config.size[0])
    buf.seek(0)
    img = np.array(Image.open(buf).convert('RGB'))
    plt.close(fig)
    return img


def get_init_state(config=None):
    """Sample a random initial state outside the obstacle."""
    if config is None:
        config = DubinsConfig()
    state = torch.zeros(3)
    while np.linalg.norm(state[:2].numpy() - np.array([config.obs_x, config.obs_y])) < config.obs_r:
        state = torch.rand(3)
        state[0] = state[0] * (config.x_max - config.buffer - config.x_min - config.buffer) + config.x_min + config.buffer
        state[1] = state[1] * (config.y_max - config.buffer - config.y_min - config.buffer) + config.y_min + config.buffer
    state[2] = torch.atan2(-state[1], -state[0]) + np.random.normal(0, 1)
    state[2] = state[2] % (2 * np.pi)
    return state


def simulate_trajectory(initial_state, actions, config=None):
    """Simulate trajectory and return (states, images)."""
    if config is None:
        config = DubinsConfig()
    states = [initial_state.clone()]
    images = [get_frame(initial_state, config)]
    state = initial_state.clone()
    for action in actions:
        state = dubins_step(state, action, config)
        states.append(state.clone())
        images.append(get_frame(state, config))
    return torch.stack(states), images


def compute_gt_margins(states, config=None):
    """Compute ground truth margins: obs_r^2 - dist^2 (positive = inside obstacle = unsafe)."""
    if config is None:
        config = DubinsConfig()
    margins = []
    for s in states:
        m = (config.obs_r ** 2
             - (s[0].item() - config.obs_x) ** 2
             - (s[1].item() - config.obs_y) ** 2)
        margins.append(m)
    return np.array(margins)
