# Dubins Car — Diffusion WM Steering

Steering the imagined future of a diffusion-based world model for the Dubins car environment.

## Environment Setup

```bash
conda create -n steering_dubins python=3.10
conda activate steering_dubins
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install h5py numpy matplotlib pillow tqdm imageio wandb einops ruamel.yaml
```

All scripts are run from the base directory (the parent of `dubins/`).

## Quick Start (Pretrained Models)

Pretrained checkpoints are in `dubins/checkpoints/`. The default run uses
`dubins/example_traj/` (a ground-truth failure trajectory) as the initial condition.

```bash
conda activate steering_dubins

# Pessimistic: steer toward failure (default)
python dubins/run_steering.py

# Optimistic: steer toward safety
python dubins/run_steering.py --mode optimistic

# Custom initial state and actions
python dubins/run_steering.py --init_state -1.2 -0.7 0.7 --actions ones --traj_length 30
python dubins/run_steering.py --actions path/to/actions.npy
```

Output: side-by-side video saved to `steering_results/steering_<mode>_<grad_mode>.mp4`.

Checkpoint paths are set in `dubins/config.yaml` (`checkpoints.vae` and `checkpoints.world_model`)
and can be overridden with `--vae_checkpoint` / `--wm_checkpoint`.

## Full Pipeline

### 1. Generate training data

```bash
python dubins/generate_dataset.py \
    --num_trajs 4000 --traj_length 100 \
    --naughty_prob 0.4 \
    --save_path dubins/checkpoints/train_data.hdf5
```

Update `data.dataset_path` in `dubins/config.yaml` to point to the generated file.

### 2. Train VAE + prediction heads

```bash
python dubins/train_vae.py --config dubins/config.yaml
```

Checkpoints are saved to `dubins/checkpoints/vae/<timestamp>/`.

### 3. Train world model (denoiser)

```bash
python dubins/train_wm.py --config dubins/config.yaml \
    --vae_checkpoint dubins/checkpoints/vae/<timestamp>/vae_final.pt
```

### 4. Run steering with trained models

Update `checkpoints.vae` and `checkpoints.world_model` in `dubins/config.yaml`, then:

```bash
python dubins/run_steering.py --mode pessimistic
```

## Key Arguments for `run_steering.py`

| Argument | Default | Description |
|---|---|---|
| `--init_state X Y THETA` | `example_traj/init_state.npy` | Initial robot state |
| `--actions` | `example_traj/actions.npy` | `ones`, `random`, `zeros`, or path to `.npy` |
| `--traj_length` | inferred from actions | Number of steps to roll out |
| `--mode` | `pessimistic` | `pessimistic` (toward failure) or `optimistic` (toward safety) |
| `--iters` | from `steering_config.yaml` | Optimization iterations per step |
| `--vae_checkpoint` | from `config.yaml` | Override VAE checkpoint path |
| `--wm_checkpoint` | from `config.yaml` | Override world model checkpoint path |
| `--output_dir` | `steering_results/` | Where to save the output video |
| `--seed` | `42` | Random seed |

Steering hyperparameters (regularizer coefficients, grad mode, etc.) live in `dubins/steering_config.yaml`.
