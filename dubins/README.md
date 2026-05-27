# StressDream / Dubins Car

> Steering image-based 3D Dubins car world model.

<p align="center">
  <video src="media/steering_pessimistic.mp4" autoplay loop muted playsinline width="100%"></video>
</p>
<p align="center"><em>Left: nominal rollout (no optimization) · Right: pessimistic steering into the failure set</em></p>

---

## 🛠️ Installation

```bash
conda create -n stressdream-dubins python=3.10
conda activate stressdream-dubins
pip install -r dubins/requirements.txt
```

All commands are run from the `StressDream/` root.

---

## 🚀 Quick Start

### Interactive notebook *(recommended)*

The fastest way to explore steering is the self-contained notebook with inline visualizations:

```bash
jupyter notebook dubins/demo.ipynb
```

### CLI

Pretrained checkpoints are in `dubins/checkpoints/`. The default run uses `dubins/example_traj/` as the initial condition.

```bash
# Nominal baseline: 5 random rollouts without optimization
python dubins/generate_nominal.py

# Pessimistic: steer toward failure (default)
python dubins/run_steering.py

# Optimistic: steer toward safety
python dubins/run_steering.py --mode optimistic

# Custom initial state / actions
python dubins/run_steering.py --init_state -1.2 -0.7 0.7 --actions ones --traj_length 30
python dubins/run_steering.py --actions path/to/actions.npy
```

Outputs are written to `steering_results/`: `steering_<mode>.mp4` (steered) and `nominal.mp4` (unoptimized samples).

---

## ⚙️ Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--init_state X Y THETA` | `example_traj/init_state.npy` | Initial robot state |
| `--actions` | `example_traj/actions.npy` | `ones`, `random`, `zeros`, or path to `.npy` |
| `--traj_length` | inferred | Number of rollout steps |
| `--mode` | `pessimistic` | `pessimistic` or `optimistic` |
| `--iters` | from config | Optimization iterations per step |
| `--vae_checkpoint` | from config | Override VAE checkpoint |
| `--wm_checkpoint` | from config | Override world model checkpoint |
| `--output_dir` | `steering_results/` | Output directory |
| `--seed` | `42` | Random seed |

Checkpoint paths default to `dubins/config.yaml` (`checkpoints.vae` / `checkpoints.world_model`). Steering hyperparameters (regularizer coefficients, grad mode, etc.) live in `dubins/steering_config.yaml`.

---

## 🏗️ Full Training Pipeline

<details>
<summary>Click to expand</summary>

### 1. Generate data

There are pretrained checkpoints available in `dubins/checkpoints/`. If you want to train the world model from scratch, you can follow the instructions below.

```bash
python dubins/generate_dataset.py \
    --num_trajs 4000 --traj_length 100 \
    --naughty_prob 0.4 \
    --save_path dubins/checkpoints/train_data.hdf5
```

Update `data.dataset_path` in `dubins/config.yaml`.

### 2. Train VAE

```bash
python dubins/train_vae.py --config dubins/config.yaml
```

Checkpoints saved to `dubins/checkpoints/vae/<timestamp>/`.

### 3. Train world model

```bash
python dubins/train_wm.py --config dubins/config.yaml \
    --vae_checkpoint dubins/checkpoints/vae/<timestamp>/vae_final.pt
```

### 4. Run steering

Update `checkpoints.vae` and `checkpoints.world_model` in `dubins/config.yaml`, then run as above.

</details>
