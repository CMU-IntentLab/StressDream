# StressDream / Ctrl-World (Manipulation)

> Steering [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) robotic manipulation video world model.

---

## 🛠️ Installation

```bash
conda env create -f ctrl_world/environment.yml
conda activate stressdream-ctrlworld
```

Python 3.11 with PyTorch ≥ 2.7, Diffusers, Transformers, and HDF5/imaging helpers.

---

## 📥 Checkpoints

Three checkpoints are required (~16.6 GB total):

| File | Source |
|------|--------|
| `ckpts/Ctrl-World/coffee_bag.pt` | [junwon-seo/StressDream](https://huggingface.co/junwon-seo/StressDream) |
| `ckpts/stable-video-diffusion-img2vid/` | [stabilityai/stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) |
| `ckpts/clip-vit-base-patch32/` | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |

```bash
pip install -U huggingface_hub

mkdir -p ckpts/Ctrl-World
huggingface-cli download junwon-seo/StressDream \
    coffee_bag.pt --local-dir ckpts/Ctrl-World

huggingface-cli download stabilityai/stable-video-diffusion-img2vid \
    --local-dir ckpts/stable-video-diffusion-img2vid
huggingface-cli download openai/clip-vit-base-patch32 \
    --local-dir ckpts/clip-vit-base-patch32
```

Qwen3-VL-4B-Instruct (~8 GB) is fetched automatically by `transformers` on first run.

---

## 🚀 Demo

A sample trajectory (`example_data/traj_0001.hdf5`, candy-coffee task) is included. Run from the `StressDream/` root:

```bash
python ctrl_world/run_steering.py \
    --hdf5_path ctrl_world/example_data/traj_0001.hdf5
```

Override the instruction or reward prompt:

```bash
python ctrl_world/run_steering.py \
    --hdf5_path ctrl_world/example_data/traj_0001.hdf5 \
    --instruction "put the coffee bag into the container without spilling" \
    --qwen_prompt "Is the coffee bean spilled? Respond Yes or No."
```

---

## 📂 Outputs

Each run writes to `outputs/ctrl_world_steering/<timestamp>/`:

| File | Description |
|------|-------------|
| `step_<n>_best.mp4` | Best multi-view rollout per interact step |
| `overlay_reward.mp4` | Concatenated rollout with Qwen p(Yes) curve |
| `history.json` | Per-iteration reward, noise norm, regularizer values |
| `optimized_noise.pt` | Final optimized noise tensor |

---

## ⚙️ Configuration

Hyperparameters live in `steering_config.yaml`. Key groups:

| Group | Controls |
|-------|----------|
| `task` | Instruction, Qwen prompt, success/failure target |
| `model` | Frames per rollout window |
| `hdf5` | Camera order, rgb_skip, crop, start window |
| `qwen` | Model name, frames per query |
| `optim` | Iters, lr, grad scale/clamp, noise-norm threshold |
| `regularizer` | Spectral / std-permutation / KL-spherical coefficients |
| `output` | Save dir, fps |

CLI flags override YAML values.

---

## 📁 Layout

```
ctrl_world/
├── run_steering.py        # CLI entrypoint
├── steering_config.yaml   # hyperparameters
├── environment.yml        # conda env
├── wm_helpers.py          # HDF5 loader, forward, decode helpers
├── rewards.py             # Qwen3-VL reward
├── models/                # Ctrl-World UNet, pipeline, action adapter
├── dataset_meta_info/     # state percentile JSON for action normalization
├── example_data/          # bundled traj_0001.hdf5
└── ckpts/                 # symlinks (re-link if you move the repo)
```
