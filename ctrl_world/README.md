# StressDream / Ctrl-World (Manipulation)

> Steering [Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) robotic manipulation video world model.

---

## 🛠️ Installation

```bash
conda env create -f ctrl_world/environment.yml
conda activate stressdream-ctrlworld
```

> Python 3.11, PyTorch 2.9.1+cu128, Diffusers 0.36, Transformers 4.57. Requires NVIDIA driver ≥ 520 (CUDA 12.8 runtime).

- Requires ~20 GB VRAM for full optimization (Qwen3-VL reward + noise optimization).

---

## 📥 Checkpoints

Three checkpoints are required (~16.6 GB total):

| File | Source |
|------|--------|
| `ckpts/Ctrl-World/coffee_bag.pt` | [junwon-seo/StressDream](https://huggingface.co/junwon-seo/StressDream) |
| `ckpts/stable-video-diffusion-img2vid/` | [stabilityai/stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) |
| `ckpts/clip-vit-base-patch32/` | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |

```bash
pip install -U huggingface_hub          # provides the new `hf` CLI

mkdir -p ckpts/Ctrl-World
hf download junwon-seo/StressDream \
    coffee_bag.pt --local-dir ckpts/Ctrl-World

hf download stabilityai/stable-video-diffusion-img2vid \
    --local-dir ckpts/stable-video-diffusion-img2vid
hf download openai/clip-vit-base-patch32 \
    --local-dir ckpts/clip-vit-base-patch32
```

> Older `huggingface-cli` is deprecated — use `hf` (shipped with recent `huggingface_hub`).

Qwen3-VL-4B-Instruct (~8 GB) is fetched automatically by `transformers` on first run.

<p align="center">
  <img src="media/full_nominal.gif" width="49%">
  <img src="media/full_steered.gif" width="49%">
</p>
<p align="center"><em>Left: nominal (random noise) · Right: steered (Qwen3-VL reward)</em></p>

---

## 🚀 Demo

A sample trajectory (`example_data/traj_0001.hdf5`, candy-coffee task) is included. Run from the `StressDream/` root:

```bash
python ctrl_world/run_steering.py \
    --hdf5_path ctrl_world/example_data/traj_0001.hdf5
```

**Nominal imagination** (no noise optimization) — compare with the steered imagination:

```bash
python ctrl_world/generate_nominal.py \
    --hdf5_path ctrl_world/example_data/traj_0001.hdf5
```


Override the instruction or reward prompt:

```bash
python ctrl_world/run_steering.py \
    --hdf5_path ctrl_world/example_data/traj_0001.hdf5 \
    --instruction "put the coffee bag into the container without spilling" \
    --qwen_prompt "Is the coffee bean spilled? Respond Yes or No."
```

Outputs are written under `outputs/ctrl_world_nominal/<timestamp>/`.

---

## 📂 Outputs

`run_steering.py` writes to `outputs/ctrl_world_steering/<timestamp>/`:

| File | Description |
|------|-------------|
| `step_<n>_best.mp4` | Best multi-view rollout per interact step (GT prefix + generated) |
| `full_steered.mp4` | All steps concatenated |
| `history.json` | Per-iteration reward, noise norm, regularizer values |
| `optimized_noise.pt` | Final optimized noise tensor |

`generate_nominal.py` writes to `outputs/ctrl_world_nominal/<timestamp>/`:

| File | Description |
|------|-------------|
| `step_<n>_nominal.mp4` | Multi-view rollout per interact step (GT prefix + generated) |
| `full_nominal.mp4` | All steps concatenated |
| `history.json` | Per-step Qwen reward |

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
├── run_steering.py        # CLI entrypoint (steered imagination)
├── generate_nominal.py    # CLI entrypoint (nominal baseline)
├── steering_config.yaml   # hyperparameters
├── environment.yml        # conda env
├── wm_helpers.py          # HDF5 loader, forward, decode helpers
├── rewards.py             # Qwen3-VL reward
├── models/                # Ctrl-World UNet, pipeline, action adapter
├── dataset_meta_info/     # state percentile JSON for action normalization
├── example_data/          # bundled traj_0001.hdf5
└── ckpts/                 # symlinks (re-link if you move the repo)
```
