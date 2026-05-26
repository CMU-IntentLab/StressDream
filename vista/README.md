# StressDream / Vista (Driving)

> Steering [Vista](https://github.com/OpenDriveLab/Vista) driving video world model.

---

## 🛠️ Installation

```bash
conda env create -f vista/environment.yml
conda activate stressdream-vista
```

- Python 3.10 with PyTorch and Vista dependencies (`omegaconf`, `pytorch-lightning`, `transformers`, etc.).
- Requires ~80GB of VRAM for running the full optimization (with Qwen2.5-VL reward), and ~40GB of VRAM for sampling without the optimization.

---

## 📥 Checkpoints

| File | Size | Source |
|------|------|--------|
| `ckpts/vista.safetensors` | ~10 GB | [OpenDriveLab/Vista](https://huggingface.co/OpenDriveLab/Vista) |

```bash
mkdir -p vista/ckpts
wget -O vista/ckpts/vista.safetensors \
    https://huggingface.co/OpenDriveLab/Vista/resolve/main/vista.safetensors
```

The X-CLIP reward (`microsoft/xclip-base-patch32`, ~400 MB) and optional Qwen2.5-VL are fetched automatically by `transformers` on first run.

---

## 🚀 Demo

A sample driving image (`example_images/truck_merge.jpg`) and default X-CLIP prompts are bundled. Run from the `StressDream/` root:

```bash
python vista/run_steering.py
```

Override the image or prompts on the CLI. `--prompts` accepts a comma-separated list or a path to a JSON list; `--target_idx` selects which prompt to maximize:

```bash
python vista/run_steering.py \
    --image_path path/to/scene.jpg \
    --prompts "the lead vehicle is far away,the lead vehicle is close,the lead vehicle is at the same distance" \
    --target_idx 0
```

**Trajectory conditioning** (optional)

```bash
python vista/run_steering.py --trajectory_json path/to/traj.json
```

`traj.json` is a flat list of `{"x": float, "y": float}` waypoints; the first two are treated as origin/calibration and skipped.

**Qwen2.5-VL reward** (optional)

```bash
python vista/run_steering.py --use_qwen
```

---

## 📂 Outputs

Each run writes to `outputs/vista_steering/<timestamp>/`:

| File | Description |
|------|-------------|
| `iter_<n>.mp4` | 25-frame rollout per iteration |
| `history.json` | Per-iter X-CLIP probs, noise norm, regularizer values |
| `optimized_noise.pt` | Final optimized noise tensor |

---

## ⚙️ Configuration

Hyperparameters live in `steering_config.yaml`. Key groups:

| Group | Controls |
|-------|----------|
| `model` | Vista config, checkpoint path, frame count, CFG scale |
| `vlm` | X-CLIP model id, frame-sampling mode, default prompts |
| `qwen` | Model name, prompt, weight, frames per query |
| `optim` | Iters, lr, grad scale/clamp, max grad norm |
| `regularizer` | Spectral / std-permutation / KL-spherical coefficients |
| `output` | Save dir, save frequency, fps |

CLI flags override YAML values.

---

## 📁 Layout

```
vista/
├── run_steering.py        # CLI entrypoint
├── steering_config.yaml   # hyperparameters
├── environment.yml        # conda env
├── wm_helpers.py          # Vista model, sampler, conditioning helpers
├── rewards.py             # X-CLIP + Qwen2.5-VL reward
├── vwm/                   # Vista core model package
├── configs/inference/vista.yaml
├── example_images/        # bundled truck_merge.jpg
└── ckpts/                 # vista.safetensors
```
