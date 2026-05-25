# Diffusion World Model Steering

Steering imaginations of a diffusion-based world model via noise optimization.
The core algorithm in `wm_steer/` is environment-agnostic; `dubins/` provides the Dubins car example.

## Structure

```
wm_steer/          # Core algorithm (model-agnostic)
  models/          # VAE, denoiser (UNet), world model, prediction heads
  steer.py         # optimize_noise() — DNO steering algorithm
  utils/
    logger.py

dubins/            # Dubins car example
  env.py
  dataset.py
  generate_dataset.py
  train_vae.py
  train_wm.py
  run_steering.py
  config.yaml
  README.md        # Setup and usage instructions
```

See `dubins/README.md` for setup and quick start.
