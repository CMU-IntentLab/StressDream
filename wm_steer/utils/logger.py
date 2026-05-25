from typing import Dict, Any
from pathlib import Path


class Logger:
    def __init__(self, log_dir=None, use_wandb=True, log_file=None):
        self.use_wandb = use_wandb
        self.step = 0
        self.log_file = log_file

        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_run = None
            except ImportError:
                print("Warning: wandb not installed.")
                self.use_wandb = False
                self.wandb = None

        if log_dir is not None:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                from datetime import datetime
                f.write(f"Training log started at {datetime.now()}\n" + "=" * 80 + "\n\n")

    def init_wandb(self, project, name, config):
        if self.use_wandb and self.wandb is not None:
            self.wandb_run = self.wandb.init(project=project, name=name, config=config, resume="allow")
            print(f"Wandb run initialized: {self.wandb_run.url}")

    def log(self, metrics, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
        if self.use_wandb and hasattr(self, 'wandb_run') and self.wandb_run is not None:
            self.wandb.log(metrics, step=self.step)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                from datetime import datetime
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Step {self.step}] {metrics}\n")

    def log_images(self, images_dict, step=None):
        if step is not None:
            self.step = step
        if self.use_wandb and hasattr(self, 'wandb_run') and self.wandb_run is not None:
            self.wandb.log(images_dict, step=self.step)

    def print(self, message):
        print(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                from datetime import datetime
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def finish(self):
        if self.use_wandb and hasattr(self, 'wandb_run') and self.wandb_run is not None:
            self.wandb.finish()
