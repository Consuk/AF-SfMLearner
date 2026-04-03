from __future__ import absolute_import, division, print_function

from trainer_end_to_end import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    try:
        trainer.train()
    finally:
        if getattr(opts, "use_wandb", False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
