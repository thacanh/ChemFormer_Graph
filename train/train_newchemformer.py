"""
Simple training script cho NewChemformer
Theo style molbart nhưng có debug geometry
"""
import time
import hydra
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import molbart.utils.data_utils as util
from molbart.models.newchemformer import NewChemformer


@hydra.main(version_base=None, config_path="config", config_name="train_newchemformer")
def main(args):
    util.seed_everything(args.seed)

    print("Training NewChemformer with Geometry Embeddings")


    # BUILD MODEL
    newchemformer = NewChemformer(args)

    # DEBUG BATCH
    dm = newchemformer.datamodule
    dm.setup()

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    print("")

    # TRAIN
    t0 = time.time()
    newchemformer.fit()
    t_fit = time.time() - t0

    print("\nTraining complete, time:", round(t_fit, 2), "s")

    # CHECK GRAD
    if hasattr(newchemformer, "geom_proj"):
        grad = newchemformer.geom_proj.weight.grad
        if grad is not None:
            print("geom_proj grad mean:", grad.abs().mean().item())
        else:
            print("geom_proj grad = None (chưa backward?)")

    print("Done!")


if __name__ == "__main__":
    main()
