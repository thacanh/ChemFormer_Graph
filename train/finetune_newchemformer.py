"""Fine-tune NewChemformer from pretrained checkpoint."""

import time
import hydra
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import molbart.utils.data_utils as util
from molbart.models.newchemformer import NewChemformer


@hydra.main(version_base=None, config_path="config", config_name="finetune_newchemformer")
def main(args):
    util.seed_everything(args.seed)

    print("Fine-tuning NewChemformer with Geometry Embeddings")
    print(f"Checkpoint: {args.model_path}")

    newchemformer = NewChemformer(args)

    t0 = time.time()
    newchemformer.fit()
    t_fit = time.time() - t0

    print(f"\nFine-tuning complete, time: {round(t_fit, 2)} s")
    print("Done!")


if __name__ == "__main__":
    main()
