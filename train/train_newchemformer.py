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

    newchemformer = NewChemformer(args)

    t0 = time.time()
    newchemformer.fit()
    t_fit = time.time() - t0

    print(f"\nTraining complete, time: {round(t_fit, 2)} s")
    print("Done!")


if __name__ == "__main__":
    main()
