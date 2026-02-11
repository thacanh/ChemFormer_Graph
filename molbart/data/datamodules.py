import numpy as np
import pandas as pd
import torch

from molbart.data.base import ReactionListDataModule


class SynthesisDataModule(ReactionListDataModule):

    datamodule_name = "synthesis"

    def __init__(
        self,
        dataset_path: str,
        geom_embeddings_path=None,
        **kwargs,
    ):
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.geom_embeddings_path = geom_embeddings_path
        self.geom_embeddings = None

    # --------------------------------------------------
    def _load_all_data(self):

        df = pd.read_csv(self.dataset_path).reset_index(drop=True)
        df["index"] = np.arange(len(df))

        self._all_data = {
            "reactants": df["reactants"].tolist(),
            "products": df["products"].tolist(),
            "index": df["index"].tolist(),
        }

        # LOAD GEOM
        if self.geom_embeddings_path is not None:
            self.geom_embeddings = np.load(self.geom_embeddings_path)
            print("Loaded geometry embeddings:", self.geom_embeddings.shape)

        self._set_split_indices_from_dataframe(df)

    # --------------------------------------------------
    def _collate(self, batch, train=True):

        batch_dict = super()._collate(batch, train)

        if "index" in batch[0]:
            batch_dict["index"] = torch.tensor([item["index"] for item in batch])

        if self.geom_embeddings is not None:
            idxs = batch_dict["index"]
            geom = torch.from_numpy(self.geom_embeddings[idxs]).float()
            batch_dict["geom"] = geom

        return batch_dict
