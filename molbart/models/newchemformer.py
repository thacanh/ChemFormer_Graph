"""NewChemformer with FiLM-based geometry conditioning."""

import torch
import torch.nn as nn
import numpy as np
from molbart.models.chemformer import Chemformer


class NewChemformer(Chemformer):

    def __init__(self, config):
        self.geom_table = None
        self.geom_dim = None

        if getattr(config, "geom_embeddings_path", None):
            print("Loading geometry:", config.geom_embeddings_path)
            self.geom_table = np.load(config.geom_embeddings_path)
            self.geom_dim = self.geom_table.shape[1]
            print("Geom shape:", self.geom_table.shape)

        super().__init__(config)

        if self.geom_table is not None:
            if hasattr(self, "datamodule") and self.datamodule is not None:
                self.datamodule.geom_embeddings_path = config.get("geom_embeddings_path")

    def build_model(self, args):
        super().build_model(args)

        if self.geom_table is None:
            print("Geometry OFF")
            return

        print("Geometry ON (FiLM encoder memory)")

        d_model = args.d_model

        self.gamma_net = nn.Linear(self.geom_dim, d_model)
        self.beta_net = nn.Linear(self.geom_dim, d_model)
        self.scale = nn.Parameter(torch.tensor(-2.0))

        model = self.model
        old_forward = model.forward

        gamma_net = self.gamma_net
        beta_net = self.beta_net
        scale = self.scale

        def new_forward(batch, *args, **kwargs):
            if "geom" not in batch:
                return old_forward(batch, *args, **kwargs)

            device = next(model.parameters()).device

            gamma_net.to(device)
            beta_net.to(device)

            geom = batch["geom"].float().to(device)
            out = old_forward(batch, *args, **kwargs)
            memory = out["model_output"]

            gamma = torch.tanh(gamma_net(geom)).unsqueeze(0)
            beta = torch.tanh(beta_net(geom)).unsqueeze(0)
            s = torch.sigmoid(scale.to(device))

            memory = memory * (1 + s * gamma) + s * beta
            out["model_output"] = memory

            return out

        model.forward = new_forward
