import torch
import torch.nn as nn

from myopenfold.model.primitives import Linear

class AuxiliaryHeads(nn.Module):
    def __init__(self, config, depth=0, log=False):
        super(AuxiliaryHeads, self).__init__()

        self.depth=depth
        self.log=log

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.config = config

    def forward(self, outputs, feat):
        aux_out = {}
        
        distogram_logits = self.distogram(outputs["pair_emb"])
        aux_out["distogram_logits"] = distogram_logits
        
        return aux_out
    
class SCHead(nn.Module):
    """
    Predict the self-consistency score.
    """
    def __init__(self, config_sc, **kwargs):
        super().__init__()

        self.config = config_sc
        
        self.latent_dim = config_sc.latent_dim
        self.ff_dim = config_sc.ff_dim
        self.c_out = config_sc.c_out
        self.n_layers = config_sc.n_layers
        self.dropout = config_sc.dropout
    
        layers = []
        for i in range(self.n_layers):
            in_dim = self.latent_dim if i == 0 else self.ff_dim
            out_dim = self.c_out if i == self.n_layers - 1 else self.ff_dim
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if i != self.n_layers - 1:
                layers.append(torch.nn.GELU())
                layers.append(torch.nn.Dropout(self.dropout))
        self.head = torch.nn.Sequential(*layers)

    def forward(self, latent_z):
        """
        Args:
            latent_z:
                [*, latent_dim] latent_z
        Returns:
            [*, c_out] self-consistency logits
        """
        # [*, c_out]
        logits = self.head(latent_z)
        return logits


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.
    Adapted from OpenFold.
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits