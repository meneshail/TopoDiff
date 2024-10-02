import torch
from torch import nn
from torch.nn import init

from typing import Optional, Tuple, Union, Literal

#. functional
from TopoDiff.model.functional import get_timestep_embedding

class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(self, config_latent):
        super().__init__()
        self.config = config_latent

        layers = []
        for i in range(config_latent.num_time_layers):
            if i == 0:
                a = config_latent.num_time_emb_channels
                b = config_latent.num_channels
            else:
                a = config_latent.num_channels
                b = config_latent.num_channels
            layers.append(nn.Linear(a, b))
            if i < config_latent.num_time_layers - 1:
                if config_latent.activation == 'relu':
                    layers.append(nn.ReLU())
                elif config_latent.activation == 'lrelu':
                    layers.append(nn.LeakyReLU(0.2))
                elif config_latent.activation == 'silu':
                    layers.append(nn.SiLU())
                else:
                    raise NotImplementedError
        self.time_embed = nn.Sequential(*layers)

        self.layers = nn.ModuleList([])
        for i in range(config_latent.num_layers):
            if i == 0:
                act = config_latent.activation
                norm = config_latent.use_norm
                cond = True
                a, b = config_latent.num_channels, config_latent.num_hid_channels
                dropout = config_latent.dropout
            elif i == config_latent.num_layers - 1:
                act = 'none'
                norm = False
                cond = False
                a, b = config_latent.num_hid_channels, config_latent.num_channels
                dropout = 0
            else:
                act = config_latent.activation
                norm = config_latent.use_norm
                cond = True
                a, b = config_latent.num_hid_channels, config_latent.num_hid_channels
                dropout = config_latent.dropout

            if i in config_latent.skip_layers:
                if config_latent.skip_type == 0 or (i == 1 and config_latent.skip_type == 1):
                    # injecting input into the hidden layers
                    a += config_latent.num_channels
                elif config_latent.skip_type == 1:
                    a += config_latent.num_hid_channels
                else:
                    raise NotImplementedError

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=config_latent.num_channels,
                    use_cond=cond,
                    condition_bias=config_latent.condition_bias,
                    dropout=dropout,
                ))

    def forward(self, x, t, **kwargs):
        t = get_timestep_embedding(t, self.config.num_time_emb_channels)
        cond = self.time_embed(t)
        h = x
        h_pre = x
        for i in range(len(self.layers)):
            if i in self.config.skip_layers:
                if self.config.skip_type == 0:
                    # injecting input into the hidden layers
                    h = torch.cat([h, x], dim=1)
                elif self.config.skip_type == 1:
                    # injecting previous layer into the hidden layers
                    # print('layer', i, h.shape, h_pre.shape)
                    h_new = torch.cat([h, h_pre], dim=1)
                    h_pre = h
                    h = h_new
                    del h_new
                else:
                    raise NotImplementedError
            else:
                h_pre = h
            h = self.layers[i].forward(x=h, cond=cond)

        return h


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        use_cond: bool,
        cond_channels: int,
        activation: Literal['relu', 'lrelu', 'silu', 'none'] = 'relu',
        condition_bias: float = 0,
        dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif self.activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()

        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'relu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == 'lrelu':
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == 'silu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x