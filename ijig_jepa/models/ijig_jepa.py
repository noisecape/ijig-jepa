import copy

import torch
import torch.nn as nn

from ijig_jepa.models.vit import ViT


class IJigJepa(nn.Module):

    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 emb_size:int=768,
                 img_size:int=224,
                 depth:int=12,
                 out_dim:int=384,
                 device:str='cuda',
                 **kwargs
                ):
        super(IJigJepa, self).__init__()

        self.context_encoder = ViT(
            in_channels=in_channels,
            patch_size=patch_size,
            emb_size=emb_size,
            img_size=img_size,
            depth=depth,
            out_dim=out_dim,
            **kwargs
        )

        self.target_encoder = copy.deepcopy(self.context_encoder)
        # TODO: shuffle is true for both. It needs to be false for target_encoder
        
        for p in self.target_encoder.parameters():
            p.requires_grad=False

    def forward(self, x):

        z = self.context_encoder(x)

        h = self.target_encoder(x)

        return z, h 