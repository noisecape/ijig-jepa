# implementation based on: https://github.com/FrancescoSaverioZuppichini/ViT

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from PIL import Image
from torch import Tensor, nn
from torchsummary import summary
from torchvision.transforms import Compose, Resize, ToTensor


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels:int = 3, patch_size:int= 16, emb_size:int=768, img_size:int=224, shuffle_patches:bool=True):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.shuffle_patches = shuffle_patches
        self.projections = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        num_positions = int(((img_size*img_size) / (patch_size*patch_size)) + 1)
        self.pos_embeddings = nn.Parameter(self.create_pos_embeddings(num_positions=num_positions, emb_size=emb_size))

    def create_pos_embeddings(self, num_positions, emb_size):
        pos_encoding = torch.zeros(num_positions, emb_size)

        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)

        # implement the div term from the paper Attention is all you need
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)
        
        return pos_encoding
    
    def apply_shuffle(self, x:torch.Tensor):
        batch_size, n_patches, _ = x.shape
        permutation = torch.randperm(n_patches)
        x_shuffled = x.clone()
        for i in range(batch_size):
            x_shuffled[i, :, :] = x[i, :, :][permutation, :]
        return x_shuffled
    
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projections(x)
        if self.shuffle_patches:
            x = self.apply_shuffle(x)
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embeddings
        return x
    

class MultiHeadAttention(nn.Module):

    def __init__(self, emb_size:int=768, num_heads:int=8, dropout:float=0, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x: torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scaling = self.emb_size ** (1/2)
        energy = energy / scaling
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)
        
        att = F.softmax(energy, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out
    

class ResidualAdd(nn.Module):

    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):

    def __init__(self,
                 emb_size:int=768,
                 drop_p:float =0.,
                 forward_expansion:int=4,
                 forward_drop_p:float=0.,
                 **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                )
            ))
        )


class TransformerEncoder(nn.Sequential):

    def __init__(self, depth:int=12, **kwargs):
        super(TransformerEncoder, self).__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ProjectionHead(nn.Sequential):
    
    def __init__(self, emb_size:int=768, out_dim:int=384):
        super(ProjectionHead, self).__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_dim)
        )


class ViT(nn.Sequential):

    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 emb_size:int=768,
                 img_size:int=224,
                 depth:int=12,
                 out_dim:int=384,
                 shuffle_patches:bool=True,
                 **kwargs
                ):
        super(ViT, self).__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size, shuffle_patches),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ProjectionHead(emb_size, out_dim)
        )
                 

# summary(ViT(), (3, 224, 224), device='cpu')
