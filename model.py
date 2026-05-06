import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
from modules.transformer import TransformerEncoder
from models import *
from timm.models.layers import to_2tuple
import einops
from complexPyTorch.complexLayers import ComplexConvTranspose2d,ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

######### 对比实验用，请按需要修改！下面有复数操作的
class TransformerModel(nn.Module):
    def __init__(self,embed_dim, num_heads, attn_dropout, relu_dropout,
                 res_dropout, out_dropout, layers, attn_mask=False,patch_size=4):

        super(TransformerModel, self).__init__()
        
        self.embed_dim = embed_dim
        final_out = embed_dim * 2
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.attn_mask = attn_mask   
        
        self.patch_proj = nn.Sequential(
            nn.Conv2d(1, self.embed_dim // 2, 3, patch_size // 2, 1),
            LayerNormProxy(self.embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, 3, patch_size // 2, 1),
            LayerNormProxy(self.embed_dim)
        )

        # Transformer networks
        self.trans = self.get_network()

        self.out_dropout = nn.Dropout(out_dropout)

        self.upsample = nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=1, kernel_size=4, stride=4)

    def get_network(self):
        return TransformerEncoder(embed_dim=self.embed_dim, num_heads=self.num_heads, layers=self.layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout, res_dropout=self.res_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x):
        a = x.real
        b = x.imag
        
        input_a = self.patch_proj(a)
        input_b = self.patch_proj(b)
        
        # 输入DAT
        h_as, h_bs = self.trans(input_a,input_b)
        
        a_real = self.upsample(h_as)
        a_imag = self.upsample(h_bs)

        output = torch.complex(a_real, a_imag)
        
        return output

######### 对比实验用，请按需修改！
# class TransformerModel(nn.Module):
#     def __init__(self, embed_dim, num_heads, attn_dropout, relu_dropout,
#                  res_dropout, out_dropout, layers, attn_mask=False, patch_size=4):
#         super(TransformerModel, self).__init__()

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.layers = layers
#         self.attn_dropout = attn_dropout
#         self.relu_dropout = relu_dropout
#         self.res_dropout = res_dropout
#         self.attn_mask = attn_mask

#         self.patch_proj = nn.Sequential(
#             ComplexConv2d(1, embed_dim // 2, 3, patch_size // 2, 1),
#             ComplexReLU(),
#             ComplexConv2d(embed_dim // 2, embed_dim, 3, patch_size // 2, 1),
#             ComplexReLU()
#         )

#         self.trans = TransformerEncoder(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             layers=layers,
#             attn_dropout=attn_dropout,
#             relu_dropout=relu_dropout,
#             res_dropout=res_dropout,
#             attn_mask=attn_mask
#         )

#         self.out_dropout = nn.Dropout(out_dropout)
#         self.upsample = ComplexConvTranspose2d(embed_dim, 1, kernel_size=4, stride=4)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         h = self.patch_proj(x) 
#         h = self.trans(h)               
#         h = self.out_dropout(h)
#         out = self.upsample(h)

#         return out