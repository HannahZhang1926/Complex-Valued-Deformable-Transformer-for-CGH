import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
from complexPyTorch.complexLayers import ComplexConvTranspose2d,ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu

import math
from models import *
import einops

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, attn_mask=False):
        super().__init__()
        self.dropout = 0.3  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        # Available: TransformerEncoderLayer(), ComplexTransformerEncoderLayer()
        self.layers.extend([
            ComplexTransformerEncoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    attn_dropout=attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    attn_mask=attn_mask)
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        
        
    def forward(self, input_A, input_B): 
        
        # For each transformer encoder layer:
        for layer in self.layers:
            input_A, input_B = layer(input_A, input_B)
        
        return input_A, input_B



######### 对比实验用，请按需要修改！下面有复数操作的
class ComplexTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn_mask = attn_mask
        self.crossmodal = True
        self.normalize = True

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = ComplexLinear(self.embed_dim//2, self.embed_dim//2)  # The "Add & Norm" part in the paper
        self.fc2 = ComplexLinear(self.embed_dim//2, self.embed_dim//2)  # The "Add & Norm" part in the paper

        self.layer_norms = nn.ModuleList([ComplexLayerNorm(self.embed_dim//2) for _ in range(2)])
        
        ###############################################################
        self.n_head_channels = 32
        self.n_heads = 2
        self.nc = self.n_head_channels * self.n_heads
        self.n_groups = 1
        self.n_group_channels = self.nc // self.n_groups
        self.stride = 8
        self.ksize = 9
        
        kk = self.ksize
        pad_size = kk // 2 if kk != self.stride else 0
        
        self.scale = self.n_head_channels ** -0.5
        
        
        
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kk, self.stride, pad_size, groups=self.embed_dim),
            LayerNormProxy(self.embed_dim),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.embed_dim, self.embed_dim,
            kernel_size=1, stride=1, padding=0
        )
        
        self.proj_k = nn.Conv2d(
            self.embed_dim, self.embed_dim,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.embed_dim, self.embed_dim,
            kernel_size=1, stride=1, padding=0
        )
    
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        return ref
    
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref
    ##############################################################################   
        
    def forward(self, x_A, x_B):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
        """
        ## Attention Part
        # # Residual and Layer Norm
        ###########################################
        B, C, H, W = x_A.size()
        dtype, device = x_A.dtype, x_A.device
        
        residual_A = x_A.reshape(B * self.n_heads, self.embed_dim//2, H * W)
        residual_A = residual_A.permute(2, 0, 1)  # B,C,T -> T,B,C
        residual_B = x_B.reshape(B * self.n_heads, self.embed_dim//2, H * W)
        residual_B = residual_B.permute(2, 0, 1)
        
        q_a = self.proj_q(x_A)
        q_b = self.proj_q(x_B)
        
        q_off_a = einops.rearrange(q_a, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.embed_dim)
        q_off_b = einops.rearrange(q_b, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.embed_dim)
        
        offset_a = self.conv_offset(q_off_a).contiguous()
        offset_b = self.conv_offset(q_off_b).contiguous()
        
        Hk, Wk = offset_a.size(2), offset_a.size(3)
        n_sample = Hk * Wk
        
        offset_a = einops.rearrange(offset_a, 'b p h w -> b h w p')
        offset_b = einops.rearrange(offset_b, 'b p h w -> b h w p')
        
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        
        pos_a = offset_a + reference
        pos_b = offset_b + reference        
        
        x_sampled_a = F.grid_sample(
            input=x_A.reshape(B * self.n_groups, self.embed_dim, H, W), 
            grid=pos_a[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        
        x_sampled_b = F.grid_sample(
            input=x_B.reshape(B * self.n_groups, self.embed_dim, H, W), 
            grid=pos_b[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        
        x_sampled_a = x_sampled_a.reshape(B, C, 1, n_sample)
        x_sampled_b = x_sampled_b.reshape(B, C, 1, n_sample)

        q_a = q_a.reshape(B * self.n_heads, self.embed_dim//2, H * W)
        k_a = self.proj_k(x_sampled_a).reshape(B * self.n_heads, self.embed_dim//2, n_sample)
        v_a = self.proj_v(x_sampled_a).reshape(B * self.n_heads, self.embed_dim//2, n_sample)
        
        q_b = q_b.reshape(B * self.n_heads, self.embed_dim//2, H * W)
        k_b = self.proj_k(x_sampled_b).reshape(B * self.n_heads, self.embed_dim//2, n_sample)
        v_b = self.proj_v(x_sampled_b).reshape(B * self.n_heads, self.embed_dim//2, n_sample)
        
        # ####################################################################
        # # Complex Multihead Attention
        query = torch.complex(q_a,q_b)
        key = torch.complex(k_a,k_b)
        value = torch.complex(v_a,v_b)
        
        attn = torch.einsum('b c m, b c n -> b m n', query, torch.conj(key)) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        
        rpe_table = nn.Parameter(torch.zeros(self.n_heads, H * 2 - 1, W * 2 - 1))
        
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        rpe_bias = rpe_bias.to(attn.device)
        
        q_grid = self._get_q_grid(H, W, B, dtype, device)
        
        displacement_a = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos_a.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
        attn_bias_a = F.grid_sample(
            input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=2, g=self.n_groups),
            grid=displacement_a[..., (1, 0)],
            mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns
        
        displacement_b = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos_b.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
        attn_bias_b = F.grid_sample(
            input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=2, g=self.n_groups),
            grid=displacement_b[..., (1, 0)],
            mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns
        
        #self.n_group_heads
        
        attn_bias_a = attn_bias_a.reshape(B * self.n_heads, H * W, n_sample)
        attn_bias_b = attn_bias_b.reshape(B * self.n_heads, H * W, n_sample)
        attn_bias = torch.complex(attn_bias_a,attn_bias_b)
        
        attn = attn + attn_bias
        attn = attn.real

        attn = F.softmax(attn, dim=2)
        attn = F.dropout(attn, p=0.0, training=self.training)
        
        out = torch.einsum('b m n, b c n -> b c m', torch.complex(attn,attn.new_zeros(attn.shape)), value)
        out = out.permute(2, 0, 1)  # B,C,T -> T,B,C
        
        x_A = out.real
        x_B = out.imag
        
        # Complex Layer Norm
        x_A, x_B = self.layer_norms[0](x_A, x_B)

        # Dropout and Residual
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)

        x_A = residual_A + x_A
        x_B = residual_B + x_B

        # ##FC Part
        residual_A = x_A
        residual_B = x_B

        # FC1
        x_A, x_B = self.fc1(x_A, x_B)
        # cReLU
        x_A = F.relu(x_A)
        x_B = F.relu(x_B)
        x_A = F.dropout(x_A, p=self.relu_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.relu_dropout, training=self.training)

        # FC2
        x_A, x_B = self.fc2(x_A, x_B)

        x_A, x_B = self.layer_norms[1](x_A, x_B)

        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)

        x_A = residual_A + x_A
        x_B = residual_B + x_B
        
        x_A = x_A.permute(1,2,0)
        x_A = x_A.reshape(B, C, H, W)
        x_B = x_B.permute(1,2,0)
        x_B = x_B.reshape(B, C, H, W)
        
        return x_A, x_B

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x



######### 对比实验用，请按需修改！
# class ComplexTransformerEncoderLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1,
#                  relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.attn_mask = attn_mask
#         self.crossmodal = True
#         self.normalize = True
#         self.relu_dropout = relu_dropout
#         self.res_dropout = res_dropout
#         self.normalize_before = True

#         self.fc1 = ComplexLinear(embed_dim, embed_dim)
#         self.fc2 = ComplexLinear(embed_dim, embed_dim)
#         self.layer_norms = nn.ModuleList([ComplexLayerNorm(embed_dim) for _ in range(2)])

#         self.n_head_channels = 32
#         self.n_heads = 2
#         self.nc = self.n_head_channels * self.n_heads
#         self.n_groups = 1
#         self.n_group_channels = self.nc // self.n_groups
#         self.stride = 8
#         self.ksize = 9
#         pad_size = self.ksize // 2 if self.ksize != self.stride else 0
#         self.scale = self.n_head_channels ** -0.5

#         self.conv_offset = nn.Sequential(
#             ComplexConv2d(embed_dim, embed_dim, self.ksize, self.stride, pad_size,
#                           groups=embed_dim),
#             ComplexLayerNorm(embed_dim),
#             nn.GELU(),
#             ComplexConv2d(embed_dim, 1, 1, 1, 0, bias=False)
#         )

#         ###########################
#         self.proj_q = ComplexConv2d(embed_dim, embed_dim, 1, 1, 0)
#         self.proj_k = ComplexConv2d(embed_dim, embed_dim, 1, 1, 0)
#         self.proj_v = ComplexConv2d(embed_dim, embed_dim, 1, 1, 0)

#         ##########################
#         self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, H * 2 - 1, W * 2 - 1))

#     def _get_ref_points(self, Hk, Wk, B, dtype, device):
#         ref_y, ref_x = torch.meshgrid(
#             torch.linspace(0.5, Hk - 0.5, Hk, dtype=dtype, device=device),
#             torch.linspace(0.5, Wk - 0.5, Wk, dtype=dtype, device=device),
#             indexing='ij'
#         )
#         ref = torch.stack((ref_y, ref_x), -1)
#         ref[..., 1] = ref[..., 1] / (Wk - 1.0) * 2.0 - 1.0
#         ref[..., 0] = ref[..., 0] / (Hk - 1.0) * 2.0 - 1.0
#         ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B*g H W 2
#         return ref

#     def _get_q_grid(self, H, W, B, dtype, device):
#         ref_y, ref_x = torch.meshgrid(
#             torch.arange(0, H, dtype=dtype, device=device),
#             torch.arange(0, W, dtype=dtype, device=device), indexing='ij'
#         )
#         ref = torch.stack((ref_y, ref_x), -1).float()
#         ref[..., 1] = ref[..., 1] / (W - 1.0) * 2.0 - 1.0
#         ref[..., 0] = ref[..., 0] / (H - 1.0) * 2.0 - 1.0
#         ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
#         return ref

#     def forward(self, x):
#         B, C, H, W = x.shape
#         dtype, device = x.real.dtype, x.real.device
#         residual = x

#         q = self.proj_q(x) 
#         offset = self.conv_offset(q) 
#         offset = offset.real        ##########根据复数操作的公式只取了real

#         Hk, Wk = offset.shape[2], offset.shape[3]
#         n_sample = Hk * Wk

#         offset = einops.rearrange(offset, 'b c h w -> b h w c')  # (B,Hk,Wk,2)
#         reference = self._get_ref_points(Hk, Wk, B, dtype, device)

#         pos = offset + reference                              # (B*g,Hk,Wk,2)
#         pos = pos[..., (1, 0)]                                # swap xy -> (x,y)


#         x_g = einops.rearrange(x, 'b (g c) h w -> (b g) c h w',
#                                g=self.n_groups, c=C // self.n_groups)
#         x_sampled = F.grid_sample(
#             x_g, pos, mode='bilinear', align_corners=True)  # (B*g,Cg,Hk,Wk)
#         x_sampled = einops.rearrange(
#             x_sampled, '(b g) c h w -> b (g c) 1 (h w)', b=B)  # (B,C,1,n_sample)

#         q = einops.rearrange(q, 'b (h c) h w -> b h c (h w)',
#                              h=self.n_heads).contiguous()
#         k = self.proj_k(x_sampled)
#         k = einops.rearrange(k, 'b (h c) 1 n -> b h c n',
#                              h=self.n_heads).contiguous()
#         v = self.proj_v(x_sampled)
#         v = einops.rearrange(v, 'b (h c) 1 n -> b h c n',
#                              h=self.n_heads).contiguous()

#         scale = self.n_head_channels ** -0.5
#         attn = torch.einsum('bhcm,bhcn->bhmn', q, k.conj()) * scale  # (B,h,HW,n_sample)

#         rpe_bias = F.grid_sample(
#             self.rpe_table[None, ...].expand(B, -1, -1, -1),
#             self._get_q_grid(H, W, B, dtype, device)[..., (1, 0)],
#             mode='bilinear', align_corners=True)          # (B,h,HW,n_sample)
#         attn = attn.real + rpe_bias                       # 只把实部加上去，和公式保持一致！

#         attn = F.softmax(attn, dim=-1)
#         attn = F.dropout(attn, p=0.0, training=self.training)

#         out = torch.einsum('bhmn,bhcn->bhcm', attn, v)    # (B,h,C_head,HW)
#         out = out.reshape(B, C, H, W)

#         # 6. dropout + residual + norm
#         out = F.dropout(out, p=self.res_dropout, training=self.training)
#         out = residual + out
#         out = self.layer_norms[0](out)

#         # ===================== FFN =====================
#         residual = out
#         out = self.fc1(out)
#         out = complex_relu(out)
#         out = F.dropout(out, p=self.relu_dropout, training=self.training)
#         out = self.fc2(out)
#         out = self.layer_norms[1](out)
#         out = F.dropout(out, p=self.res_dropout, training=self.training)
#         out = residual + out

#         return out



class TransformerConcatEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, attn_mask=False):
        super().__init__()
        self.dropout = 0.3  # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerConcatEncoderLayer(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          attn_dropout=attn_dropout,
                                          relu_dropout=relu_dropout,
                                          res_dropout=res_dropout,
                                          attn_mask=attn_mask)
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, x):
        x = self.scale_embed_position_dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerConcatEncoderLayer(nn.Module):
    """Encoder layer block.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True
        )
        self.attn_mask = attn_mask
        self.normalize = True

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x):
        ## Attention Part
        # Residual and Layer Norm
        residual = x
        # Multihead Attention
        x = self.attention_block(x, x, x)

        x = self.layer_norms[0](x)
        # Dropout and Residual
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x

        # ##FC Part
        residual = x

        # FC1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)

        x = self.fc2(x)
        x = self.layer_norms[1](x)

        x = F.dropout(x, p=self.res_dropout, training=self.training)

        x = residual + x

        return x

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def attention_block(self, x, x_k, x_v):
        mask = None
        x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        return x


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def fill_with_one(t):
    return t.float().fill_(float(1)).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.tril(fill_with_one(torch.ones(dim1, dim2)), 0)
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim, eps=1e-20)
    return m

def ComplexLayerNorm(normalized_shape):
    m = ComplexLayerNorm1d(normalized_shape, eps=1e-20)
    return m