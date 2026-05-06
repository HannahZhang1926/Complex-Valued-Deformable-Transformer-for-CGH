import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from models import ComplexDropout

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        self.bias = bias
        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_i = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_r.weight)
        nn.init.xavier_uniform_(self.fc_i.weight)
        if self.bias:
            nn.init.constant_(self.bias_r, 0.)
            nn.init.constant_(self.bias_i, 0.)

    def forward(self, input):
        input_r = input.real
        input_i = input.imag
        if self.bias:
            return torch.complex(self.fc_r(input_r)-self.fc_i(input_i)+self.bias_r, 
                                 self.fc_r(input_i)+self.fc_i(input_r)+self.bias_i)
        else:
            return torch.complex(self.fc_r(input_r)-self.fc_i(input_i), 
                                 self.fc_r(input_i)+self.fc_i(input_r))


class RIComplexMultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0., qkv_same=True, kv_same=False, bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = ComplexDropout(p=attn_dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.qkv_same = qkv_same
        self.kv_same = kv_same
        self.bias = bias

        if qkv_same:
            self.in_proj_qkv = ComplexLinear(embed_dim, 3 * embed_dim, bias=bias)
        elif kv_same:
            self.in_proj_q = ComplexLinear(embed_dim, embed_dim, bias=bias)
            self.in_proj_kv = ComplexLinear(embed_dim, 2 * embed_dim, bias=bias)
        else:
            self.in_proj_q = ComplexLinear(embed_dim, embed_dim, bias=bias)
            self.in_proj_k = ComplexLinear(embed_dim, embed_dim, bias=bias)
            self.in_proj_v = ComplexLinear(embed_dim, embed_dim, bias=bias)
        
        self.out_proj = ComplexLinear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k_r = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_k_i = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v_r = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v_i = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k_r = self.bias_k_i = self.bias_v_r = self.bias_v_i = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):        
        if self.bias_k_r is not None:
            assert all(_ is not None for _ in [self.bias_k_i, self.bias_v_r, self.bias_v_i])
            nn.init.xavier_normal_(self.bias_k_r)
            nn.init.xavier_normal_(self.bias_k_i)
            nn.init.xavier_normal_(self.bias_v_r)
            nn.init.xavier_normal_(self.bias_v_i)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        # TODO key_padding_mask

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()
        
        # projecting q, k, v
        if self.qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query).chunk(3, dim=-1)
        elif self.kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key).chunk(2, dim=-1)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        # extending k, v by one time step at the end, with self.bias_k and self.bias_v (WHY??)
        if self.bias_k_r is not None:
            assert all(_ is not None for _ in [self.bias_k_i, self.bias_v_r, self.bias_v_i])

            bias_k = torch.complex(self.bias_k_r, self.bias_k_i)
            bias_v = torch.complex(self.bias_v_r, self.bias_v_i)
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])

            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # extending k, v by another time step at the end, (bsz * num_heads, 1, head_dim) of zeros (WHY??)
        if self.add_zero_attn:

            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)

            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        qk_inner = torch.bmm(q, torch.conj(k.transpose(1, 2)))
        attn_weights_r, attn_weights_i = qk_inner.real, qk_inner.imag
        assert list(attn_weights_r.size()) == [bsz * self.num_heads, tgt_len, src_len]
        assert list(attn_weights_i.size()) == [bsz * self.num_heads, tgt_len, src_len]
        attn_weights_r = attn_weights_r * self.scaling
        attn_weights_i = attn_weights_i * self.scaling

        if attn_mask is not None:
            try:
                attn_weights_r = attn_weights_r * attn_mask.unsqueeze(0)
                attn_weights_i = attn_weights_i * attn_mask.unsqueeze(0)
            except:
                print(attn_weights_r.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        # attn_weights_r = (attn_weights_r - torch.min(attn_weights_r)) / (torch.max(attn_weights_r) - torch.min(attn_weights_r))
        # attn_weights_i = (attn_weights_i - torch.min(attn_weights_i)) / (torch.max(attn_weights_i) - torch.min(attn_weights_i))
        attn_weights_r = F.softmax(attn_weights_r, dim=-1)
        attn_weights_i = F.softmax(attn_weights_i, dim=-1)
        
        attn_weights_r, attn_weights_i = self.attn_dropout(attn_weights_r, attn_weights_i)

        attn_weights = torch.complex(attn_weights_r, attn_weights_i)
        
        attn = torch.bmm(attn_weights, torch.conj(v))
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights
