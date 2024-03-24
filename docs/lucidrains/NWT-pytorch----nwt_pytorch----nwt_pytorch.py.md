# `.\lucidrains\NWT-pytorch\nwt_pytorch\nwt_pytorch.py`

```py
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import EinMix as Mix

# 定义一个名为Memcodes的神经网络模型，继承自nn.Module类
class Memcodes(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 输入数据的维度
        num_codes,  # 编码的数量
        heads = 8,  # 多头注意力机制中的头数，默认为8
        temperature = 1.,  # 温度参数，默认为1
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5  # 缩放因子
        self.temperature = temperature
        self.num_codes = num_codes

        num_codebooks = heads
        codebook_dim = dim // heads

        # 初始化编码参数
        self.codes = nn.Parameter(torch.randn(num_codebooks, num_codes, codebook_dim))
        # 初始化转换矩阵，用于将编码转换为key
        self.to_k = Mix('h n d -> h n c', weight_shape = 'h d c', h = heads, d = codebook_dim, c = codebook_dim)
        # 初始化转换矩阵，用于将编码转换为value
        self.to_v = Mix('h n d -> h n c', weight_shape = 'h d c', h = heads, d = codebook_dim, c = codebook_dim)

    # 根据编码的索引获取编码
    def get_codes_from_indices(self, codebook_indices, *, merge_output_heads = True):
        batch = codebook_indices.shape[0]

        values = self.to_v(self.codes)
        values = repeat(values, 'h n d -> b h n d', b = batch)

        codebook_indices = repeat(codebook_indices, '... -> ... d', d = values.shape[-1])
        out = values.gather(2, codebook_indices)

        if not merge_output_heads:
            return out

        return rearrange(out, 'b h n d -> b n (h d)')

    # 前向传播函数
    def forward(self, x, *, merge_output_heads = True):
        assert x.shape[-1] == self.dim

        # 将输入数据分成多个头
        q = rearrange(x, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        # 获取编码的key和value
        k, v = self.to_k(self.codes), self.to_v(self.codes)

        # 使用直通Gumbel Softmax
        logits = einsum('b h i d, h j d -> b h i j', q, k)

        if self.training:
            attn = F.gumbel_softmax(logits, tau = self.temperature, dim = -1, hard = True)
            codebook_indices = attn.argmax(dim = -1)
        else:
            codebook_indices = logits.argmax(dim = -1)
            attn = F.one_hot(codebook_indices, num_classes = self.num_codes).float()

        out = einsum('b h i j, h j d -> b h i d', attn, v)

        if not merge_output_heads:
            return out, codebook_indices

        # 如果指定了合并头部，则合并头部
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out, codebook_indices
```