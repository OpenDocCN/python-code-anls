# `.\lucidrains\vit-pytorch\vit_pytorch\na_vit.py`

```py
from functools import partial
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

# 检查值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 返回固定值的函数
def always(val):
    return lambda *args: val

# 将输入转换为元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 检查一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# auto grouping images

# 根据最大序列长度对图像进行分组
def group_images_by_max_seq_len(
    images: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 2048

) -> List[List[Tensor]]:

    calc_token_dropout = default(calc_token_dropout, always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, image_dims)

        image_seq_len = (ph * pw)
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups

# normalization
# they use layernorm without bias, something that pytorch does not offer

# 自定义 LayerNorm 类
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper

# 自定义 RMSNorm 类
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# feedforward

# 定义 FeedForward 函数
def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

# 定义 Attention 类
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None
        ):
        # 对输入进行归一化处理
        x = self.norm(x)
        # 从上下文中获取默认的键值对输入
        kv_input = default(context, x)

        # 将输入数据转换为查询、键、值三部分
        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        # 将查询、键、值进行维度重排，以适应多头注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # 对查询和键进行归一化处理
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 计算查询和键的点积
        dots = torch.matmul(q, k.transpose(-1, -2))

        # 如果存在掩码，则进行掩码处理
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        # 如果存在注意力掩码，则进行掩码处理
        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        # 进行注意力计算
        attn = self.attend(dots)
        # 对注意力结果进行dropout处理
        attn = self.dropout(attn)

        # 将注意力结果与值进行加权求和
        out = torch.matmul(attn, v)
        # 将结果进行维度重排，返回输出
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class Transformer(nn.Module):
    # 定义 Transformer 类，继承自 nn.Module
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        # 初始化函数，接受维度、深度、头数、头维度、MLP维度和dropout参数
        super().__init__()
        # 调用父类的初始化函数
        self.layers = nn.ModuleList([])
        # 初始化层列表
        for _ in range(depth):
            # 循环创建指定深度的层
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = LayerNorm(dim)
        # 初始化 LayerNorm 层

    def forward(
        self,
        x,
        mask = None,
        attn_mask = None
    ):
        # 前向传播函数
        for attn, ff in self.layers:
            # 遍历层列表
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            # 使用注意力机制处理输入并加上残差连接
            x = ff(x) + x
            # 使用前馈网络处理输入并加上残差连接

        return self.norm(x)
        # 返回经过 LayerNorm 处理后的结果

class NaViT(nn.Module):
    # 定义 NaViT 类，继承自 nn.Module
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., token_dropout_prob = None):
        # 初始化函数，接受图像大小、补丁大小、类别数、维度、深度、头数、MLP维度等参数
        super().__init__()
        # 调用父类的初始化函数
        image_height, image_width = pair(image_size)

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0. < token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        batched_images: Union[List[Tensor], List[List[Tensor]]], # assume different resolution images already grouped correctly
        group_images = False,
        group_max_seq_len = 2048
```