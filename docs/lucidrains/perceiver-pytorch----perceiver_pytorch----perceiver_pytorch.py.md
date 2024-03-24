# `.\lucidrains\perceiver-pytorch\perceiver_pytorch\perceiver_pytorch.py`

```py
# 从 math 模块中导入 pi 和 log 函数
# 从 functools 模块中导入 wraps 装饰器
# 导入 torch 库及其相关模块
# 从 torch.nn 模块中导入 nn 和 einsum
# 从 torch.nn.functional 模块中导入 F
# 导入 einops 库中的 rearrange 和 repeat 函数
# 从 einops.layers.torch 模块中导入 Reduce 类
from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

# 定义一些辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 缓存函数结果的装饰器
def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

# 对输入进行傅立叶编码的函数
def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# 定义一些辅助类

# 实现预层归一化的类
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

# 实现GEGLU激活函数的类
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# 实现前馈神经网络的类
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 实现注意力机制的类
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # 注意力机制，获取重要信息
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# 主类

class Perceiver(nn.Module):
    # 初始化函数，设置Transformer模型的参数
    def __init__(
        self,
        *,
        num_freq_bands,  # 频率带数量
        depth,  # Transformer的深度
        max_freq,  # 最大频率
        input_channels = 3,  # 输入通道数，默认为3
        input_axis = 2,  # 输入轴，默认为2
        num_latents = 512,  # 潜在变量数量，默认为512
        latent_dim = 512,  # 潜在维度，默认为512
        cross_heads = 1,  # 交叉头数，默认为1
        latent_heads = 8,  # 潜在头数，默认为8
        cross_dim_head = 64,  # 交叉维度头数，默认为64
        latent_dim_head = 64,  # 潜在维度头数，默认为64
        num_classes = 1000,  # 类别数量，默认为1000
        attn_dropout = 0.,  # 注意力机制的dropout，默认为0
        ff_dropout = 0.,  # 前馈网络的dropout，默认为0
        weight_tie_layers = False,  # 是否权重绑定层，默认为False
        fourier_encode_data = True,  # 是否对数据进行傅立叶编码，默认为True
        self_per_cross_attn = 1,  # 自注意力与交叉注意力的比例，默认为1
        final_classifier_head = True  # 是否使用最终分类头，默认为True
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        return_embeddings = False
        ):
        # 解构 data 的 shape，获取除了最后两个元素外的所有元素，分别赋值给 b 和 axis
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        # 断言 axis 的长度等于 self.input_axis，确保输入数据具有正确数量的轴
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # 如果需要对数据进行傅立叶编码
            # 计算每个轴上范围为[-1, 1]的傅立叶编码位置

            # 为每个轴生成均匀分布的位置
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            # 将每个轴的位置组合成多维网格
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            # 对位置进行傅立叶编码
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            # 重新排列编码后的位置
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            # 将编码后的位置重复 b 次
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            # 将编码后的位置拼接到数据的通道中
            data = torch.cat((data, enc_pos), dim=-1)

        # 将数据拼接到通道并展平轴
        data = rearrange(data, 'b ... d -> b (...) d')

        # 将 latents 重复 b 次
        x = repeat(self.latents, 'n d -> b n d', b=b)

        # 循环处理每一层
        for cross_attn, cross_ff, self_attns in self.layers:
            # 跨通道注意力和前馈网络
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            # 处理每个自注意力和前馈网络
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # 如果需要返回嵌入向量
        if return_embeddings:
            return x

        # 转换为 logits
        return self.to_logits(x)
```