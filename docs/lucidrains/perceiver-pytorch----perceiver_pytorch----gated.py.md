# `.\lucidrains\perceiver-pytorch\perceiver_pytorch\gated.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块、einsum 函数
from torch import nn, einsum
# 从 torch.nn.functional 中导入 F 模块
import torch.nn.functional as F

# 从 einops 库中导入 rearrange、repeat 函数
from einops import rearrange, repeat

# 从 perceiver_pytorch.perceiver_pytorch 中导入 exists、default、cache_fn、fourier_encode、PreNorm、FeedForward、Attention

# helpers

# 定义 Residual 类，继承 nn.Module 类
class Residual(nn.Module):
    # 初始化函数
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    # 前向传播函数
    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

# 定义 GRUGating 类，继承 nn.Module 类
class GRUGating(nn.Module):
    # 初始化函数
    def __init__(self, dim, fn):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nn.GRUCell(dim, dim)

    # 前向传播函数
    def forward(self, x, **kwargs):
        b, dim = x.shape[0], self.dim
        y = self.fn(x, **kwargs)

        gated_output = self.gru(
            rearrange(y, '... d -> (...) d'),
            rearrange(x, '... d -> (...) d')
        )

        gated_output = rearrange(gated_output, '(b n) d -> b n d', b = b)
        return gated_output

# main class

# 定义 Perceiver 类，继承 nn.Module 类
class Perceiver(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False
    ):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn  = lambda: GRUGating(latent_dim, PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim))
        get_latent_attn = lambda: GRUGating(latent_dim, PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_cross_ff    = lambda: Residual(PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout)))
        get_latent_ff   = lambda: Residual(PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout)))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )
    # 前向传播函数，接受数据和掩码作为输入
    def forward(self, data, mask = None):
        # 获取数据的形状和设备信息
        b, *axis, _, device = *data.shape, data.device
        # 断言数据的轴数与输入轴数相同
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # 计算傅立叶编码的位置，范围为[-1, 1]，对所有轴

        # 生成每个轴上的位置信息
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
        # 生成位置的网格
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
        # 对位置信息进行傅立叶编码
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
        # 重新排列编码后的位置信息
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        # 复制编码后的位置信息，使其与数据维度相匹配
        enc_pos = repeat(enc_pos, '... -> b ...', b = b)

        # 将编码后的位置信息连接到数据的通道上，并展平轴

        data = torch.cat((data, enc_pos), dim = -1)
        data = rearrange(data, 'b ... d -> b (...) d')

        # 复制潜在变量，使其与数据维度相匹配
        x = repeat(self.latents, 'n d -> b n d', b = b)

        # 遍历每个层，进行交叉注意力、交叉前馈、潜在注意力和潜在前馈操作
        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            x = cross_attn(x, context = data, mask = mask)
            x = cross_ff(x)
            x = latent_attn(x)
            x = latent_ff(x)

        # 对最终结果进行平均处理，并返回logits
        x = x.mean(dim = -2)
        return self.to_logits(x)
```