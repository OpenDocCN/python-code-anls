# `.\lucidrains\perceiver-pytorch\perceiver_pytorch\experimental.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch.nn.functional 模块中导入 F 别名
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 从 perceiver_pytorch.perceiver_pytorch 模块中导入 exists, default, cache_fn, fourier_encode, PreNorm, FeedForward, Attention 类

# 定义线性注意力类 LinearAttention
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 4,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 定义线性变换层，将输入维度转换为内部维度的三倍
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # 定义输出层，包含线性变换和 dropout 操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    # 前向传播函数
    def forward(self, x, mask = None):
        h = self.heads
        # 将输入 x 经过线性变换层得到查询、键、值
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # 重排查询、键、值的维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # 缩放查询
        q = q * self.scale
        # 对查询和键进行 softmax 操作
        q, k = q.softmax(dim = -1), k.softmax(dim = -2)

        # 如果存在 mask，则对键进行填充
        if exists(mask):
            k.masked_fill_(mask, 0.)

        # 计算上下文信息
        context = einsum('b n d, b n e -> b d e', q, k)
        # 计算输出
        out = einsum('b d e, b n d -> b n e', context, v)
        # 重排输出的维度
        out = rearrange(out, ' (b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# 主类 Perceiver
class Perceiver(nn.Module):
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
        weight_tie_layers = False,
        fourier_encode_data = True
        ):
        # 调用父类的构造函数
        super().__init__()
        # 设置输入数据的轴数
        self.input_axis = input_axis
        # 设置最大频率
        self.max_freq = max_freq
        # 设置频率带数量
        self.num_freq_bands = num_freq_bands
        # 是否对数据进行傅立叶编码
        self.fourier_encode_data = fourier_encode_data

        # 计算输入维度
        input_dim = input_channels

        # 如果需要对数据进行傅立叶编码
        if fourier_encode_data:
            # 更新输入维度
            input_dim += input_axis * ((num_freq_bands * 2) + 1) + input_channels

        # 初始化潜在变量
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # 数据投影层
        self.data_proj = nn.Linear(input_dim, input_dim)

        # 定义获取交叉注意力的函数
        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        # 定义获取交叉前馈网络的函数
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        # 定义获取输入注意力的函数
        get_input_attn = lambda: PreNorm(input_dim, LinearAttention(input_dim, dropout = attn_dropout))
        # 定义获取反向交叉注意力的函数
        get_rev_cross_attn = lambda: PreNorm(input_dim, Attention(input_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = latent_dim)
        # 定义获取反向交叉前馈网络的函数
        get_rev_cross_ff = lambda: PreNorm(input_dim, FeedForward(input_dim, dropout = ff_dropout))

        # 定义获取潜在注意力的函数
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        # 定义获取潜在前馈网络的函数
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        # 使用缓存函数对获取函数进行缓存
        get_cross_attn, get_cross_ff, get_rev_cross_attn, get_rev_cross_ff, get_input_attn, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_rev_cross_attn, get_rev_cross_ff, get_input_attn, get_latent_attn, get_latent_ff))

        # 初始化网络层
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                get_rev_cross_attn(**cache_args),
                get_rev_cross_ff(**cache_args),
                get_input_attn(**cache_args),
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # 输出层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, data, mask = None):
        # 获取数据的维度信息
        b, *axis, _, device = *data.shape, data.device
        # 断言数据维度与输入轴数相符
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # 如果需要对数据进行傅立叶编码
        if self.fourier_encode_data:
            # 计算在[-1, 1]范围内的傅立叶编码位置，对所有轴
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            # 将编码位置与数据的通道连接并展平轴
            data = torch.cat((data, enc_pos), dim = -1)

        data = rearrange(data, 'b ... d -> b (...) d')

        # 数据投影
        data = self.data_proj(data)

        # 重复潜在变量
        x = repeat(self.latents, 'n d -> b n d', b = b)

        # 遍历网络层
        for i, (cross_attn, cross_ff, rev_cross_attn, rev_cross_ff, input_attn, latent_attn, latent_ff) in enumerate(self.layers):
            is_last = i == (len(self.layers) - 1)

            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            if not is_last:
                data = input_attn(data, mask = mask) + data
                data = rev_cross_attn(data, context = x) + data
                data = rev_cross_ff(data) + data

            x = latent_attn(x) + x
            x = latent_ff(x) + x

        # 对最后的输出进行平均处理
        x = x.mean(dim = -2)
        return self.to_logits(x)
```