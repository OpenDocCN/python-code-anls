# `.\lucidrains\Mega-pytorch\mega_pytorch\mega_pytorch.py`

```
# 导入数学库
import math
# 从 functools 库中导入 partial 函数
from functools import partial

# 导入 torch 库
import torch
# 从 torch.nn.functional 模块中导入 F
import torch.nn.functional as F
# 从 torch 模块中导入 nn 和 einsum
from torch import nn, einsum
# 从 torch.fft 模块中导入 rfft 和 irfft
from torch.fft import rfft, irfft

# 从 einops 库中导入 rearrange 和 Rearrange
from einops import rearrange
from einops.layers.torch import Rearrange

# 从 scipy.fftpack 模块中导入 next_fast_len 函数

# functions

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 返回输入的函数
def identity(t, *args, **kwargs):
    return t

# 如果输入值存在则返回输入值，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 在输入张量的末尾添加指定数量的维度的函数
def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

# 使用傅立叶技巧进行 O(N log(N)) 的 1D 卷积的函数
def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = rfft(x, n = fast_len, dim = dim)
    f_weight = rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)
    out = irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

# 用于单头注意力的位置偏置类
class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# classes

# 拉普拉斯注意力函数类
class LaplacianAttnFn(nn.Module):
    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt((4 * math.pi) ** -1)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

# 偏移和缩放类
class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

# 单头注意力类
class SingleHeadedAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_qk,
        dim_value,
        causal = False,
        laplacian_attn_fn = False
    # 初始化 Transformer 层
    def __init__(
        self,
        causal: bool = False,
        laplacian_attn_fn: bool = False
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置是否使用因果关系和 Laplacian 注意力函数
        self.causal = causal
        self.laplacian_attn_fn = laplacian_attn_fn

        # 根据是否使用 Laplacian 注意力函数选择不同的注意力函数
        self.attn_fn = partial(F.softmax, dim = -1) if not laplacian_attn_fn else LaplacianAttnFn()

        # 初始化相对位置偏置
        self.rel_pos_bias = T5RelativePositionBias(causal = causal, scale = dim_qk ** 0.5)

        # 将输入转换为查询和键值对
        self.to_qk = nn.Sequential(
            nn.Linear(dim, dim_qk),
            nn.SiLU()
        )

        # 初始化偏移和缩放层
        self.offsetscale = OffsetScale(dim_qk, heads = 2)

        # 将输入转换为值
        self.to_v = nn.Sequential(
            nn.Linear(dim, dim_value),
            nn.SiLU()
        )

    # 前向传播函数
    def forward(self, x, v_input = None):
        # 获取序列长度、维度、设备和数据类型
        seq_len, dim, device, dtype = *x.shape[-2:], x.device, x.dtype

        # 如果未提供值输入，则使用 x 作为值输入
        v_input = default(v_input, x)

        # 将输入转换为查询、键和值
        qk, v = self.to_qk(x), self.to_v(v_input)
        q, k = self.offsetscale(qk)

        # 计算缩放因子
        scale = (seq_len ** -1) if self.laplacian_attn_fn else (dim ** -0.5)

        # 计算注意力矩阵
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

        # 添加相对位置偏置
        sim = sim + self.rel_pos_bias(sim)

        # 如果使用因果关系，则创建因果 mask
        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).triu(1)

        # 如果使用因果关系且不使用 Laplacian 注意力函数，则对注意力矩阵进行 mask 处理
        if self.causal and not self.laplacian_attn_fn:
            # 如果是 softmax 注意力并且使用大的负值作为 softmax 前的值
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 计算注意力权重
        attn = self.attn_fn(sim)

        # 如果使用因果关系且使用 Laplacian 注意力函数，则将上三角部分置为 0
        if self.causal and self.laplacian_attn_fn:
            # 如果使用 Laplacian 注意力函数，则将上三角部分置为 0
            attn = attn.masked_fill(causal_mask, 0.)

        # 计算输出值
        return einsum('b i j, b j d -> b i d', attn, v)
class MultiHeadedEMA(nn.Module):
    # 定义多头EMA模块
    def __init__(
        self,
        *,
        dim,
        heads,
        bidirectional = False,
        norm_mhesa_heads = False
    ):
        # 初始化函数
        super().__init__()
        self.bidirectional = bidirectional

        # 初始化参数
        self.expansion = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))
        self.reduction = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))

        # 学习的alpha和阻尼因子

        self.alphas = nn.Parameter(torch.randn(heads))
        self.dampen_factors = nn.Parameter(torch.randn(heads))

        if bidirectional:
            self.reverse_alphas = nn.Parameter(torch.randn(heads))
            self.reverse_dampen_factors = nn.Parameter(torch.randn(heads))

        self.heads = heads

        self.norm_heads = nn.Identity()

        if norm_mhesa_heads:
            # 使用子层归一化作为组归一化
            self.norm_heads = nn.Sequential(
                Rearrange('b n h d -> b (h d) n'),
                nn.GroupNorm(heads, dim * heads),
                Rearrange('b (h d) n -> b n h d', h = heads)
            )

    def forward(self, x):
        # 前向传播函数
        device, seq_len = x.device, x.shape[1]

        # 投影并分割头部
        x = einsum('... d, h d -> ... h d', x, self.expansion)

        if self.bidirectional:
            x, x_reversed = x.chunk(2, dim = -2)
            x_reversed = torch.flip(x_reversed, dims = (1,))

        # 从alphas派生的权重（学习的指数平滑衰减率）
        def apply_learned_ema_with_damping(x, alphas, dampen_factors):
            alphas = alphas.sigmoid()
            dampen_factors = dampen_factors.sigmoid()

            reversed_powers = torch.arange(seq_len - 1, -1, -1, device = device)
            K = alphas * (((1 - alphas) * dampen_factors) ** rearrange(reversed_powers, '... l -> ... l 1'))

            # 使用conv1d fft计算
            return conv1d_fft(x, K, dim = -3, weight_dim = -2)

        x = apply_learned_ema_with_damping(x, self.alphas, self.dampen_factors)

        if self.bidirectional:
            x_reversed = apply_learned_ema_with_damping(x_reversed, self.reverse_alphas, self.reverse_dampen_factors)
            x_reversed = torch.flip(x_reversed, dims = (1,))
            x = torch.cat((x, x_reversed), dim = -2)

        # 可能归一化头部
        x = self.norm_heads(x)

        # 合并头部和输出
        return einsum('... h d, h d -> ... d', x, self.reduction)

# Mega Layer
# 单头注意力 + 多头EMA，然后是类似GRU的门控

class MegaLayer(nn.Module):
    # 定义MegaLayer模块
    def __init__(
        self,
        *,
        dim = 128,
        ema_heads = 16,
        attn_dim_qk = 64,
        attn_dim_value = 256,
        laplacian_attn_fn = False,
        causal = True,
        norm_mhesa_heads = False
    ):
        # 初始化函数
        super().__init__()

        # 单头注意力
        self.single_headed_attn = SingleHeadedAttention(
            dim = dim,
            dim_qk = attn_dim_qk,
            dim_value = attn_dim_value,
            causal = causal,
            laplacian_attn_fn = laplacian_attn_fn
        )

        # 多头EMA
        self.multi_headed_ema = MultiHeadedEMA(
            dim = dim,
            heads = ema_heads,
            bidirectional = not causal,
            norm_mhesa_heads = norm_mhesa_heads
        )

        # 重置门
        self.to_reset_gate = nn.Sequential(
            nn.Linear(dim, attn_dim_value),
            nn.SiLU()
        )

        # 更新门
        self.to_update_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # 计算H的方程式14
        self.Wh = nn.Parameter(torch.randn(dim, dim))
        self.Uh = nn.Parameter(torch.randn(attn_dim_value, dim))
        self.bh = nn.Parameter(torch.randn(dim))
    # 定义前向传播函数，接受输入 x 和残差 residual，默认为 None
    def forward(self, x, residual = None):
        # 如果没有传入残差，则使用 x 作为默认值
        residual = default(residual, x)

        # 使用多头 EMA 模型处理输入 x
        ema_output = self.multi_headed_ema(x)
        # 使用单头注意力模型处理 EMA 输出和输入 x
        attn_output = self.single_headed_attn(ema_output, x)

        # 计算重置门和更新门
        reset_gate = self.to_reset_gate(ema_output)
        update_gate = self.to_update_gate(ema_output)

        # 使用重置门对注意力输出进行门控
        gated_attn_output = attn_output * reset_gate

        # 计算 H，根据方程式 14
        H = F.silu(ema_output @ self.Wh + gated_attn_output @ self.Uh + self.bh)

        # 更新门
        return update_gate * H + (1 - update_gate) * residual
# 定义一个前馈神经网络层，包括线性层、GELU激活函数和另一个线性层
def FeedForward(dim, ff_mult):
    # 计算隐藏层维度
    dim_hidden = int(dim * ff_mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),  # 输入维度为dim，输出维度为dim_hidden的线性层
        nn.GELU(),  # GELU激活函数
        nn.Linear(dim_hidden, dim)  # 输入维度为dim_hidden，输出维度为dim的线性层
    )

# 定义一个Mega类，继承自nn.Module
class Mega(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        ff_mult = 2,
        pre_norm = False,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)  # 创建一个嵌入层，用于将token映射为dim维向量
        self.pre_norm = pre_norm  # 是否使用预层归一化

        self.layers = nn.ModuleList([])  # 创建一个空的ModuleList，用于存储多个MegaLayer

        # 循环depth次，创建多个MegaLayer及其相关层，并添加到layers中
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MegaLayer(dim = dim, **kwargs),  # MegaLayer层
                nn.LayerNorm(dim),  # LayerNorm层
                FeedForward(dim = dim, ff_mult = ff_mult),  # FeedForward层
                nn.LayerNorm(dim)  # LayerNorm层
            ]))

        # 创建一个Sequential模块，用于将模型输出映射为num_tokens维度
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim) if pre_norm else nn.Identity(),  # 如果使用预层归一化，则使用LayerNorm，否则使用Identity
            nn.Linear(dim, num_tokens)  # 线性层，将dim维度映射为num_tokens维度
        )

    # 前向传播函数
    def forward(self, x):
        pre_norm = self.pre_norm
        post_norm = not self.pre_norm

        x = self.token_emb(x)  # 将输入的token映射为dim维度的向量

        # 遍历layers中的每个MegaLayer及其相关层
        for mega_layer, mega_norm, ff, ff_norm in self.layers:
            mega_maybe_prenorm = mega_norm if pre_norm else identity
            ff_maybe_prenorm = ff_norm if pre_norm else identity

            mega_maybe_postnorm = mega_norm if post_norm else identity
            ff_maybe_postnorm = ff_norm if post_norm else identity

            x = mega_layer(mega_maybe_prenorm(x), x)  # MegaLayer的前向传播

            x = mega_maybe_postnorm(x)  # 可能的后层归一化

            x = ff(ff_maybe_prenorm(x)) + x  # FeedForward层的前向传播

            x = ff_maybe_postnorm(x)  # 可能的后层归一化

        return self.to_logits(x)  # 将输出映射为num_tokens维度
```