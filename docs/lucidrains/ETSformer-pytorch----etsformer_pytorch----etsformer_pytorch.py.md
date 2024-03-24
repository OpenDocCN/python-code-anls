# `.\lucidrains\ETSformer-pytorch\etsformer_pytorch\etsformer_pytorch.py`

```
# 从 math 模块中导入 pi 常数
from math import pi
# 从 collections 模块中导入 namedtuple 类
from collections import namedtuple

# 导入 torch 库
import torch
# 从 torch.nn.functional 模块中导入 F
import torch.nn.functional as F
# 从 torch 模块中导入 nn 和 einsum
from torch import nn, einsum

# 从 scipy.fftpack 模块中导入 next_fast_len 函数
from scipy.fftpack import next_fast_len
# 从 einops 模块中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 einops.layers.torch 模块中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 定义一个名为 Intermediates 的命名元组，包含 growth_latents、seasonal_latents 和 level_output 三个字段
Intermediates = namedtuple('Intermediates', ['growth_latents', 'seasonal_latents', 'level_output'])

# 定义一个名为 exists 的函数，用于判断值是否存在
def exists(val):
    return val is not None

# 定义一个名为 fourier_extrapolate 的函数，用于对信号进行傅立叶外推
def fourier_extrapolate(signal, start, end):
    # 获取信号所在设备
    device = signal.device
    # 对信号进行傅立叶变换
    fhat = torch.fft.fft(signal)
    fhat_len = fhat.shape[-1]
    # 生成时间序列
    time = torch.linspace(start, end - 1, end - start, device=device, dtype=torch.complex64)
    # 生成频率序列
    freqs = torch.linspace(0, fhat_len - 1, fhat_len, device=device, dtype=torch.complex64)
    # 计算傅立叶外推结果
    res = fhat[..., None, :] * (1.j * 2 * pi * freqs[..., None, :] * time[..., :, None] / fhat_len).exp() / fhat_len
    return res.sum(dim=-1).real

# 定义一个名为 InputEmbedding 的函数，用于输入嵌入
def InputEmbedding(time_features, model_dim, kernel_size=3, dropout=0.):
    return nn.Sequential(
        Rearrange('b n d -> b d n'),
        nn.Conv1d(time_features, model_dim, kernel_size=kernel_size, padding=kernel_size // 2),
        nn.Dropout(dropout),
        Rearrange('b d n -> b n d'),
    )

# 定义一个名为 FeedForward 的函数，用于前馈网络
def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.Sigmoid(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
        nn.Dropout(dropout)
    )

# 定义一个名为 FeedForwardBlock 的类，用于前馈网络块
class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        **kwargs
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, **kwargs)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.post_norm(x + self.ff(x))

# encoder 相关类

## 多头指数平滑注意力机制
# 定义一个名为 conv1d_fft 的函数，用于一维卷积和快速傅立叶变换
def conv1d_fft(x, weights, dim=-2, weight_dim=-1):
    # 算法 3
    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = torch.fft.rfft(x, n=fast_len, dim=dim)
    f_weight = torch.fft.rfft(weights, n=fast_len, dim=weight_dim)

    f_v_weight = f_x * rearrange(f_weight.conj(), '... -> ... 1')
    out = torch.fft.irfft(f_v_weight, fast_len, dim=dim)
    out = out.roll(-1, dims=(dim,))

    indices = torch.arange(start=fast_len - N, end=fast_len, dtype=torch.long, device=x.device)
    out = out.index_select(dim, indices)
    return out

# 定义一个名为 MHESA 的类，用于多头指数平滑注意力机制
class MHESA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        dropout=0.,
        norm_heads=False
    ):
        super().__init__()
        self.heads = heads
        self.initial_state = nn.Parameter(torch.randn(heads, dim // heads))

        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.randn(heads))

        self.norm_heads = nn.Sequential(
            Rearrange('b n (h d) -> b (h d) n', h=heads),
            nn.GroupNorm(heads, dim),
            Rearrange('b (h d) n -> b n (h d)', h=heads)
        ) if norm_heads else nn.Identity()

        self.project_in = nn.Linear(dim, dim)
        self.project_out = nn.Linear(dim, dim)

    # 定义一个名为 naive_Aes 的方法，用于执行简单指数平滑
    def naive_Aes(self, x, weights):
        n, h = x.shape[-2], self.heads

        # 在附录 A.1 中 - 算法 2

        arange = torch.arange(n, device=x.device)

        weights = repeat(weights, '... l -> ... t l', t=n)
        indices = repeat(arange, 'l -> h t l', h=h, t=n)

        indices = (indices - rearrange(arange + 1, 't -> 1 t 1')) % n

        weights = weights.gather(-1, indices)
        weights = self.dropout(weights)

        # 因果关系

        weights = weights.tril()

        # 矩阵相乘

        output = einsum('b h n d, h m n -> b h m d', x, weights)
        return output
    # 定义前向传播函数，接受输入 x 和是否使用 naive 模式的标志
    def forward(self, x, naive = False):
        # 获取输入 x 的形状信息，包括 batch size (b), 序列长度 (n), 特征维度 (d), 头数 (h), 设备信息 (device)
        b, n, d, h, device = *x.shape, self.heads, x.device

        # 线性投影输入数据
        x = self.project_in(x)

        # 将投影后的数据按头数拆分
        x = rearrange(x, 'b n (h d) -> b h n d', h = h)

        # 计算时间差异
        x = torch.cat((
            repeat(self.initial_state, 'h d -> b h 1 d', b = b),
            x
        ), dim = -2)

        x = x[:, :, 1:] - x[:, :, :-1]

        # 准备指数 alpha
        alpha = self.alpha.sigmoid()
        alpha = rearrange(alpha, 'h -> h 1')

        # 计算权重
        arange = torch.arange(n, device = device)
        weights = alpha * (1 - alpha) ** torch.flip(arange, dims = (0,))

        # 根据是否使用 naive 模式选择不同的计算方式
        if naive:
            output = self.naive_Aes(x, weights)
        else:
            output = conv1d_fft(x, weights)

        # 计算初始状态的贡献
        init_weight = (1 - alpha) ** (arange + 1)
        init_output = rearrange(init_weight, 'h n -> h n 1') * rearrange(self.initial_state, 'h d -> h 1 d')

        output = output + init_output

        # 合并头部信息
        output = rearrange(output, 'b h n d -> b n (h d)')

        # 对输出进行规范化处理
        output = self.norm_heads(output)

        # 返回输出结果
        return self.project_out(output)
## frequency attention

# 定义频率注意力模块
class FrequencyAttention(nn.Module):
    def __init__(
        self,
        *,
        K = 4,
        dropout = 0.
    ):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 对输入数据进行傅立叶变换
        freqs = torch.fft.rfft(x, dim = 1)

        # 获取振幅

        amp = freqs.abs()
        amp = self.dropout(amp)

        # 获取前K个振幅值 - 用于季节性，被标记为注意力

        topk_amp, _ = amp.topk(k = self.K, dim = 1, sorted = True)

        # 掩盖所有振幅低于前K个最小值的频率

        topk_freqs = freqs.masked_fill(amp < topk_amp[:, -1:], 0.+0.j)

        # 反向傅立叶变换

        return torch.fft.irfft(topk_freqs, dim = 1)

## level module

# 定义水平模块
class Level(nn.Module):
    def __init__(self, time_features, model_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.]))
        self.to_growth = nn.Linear(model_dim, time_features)
        self.to_seasonal = nn.Linear(model_dim, time_features)

    def forward(self, x, latent_growth, latent_seasonal):
        # 按附录A.2中的方程式

        n, device = x.shape[1], x.device

        alpha = self.alpha.sigmoid()

        arange = torch.arange(n, device = device)
        powers = torch.flip(arange, dims = (0,))

        # 用于具有季节性项的原始时间序列信号（从频率注意力中减去）的Aes

        seasonal =self.to_seasonal(latent_seasonal)
        Aes_weights = alpha * (1 - alpha) ** powers
        seasonal_normalized_term = conv1d_fft(x - seasonal, Aes_weights)

        # 辅助项

        growth = self.to_growth(latent_growth)
        growth_smoothing_weights = (1 - alpha) ** powers
        growth_term = conv1d_fft(growth, growth_smoothing_weights)

        return seasonal_normalized_term + growth_term

# 解码器类

class LevelStack(nn.Module):
    def forward(self, x, num_steps_forecast):
        return repeat(x[:, -1], 'b d -> b n d', n = num_steps_forecast)

class GrowthDampening(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.dampen_factor = nn.Parameter(torch.randn(heads))

    def forward(self, growth, *, num_steps_forecast):
        device, h = growth.device, self.heads

        dampen_factor = self.dampen_factor.sigmoid()

        # 类似于level stack，它获取最后一个增长用于预测

        last_growth = growth[:, -1]
        last_growth = rearrange(last_growth, 'b l (h d) -> b l 1 h d', h = h)

        # 准备每个头部的减弱因子和幂

        dampen_factor = rearrange(dampen_factor, 'h -> 1 1 1 h 1')
        powers = (torch.arange(num_steps_forecast, device = device) + 1)
        powers = rearrange(powers, 'n -> 1 1 n 1 1')

        # 遵循论文中的Eq(2)

        dampened_growth = last_growth * (dampen_factor ** powers).cumsum(dim = 2)
        return rearrange(dampened_growth, 'b l n h d -> b l n (h d)')

# 主类

class ETSFormer(nn.Module):
    def __init__(
        self,
        *,
        model_dim,
        time_features = 1,
        embed_kernel_size = 3,
        layers = 2,
        heads = 8,
        K = 4,
        dropout = 0.
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言模型维度必须能够被头数整除
        assert (model_dim % heads) == 0, 'model dimension must be divisible by number of heads'
        # 初始化模型维度和时间特征
        self.model_dim = model_dim
        self.time_features = time_features

        # 创建输入嵌入层
        self.embed = InputEmbedding(time_features, model_dim, kernel_size = embed_kernel_size, dropout = dropout)

        # 初始化编码器层列表
        self.encoder_layers = nn.ModuleList([])

        # 循环创建编码器层
        for ind in range(layers):
            is_last_layer = ind == (layers - 1)

            # 添加编码器层
            self.encoder_layers.append(nn.ModuleList([
                FrequencyAttention(K = K, dropout = dropout),
                MHESA(dim = model_dim, heads = heads, dropout = dropout),
                FeedForwardBlock(dim = model_dim) if not is_last_layer else None,
                Level(time_features = time_features, model_dim = model_dim)
            ]))

        # 创建增长阻尼模块
        self.growth_dampening_module = GrowthDampening(dim = model_dim, heads = heads)

        # 线性层将潜在变量转换为时间特征
        self.latents_to_time_features = nn.Linear(model_dim, time_features)
        # 创建级别堆栈
        self.level_stack = LevelStack()

    def forward(
        self,
        x,
        *,
        num_steps_forecast = 0,
        return_latents = False
    ):
        # 检查输入是否只有一个时间特征
        one_time_feature = x.ndim == 2

        if one_time_feature:
            x = rearrange(x, 'b n -> b n 1')

        z = self.embed(x)

        latent_growths = []
        latent_seasonals = []

        # 遍历编码器层
        for freq_attn, mhes_attn, ff_block, level in self.encoder_layers:
            latent_seasonal = freq_attn(z)
            z = z - latent_seasonal

            latent_growth = mhes_attn(z)
            z = z - latent_growth

            if exists(ff_block):
                z = ff_block(z)

            x = level(x, latent_growth, latent_seasonal)

            latent_growths.append(latent_growth)
            latent_seasonals.append(latent_seasonal)

        latent_growths = torch.stack(latent_growths, dim = -2)
        latent_seasonals = torch.stack(latent_seasonals, dim = -2)

        latents = Intermediates(latent_growths, latent_seasonals, x)

        if num_steps_forecast == 0:
            return latents

        latent_seasonals = rearrange(latent_seasonals, 'b n l d -> b l d n')
        extrapolated_seasonals = fourier_extrapolate(latent_seasonals, x.shape[1], x.shape[1] + num_steps_forecast)
        extrapolated_seasonals = rearrange(extrapolated_seasonals, 'b l d n -> b l n d')

        dampened_growths = self.growth_dampening_module(latent_growths, num_steps_forecast = num_steps_forecast)
        level = self.level_stack(x, num_steps_forecast = num_steps_forecast)

        summed_latents = dampened_growths.sum(dim = 1) + extrapolated_seasonals.sum(dim = 1)
        forecasted = level + self.latents_to_time_features(summed_latents)

        if one_time_feature:
            forecasted = rearrange(forecasted, 'b n 1 -> b n')

        if return_latents:
            return forecasted, latents

        return forecasted
# 分类包装器

class MultiheadLayerNorm(nn.Module):
    def __init__(self, dim, heads = 1, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(heads, 1, dim))  # 初始化可学习参数 g
        self.b = nn.Parameter(torch.zeros(heads, 1, dim))  # 初始化可学习参数 b

    def forward(self, x):
        std = torch.var(x, dim = -1, unbiased = False, keepdim = True).sqrt()  # 计算标准差
        mean = torch.mean(x, dim = -1, keepdim = True)  # 计算均值
        return (x - mean) / (std + self.eps) * self.g + self.b  # 返回归一化后的结果

class ClassificationWrapper(nn.Module):
    def __init__(
        self,
        *,
        etsformer,
        num_classes = 10,
        heads = 16,
        dim_head = 32,
        level_kernel_size = 3,
        growth_kernel_size = 3,
        seasonal_kernel_size = 3,
        dropout = 0.
    ):
        super().__init__()
        assert isinstance(etsformer, ETSFormer)
        self.etsformer = etsformer
        model_dim = etsformer.model_dim
        time_features = etsformer.time_features

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.queries = nn.Parameter(torch.randn(heads, dim_head))  # 初始化查询参数

        self.growth_to_kv = nn.Sequential(
            Rearrange('b n d -> b d n'),  # 重新排列张量维度
            nn.Conv1d(model_dim, inner_dim * 2, growth_kernel_size, bias = False, padding = growth_kernel_size // 2),  # 一维卷积层
            Rearrange('... (kv h d) n -> ... (kv h) n d', kv = 2, h = heads),  # 重新排列张量维度
            MultiheadLayerNorm(dim_head, heads = 2 * heads),  # 多头层归一化
        )

        self.seasonal_to_kv = nn.Sequential(
            Rearrange('b n d -> b d n'),  # 重新排列张量维度
            nn.Conv1d(model_dim, inner_dim * 2, seasonal_kernel_size, bias = False, padding = seasonal_kernel_size // 2),  # 一维卷积层
            Rearrange('... (kv h d) n -> ... (kv h) n d', kv = 2, h = heads),  # 重新排列张量维度
            MultiheadLayerNorm(dim_head, heads = 2 * heads),  # 多头层归一化
        )

        self.level_to_kv = nn.Sequential(
            Rearrange('b n t -> b t n'),  # 重新排列张量维度
            nn.Conv1d(time_features, inner_dim * 2, level_kernel_size, bias = False, padding = level_kernel_size // 2),  # 一维卷积层
            Rearrange('b (kv h d) n -> b (kv h) n d', kv = 2, h = heads),  # 重新排列张量维度
            MultiheadLayerNorm(dim_head, heads = 2 * heads),  # 多头层归一化
        )

        self.to_out = nn.Linear(inner_dim, model_dim)  # 线性变换层

        self.to_logits = nn.Sequential(
            nn.LayerNorm(model_dim),  # 层归一化
            nn.Linear(model_dim, num_classes)  # 线性变换层
        )

    def forward(self, timeseries):
        latent_growths, latent_seasonals, level_output = self.etsformer(timeseries)  # 获取ETSFormer的输出

        latent_growths = latent_growths.mean(dim = -2)  # 沿着指定维度计算均值
        latent_seasonals = latent_seasonals.mean(dim = -2)  # 沿着指定维度计算均值

        # queries, key, values

        q = self.queries * self.scale  # 缩放查询参数

        kvs = torch.cat((
            self.growth_to_kv(latent_growths),  # 经过growth_to_kv处理
            self.seasonal_to_kv(latent_seasonals),  # 经过seasonal_to_kv处理
            self.level_to_kv(level_output)  # 经过level_to_kv处理
        ), dim = -2)

        k, v = kvs.chunk(2, dim = 1)  # 按维度切分张量

        # cross attention pooling

        sim = einsum('h d, b h j d -> b h j', q, k)  # 执行张量乘法
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()  # 减去最大值并断开梯度

        attn = sim.softmax(dim = -1)  # softmax操作
        attn = self.dropout(attn)  # dropout操作

        out = einsum('b h j, b h j d -> b h d', attn, v)  # 执行张量乘法
        out = rearrange(out, 'b ... -> b (...)')  # 重新排列张量维度

        out = self.to_out(out)  # 线性变换

        # project to logits

        return self.to_logits(out)  # 返回logits
```