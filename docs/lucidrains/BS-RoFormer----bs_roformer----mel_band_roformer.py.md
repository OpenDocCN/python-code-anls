# `.\lucidrains\BS-RoFormer\bs_roformer\mel_band_roformer.py`

```
# 导入所需的库
from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# 导入自定义的模块
from bs_roformer.attend import Attend

# 导入类型提示相关的库
from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

# 导入旋转嵌入相关的库
from rotary_embedding_torch import RotaryEmbedding

# 导入 einops 库中的函数和层
from einops import rearrange, pack, unpack, reduce, repeat
from einops.layers.torch import Rearrange

# 导入 librosa 库中的滤波器
from librosa import filters

# 定义一些辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 将张量打包成指定模式的形状
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包后的张量解包成原始形状
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 在指定维度上进行填充
def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# 定义 RMS 归一化层
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# 定义前馈神经网络层
class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 定义注意力机制层
class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        rotary_embed = None,
        flash = True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash = flash, dropout = dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义线性注意力机制层
class LinearAttention(Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_head = 32,
        heads = 8,
        scale = 8,
        flash = False,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv = 3, h = heads)
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = Attend(
            scale = scale,
            dropout = dropout,
            flash = flash
        )

        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        x
        ):
        # 对输入进行归一化处理
        x = self.norm(x)

        # 将输入转换为查询、键、值
        q, k, v = self.to_qkv(x)

        # 对查询、键进行 L2 归一化
        q, k = map(l2norm, (q, k))
        # 对查询进行温度调节
        q = q * self.temperature.exp()

        # 进行注意力计算
        out = self.attend(q, k, v)

        # 将输出转换为最终输出
        return self.to_out(out)
# 定义一个名为 Transformer 的类，继承自 Module 类
class Transformer(Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        norm_output = True,
        rotary_embed = None,
        flash_attn = True,
        linear_attn = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化 layers 属性为一个空的 ModuleList
        self.layers = ModuleList([])

        # 循环 depth 次
        for _ in range(depth):
            # 根据 linear_attn 参数选择不同的注意力机制
            if linear_attn:
                attn = LinearAttention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = flash_attn)
            else:
                attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, flash = flash_attn)

            # 将注意力机制和前馈网络添加到 layers 中
            self.layers.append(ModuleList([
                attn,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        # 根据 norm_output 参数选择是否使用 RMSNorm 或者 nn.Identity
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    # 前向传播函数
    def forward(self, x):

        # 遍历 layers 中的每个注意力机制和前馈网络
        for attn, ff in self.layers:
            # 执行注意力机制并将结果与输入相加
            x = attn(x) + x
            # 执行前馈网络并将结果与输入相加
            x = ff(x) + x

        # 对结果进行归一化处理
        return self.norm(x)

# 定义一个名为 BandSplit 的类，继承自 Module 类
class BandSplit(Module):
    # 初始化函数，接收多个参数
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: Tuple[int, ...]
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化 dim_inputs 属性
        self.dim_inputs = dim_inputs
        # 初始化 to_features 属性为一个空的 ModuleList
        self.to_features = ModuleList([])

        # 遍历 dim_inputs 中的每个维度
        for dim_in in dim_inputs:
            # 创建一个包含 RMSNorm 和 Linear 层的网络
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            # 将网络添加到 to_features 中
            self.to_features.append(net)

    # 前向传播函数
    def forward(self, x):
        # 将输入 x 按照 dim_inputs 进行分割
        x = x.split(self.dim_inputs, dim = -1)

        outs = []
        # 遍历分割后的输入和对应的网络
        for split_input, to_feature in zip(x, self.to_features):
            # 对分割后的输入进行处理并添加到 outs 中
            split_output = to_feature(split_input)
            outs.append(split_output)

        # 在指定维度上将结果拼接起来
        return torch.stack(outs, dim = -2)

# 定义一个名为 MLP 的函数
def MLP(
    dim_in,
    dim_out,
    dim_hidden = None,
    depth = 1,
    activation = nn.Tanh
):
    # 如果未指定隐藏层维度，则设置为输入维度
    dim_hidden = default(dim_hidden, dim_in)

    # 初始化网络列表
    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    # 遍历每一层的输入和输出维度
    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        # 添加线性层
        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        # 如果不是最后一层，则添加激活函数
        if is_last:
            continue

        net.append(activation())

    # 返回一个包含所有层的序列网络
    return nn.Sequential(*net)

# 定义一个名为 MaskEstimator 的类，继承自 Module 类
class MaskEstimator(Module):
    # 初始化函数，接收多个参数
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: Tuple[int, ...],
        depth,
        mlp_expansion_factor = 4
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化 dim_inputs 属性
        self.dim_inputs = dim_inputs
        # 初始化 to_freqs 属性为一个空的 ModuleList
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        # 遍历 dim_inputs 中的每个维度
        for dim_in in dim_inputs:
            net = []

            # 创建一个包含 MLP 和 GLU 层的网络
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden = dim_hidden, depth = depth),
                nn.GLU(dim = -1)
            )

            # 将网络添加到 to_freqs 中
            self.to_freqs.append(mlp)

    # 前向传播函数
    def forward(self, x):
        # 将输入 x 按照指定维度解绑
        x = x.unbind(dim = -2)

        outs = []

        # 遍历解绑后的输入和对应的网络
        for band_features, mlp in zip(x, self.to_freqs):
            # 对输入进行处理并添加到 outs 中
            freq_out = mlp(band_features)
            outs.append(freq_out)

        # 在指定维度上将结果拼接起来
        return torch.cat(outs, dim = -1)

# 定义一个名为 MelBandRoformer 的类，继承自 Module 类
class MelBandRoformer(Module):

    # 初始化函数
    @beartype
    # 初始化函数，设置模型参数
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo = False,
        num_stems = 1,
        time_transformer_depth = 2,
        freq_transformer_depth = 2,
        linear_transformer_depth = 1,
        num_bands = 60,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.1,
        ff_dropout = 0.1,
        flash_attn = True,
        dim_freqs_in = 1025,
        sample_rate = 44100,     # needed for mel filter bank from librosa
        stft_n_fft = 2048,
        stft_hop_length = 512,   # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length = 2048,
        stft_normalized = False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth = 1,
        multi_stft_resolution_loss_weight = 1.,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size = 147,
        multi_stft_normalized = False,
        multi_stft_window_fn: Callable = torch.hann_window,
        match_input_audio_length = False, # if True, pad output tensor to match length of input tensor
    # 前向传播函数，接收原始音频数据和目标数据，返回损失细分结果
    def forward(
        self,
        raw_audio,
        target = None,
        return_loss_breakdown = False
```