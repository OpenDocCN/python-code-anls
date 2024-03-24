# `.\lucidrains\BS-RoFormer\bs_roformer\bs_roformer.py`

```py
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

# 导入 einops 库
from einops import rearrange, pack, unpack

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 将单个张量按照指定模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 将单个张量按照指定模式解包
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 归一化模块

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# 注意力模块

# 前馈神经网络模块
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

# 注意力模块
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

# Transformer 模块
class Transformer(Module):
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
        flash_attn = True
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, flash = flash_attn),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# bandsplit 模块

class BandSplit(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 将输入 x 沿着指定维度进行分割
        x = x.split(self.dim_inputs, dim = -1)

        # 初始化一个空列表用于存储输出结果
        outs = []
        # 遍历分割后的输入和对应的特征函数
        for split_input, to_feature in zip(x, self.to_features):
            # 对每个分割后的输入应用对应的特征函数，得到分割后的输出
            split_output = to_feature(split_input)
            # 将分割后的输出添加到输出列表中
            outs.append(split_output)

        # 将所有输出结果堆叠在一起，沿着指定维度
        return torch.stack(outs, dim = -2)
# 定义一个多层感知机（MLP）模型
def MLP(
    dim_in,
    dim_out,
    dim_hidden = None,
    depth = 1,
    activation = nn.Tanh
):
    # 如果未指定隐藏层维度，则默认为输入维度
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    # 构建每一层的维度信息
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    # 遍历每一层，构建网络结构
    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        # 添加线性层
        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        # 如果是最后一层，则跳过激活函数
        if is_last:
            continue

        # 添加激活函数
        net.append(activation())

    return nn.Sequential(*net)

# 定义一个MaskEstimator类，继承自Module
class MaskEstimator(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: Tuple[int, ...],
        depth,
        mlp_expansion_factor = 4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        # 遍历输入维度，构建MLP网络
        for dim_in in dim_inputs:
            net = []

            # 构建MLP网络
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden = dim_hidden, depth = depth),
                nn.GLU(dim = -1)
            )

            self.to_freqs.append(mlp)

    # 前向传播函数
    def forward(self, x):
        # 沿着倒数第二维度拆分输入
        x = x.unbind(dim = -2)

        outs = []

        # 遍历每个频段特征和对应的MLP网络
        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim = -1)

# 主类

# 默认频率带数目
DEFAULT_FREQS_PER_BANDS = (
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  12, 12, 12, 12, 12, 12, 12, 12,
  24, 24, 24, 24, 24, 24, 24, 24,
  48, 48, 48, 48, 48, 48, 48, 48,
  128, 129,
)

# 定义BSRoformer类，继承自Module
class BSRoformer(Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo = False,
        num_stems = 1,
        time_transformer_depth = 2,
        freq_transformer_depth = 2,
        freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,  # 在论文中，它们将其分成约60个频带，测试时先用1
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        flash_attn = True,
        dim_freqs_in = 1025,
        stft_n_fft = 2048,
        stft_hop_length = 512, # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length = 2048,
        stft_normalized = False,
        stft_window_fn: Optional[Callable] = None,
        mask_estimator_depth = 2,
        multi_stft_resolution_loss_weight = 1.,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size = 147,
        multi_stft_normalized = False,
        multi_stft_window_fn: Callable = torch.hann_window
        ):
        # 调用父类的构造函数
        super().__init__()

        # 设置音频是否为立体声
        self.stereo = stereo
        # 根据音频是否为立体声确定音频通道数
        self.audio_channels = 2 if stereo else 1
        # 设置音频分离的声音轨道数
        self.num_stems = num_stems

        # 初始化神经网络层列表
        self.layers = ModuleList([])

        # 设置变压器的参数
        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            flash_attn = flash_attn,
            norm_output = False
        )

        # 创建时间旋转嵌入和频率旋转嵌入
        time_rotary_embed = RotaryEmbedding(dim = dim_head)
        freq_rotary_embed = RotaryEmbedding(dim = dim_head)

        # 根据深度循环创建变压器层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(depth = time_transformer_depth, rotary_embed = time_rotary_embed, **transformer_kwargs),
                Transformer(depth = freq_transformer_depth, rotary_embed = freq_rotary_embed, **transformer_kwargs)
            ]))

        # 初始化最终的归一化层
        self.final_norm = RMSNorm(dim)

        # 设置短时傅里叶变换的参数
        self.stft_kwargs = dict(
            n_fft = stft_n_fft,
            hop_length = stft_hop_length,
            win_length = stft_win_length,
            normalized = stft_normalized
        )

        # 设置短时傅里叶变换的窗口函数
        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        # 计算频率数量
        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex = True).shape[1]

        # 断言频率段数大于1
        assert len(freqs_per_bands) > 1
        # 断言频率段数之和等于总频率数
        assert sum(freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        # 计算每个频率段的复数频率数量
        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        # 初始化频率段分割层
        self.band_split = BandSplit(
            dim = dim,
            dim_inputs = freqs_per_bands_with_complex
        )

        # 初始化掩蔽估计器列表
        self.mask_estimators = nn.ModuleList([])

        # ��据声音轨道数循环创建掩蔽估计器
        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim = dim,
                dim_inputs = freqs_per_bands_with_complex,
                depth = mask_estimator_depth
            )

            self.mask_estimators.append(mask_estimator)

        # 设置多分辨率短时傅里叶变换损失的权重和参数
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length = multi_stft_hop_size,
            normalized = multi_stft_normalized
        )

    # 前向传播函数
    def forward(
        self,
        raw_audio,
        target = None,
        return_loss_breakdown = False
```