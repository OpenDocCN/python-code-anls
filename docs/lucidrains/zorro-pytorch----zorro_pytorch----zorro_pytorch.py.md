# `.\lucidrains\zorro-pytorch\zorro_pytorch\zorro_pytorch.py`

```
# 导入所需的模块和类
from enum import Enum
import functools
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Tuple, Optional, Union

from torchaudio.transforms import Spectrogram

# 定义枚举类型 TokenTypes，包含音频、视频、融合和全局四种类型
class TokenTypes(Enum):
    AUDIO = 0
    VIDEO = 1
    FUSION = 2
    GLOBAL = 3

# 定义一些通用的函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 返回参数列表中第一个存在的参数，如果都不存在则返回 None
def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# 返回小于等于 n 的最接近的 divisor 的倍数
def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor

# 将输入转换为元组，如果输入不是元组则返回 (t, t)
def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

# 对可迭代对象进行累积乘法
def cum_mul(it):
    return functools.reduce(lambda x, y: x * y, it, 1)

# 判断 numer 是否能被 denom 整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 装饰器

# 保证函数只调用一次的装饰器
def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# 用 once 装饰的 print 函数，确保只打印一次
print_once = once(print)

# 无偏置的 Layernorm 类
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# GEGLU 激活函数
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

# FeedForward 网络结构
def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

# 注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        attn_mask = None
    ):
        x = self.norm(x)
        kv_x = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_x).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 主类 Zorro
class Zorro(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_fusion_tokens = 16,
        audio_patch_size: Union[int, Tuple[int, int]] = 16,
        video_patch_size: Union[int, Tuple[int, int]] = 16,
        video_temporal_patch_size = 2,
        video_channels = 3,
        spec_n_fft = 128,
        spec_power = 2,
        spec_win_length = 24,
        spec_hop_length = None,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 80,
        spec_aug_time_mask = 80,
        return_token_types: Tuple[TokenTypes] = (TokenTypes.AUDIO, TokenTypes.VIDEO, TokenTypes.FUSION)
        ):
        # 调用父类的构造函数
        super().__init__()
        # 设置最大返回标记数为返回标记类型列表的长度
        self.max_return_tokens = len(return_token_types)

        # 存储返回标记类型列表
        self.return_token_types = return_token_types
        # 将返回标记类型列表转换为张量
        return_token_types_tensor = torch.tensor(list(map(lambda t: t.value, return_token_types)))
        # 将返回标记类型张量注册为缓冲区
        self.register_buffer('return_token_types_tensor', return_token_types_tensor, persistent=False)

        # 初始化返回标记张量
        self.return_tokens = nn.Parameter(torch.randn(self.max_return_tokens, dim))
        # 初始化注意力池
        self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)

        # 音频输入

        # 设置音频块大小
        self.audio_patch_size = audio_patch_height, audio_patch_width = pair(audio_patch_size)

        # 初始化频谱图
        self.spec = Spectrogram(
            n_fft=spec_n_fft,
            power=spec_power,
            win_length=spec_win_length,
            hop_length=spec_hop_length,
            pad=spec_pad,
            center=spec_center,
            pad_mode=spec_pad_mode
        )

        # 计算音频输入维度
        audio_input_dim = cum_mul(self.audio_patch_size)
        # 将音频转换为标记
        self.audio_to_tokens = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1=audio_patch_height, p2=audio_patch_width),
            nn.LayerNorm(audio_input_dim),
            nn.Linear(audio_input_dim, dim),
            nn.LayerNorm(dim)
        )

        # 视频输入

        # 设置视频块大小
        self.video_patch_size = (video_temporal_patch_size, *pair(video_patch_size))

        # 计算视频输入维度
        video_input_dim = cum_mul(self.video_patch_size) * video_channels
        video_patch_time, video_patch_height, video_patch_width = self.video_patch_size

        # 将视频转换为标记
        self.video_to_tokens = nn.Sequential(
            Rearrange('b c (t p1) (h p2) (w p3) -> b t h w (c p1 p2 p3)', p1=video_patch_time, p2=video_patch_height, p3=video_patch_width),
            nn.LayerNorm(video_input_dim),
            nn.Linear(video_input_dim, dim),
            nn.LayerNorm(dim)
        )

        # 融合标记

        # 初始化融合标记
        self.fusion_tokens = nn.Parameter(torch.randn(num_fusion_tokens, dim))

        # transformer

        # 初始化层列表
        self.layers = nn.ModuleList([])

        # 循环创建指定数量的层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

        # 初始化层归一化
        self.norm = LayerNorm(dim)

    def forward(
        self,
        *,
        audio,
        video,
        return_token_indices: Optional[Tuple[int]] = None
        ):
        # 获取音频的批次大小和设备信息
        batch, device = audio.shape[0], audio.device
    
        # 验证视频是否可以被分块
        assert all([divisible_by(numer, denom) for denom, numer in zip(self.video_patch_size, tuple(video.shape[-3:]))]), f'video shape {video.shape[-3:]} needs to be divisible by {self.video_patch_size}'

        # 如果音频产生的二维频谱图不是patch大小的倍数，则自动裁剪
        audio = self.spec(audio)

        height, width = audio.shape[-2:]
        patch_height, patch_width = self.audio_patch_size

        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): # 只要打印，直到修复为止
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        audio = audio[..., :rounded_height, :rounded_width]

        # 转换为tokens
        audio_tokens = self.audio_to_tokens(audio)
        video_tokens = self.video_to_tokens(video)
        fusion_tokens = repeat(self.fusion_tokens, 'n d -> b n d', b = batch)

        # 构建所有tokens
        audio_tokens, fusion_tokens, video_tokens = map(lambda t: rearrange(t, 'b ... d -> b (...) d'), (audio_tokens, fusion_tokens, video_tokens))
        tokens, ps = pack((
            audio_tokens,
            fusion_tokens,
            video_tokens
        ), 'b * d')

        # 构建mask（即zorro）
        token_types = torch.tensor(list((
            *((TokenTypes.AUDIO.value,) * audio_tokens.shape[-2]),
            *((TokenTypes.FUSION.value,) * fusion_tokens.shape[-2]),
            *((TokenTypes.VIDEO.value,) * video_tokens.shape[-2]),
        )), device = device, dtype = torch.long)

        token_types_attend_from = rearrange(token_types, 'i -> i 1')
        token_types_attend_to = rearrange(token_types, 'j -> 1 j')

        # 逻辑是每个模态，包括融合，都可以关注自己
        zorro_mask = token_types_attend_from == token_types_attend_to

        # 融合可以关注所有
        zorro_mask = zorro_mask | (token_types_attend_from == TokenTypes.FUSION.value)

        # 注意力和前馈
        for attn, ff in self.layers:
            tokens = attn(tokens, attn_mask = zorro_mask) + tokens
            tokens = ff(tokens) + tokens

        tokens = self.norm(tokens)

        # 最终注意力池化 - 每个模态池token只能关注自己的tokens
        return_tokens = self.return_tokens
        return_token_types_tensor = self.return_token_types_tensor

        if exists(return_token_indices):
            assert len(set(return_token_indices)) == len(return_token_indices), 'all indices must be unique'
            assert all([indice < self.max_return_tokens for indice in return_token_indices]), 'indices must range from 0 to max_num_return_tokens - 1'

            return_token_indices = torch.tensor(return_token_indices, dtype = torch.long, device = device)

            return_token_types_tensor = return_token_types_tensor[return_token_indices]
            return_tokens = return_tokens[return_token_indices]

        return_tokens = repeat(return_tokens, 'n d -> b n d', b = batch)
        pool_mask = rearrange(return_token_types_tensor, 'i -> i 1') == token_types_attend_to
        # 全局查询可以关注所有tokens
        pool_mask = pool_mask | rearrange(return_token_types_tensor, 'i -> i 1') == torch.ones_like(token_types_attend_to, dtype=torch.long) * TokenTypes.GLOBAL.value

        pooled_tokens = self.attn_pool(return_tokens, context = tokens, attn_mask = pool_mask) + return_tokens

        return pooled_tokens
```