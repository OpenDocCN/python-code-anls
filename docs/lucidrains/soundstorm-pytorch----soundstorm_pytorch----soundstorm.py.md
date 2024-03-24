# `.\lucidrains\soundstorm-pytorch\soundstorm_pytorch\soundstorm.py`

```py
import math
from random import random, randrange  # 导入随机数生成相关函数
from functools import wraps  # 导入wraps装饰器
from contextlib import nullcontext  # 导入nullcontext上下文管理器
from collections import namedtuple  # 导入namedtuple命名元组
from pathlib import Path  # 导入Path路径操作模块

import torch  # 导入PyTorch深度学习库
from torch.cuda.amp import autocast  # 导入自动混合精度计算
from torch import Tensor, nn, einsum  # 导入张量、神经网络、einsum函数
import torch.nn.functional as F  # 导入PyTorch中的函数模块

from einops import rearrange, reduce, repeat, unpack, pack  # 导入einops库中的函数
from einops.layers.torch import Rearrange, EinMix  # 导入einops库中的层函数

from beartype import beartype  # 导入beartype类型检查库
from beartype.door import is_bearable  # 导入is_bearable函数
from beartype.typing import Union, Dict, Optional, List, Optional  # 导入beartype中的类型注解

from soundstorm_pytorch.attend import Attend  # 导入Attend模块

from spear_tts_pytorch import TextToSemantic  # 导入TextToSemantic模块

from audiolm_pytorch import SoundStream  # 导入SoundStream模块
from audiolm_pytorch import HubertWithKmeans, FairseqVQWav2Vec  # 导入HubertWithKmeans和FairseqVQWav2Vec模块

from gateloop_transformer import SimpleGateLoopLayer as GateLoop  # 导入SimpleGateLoopLayer模块

from tqdm import tqdm  # 导入tqdm进度条模块

# helpers

def exists(val):
    return val is not None  # 判断值是否存在

def default(val, d):
    return val if exists(val) else d  # 如果值存在则返回值，否则返回默认值

def divisible_by(numer, denom):
    return (numer % denom) == 0  # 判断是否可以整除

def calc_same_padding(kernel_size):
    pad = kernel_size // 2  # 计算padding值
    return (pad, pad - (kernel_size + 1) % 2)  # 返回padding元组

def eval_decorator(fn):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()  # 设置模型为评估模式
        out = fn(model, *args, **kwargs)  # 调用函数
        model.train(was_training)  # 恢复模型训练模式
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])  # 计算top-k值
    val, ind = logits.topk(k, dim = -1)  # 获取top-k值和索引
    probs = torch.full_like(logits, float('-inf'))  # 创建与logits相同形状的全为负无穷的张量
    probs.scatter_(2, ind, val)  # 根据索引填充top-k值
    return probs  # 返回top-k值

def log(t, eps = 1e-10):
    return torch.log(t + eps)  # 计算对数

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)  # 生成均匀分布的噪声
    return -log(-log(noise))  # 计算Gumbel噪声

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)  # 计算Gumbel采样

# prob helpers

def sample_prob(prob):
    return random() < prob  # 根据概率进行采样

def coin_flip():
    return sample_prob(0.5)  # 以0.5的概率进行翻转

# tensor helpers

@beartype
def get_mask_subset_prob(
    mask: Tensor,
    prob: Union[float, Tensor],
    min_mask: int = 0
):
    batch, seq, device = *mask.shape, mask.device  # 获取批次大小、序列长度和设备信息

    if isinstance(prob, Tensor):
        prob = rearrange(prob, 'b -> b 1')  # 重排概率张量的维度

    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)  # 计算要屏蔽的数量
    logits = torch.rand((batch, seq), device = device)  # 生成随机数张量
    logits = logits.masked_fill(~mask, -1)  # 根据mask进行填充

    randperm = logits.argsort(dim = -1).argsort(dim = -1).float()  # 对logits进行排序

    num_padding = (~mask).sum(dim = -1, keepdim = True)  # 计算填充数量
    randperm -= num_padding  # 减去填充数量

    subset_mask = randperm < num_to_mask  # 生成子集mask
    subset_mask.masked_fill_(~mask, False)  # 根据mask进行填充
    return subset_mask  # 返回子集mask

# schedules

def linear_schedule(t):
    return 1 - t  # 线性调度函数

def cosine_schedule(t):
    """ https://arxiv.org/abs/2202.04200 """
    return torch.cos(t * math.pi / 2)  # 余弦调度函数

# rotary embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # 计算频率
        self.register_buffer("inv_freq", inv_freq, persistent = False)  # 注册缓冲区

    @property
    def device(self):
        return next(self.buffers()).device  # 获取设备信息

    @autocast(enabled = False)
    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)  # 生成序列长度张量
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)  # 计算频率
        freqs = torch.cat((freqs, freqs), dim = -1)  # 拼接频率
        return freqs  # 返回频率

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)  # 将张量分成两部分
    return torch.cat((-x2, x1), dim=-1)  # 拼接张量

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())  # 应用旋转位置嵌入

# t5 relative positional bias

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale = 1.,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化缩放因子
        self.scale = scale
        # 初始化桶的数量
        self.num_buckets = num_buckets
        # 初始化最大距离
        self.max_distance = max_distance
        # 创建相对注意力偏置的嵌入层
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        # 初始化返回值
        ret = 0
        # 计算相对位置的负值
        n = -relative_position

        # 将桶的数量减半
        num_buckets //= 2
        # 根据n是否小于0来更新ret
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        # 计算最大精确值
        max_exact = num_buckets // 2
        # 判断n是否小于最大精确值
        is_small = n < max_exact

        # 计算大值时的结果
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()

        # 将大值结果限制在桶的范围内
        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1)
        )

        # 根据is_small选择n或者val_if_large
        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        # 返回参数的设备信息
        return next(self.parameters()).device

    def forward(self, n):
        # 生成长度为n的张量
        pos = torch.arange(n, device = self.device).long()
        # 计算相对位置
        rel_pos = rearrange(pos, 'j -> 1 j') - rearrange(pos, 'i -> i 1')

        # 计算相对位置的桶
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        # 获取相对注意力偏置的值
        values = self.relative_attention_bias(rp_bucket)

        # 重排values的维度
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale
# 定义 Swish 激活函数模块
class Swish(nn.Module):
    # 前向传播函数
    def forward(self, x):
        return x * x.sigmoid()

# 定义 GLU 模块
class GLU(nn.Module):
    # 初始化函数
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    # 前向传播函数
    def forward(self, x):
        # 将输入张量按维度分割成两部分
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

# 定义 DepthWiseConv1d 模块
class DepthWiseConv1d(nn.Module):
    # 初始化函数
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        # 创建深度卷积层
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    # 前向传播函数
    def forward(self, x, mask=None):
        # 如果存在掩码，则将掩码应用到输入张量上
        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 n')
            x = x.masked_fill(~mask, 0.)

        # 对输入张量进行填充
        x = F.pad(x, self.padding)
        # 进行卷积操作
        out = self.conv(x)

        # 如果存在掩码，则将掩码应用到输出张量上
        if exists(mask):
            out = out.masked_fill(~mask, 0.)

        return out

# 定义 Scale 模块
class Scale(nn.Module):
    # 初始化函数
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    # 前向传播函数
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

# 定义 ChanLayerNorm 模块
class ChanLayerNorm(nn.Module):
    # 初始化函数
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    # 前向传播函数
    def forward(self, x):
        eps = 1e-6 if x.dtype == torch.float32 else 1e-4
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * var.clamp(min=eps).rsqrt() * self.gamma

# 定义 PreNorm 模块
class PreNorm(nn.Module):
    # 初始化函数
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    # 前向传播函数
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 定义 Attention 模块
class Attention(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        flash=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = Attend(
            flash=flash,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    # 前向传播函数
    def forward(
        self,
        x,
        context=None,
        mask=None,
        rotary_emb=None,
        attn_bias=None
    ):
        n, device, h, has_context = x.shape[-2], x.device, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v, mask=mask, attn_bias=attn_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义 FeedForward 模块
class FeedForward(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    # 前向传播函数
    def forward(self, x):
        return self.net(x)

# 定义 ConformerConvModule 模块
class ConformerConvModule(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        causal=False,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.
    # 定义一个类，继承自 nn.Module
    ):
        # 调用父类的构造函数
        super().__init__()

        # 计算内部维度
        inner_dim = dim * expansion_factor
        # 计算填充大小，如果是因果卷积则填充为 (kernel_size - 1, 0)
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        # 定义网络结构 net1，包括 LayerNorm、Rearrange、Conv1d 和 GLU 激活函数
        self.net1 = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1)
        )

        # 定义深度卷积层 ds_conv
        self.ds_conv = DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding)

        # 定义网络结构 net2，包括 Swish 激活函数、ChanLayerNorm、Conv1d、Rearrange 和 Dropout
        self.net2 = nn.Sequential(
            Swish(),
            ChanLayerNorm(inner_dim),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    # 定义前向传播函数
    def forward(self, x, mask = None):
        # 使用 net1 进行前向传播
        x = self.net1(x)
        # 使用 ds_conv 进行前向传播
        x = self.ds_conv(x, mask = mask)
        # 使用 net2 进行前向传播
        return self.net2(x)
# Conformer Block

# 定义 ConformerBlock 类
class ConformerBlock(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 维度
        dim_head = 64,  # 头的维度
        heads = 8,  # 头的数量
        ff_mult = 4,  # FeedForward 层的倍数
        conv_expansion_factor = 2,  # 卷积扩展因子
        conv_kernel_size = 31,  # 卷积核大小
        attn_dropout = 0.,  # 注意力机制的 dropout
        attn_flash = True,  # 是否使用闪存注意力
        ff_dropout = 0.,  # FeedForward 层的 dropout
        conv_dropout = 0.,  # 卷积层的 dropout
        conv_causal = False,  # 是否是因果卷积
        use_gateloop_layers = False  # 是否使用门循环层
    ):
        super().__init__()
        # 创建第一个 FeedForward 层
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        # 如果使用门循环层，则创建 GateLoop 层
        self.gateloop = GateLoop(dim) if use_gateloop_layers else None

        # 创建注意力机制层
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, flash = attn_flash)
        # 创建 ConformerConvModule 层
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        # 创建第二个 FeedForward 层
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        # 对注意力机制层进行预归一化
        self.attn = PreNorm(dim, self.attn)
        # 对第一个 FeedForward 层进行预归一化
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        # 对第二个 FeedForward 层进行预归一化
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        # 创建 LayerNorm 层
        self.post_norm = nn.LayerNorm(dim)

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None,
        rotary_emb = None,
        attn_bias = None
    ):
        # 第一个 FeedForward 层
        x = self.ff1(x) + x

        # 如果存在门循环层，则应用门循环层
        if exists(self.gateloop):
            x = self.gateloop(x) + x

        # 注意力机制层
        x = self.attn(x, mask = mask, rotary_emb = rotary_emb, attn_bias = attn_bias) + x
        # 卷积层
        x = self.conv(x, mask = mask) + x
        # 第二个 FeedForward 层
        x = self.ff2(x) + x
        # LayerNorm 层
        x = self.post_norm(x)
        return x

# Conformer

# 定义 Conformer 类
class Conformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        *,
        depth,  # 深度
        dim_head = 64,  # 头的维度
        heads = 8,  # 头的数量
        ff_mult = 4,  # FeedForward 层的倍数
        conv_expansion_factor = 2,  # 卷积扩展因子
        conv_kernel_size = 31,  # 卷积核大小
        attn_dropout = 0.,  # 注意力机制的 dropout
        ff_dropout = 0.,  # FeedForward 层的 dropout
        conv_dropout = 0.,  # 卷积层的 dropout
        conv_causal = False,  # 是否是因果卷积
        attn_flash = True,  # 是否使用闪存注意力
        t5_rel_pos_bias = False,  # 是否使用 T5 相对位置偏置
        use_gateloop_layers = True  # 是否使用门循环层
    ):
        super().__init__()

        # 断言，确保闪存注意力和学习偏置不兼容
        assert not (t5_rel_pos_bias and attn_flash), 'flash attention is not compatible with learned bias'

        self.dim = dim
        self.layers = nn.ModuleList([])

        # 如果不使用 T5 相对位置偏置，则创建 RotaryEmbedding 层
        self.rotary_emb = RotaryEmbedding(dim_head) if not t5_rel_pos_bias else None
        # 如果使用 T5 相对位置偏置，则创建 T5RelativePositionBias 层
        self.rel_pos_bias = T5RelativePositionBias(dim_head ** 0.5, heads = heads) if t5_rel_pos_bias else None

        # 根据深度循环创建 ConformerBlock 层
        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                conv_dropout = conv_dropout,
                conv_causal = conv_causal,
                attn_flash = attn_flash,
                use_gateloop_layers = use_gateloop_layers
            ))

    # 前向传播函数
    def forward(self, x, mask = None):
        seq_len = x.shape[-2]

        # 如果存在 RotaryEmbedding 层，则创建旋转嵌入
        rotary_emb = self.rotary_emb(seq_len) if exists(self.rotary_emb) else None
        # 如果存在 T5RelativePositionBias 层，则创建注意力偏置
        attn_bias = self.rel_pos_bias(seq_len) if exists(self.rel_pos_bias) else None

        # 遍历 ConformerBlock 层进行前向传播
        for block in self.layers:
            x = block(
                x,
                mask = mask,
                rotary_emb = rotary_emb,
                attn_bias = attn_bias
            )

        return x

# conformer with sum reduction across quantized tokens at the beginning, along with heads

# 定义 ConformerWrapper 类
class ConformerWrapper(nn.Module):

    @beartype
    # 初始化函数
    def __init__(
        self,
        *,
        codebook_size,  # 代码本大小
        num_quantizers,  # 量化器数量
        conformer: Union[Conformer, Dict[str, any]],  # Conformer 模型
        grouped_quantizers = 1  # 分组量化器数量
        ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化属性conformer
        self.conformer = conformer

        # 如果conformer是字典类型，则使用Conformer类初始化self.conformer
        if isinstance(conformer, dict):
            self.conformer = Conformer(**self.conformer)

        # 获取conformer的维度
        dim = self.conformer.dim

        # 根据grouped_quantizers的值判断是否需要进行embedding投影
        self.embedding_proj = nn.Sequential(
            nn.Linear(dim * grouped_quantizers, dim),
            nn.LayerNorm(dim)
        ) if grouped_quantizers > 1 else nn.Identity()

        # 计算带有mask的量化器代码数量
        num_codes_with_mask = codebook_size + 1
        num_effective_quantizers = num_quantizers * grouped_quantizers

        # 初始化代码嵌入层
        self.code_embeds = nn.Embedding(num_codes_with_mask * num_effective_quantizers, dim)

        # 注册缓冲区，存储量化器偏移和mask标记
        self.register_buffer('quantizer_offsets', torch.arange(num_effective_quantizers) * num_codes_with_mask, persistent=False)
        self.register_buffer('mask_tokens', self.quantizer_offsets + num_codes_with_mask, persistent=False)

        # 初始化其他属性
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codes_with_mask = num_codes_with_mask
        self.num_quantizers = num_quantizers
        self.grouped_quantizers = grouped_quantizers

        # 初始化头部
        self.heads = nn.Sequential(
            nn.Linear(dim, dim * num_effective_quantizers),
            Rearrange('b n (h d) -> b (n h) d', h=num_effective_quantizers)
        )

        # 每个量化器代码本都需要自己的logits权重和偏置矩阵
        # 使用EinMix和einops实现
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b (n gq) d -> b n gq d', gq=num_effective_quantizers),
            EinMix(
                'b n gq d -> b n gq l',
                weight_shape='gq d l',
                bias_shape='gq l',
                gq=num_effective_quantizers,
                l=codebook_size,
                d=dim
            ),
            Rearrange('b ... d -> b (...) d')
        )

    def forward(
        self,
        x,
        *,
        mask=None,
        cond=None,
        sum_embeds=None,
        return_embeddings=False,
        return_logits_and_embeddings=False
    ):
        """
        einops notation:
        b - batch
        n - sequence
        g - groups
        q - quantizers
        d - feature dimension
        """

        # 获取x的维度信息
        n, q, g = x.shape[-1], self.num_quantizers, self.grouped_quantizers
        assert divisible_by(n, g * q), 'sequence must be divisible by number of quantizers'

        # 重排x的维度
        x = rearrange(x, 'b (n gq) -> b n gq', gq=g * q)
        x = x + self.quantizer_offsets

        # 对x进行代码嵌入
        x = self.code_embeds(x)

        # 对x进行降维操作
        x = reduce(x, 'b n (g q) d -> b n (g d)', 'sum', g=g)

        # 对x进行嵌入投影
        x = self.embedding_proj(x)

        # 如果存在sum_embeds，则将其加到x上
        if exists(sum_embeds):
            x = x + sum_embeds

        # 如果存在cond，则将其加到x上
        if exists(cond):
            if cond.ndim == 2:
                cond = rearrange(cond, 'b d -> b 1 d')

            x = x + cond

        # 对x进行Conformer处理
        x = self.conformer(x, mask=mask)
        embeds = self.heads(x)

        # 如果需要返回嵌入向量或者没有to_logits，则返回embeds
        if return_embeddings or not exists(self.to_logits):
            return embeds

        # 获取logits
        logits = self.to_logits(embeds)

        # 如果需要返回logits和嵌入向量，则返回logits和embeds
        if return_logits_and_embeddings:
            return logits, embeds

        return logits
# 定义 LogitHead 类，用于处理主要的 logits 以及自我 token 评论
class LogitHead(nn.Module):
    def __init__(
        self,
        net: ConformerWrapper,
        logit_dim
    ):
        super().__init__()
        self.net = net
        dim = net.dim
        self.to_logits = nn.Linear(dim, logit_dim)

    def forward(self, x):
        # 获取网络的嵌入表示
        embed = self.net(x, return_embeddings = True)
        return self.to_logits(embed)

# 定义 LossBreakdown 命名元组，包含生成器损失和评论家损失
LossBreakdown = namedtuple('LossBreakdown', ['generator_loss', 'critic_loss'])

# 定义 SoundStorm 类，用于处理声音数据
class SoundStorm(nn.Module):

    @beartype
    def __init__(
        self,
        net: ConformerWrapper,
        *,
        soundstream: Optional[SoundStream] = None,
        spear_tts_text_to_semantic: Optional[TextToSemantic] = None,
        wav2vec: Optional[Union[HubertWithKmeans, FairseqVQWav2Vec]] = None,
        steps = 18,
        self_cond = False,
        self_cond_train_prob = 0.75,
        no_replace_prob = 0.15,          # 原始 MLM 论文中指定的一定比例的 tokens 会保持不变
        random_token_prob = 0.1,         # 原始 MLM 论文中指定的一定比例的 tokens 会被替换为随机 token
        schedule = 'linear',
        can_mask_prev_unmasked = False,  # 当解除 mask 时，是否可以重新 mask 之前未 mask 的 tokens
        self_token_critic = False,       # 是否使用自我 token 评论家
        critic_loss_weight = 1.,
        num_semantic_token_ids = None,
        semantic_pad_id = -1,
        pad_id = None,
        wav2vec_target_sample_hz = None,
        wav2vec_downsample_factor = None,
        codec_target_sample_hz = None,
        codec_downsample_factor = None,
    @property
    def device(self):
        return next(self.net.parameters()).device

    def load(self, path, strict = True):
        # 加载模型参数
        # 返回 pkg，以便如果此函数从 Trainer 函数调用中调用，则 Trainer 也可以访问从检查点加载的 package
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        self.load_state_dict(pkg['model'], strict = strict)
        return pkg

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        num_latents = None,
        *,
        mask = None,
        texts: Optional[Union[List[str], Tensor]] = None,
        cond_semantic_token_ids = None,
        prompt_acoustic_token_ids = None,
        seconds = None,
        batch_size = None,
        start_temperature = 1.,
        filter_thres = 0.7,
        noise_level_scale = 1.,
        num_full_sampling_levels = 1,
        text_to_semantic_generate_kwargs: dict = {},
        spec_decode = False,
        spec_decode_gamma = 5,
        **kwargs
    # 定义一个方法，用于获取条件信息
    def maybe_get_condition(self, token_ids = None, length = None):
        # 断言条件：如果传入的 token_ids 存在，则应该开启文本条件化，反之亦然
        assert not (exists(token_ids) ^ self.should_condition), 'you either have text-conditioning turned on and have not passed in any conditioning semantic token ids, or vice versa'

        # 如果 token_ids 不存在，则返回 None
        if not exists(token_ids):
            return None

        # 根据是否存在文本到语义的映射，选择是否开启 torch 的无梯度上下文
        context = torch.no_grad if exists(self.text_to_semantic) else nullcontext

        # 在上下文中执行以下代码块
        with context():
            # 创建一个 mask，用于过滤掉语义填充标记
            mask = token_ids != self.semantic_pad_id

            # 如果存在文本到语义的映射，并且自动设置了 eos 语义标记 id
            if exists(self.text_to_semantic) and self.text_to_semantic.autoset_eos_id['speech']:
                # 进一步过滤掉 eos 语义标记 id
                mask &= token_ids != self.num_semantic_token_ids

            # 将不符合 mask 的 token_ids 替换为 0
            token_ids = token_ids.masked_fill(~mask, 0)

            # 获取语义标记的嵌入
            semantic_tokens = self.semantic_token_emb(token_ids)
            # 将语义标记转换为模型维度的条件 tokens
            cond_tokens = self.semantic_cond_to_model_dim(semantic_tokens)

            # 将填充部分的值设为 0，让网络学习处理
            cond_tokens = cond_tokens.masked_fill(~rearrange(mask, '... -> ... 1'), 0.)

        # 需要插值条件 tokens，以使语义和向量量化 tokens 在时间上对齐
        cond_length = cond_tokens.shape[-2]

        # 计算目标条件长度
        target_cond_length = math.ceil(cond_length * (self.wav2vec_downsample_factor / self.wav2vec_target_sample_hz) / (self.codec_downsample_factor / self.codec_target_sample_hz))

        # 由于 PyTorch 不支持 1D 插值，将数据转换为 2D 进行插值
        if cond_length != target_cond_length:
            cond_tokens = rearrange(cond_tokens, 'b n d -> b d n 1')
            cond_tokens = F.interpolate(cond_tokens, (target_cond_length, 1), mode = 'bilinear')
            cond_tokens = rearrange(cond_tokens, 'b d n 1 -> b n d')

        # 根据长度是否存在，决定是截断还是填充条件 tokens
        cond_length = cond_tokens.shape[-2]

        if exists(length):
            if cond_length < length:
                cond_tokens = F.pad(cond_tokens, (0, 0, 0, length - cond_length), value = 0.)
            elif cond_length > length:
                cond_tokens = cond_tokens[:, :length]

        # 返回处理后的条件 tokens
        return cond_tokens

    # 定义前向传播方法
    def forward(
        self,
        x,
        *,
        mask = None,
        cond_semantic_token_ids = None,
        only_train_generator = False,
        only_train_critic = False,
        generator_sample_temperature = None,
        **kwargs
```