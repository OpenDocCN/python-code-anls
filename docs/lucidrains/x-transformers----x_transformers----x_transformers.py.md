# `.\lucidrains\x-transformers\x_transformers\x_transformers.py`

```py
# 导入数学库
import math
# 从 random 模块中导入 random 函数
from random import random
# 从 typing 模块中导入 Dict 类型提示
from typing import Dict
# 从 packaging 模块中导入 version 版本信息
from packaging import version

# 导入 torch 库
import torch
# 从 torch 库中导入 nn, einsum, Tensor
from torch import nn, einsum, Tensor
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F
# 从 torch.cuda.amp 模块中导入 autocast 函数
from torch.cuda.amp import autocast

# 导入 functools 模块中的 partial, wraps 函数
from functools import partial, wraps
# 导入 collections 模块中的 namedtuple 类
from collections import namedtuple
# 导入 dataclasses 模块中的 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块中导入 List, Callable, Optional, Union 类型提示
from typing import List, Callable, Optional, Union

# 从 einops 模块中导入 rearrange, repeat, reduce, pack, unpack 函数
from einops import rearrange, repeat, reduce, pack, unpack
# 从 einops.layers.torch 模块中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 从 x_transformers.attend 模块中导入 Attend, Intermediates 类
from x_transformers.attend import Attend, Intermediates
# 从 x_transformers.autoregressive_wrapper 模块中导入 AutoregressiveWrapper 类

# 常量定义

# 默认头部维度
DEFAULT_DIM_HEAD = 64

# 定义 LayerIntermediates 数据类
@dataclass
class LayerIntermediates:
    hiddens:            Optional[List[Tensor]] = None   # 所有隐藏层，在最终规范化之前（在预规范化架构中）
    last_hidden:        Optional[Tensor] = None         # 所有注意力层之后的最后一个隐藏层，在最终规范化之后
    attn_intermediates: Optional[List[Intermediates]] = None
    layer_hiddens:      Optional[List[Tensor]] = None
    attn_z_loss:        Optional[Tensor] = None
    mems:               Optional[Tensor] = None
    memory_tokens:      Optional[Tensor] = None

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 将变量转换为元组
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# 判断一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# 如果变量存在则执行函数
def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

# 至多一个为真
def at_most_one_of(*bools):
    return sum(map(int, bools)) <= 1

# 始终返回相同值
class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

# 不等于某个值
class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

# 等于某个值
class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

# 创建序列模块
def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# 张量辅助函数

# 返回张量的最小负值
def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# 对张量进行 L2 归一化
def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

# 在指定维度上填充张量
def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# 对多个掩码进行逻辑或操作
def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# 辅助损失函数

# 计算 z 损失
def calc_z_loss(
    pre_softmax_attns: List[Tensor],
    mask = None,
    weight = 1.
):
    # 在 https://arxiv.org/abs/2202.08906 中应用于专家混合路由器对数的相同损失
    # 在论文中，他们在一个小脚注中提到将其应用于注意力对数，具有稳定效果
    # 在 PaLM 中也作为措施之一使用

    lse = 0.

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim = -1)

    loss = torch.square(lse)
    loss = reduce(loss, 'b h n -> b n', 'sum')

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min = 1e-5)
    return loss * weight

# 初始化辅助函数

# 初始化为零
def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# 关键字参数辅助函数

# 选择并弹出键值对
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

# 根据条件将字典分组
def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)
# 检查字符串是否以指定前缀开头
def string_begins_with(prefix, str):
    return str.startswith(prefix)

# 根据键的前缀对字典进行分组
def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

# 根据前缀对字典进行分组并修剪前缀
def groupby_prefix_and_trim(prefix, d):
    # 根据前缀对字典进行分组
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    # 剔除前缀，生成新的字典
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# 结构化的 dropout，比传统的注意力 dropout 更有效
def dropout_seq(seq, mask, dropout):
    # 获取序列的形状和设备信息
    b, n, *_, device = *seq.shape, seq.device
    # 生成服从标准正态分布的随机数
    logits = torch.randn(b, n, device=device)

    # 如果存在掩码
    if exists(mask):
        # 获取 logits 中的最大负值
        mask_value = max_neg_value(logits)
        # 使用 mask_value 替换掩码为 False 的位置
        logits = logits.masked_fill(~mask, mask_value)

    # 计算保留的概率和保留的数量
    keep_prob = 1. - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    # 生成批次索引
    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    # 根据保留的索引获取序列的子集
    seq = seq[batch_indices, keep_indices]

    # 如果存在掩码
    if exists(mask):
        # 计算序列中每个样本的非零元素数量
        seq_counts = mask.sum(dim=-1)
        # 计算保留的元素数量
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(seq_keep_counts, 'b -> b 1')

        # 更新掩码
        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# 激活函数
class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

# 词嵌入
class TokenEmbedding(nn.Module):
    def __init__(self, dim, num_tokens, l2norm_embed=False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return l2norm(token_emb) if self.l2norm_embed else token_emb

# 绝对位置嵌入
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

# 缩放的正弦位置嵌入
class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale

class RelativePositionBias(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的缩放比例
        self.scale = scale
        # 设置是否使用因果关系
        self.causal = causal
        # 设置桶的数量
        self.num_buckets = num_buckets
        # 设置最大距离
        self.max_distance = max_distance
        # 创建相对注意力偏置的嵌入层
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    # 静态方法，用于计算相对位置的桶索引
    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        # 初始化返回值
        ret = 0
        # 计算相对位置的负值
        n = -relative_position
        # 如果不是因果关系，调整桶的数量
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        # 计算最大精确值
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # 计算大值时的桶索引
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        # 根据大小选择桶索引
        ret += torch.where(is_small, n, val_if_large)
        return ret

    # 返回设备信息
    @property
    def device(self):
        return next(self.parameters()).device

    # 前向传播函数
    def forward(self, i, j):
        # 获取设备信息
        device = self.device
        # 生成查询位置
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        # 生成键位置
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        # 计算相对位置
        rel_pos = k_pos[None, :] - q_pos[:, None]
        # 计算相对位置的桶索引
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        # 获取相对注意力偏置值
        values = self.relative_attention_bias(rp_bucket)
        # 重排形状
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale
class DynamicPositionBias(nn.Module):
    # 定义动态位置偏置类，继承自 nn.Module
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        # 初始化函数，接受维度、头数、深度、是否对距离取对数、是否进行归一化等参数
        super().__init__()
        # 调用父类的初始化函数
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        # 断言深度必须大于等于1
        self.log_distance = log_distance
        # 设置是否对距离取对数的标志

        self.mlp = nn.ModuleList([])
        # 初始化多层感知机模块列表

        self.mlp.append(Sequential(
            nn.Linear(1, dim),
            LayerNorm(dim) if norm else None,
            nn.SiLU()
        ))
        # 向多层感知机模块列表中添加线性层、归一化层和激活函数

        for _ in range(depth - 1):
            self.mlp.append(Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else None,
                nn.SiLU()
            ))
        # 根据深度循环添加多层感知机模块

        self.mlp.append(nn.Linear(dim, heads)
        # 向多层感知机模块列表中添加线性层，输出头数

    @property
    def device(self):
        # 定义设备属性，返回参数的设备
        return next(self.parameters()).device

    def forward(self, i, j):
        # 前向传播函数，接受输入i和j
        assert i == j
        # 断言i等于j
        n, device = j, self.device
        # 设置n为j，获取设备信息

        # get the (n x n) matrix of distances
        # 获取距离的(n x n)矩阵
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        # 连续位置多层感知机的输入
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)
        # 如果需要对距离取对数，则进行对数操作

        for layer in self.mlp:
            pos = layer(pos)
        # 遍历多层感知机模块列表，对位置进行处理

        # get position biases        
        # 获取位置偏置
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias
        # 返回位置偏置

class AlibiPositionalBias(nn.Module):
    # 定义Alibi位置偏置类，继承自 nn.Module
    def __init__(self, heads, total_heads, **kwargs):
        # 初始化函数，接受头数和总头数等参数
        super().__init__()
        # 调用父类的初始化函数
        self.heads = heads
        self.total_heads = total_heads

        slopes = Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
        # 初始化斜率和偏置

    def get_bias(self, i, j, device):
        # 定义获取偏置的函数，接受i、j和设备参数
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias
        # 返回偏置

    @staticmethod
    def _get_slopes(heads):
        # 定义获取斜率的静态方法，接受头数参数
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        # 定义获取2的幂次方斜率的函数

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)
        # 如果头数是2的幂次方，则返回对应斜率

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]
        # 否则返回最接近的2的幂次方的斜率和补充的斜率

    @property
    def device(self):
        # 定义设备属性，返回缓冲区的设备
        return next(self.buffers()).device

    def forward(self, i, j):
        # 前向传播函数，接受输入i和j
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]
        # 如果偏置存在且形状符合要求，则返回偏置

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes
        # 计算偏置并乘以斜率

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = 0)
        self.register_buffer('bias', bias, persistent = False)
        # 对未校准的头数进行填充

        return self.bias
        # 返回偏置

class RotaryEmbedding(nn.Module):
    # 定义旋转嵌入类，继承自 nn.Module
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
        # 初始化函数，接受维度、是否使用x位置、缩放基数、插值因子、基数和基数重新缩放因子等参数
    ):
        # 调用父类的构造函数
        super().__init__()
        # 根据 reddit 用户 bloc97 的建议，将旋转嵌入重新缩放到更长的序列长度，而无需微调
        # 与 NTK 文献有一定联系
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        # 计算频率的倒数
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        # 将频率的倒数作为缓冲区
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        # 设置插值因子
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            # 如果不使用 xpos，则将缩放设置为 None
            self.register_buffer('scale', None)
            return

        # 计算缩放
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        # 将缩放作为缓冲区
        self.register_buffer('scale', scale)

    # 根据序列长度进行前向传播
    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    # 禁用自动混合精度
    @autocast(enabled = False)
    def forward(self, t):
        # 计算最大位置
        max_pos = t.max()+1

        # 计算频率
        freqs = torch.einsum('i , j -> i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not exists(self.scale):
            return freqs, 1.

        # 计算幂次
        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale
# 定义一个函数，将输入张量 x 进行重新排列，将最后两个维度中的第一个维度 j 换到倒数第二个维度
def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    # 将 x 拆分为两部分 x1 和 x2，根据倒数第二个维度进行拆分
    x1, x2 = x.unbind(dim = -2)
    # 将 x2 取负值，然后与 x1 进行拼接，得到旋转后的张量
    return torch.cat((-x2, x1), dim = -1)

# 定义一个函数，应用旋转位置嵌入到输入张量 t 上
@autocast(enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    # 获取旋转维度和序列长度
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    # 截取与序列长度相同的频率信息
    freqs = freqs[-seq_len:, :]
    # 如果 scale 是张量，则截取与序列长度相同的部分
    scale = scale[-seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    # 如果输入张量 t 和频率信息 freqs 的维度分别为 4 和 3
    if t.ndim == 4 and freqs.ndim == 3:
        # 将频率信息维度扩展为 4 维
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # 部分旋转嵌入，Wang et al. GPT-J
    # 将输入张量 t 拆分为旋转部分 t 和未旋转部分 t_unrotated
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    # 计算旋转后的张量 t
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    # 将旋转后的张量 t 与未旋转部分拼接，返回结果
    return torch.cat((t, t_unrotated), dim = -1)

# norms

# 定义一个缩放层，用于对输入进行缩放
class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        # 对输入进行处理
        out = self.fn(x, **kwargs)
        # 定义缩放函数
        scale_fn = lambda t: t * self.value

        # 如果输出不是元组，则对输出进行缩放处理
        if not isinstance(out, tuple):
            return scale_fn(out)

        # 如果输出是元组，则对第一个元素进行缩放处理
        return (scale_fn(out[0]), *out[1:])

# 定义一个缩放归一化层
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim ** -0.5))

    def forward(self, x):
        # 计算输入张量的范数，并进行归一化处理
        norm = torch.norm(x, dim = -1, keepdim = True)
        return x / norm.clamp(min = self.eps) * self.g

# 定义一个 LayerNorm ��
class LayerNorm(nn.Module):
    def __init__(self, dim):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        # 使用 F.layer_norm 进行 LayerNorm 处理
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 如果 PyTorch 版本大于等于 2.1.0，则将 LayerNorm 替换为 nn.LayerNorm，并设置 bias 为 False
if version.parse(torch.__version__) >= version.parse('2.1.0'):
    LayerNorm = partial(nn.LayerNorm, bias = False)

# 定义一个 RMSNorm 层
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 对输入进行归一化处理，并乘以缩放因子和参数 g
        return F.normalize(x, dim = -1) * self.scale * self.g

# 定义一个简单的 RMSNorm 层
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        # 对输入进行归一化处理，并乘以缩放因子
        return F.normalize(x, dim = -1) * self.scale

# residual and residual gates

# 定义一个残差连接层
class Residual(nn.Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        # 如果存在残差缩放参数，则对残差进行缩放处理
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        # 如果缩放常数不为 1，则对残差进行缩放处理
        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        # 返回残差连接结果
        return x + residual

# 定义一个 GRU 门控单元层
class GRUGating(nn.Module):
    def __init__(self, dim, scale_residual = False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def forward(self, x, residual):
        # 如果存在残差缩放参数，则对残差进行缩放处理
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        # 使用 GRU 单元进行门控处理
        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        # 将门控输出重塑为与输入相同的形状
        return gated_output.reshape_as(x)

# token shifting

# 定义一个函数，对输入张量进行平移操作
def shift(t, amount, mask = None):
    if amount == 0:
        return t
    else:
        # 如果平移量大于输入张量的长度，则取最大值
        amount = min(amount, t.shape[1])

    # 如果存在掩码，则对输入张量进行掩码填充
    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    # 在指定维度上对输入张量进行填充操作
    return pad_at_dim(t, (amount, -amount), dim = - 2, value = 0.)

# 定义一个 ShiftTokens 类，用于对输入进行平移操作
class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)
    # 定义一个前向传播函数，接受输入 x 和关键字参数 kwargs
    def forward(self, x, **kwargs):
        # 从关键字参数 kwargs 中获取名为 'mask' 的值，如果没有则为 None
        mask = kwargs.get('mask', None)
        # 获取位移列表
        shifts = self.shifts
        # 计算段数
        segments = len(shifts)
        # 计算每个段的特征数
        feats_per_shift = x.shape[-1] // segments
        # 将输入 x 按特征数分割成多个张量
        splitted = x.split(feats_per_shift, dim=-1)
        # 将分割后的张量分为需要进行位移的段和剩余部分
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        # 对需要进行位移的段进行位移操作，使用 map 函数和 lambda 表达式
        segments_to_shift = list(map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts)))
        # 将位移后的段和剩余部分拼接在一起
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        # 调用 self.fn 函数对拼接后的张量进行处理，返回结果
        return self.fn(x, **kwargs)
# 定义 GLU 类，用于实现门控线性单元
class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        mult_bias = False
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.

    # 前向传播函数
    def forward(self, x):
        # 将输入通过线性变换后分成两部分
        x, gate = self.proj(x).chunk(2, dim = -1)
        # 返回门控线性单元的输出
        return x * self.act(gate) * self.mult_bias

# 定义 FeedForward 类，用于实现前馈神经网络
class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        glu_mult_bias = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        # 根据参数选择激活函数
        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        # 根据参数选择网络结构
        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias = glu_mult_bias)
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias = not no_bias),
                activation
            )

        # 构建前馈神经网络
        self.ff = Sequential(
            project_in,
            LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

        # 初始化最后一层线性层的权重为0
        if zero_init_output:
            init_zero_(self.ff[-1])

    # 前向传播函数
    def forward(self, x):
        return self.ff(x)

# 定义 Attention 类，用于实现注意力机制
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        dim_context = None,
        heads = 8,
        causal = False,
        flash = False,
        talking_heads = False,
        head_scale = False,
        sparse_topk = None,
        num_mem_kv = 0,
        dropout = 0.,
        on_attn = False,
        gate_value_heads = False,
        swiglu_values = False,
        gate_values = False,
        zero_init_output = False,
        max_attend_past = None,
        qk_norm = False,
        qk_norm_groups = 1,
        qk_norm_scale = 10,
        qk_norm_dim_scale = False,
        one_kv_head = False,
        kv_heads = None,
        shared_kv = False,
        value_dim_head = None,
        tensor_product = False,      # https://arxiv.org/abs/2208.06061
        add_zero_kv = False,         # same as add_zero_attn in pytorch
        rotary_embed_values = False,
        onnxable = False
    # 前向传播函数
    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None,
        mem_mask = None,
        return_intermediates = False,
        cache: Optional[Intermediates] = None,
class AttentionLayers(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        use_simple_rmsnorm = False,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        dynamic_pos_bias = False,
        dynamic_pos_bias_log_distance = False,
        dynamic_pos_bias_mlp_depth = 2,
        dynamic_pos_bias_norm = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        rotary_xpos = False,
        rotary_interpolation_factor = 1.,
        rotary_xpos_scale_base = 512,
        rotary_base_rescale_factor = 1.,
        custom_layers = None,
        sandwich_coef = None,
        par_ratio = None,
        weight_tie_layers = False,   # Albert - https://arxiv.org/abs/1909.11942
        layers_execute_order = None, # generalizes weight tying, can do arbitrary layer execution orders
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        pre_norm_has_final_norm = True,
        gate_residual = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        shift_tokens = 0,
        sandwich_norm = False,
        resi_dual = False,
        resi_dual_scale = 1.,
        zero_init_branch_output = False,
        layer_dropout = 0.,
        cross_attn_tokens_dropout = 0.,
        disable_abs_pos_emb = None,
        **kwargs
    # 前向传播函数，接收输入数据并进行模型计算
    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        self_attn_kv_mask = None,
        mems = None,
        mem_masks = None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        cache_age = 1,
        return_hiddens = False,
        rotary_pos_emb = None
class Encoder(AttentionLayers):
    # 定义编码器类，继承自AttentionLayers类
    def __init__(self, **kwargs):
        # 初始化函数，接受任意关键字参数
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        # 断言关键字参数中不包含'causal'，否则抛出异常
        super().__init__(causal = False, **kwargs)
        # 调用父类的初始化函数，设置causal参数为False，并传入其他关键字参数

class Decoder(AttentionLayers):
    # 定义解码器类，继承自AttentionLayers类
    def __init__(self, **kwargs):
        # 初始化函数，接受任意关键字参数
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        # 断言关键字参数中不包含'causal'，否则抛出异常
        super().__init__(causal = True, **kwargs)
        # 调用父类的初始化函数，设置causal参数为True，并传入其他关键字参数

class PrefixDecoder(AttentionLayers):
    # 定义前缀解码器类，继承自AttentionLayers类
    def __init__(self, **kwargs):
        # 初始化函数，接受任意关键字参数
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        # 断言关键字参数中不包含'causal'，否则抛出异常
        super().__init__(causal = False, **kwargs)
        # 调用父类的初始化函数，设置causal参数为False，并传入其他关键字参数

    def forward(
        self,
        x,
        *args,
        attn_mask = None,
        prefix_attn_len = None,
        **kwargs
    ):
        # 前向传播函数，接受输入x和任意位置参数args，注意力掩码attn_mask和前缀注意力长度prefix_attn_len，以及任意关键字参数kwargs
        b, n, device = x.shape[0], x.shape[1], x.device
        # 获取输入x的批量大小b，序列长度n，设备device
        causal_mask = torch.ones((n, n), device = device, dtype = torch.bool).triu(1)
        # 创建一个全为1的张量作为因果掩码，上三角部分为True，下三角部分为False

        forwarded_mask = ~causal_mask
        # 计算非因果掩码，即上三角部分为False，下三角部分为True

        if exists(prefix_attn_len):
            # 如果前缀注意力长度存在
            if isinstance(prefix_attn_len, int):
                # 如果前缀注意力长度是整数
                prefix_attn_len = torch.full((b,), prefix_attn_len, device = device)
                # 创建一个形状为(b,)的张量，填充值为前缀注意力长度，设备为device

            prefix_mask = torch.arange(n, device = device) < rearrange(prefix_attn_len, 'b -> b 1 1 1')
            # 创建前缀掩码，根据前缀注意���长度生成

            forwarded_mask = forwarded_mask | prefix_mask
            # 更新前向掩码，将前缀掩码应用到前向掩码中

        if exists(attn_mask):
            # 如果注意力掩码存在
            forwarded_mask = forwarded_mask & attn_mask
            # 更新前向掩码，将注意力掩码应用到前向掩码中

        return super().forward(x, *args, attn_mask = forwarded_mask, **kwargs)
        # 调用父类的前向传播函数，传入更新后的注意力掩码参数

class CrossAttender(AttentionLayers):
    # 定义交叉注意力层类，继承自AttentionLayers类
    def __init__(self, **kwargs):
        # 初始化函数，接受任意关键字参数
        super().__init__(cross_attend = True, only_cross = True, **kwargs)
        # 调用父类的初始化函数，设置cross_attend和only_cross参数为True，并传入其他关键字参数

class ViTransformerWrapper(nn.Module):
    # 定义ViTransformerWrapper类，继承自nn.Module类
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers: Encoder,
        channels = 3,
        num_classes = None,
        post_emb_norm = False,
        num_register_tokens = 0,
        emb_dropout = 0.
    ):
        # 初始化函数，接受命名关键字参数
        super().__init__()
        # 调用父类的初始化函数
        assert divisible_by(image_size, patch_size), 'image dimensions must be divisible by the patch size'
        # 断言图像尺寸能被补丁尺寸整除，否则抛出异常
        dim = attn_layers.dim
        # 获取注意力层的维度
        num_patches = (image_size // patch_size) ** 2
        # 计算图像中的补丁数量
        patch_dim = channels * patch_size ** 2
        # 计算补丁的维度

        self.patch_size = patch_size
        # 设置对象属性patch_size为传入的补丁尺寸

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 创建位置嵌入参数，形状为(1, num_patches, dim)，初始化为随机值

        has_register_tokens = num_register_tokens > 0
        # 判断是否存在注册令牌
        self.has_register_tokens = has_register_tokens
        # 设置对象属性has_register_tokens为判断结果

        if has_register_tokens:
            # 如果存在注册令牌
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))
            # 创建注册令牌参数，形状为(num_register_tokens, dim)，初始化为随机值

        self.patch_to_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim)
        )
        # 创建补丁到嵌入的序列模块

        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        # 根据post_emb_norm参数选择是否进行嵌入后的归一化
        self.dropout = nn.Dropout(emb_dropout)
        # 创建丢弃层，用于嵌入的丢弃

        self.attn_layers = attn_layers
        # 设置对象属性attn_layers为传入的注意力层

        self.mlp_head = nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()
        # 创建MLP头部，根据是否存在类别数量选择是否添加线性层

    def forward(
        self,
        img,
        return_embeddings = False,
        return_logits_and_embeddings = False
    ):
        # 前向传播函数，接受输入图像img，返回嵌入、逻辑和嵌入的标志
        b, p = img.shape[0], self.patch_size
        # 获取输入图像的批量大小b和补丁大小p

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # 重排输入图像，将其转换为形状为(b, h*w, p1*p2*c)的张量
        x = self.patch_to_embedding(x)
        # 将补丁转换为嵌入

        n = x.shape[1]
        # 获取嵌入的序列长度n

        x = x + self.pos_embedding[:, :n]
        # 添加位置嵌入到嵌入中

        x = self.post_emb_norm(x)
        # 对嵌入进行归一化
        x = self.dropout(x)
        # 对嵌入进行丢弃

        if self.has_register_tokens:
            # 如果存在注册令牌
            r = repeat(self.register_tokens, 'n d -> b n d', b = b)
            # 重复注册令牌，形状为(b, num_register_tokens, dim)
            x, ps = pack((x, r), 'b * d')
            # 打包嵌入和注册令牌

        embed = self.attn_layers(x)
        # 使用注意力层处理嵌入

        if self.has_register_tokens:
            # 如果存在注册令牌
            embed, _ = unpack(embed, ps, 'b * d')
            # 解包嵌入

        assert at_most_one_of(return_embeddings, return_logits_and_embeddings)
        # 断言返回嵌入和逻辑的标志中最多只有一个为True

        if not exists(self.mlp_head) or return_embeddings:
            # 如果MLP头部不存在或者需要返回嵌入
            return embed
            # 返回嵌入

        pooled = embed.mean(dim = -2)
        # 对嵌入进行平均池化
        logits = self.mlp_head(pooled)
        # 使用MLP头部生成逻辑

        if not return_logits_and_embeddings:
            # 如果不需要返回逻辑和嵌入
            return logits
            # 返回逻辑

        return logits, embed
        # 返回逻辑和嵌入
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        num_tokens,  # 令牌数量
        max_seq_len,  # 最大序列长度
        attn_layers: AttentionLayers,  # 注意力层对象
        embed_num_tokens: Dict[str, int] = dict(),  # 嵌入令牌数量的字典，默认为空
        emb_dim = None,  # 嵌入维度，默认为空
        max_mem_len = 0,  # 最大记忆长度，默认为0
        shift_mem_down = 0,  # 记忆向下移动的步数，默认为0
        emb_dropout = 0.,  # 嵌入层的dropout率，默认为0
        post_emb_norm = False,  # 是否对嵌入后进行归一化，默认为False
        num_memory_tokens = None,  # 记忆令牌数量，默认为空
        memory_tokens_interspersed_every = None,  # 记忆令牌插入间隔，默认为空
        tie_embedding = False,  # 是否共享嵌入权重，默认为False
        logits_dim = None,  # logits维度，默认为空
        use_abs_pos_emb = True,  # 是否使用绝对位置编码，默认为True
        scaled_sinu_pos_emb = False,  # 是否使用缩放的正弦位置编码，默认为False
        l2norm_embed = False,  # 是否对嵌入进行L2归一化，默认为False
        emb_frac_gradient = 1.,  # 梯度分配给嵌入的比例，默认为1
        attn_z_loss_weight = 1e-4,  # 注意力z损失的权重，默认为1e-4
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 获取注意力层的维度
        dim = attn_layers.dim
        # 如果嵌入维度为空，则设置为注意力层的维度
        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed
        # 创建令牌嵌入层对象
        self.token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed = l2norm_embed)

        # 判断是否不需要绝对位置编码
        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        # 根据条件选择不同的位置编码方式
        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed)

        # 初始化额外的嵌入层
        self.embeds = None

        # 如果有额外的嵌入令牌数量，则创建对应的嵌入层
        if len(embed_num_tokens) > 0:
            self.embeds = nn.ModuleDict({f'{name}_embed': nn.Embedding(num_tokens, emb_dim) for name, num_tokens in embed_num_tokens.items()})

        # 设置梯度分配给嵌入的比例
        self.emb_frac_gradient = emb_frac_gradient

        # 对嵌入后的结果进行归一化
        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        # 投影嵌入到指定维度
        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        # 初始化模型参数
        self.init_()

        # 设置logits的维度
        logits_dim = default(logits_dim, num_tokens)
        # 如果不共享嵌入权重，则创建线性层
        self.to_logits = nn.Linear(dim, logits_dim, bias = False) if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()

        # 设置记忆令牌
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # 判断是否可以进行缓存的kv解码
        self.can_cache_kv = self.num_memory_tokens == 0
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    # 初始化函数，根据是否进行L2归一化初始化权重
    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb.emb.weight, std = 1e-5)
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)
            return

        nn.init.kaiming_normal_(self.token_emb.emb.weight)

    # 前向传播函数
    def forward(
        self,
        x,  # 输入数据
        return_embeddings = False,  # 是否返回嵌入结果
        return_logits_and_embeddings = False,  # 是否返回logits和嵌入结果
        return_intermediates = False,  # 是否返回中间结果
        mask = None,  # 掩码
        return_mems = False,  # 是否返回记忆
        return_attn = False,  # 是否返回注意力
        mems = None,  # 记忆
        mem_masks = None,  # 记忆掩码
        pos = None,  # 位置编码
        prepend_embeds = None,  # 前置嵌入
        prepend_mask = None,  # 前置掩码
        embed_ids: Dict[str, Tensor] = dict(),  # 嵌入ID的字典
        sum_embeds = None,  # 嵌入求和
        return_attn_z_loss = False,  # 是否返回注意力z损失
        attn_z_loss_weight = 1e-4,  # 注意力z损失的权重
        seq_start_pos = None,  # 序列起始位置
        cache: Optional[LayerIntermediates] = None,  # 缓存
        **kwargs  # 其他参数
class XTransformer(nn.Module):
    # 定义 XTransformer 类，继承自 nn.Module
    def __init__(
        self,
        *,
        dim,
        tie_token_emb = False,
        ignore_index = -100,
        pad_value = 0,
        cross_attn_tokens_dropout = 0.,
        **kwargs
    ):
        # 初始化函数，接受一系列参数
        super().__init__()
        # 调用父类的初始化函数

        # 将参数按照前缀分组并修剪
        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        # 断言确保编码器或解码器的维度必须使用 `dim` 关键字设置
        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'

        # 从参数中选择并弹出 'num_tokens' 和 'max_seq_len'，并设置默认值
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        enc_transformer_kwargs['emb_dropout'] = enc_kwargs.pop('emb_dropout', 0)
        enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop('num_memory_tokens', None)
        enc_transformer_kwargs['scaled_sinu_pos_emb'] = enc_kwargs.pop('scaled_sinu_pos_emb', False)
        enc_transformer_kwargs['use_abs_pos_emb'] = enc_kwargs.pop('use_abs_pos_emb', True)

        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        # 设置交叉注意力的 tokens dropout 参数
        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # 创建编码器和解码器的 TransformerWrapper 对象
        self.encoder = TransformerWrapper(
            **enc_transformer_kwargs,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )

        self.decoder = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(dim = dim, cross_attend = True, **dec_kwargs)
        )

        # 如果 tie_token_emb 为 True，则共享解码器的 token_emb 层和编码器的 token_emb 层
        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb

        # 将解码器包装在 AutoregressiveWrapper 中
        self.decoder = AutoregressiveWrapper(self.decoder, ignore_index=ignore_index, pad_value=pad_value)

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, mask = None, attn_mask = None, **kwargs):
        # 生成函数，接受输入序列和输出序列的起始位置、长度等参数
        encodings = self.encoder(seq_in, mask = mask, attn_mask = attn_mask, return_embeddings = True)
        # 使用编码器对输入序列进行编码，返回编码结果
        return self.decoder.generate(seq_out_start, seq_len, context = encodings, context_mask = mask, **kwargs)
        # 使用解码器生成输出序列

    def forward(self, src, tgt, mask = None, attn_mask = None, src_prepend_embeds = None):
        # 前向传播函数，接受源序列、目标序列、掩码等参数

        # 使用编码器对源序列进行编码
        enc = self.encoder(src, mask = mask, attn_mask = attn_mask, prepend_embeds = src_prepend_embeds, return_embeddings = True)

        # 如果存在源序列的前置嵌入和掩码，则在掩码上进行填充
        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        # 如果处于训练状态且交叉注意力 tokens dropout 大于 0，则对编码结果进行 dropout
        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)

        # 使用解码器生成输出序列
        out = self.decoder(tgt, context = enc, context_mask = mask)
        return out
```