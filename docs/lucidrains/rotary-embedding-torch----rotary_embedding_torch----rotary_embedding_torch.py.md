# `.\lucidrains\rotary-embedding-torch\rotary_embedding_torch\rotary_embedding_torch.py`

```
# 从 math 模块中导入 pi 和 log 函数
from math import pi, log

# 导入 torch 模块
import torch
# 从 torch.nn 模块中导入 Module 和 ModuleList 类
from torch.nn import Module, ModuleList
# 从 torch.cuda.amp 模块中导入 autocast 函数
from torch.cuda.amp import autocast
# 从 torch 模块中导入 nn, einsum, broadcast_tensors, Tensor 类
from torch import nn, einsum, broadcast_tensors, Tensor

# 从 einops 模块中导入 rearrange, repeat 函数
from einops import rearrange, repeat

# 从 beartype 模块中导入 beartype 函数和相关类型
from beartype import beartype
from beartype.typing import Literal, Union, Optional

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 广播函数，用于 tortoise-tts

def broadcat(tensors, dim = -1):
    # 广播输入的张量
    broadcasted_tensors = broadcast_tensors(*tensors)
    # 沿指定维度拼接张量
    return torch.cat(broadcasted_tensors, dim = dim)

# 旋转嵌入的辅助函数

# 将输入张量沿最后两个维度旋转一半
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

# 应用旋转嵌入
@autocast(enabled = False)
def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1)

# 应用学习到的旋转

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# 类

# 旋转嵌入类
class RotaryEmbedding(Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        dim,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Union[
            Literal['lang'],
            Literal['pixel'],
            Literal['constant']
        ] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True
    ):
        # 调用父类的构造函数
        super().__init__()
        # 提议由 Reddit 用户 bloc97 提出，将旋转嵌入重新缩放到更长的序列长度，无需微调
        # 与 NTK 文献有一定联系
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        # 根据维度调整旋转角度
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        # 为频率设置参数
        self.freqs_for = freqs_for

        # 如果存在自定义频率，则使用自定义频率；否则根据不同的频率类型生成频率
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        # 设置是否缓存频率
        self.cache_if_possible = cache_if_possible

        # 初始化缓存频率和缩放
        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)

        # 将频率设置为可学习参数
        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        # 设置是否学习频率
        self.learned_freq = learned_freq

        # 为设备设置虚拟值
        self.tmp_store('dummy', torch.tensor(0))

        # 默认序列维度
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # 插值因子
        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # 是否使用 x 位置编码
        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store('scale', None)
            return

        # 计算 x 位置编码的缩放
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

    @property
    def device(self):
        # 返回虚拟值的设备
        return self.dummy.device

    def tmp_store(self, key, value):
        # 临时存储函数
        self.register_buffer(key, value, persistent = False)

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        # 获取序列位置
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, freq_seq_len = None):
        # 旋转查询或键
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        if exists(freq_seq_len):
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        freqs = self.forward(self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset), seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        # 旋转查询并使用缓存的键
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len
        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, freq_seq_len = k_len)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k
    # 旋转查询和键，用于生成旋转后的查询和键
    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        # 设置默认的序列维度
        seq_dim = default(seq_dim, self.default_seq_dim)

        # 断言是否使用了 xpos
        assert self.use_xpos
        # 获取设备、数据类型和序列长度
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        # 获取序列位置信息
        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        # 计算频率
        freqs = self.forward(seq, seq_len = seq_len)
        # 获取缩放比例
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        # 如果序列维度为 -3，则重新排列频率和缩放
        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        # 应用旋转嵌入到查询和键上
        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        # 转换旋转后的查询和键的数据类型
        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        # 返回旋转后的查询和键
        return rotated_q, rotated_k

    # 获取缩放比例
    @beartype
    def get_scale(
        self,
        t: Tensor,
        seq_len: Optional[int] = None,
        offset = 0
    ):
        # 断言是否使用了 xpos
        assert self.use_xpos

        # 判断是否应该缓存
        should_cache = (
            self.cache_if_possible and
            exists(seq_len)
        )

        # 如果应该缓存且缓存存在，则返回缓存的缩放比例
        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales.shape[0]
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        # 初始化缩放比例为 1
        scale = 1.
        # 如果使用了 xpos，则计算缩放比例
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim = -1)

        # 如果应该缓存，则缓存缩放比例
        if should_cache:
            self.tmp_store('cached_scales', scale)

        # 返回缩放比例
        return scale

    # 获取轴向频率
    def get_axial_freqs(self, *dims):
        # 定义切片
        Colon = slice(None)
        all_freqs = []

        # 遍历维度
        for ind, dim in enumerate(dims):
            # 根据频率类型生成位置信息
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            # 计算频率
            freqs = self.forward(pos, seq_len = dim)

            # 构建新的轴向切片
            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        # 广播所有频率并拼接
        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    # 前向传播函数
    @autocast(enabled = False)
    def forward(
        self,
        t: Tensor,
        seq_len = None,
        offset = 0
    ):
        # 判断是否应该缓存频率
        should_cache = (
            self.cache_if_possible and \
            not self.learned_freq and \
            exists(seq_len) and \
            self.freqs_for != 'pixel'
        )

        # 如果应该缓存且缓存存在，则返回缓存的频率
        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        # 获取频率
        freqs = self.freqs

        # 计算频率
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        # 如果应该缓存，则缓存频率
        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())

        # 返回频率
        return freqs
```