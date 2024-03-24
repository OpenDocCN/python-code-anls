# `.\lucidrains\block-recurrent-transformer-pytorch\block_recurrent_transformer_pytorch\block_recurrent_transformer_pytorch.py`

```
# 导入数学库
import math
# 从 random 模块中导入 random 函数
from random import random
# 从 functools 模块中导入 wraps 和 partial 函数
from functools import wraps, partial
# 从 itertools 模块中导入 zip_longest 函数
from itertools import zip_longest
# 从 collections 模块中导入 namedtuple 和 defaultdict 类
from collections import namedtuple, defaultdict
# 从 packaging 模块中导入 version 类
from packaging import version

# 导入 torch 库
import torch
# 从 torch.nn.functional 模块中导入 F 函数
import torch.nn.functional as F
# 从 torch 模块中导入 nn 和 einsum 函数
from torch import nn, einsum

# 从 einops 库中导入 rearrange、repeat、pack、unpack 函数和 Rearrange 类
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# 从 beartype 库中导入 beartype 函数
from beartype import beartype
# 从 beartype.door 模块中导入 is_bearable 函数
from beartype.door import is_bearable
# 从 beartype.typing 模块中导入 Optional、List、Tuple 类
from beartype.typing import Optional, List, Tuple

# helpers

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 判断张量是否为空
def is_empty(t: torch.Tensor):
    return t.numel() == 0

# 将输入转换为元组
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 判断列表中的元素是否唯一
def all_unique(arr):
    return len(arr) == len(set(arr))

# 评估装饰器
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# 仅执行一次的装饰器
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

# 仅打印一次的装饰器
print_once = once(print)

# 过滤掉空值
def compact(arr):
    return [*filter(exists, arr)]

# 对张量列表进行按位与操作
def and_reduce(arr: List[torch.Tensor]):
    if len(arr) == 0:
        return None
    head, *rest = arr
    for t in rest:
        head = head & t
    return head

# 安全拼接张量
def safe_cat(*args, dim = 1):
    args = compact(args)

    if len(args) == 0:
        return None

    return torch.cat(args, dim = dim)

# 判断是否可以整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 计算张量的 L2 范数
def l2norm(t):
    return F.normalize(t, dim = -1)

# 将张量打包成指定模式
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包的张量解包成指定模式
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 在指定维度上填充张量
def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# 无偏置的 Layernorm 类
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 采样辅助函数

# 计算张量的对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# Gumbel 采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# Top-k 采样
def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 旋转位置嵌入类
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        width,
        scale_base = 512,
        theta = 10000
    ):
        super().__init__()
        self.width = width

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

        self.register_buffer('cached_freqs', None, persistent = False)
        self.register_buffer('cached_scales', None, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device
    # 定义一个方法用于前向传播，self代表类的实例
    def forward(self):
        # 获取设备和序列长度
        device, seq_len = self.device, self.width

        # 如果已经存在缓存的频率信息
        if exists(self.cached_freqs):
            # 获取缓存的序列长度
            cached_seq_len = self.cached_freqs.shape[-2]
            # 如果缓存的序列长度大于等于当前序列长度，则直接返回缓存的频率和尺度
            if cached_seq_len >= seq_len:
                return self.cached_freqs[:seq_len], self.cached_scales[:seq_len]

        # 生成一个序列t，长度为seq_len，设备为device，数据类型与self.inv_freq相同
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        # 计算频率信息，使用torch.einsum进行张量乘法
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        # 将频率信息复制一份，拼接在一起，维度为-1
        freqs = torch.cat((freqs, freqs), dim=-1)

        # 计算尺度信息，根据公式计算得到
        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        # 将尺度信息复制一份，拼接在一起，维度为-1
        scale = torch.cat((scale, scale), dim=-1)

        # 将频率信息和尺度信息注册为缓存，persistent=False表示不持久化
        self.register_buffer('cached_freqs', freqs, persistent=False)
        self.register_buffer('cached_scales', scale, persistent=False)
        # 返回频率信息和尺度信息
        return freqs, scale
# 将输入张量 x 沿着最后一个维度分成两部分 x1 和 x2
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    # 返回将 x2 逆时针旋转 180 度后与 x1 拼接的结果
    return torch.cat((-x2, x1), dim=-1)


# 对输入张量 t 应用旋转位置编码 pos，并乘以缩放因子 scale
def apply_rotary_pos_emb(t, pos, scale = 1.):
    # 如果未提供缩放因子，则默认为 1
    scale = default(scale, 1.)

    # 获取序列长度
    seq_len = t.shape[-2]

    # 断言位置编码的长度大于等于序列长度
    assert pos.shape[-2] >= seq_len

    # 截取位置编码，保留与序列长度相同的部分
    pos = pos[-seq_len:]

    # 如果缩放因子是张量，则断言其长度大于等于序列长度，并截取与序列长度相同的部分
    if isinstance(scale, torch.Tensor):
        assert scale.shape[-2] >= seq_len
        scale = scale[-seq_len:]

    # 返回应用旋转位置编码后的结果
    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


# 内存管理类
class MemoryManager(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers = 1,
        mem_lengths = 512,
        compress_factors = 1
    ):
        super().__init__()
        # 将内存长度和压缩因子转换为元组形式
        mem_lengths = cast_tuple(mem_lengths)
        compress_factors = cast_tuple(compress_factors)

        # 断言所有内存长度大于 0
        assert all([mem_length > 0 for mem_length in mem_lengths])
        # 断言内存长度和压缩因子长度相同
        assert len(mem_lengths) == len(compress_factors)
        # 断言层数大于等于 1
        assert layers >= 1

        self.mem_lengths = mem_lengths
        self.compress_factors = compress_factors

        # 初始化层列表
        self.layers = nn.ModuleList([])

        # 遍历层数
        for _ in range(layers):
            compress_fns = nn.ModuleList([])

            # 遍历压缩因子
            for compress_factor in compress_factors:
                compress_fn = nn.Identity()
                # 如果压缩因子大于 1，则使用卷积进行压缩
                if compress_factor > 1:
                    compress_fn = nn.Sequential(
                        Rearrange('b n d -> b d n'),
                        nn.Conv1d(
                            dim * 2,
                            dim * 2,
                            compress_factor,
                            stride = compress_factor,
                            groups = 2
                        ),
                        Rearrange('b d n -> b n d'),
                    )

                compress_fns.append(compress_fn)

            self.layers.append(compress_fns)

    def forward(
        self,
        past_memories: List[torch.Tensor],
        new_memories: List[torch.Tensor]
        ):
        # 初始化一个空列表，用于存储下一个时间步的记忆
        next_memories = []

        # 遍历过去记忆、新记忆和压缩函数的组合
        for past_memory, new_memory, compress_fns in zip_longest(past_memories, new_memories, self.layers):

            # 处理当过去记忆和新记忆都不存在的情况
            if not (exists(past_memory) or exists(new_memory)):
                next_memories.append(None)
                continue

            next_memory = None

            # 遍历记忆长度、压缩因子和压缩函数的组合
            for mem_length, compress_factor, compress_fn in zip(self.mem_lengths, self.compress_factors, compress_fns):

                # 获取给定压缩因子下的记忆 "current_memory"
                current_memory = None
                if exists(past_memory):
                    past_memory, current_memory = past_memory[..., :-mem_length, :], past_memory[..., -mem_length:, :]

                # 基于初始化设置的压缩因子，压缩新进来的记忆
                if (not is_empty(new_memory)) and compress_factor > 1:
                    # 确保记忆长度可以被压缩因子整除
                    new_mem_length = new_memory.shape[-2]
                    curtailed_length = (new_mem_length // compress_factor) * compress_factor
                    curtailed_slice = slice(-curtailed_length, None) if curtailed_length > 0 else slice(0, 0)
                    new_memory = new_memory[..., curtailed_slice, :]

                    # 压缩推送到下一阶段的记忆
                    if new_memory.shape[-2] > 0:
                        new_memory = rearrange(new_memory, 'm b n d -> b n (m d)')
                        new_memory = compress_fn(new_memory)
                        new_memory = rearrange(new_memory, 'b n (m d) -> m b n d', m = 2)

                # FIFO 记忆队列
                # 将新记忆添加到右侧
                current_memory = safe_cat(current_memory, new_memory, dim = -2)
                # "new" 记忆是相对于下一个压缩段的新记忆

                new_memory, current_memory = current_memory[..., :-mem_length, :], current_memory[..., -mem_length:, :]
                # 将新记忆连接到过去记忆的左侧

                next_memory = safe_cat(current_memory, next_memory, dim = -2)

            next_memories.append(next_memory)

        return next_memories
# maybe flash attention, if using pytorch 2.0

# 定义一个命名元组 Config，包含三个布尔类型的配置参数
Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# state container

# 定义状态容器类 StateContainer，继承自 nn.Module
class StateContainer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_state_vectors,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        qk_rmsnorm_scale = 8,
        use_flash_attn = False
    ):
        super().__init__()
        assert num_state_vectors > 0
        self.heads = heads
        inner_dim = dim_head * heads

        # 对状态进行归一化
        self.state_norm = LayerNorm(dim)

        # 定义线性层，用于将输入转换为查询向量
        self.q_to_state = nn.Linear(dim, inner_dim, bias = False)
        self.q_from_state = nn.Linear(dim, inner_dim, bias = False)

        # 定义线性层，用于将状态转换为查询向量和键值对
        self.state_to_q = nn.Linear(dim, inner_dim, bias = False)
        self.state_to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        # 初始化状态和位置编码
        self.init_state = nn.Parameter(torch.randn(num_state_vectors, dim))
        self.state_pos_ids = nn.Parameter(torch.randn(num_state_vectors, dim))

        # 定义线性层，用于将输出转换为状态
        self.to_state_out = nn.Linear(inner_dim * 2, dim, bias = False)

        # 定义注意力机制，用于状态之间的交叉注意力和自注意力
        self.to_state_cross_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)
        self.state_self_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)
        self.from_state_cross_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

        # gating related parameters - using the fixed simple config

        # 定义线性层和参数，用于门控机制
        self.state_out_to_gate = nn.Linear(dim, dim)
        self.learned_ema_beta = nn.Parameter(torch.randn(dim))

        # since each read should be followed by a write, just store cache in the container

        # 初始化缓存和下一个读取状态
        self.cache = None
        self.next_read_state = None

    # 设置下一个读取状态
    def set_next_read_state(
        self,
        states
    ):
        if not exists(states):
            states = self.init_state

        self.next_read_state = (states,)

    # 读取状态
    def read(self, x):
        assert exists(self.next_read_state), 'states to be read must be set with .set_next_read_state'

        states, = self.next_read_state
        self.next_read_state = None

        # 对状态进行注意力前的归一化
        normed_states = self.state_norm(states)

        # 添加位置编码
        normed_states = normed_states + self.state_pos_ids

        # 获取查询向量用于交叉注意力
        q_to_state = self.q_to_state(x)
        q_to_state = rearrange(q_to_state, '... n (h d) -> ... h n d', h = self.heads)

        # 状态的自注意力机制
        state_k, state_v = self.state_to_kv(normed_states).chunk(2, dim = -1)

        # 交叉注意力
        to_state_out = self.to_state_cross_attn(q_to_state, state_k, state_v)

        to_state_out = rearrange(to_state_out, 'b h n d -> b n (h d)')

        # 缓存下一个写入状态
        self.cache = (states, normed_states, state_k, state_v)

        return to_state_out

    # 写入状态
    def write(
        self,
        *,
        memories
    ):
        # 断言缓存存在
        assert exists(self.cache)

        # 解包记忆
        k, v = memories
        batch = k.shape[0]

        # 从先前读取的缓存中获取缓存的值

        states, normed_states, state_k, state_v = self.cache

        self.cache = None

        # 推导查询

        q_from_state = self.q_from_state(normed_states)
        q_from_state = rearrange(q_from_state, '... n (h d) -> ... h n d', h = self.heads)

        state_q = self.state_to_q(normed_states)
        state_q_einsum = 'n (h d)' if state_q.ndim == 2 else 'b n (h d)'
        state_q = repeat(state_q, f'{state_q_einsum} -> b h n d', h = self.heads, b = batch)

        # 状态也必须经过自注意力

        if q_from_state.ndim == 3:
            q_from_state = repeat(q_from_state, '... -> b ...', b = batch)

        state_out = self.state_self_attn(state_q, state_k, state_v)

        from_state_out = self.from_state_cross_attn(q_from_state, k, v)

        state_out = torch.cat((state_out, from_state_out), dim = -1)
        state_out = rearrange(state_out, 'b h n d -> b n (h d)')

        state_out = self.to_state_out(state_out)

        # 使用表现最佳的配置
        # 固定简单门 - 仅仅是一个学习的EMA，与高速公路网络有些相似

        z = self.state_out_to_gate(state_out)
        learned_ema_decay = self.learned_ema_beta.sigmoid()

        # 使用学习的EMA门设置新状态

        return learned_ema_decay * z + (1 - learned_ema_decay) * states

    def forward(self, x):
        raise NotImplementedError
# 主类
class Attend(nn.Module):
    # 初始化函数
    def __init__(
        self,
        causal = False,
        use_flash_attn = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化 causal 和 use_flash_attn 属性
        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash_attn = use_flash_attn
        # 检查是否满足使用 flash attention 的条件
        assert not (use_flash_attn and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 确定 CUDA 和 CPU 的高效注意力配置

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash_attn:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    # 获取 mask
    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    # flash attention 函数
    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # 推荐使用 Tri Dao 的多查询单键值注意力
        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h = q.shape[1])

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h = q.shape[1])

        # 检查 mask 是否存在并扩展到兼容的形状
        masks = []

        if self.causal:
            i, j = q_len, k_len
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = q.device).triu(j - i + 1)
            masks.append(~causal_mask)

        if exists(mask):
            if mask.ndim != 2:
                mask = repeat(mask, 'w ... -> (b w) ...', b = q.shape[0] // mask.shape[0])

            masks.append(mask)

        attn_mask = and_reduce(masks)

        # 检查是否有兼容的设备用于 flash attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # 使用 torch.backends.cuda.sdp_kernel 函数进行 flash attention
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = attn_mask
            )

        return out
    # 实现 Transformer 模型的前向传播函数，接受查询(q)、键(k)、值(v)和掩码(mask)，以及是否使用 Flash Attention(use_flash_attn)的参数
    def forward(self, q, k, v, mask = None, use_flash_attn = None):
        # 如果未提供 use_flash_attn 参数，则使用默认值 self.use_flash_attn
        use_flash_attn = default(use_flash_attn, self.use_flash_attn)

        # 获取查询张量的形状信息
        b, n, device = q.shape[0], q.shape[-2], q.device

        # 将查询(q)、键(k)、值(v)打包成特定形状
        q, ps = pack_one(q, '* h n d')
        k, _ = pack_one(k, '* n d')
        v, _ = pack_one(v, '* n d')

        # 如果使用 Flash Attention，则调用 flash_attn 函数进行注意力计算
        if use_flash_attn:
            out = self.flash_attn(q, k, v, mask = mask)
            return unpack_one(out, ps, '* h n d')

        # 计算缩放因子
        scale = q.shape[-1] ** -0.5

        # 根据键(k)的维度确定 einsum 中的字符串
        k_einsum = 'b j d' if k.ndim == 3 else 'b h j d'
        v_einsum = 'b j d' if v.ndim == 3 else 'b h j d'

        # 计算相似度矩阵
        sim = einsum(f"b h i d, {k_einsum} -> b h i j", q, k) * scale

        # 处理键的填充掩码
        if exists(mask):
            if mask.ndim != 2:
                mask = repeat(mask, 'w ... -> (b w) ...', b = b)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 处理因果掩码
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = q.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 计算注意力权重
        attn = sim.softmax(dim=-1)

        # 聚合值
        out = einsum(f"b h i j, {v_einsum} -> b h i d", attn, v)

        return unpack_one(out, ps, '* h n d')
# 定义 GEGLU 类，用于实现 GEGLU 激活函数
class GEGLU(nn.Module):
    # GEGLU 类的前向传播函数
    def forward(self, x):
        # 将输入张量 x 按照最后一个维度分成两部分，分别赋值给 x 和 gate
        x, gate = x.chunk(2, dim=-1)
        # 返回 GEGLU 激活函数的计算结果
        return F.gelu(gate) * x

# 定义 FeedForward 函数，用于创建前馈神经网络
def FeedForward(dim, mult=4):
    # 计算内部维度
    inner_dim = int(dim * mult * 2 / 3)
    # 返回一个包含多个层的神经网络模型
    return nn.Sequential(
        LayerNorm(dim),  # 对输入进行层归一化
        nn.Linear(dim, inner_dim * 2, bias=False),  # 线性变换层
        GEGLU(),  # GEGLU 激活函数
        nn.Linear(inner_dim, dim, bias=False)  # 线性变换层
    )

# 定义 Attention 类，用于实现注意力机制
class Attention(nn.Module):
    # Attention 类的初始化函数
    def __init__(
        self,
        dim_head,
        causal=False,
        qk_rmsnorm=False,
        qk_rmsnorm_scale=8,
        use_flash_attn=False
    ):
        super().__init__()
        self.causal = causal  # 是否使用因果注意力机制

        self.qk_rmsnorm = qk_rmsnorm  # 是否进行 RMS 归一化
        self.qk_rmsnorm_scale = qk_rmsnorm_scale  # RMS 归一化的缩放因子

        self.attend = Attend(causal=causal, use_flash_attn=use_flash_attn)  # 创建 Attend 对象

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))  # 创建可学习参数 q_scale
            self.k_scale = nn.Parameter(torch.ones(dim_head))  # 创建可学习参数 k_scale

    # Attention 类的前向传播函数
    def forward(
        self,
        q, k, v,
        mask=None,
        rotary_pos_emb=None,
        xpos_scale=None
    ):

        scale = q.shape[-1] ** -0.5  # 缩放因子

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))  # 对 q 和 k 进行 L2 归一化
            scale = self.qk_rmsnorm_scale  # 更新缩放因子

        if self.qk_rmsnorm:
            q = q * self.q_scale  # 对 q 进行缩放
            k = k * self.k_scale  # 对 k 进行缩放

        # 使用旋转位置嵌入进行位置编码
        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb, xpos_scale)
            k = apply_rotary_pos_emb(k, rotary_pos_emb, xpos_scale ** -1)

        # 注意力计算
        out = self.attend(q, k, v, mask=mask)

        return out

# 定义 AttentionBlock 类，用于实现注意力块
class AttentionBlock(nn.Module):
    # AttentionBlock 类的初始化函数
    def __init__(
        self,
        dim,
        block_width,
        dim_head=64,
        heads=8,
        qk_rmsnorm=False,
        qk_rmsnorm_scale=8,
        use_flash_attn=False,
        num_state_vectors=0,
        num_external_state_reads=0,
        state_read_before_write=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.norm = LayerNorm(dim)  # 对输入进行层归一化

        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # 线性变换层
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)  # 线性变换层

        self.attn = Attention(dim_head, qk_rmsnorm=qk_rmsnorm, qk_rmsnorm_scale=qk_rmsnorm_scale, use_flash_attn=use_flash_attn)  # 创建 Attention 对象

        self.block_width = block_width
        self.is_recurrent_layer = num_state_vectors > 0  # 是否为循环层

        num_state_reads = int(self.is_recurrent_layer and state_read_before_write) + num_external_state_reads  # 确定从状态容器中读取的状态数量

        self.to_out = nn.Linear(inner_dim * (1 + num_state_reads), dim, bias=False)  # 线性变换层

        if not self.is_recurrent_layer:
            return

        self.state_read_before_write = state_read_before_write

        self.state_container = StateContainer(
            dim,
            dim_head=dim_head,
            heads=heads,
            num_state_vectors=num_state_vectors,
            qk_rmsnorm=qk_rmsnorm,
            qk_rmsnorm_scale=qk_rmsnorm_scale,
            use_flash_attn=use_flash_attn
        )

    @property
    def device(self):
        return next(self.parameters()).device

    # AttentionBlock 类的前向传播函数
    def forward(
        self,
        x,
        rotary_pos_emb=None,
        xpos_scale=None,
        attn_mask=None,
        xl_memories: Optional[torch.Tensor] = None,
        read_from_state_containers: List[StateContainer] = []
        ):
            # 解构输入张量 x 的形状，获取 batch, seq_len, _, width, device
            batch, seq_len, _, width, device = *x.shape, self.block_width, self.device

            # 预归一化处理
            x = self.norm(x)

            # 分别提取 queries, keys, values，并拆分出多头
            q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

            split_head = partial(rearrange, pattern='b n (h d) -> b h n d', h=self.heads)
            q = split_head(q)

            # 将最后的 key / values 作为记忆以供后续使用
            memories = torch.stack((k, v))
            mem_len = 0

            if exists(xl_memories):
                # 如果传入了过去的记忆，将其连接为第一个 bucket
                mem_len = xl_memories.shape[-2]
                past_k, past_v = xl_memories
                k = torch.cat((past_k, k), dim=1)
                v = torch.cat((past_v, v), dim=1)

            # 处理注意力掩码和位置嵌入的裁剪
            if exists(attn_mask):
                attn_mask = attn_mask[:seq_len, :seq_len]
                attn_mask = F.pad(attn_mask, (mem_len, 0), value=True)

            # 进行注意力计算
            out = self.attn(
                q, k, v,
                rotary_pos_emb=rotary_pos_emb,
                xpos_scale=xpos_scale,
                mask=attn_mask
            )

            # 合并多头
            out = rearrange(out, 'b h n d -> b n (h d)')

            # 如果不是循环层且没有从状态容器中读取数据，则直接返回结果
            if not self.is_recurrent_layer and len(read_from_state_containers) == 0:
                return self.to_out(out), memories, None

            # 是否从自身状态容器中读取数据，默认为是，但也可以传入更多
            if self.is_recurrent_layer and self.state_read_before_write:
                read_from_state_containers = [self.state_container, *read_from_state_containers]

            for read_state_container in read_from_state_containers:
                # 从状态容器中读取数据
                to_state_out = read_state_container.read(x)

                # 将读取的数据连接到自注意力输出中
                out = torch.cat((out, to_state_out), dim=-1)

            new_states = None

            if self.is_recurrent_layer:
                # 如果是循环层，则将记忆写入状态容器
                new_states = self.state_container.write(memories=memories)

            return self.to_out(out), memories, new_states
# 定义一个装饰器函数 @beartype，用于类型检查
# 定义一个类 BlockRecurrentTransformer，继承自 nn.Module
class BlockRecurrentTransformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        num_tokens,  # 输入参数：标记的数量
        dim,  # 输入参数：维度
        depth,  # 输入参数：深度
        dim_head = 64,  # 输入参数：头的维度，默认为64
        heads = 8,  # 输入参数：头的数量，默认为8
        all_layers_qk_rmsnorm = False,  # 输入参数：是否对所有层的查询和键进行均方根归一化，默认为False
        ff_mult = 4,  # 输入参数：前馈网络的倍数，默认为4
        max_seq_len = 1024,  # 输入参数：最大序列长度，默认为1024
        block_width = 512,  # 输入参数：块的宽度，默认为512
        recurrent_layers: Optional[Tuple[int, ...]] = None,  # 输入参数：循环层的索引元组，默认为None
        read_recurrent_layers: Optional[Tuple[int, ...]] = None,  # 输入参数：读取循环层的索引元组，默认为None
        num_state_vectors = None,  # 输入参数：状态向量的数量，默认为None
        ignore_index = -100,  # 输入参数：忽略的索引，默认为-100
        use_flash_attn = False,  # 输入参数：是否使用快闪注意力，默认为False
        use_compressed_mem = False,  # 输入参数：是否使用压缩内存，默认为False
        compressed_mem_factor = 4  # 输入参数：压缩内存因子，默认为4
        ):
        # 调用父类的构造函数
        super().__init__()
        # 设置状态向量的数量，默认为块宽度
        num_state_vectors = default(num_state_vectors, block_width)

        # 设置循环层

        # 默认为网络中间的一个循环层
        recurrent_layers = default(recurrent_layers, (depth // 2,))

        # 断言循环层的范围在1到深度之间
        assert all([0 < layer <= depth for layer in recurrent_layers]), f'recurrent layers must range from 1 to the depth {depth}'
        # 断言循环层是唯一的，没有重复的层
        assert all_unique(recurrent_layers), 'recurrent layers must be all unique. no duplicate layers'

        self.recurrent_layers = recurrent_layers

        # 设置读取循环层

        read_recurrent_layers = default(read_recurrent_layers, recurrent_layers)

        # 断言读取循环层小于等于写入循环层
        assert all([read_layer <= write_layer for read_layer, write_layer in zip(read_recurrent_layers, recurrent_layers)]), 'the recurrent read layer must be always less than or equal to the write layer'
        assert all([0 < layer <= depth for layer in read_recurrent_layers])
        assert len(read_recurrent_layers) == len(recurrent_layers)

        self.read_recurrent_layers = read_recurrent_layers

        # 令牌嵌入

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.rotary_pos_emb = RotaryEmbedding(dim = dim_head, width = (2 if not use_compressed_mem else 3) * block_width)

        self.layers = nn.ModuleList([])

        self.write_to_read_map = {write_layer: read_layer for write_layer, read_layer in zip(recurrent_layers, read_recurrent_layers)}

        self.read_state_router = defaultdict(list)

        for layer in range(1, depth + 1):
            is_recurrent_layer = layer in self.recurrent_layers

            layer_num_state_vectors = num_state_vectors if is_recurrent_layer else 0

            num_external_state_reads = sum([int(layer == read_layer) for read_layer in read_recurrent_layers])

            # 只有具有xl记忆的层或在水平方向上具有循环的层使用qk rmsnorm
            qk_rmsnorm = all_layers_qk_rmsnorm or is_recurrent_layer

            attn_block = AttentionBlock(
                dim,
                block_width = block_width,
                dim_head = dim_head,
                heads = heads,
                qk_rmsnorm = qk_rmsnorm,
                num_state_vectors = layer_num_state_vectors,
                use_flash_attn = use_flash_attn,
                num_external_state_reads = num_external_state_reads,
                state_read_before_write = False,
            )

            ff_block = FeedForward(dim, mult = ff_mult)

            if is_recurrent_layer:
                read_layer = self.write_to_read_map[layer]
                self.read_state_router[read_layer].append(attn_block.state_container)

            self.layers.append(nn.ModuleList([
                attn_block,
                ff_block
            ]))

        # (compressed) memory management

        self.mem_manager = MemoryManager(
            dim = dim_head,
            layers = depth,
            mem_lengths = block_width if not use_compressed_mem else (block_width, block_width // 2),
            compress_factors = 1 if not use_compressed_mem else (1, compressed_mem_factor)
        )

        # 转换为logits

        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.max_seq_len = max_seq_len
        self.block_width = block_width

        # 断言最大序列长度能被块宽度整除
        assert divisible_by(max_seq_len, block_width)

        self.ignore_index = ignore_index

        self.register_buffer('cached_causal_attn_mask', None, persistent = False)

    @property
    def device(self):
        # 返回参数的设备
        return next(self.parameters()).device
    # 获取因果注意力掩码
    def get_causal_attn_mask(self, width):
        # 如果缓存中存在因果注意力掩码
        if exists(self.cached_causal_attn_mask):
            # 获取缓存中的掩码
            cached_mask = self.cached_causal_attn_mask
            # 获取缓存掩码的宽度
            cached_width = cached_mask.shape[-2]
            # 计算填充量
            padding = (width - cached_width) // 2
            # 创建切片对象
            j_slice = Ellipsis if padding == 0 else slice(padding, -padding)
            # 返回缓存中的掩码
            return cached_mask[:cached_width, j_slice]

        # 获取设备信息
        device = self.device
        # 创建全为1的因果掩码
        causal_mask = torch.ones((width, width), device=device, dtype=torch.bool).triu(1)
        # 返回取反的因果掩码
        return ~causal_mask

    # 生成序列
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prime,
        length=None,
        xl_memories: List[torch.Tensor] = [],
        states: List[torch.Tensor] = [],
        temperature=1.,
        filter_thres=0.9,
        return_memories_and_states=False
    ):
        # 设置生成序列的长度
        length = default(length, self.max_seq_len + 1)
        # 获取起始序列的长度
        start_len = prime.shape[-1]

        # 断言起始序列长度小于最大序列长度
        assert start_len < self.max_seq_len
        # 断言生成序列长度不超过最大序列长度
        assert length <= (self.max_seq_len + 1)
        # 断言起始序列长度小于生成序列长度
        assert start_len < length

        # 初始化输出为起始序列
        output = prime

        # 初始化记忆
        memories = []

        # 循环生成序列
        for ind in range(length - start_len):

            # 前向传播
            logits, next_memories, next_states = self.forward(
                output,
                xl_memories=xl_memories,
                states=states
            )

            # 获取最后一个位置的logits
            logits = logits[:, -1]

            # 过滤logits
            filtered_logits = top_k(logits, thres=filter_thres)
            # 采样
            sampled = gumbel_sample(filtered_logits, temperature=temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            # 拼接采样结果到输出序列
            output = torch.cat((output, sampled), dim=-1)

            # 如果当前窗口的最后一个token采样完成，更新记忆和状态
            if divisible_by(output.shape[-1] - 1, self.max_seq_len):
                memories = next_memories
                states = next_states

        # 去除起始序列部分，得到最终生成序列
        output = output[:, start_len:]

        # 如果需要返回记忆和状态信息
        if return_memories_and_states:
            return output, memories, states

        return output

    # 前向传播
    def forward(
        self,
        x,
        return_loss=False,
        xl_memories: List[torch.Tensor] = [],
        states: List[torch.Tensor] = [],
        return_memories_and_states=None  # 可以强制返回记忆和状态，或者不返回。默认只有在token数量等于最大序列长度时才返回
        ):
            # 获取输入张量的设备信息
            device = x.device

            if return_loss:
                # 如果需要返回损失，则将输入张量切片，去掉最后一个元素作为标签
                x, labels = x[:, :-1], x[:, 1:]

            # 获取动态位置偏置的序列长度 i 和 j

            assert x.shape[-1] <= self.max_seq_len

            w = self.block_width

            # 令牌嵌入

            x = self.token_emb(x)

            # 动态位置偏置

            attn_mask = self.get_causal_attn_mask(w)
            rotary_pos_emb, xpos_scale = self.rotary_pos_emb()

            # 只有在完整的块宽度时才返回记忆和状态，但可以被覆盖

            return_memories_and_states = default(return_memories_and_states, self.max_seq_len == x.shape[-2])

            # 准备输出张量，以便按块连接

            batch, _, dim = x.shape

            out = torch.empty(batch, 0, dim, dtype = x.dtype, device = self.device)

            # 将输入分割成宽度为 w 的块

            input_blocks = x.split(w, dim = -2)

            # 逐个处理每个块

            for input_block in input_blocks:
                input_block_length = input_block.shape[-2]

                # 准备 xl 记忆和状态

                iter_xl_memories = iter(xl_memories)
                iter_states = iter(states)

                next_xl_memories = []
                next_states = []

                # 在适当的状态容器上设置状态

                for attn, _ in self.layers:
                    if not attn.is_recurrent_layer:
                        continue

                    attn.state_container.set_next_read_state(next(iter_states, None))

                # 遍历层

                for ind, (attn, ff) in enumerate(self.layers):

                    # 确定层是否需要 transformer xl 记忆

                    layer = ind + 1

                    # 是否传入 xl 记忆

                    attn_kwargs = dict(
                        rotary_pos_emb = rotary_pos_emb,
                        xpos_scale = xpos_scale,
                        attn_mask = attn_mask,
                        xl_memories = next(iter_xl_memories, None),
                        read_from_state_containers = self.read_state_router[layer]
                    )

                    # 注意力层

                    residual = input_block
                    attn_branch_out, layer_xl_memories, layer_next_states = attn(input_block, **attn_kwargs)

                    if exists(layer_xl_memories):
                        next_xl_memories.append(layer_xl_memories)

                    if exists(layer_next_states):
                        next_states.append(layer_next_states)

                    input_block = attn_branch_out + residual

                    # 前馈层

                    input_block = ff(input_block) + input_block

                # 连接到输出

                out = torch.cat((out, input_block), dim = -2)

                # 设置新的 xl 记忆和状态

                states = next_states

                if input_block_length == w:
                    xl_memories = self.mem_manager(xl_memories, next_xl_memories)


            # 投影到对数

            logits = self.to_logits(out)

            # 分离状态和记忆

            returned_next_states = list(map(torch.detach, states)) if return_memories_and_states else None
            returned_next_xl_memories = list(map(torch.detach, xl_memories)) if return_memories_and_states else None

            # 是否返回对数

            if not return_loss:
                return logits, returned_next_xl_memories, returned_next_states

            # 交叉熵损失

            logits = rearrange(logits, 'b n c -> b c n')
            loss = F.cross_entropy(logits, labels, ignore_index = self.ignore_index)

            return loss, returned_next_xl_memories, returned_next_states
# recurrent trainer wrapper

# 定义一个装饰器，用于验证输入参数类型
@beartype
# 定义一个类，继承自 nn.Module
class RecurrentTrainerWrapper(nn.Module):
    # 初始化方法
    def __init__(
        self,
        transformer: BlockRecurrentTransformer,
        xl_memories_dropout = 0.,
        state_dropout = 0.
    ):
        super().__init__()
        self.transformer = transformer
        self.seq_len = transformer.max_seq_len

        self.xl_memories_dropout = xl_memories_dropout
        self.state_dropout = state_dropout

    # 生成方法，用于生成序列
    @eval_decorator
    @torch.no_grad()
    def generate(
        self,
        prime,
        length,
        **kwargs
    ):
        seq_len = self.seq_len
        start_len = prime.shape[-1]
        assert start_len < length

        output = prime
        current_len = start_len

        memories = []
        states = []

        # 确定长度

        has_remainder = not divisible_by(length, seq_len)
        remainder_amount = length % seq_len
        total_segments = math.ceil(length / seq_len)

        if not has_remainder:
            lengths = (*((seq_len + 1,) * (total_segments - 1)), seq_len)
        elif remainder_amount == 1:
            lengths = (seq_len + 1,) * (total_segments - 1)
        else:
            lengths = (*((seq_len + 1,) * (total_segments - 1)), remainder_amount)

        # 循环遍历长度

        for next_length in lengths:

            segment_output, memories, states = self.transformer.generate(
                output[:, -current_len:],
                length = next_length,
                xl_memories = memories,
                states = states,
                return_memories_and_states = True,
                **kwargs
            )

            output = torch.cat((output, segment_output), dim = -1)
            current_len = 1

        return output[:, start_len:]

    # 前向传播方法
    def forward(
        self,
        x,
        return_memories_and_states = False
    ):
        total_seq_len, seq_len = x.shape[1], self.seq_len

        assert divisible_by(total_seq_len - 1, seq_len), f'length of sequence ({total_seq_len}) must be equal to a multiple of {seq_len} + 1 (one extra token) during training'
        segments = total_seq_len // seq_len

        total_loss = 0.

        memories = []
        states = []

        for ind in range(segments):
            start = ind * seq_len
            end = start + seq_len + 1

            if self.training and random() < self.xl_memories_dropout:
                memories.clear()

            if self.training and random() < self.state_dropout:
                states.clear()

            loss, memories, states = self.transformer(
                x[:, start:end],
                xl_memories = memories,
                states = states,
                return_loss = True
            )

            total_loss = total_loss + (loss / segments)

        if return_memories_and_states:
            return total_loss, memories, states

        return total_loss
```