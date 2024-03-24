# `.\lucidrains\ring-attention-pytorch\ring_attention_pytorch\ring_attention.py`

```py
# 导入必要的库
from typing import Optional, Tuple, Union

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module, ModuleList

import einx
from einx import rearrange

from beartype import beartype

# 导入自定义模块和函数
from ring_attention_pytorch.ring import (
    all_ring_pass,
    is_distributed,
    get_rank,
    get_world_size
)

from ring_attention_pytorch.ring_flash_attention import (
    ring_flash_attn
)

from ring_attention_pytorch.distributed import (
    split_by_rank,
    AllGather
)

# 辅助函数

# 检查变量是否存在
def exists(v):
    return v is not None

# 如果变量存在则返回变量，否则返回默认值
def default(v, d):
    return v if exists(v) else d

# 将输入转换为元组，如果输入已经是元组则返回，否则返回包含输入的元组
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 检查一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# 默认的注意力函数
def default_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    causal: bool = False
):
    q = q * (q.shape[-1] ** -0.5)

    mask_value = -torch.finfo(q.dtype).max

    # 相似度计算

    sim = einsum('b i h d, b j h d -> b h i j', q, k)

    # 掩码处理

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
        sim = torch.where(causal_mask, mask_value, sim)

    elif exists(mask):
        sim = einx.where('b j, b h i j, -> b h i j', mask, sim, mask_value)

    # 注意力计算

    attn = einx.softmax('b h i [j]', sim)

    # 聚合

    out = einsum('b h i j, b j h d -> b i h d', attn, v)

    return out

# 旋转嵌入，支持条纹注意力的修改
class RingRotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        ring: bool = False,
        striped: bool = False,
        buckets: int = 1,        # 在带有 flash buckets > 1 的条纹注意力中，需要指定每台机器的桶数
        theta = 10000
    ):
        super().__init__()
        self.ring = ring
        self.striped = striped
        self.buckets = buckets

        inv_freq = theta ** -(torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled = False)
    def forward(
        self,
        seq_len: int,
        offset = 0
    ):
        device = self.device
        pos = None

        if self.ring:
            if self.striped:
                buckets = self.buckets
                ring_stride = get_world_size() * buckets
                ring_offset = buckets

                pos = torch.arange(seq_len // buckets, device = device)
                pos = rearrange('n -> n b', pos, b = buckets)

                pos = pos * ring_stride
                pos += torch.arange(buckets, device = device) + (get_rank() * buckets)
                pos = rearrange('n b -> (b n)', pos)

            else:
                pos = torch.arange(seq_len, device = device)
                pos += seq_len * get_rank()
        else:
            pos = torch.arange(seq_len, device = device)

        pos = pos.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', pos, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

# 旋转半部分
def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim=-1)

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    pos = rearrange('n d -> n 1 d', pos)
    return t * pos.cos() + rotate_half(t) * pos.sin()

# 批量到序列分片和反向操作

# 将张量填充到指定长度的倍数
def pad_to_multiple(
    x: Tensor,
    length: int,
    pad_value = 0
):
    seq_len = x.shape[-1]
    remainder = seq_len % length

    if remainder == 0:
        return x, 0

    pad_length = length - remainder
    return F.pad(x, (0, pad_length), value = pad_value), pad_length

# 可能填充序列和掩码
def maybe_pad_seq_and_mask(
    x: Tensor,
    mask: Optional[Tensor],
    seq_size: int
):
    orig_x, seq_len = x, x.shape[-1]
    # 自动填充序列和掩码，因为环传递假设张量的形状都相同

    # 调用函数将输入张量 x 填充到 seq_size 的倍数，并返回填充后的张量和填充长度
    x, pad_length = pad_to_multiple(x, seq_size)

    # 如果填充长度为 0，则直接返回填充后的张量 x 和掩码 mask
    if pad_length == 0:
        return x, mask

    # 如果掩码 mask 不存在，则创建一个与原始输入 orig_x 相同形状的全为 True 的掩码
    if not exists(mask):
        mask = torch.ones_like(orig_x).bool()

    # 调用函数将掩码 mask 填充到 seq_size 的倍数，并使用 False 值进行填充
    mask, _ = pad_to_multiple(mask, seq_size, pad_value = False)

    # 返回填充后的张量 x 和掩码 mask
    return x, mask
def sharded_batch_to_sharded_seq(
    x: Tensor,
    mask: Optional[Tensor],
    seq_size: int
):
    assert is_distributed()

    # 创建 AllGather 对象，用于在批次维度上进行全局收集
    all_gather = AllGather(dim = 0)

    # 在批次维度上对输入张量 x 进行全局收集
    x, sizes = all_gather(x)

    if exists(mask):
        # 如果存在 mask，则在批次维度上对 mask 进行全局收集
        mask, _ = all_gather(mask)

    # 确保世界大小可以被序列大小整除
    world_size = get_world_size()
    total_split_seq = x.shape[-1] // seq_size
    assert divisible_by(world_size, total_split_seq)

    num_sharded_batches = world_size // total_split_seq

    # 重新排列输入张量 x，以便在序列维度上进行分片
    x = rearrange('(b s) n -> b (s n)', x, s = num_sharded_batches)

    # 在序列维度上对 x 进行分片
    x = x.split(seq_size, dim = -1)

    # 根据排名对 x 进行分割
    x, _ = split_by_rank(x)

    if exists(mask):
        # 如果存在 mask，则重新排列 mask，并在序列维度上对其进行分片
        mask = rearrange('(b s) n -> b (s n)', mask, s = num_sharded_batches)
        mask = mask.split(seq_size, dim = -1)
        mask, _ = split_by_rank(mask)

    return (x, mask), sizes, num_sharded_batches

def sharded_seq_to_sharded_batch(
    logits: Tensor,
    sizes,
    num_sharded_batches = 1
):
    all_gather = AllGather(dim = -2) # 在序列维度上进行全局收集

    # 在序列维度上对 logits 进行全局收集
    logits, _ = all_gather(logits)

    # 重新排列 logits，以便在批次维度上进行分片
    logits = rearrange('b (s n) c -> (b s) n c', logits, s = num_sharded_batches)

    # 在批次维度上对 logits 进行分片
    logits = logits.split(sizes.tolist(), dim = 0)

    # 根据排名对 logits 进行分割
    logits, _ = split_by_rank(logits)

    return logits

# 主类 RingAttention
class RingAttention(Module):
    @beartype
    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 64,
        heads: int = 8,
        causal: bool = False,
        eps: float = 1e-10,
        bucket_size: int = 512,
        ring_attn: bool = False,
        ring_seq_size: int = 512,
        max_lookback_seq_len: Optional[int] = None,
        striped_ring_attn: bool = False,
        auto_shard_seq: Optional[bool] = None,
        prenorm: bool = True,
        force_regular_attn: bool = False,
        rotary_embed: bool = False,
        rotary_embed_theta: int = 10000,
        use_cuda_kernel: bool = None
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        assert divisible_by(ring_seq_size, bucket_size)

        self.ring_attn = ring_attn
        self.max_lookback_seq_len = max_lookback_seq_len
        self.striped_ring_attn = striped_ring_attn

        self.force_regular_attn = force_regular_attn
        self.auto_shard_seq = default(auto_shard_seq, ring_attn) # 这应该在 token ids 的转换器级别上完成，但出于测试目的

        assert not (not self.ring_attn and self.auto_shard_seq)

        self.ring_seq_size = ring_seq_size
        self.bucket_size = bucket_size

        # 初始化旋转嵌入
        self.rotary_embed = None
        if rotary_embed:
            self.rotary_embed = RingRotaryEmbedding(
                dim = dim_head,
                ring = ring_attn,
                striped = striped_ring_attn,
                theta = rotary_embed_theta,
                buckets = ring_seq_size // bucket_size
            )

        # 投影层
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            RMSNorm(dim) if prenorm else nn.Identity(),
            nn.Linear(dim, dim_inner * 3, bias = False)
        )

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

        # 是否使用 flash attention cuda kernel
        self.use_cuda_kernel = default(use_cuda_kernel, torch.cuda.is_available())
        assert not (use_cuda_kernel and not torch.cuda.is_available())

    def forward(
        self,
        x,
        mask = None,
        rotary_emb = None,
        force_ring_reduce_off = False,
        ring_size = None,
        ):
        """
        einstein notation

        b - batch
        h - heads
        d - feature dimension
        n, i, j - sequence
        """

        # 设置环的大小为默认值或者获取当前环的大小
        ring_size = default(ring_size, get_world_size())
        # 判断是否使用环形注意力，并且当前环是否分布式
        ring_attn = self.ring_attn & is_distributed()
        # 判断是否自动分片序列，并且当前环是否分布式
        auto_shard_seq = self.auto_shard_seq & is_distributed()

        # 获取序列的长度
        seq_len = x.shape[-1]

        # 如果自动分片序列为真
        if auto_shard_seq:
            # 可能填充序列和掩码，使其长度符合环形序列的大小
            x, mask = maybe_pad_seq_and_mask(x, mask, self.ring_seq_size)

            # 如果使用条纹环形注意力
            if self.striped_ring_attn:
                # 重新排列张量维度，以适应条纹环形注意力
                x = rearrange('b (i j) d -> b (j i) d', x, i = self.bucket_size)

                # 如果存在掩码
                if exists(mask):
                    # 重新排列掩码张量维度，以适应条纹环形注意力
                    mask = rearrange('b (i j) -> b (j i)', mask, i = self.bucket_size)

            # 将批次转换为序列，并返回批次大小
            (x, mask), batch_sizes = sharded_batch_to_sharded_seq(x, mask, self.ring_seq_size)

        # 获取设备信息
        device = x.device

        # 将输入张量转换为查询、键、值
        qkv = self.to_qkv(x)
        q, k, v = rearrange('b n (qkv h d) -> qkv b n h d', qkv, qkv = 3, h = self.heads)

        # 旋转相对位置

        # 如果旋转嵌入不存在且存在旋转嵌入
        if not exists(rotary_emb) and exists(self.rotary_embed):
            # 生成旋转嵌入
            rotary_emb = self.rotary_embed(q.shape[-2])

        # 如果存在旋转嵌入
        if exists(rotary_emb):
            # 应用旋转位置嵌入到查询和键
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        # 常规注意力 vs 闪存注意力（带或不带 kv 环减少）

        # 判断是否有任何 CUDA 输入
        any_cuda_inputs = any([t.is_cuda for t in (q, k, v)])

        # 如果强制使用常规注意力
        if self.force_regular_attn:
            # 使用默认的注意力机制
            out = default_attention(q, k, v, mask = mask, causal = self.causal)

        # 如果有任何 CUDA 输入并且使用 CUDA 内核
        elif any_cuda_inputs and self.use_cuda_kernel:
            # 导入 CUDA 实现的闪存注意力
            from ring_attention_pytorch.ring_flash_attention_cuda import ring_flash_attn_cuda

            # 使用 CUDA 实现的闪存注意力
            out = ring_flash_attn_cuda(
                q, k, v,
                mask,
                self.causal,
                self.bucket_size,
                ring_attn and not force_ring_reduce_off,
                self.striped_ring_attn and not force_ring_reduce_off,
                self.max_lookback_seq_len,
                ring_size
            )

        else:
            # 使用 Python 实现的闪存注意力
            out = ring_flash_attn(
                q, k, v,
                mask,
                self.causal,
                self.bucket_size,
                ring_attn and not force_ring_reduce_off,
                self.striped_ring_attn and not force_ring_reduce_off,
                self.max_lookback_seq_len,
                ring_size
            )

        # 合并头部
        out = rearrange('b n h d -> b n (h d)', out)
        out = self.to_out(out)

        # 如果自动分片序列为真
        if auto_shard_seq:
            # 将序列转换为批次，并截取到原始序列长度
            out, _ = sharded_seq_to_sharded_batch(out, batch_sizes)
            out = out[:, :seq_len]

        # 返回结果
        return out
# 定义一个简单的端到端测试的转换器

class RMSNorm(Module):
    # 初始化函数，接受一个维度参数
    def __init__(self, dim):
        super().__init__()
        # 计算缩放因子
        self.scale = dim ** 0.5
        # 初始化可学习参数 gamma
        self.gamma = nn.Parameter(torch.ones(dim))

    # 前向传播函数
    def forward(self, x):
        # 对输入进行归一化处理，乘以缩放因子和 gamma
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# 定义一个前馈神经网络模块
def FeedForward(dim, mult = 4):
    # 计算内部维度
    dim_inner = int(dim * mult)
    return nn.Sequential(
        RMSNorm(dim),  # 使用 RMSNorm 进行归一化
        nn.Linear(dim, dim_inner),  # 线性变换
        nn.GELU(),  # GELU 激活函数
        nn.Linear(dim_inner, dim)  # 线性变换
    )

# 定义一个环形注意力机制模块
class RingTransformer(Module):
    # 初始化函数，接受多个参数
    @beartype
    def __init__(
        self,
        *,
        num_tokens: int,
        dim: int,
        depth: int,
        causal: bool = False,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        bucket_size: int = 512,
        ring_attn: bool = False,
        striped_ring_attn: bool = False,
        ring_seq_size: int = 512,
        auto_shard_seq: Optional[bool] = None,
        max_lookback_seq_len: Optional[Union[Tuple[int, ...], int]] = None,
        rotary_embed_theta: int = 10000,    # 需要根据上下文中的百万标记进行更改
        ignore_index: int = -1
    ):
        super().__init__()
        # 初始化环形注意力机制相关参数
        self.ring_attn = ring_attn
        self.striped_ring_attn = striped_ring_attn

        self.ring_seq_size = ring_seq_size
        self.bucket_size = bucket_size
        assert divisible_by(ring_seq_size, bucket_size)

        self.auto_shard_seq = default(auto_shard_seq, ring_attn) # 如果环形注意力机制打开，则自动在序列维度上进行分片。这也可以关闭，在数据加载的其他地方手动完成

        assert not (not self.ring_attn and self.auto_shard_seq)
        assert not (not self.ring_attn and self.striped_ring_attn)
        assert not (self.striped_ring_attn and not causal), 'striped ring attention only applies to autoregressive models'

        # 初始化标记嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)

        # 初始化旋转嵌入层
        self.rotary_emb = RingRotaryEmbedding(
            dim = dim_head,
            ring = ring_attn,
            striped = striped_ring_attn,
            theta = rotary_embed_theta,
            buckets = ring_seq_size // bucket_size
        )

        # 初始化层列表
        self.layers = ModuleList([])

        max_lookback_seq_len = cast_tuple(max_lookback_seq_len, depth)
        assert len(max_lookback_seq_len) == depth

        for layer_max_lookback_seq_len in max_lookback_seq_len:

            self.layers.append(ModuleList([
                RingAttention(
                    dim = dim,
                    causal = causal,
                    dim_head = dim_head,
                    heads = heads,
                    bucket_size = bucket_size,
                    ring_attn = ring_attn,
                    ring_seq_size = ring_seq_size,
                    max_lookback_seq_len = layer_max_lookback_seq_len,
                    striped_ring_attn = striped_ring_attn,
                    auto_shard_seq = False,
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        # 输出层
        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        # 训练相关

        self.ignore_index = ignore_index

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None,
        labels = None,
        return_loss = False,
        force_ring_reduce_off = False,
        ring_size = None
        ):
        # 获取序列长度和设备信息
        seq_len, device = x.shape[-1], x.device

        # 是否自动分片序列，如果不强制关闭环形归约且自动分片序列且处于分布式环境下
        auto_shard_seq = not force_ring_reduce_off and self.auto_shard_seq and is_distributed()

        # 如果没有传入标签，则获取标签
        return_loss |= exists(labels)

        # 如果需要返回损失值且没有传入标签，则将输入数据切片为输入和标签
        if return_loss and not exists(labels):
            x, labels = x[:, :-1], x[:, 1:]

        # 处理填充以便将序列分割到不同机器上
        ring_size = default(ring_size, get_world_size())

        # 如果自动分片序列
        if auto_shard_seq:
            # 首先填充到右侧的倍数
            x, mask = maybe_pad_seq_and_mask(x, mask, self.ring_seq_size)

            # 处理标签
            if exists(labels):
                labels, label_mask = maybe_pad_seq_and_mask(labels, mask[:, 1:], self.ring_seq_size)
                labels.masked_fill_(~label_mask, self.ignore_index)

            # 考虑条纹注意力以进行工作负载平衡
            if self.striped_ring_attn:
                x = rearrange('b (i j) -> b (j i)', x, i = self.bucket_size)

                if exists(labels):
                    labels = rearrange('b (i j) -> b (j i)', labels, i = self.bucket_size)

                if exists(mask):
                    mask = rearrange('b (i j) -> b (j i)', mask, i = self.bucket_size)

            # 在批次之间收集并在世界中分割
            (x, mask), batch_sizes, num_sharded_batches = sharded_batch_to_sharded_seq(x, mask, self.ring_seq_size)

            if exists(labels):
                (labels, _), *_ = sharded_batch_to_sharded_seq(labels, None, self.ring_seq_size)

            # 根据分片批次数计算环大小
            ring_size = get_world_size() // num_sharded_batches

        # 旋转位置，考虑环和条纹
        rotary_emb = self.rotary_emb(x.shape[-1])

        # 主要的Transformer逻辑
        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(
                x,
                mask = mask,
                rotary_emb = rotary_emb,
                force_ring_reduce_off = force_ring_reduce_off,
                ring_size = ring_size
            ) + x

            x = ff(x) + x

        logits = self.to_logits(x)

        # 处理返回损失值
        if return_loss:
            logits = rearrange('b n c -> b c n', logits)

            ce_loss = F.cross_entropy(
                logits,
                labels,
                ignore_index = self.ignore_index
            )

            return ce_loss

        # 否则收集所有机器上的序列块以获取logits并分片批次维度
        if not auto_shard_seq:
            return logits

        logits = sharded_seq_to_sharded_batch(logits, batch_sizes, num_sharded_batches)

        if self.striped_ring_attn:
            logits = rearrange('b (i j) d -> b (j i) d', logits, j = self.bucket_size)

        return logits[:, :seq_len]
```