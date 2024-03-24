# `.\lucidrains\reformer-pytorch\reformer_pytorch\reformer_pytorch.py`

```
# 导入数学库
import math
# 导入 PyTorch 库
import torch
import torch.nn as nn
# 从 torch.nn 模块导入 Identity 类
from torch.nn import Identity
# 导入 torch.nn.functional 模块
import torch.nn.functional as F
# 从 torch.autograd 模块导入 Function 类
from torch.autograd import Function
# 从 functools 模块导入 partial、reduce、wraps 函数
from functools import partial, reduce, wraps
# 从 itertools 模块导入 chain 函数
from itertools import chain
# 从 operator 模块导入 mul 函数
from operator import mul

# 导入自定义模块
from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from product_key_memory import PKM
from reformer_pytorch.reversible import ReversibleSequence

# 导入 einops 库
from einops import rearrange, repeat

# 常量定义

# 用于自注意力机制的特殊值，用于半精度计算
TOKEN_SELF_ATTN_VALUE = -5e4

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 对两个张量进行排序，并返回排序后的值和对应的张量
def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

# 在指定维度上对张量进行批量索引选择
def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

# 对输入进行分块处理
def process_inputs_chunk(fn, chunks=1, dim=0):
    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), list(args) + list(values))))
        all_args = map(lambda x: (x[:len_args], dict(zip(keys, x[len_args:]))), chunked_args)
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))
    return inner_fn

# 对张量进行分块求和
def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)

# 返回默认值
def default(val, default_val):
    return default_val if val is None else val

# 将输入转换为元组
def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)

# 返回张量的最大负值
def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# 缓存函��的计算结果
def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# 缓存方法的计算结果
def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val
        return wrapper
    return inner_fn

# 在指定维度上扩展张量的维度
def expand_dim(dim, k, t):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 合并张量的维度
def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

# 在指定维度上将张量拆分为两部分
def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# 辅助类

# 始终返回固定值的模块
class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

# 矩阵乘法模块
class MatrixMultiply(nn.Module):
    def __init__(self, tensor, transpose=False, normalize=False):
        super().__init__()
        self.tensor = tensor
        self.transpose = transpose
        self.normalize = normalize
    # 定义一个前向传播函数，接受输入张量 x
    def forward(self, x):
        # 将类中的张量赋值给变量 tensor
        tensor = self.tensor
        # 如果需要进行标准化操作
        if self.normalize:
            # 对张量进行标准化操作，沿着最后一个维度进行标准化
            tensor = F.normalize(tensor, dim=-1)
        # 如果需要进行转置操作
        if self.transpose:
            # 对张量进行转置操作
            tensor = tensor.t()
        # 返回输入张量与处理后的张量的矩阵乘法结果
        return x @ tensor
# 定义 ReZero 类，继承自 nn.Module
class ReZero(nn.Module):
    # 初始化函数，接受一个函数 fn 作为参数
    def __init__(self, fn):
        super().__init__()
        # 创建一个可学习的参数 g，初始化为零
        self.g = nn.Parameter(torch.zeros(1))
        # 将传入的函数 fn 赋值给 self.fn
        self.fn = fn

    # 前向传播函数，接受输入 x 和其他关键字参数
    def forward(self, x, **kwargs):
        # 返回经过函数 fn 处理后的结果乘以参数 g
        return self.fn(x, **kwargs) * self.g

# 定义 ScaleNorm 类，继承自 nn.Module
class ScaleNorm(nn.Module):
    # 初始化函数，接受维度 dim 和一个小数 eps 作为参数
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        # 创建一个可学习的参数 g，初始化为一
        self.g = nn.Parameter(torch.ones(1))
        # 将传入的 eps 赋值给 self.eps
        self.eps = eps

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 计算 x 在指定维度上的范数，并限制最小值为 eps
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        # 返回 x 除以范数后乘以参数 g 的结果
        return x / n * self.g

# 定义 PreNorm 类，继承自 nn.Module
class PreNorm(nn.Module):
    # 初始化函数，接受一个规范化类 norm_class、维度 dim 和一个函数 fn 作为参数
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        # 创建一个 norm_class 类型的规范化对象，并赋值给 self.norm
        self.norm = norm_class(dim)
        # 将传入的函数 fn 赋值给 self.fn
        self.fn = fn

    # 前向传播函数，接受输入 x 和其他关键字参数
    def forward(self, x, **kwargs):
        # 对输入 x 进行规范化
        x = self.norm(x)
        # 返回经过函数 fn 处理后的结果
        return self.fn(x, **kwargs)

# 定义 Chunk 类，继承自 nn.Module
class Chunk(nn.Module):
    # 初始化函数，接受块数 chunks、函数 fn 和沿着的维度 along_dim 作为参数
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        # 将 along_dim 赋值给 self.dim
        self.dim = along_dim
        # 将 chunks 和 fn 赋值给 self.chunks 和 self.fn
        self.chunks = chunks
        self.fn = fn

    # 前向传播函数，接受输入 x 和其他关键字参数
    def forward(self, x, **kwargs):
        # 如果 chunks 等于 1，则直接返回经过函数 fn 处理后的结果
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        # 将输入 x 沿着维度 self.dim 切分成多个块
        chunks = x.chunk(self.chunks, dim=self.dim)
        # 对每个块应用函数 fn，并在指定维度上拼接结果
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)

# LSH attention 类，实现了论文中描述的 LSH 注意力机制
class LSHAttention(nn.Module):
    # 初始化函数，接受多个参数设置
    def __init__( self,
                  dropout=0.,
                  bucket_size=64,
                  n_hashes=8,
                  causal=False,
                  allow_duplicate_attention=True,
                  attend_across_buckets=True,
                  rehash_each_round=True,
                  drop_for_hash_rate=0.0,
                  random_rotations_per_head=False,
                  return_attn=False):
        super().__init__()
        # 如果 dropout 大于等于 1，则抛出异常
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        # 创建一个 dropout 层，用于在训练时随机丢弃部分数据
        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        # 确保每轮重新哈希或允许重复注意力的设置
        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        # 设置是否是因果关系
        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # 是否返回注意力矩阵
        self._return_attn = return_attn

        # 用于缓存可逆网络的桶，作者报告这样可以使 Reformer 在深度上工作
        self._cache = {}

    # 缓存方法装饰器，用于缓存 buckets
    @cache_method_decorator('_cache', 'buckets', reexecute=True)
    # 对输入的向量进行哈希处理，将其映射到指定数量的桶中
    def hash_vectors(self, n_buckets, vecs):
        # 获取输入向量的批量大小
        batch_size = vecs.shape[0]
        # 获取输入向量所在设备
        device = vecs.device

        # 参考论文 https://arxiv.org/pdf/1509.02897.pdf
        # 为每一轮哈希采样不同的随机旋转，以减少哈希失配的概率
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        # 生成随机旋转矩阵
        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        # 对输入向量进行哈希前的丢弃处理
        dropped_vecs = self.dropout_for_hash(vecs)
        # 对丢弃后的向量进行旋转操作
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            # 如果每轮都重新哈希，则将旋转后的向量进行拼接
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # 获取每个向量对应的桶索引
            buckets = torch.argmax(rotated_vecs, dim=-1)
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # 在这种配置下，将每个项目映射到前 self.n_hashes 个桶中
            rotated_vecs = torch.squeeze(rotated_vecs, 1)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)

            # 对旋转后的向量进行排序，获取对应的桶索引
            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            # 调整桶索引的维度
            buckets = buckets[... , -self.n_hashes:].transpose(1, 2)

        # 每个哈希轮次的桶索引现在是 (self.n_hashes, seq_len) 的形状。接下来添加偏移量，以避免不同哈希轮次的桶号重叠
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        # 返回最终的桶索引
        return buckets
# 定义全连接的注意力机制类
class FullQKAttention(nn.Module):
    def __init__(self, causal = False, dropout = 0.):
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None, **kwargs):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type_as(q)

        dot = torch.einsum('bie,bje->bij', q, qk) * (dim ** -0.5)

        # qk attention requires tokens not attend to self
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seq_len - input_attn_mask.shape[-1]), value=True)
            dot.masked_fill_(~input_attn_mask, masked_value)

        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, i, j] = masked_value

        dot = dot.softmax(dim=-1)
        dot = self.dropout(dot)

        out = torch.einsum('bij,bje->bie', dot, v)

        return out, dot, torch.empty(0)

# 共享的 qk 注意力机制，使用全局或 LSH 注意力机制
class LSHSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, bucket_size = 64, n_hashes = 8, causal = False, dim_head = None, attn_chunks = 1, random_rotations_per_head = False, attend_across_buckets = True, allow_duplicate_attention = True, num_mem_kv = 0, one_value_head = False, use_full_attn = False, full_attn_thres = None, return_attn = False, post_attn_dropout = 0., dropout = 0., n_local_attn_heads = 0, **kwargs):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks = default(attn_chunks, 1)

        self.v_head_repeats = (heads if one_value_head else 1)
        v_dim = dim_heads // self.v_head_repeats

        self.toqk = nn.Linear(dim, dim_heads, bias = False)
        self.tov = nn.Linear(dim, v_dim, bias = False)
        self.to_out = nn.Linear(dim_heads, dim)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal, random_rotations_per_head=random_rotations_per_head, attend_across_buckets = attend_across_buckets,  allow_duplicate_attention = allow_duplicate_attention, return_attn = return_attn, dropout = dropout, **kwargs)
        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(window_size=bucket_size * 2, causal=causal, dropout=dropout, shared_qk=True, look_forward=(1 if not causal else 0))

        self.callback = None
    # 定义前向传播函数，接受输入 x 和其他可选参数
    def forward(self, x, keys = None, input_mask = None, input_attn_mask = None, context_mask = None, pos_emb = None, **kwargs):
        # 获取输入 x 的设备和数据类型
        device, dtype = x.device, x.dtype
        # 获取输入 x 的形状信息
        b, t, e, h, dh, m, l_h = *x.shape, self.heads, self.dim_head, self.num_mem_kv, self.n_local_attn_heads

        # 初始化记忆键值对
        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem = mem_kv.expand(b, m, -1)

        # 初始化键
        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]

        # 计算键值对的长度
        kv_len = t + m + c
        # 判断是否使用全局注意力
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres

        # 将输入 x、记忆和键连接起来
        x = torch.cat((x, mem, keys), dim=1)
        # 将输入 x 转换为查询和键
        qk = self.toqk(x)
        # 将输入 x 转换为值
        v = self.tov(x)
        # 复制值以匹配头数
        v = v.repeat(1, 1, self.v_head_repeats)

        # 定义合并头部的函数
        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        # 定义分割头部的函数
        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        # 合并批次和头部维度
        merge_batch_and_heads = partial(merge_dims, 0, 1)

        # 对查询和键值对进行头部合并
        qk, v = map(merge_heads, (qk, v))

        # 判断是否有局部注意力
        has_local = l_h > 0
        lsh_h = h - l_h

        # 分割索引函数
        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
        lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))

        # 初始化掩码字典
        masks = {}
        # 如果存在输入掩码或上下文掩码
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks['input_mask'] = mask

        # 如果存在输入注意力掩码
        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(expand_dim(1, lsh_h, input_attn_mask))
            masks['input_attn_mask'] = input_attn_mask

        # 根据是否使用全局注意力选择不同的注意力函数
        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn
        partial_attn_fn = partial(attn_fn, query_len = t, pos_emb = pos_emb, **kwargs)
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks = self.attn_chunks)

        # 执行注意力函数
        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)

        # 如果存在回调函数，则执行回调
        if self.callback is not None:
            self.callback(attn.reshape(b, lsh_h, t, -1), buckets.reshape(b, lsh_h, -1))

        # 如果存在局部注意力
        if has_local:
            lqk, lv = lqk[:, :t], lv[:, :t]
            local_out = self.local_attn(lqk, lqk, lv, input_mask=input_mask)
            local_out = local_out.reshape(b, l_h, t, -1)
            out = out.reshape(b, lsh_h, t, -1)
            out = torch.cat((local_out, out), dim=1)

        # 分割头部并重塑输出
        out = split_heads(out).view(b, t, -1)
        out = self.to_out(out)
        return self.post_attn_dropout(out)
# 定义 GELU 激活函数类，继承自 nn.Module
class GELU_(nn.Module):
    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 使用 GELU 激活函数计算输出
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))

# 如果 nn 模块中存在 GELU 类，则使用 nn.GELU，否则使用自定义的 GELU_ 类
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# 定义前馈神经网络类 FeedForward，继承自 nn.Module
class FeedForward(nn.Module):
    # 初始化函数，接受维度 dim、倍数 mult、dropout 概率、激活函数 activation 和 glu 标志
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        # 设置激活函数为默认值 GELU
        activation = default(activation, GELU)

        self.glu = glu
        # 第一层全连接层，输入维度为 dim，输出维度为 dim * mult * (2 if glu else 1)
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        # 激活函数层
        self.act = activation()
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        # 第二层全连接层，输入维度为 dim * mult，输出维度为 dim
        self.w2 = nn.Linear(dim * mult, dim)

    # 前向传播函数，接受输入 x 和其他参数
    def forward(self, x, **kwargs):
        # 如果不使用 glu
        if not self.glu:
            # 进行第一层全连接层和激活函数的计算
            x = self.w1(x)
            x = self.act(x)
        else:
            # 如果使用 glu，进行特殊处理
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        # Dropout
        x = self.dropout(x)
        # 第二层全连接层计算结果
        x = self.w2(x)
        return x

# 绝对位置嵌入类，继承自 nn.Module
class AbsolutePositionalEmbedding(nn.Module):
    # 初始化函数，接受维度 dim 和最大序列长度 max_seq_len
    def __init__(self, dim, max_seq_len):
        super().__init__()
        # 创建 Embedding 层，输入维度为最大序列长度，输出维度为 dim
        self.emb = nn.Embedding(max_seq_len, dim)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 生成序列长度的张量 t
        t = torch.arange(x.shape[1], device=x.device)
        # 返回位置嵌入结果
        return self.emb(t)

# 固定位置嵌入类，继承自 nn.Module
class FixedPositionalEmbedding(nn.Module):
    # 初始化函数，接受维度 dim
    def __init__(self, dim):
        super().__init__()
        # 计算频率
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 将频率作为缓冲区
        self.register_buffer('inv_freq', inv_freq)

    # 前向传播函数，接受输入 x 和序列维度 seq_dim
    def forward(self, x, seq_dim=1):
        # 生成序列长度的张量 t
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        # 计算正弦和余弦位置嵌入
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :].type_as(x)

# 旋转位置嵌入辅助函数，用于旋转每两个元素
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')

# 应用旋转位置嵌入函数，接受查询键 qk 和正弦位置 sinu_pos
def apply_rotary_pos_emb(qk, sinu_pos):
    sinu_pos = sinu_pos.type(qk.dtype)
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, 'n d -> n (d j)', j=2), (sin, cos))
    seq_len = sin.shape[0]
    qk, qk_pass = qk[:, :seq_len], qk[:, seq_len:]
    qk = (qk * cos) + (rotate_every_two(qk) * sin)
    return torch.cat((qk, qk_pass), dim=1)

# Reformer 语言模型类，继承自 nn.Module
class Reformer(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, dim, depth, heads = 8, dim_head = None, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., ff_dropout = 0., ff_activation = None, ff_mult = 4, ff_glu = False, post_attn_dropout = 0., layer_dropout = 0., lsh_attend_across_buckets = True, lsh_allow_duplicate_attention = True, random_rotations_per_head = False, use_scale_norm = False, use_rezero = False, use_full_attn = False, full_attn_thres = 0, reverse_thres = 0, num_mem_kv = 0, one_value_head = False, n_local_attn_heads = 0, pkm_layers = tuple(), pkm_num_keys = 128):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的维度和深度
        self.dim = dim
        self.depth = depth

        # 设置桶的大小和记忆键值对的数量
        self.bucket_size = bucket_size
        self.num_mem_kv = num_mem_kv

        # 设置全局注意力的阈值
        self.full_attn_thres = full_attn_thres

        # 定义获取注意力和前馈网络的函数
        get_attn = lambda: LSHSelfAttention(dim, heads, bucket_size, n_hashes, causal = causal, dim_head = dim_head, dropout = lsh_dropout, post_attn_dropout = post_attn_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head, num_mem_kv = num_mem_kv, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, one_value_head = one_value_head, n_local_attn_heads = n_local_attn_heads)
        get_ff = lambda: Chunk(ff_chunks, FeedForward(dim, dropout = ff_dropout, activation = ff_activation, mult = ff_mult, glu = ff_glu), along_dim = -2)
        get_pkm = lambda: PKM(dim, num_keys = pkm_num_keys)

        # 如果权重共享为真，则对获取注意力和前馈网络的函数进行缓存
        if weight_tie:
            get_attn, get_ff, get_pkm = map(cache_fn, (get_attn, get_ff, get_pkm))

        # 初始化块列表
        blocks = []

        # 根据是否使用标准化类型，选择不同的标准化函数
        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm

        # 根据是否使用 ReZero，选择不同的残差函数
        residual_fn_wrapper = ReZero if use_rezero else partial(PreNorm, norm_type, dim)

        # 循环构建深度个块
        for ind in range(depth):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)
            parallel_net = None

            # 获取注意力和前馈网络
            attn = get_attn()

            if use_pkm:
                parallel_net = get_pkm()
            else:
                parallel_net = get_ff()

            f = residual_fn_wrapper(attn)
            g = residual_fn_wrapper(parallel_net)

            blocks.append(nn.ModuleList([f, g]))

        # 构建可逆序列
        self.layers = ReversibleSequence(nn.ModuleList(blocks), layer_dropout = layer_dropout, reverse_thres = reverse_thres, send_signal = True)

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 在最后一个维度上拼接输入张量
        x = torch.cat([x, x], dim = -1)
        # 使用可逆序列进行前向传播
        x = self.layers(x, **kwargs)
        # 将结果张量按最后一个维度分块，取均值
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)
class ReformerLM(nn.Module):
    # 定义 ReformerLM 类，继承自 nn.Module
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads = 8, dim_head = 64, bucket_size = 64, n_hashes = 4, ff_chunks = 100, attn_chunks = 1, causal = False, weight_tie = False, lsh_dropout = 0., ff_dropout = 0., ff_mult = 4, ff_activation = None, ff_glu = False, post_attn_dropout = 0., layer_dropout = 0., random_rotations_per_head = False, use_scale_norm = False, use_rezero = False, use_full_attn = False, full_attn_thres = 0, reverse_thres = 0, num_mem_kv = 0, one_value_head = False, emb_dim = None, return_embeddings = False, weight_tie_embedding = False, fixed_position_emb = False, absolute_position_emb = False, axial_position_emb = False, axial_position_shape = None, n_local_attn_heads = 0, pkm_layers = tuple(), pkm_num_keys = 128):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数

        emb_dim = default(emb_dim, dim)
        # 如果 emb_dim 为 None，则使用 dim

        self.max_seq_len = max_seq_len
        # 设置最大序列长度

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        # 创建一个嵌入层，用于将输入的 token 转换为向量表示

        self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)
        # 如果 emb_dim 等于 dim，则使用 Identity()，否则使用线性层将 emb_dim 转换为 dim

        self.pos_emb = Always(0)
        self.layer_pos_emb = Always(None)
        # 初始化位置编码

        if axial_position_emb:
            # 如果启用轴向位置编码
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / bucket_size), bucket_size))
            # 计算轴向位置编码的形状
            self.pos_emb = AxialPositionalEmbedding(emb_dim, axial_position_shape)
            # 创建轴向位置编码
        elif absolute_position_emb:
            # 如果启用绝对位置编码
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
            # 创建绝对位置编码
        elif fixed_position_emb:
            # 如果启用固定位置编码
            self.pos_emb = FixedPositionalEmbedding(emb_dim)
            # 创建固定位置编码
        else:
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head)
            # 创建固定位置编码

        self.reformer = Reformer(dim, depth, heads = heads, dim_head = dim_head, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, ff_mult = ff_mult, ff_activation = ff_activation, ff_glu = ff_glu, ff_dropout = ff_dropout, post_attn_dropout = 0., layer_dropout = layer_dropout, random_rotations_per_head = random_rotations_per_head, use_scale_norm = use_scale_norm, use_rezero = use_rezero, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, reverse_thres = reverse_thres, num_mem_kv = num_mem_kv, one_value_head = one_value_head, n_local_attn_heads = n_local_attn_heads, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys)
        # 创建 Reformer 模型

        self.norm = nn.LayerNorm(dim)
        # 创建 LayerNorm 层

        if return_embeddings:
            self.out = Identity()
            return
            # 如果需要返回嵌入向量，则直接返回

        self.out = nn.Sequential(
            nn.Linear(dim, emb_dim) if emb_dim != dim else Identity(),
            nn.Linear(emb_dim, num_tokens) if not weight_tie_embedding else MatrixMultiply(self.token_emb.weight, transpose=True, normalize=True)
        )
        # 创建输出层，根据是否需要权重共享选择不同的操作

    def forward(self, x, **kwargs):
        # 前向传播函数
        x = self.token_emb(x)
        # 将输入的 token 转换为向量表示
        x = x + self.pos_emb(x)
        # 添加位置编码到输入向量中

        layer_pos_emb = self.layer_pos_emb(x)
        # 获取层级位置编码
        x = self.to_model_dim(x)
        # 将输入向量转换为模型维度
        x = self.reformer(x, pos_emb = layer_pos_emb, **kwargs)
        # 使用 Reformer 模型进行处理
        x = self.norm(x)
        # 对输出进行 LayerNorm 处理
        return self.out(x)
        # 返回输出结果
```