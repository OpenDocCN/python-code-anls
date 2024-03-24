# `.\lucidrains\routing-transformer\routing_transformer\routing_transformer.py`

```
# 导入 torch 库
import torch
# 导入 torch 中的神经网络模块
import torch.nn as nn
# 导入 torch 中的函数操作模块
import torch.nn.functional as F
# 导入 math 库
import math
# 从 inspect 模块中导入 isfunction 函数
from inspect import isfunction
# 从 operator 模块中导入 mul 函数
from operator import mul
# 从 functools 模块中导入 partial, reduce, wraps 函数
from functools import partial, reduce, wraps

# 从 einops 库中导入 rearrange, repeat 函数
from einops import rearrange, repeat
# 从 einops.layers.torch 模块中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 从 local_attention 模块中导入 LocalAttention 类
from local_attention import LocalAttention
# 从 product_key_memory 模块中导入 PKM 类
from product_key_memory import PKM
# 从 mixture_of_experts 模块中导入 MoE 类
from mixture_of_experts import MoE
# 从 routing_transformer.reversible 模块中导入 ReversibleSequence, SequentialSequence 类

# 常量定义

# 定义 TOKEN_SELF_ATTN_VALUE 常量为 -5e4
TOKEN_SELF_ATTN_VALUE = -5e4
# 定义 KMEAN_INIT_ITERS 常量为 10
KMEAN_INIT_ITERS = 10

# 辅助函数

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 返回输入值的函数
def identity(x, *args, **kwargs):
    return x

# 如果输入值不存在，则返回默认值的函数
def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

# 将输入值转换为元组的函数
def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)

# 缓存函数的装饰器
def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if exists(cache):
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# 组合多个函数的函数
def compose(*fns):
    def inner(x, *args, **kwargs):
        for fn in reversed(fns):
            x = fn(x, *args, **kwargs)
        return x
    return inner

# 返回输入张量的设备和数据类型的字典的函数
def to(t):
    return {'device': t.device, 'dtype': t.dtype}

# 查找神经网络模块中指定类型的模块的函数
def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# 判断张量是否为空的函数
def is_empty(t):
    return t.nelement() == 0

# 返回指定张量数据类型的最大负值的函数
def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# 在指定维度上对张量进行批量索引选择的函数
def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(2, expand_dim(indices, -1, last_dim))

# 合并张量的维度的函数
def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

# 在指定维度上扩展张量的函数
def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 在指定维度上对张量进行均值散开的函数
def scatter_mean(src, t, index, dim, eps = 1e-5):
    numer = src.scatter_add(dim, index, t)
    denom = src.scatter_add(dim, index, torch.ones_like(t))
    return numer / (denom + eps)

# 在指定维度上将张量拆分为两部分的函数
def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# 重塑张量的维度的函数
def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim+1] = split_dims
    return t.reshape(shape)

# 指数移动平均的函数
def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

# 就地指数移动平均的函数
def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))

# 辅助类

# 对第一个元组或元素应用函数的类
class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks <= 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

# 具有预处理的模块列表的类
class PreNorm(nn.ModuleList):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# ReZero 模块
class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.residual_weight = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs)
        return map_first_tuple_or_el(x, lambda t: t * self.residual_weight)
# 定义 ScaleNorm 类，用于对输入进行归一化处理
class ScaleNorm(nn.Module):
    # 初始化函数，设置归一化参数和阈值
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    # 前向传播函数，对输入进行归一化处理
    def forward(self, x):
        # 定义内部函数 norm，用于计算归一化后的值
        def norm(t):
            # 计算输入张量 t 在指定维度上的 L2 范数，并进行归一化处理
            n = torch.norm(t, dim=-1, keepdim=True).clamp(min=self.eps)
            return t / n * self.g
        # 调用 map_first_tuple_or_el 函数，对输入进行处理
        return map_first_tuple_or_el(x, norm)

# 定义 ProjectInOut 类，用于对输入进行线性投影
class ProjectInOut(nn.Module):
    # 初始化函数，设置投影函数和维度参数
    def __init__(self, fn, dim_in, dim_out, project_out = True):
        super().__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    # 前向传播函数，对输入进行线性投影处理
    def forward(self, x, **kwargs):
        # 对输入进行投影处理
        x = self.project_in(x)
        # 调用 fn 函数处理投影后的结果
        x, loss = self.fn(x, **kwargs)
        # 对输出进行反向投影处理
        x = self.project_out(x)
        return x, loss

# 定义 MatrixMultiply 类，用于矩阵乘法操作
class MatrixMultiply(nn.Module):
    # 初始化函数，设置矩阵和是否转置参数
    def __init__(self, tensor, transpose = False):
        super().__init__()
        self.tensor = tensor
        self.transpose = transpose

    # 前向传播函数，进行矩阵乘法操作
    def forward(self, x):
        tensor = self.tensor
        # 如果需要转置，则对矩阵进行转置操作
        if self.transpose:
            tensor = tensor.t()
        return x @ tensor

# 定义 token shift 函数，用于对输入进行位移操作
def shift(t, amount, mask = None):
    # 如果位移量为 0，则直接返回输入
    if amount == 0:
        return t

    # 如果存在掩码，则根据掩码进行填充操作
    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

# 定义 PreShiftTokens 类，用于对输入进行预位移操作
class PreShiftTokens(nn.Module):
    # 初始化函数，设置位移量和处理函数
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    # 前向传播函数，对输入进行预位移处理
    def forward(self, x, **kwargs):
        # 获取掩码信息
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# 定义 FixedPositionalEmbedding 类，用于固定位置编码
class FixedPositionalEmbedding(nn.Module):
    # 初始化函数，设置维度和最大序列长度
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    # 前向传播函数，返回固定位置编码结果
    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)

# 定义 rotate_every_two 函数，用于对输入进行旋转操作
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

# 定义 apply_rotary_pos_emb 函数，用于应用旋转位置编码
def apply_rotary_pos_emb(q, k, v, sinu_pos):
    sinu_pos = sinu_pos.type(q.dtype)
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k, v = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k, v))
    return q, k, v

# 定义 update_kmeans_on_backwards 函数，用于在反向传播时更新 kmeans 模块
def update_kmeans_on_backwards(module):
    module.kmean_modules = find_modules(module, Kmeans)
    def hook(_, grad_in, grad_out):
        for m in module.kmean_modules:
            m.update()

    return module.register_backward_hook(hook)

# 定义 similarity 函数，用于计算输入与均值之间的相似度
def similarity(x, means):
    return torch.einsum('bhld,hcd->bhlc', x, means)

# 定义 dists_and_buckets 函数，用于计算距离和分桶
def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

# 定义 batched_bincount 函数，用于批量计算索引的频次
def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

# 定义 kmeans_iter 函数，用于执行 kmeans 迭代
def kmeans_iter(x, means, buckets = None):
    b, h, l, d, dtype, num_clusters = *x.shape, x.dtype, means.shape[1]
    # 如果 buckets 不存在，则通过 dists_and_buckets 函数计算出来
    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    # 对 buckets 进行批量计数，然后对结果进行求和
    bins = batched_bincount(buckets, num_clusters).sum(0, keepdim=True)
    # 创建一个与 bins 形状相同的布尔张量，标记 bins 中为 0 的位置
    zero_mask = bins.long() == 0

    # 创建一个与 buckets 相同形状的全零张量 means_
    means_ = buckets.new_zeros(b, h, num_clusters, d, dtype=dtype)
    # 在指定维度上对 means_ 进行 scatter_add_ 操作，将 x 散射到 means_ 上
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    # 对 means_ 沿着指定维度求和，并进行归一化，然后转换为指定数据类型
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)

    # 使用 torch.where 函数根据 zero_mask 的值选择更新后的 means_ 或保持原来的 means
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    # 去除 means 的第一个维度，返回结果
    means = means.squeeze(0)
    # 返回计算得到的 means
    return means
# 根据距离矩阵和窗口大小，获取最大的 k 个索引
_, topk_indices = dists.topk(k=window_size, dim=-2)
# 转置索引矩阵
indices = topk_indices.transpose(-2, -1)
# 重新整形索引矩阵
return indices.reshape(*indices.size()[:2], -1)

# Kmeans 类定义
class Kmeans(nn.Module):
    def __init__(self, num_heads, head_dim, num_clusters, ema_decay = 0.999, commitment = 1e-4):
        super().__init__()
        self.commitment = commitment
        self.ema_decay = ema_decay

        # 注册缓冲区，存储聚类中心和初始化状态
        self.register_buffer('means', torch.randn(num_heads, num_clusters, head_dim))
        self.register_buffer('initted', torch.tensor(False))
        self.num_new_means = 0
        self.new_means = None

    @torch.no_grad()
    def init(self, x):
        if self.initted:
            return
        _, h, _, d, device, dtype = *x.shape, x.device, x.dtype

        num_clusters = self.means.shape[1]

        # 调整输入数据形状
        means = x.transpose(0, 1).contiguous().view(h, -1, d)
        num_samples = means.shape[1]

        # 初始化聚类中心
        if num_samples >= num_clusters:
            indices = torch.randperm(num_samples, device=device)[:num_clusters]
        else:
            indices = torch.randint(0, num_samples, (num_clusters,), device=device)

        means = means[:, indices]

        # 迭代更新聚类中心
        for _ in range(KMEAN_INIT_ITERS):
            means = kmeans_iter(x, means)

        self.num_new_means = 0
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))

    @torch.no_grad()
    def update(self, new_means = None):
        new_means = default(new_means, self.new_means)
        assert exists(new_means), 'new kmeans has not been supplied'
        # 更新聚类中心
        ema_inplace(self.means, new_means, self.ema_decay)

        del self.new_means
        self.new_means = None
        self.num_new_means = 0

    def forward(self, x, update_means = False):
        self.init(x)

        b, dtype = x.shape[0], x.dtype
        means = self.means.type(dtype)
        x = F.normalize(x, 2, dim=-1).type(dtype)

        with torch.no_grad():
            dists, buckets = dists_and_buckets(x, means)

        routed_means = batched_index_select(expand_dim(means, 0, b), buckets)
        loss = F.mse_loss(x, routed_means) * self.commitment

        if update_means:
            with torch.no_grad():
                means = kmeans_iter(x, means, buckets)
            self.new_means = ema(self.new_means, means, self.num_new_means / (self.num_new_means + 1))
            self.num_new_means += 1

        return dists, loss

# KmeansAttention 类定义
class KmeansAttention(nn.Module):
    def __init__(self, num_clusters, window_size, num_heads, head_dim, causal = False, dropout = 0., ema_decay = 0.999, commitment = 1e-4, context_window_size = None, receives_context = False, num_mem_kv = 0, shared_qk = False):
        super().__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = head_dim

        self.window_size = window_size
        self.context_window_size = default(context_window_size, window_size)
        self.causal = causal

        self.shared_qk = shared_qk
        self.receives_context = receives_context
        self.kmeans = Kmeans(num_heads, head_dim, num_clusters, ema_decay, commitment)
        self.dropout = nn.Dropout(dropout)

        self.num_mem_kv = max(num_mem_kv, 1 if causal and not shared_qk else 0)
        self.mem_key = nn.Parameter(torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim))
        self.mem_value = nn.Parameter(torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim))
    # 定义前向传播函数，接受查询 q、键 k、值 v，以及可选的查询和键的掩码
    def forward(self, q, k, v, query_mask = None, key_mask = None, **kwargs):
        # 解包变量 b、h、t、d、kv_t、wsz、c_wsz、nc、device、dtype
        b, h, t, d, kv_t, wsz, c_wsz, nc, device, dtype = *q.shape, k.shape[2], self.window_size, self.context_window_size, self.num_clusters, q.device, q.dtype
        # 从 kwargs 中弹出 '_reverse' 键值对，默认为 False
        is_reverse = kwargs.pop('_reverse', False)

        # 创建与 q 相同形状的零张量 out
        out = torch.zeros_like(q, dtype=dtype)

        # 更新 kmeans 模型的标志，训练中且非反向传播时更新
        update_kmeans = self.training and not is_reverse
        
        # 如果不接收上下文信息，则 key_mask 默认为 query_mask
        key_mask = default(key_mask, query_mask) if not self.receives_context else key_mask
        # 如果不接收上下文信息，则 kv_wsz 为 wsz，否则为 c_wsz
        kv_wsz = wsz if not self.receives_context else c_wsz

        # 更新 wsz 和 kv_wsz 为 t 和 kv_t 的最小值
        wsz = min(wsz, t)
        kv_wsz = min(kv_wsz, kv_t)

        # 如果不共享查询和键或者接收上下文信息
        if not self.shared_qk or self.receives_context:
            # 使用 kmeans 模型计算 q 和 k 的聚类中心距离，返回聚类中心距离和辅助损失
            dists, aux_loss = self.kmeans(torch.cat((q, k), dim=2), update_kmeans)
            # 将 dists 按索引 2 分割为 q_dists 和 k_dists
            q_dists, k_dists = split_at_index(2, t, dists)
            # 根据 q_dists 和 wsz 计算索引
            indices = distribution(q_dists, wsz)
            # 根据 k_dists 和 kv_wsz 计算索引
            kv_indices = distribution(k_dists, kv_wsz)
        else:
            # 使用 kmeans 模型计算 q 的聚类中心距离，返回聚类中心距离和辅助损失
            dists, aux_loss = self.kmeans(q, update_kmeans)
            # 对 k 进行归一化，并转换为与 q 相同的类型
            k = F.normalize(k, dim=-1).to(q)
            # 根据 dists 和 wsz 计算索引
            indices = distribution(dists, wsz)
            # kv_indices 与 indices 相同
            kv_indices = indices

        # 根据索引选择 q、k、v 的子集
        q = batched_index_select(q, indices)
        k = batched_index_select(k, kv_indices)
        v = batched_index_select(v, kv_indices)

        # 定义 reshape_with_window 函数，用于将张量重塑为指定形状
        reshape_with_window = lambda x: x.reshape(b, h, nc, -1, d)
        # 将 q、k、v 分别应用 reshape_with_window 函��
        q, k, v = map(reshape_with_window, (q, k, v))

        # 将 self.mem_key 和 self.mem_value 扩展为与 q 相同的形状
        m_k, m_v = map(lambda x: expand_dim(x, 0, b).to(q), (self.mem_key, self.mem_value))
        # 将 k、v 与 m_k、m_v 连接在最后一个维度上
        k, v = map(lambda x: torch.cat(x, dim=3), ((m_k, k), (m_v, v)))

        # 计算点积，乘以缩放因子
        dots = torch.einsum('bhnid,bhnjd->bhnij', q, k) * (d ** -0.5)

        # 计算掩码值
        mask_value = max_neg_value(dots)

        # 如果存在查询或键的掩码
        if exists(query_mask) or exists(key_mask):
            # 默认创建查询掩码为全 1，键掩码为全 1
            query_mask = default(query_mask, lambda: torch.ones((b, t), device=device).bool())
            key_mask = default(key_mask, lambda: torch.ones((b, kv_t), device=device).bool())

            # 根据 indices 和 kv_indices 从掩码中选择子集
            q_mask = expand_dim(query_mask, 1, h).gather(2, indices)
            kv_mask = expand_dim(key_mask, 1, h).gather(2, kv_indices)
            # 将 q_mask、kv_mask 重塑为指定形状
            q_mask, kv_mask = map(lambda t: t.reshape(b, h, nc, -1), (q_mask, kv_mask))
            # 创建掩码，填充边界
            mask = q_mask[:, :, :, :, None] * kv_mask[:, :, :, None, :]
            mask = F.pad(mask, (self.num_mem_kv, 0), value=True)
            # 将 dots 中不符合掩码条件的位置填充为 mask_value
            dots.masked_fill_(~mask, mask_value)
            del mask

        # 如果是因果注意力机制
        if self.causal:
            # 将 indices、kv_indices 重塑为指定形状
            q_mask, kv_mask = map(lambda t: t.reshape(b, h, nc, -1), (indices, kv_indices))
            # 创建因果掩码
            mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
            mask = F.pad(mask, (self.num_mem_kv, 0), value=True)
            # 将 dots 中不符合掩码条件的位置填充为 mask_value
            dots.masked_fill_(~mask, mask_value)
            del mask            

        # 如果共享查询和键
        if self.shared_qk:
            # 将 indices、kv_indices 重塑为指定形状
            q_mask, kv_mask = map(lambda t: t.reshape(b, h, nc, -1), (indices, kv_indices))
            # 创建自注意力掩码
            mask = q_mask[:, :, :, :, None] == kv_mask[:, :, :, None, :]
            mask = F.pad(mask, (self.num_mem_kv, 0), value=False)
            # 将 dots 中符合掩码条件的位置填充为 TOKEN_SELF_ATTN_VALUE
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        # 对 dots 进行 softmax 操作
        dots = dots.softmax(dim=-1)
        # 对 dots 进行 dropout 操作
        dots = self.dropout(dots)

        # 计算输出张量 bo
        bo = torch.einsum('bhcij,bhcjd->bhcid', dots, v)
        # 将 bo 重塑为指定形状
        so = torch.reshape(bo, (b, h, -1, bo.shape[-1])).type(dtype)
        # 对输出张量 out 进行 scatter_mean 操作
        out = scatter_mean(out, so, indices.unsqueeze(-1).expand_as(so), -2)
        # 返回输出张量 out 和辅助损失
        return out, aux_loss
# 定义 GELU 激活函数类
class GELU_(nn.Module):
    # 前向传播函数
    def forward(self, x):
        # GELU 激活函数的计算公式
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))

# 如果 nn 模块中存在 GELU 函数，则使用 nn.GELU，否则使用自定义的 GELU_ 函数
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# 定义前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        # 设置激活函数为 GELU
        activation = default(activation, GELU)

        self.glu = glu
        # 第一个全连接层
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        # 激活函数层
        self.act = activation()
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        # 第二个全连接层
        self.w2 = nn.Linear(dim * mult, dim)

    # 前向传播函数
    def forward(self, x, **kwargs):
        if not self.glu:
            # 非 GLU 模式下的前向传播
            x = self.w1(x)
            x = self.act(x)
        else:
            # GLU 模式下的前向传播
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# 自注意力机制类
class SelfAttention(nn.Module):
    def __init__(self,  dim, depth, max_seq_len, heads, local_attn_heads, window_size, dim_head = None, local_attn_window_size = None, local_attn_radius_blocks = 1, causal = False, attn_dropout = 0., dropout = 0., kmeans_ema_decay = 0.999, commitment_factor = 1e-4, receives_context = False, context_window_size = None, rel_pos_emb = True, num_mem_kv = 0, shared_qk = False, conv_query_kernel = 9):
        super().__init__()
        # 断言确保隐藏维度可以被头数整除
        assert dim_head or (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        # 断言确保最大序列长度可以被窗口大小整除
        assert (max_seq_len % window_size) == 0, 'maximum sequence length must be divisible by the target window size'
        # 断言确保本地注意力头数小于总头数
        assert local_attn_heads <= heads, 'number of local attention heads must be less than total heads'
        # 断言确保本地注意力和上下文注意力不能同时使用
        assert not (receives_context and local_attn_heads > 0), 'local attention cannot be used for self attention with context'
        # 断言确保上下文注意力和因果��不能同时使用
        assert not (receives_context and causal), 'contextual attention layer cannot be causal'

        local_attn_window_size = default(local_attn_window_size, window_size)
        context_window_size = default(context_window_size, window_size)

        self.shared_qk = shared_qk
        self.receives_context = receives_context
        self.heads = heads
        self.local_attn_heads = local_attn_heads
        self.global_attn_heads = heads - local_attn_heads

        self.causal = causal
        self.window_size = window_size

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads
        self.dim_head = dim_head

        num_clusters = max_seq_len // window_size

        # 本地注意力
        local_dim_heads = dim_head * self.local_attn_heads
        if self.local_attn_heads > 0:
            rel_pos_emb_config = (dim_head, local_attn_heads) if rel_pos_emb else None
            self.local_attn = LocalAttention(dim = dim_head, window_size = local_attn_window_size, causal = causal, dropout = attn_dropout, rel_pos_emb_config = rel_pos_emb_config, look_backward = local_attn_radius_blocks, look_forward = 0 if causal else local_attn_radius_blocks)
            self.local_to_qkv = nn.Linear(dim, 3 * local_dim_heads)

        # 全局注意力
        global_dim_heads = dim_head * self.global_attn_heads
        if self.global_attn_heads > 0:
            self.global_attn = KmeansAttention(num_clusters, window_size, self.global_attn_heads, dim_head, causal = causal, dropout = attn_dropout, ema_decay = kmeans_ema_decay, commitment = commitment_factor, receives_context = receives_context, num_mem_kv = num_mem_kv, shared_qk = shared_qk)

        self.to_q = nn.Linear(dim, global_dim_heads, bias = False)
        self.to_v = nn.Linear(dim, global_dim_heads, bias = False)

        if not self.shared_qk:
            self.to_k = nn.Linear(dim, global_dim_heads, bias = False)

        # 输出
        self.to_out = nn.Linear(dim_heads, dim, bias = False)
        self.dropout = nn.Dropout(dropout)
    # 定义前向传播函数，接受输入 x 和其他参数
    def forward(self, x, context = None, input_mask = None, context_mask = None, pos_emb = None, **kwargs):
        # 断言如果需要上下文信息但未传入，则抛出异常
        assert not (self.receives_context and not exists(context)), 'context must be passed if self attention is set to receive context'
        # 获取输入 x 的形状信息
        b, t, e, h, dh = *x.shape, self.heads, self.dim_head
        # 判断是否存在局部和全局注意力头
        has_local, has_global = map(lambda x: x > 0, (self.local_attn_heads, self.global_attn_heads))

        # 定义函数用于将输入张量按照头数进行分割
        split_heads = lambda v: reshape_dim(v, -1, (-1, dh)).transpose(1, 2).contiguous()

        # 如果存在局部注意力头
        if has_local:
            # 将局部注意力头的查询、键、值分别提取出来并按头数分割
            local_qkv = self.local_to_qkv(x).chunk(3, dim=-1)
            lq, lk, lv = map(split_heads, local_qkv)

        # 如果存在全局注意力头
        if has_global:
            # 根据是否接收上下文信息选择输入作为查询和值
            kv_input = x if not self.receives_context else context

            # 将查询和值分别转换为 Q 和 V，并按头数分割
            q, v = self.to_q(x), self.to_v(kv_input)

            # 如果不共享 Q 和 K，则将键也转换为 K，否则根据是否接收上下文信息选择使用 Q 或者 K
            if not self.shared_qk:
                k = self.to_k(kv_input)
            else:
                k = self.to_q(kv_input) if self.receives_context else q

            q, k, v = map(split_heads, (q, k, v))

        # 初始化输出列表和总损失
        out = []
        total_loss = torch.tensor(0., requires_grad=True, **to(x))

        # 如果存在局部注意力头
        if has_local:
            # 使用局部注意力计算输出
            local_out = self.local_attn(lq, lk, lv, input_mask = input_mask)
            out.append(local_out)

        # 如果存在全局注意力头
        if has_global:
            # 如果不接收上下文信息且存在位置编码，则应用位置编码
            if not self.receives_context and exists(pos_emb):
                q, k, v = apply_rotary_pos_emb(q, k, v, pos_emb)

            # 使用全局注意力计算输出和损失
            global_out, loss = self.global_attn(q, k, v, query_mask = input_mask, key_mask = context_mask)
            total_loss = total_loss + loss

            out.append(global_out)

        # 将所有输出拼接在一起
        out = torch.cat(out, dim=1)
        # 重塑输出张量的形状
        out = out.reshape(b, h, t, -1).transpose(1, 2).reshape(b, t, -1)
        # 将输出传递给输出层，并应用 dropout
        out = self.to_out(out)
        return self.dropout(out), total_loss
class RoutingTransformer(nn.Module):
    # 定义一个路由变换器类，继承自 nn.Module
    def __init__(
        self,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = None,
        window_size = 64,
        local_attn_window_size = 256,
        local_attn_radius_blocks = 1,
        causal = False,
        weight_tie = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        layer_dropout = 0.,
        n_local_attn_heads = 0,
        ff_glu = False,
        reversible = False,
        ff_chunks = 1,
        kmeans_ema_decay = 0.999,
        commitment_factor = 1e-4,
        receives_context = False,
        context_window_size = None,
        _register_kmeans_update = False,
        rel_pos_emb = True,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        moe_layers = tuple(),
        moe_num_experts = 4,
        moe_loss_coef = 1e-2,
        num_mem_kv = 0,
        shared_qk = None,
        context_shared_qk = False,
        use_rezero = False,
        use_scale_norm = False,
        ff_activation = None,
        shift_tokens = False
    # 初始化函数，设置路由变换器的各种参数
    def cancel_kmeans_update(self):
        # 取消 K-means 更新
        if not exists(self._handle):
            return
        self._handle.remove()
        self._handle = None

    def register_kmeans_update(self):
        # 注册 K-means 更新
        self._handle = update_kmeans_on_backwards(self)

    def forward(self, x, **kwargs):
        # 前向传播函数
        x, loss = self.layers(x, **kwargs)
        return x, loss

class RoutingTransformerLM(nn.Module):
    # 定义一个路由变换器语言模型类，继承自 nn.Module
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        window_size = 64,
        local_attn_window_size = None,
        local_attn_radius_blocks = 1,
        causal = False,
        emb_dim = None,
        weight_tie = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        layer_dropout = 0.,
        ff_mult = 4,
        ff_activation = None,
        ff_glu = False,
        return_embeddings = False,
        n_local_attn_heads = 0,
        reversible = False,
        ff_chunks = 1,
        kmeans_ema_decay = 0.999,
        commitment_factor = 1e-4,
        receives_context = False,
        context_window_size = None,
        rel_pos_emb = True,
        _register_kmeans_update = True,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        moe_layers = tuple(),
        moe_num_experts = 4,
        moe_loss_coef = 1e-2,
        num_mem_kv = 0,
        shared_qk = None,
        context_shared_qk = False,
        use_rezero = False,
        use_scale_norm = False,
        tie_embedding = False,
        use_absolute_pos_emb = False,
        shift_tokens = False
    # 初始化函数，设置路由变换器语言模型的各种参数
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言最大序列长度必须能被窗口大小整除，以计算 kmeans 簇的数量
        assert (max_seq_len % window_size) == 0, 'max sequence length must be divisible by the window size, to calculate number of kmeans cluster'
        # 如果未指定嵌入维度，则使用默认维度
        emb_dim = default(emb_dim, dim)

        # 初始化最大序列长度和正弦位置编码
        self.max_seq_len = max_seq_len
        self.sinu_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)

        # 初始化标记嵌入层
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        # 使用正态分布初始化权重
        nn.init.normal_(self.token_emb.weight, std = 0.02)

        # 初始化路由变换器
        self.routing_transformer = RoutingTransformer(dim, depth, max_seq_len, heads = heads, dim_head = dim_head, window_size = window_size, local_attn_window_size = local_attn_window_size, local_attn_radius_blocks = local_attn_radius_blocks, causal = causal, weight_tie = weight_tie, ff_dropout = ff_dropout, attn_dropout = attn_dropout, attn_layer_dropout = attn_layer_dropout, layer_dropout = layer_dropout, n_local_attn_heads = n_local_attn_heads, ff_glu = ff_glu, reversible = reversible, ff_chunks = ff_chunks, kmeans_ema_decay = kmeans_ema_decay, receives_context = receives_context, context_window_size = context_window_size, rel_pos_emb = rel_pos_emb, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys,  moe_layers = moe_layers, moe_num_experts = moe_num_experts, moe_loss_coef = moe_loss_coef, num_mem_kv = num_mem_kv, shared_qk = shared_qk, context_shared_qk = context_shared_qk, _register_kmeans_update = _register_kmeans_update, use_rezero = use_rezero, use_scale_norm = use_scale_norm, ff_activation = ff_activation, shift_tokens = shift_tokens)

        # 如果嵌入维度不等于维度，则使用 ProjectInOut 进行维度转换
        if emb_dim != dim:
            self.routing_transformer = ProjectInOut(self.routing_transformer, emb_dim, dim, project_out = not return_embeddings)

        # 初始化 LayerNorm 层
        self.norm = nn.LayerNorm(emb_dim)

        # 根据返回嵌入标志选择输出层
        if return_embeddings:
            self.out = nn.Identity()
        elif tie_embedding:
            self.out = MatrixMultiply(self.token_emb.weight, transpose = True)
        else:
            self.out = nn.Linear(emb_dim, num_tokens)

    # 取消 kmeans 更新
    def cancel_kmeans_update(self):
        # 找到 RoutingTransformer 模块并取消 kmeans 更新
        transformer = find_modules(self, RoutingTransformer)[0]
        transformer.cancel_kmeans_update()

    # ���新 kmeans
    def update_kmeans(self):
        # 对于所有的 Kmeans 模块，执行更新
        for m in find_modules(self, Kmeans):
            m.update()

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 对输入进行标记嵌入
        x = self.token_emb(x)

        # 计算旋转位置编码
        rotary_pos_emb = self.sinu_pos_emb(x)
        # 使用路由变换器进行前向传播
        x, loss = self.routing_transformer(x, pos_emb = rotary_pos_emb, **kwargs)

        # 对输出进行 LayerNorm
        x = self.norm(x)
        # 返回输出和损失
        return self.out(x), loss
```