# `.\lucidrains\sinkhorn-transformer\sinkhorn_transformer\sinkhorn_transformer.py`

```
# 导入 math 模块
import math
# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 子模块
from torch import nn
# 从 operator 模块中导入 mul 函数
from operator import mul
# 从 math 模块中导入 gcd 函数
from math import gcd
# 从 torch.nn.functional 模块中导入 F 别名
import torch.nn.functional as F
# 从 inspect 模块中导入 isfunction 函数
from inspect import isfunction
# 从 functools 模块中导入 partial, wraps, reduce 函数
from functools import partial, wraps, reduce

# 导入自定义模块
from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from product_key_memory import PKM
from sinkhorn_transformer.reversible import ReversibleSequence, SequentialSequence

# 辅助函数

# 定义返回输入的函数
def identity(x, *args, **kwargs): return x

# 定义返回默认值的函数
def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

# 将输入转换为元组的函数
def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)

# 判断一个数是否能被另一个数整除的函数
def divisible_by(num, divisor):
    return num % divisor == 0

# 计算多个数的最小公倍数的函数
def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1)

# 判断多个元素是否都为 None 的函数
def all_none(*arr):
    return all(el is None for el in arr)

# 缓存函数的装饰器
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

# 将张量向左旋转的函数
def rotate_left(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(n, None))
    r = (*pre_slices, slice(0, n))
    return torch.cat((t[l], t[r]), dim=dim)

# 将张量向右旋转的函数
def rotate_right(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(-n, None))
    r = (*pre_slices, slice(None, -n))
    return torch.cat((t[l], t[r]), dim=dim)

# 合并张量的维度的函数
def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

# 合并张量的头部的函数
def merge_heads(h, v):
    b, t, d = v.shape
    return v.view(b, t, h, -1).transpose(1, 2).reshape(b, h, t, -1)

# 分割张量的头部的函数
def split_heads(h, v):
    *_, t, d = v.shape
    return v.view(-1, h, t, d).transpose(1, 2).reshape(-1, t, d * h)

# 在指定索引处分割张量的函数
def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# 将张量分桶的函数
def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [buckets, -1]
    return t.reshape(*shape)

# 将分桶后的张量还原的函数
def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)

# 采样 Gumbel 分布的函数
def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

# Sinkhorn 排序算子的函数
def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)

# Gumbel Sinkhorn 函数
def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)

# 重新排序分桶后的张量的函数
def reorder_buckets(t, r):
    return torch.einsum('buv,bvtd->butd', r, t)

# 对张量取对数的函数
def log(t, eps = 1e-6):
    return torch.log(t + eps)

# 获取张量最大负值的函数
def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# 沿指定维度计算累积平均值的函数
def cumavg(t, dim):
    r = torch.arange(1, t.shape[dim] + 1, device=t.device, dtype=t.dtype)
    expand_slice = [None] * len(t.shape)
    expand_slice[dim] = slice(None, None)
    return t.cumsum(dim=dim) / r[tuple(expand_slice)]

# 批量索引选择的函数
def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

# 在指定维度扩展张量的函数
def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 扩展批次并合并头部的函数
def expand_batch_and_merge_head(b, t):
    shape = list(t.squeeze(0).shape)
    t = expand_dim(t, 0, b)
    shape[0] = shape[0] * b
    return t.reshape(*shape)

# 可微分的 Top-K 函数
def differentiable_topk(x, k, temperature=1.):
    *_, n, dim = x.shape
    topk_tensors = []
    # 遍历 k 次，每次生成一个 topk tensor
    for i in range(k):
        # 判断是否是最后一次循环
        is_last = i == (k - 1)
        # 对输入 x 进行 softmax 操作，然后取最大的值和对应的索引
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        # 根据索引和值生成一个新的 tensor，并替换原来的值
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        # 将生成的 topk tensor 添加到列表中
        topk_tensors.append(topks)
        # 如果不是最后一次循环，则将对应索引的值设为负无穷
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    # 将所有生成的 topk tensor 拼接在一起
    topks = torch.cat(topk_tensors, dim=-1)
    # 将拼接后的 tensor 重新 reshape 成指定形状
    return topks.reshape(*_, k * n, dim)
# 定义一个名为 Chunk 的类，继承自 nn.Module
class Chunk(nn.Module):
    # 初始化方法，接受参数 chunks、fn 和 along_dim，默认值为 -1
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        # 设置对象属性
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    # 前向传播方法
    def forward(self, x):
        # 将输入 x 按照指定维度分块
        chunks = x.chunk(self.chunks, dim = self.dim)
        # 对每个分块应用函数 fn，并在指定维度上拼接结果
        return torch.cat([self.fn(c) for c in chunks], dim = self.dim)

# 定义一个名为 GELU_ 的类，继承自 nn.Module
class GELU_(nn.Module):
    # 前向传播方法
    def forward(self, x):
        # 计算 GELU 激活函数的值
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))

# 如果 nn 模块中存在 GELU 类，则使用 nn.GELU，否则使用 GELU_
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

# 定义一个名为 FeedForward 的类，继承自 nn.Module
class FeedForward(nn.Module):
    # 初始化方法，接受参数 dim、mult、dropout、activation 和 glu，默认值为 False
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        # 设置对象属性
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    # 前向传播方法
    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# 定义一个名为 ReZero 的类，继承自 nn.Module
class ReZero(nn.Module):
    # 初始化方法，接受参数 fn
    def __init__(self, fn):
        super().__init__()
        # 定义可学习参数 g
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    # 前向传播方法
    def forward(self, x, **kwargs):
        # 返回 fn 函数的结果乘以可学习参数 g
        return self.fn(x, **kwargs) * self.g

# 定义一个名为 PreNorm 的类，继承自 nn.Module
class PreNorm(nn.Module):
    # 初始化方法，接受参数 norm_class、dim 和 fn
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        # 实例化规范化层对象
        self.norm = norm_class(dim)
        self.fn = fn

    # 前向传播方法
    def forward(self, x, **kwargs):
        # 对输入 x 进行规范化
        x = self.norm(x)
        # 返回 fn 函数的结果
        return self.fn(x, **kwargs)

# 定义一个名为 ProjectInOut 的类，继承自 nn.Module
class ProjectInOut(nn.Module):
    # 初始化方法，接受参数 fn、dim_in、dim_out 和 project_out，默认为 True
    def __init__(self, fn, dim_in, dim_out, project_out = True):
        super().__init__()
        # 设置对象属性
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    # 前向传播方法
    def forward(self, x, **kwargs):
        # 对输入 x 进行投影
        x = self.project_in(x)
        # 对投影后的结果应用函数 fn
        x = self.fn(x, **kwargs)
        # 对结果进行逆投影
        x = self.project_out(x)
        return x

# 定义一个名为 SimpleSortNet 的类，继承自 nn.Module
class SimpleSortNet(nn.Module):
    # 初始化方法，接受参数 heads、bucket_size、max_buckets、dim、non_permutative、temperature 和 sinkhorn_iter
    def __init__(self, heads, bucket_size, max_buckets, dim, non_permutative, temperature, sinkhorn_iter):
        super().__init__()
        # 设置对象属性
        self.dim = dim
        self.heads = heads
        self.max_buckets = max_buckets
        self.bucket_size = bucket_size
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.linear = nn.Parameter(torch.randn(1, heads, dim, max_buckets))
        self.act = nn.ReLU()

    # 前向传播方法
    def forward(self, q, k, topk=1):
        bh, t, _ = q.shape
        b = bh // self.heads
        buckets = t // self.bucket_size

        b_q, b_k = bucket(buckets, q), bucket(buckets, k)
        x = torch.cat((b_q.sum(dim=2), b_k.sum(dim=2)), dim=-1)

        W = expand_batch_and_merge_head(b, self.linear)
        R = self.act(x @ W)

        return differentiable_topk(R, k=topk, temperature=self.temperature) if self.non_permutative else gumbel_sinkhorn(R, self.sinkhorn_iter, self.temperature)

# 定义一个名为 AttentionSortNet 的类，继承自 nn.Module
class AttentionSortNet(nn.Module):
    # 初始化方法，接受参数 heads、bucket_size、kv_bucket_size、dim、non_permutative、temperature、sinkhorn_iter 和 n_sortcut，默认为 0
    def __init__(self, heads, bucket_size, kv_bucket_size, dim, non_permutative, temperature, sinkhorn_iter, n_sortcut = 0):
        super().__init__()
        # 设置对象属性
        self.heads = heads
        self.bucket_size = bucket_size
        self.kv_bucket_size = kv_bucket_size
        self.dim = dim
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut
    # 定义一个前向传播函数，接受查询向量 q、键向量 k 和 topk 参数，默认为 1
    def forward(self, q, k, topk=1):
        # 解构赋值，获取查询向量 q 的形状信息
        bh, *_, bucket_size, kv_bucket_size, device, dtype, dim = *q.shape, self.bucket_size, self.kv_bucket_size, q.device, q.dtype, self.dim
        # 计算每个头部的批次大小
        b = bh // self.heads

        # 计算查询向量 q 的桶数
        buckets = q.shape[1] // bucket_size
        # 计算键向量 k 的桶数
        kv_buckets = k.shape[1] // kv_bucket_size

        # 将查询向量 q 分桶，如果 n_sortcut 为 0 则只有一个桶，否则按照桶大小分桶
        b_q = bucket(buckets, q) if self.n_sortcut == 0 else bucket(1, q)
        # 将键向量 k 分桶
        b_k = bucket(kv_buckets, k)

        # 计算查询向量 q 的均值
        sq = b_q.mean(dim=2)
        # 计算键向量 k 的均值
        sk = b_k.mean(dim=2)

        # 计算 R 矩阵，使用 einsum 函数计算点积并乘以缩放因子
        R = torch.einsum('bie,bje->bij', sq, sk).to(q) * (dim ** -0.5)

        # 如果是非排列不变的注意力机制
        if self.non_permutative:
            # 如果 n_sortcut 为 0，则返回前 k 个最大值，否则返回前 n_sortcut 个最大值
            k = topk if self.n_sortcut == 0 else self.n_sortcut
            return differentiable_topk(R, k=k)

        # 如果是排列不变的注意力机制，则使用 Gumbel Sinkhorn 进行计算
        return gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)
# 定义 SinkhornAttention 类，继承自 nn.Module
class SinkhornAttention(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(self, bucket_size, dim, dim_heads, heads, max_seq_len, temperature = 0.75, non_permutative = True, sinkhorn_iter = 7, n_sortcut = 0, dropout = 0., kv_bucket_size = None, use_simple_sort_net = False, n_top_buckets = 1):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化各个参数
        self.bucket_size = bucket_size
        # 如果 kv_bucket_size 为 None，则使用 bucket_size
        self.kv_bucket_size = default(kv_bucket_size, bucket_size)

        self.dim = dim
        self.heads = heads
        self.temperature = temperature
        self.non_permutative = non_permutative
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

        # 根据 use_simple_sort_net 的值选择不同的排序网络
        if use_simple_sort_net:
            self.sort_net = SimpleSortNet(heads, self.kv_bucket_size, max_seq_len // self.kv_bucket_size, dim_heads * 2, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter)
        else:
            self.sort_net = AttentionSortNet(heads, self.bucket_size, self.kv_bucket_size, dim_heads, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut)

        self.n_top_buckets = n_top_buckets
        # 初始化一个 dropout 层
        self.dropout = nn.Dropout(dropout)
    # 定义前向传播函数，接受查询、键、值以及查询和键值的掩码作为输入
    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        # 解包变量，获取批次大小、头数、序列长度、隐藏维度、前n个桶、维度、头数、温度、桶大小、键值桶大小、设备
        b, h, t, d_h, n_top, d, heads, temperature, bucket_size, kv_bucket_size, device = *q.shape, self.n_top_buckets, self.dim, self.heads, self.temperature, self.bucket_size, self.kv_bucket_size, q.device

        # 计算批次头数
        bh = b * h
        # 计算查询的桶数和键值的桶数
        buckets = q.shape[2] // bucket_size
        kv_buckets = k.shape[2] // kv_bucket_size
        # 确保前n个桶不超过键值桶数
        n_top = min(n_top, kv_buckets)

        # 合并批次和头维度
        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v))

        # 桶化查询、键、值
        b_q = bucket(buckets, q)
        b_k, b_v = map(partial(bucket, kv_buckets), (k, v))

        bsz = b_k.shape[2]

        # 使用简单排序网络计算重新排序矩阵R
        R = self.sort_net(q, k, topk=n_top)
        R = R.type_as(q).to(q)

        # 拼接重新排序后的桶
        b_k_r = reorder_buckets(b_k, R)
        b_v_r = reorder_buckets(b_v, R)

        # 选择前n个排名的桶作为所有查询桶的顶部n个桶
        if self.n_sortcut > 0:
            b_k_r = b_k_r[:, 0:self.n_sortcut].reshape(bh, 1, -1, d_h)
            b_v_r = b_v_r[:, 0:self.n_sortcut].reshape(bh, 1, -1, d_h)
            b_k_r = expand_dim(b_k_r, 1, buckets)
            b_v_r = expand_dim(b_v_r, 1, buckets)
        else:
            b_k_r = b_k_r.reshape(bh, buckets, -1, d_h)
            b_v_r = b_k_r.reshape(bh, buckets, -1, d_h)

        # 拼接查询桶和键值桶
        b_k = torch.cat((b_k_r, b_k), dim=2) if buckets == kv_buckets else b_k_r
        b_v = torch.cat((b_v_r, b_v), dim=2) if buckets == kv_buckets else b_v_r

        # 计算点积
        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d_h ** -0.5)

        # 掩码
        mask_value = max_neg_value(dots)

        # 如果查询和键值掩码不全为空
        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)
            mq, mk = bucket(buckets, q_mask), bucket(kv_buckets, kv_mask)
            expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, h))
            mq, mk = map(expand_head_and_merge_into_batch, (mq, mk))

            mk_r = batched_index_select(mk, R.abs().argmax(dim=-1))

            if self.n_sortcut > 0:
                mk_r = mk_r[:, 0:self.n_sortcut].reshape(-1, 1, bsz * self.n_sortcut)
                mk_r = expand_dim(mk_r, 1, buckets)
            else:
                mk_r = mk_r.reshape(bh, buckets, -1)

            mk = torch.cat((mk_r, mk), dim=2) if buckets == kv_buckets else mk_r
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask            

        # 注意力
        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        return out
# 定义函数，生成一个掩码矩阵，用于重新排序
def mask_reordering_matrix(R, topk, temperature):
    # 获取矩阵的列数，即桶的数量
    buckets = R.shape[1]

    # 获取矩阵中的最大值，用于生成掩码
    mask_value = max_neg_value(R)
    # 创建一个与 R 相同形状的全零张量，用于存储掩码
    mask = torch.zeros(R.shape, device=R.device).bool()
    # 获取上三角矩阵的索引
    i, j = torch.triu_indices(buckets, buckets)
    # 将掩码应用到 R 上，将指定位置的值替换为 mask_value
    mask[:, i, j + topk] = True

    # 使用掩码将 R 中的值替换为 mask_value
    R.masked_fill_(mask, mask_value)
    # 返回经过不同iable_topk 函数处理后的结果
    return differentiable_topk(R, topk, temperature)

# 定义一个简单的排序网络模型
class CausalSimpleSortNet(nn.Module):
    def __init__(self, heads, bucket_size, max_buckets, n_top_buckets, dim, temperature):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.bucket_size = bucket_size
        self.max_buckets = max_buckets
        self.n_top_buckets = n_top_buckets
        self.temperature = temperature
        # 初始化线性层参数
        self.linear = nn.Parameter(torch.randn(1, heads, dim, max_buckets + n_top_buckets))
        # 初始化激活函数
        self.act = nn.LeakyReLU()

    # 前向传播函数
    def forward(self, q, k, topk=1):
        # 获取张量的维度信息
        bh, *_, h, max_buckets = *q.shape, self.heads, self.max_buckets
        b = bh // h
        # 计算桶的数量
        buckets = k.shape[1] // self.bucket_size

        # 对 k 进行处理，将其累积平均后进行桶化
        k_r = torch.cat((cumavg(k, dim=1), k), dim=-1)
        k_r = bucket(buckets, k_r)

        # 对于因果排序网络，取每个桶的第一个标记以防止未来信息泄漏到过去
        x = k_r[:, :, 0]

        # 扩展线性层参数并合并头部
        W = expand_batch_and_merge_head(b, self.linear)
        R = self.act(x @ W)
        R = R[:, 0:buckets, 0:(buckets + self.n_top_buckets)]

        # 返回经过 mask_reordering_matrix 函数处理后的结果
        return mask_reordering_matrix(R, topk, self.temperature)

# 定义一个因果注意力排序网络模型
class CausalAttentionSortNet(nn.Module):
    def __init__(self, heads, bucket_size, dim, temperature):
        super().__init__()
        self.heads = heads
        self.bucket_size = bucket_size
        self.dim = dim
        self.temperature = temperature

    # 前向传播函数
    def forward(self, q, k, topk=1):
        bh, *_, h, dim = *q.shape, self.heads, self.dim

        b = bh // h
        buckets = q.shape[1] // self.bucket_size
        kv_buckets = k.shape[1] // self.bucket_size

        q_r = bucket(buckets, cumavg(q, dim=1))
        k_r = bucket(kv_buckets, cumavg(k, dim=1))

        sq = q_r[:, :, 0]
        sk = k_r.sum(dim=2)
        sk = F.pad(sk, (0, 0, topk, 0))

        R = torch.einsum('bie,bje->bij', sq, sk) * (dim ** -0.5)
        # 返回经过 mask_reordering_matrix 函数处理后的结果
        return mask_reordering_matrix(R, topk, self.temperature)

# 在指定索引处对张量进行分割，并对分割后的部分应用函数后再拼接
def apply_fn_after_split_ind(dim, ind, fn, t):
    l, r = split_at_index(dim, ind, t)
    return torch.cat((l, fn(r)), dim=dim)

# 定义 Sinkhorn 因果注意力模型
class SinkhornCausalAttention(nn.Module):
    def __init__(self, bucket_size, dim, dim_heads, heads, max_seq_len, dropout = 0., kv_bucket_size = None, use_simple_sort_net = False, n_top_buckets = 2, temperature = 1.):
        super().__init__()
        assert kv_bucket_size is None or bucket_size == kv_bucket_size, 'different bucketing for key/values for causal reordering not supported yet'

        self.dim = dim
        self.heads = heads
        self.bucket_size = bucket_size

        # 用于第一个桶的学习到的空键/值（过去没有内容需要排序）
        self.null_keys = nn.Parameter(torch.randn(heads, 1, dim_heads))
        self.null_values = nn.Parameter(torch.randn(heads, 1, dim_heads))

        # 根据 use_simple_sort_net 参数选择不同的排序网络模型
        if use_simple_sort_net:
            self.sort_net = CausalSimpleSortNet(heads, bucket_size, max_seq_len // bucket_size, n_top_buckets, dim_heads * 2, temperature)
        else:
            self.sort_net = CausalAttentionSortNet(heads, bucket_size, dim_heads, temperature)

        self.n_top_buckets = n_top_buckets
        self.dropout = nn.Dropout(dropout)
    # 定义一个前向传播函数，接受查询(q)、键(k)、值(v)以及查询掩码(q_mask)和键值掩码(kv_mask)作为输入参数
    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        # 获取输入张量的形状信息
        b, h, t, d_h, n_top, d, bsz, device = *q.shape, self.n_top_buckets, self.dim, self.bucket_size, q.device

        # 计算一些常用的值
        bh = b * h
        hh = h // 2
        buckets = t // bsz
        n_top = min(n_top, buckets)

        # 定义一个切片，用于获取后半部分的头信息
        hh_slice = (slice(None), slice(hh, None))

        # 定义一个部分函数，用于对输入张量进行旋转操作
        rotate_fn = partial(apply_fn_after_split_ind, 1, hh, lambda t: rotate_left(t, bsz-1, dim=2))
        q, k, v = map(rotate_fn, (q, k, v))

        # 合并批次和头信息
        merge_batch_head = partial(merge_dims, 0, 1)
        q, k, v = map(merge_batch_head, (q, k, v))

        # 对查询、键、值进行分桶操作
        b_q, b_k, b_v = map(partial(bucket, buckets), (q, k, v))

        # 计算排序矩阵R
        R = self.sort_net(q, k, topk=n_top)
        R = R.type_as(q).to(q)

        # 添加空键/值
        b_null_k = self.null_keys[None, :, None, :, :].expand(b, h, n_top, bsz, -1).reshape(bh, n_top, bsz, -1).to(k)
        b_null_v = self.null_values[None, :, None, :, :].expand(b, h, n_top, bsz, -1).reshape(bh, n_top, bsz, -1).to(v)

        b_k_r = torch.cat((b_null_k, b_k), dim=1)
        b_v_r = torch.cat((b_null_v, b_v), dim=1)

        # 重新排序桶以便进行本地注意力计算
        b_k_r = reorder_buckets(b_k_r, R)
        b_v_r = reorder_buckets(b_v_r, R)

        b_k_r = b_k_r.reshape(bh, buckets, bsz * n_top, -1)
        b_v_r = b_v_r.reshape(bh, buckets, bsz * n_top, -1)

        # 将原始桶本身连接到重新排序的桶中
        b_k = torch.cat((b_k_r, b_k), dim=2)
        b_v = torch.cat((b_v_r, b_v), dim=2)

        # 计算点积
        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d_h ** -0.5)

        # 定义掩码值
        mask_value = max_neg_value(q)

        # 如果存在查询掩码和键值掩码，则进行掩码操作
        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)

            expand_head = lambda x: x.unsqueeze(1).repeat(1, h, 1)
            q_mask, kv_mask = map(expand_head, (q_mask, kv_mask))

            q_mask[hh_slice] = rotate_left(q_mask[hh_slice], bsz-1, dim=2)
            kv_mask[hh_slice] = rotate_left(kv_mask[hh_slice], bsz-1, dim=2)

            q_mask, kv_mask = map(lambda x: merge_dims(0, 1, x), (q_mask, kv_mask))
            mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask)

            mk_with_null = F.pad(mk, (0, 0, 2, 0), value=True)
            mk_r = batched_index_select(mk_with_null, R.abs().argmax(dim=-1))

            mk_r = mk_r.reshape(bh, buckets, -1)
            mk = torch.cat((mk_r, mk), dim=2)
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # 为半头旋转进行掩码操作
        shift = n_top * bsz
        total_shift = shift + bsz

        mask = torch.ones((b, h, buckets, bsz, total_shift), device=device).bool()
        i, j = torch.triu_indices(bsz, bsz, 1)
        mask[:, :, :, i, j + shift] = False
        mask[:, hh:, -1, 0:shift, 0:shift+1] = False
        mask[:, hh:, -1, 0, 0:shift+1] = True
        mask = mask.reshape(b * h, buckets, bsz, total_shift)

        dots.masked_fill_(~mask, mask_value)
        del mask

        # 注意力计算
        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        out = apply_fn_after_split_ind(1, hh, lambda t: rotate_right(t, bsz-1, dim=2), out)
        return out
# 定义 SinkhornSelfAttention 类，继承自 nn.Module
class SinkhornSelfAttention(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(self, dim, bucket_size, max_seq_len, heads = 8, dim_head = None, kv_bucket_size = None, causal = False, non_permutative = True, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, attn_dropout = 0., dropout = 0., context_only = False, use_simple_sort_net = False, n_local_attn_heads = 0, n_top_buckets = 1):
        # 调用父类的初始化函数
        super().__init__()
        # 断言确保 dim_head 不为空或者 dim 可以被 heads 整除
        assert dim_head or divisible_by(dim, heads), f'If dim_head is None, dimension {dim} must be divisible by the number of heads {heads}'
        # 断言确保 sortcut 只能用于非因果注意力
        assert not (causal and n_sortcut > 0), 'sortcut can only be used for non causal attention'
        # 断言确保 context_only 自注意力层不能是因果的
        assert not (causal and context_only), 'context only self attention layer cannot be causal'
        # 断言确保本地注意力头数不超过总头数
        assert n_local_attn_heads <= heads, 'number of local attention heads cannot exceed total heads'

        # 如果 dim_head 为空，则设置为 dim 除以 heads
        dim_head = default(dim_head, dim // heads)
        # 计算 dim_heads
        dim_heads = dim_head * heads
        self.dim_head = dim_head

        self.heads = heads
        self.bucket_size = bucket_size
        self.kv_bucket_size = default(kv_bucket_size, bucket_size)

        self.context_only = context_only
        # 将输入转换为查询向量
        self.to_q = nn.Linear(dim, dim_heads, bias=False)
        # 如果不是仅上下文自注意力，则将输入转换为键值对
        self.to_kv = nn.Linear(dim, dim_heads * 2, bias=False) if not context_only else None

        # 将输出转换为线性层
        self.to_out = nn.Linear(dim_heads, dim)

        self.n_local_attn_heads = n_local_attn_heads
        # 创建本地注意力对象
        self.local_attention = LocalAttention(bucket_size, causal, dropout = attn_dropout, look_forward=(1 if not causal else 0))

        # 计算 Sinkhorn 注意力头数
        sink_heads = heads - n_local_attn_heads

        # 如果是因果的，则创建 SinkhornCausalAttention 对象，否则创建 SinkhornAttention 对象
        if causal:
            attn = SinkhornCausalAttention(bucket_size, dim, dim_head, sink_heads, max_seq_len, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets, temperature = temperature)
        else:
            attn = SinkhornAttention(bucket_size, dim, dim_head, sink_heads, max_seq_len, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, dropout = attn_dropout, kv_bucket_size = kv_bucket_size, use_simple_sort_net = use_simple_sort_net, n_top_buckets = n_top_buckets)

        # 设置 Sinkhorn 注意力对象
        self.sinkhorn_attention = attn

        # 创建丢弃层
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数，接受输入 x、输入掩码 input_mask、上下文 context 和上下文掩码 context_mask
    def forward(self, x, input_mask = None, context = None, context_mask = None):
        # 获取输入 x 的形状信息
        b, t, d, h, dh, l_h = *x.shape, self.heads, self.dim_head, self.n_local_attn_heads
        # 断言确保序列 t 可以被 bucket_size 整除
        assert divisible_by(t, self.bucket_size), f'sequence {t} needs to be divisible by bucket size {self.bucket_size}'
        # 断言确保如果是仅上下文自注意力，则必须提供上下文键/值
        assert not (self.context_only and context is None), 'context key / values must be supplied if context self attention layer'
        # 断言确保如果提供上下文，则上下文的批次和维度与解码器相同
        assert not (context is not None and (context.shape[0], context.shape[2]) !=  (b, d)), 'contextual key / values must have the same batch and dimensions as the decoder'

        # 将输入转换为查询向量
        q = self.to_q(x)

        # 如果不是仅上下文自注意力，则将输入转换为键值对，并根据维度切分
        kv = self.to_kv(x).chunk(2, dim=-1) if not self.context_only else (context, context)
        kv_mask = input_mask if not self.context_only else context_mask

        # 断言确保键/值序列可以被键/值 bucket_size 整除
        assert divisible_by(kv[0].shape[1], self.kv_bucket_size), 'key/value sequences need to be divisible by key/value bucket size'

        # 将查询向量和键值对合并
        qkv = (q, *kv)
        merge_heads_fn = partial(merge_heads, h)
        q, k, v = map(merge_heads_fn, qkv)

        # 部分函数，用于在特定索引处切分张量
        split_index_fn = partial(split_at_index, 1, l_h)
        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
        # 检查是否存在本地和 Sinkhorn 注意力
        has_local, has_sinkhorn = map(lambda x: x.shape[1] > 0, (lq, q))

        out = []

        # 如果存在本地注意力，则将结果添加到输出列表中
        if has_local > 0:
            out.append(self.local_attention(lq, lk, lv, input_mask = input_mask))

        # 如果存在 Sinkhorn 注意力，则将结果添加到输出列表中
        if has_sinkhorn > 0:
            out.append(self.sinkhorn_attention(q, k, v, q_mask = input_mask, kv_mask = kv_mask))

        # 在指定维度上连接输出列表中的张量
        out = torch.cat(out, dim=1)
        # 将输出张量按头数拆分
        out = split_heads(h, out)
        # 将输出转换为指定维度
        out = self.to_out(out)
        # 应用丢弃层
        out = self.dropout(out)
        return out

# 定义 SinkhornTransformer 类，继承自 nn.Module
class SinkhornTransformer(nn.Module):
    # 初始化函数，设置模型的各种参数
    def __init__(self, dim, depth, max_seq_len = None, causal = False, heads = 8, dim_head = None, bucket_size = 64, kv_bucket_size = None, context_bucket_size = None, non_permutative = False, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, reversible = False, ff_chunks = 1, ff_dropout = 0., attn_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., weight_tie = False, ff_glu = False, use_simple_sort_net = None, receives_context = False, context_n_sortcut = 2, n_local_attn_heads = 0, use_rezero = False, n_top_buckets = 1,  pkm_layers = tuple(), pkm_num_keys = 128):
        # 调用父类的初始化函数
        super().__init__()
        # 创建空的模型层列表
        layers = nn.ModuleList([])

        # 设置默认的 kv_bucket_size 和 context_bucket_size
        kv_bucket_size = default(kv_bucket_size, bucket_size)
        context_bucket_size = default(context_bucket_size, bucket_size)

        # 定义获取注意力层、前馈层和 PKM 层的 lambda 函数
        get_attn = lambda: SinkhornSelfAttention(dim, bucket_size, max_seq_len, causal = causal, heads = heads, dim_head = dim_head, kv_bucket_size = kv_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, use_simple_sort_net = use_simple_sort_net, n_local_attn_heads = n_local_attn_heads, n_top_buckets = n_top_buckets)
        get_ff = lambda: Chunk(ff_chunks, FeedForward(dim, dropout = ff_dropout, glu = ff_glu), along_dim=1)
        get_pkm = lambda: PKM(dim, num_keys = pkm_num_keys)

        # 定义获取上下文注意力层和上下文前馈层的 lambda 函数
        get_attn_context = lambda: SinkhornSelfAttention(dim, bucket_size, max_seq_len, context_only = True, heads = heads, dim_head = dim_head, kv_bucket_size = context_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = context_n_sortcut, temperature = temperature, attn_dropout = attn_dropout, dropout = attn_layer_dropout, n_top_buckets = n_top_buckets)
        get_ff_context = lambda: FeedForward(dim, dropout = ff_dropout, glu = ff_glu)

        # 如果权重共享为真，则缓存获取注意力层和前馈层的函数
        if weight_tie:
            get_attn, get_attn_context, get_ff, get_ff_context = map(cache_fn, (get_attn, get_attn_context, get_ff, get_ff_context))

        # 根据是否使用 PKM 层，选择获取并行函数
        for ind in range(depth):
            layer_num = ind + 1
            use_pkm = layer_num in pkm_layers

            get_parallel_fn = get_ff if not use_pkm else get_pkm

            # 将注意力层和并行函数添加到模型层列表中
            layers.append(nn.ModuleList([
                fn_wrapper(get_attn()),
                fn_wrapper(get_parallel_fn())
            ]))

            # 如果不接收上下文，则继续下一个循环
            if not receives_context:
                continue

            # 将上下文注意力层和上下文前馈层添加到模型层列表中
            layers.append(nn.ModuleList([
                fn_wrapper(get_attn_context()),
                fn_wrapper(get_ff_context())
            ]))

        # 根据是否可逆选择执行类型
        execute_type = ReversibleSequence if reversible else SequentialSequence

        # 设置上下文路由和注意力路由
        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *attn_context_layer) * depth
        route_context = ((False, False), *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn}

        # 创建模型层序列，设置参数路由和层丢弃率
        self.layers = execute_type(layers, args_route = {**context_route_map, **attn_route_map}, layer_dropout = layer_dropout)
        self.receives_context = receives_context

        # 设置最大序列长度、填充到桶大小、上下文桶大小和是否固定长度
        self.max_seq_len = max_seq_len
        self.pad_to_bucket_size = lcm(bucket_size, kv_bucket_size)
        self.context_bucket_size = context_bucket_size
        self.is_fixed_length = use_simple_sort_net and not causal

        # 如果不使用注意力排序且不是因果的，强制固定序列长度
        assert not (self.is_fixed_length and self.max_seq_len is None), 'maximum sequence length must be specified if length is fixed'
    # 定义一个前向传播函数，接受输入 x 和其他关键字参数
    def forward(self, x, **kwargs):
        # 如果模型要求输入是固定长度的序列，并且输入 x 的长度不等于最大序列长度，则抛出断言错误
        assert not (self.is_fixed_length and x.shape[1] != self.max_seq_len), f'you must supply a sequence of length {self.max_seq_len}'
        # 如果关键字参数中包含 'context' 且模型接收上下文信息，则通过，否则抛出断言错误
        assert ('context' not in kwargs or self.receives_context), 'needs to be initted with receives_context True if passing contextual key / values'
        # 调用模型的 layers 方法进行前向传播，并返回结果
        return self.layers(x, **kwargs)
class SinkhornTransformerLM(nn.Module):
    # 定义 SinkhornTransformerLM 类，继承自 nn.Module
    def __init__(self, num_tokens, dim, max_seq_len, depth, heads = 8, dim_head = None, bucket_size = 64, kv_bucket_size = None, context_bucket_size = None, causal = False, non_permutative = True, sinkhorn_iter = 5, n_sortcut = 0, temperature = 0.75, reversible = False, ff_chunks = 1, ff_glu = False, return_embeddings = False, ff_dropout = 0., attn_dropout = 0., attn_layer_dropout = 0., layer_dropout = 0., emb_dropout = 0., weight_tie = False, emb_dim = None, use_simple_sort_net = None, receives_context = False, context_n_sortcut = 0, n_local_attn_heads = 0, use_rezero = False, n_top_buckets = 2, pkm_layers = tuple(), pkm_num_keys = 128):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数

        emb_dim = default(emb_dim, dim)
        # 如果 emb_dim 为 None，则使用 dim

        self.max_seq_len = max_seq_len
        # 设置最大序列长度

        self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
        # 创建一个嵌入层，将输入的 token 映射为嵌入向量
        self.axial_pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape = (max_seq_len // bucket_size, bucket_size))
        # 创建轴向位置编码层
        self.emb_dropout = nn.Dropout(emb_dropout)
        # 创建一个丢弃层，用于嵌入向量的丢弃

        self.sinkhorn_transformer = SinkhornTransformer(dim, depth, max_seq_len = max_seq_len, causal = causal, heads = heads, dim_head = dim_head, bucket_size = bucket_size, kv_bucket_size = kv_bucket_size, context_bucket_size = context_bucket_size, non_permutative = non_permutative, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut, temperature = temperature, reversible = reversible, ff_chunks = ff_chunks, ff_dropout = ff_dropout, attn_dropout = attn_dropout, attn_layer_dropout = attn_layer_dropout, layer_dropout = layer_dropout, weight_tie = weight_tie, ff_glu = ff_glu, use_simple_sort_net = use_simple_sort_net, receives_context = receives_context, context_n_sortcut = context_n_sortcut, n_local_attn_heads = n_local_attn_heads, use_rezero = use_rezero, n_top_buckets = n_top_buckets,  pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys)
        # 创建 SinkhornTransformer 模型

        if emb_dim != dim:
            # 如果嵌入维度不等于 dim
            self.sinkhorn_transformer = ProjectInOut(self.sinkhorn_transformer, emb_dim, dim, project_out =(not return_embeddings))
            # 使用 ProjectInOut 对象将嵌入维度转换为 dim

        self.norm = nn.LayerNorm(emb_dim)
        # 创建一个 LayerNorm 层，用于归一化
        self.to_logits = identity if return_embeddings else nn.Linear(emb_dim, num_tokens)
        # 如果 return_embeddings 为真，则使用 identity 函数，否则使用线性层将嵌入向量映射为输出 logits

    def forward(self, x, **kwargs):
        # 前向传播函数，接受输入 x 和关键字参数 kwargs
        _, t, device = *x.shape, x.device
        # 获取输入 x 的形状和设备信息
        assert t <= self.max_seq_len, f'sequence length {t} is greater than maximum sequence length {self.max_seq_len}'
        # 断言序列长度不超过最大序列长度

        x = self.to_token_emb(x)
        # 将输入 x 映射为嵌入向量
        x = self.axial_pos_emb(x) + x
        # 添加轴向位置编码到嵌入向量上
        x = self.emb_dropout(x)
        # 对嵌入向量进行丢弃
        x = self.sinkhorn_transformer(x, **kwargs)
        # 使用 SinkhornTransformer 处理嵌入向量
        x = self.norm(x)
        # 对处理后的向量进行归一化
        return self.to_logits(x)
        # 返回最终的 logits
```