# `.\lucidrains\product-key-memory\product_key_memory\product_key_memory.py`

```
# 导入 math、torch 库
import math
import torch
# 从 torch 库中导入 nn、einsum 模块
from torch import nn, einsum
# 从 einops 库中导入 rearrange、Rearrange、Reduce 函数
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
# 从 colt5_attention.py 文件中导入 topk 函数作为 coor_descent_topk

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 计算输入张量的自然对数，避免输入值小于给定的最小值
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 初始化函数

# 初始化张量，均值为 0，标准差为 1/sqrt(dim)
def init_(t, dim = None):
    dim = default(dim, t.shape[-1])
    std = 1. / math.sqrt(dim)
    return nn.init.normal_(t, mean=0, std=std)

# 优化器函数

# 从列表 l 中减去列表 r 中的元素
def list_subtract(l, r):
    return [el for el in l if el not in set(r)]

# 获取 PKM 模块中的值参数
def fetch_pkm_value_parameters(module):
    params = []
    for m in module.modules():
        if isinstance(m, PKM):
            params.append(m.values.weight)
    rest = list_subtract(module.parameters(), params)
    return params, rest

# 获取优化器参数
def fetch_optimizer_parameters(module, pkm_learning_rate = 1e-2):
    pkm_params, rest = fetch_pkm_value_parameters(module)
    return [{'params': rest}, {'params': pkm_params, 'lr': pkm_learning_rate}]

# 归一化函数

# 一维掩码批归一化类
class MaskedBatchNorm1D(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(
        self,
        x,
        mask = None
    ):
        if exists(mask):
            initial_x = x
            x = x[mask]

        x = self.fn(x)

        if exists(mask):
            initial_x[mask] = x
            x = initial_x

        return x

# PKM 模块
class PKM(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        num_keys = 128,
        topk = 32,
        dim_head = 128,
        input_dropout = 0.,
        query_dropout = 0.,
        value_dropout = 0.,
        attn_dropout = 0.,
        use_layernorm = True,
        pre_layernorm = False,
        differentiable_topk = False,
        concat_values_and_combine = False,
        norm_output = False
    ):
        super().__init__()
        self.topk = topk
        self.heads = heads
        self.num_keys = num_keys

        dim_query = dim_head * heads * 2
        self.to_queries = nn.Linear(dim, dim_query, bias = False)

        # 预层归一化模式

        self.pre_layernorm = nn.LayerNorm(dim) if pre_layernorm else nn.Identity()

        # 批归一化会破坏因果性

        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm = nn.LayerNorm(dim_head)
        else:
            self.norm = MaskedBatchNorm1D(nn.BatchNorm1d(dim_head))

        # 键

        self.keys = nn.Parameter(torch.zeros(heads, num_keys, 2, dim_head))
        init_(self.keys)

        # 值

        self.concat_values_and_combine = concat_values_and_combine

        if concat_values_and_combine:
            values = nn.Embedding(num_keys ** 2, dim_head)

            self.values = nn.Sequential(
                values,
                Reduce('b (h k) d -> b h d', 'sum', h = heads),
                Rearrange('b n d -> b (n d)'),
                nn.Linear(dim_head * heads, dim, bias = False)
            )
        else:
            values = nn.EmbeddingBag(num_keys ** 2, dim, mode = 'sum')
            self.values = values

        init_(values.weight)

        # 丢弃

        self.input_dropout = nn.Dropout(input_dropout)
        self.query_dropout = nn.Dropout(query_dropout)
        self.value_dropout = nn.Dropout(value_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # 使用可微分的 topk，基于坐标下降

        self.differentiable_topk = differentiable_topk

        # https://arxiv.org/abs/2302.06461
        # 声称通过简单地对输出进行 layernorm 来提高 softmax 键/值网络的性能

        self.output_norm = nn.LayerNorm(dim) if norm_output else nn.Identity()

    def forward(
        self,
        x,
        input_mask = None,
        gumbel_noise_scale = 0.,
        **kwargs
        ):
        # 解构 x 的形状，分别赋值给 b, t, h
        b, t, h = *x.shape[:2], self.heads

        # 对输入进行预层归一化
        x = self.pre_layernorm(x)
        # 对输入进行输入层的 dropout
        x = self.input_dropout(x)

        # 将输入转换为查询
        queries = self.to_queries(x)

        # 分割查询头

        queries = rearrange(queries, 'b t (p h d) -> (b p h) t d', p = 2, h = h)

        # 对查询进行归一化和 dropout

        norm_kwargs = dict(mask = input_mask) if not self.use_layernorm else dict()
        queries = self.norm(queries, **norm_kwargs)
        queries = self.query_dropout(queries)

        # 准备查询

        queries = rearrange(queries, '(b p h) t d -> p b t h d', p = 2, h = h)

        # 与键计算相似度

        dots = einsum('p b t h d, h n p d -> b t h p n', queries, self.keys)

        # gumbel 噪声

        if gumbel_noise_scale > 0.:
            dots = dots + gumbel_noise(dots) * gumbel_noise_scale

        # topk 分数

        if self.differentiable_topk:
            scores, indices, *_ = coor_descent_topk(dots, k = self.topk, fused = True)
        else:
            scores, indices = dots.topk(k = self.topk, dim = -1)

        # 分数进行因式分解

        (scores_x, scores_y), (indices_x, indices_y) = map(lambda t: t.chunk(2, dim = 3), (scores, indices))

        all_topk = self.topk ** 2

        all_scores = rearrange((
            rearrange(scores_x, '... k -> ... k 1') +
            rearrange(scores_y, '... k -> ... 1 k')
        ), 'b t h ... -> b t h (...)')

        all_indices = rearrange((
            rearrange(indices_x, '... k -> ... k 1') * self.num_keys +
            rearrange(indices_y, '... k -> ... 1 k')
        ), 'b t h ... -> b t h (...)')

        final_topk, final_indices = all_scores.topk(self.topk, dim=-1)
        value_indices = all_indices.gather(-1, final_indices)

        # 注意力

        attn = final_topk.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        value_indices, attn = map(lambda t: rearrange(t, 'b t h k -> (b t) (h k)'), (value_indices, attn))

        # 聚合

        if self.concat_values_and_combine:
            out = self.values(value_indices)
        else:
            out = self.values(value_indices, per_sample_weights = attn)

        out = self.value_dropout(out)

        # 可能对输出进行层归一化

        out = self.output_norm(out)

        return rearrange(out, '(b t) d -> b t d', b = b)
```