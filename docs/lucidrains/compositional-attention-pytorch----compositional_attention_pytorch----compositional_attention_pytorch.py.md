# `.\lucidrains\compositional-attention-pytorch\compositional_attention_pytorch\compositional_attention_pytorch.py`

```py
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from einops_exts import rearrange_many

# 检查变量是否存在的函数
def exists(val):
    return val is not None

# 计算稳定的 softmax 函数
def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True).detach()
    return t.softmax(dim = dim)

# 组合注意力机制类
class CompositionalAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        num_searches = 8,
        num_retrievals = 2,
        dropout = 0.,
        prenorm = False,
        causal = False
    ):
        super().__init__()
        # 根据 prenorm 参数选择是否使用 LayerNorm 或 Identity
        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()

        self.scale = dim_head ** -0.5
        inner_search_dim = dim_head * num_searches
        inner_retrieval_dim = dim_head * num_retrievals

        self.num_searches = num_searches
        self.num_retrievals = num_retrievals

        # 线性变换层，将输入映射到搜索查询和键
        self.to_searches_queries = nn.Linear(dim, inner_search_dim, bias = False)
        self.to_searches_keys = nn.Linear(dim, inner_search_dim, bias = False)
        self.to_retrieval_values = nn.Linear(dim, inner_retrieval_dim, bias = False)

        # 线性变换层，将输入映射到检索查询和键
        self.to_retrieval_queries = nn.Linear(dim, inner_search_dim, bias = False)
        self.to_retrieval_keys = nn.Linear(dim_head, dim_head, bias = False)

        # 线性变换层，将检索结果映射回输出维度
        self.to_out = nn.Linear(inner_search_dim, dim, bias = False)

        self.search_dropout = nn.Dropout(dropout)
        self.retrieval_dropout = nn.Dropout(dropout)

        # 是否使用自回归变体进行自我实验
        self.causal = causal

    def forward(self, x, mask = None):
        """
        einstein notation:
        b - batch
        n - sequence dimension
        i - sequence dimension (source)
        j - sequence dimension (target, aggregation dimension)
        s - number of searches
        r - number of retrievals
        d - feature dimension
        """
        x = self.norm(x)

        s = self.num_searches
        r = self.num_retrievals

        # 获取搜索查询和键
        sq, sk = self.to_searches_queries(x), self.to_searches_keys(x)
        sq, sk = rearrange_many((sq, sk), 'b n (s d) -> b s n d', s = s)

        sq = sq * self.scale

        # 计算搜索相似度和注意力
        search_sim = einsum('b s i d, b s j d -> b s i j', sq, sk)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            search_sim = search_sim.masked_fill(~mask, -torch.finfo(search_sim.dtype).max)

        if self.causal:
            i, j = search_sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
            search_sim = search_sim.masked_fill(causal_mask, -torch.finfo(search_sim.dtype).max)

        search_attn = stable_softmax(search_sim, dim = -1)
        search_attn = self.search_dropout(search_attn)

        # 获取检索值
        rv = self.to_retrieval_values(x)
        rv = rearrange(rv, 'b n (r d) -> b r n d', r = r)

        retrieved = einsum('b s i j, b r j d -> b s r i d', search_attn, rv)

        # 获取检索查询和键
        rq, rk = self.to_retrieval_queries(x), self.to_retrieval_keys(retrieved)
        rq = rearrange(rq, 'b n (s d) -> b s n d', s = s)
        rq = rq * self.scale

        # 获取检索注意力
        retrieval_sim = einsum('b s n d , b s r n d -> b s n r', rq, rk)

        retrieval_attn = stable_softmax(retrieval_sim, dim = -1)
        retrieval_attn = self.retrieval_dropout(retrieval_attn)

        # 聚合检索结果
        out = einsum('b s n r, b s r n d -> b s n d', retrieval_attn, retrieved)

        # 组合搜索结果
        out = rearrange(out, 'b s n d -> b n (s d)')
        return self.to_out(out)
```