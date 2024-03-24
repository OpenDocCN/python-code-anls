# `.\lucidrains\coordinate-descent-attention\coordinate_descent_attention\coordinate_descent_attention.py`

```py
# 导入 torch 库
import torch
# 导入 torch.nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 torch 库中导入 nn 和 einsum 模块
from torch import nn, einsum
# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 colt5_attention 模块中导入 coor_descent 和 triton_coor_descent 函数

from colt5_attention import coor_descent
from colt5_attention.triton_coor_descent import triton_coor_descent

# helpers

# 定义函数 exists，判断变量是否存在
def exists(val):
    return val is not None

# 定义函数 default，如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# classes

# 定义类 FeedForward，继承自 nn.Module 类
class FeedForward(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        mult = 4,
        use_coor_descent = False,
        coor_descent_iters = 20,
        coor_descent_sparsity_k = None,
        coor_descent_eps = 1e-1,
        coor_descent_eps_init = 4.,
        coor_descent_eps_decay = 0.7,
    ):
        super().__init__()

        dim_hidden = int(dim * mult)

        self.use_coor_descent = use_coor_descent

        self.coor_descent_iters = coor_descent_iters
        self.coor_descent_sparsity_k = default(coor_descent_sparsity_k, dim_hidden // 10)
        self.coor_descent_eps = coor_descent_eps
        self.coor_descent_eps_init = coor_descent_eps_init
        self.coor_descent_eps_decay = coor_descent_eps_decay

        self.proj_in = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_hidden),
        )

        self.proj_out = nn.Linear(dim_hidden, dim)

    # 前向传播函数
    def forward(self, x):
        x = self.proj_in(x)

        if self.use_coor_descent:
            x = triton_coor_descent(
                x,
                n_iters = self.coor_descent_iters,
                k = self.coor_descent_sparsity_k,
                eps = self.coor_descent_eps,
                eps_init = self.coor_descent_eps_init,
                eps_decay = eslf.coor_descent_eps_decay,
                checkpoint_segments = self.coor_descent_iters // 5
            )
        else:
            x = F.gelu(x)

        return self.proj_out(x)

# 定义类 Attention，继承自 nn.Module 类
class Attention(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        use_coor_descent = False,
        coor_descent_iters = 20,
        coor_descent_sparsity_k = 1,
        coor_descent_eps = 1e-1,
        coor_descent_eps_init = 4.,
        coor_descent_eps_decay = 0.7,
        attn_null_kv = 0,
        learned_sparsity_k = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.use_coor_descent = use_coor_descent

        self.coor_descent_iters = coor_descent_iters
        self.coor_descent_sparsity_k = coor_descent_sparsity_k

        self.coor_descent_eps = coor_descent_eps
        self.coor_descent_eps_init = coor_descent_eps_init
        self.coor_descent_eps_decay = coor_descent_eps_decay

        self.to_learned_k = None
        if learned_sparsity_k:
            self.to_learned_k = nn.Linear(dim, heads)
            nn.init.constant_(self.to_learned_k.bias, -10)

        self.norm = nn.LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, attn_null_kv, dim_head))

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)
    # 定义前向传播函数，接受输入张量 x
    def forward(self, x):
        # 解构 x 的形状，获取批大小 b，序列长度 n，头数 h，设备信息 device，数据类型 dtype
        b, n, h, device, dtype = *x.shape[:2], self.heads, x.device, x.dtype
        # 对输入 x 进行归一化处理
        x = self.norm(x)

        # 获取查询（q）、键（k）、值（v），并将它们按头数拆分

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 如果需要添加空键值对

        if self.null_kv.numel() > 0:
            nk, nv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.null_kv)
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        # 计算相似度

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

        # 是否使用坐标下降

        if self.use_coor_descent:

            if exists(self.to_learned_k):
                sparsity_k = self.to_learned_k(x).sigmoid() * (self.coor_descent_sparsity_k - 1) + 1
                sparsity_k = rearrange(sparsity_k, 'b i h -> (b h i)')
            else:
                sparsity_k = torch.ones(i, device = device, dtype = dtype) * self.coor_descent_sparsity_k

            causal_mask = repeat(causal_mask, 'i j -> b h i j', b = sim.shape[0], h = sim.shape[1])

            attn = triton_coor_descent(
                sim,
                n_iters = self.coor_descent_iters,
                k = sparsity_k,
                eps = self.coor_descent_eps,
                eps_decay = self.coor_descent_eps_decay,
                eps_init = self.coor_descent_eps_init,
                mask = ~causal_mask,
                checkpoint_segments = self.coor_descent_iters // 5
            )

        else:
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
            attn = sim.softmax(dim = -1)

        # 聚合

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并头部

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# 定义 Transformer 类，继承自 nn.Module
class Transformer(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        num_tokens,  # 标记的数量
        dim,  # 向量维度
        seq_len,  # 序列长度
        depth,  # 层数
        dim_head = 64,  # 注意力头的维度
        heads = 8,  # 注意力头的数量
        ff_mult = 4,  # FeedForward 层的倍数
        attn_use_coor_descent = False,  # 是否使用坐标下降优化注意力
        ff_use_coor_descent = False,  # 是否使用坐标下降优化 FeedForward
        attn_coor_descent_sparsity_k = 2,  # 注意力坐标下降的稀疏度参数
        ff_coor_descent_sparsity_k = 2,  # FeedForward 坐标下降的稀疏度参数
        coor_descent_iters = 15,  # 坐标下降的迭代次数
        coor_descent_eps = 1e-1,  # 坐标下降的收敛阈值
        attn_null_kv = 0,  # 注意力的 null key 和 value
        learned_sparsity_k = False  # 是否学习稀疏度参数
    ):
        super().__init__()
        self.seq_len = seq_len  # 保存序列长度

        # 创建标记嵌入层和位置嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)

        self.layers = nn.ModuleList([])  # 初始化层列表

        # 定义坐标下降参数字典
        coor_kwargs = dict(
            coor_descent_iters = coor_descent_iters,
            coor_descent_eps = coor_descent_eps,
        )

        # 根据层数循环创建多个注意力和 FeedForward 层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim,
                    dim_head = dim_head,
                    heads = heads,
                    use_coor_descent = attn_use_coor_descent,
                    coor_descent_sparsity_k = attn_coor_descent_sparsity_k,
                    attn_null_kv = attn_null_kv,
                    learned_sparsity_k = learned_sparsity_k,
                    **coor_kwargs
                ),
                FeedForward(
                    dim,
                    ff_mult,
                    use_coor_descent = ff_use_coor_descent,
                    coor_descent_sparsity_k = ff_coor_descent_sparsity_k,
                    **coor_kwargs
                )
            ]))

        # 定义输出层，包括 LayerNorm 和线性层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    # 前向传播函数
    def forward(self, x):
        n, device = x.shape[-1], x.device
        assert n <= self.seq_len  # 断言序列长度不超过设定的最大长度

        x = self.token_emb(x)  # 对输入进行标记嵌入
        x = x + self.pos_emb(torch.arange(n, device = device))  # 加上位置嵌入

        # 遍历每个注意力和 FeedForward 层，进行前向传播
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.to_logits(x)  # 返回最终输出
```