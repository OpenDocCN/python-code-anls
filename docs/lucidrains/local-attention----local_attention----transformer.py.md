# `.\lucidrains\local-attention\local_attention\transformer.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 从 local_attention 包中导入 LocalAttention 类
from local_attention.local_attention import LocalAttention

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 评估装饰器函数
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 采样函数

# 返回 logits 中大于阈值的前 k 个值
def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# 多头注意力机制

class LocalMHA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        prenorm = False,
        qk_rmsnorm = False,
        qk_scale = 8,
        use_xpos = False,
        xpos_scale_base = None,
        exact_windowsize = None,
        gate_values_per_head = False,
        **kwargs
    ):
        super().__init__()        
        inner_dim = dim_head * heads

        # 如果 prenorm 为 True，则使用 LayerNorm 进行归一化
        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        # 将输入映射到查询、键、值空间
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 使用 LocalAttention 进行局部注意力计算
        self.attn_fn = LocalAttention(
            dim = dim_head,
            window_size = window_size,
            causal = causal,
            autopad = True,
            scale = (qk_scale if qk_rmsnorm else None),
            exact_windowsize = default(exact_windowsize, True),
            use_xpos = use_xpos,
            xpos_scale_base = xpos_scale_base,
            **kwargs
        )

        self.to_v_gate = None

        if gate_values_per_head:
            self.to_v_gate = nn.Sequential(
                nn.Linear(dim, heads)
            )

        # 将输出映射回原始维度
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None, attn_bias = None):
        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) 

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        out = self.attn_fn(q, k, v, mask = mask, attn_bias = attn_bias)

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            gates = rearrange(gates, 'b n h -> b h n 1')
            out = out * gates.sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 前馈网络

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

# 创建前馈网络
def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# 动态位置偏置

class DynamicPositionBias(nn.Module):
    def __init__(
        self,
        dim,
        heads
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, heads)
        )

    @property
    def device(self):
        return next(self.parameters()).device
    # 定义一个前向传播函数，接受输入参数 i 和 j
    def forward(self, i, j):
        # 获取设备信息
        device = self.device
        # 断言 j 大于等于 i
        assert j >= i

        # 创建一个相对距离张量，从 i 到 j，数据类型为浮点数，使用指定设备
        rel_dist = torch.arange(j, dtype=torch.float, device=device)
        # 使用 MLP 模型处理重新排列后的相对距离张量，得到偏置
        bias = self.mlp(rearrange(rel_dist, '... -> ... 1'))

        # 创建从 i 到 j-1 的序列张量，使用指定设备
        i_seq = torch.arange(j - i, j, device=device)
        # 创建从 0 到 j-1 的序列张量，使用指定设备
        j_seq = torch.arange(j, device=device)

        # 计算相对距离的索引，取绝对值
        rel_dist_indices = (rearrange(i_seq, 'i -> i 1') - rearrange(j_seq, 'j -> 1 j')).abs()

        # 重新排列偏置张量，根据相对距离索引，维度顺序为 h i j
        bias = rearrange(bias[rel_dist_indices], 'i j h -> h i j')
        # 返回处理后的偏置张量
        return bias
# 主要的转换器类

class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,  # 标记的数量
        max_seq_len,  # 最大序列长度
        dim,  # 维度
        depth,  # 深度
        causal = True,  # 是否使用因果注意力
        local_attn_window_size = 512,  # 本地注意力窗口大小
        dim_head = 64,  # 头部维度
        heads = 8,  # 头部数量
        ff_mult = 4,  # FeedForward 层的倍数
        attn_dropout = 0.,  # 注意力层的丢弃率
        ff_dropout = 0.,  # FeedForward 层的丢弃率
        ignore_index = -1,  # 忽略的索引
        use_xpos = False,  # 是否使用位置编码
        xpos_scale_base = None,  # 位置编码的缩放基数
        use_dynamic_pos_bias = False,  # 是否使用动态位置偏置
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)  # 标记嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, dim)  # 位置嵌入层

        self.max_seq_len = max_seq_len  # 最大序列长度
        self.layers = nn.ModuleList([])  # 层列表

        self.local_attn_window_size = local_attn_window_size  # 本地注意力窗口大小
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)  # 动态位置偏置

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalMHA(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = causal, window_size = local_attn_window_size, use_xpos = use_xpos, xpos_scale_base = xpos_scale_base, use_rotary_pos_emb = not use_dynamic_pos_bias, prenorm = True, **kwargs),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))  # 添加多层局部多头注意力和前馈网络

        self.ignore_index = ignore_index  # 忽略的索引
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_tokens, bias = False)  # 线性层
        )

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prime,  # 初始序列
        seq_len,  # 生成序列的长度
        temperature = 1.,  # 温度参数
        filter_thres = 0.9,  # 过滤阈值
        **kwargs
    ):
        n, device = prime.shape[1], prime.device

        out = prime

        for _ in range(seq_len):
            logits = self.forward(out[:, -self.max_seq_len:], **kwargs)  # 前向传播获取 logits
            filtered_logits = top_k(logits[:, -1], thres = filter_thres)  # 获取 top-k logits
            probs = F.softmax(filtered_logits / temperature, dim = -1)  # softmax 计算概率
            sampled = torch.multinomial(probs, 1)  # 多项式采样
            out = torch.cat((out, sampled), dim = -1)  # 将采样结果拼接到输出序列

        return out[:, n:]  # 返回生成的序列

    def forward(self, x, mask = None, return_loss = False):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]  # 获取输入和标签序列

        n, device = x.shape[1], x.device
        x = self.token_emb(x)  # 标记嵌入

        assert n <= self.max_seq_len
        x = x + self.pos_emb(torch.arange(n, device = device))  # 添加位置编码

        # 动态位置偏置

        attn_bias = None
        if exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)  # 计算注意力偏置

        # 通过层

        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_bias = attn_bias) + x  # 多头注意力层
            x = ff(x) + x  # 前馈网络

        logits = self.to_logits(x)  # 线性层得到 logits

        if not return_loss:
            return logits

        logits = rearrange(logits, 'b n c -> b c n')  # 重新排列 logits
        loss = F.cross_entropy(logits, labels, ignore_index = self.ignore_index)  # 计算交叉熵损失
        return loss  # 返回损失
```