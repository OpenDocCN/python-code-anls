# `.\lucidrains\feedback-transformer-pytorch\feedback_transformer_pytorch\feedback_transformer_pytorch.py`

```py
# 导入数学库
import math
# 导入命名元组
from collections import namedtuple

# 导入 PyTorch 库
import torch
# 导入神经网络模块、矩阵乘法函数
from torch import nn, einsum
# 导入 PyTorch 中的函数库
import torch.nn.functional as F
# 从 einops 库中导入重新排列函数
from einops import rearrange

# 定义命名元组 Memory，包含 keys 和 values 两个字段
Memory = namedtuple('Memory', ['keys', 'values'])

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 安全地拼接张量
def safe_cat(arr, el, dim = 1):
    if not exists(arr):
        return el
    return torch.cat((arr, el), dim = dim)

# 位置嵌入

# 定义相对位置偏置类
class RelativePositionBias(nn.Module):
    def __init__(
        self,
        causal = False,
        num_buckets = 32,
        max_distance = 128,
        heads = 8
    ):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    # 静态方法，计算相对位置的桶索引
    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    # 前向传播函数
    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return bias

# 辅助类

# 残差连接类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 预层归一化类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 如果条件成立则跳过的类
class SkipIf(nn.Module):
    def __init__(self, cond, fn):
        super().__init__()
        self.cond = cond
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        if self.cond(x, *args, **kwargs):
            return x
        return self.fn(x, *args, **kwargs)

# 前馈网络

# GEGLU 激活函数类
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

# 前馈网络类
class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# 注意力机制

# 注意力类
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)
    # 定义前向传播函数，接受输入 x、记忆 memory 和位置编码 pos_emb
    def forward(self, x, memory, pos_emb = None):
        # 获取头数 h、序列长度 n 和设备信息
        h, n, device = self.heads, x.shape[1], x.device

        # 判断是否进行自注意力计算，只有在大于1个标记时才进行自注意力计算
        self_attend = n > 1 

        # 将输入 x 转换为查询向量 q，并乘以缩放因子
        q = self.to_q(x) * self.scale

        # 解包记忆 memory 中的键 k 和值 v，如果不存在则设为 None
        k, v = memory if exists(memory) else (None, None)

        # 如果需要进行自注意力计算
        if self_attend:
            # 将输入 x 转换为键 k 和值 v
            self_k, self_v = self.to_kv(x).chunk(2, dim = -1)
            # 将自注意力计算得到的键 k 和值 v 与原有的键 k 和值 v 进行拼接
            k = safe_cat(k, self_k, dim = 1)
            v = safe_cat(v, self_v, dim = 1)

        # 将查询 q、键 k 和值 v 重排维度，以适应多头注意力计算
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 计算注意力分数矩阵 sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        i, j = sim.shape[-2:]

        # 如果存在位置编码 pos_emb，则加上位置编码
        if exists(pos_emb):
            sim = sim + pos_emb(sim)

        # 如果需要进行自注意力计算
        if self_attend:
            # 生成因果掩码，用于屏蔽未来信息
            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            causal_mask = rearrange(causal_mask, 'i j -> () () i j')
            mask_value = -torch.finfo(q.dtype).max
            sim.masked_fill_(causal_mask, mask_value)

        # 对注意力分数矩阵进行 softmax 操作
        attn = sim.softmax(dim = -1)
        # 对注意力分数矩阵应用 dropout
        attn = self.dropout(attn)

        # 计算加权后的值向量 out
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # 重排维度，恢复原始形状
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出结果传递给输出层
        return self.to_out(out)
# 主类定义

class FeedbackTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,  # 标记数量
        dim,  # 维度
        depth,  # 深度
        mem_len,  # 记忆长度
        seq_len = 2,  # 序列长度，默认为2
        heads = 8,  # 头数
        dim_head = 64,  # 头维度
        attn_dropout = 0.,  # 注意力机制的dropout
        ff_dropout = 0.,  # 前馈网络的dropout
        keep_last_hidden = False  # 是否保留最后一个隐藏层
    ):
        super().__init__()
        self.seq_len = seq_len
        self.mem_len = mem_len

        self.token_emb = nn.Embedding(num_tokens, dim)  # 标记嵌入层
        self.pos_emb = RelativePositionBias(causal = True, heads = heads)  # 相对位置偏置

        # 主要层

        self.layers = nn.ModuleList([])
        shared_kv_proj = None

        for _ in range(depth):
            attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)  # 注意力机制
            ff = FeedForward(dim = dim, dropout = ff_dropout)  # 前馈网络

            shared_kv_proj = default(shared_kv_proj, attn.to_kv)  # 共享键值投影
            attn.to_kv = shared_kv_proj

            attn, ff = map(lambda fn: Residual(PreNorm(dim, fn)), (attn, ff))  # 添加残差连接和层归一化

            if seq_len == 1:
                memory_is_empty = lambda *args, **kwargs: not exists(kwargs['memory'])
                attn = SkipIf(memory_is_empty, attn)  # 如果记忆为空，则跳过

            self.layers.append(nn.ModuleList([
                attn,
                ff
            ]))

        # 记忆参数

        self.layer_weight = nn.Parameter(torch.ones(depth + 1))  # 层权重
        self.shared_kv_proj = shared_kv_proj
        self.keep_last_hidden = keep_last_hidden

        # 最终投影到logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_tokens)  # 线性层
        )

    def forward(self, x, memory = None, return_memory = False):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)  # 标记嵌入

        memory_keys = None
        memory_values = None

        if exists(memory):
            memory_keys, memory_values = memory

        outputs = []

        # 计算层的权重以存储到记忆中

        layer_weight = self.layer_weight.softmax(dim = -1)
        layer_weight = rearrange(layer_weight, 'd -> d () () ()')

        for x in x.split(self.seq_len, dim = 1):
            hiddens = [x]

            # 准备用于注意力的记忆，如果存在

            memory = None
            if exists(memory_keys):
                memory = (memory_keys, memory_values)

            for attn, ff in self.layers:

                x = attn(x, memory = memory, pos_emb = self.pos_emb)  # 注意力机制
                x = ff(x)  # 前馈网络

                hiddens.append(x)

            outputs.append(x)

            # 计算新的记忆键/值并存储到FIFO队列

            if self.keep_last_hidden:  # 保留最后一个隐藏层
                agg_hiddens = hiddens[-1]
            else:
                hiddens = torch.stack(hiddens)
                agg_hiddens = (hiddens * layer_weight).sum(dim = 0)

            # 预先计算记忆键/值并存储到缓冲区

            mem_k, mem_v = self.shared_kv_proj(agg_hiddens).chunk(2, dim = -1)
            memory_keys = safe_cat(memory_keys, mem_k, dim = 1)
            memory_values = safe_cat(memory_values, mem_v, dim = 1)

            # 强制在记忆缓冲区上施加最大长度限制

            memory_keys = memory_keys[:, -self.mem_len:]
            memory_values = memory_values[:, -self.mem_len:]

        x = torch.cat((outputs), dim = 1)
        out = self.to_logits(x)

        if not return_memory:
            return out

        return out, Memory(memory_keys, memory_values)
```