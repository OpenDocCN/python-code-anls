# `.\lucidrains\fast-transformer-pytorch\fast_transformer_pytorch\fast_transformer_pytorch.py`

```py
import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch import nn, einsum  # 从 PyTorch 中导入 nn 和 einsum 模块

from einops import rearrange, reduce  # 从 einops 库中导入 rearrange 和 reduce 函数
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding  # 从 rotary_embedding_torch 库中导入 apply_rotary_emb 和 RotaryEmbedding 类

# helper functions

def exists(val):
    return val is not None  # 判断值是否为 None 的辅助函数

def default(val, d):
    return val if exists(val) else d  # 如果值存在则返回该值，否则返回默认值的辅助函数

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 对输入进行 LayerNorm 归一化
        self.fn = fn  # 传入的函数

    def forward(self, x, **kwargs):
        x = self.norm(x)  # 对输入进行归一化
        return self.fn(x, **kwargs)  # 调用传入的函数处理归一化后的输入

# blocks

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),  # 线性变换
        nn.GELU(),  # GELU 激活函数
        nn.Linear(dim * mult, dim)  # 线性变换
    )

class FastAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64,
        max_seq_len = None,
        pos_emb = None
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 线性变换将输入转换为查询、键、值

        # rotary positional embedding

        assert not (exists(pos_emb) and not exists(max_seq_len)), 'max_seq_len must be passed in if to use rotary positional embeddings'  # 断言语句，确保条件成立

        self.pos_emb = pos_emb  # 位置编码
        self.max_seq_len = max_seq_len  # 最大序列长度

        # if using relative positional encoding, make sure to reduce pairs of consecutive feature dimension before doing projection to attention logits

        kv_attn_proj_divisor = 1 if not exists(pos_emb) else 2  # 如果使用相对位置编码，则将连续特征维度减少一半再进行注意力机制的投影

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias = False)  # 用于将查询投影到查询注意力得分的线性变换
        self.to_k_attn_logits = nn.Linear(dim_head // kv_attn_proj_divisor, 1, bias = False)  # 用于将键��影到键注意力得分的线性变换

        # final transformation of values to "r" as in the paper

        self.to_r = nn.Linear(dim_head // kv_attn_proj_divisor, dim_head)  # 将值最终转换为 "r"，与论文中描述的一致

        self.to_out = nn.Linear(inner_dim, dim)  # 最终输出的线性变换
    # 定义前向传播函数，接受输入张量 x 和可选的 mask 参数
    def forward(self, x, mask = None):
        # 获取输入张量 x 的形状信息
        n, device, h, use_rotary_emb = x.shape[1], x.device, self.heads, exists(self.pos_emb)

        # 将输入张量 x 经过线性变换得到 qkv，并按照通道数分割为 q、k、v
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # 初始化 mask_value 为 x 数据类型的最小值
        mask_value = -torch.finfo(x.dtype).max
        # 将 mask 重排为 'b () n' 形状
        mask = rearrange(mask, 'b n -> b () n')

        # 如果需要使用相对位置编码
        if use_rotary_emb:
            # 获取位置编码频率信息
            freqs = self.pos_emb(torch.arange(self.max_seq_len, device = device), cache_key = self.max_seq_len)
            freqs = rearrange(freqs[:n], 'n d -> () () n d')
            # 对 q、k、v 应用旋转编码
            q_aggr, k_aggr, v_aggr = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))
        else:
            q_aggr, k_aggr, v_aggr = q, k, v

        # 计算查询注意力 logits
        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = q_attn_logits.softmax(dim = -1)

        # 计算全局查询 token
        global_q = einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # 用全局查询 token 偏置键
        k = k * global_q

        # 如果使用旋转编码，对特征维度中相邻对进行内积
        if use_rotary_emb:
            k = reduce(k, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # 计算键注意力 logits
        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = k_attn_logits.softmax(dim = -1)

        # 计算全局键 token
        global_k = einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        # 偏置值
        u = v_aggr * global_k

        # 如果使用旋转编码，对特征维度中相邻对进行内积
        if use_rotary_emb:
            u = reduce(u, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # 转换步骤
        r = self.to_r(u)

        # 论文中指出将查询作为残差添加
        r = r + q

        # 合并头部
        r = rearrange(r, 'b h n d -> b n (h d)')
        # 返回输出结果
        return self.to_out(r)
# 主类 FastTransformer
class FastTransformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        num_tokens,  # 标记数量
        dim,  # 维度
        depth,  # 深度
        max_seq_len,  # 最大序列长度
        heads = 8,  # 头数
        dim_head = 64,  # 头的维度
        ff_mult = 4,  # FeedForward 的倍数
        absolute_pos_emb = False  # 是否使用绝对位置编码
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)  # 标记嵌入层

        # 位置编码
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if absolute_pos_emb else None

        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (dim_head % 4) == 0, 'dimension of the head must be divisible by 4 to use rotary embeddings'
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        # 层
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = FastAttention(dim, dim_head = dim_head, heads = heads, pos_emb = layer_pos_emb, max_seq_len = max_seq_len)  # 快速注意力机制
            ff = FeedForward(dim, mult = ff_mult)  # 前馈网络

            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),  # 预归一化
                PreNorm(dim, ff)  # 预归一化
            ]))

        # 在所有层之间进行权重绑定投影
        first_block, _ = self.layers[0]
        for block, _ in self.layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        # 转换为 logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, num_tokens)  # 线性层
        )

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)  # 标记嵌入

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> () n d')  # 重排位置编码

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x  # 注意力机制
            x = ff(x) + x  # 前馈网络

        return self.to_logits(x)  # 返回 logits
```