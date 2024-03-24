# `.\lucidrains\memory-efficient-attention-pytorch\memory_efficient_attention_pytorch\transformer.py`

```
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 functools 库中导入 partial 函数
from functools import partial
# 从 einops 库中导入 rearrange 函数
from einops import rearrange
# 从 memory_efficient_attention_pytorch 库中导入 FlashAttention 和 Attention 类
from memory_efficient_attention_pytorch import FlashAttention, Attention
# 从 memory_efficient_attention_pytorch.reversible 库中导入 ReversibleSequence 类
from memory_efficient_attention_pytorch.reversible import ReversibleSequence

# 定义一个函数，用于检查变量是否存在
def exists(val):
    return val is not None

# 定义一个继承自 nn.Module 的类 PreNorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        # 对输入数据进行 LayerNorm 归一化
        x = self.norm(x)
        # 调用传入的函数处理归一化后的数据
        return self.fn(x, **kwargs)

# 定义一个继承自 nn.Module 的类 FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, chunks = 1):
        super().__init__()
        self.chunks = chunks

        # 定义一个包含线性层和 GELU 激活函数的神经网络
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        # 如果 chunks 小于等于 1，则直接对输入数据进行处理
        if self.chunks <= 1:
            return self.net(x)

        # 将输入数据按照指定维度进行切分
        chunks = x.chunk(self.chunks, dim = 1)
        # 对每个切分后的数据块进行处理
        out = [self.net(chunk) for chunk in chunks]
        # 将处理后的数据块拼接在一起
        return torch.cat(out, dim = 1)

# 定义一个继承自 nn.Module 的类 Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        ff_chunks = 1,
        use_flash_attn = True,
        **kwargs
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # 定义一个 token 的 Embedding 层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # ���义一个位置编码的 Embedding 层
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 根据 use_flash_attn 参数选择不同的注意力机制类
        attn_klass = FlashAttention if use_flash_attn else partial(Attention, memory_efficient = True)

        # 初始化一个空的神经网络层列表
        self.layers = nn.ModuleList([])
        # 根据深度循环创建多个层
        for _ in range(depth):
            # 每个层包含一个注意力机制和一个前馈神经网络
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn_klass(dim = dim, dim_head = dim_head, heads = heads, causal = causal, **kwargs)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, chunks = ff_chunks)),
            ]))

        # 创建一个可逆序列
        self.net = ReversibleSequence(self.layers)

        # 定义一个输出层，用于将模型输出转换为预测标签
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, labels = None):
        device = x.device
        # 对输入数据进行 token embedding
        x = self.token_emb(x)

        # 生成位置编码
        pos_emb = self.pos_emb(torch.arange(x.shape[-2], device = device))
        x = x + pos_emb

        # 通过网络层进行前向传播
        x = self.net(x)

        # 将输出数据转换为预测标签
        logits = self.to_logits(x)

        # 如果不存在标签，则直接返回预测结果
        if not exists(labels):
            return logits

        # 计算交叉熵损失
        return F.cross_entropy(rearrange(logits, 'b n d -> b d n'), labels)
```