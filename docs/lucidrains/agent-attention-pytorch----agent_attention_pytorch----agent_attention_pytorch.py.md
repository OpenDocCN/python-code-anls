# `.\lucidrains\agent-attention-pytorch\agent_attention_pytorch\agent_attention_pytorch.py`

```
# 导入 torch 库
import torch
# 从 torch.nn 模块中导入 Module 类
from torch.nn import Module
# 从 torch 模块中导入 nn、einsum、Tensor
from torch import nn, einsum, Tensor
# 从 einops 库中导入 rearrange、repeat
from einops import rearrange, repeat
# 从 einops.layers.torch 中导入 Rearrange 类

# 定义函数

# 判断变量是否存在的函数
def exists(v):
    return v is not None

# 主要类

# 自注意力机制的代理类
class AgentSelfAttention(Module):
    def __init__(
        self,
        dim,
        *,
        num_agent_tokens,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        talking_heads = True,
        gate = True,
        combine_agent_tokens = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        # 将输入转换为查询、键、值
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        # 生成门控信息
        self.to_gates = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if gate else None

        # 初始化代理令牌
        self.agent_tokens = nn.Parameter(torch.zeros(heads, num_agent_tokens, dim_head))
        nn.init.normal_(self.agent_tokens, std = 0.02)

        # 对查询和键进行对话操作
        self.qa_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.ak_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

        # 对查询和键进行 dropout 操作
        self.qa_dropout = nn.Dropout(dropout)
        self.ak_dropout = nn.Dropout(dropout)

        # 输出层
        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None,
        agent_tokens = None,
        return_agent_tokens = False
    ):
        batch = x.shape[0]

        q, k, v = self.to_qkv(x)

        if exists(agent_tokens):
            a = agent_tokens
        else:
            a = repeat(self.agent_tokens, 'h m d -> b h m d', b = batch)

        a = a * self.scale

        qa_sim = einsum('b h i d, b h j d -> b h i j', q, a)
        ak_sim = einsum('b h i d, b h j d -> b h i j', a, k)

        if exists(mask):
            max_neg_value = -torch.finfo(qa_sim.dtype).max
            ak_sim = ak_sim.masked_fill(~rearrange(mask, 'b j -> b 1 1 j'), max_neg_value)

        qa_attn = qa_sim.softmax(dim = -1)
        ak_attn = ak_sim.softmax(dim = -1)

        qa_attn = self.qa_dropout(qa_attn)
        ak_attn = self.ak_dropout(ak_attn)

        qa_attn = self.qa_talking_heads(qa_attn)
        ak_attn = self.ak_talking_heads(ak_attn)

        agent_gathered_tokens = einsum('b h i j, b h j d -> b h i d', ak_attn, v)

        out = einsum('b h i j, b h j d -> b h i d', qa_attn, agent_gathered_tokens)

        if exists(mask):
            out = out.masked_fill(~rearrange(mask, 'b n -> b 1 n 1'), 0.)

        if exists(self.to_gates):
            out = out * self.to_gates(x)

        out = self.to_out(out)

        if not return_agent_tokens:
            return out

        return out, agent_gathered_tokens
```