# `.\lucidrains\PaLM-pytorch\palm_pytorch\palm_lite.py`

```py
# 导入 torch 库
import torch
# 导入 torch.nn.functional 模块
import torch.nn.functional as F
# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 torch 库中导入 einsum 和 nn 模块
from torch import einsum, nn
# 从 math 库中导入 log2 和 floor 函数
from math import log2, floor

# 定义函数，判断变量是否存在
def exists(val):
    return val is not None

# normalization

# 定义 RMSNorm 类，继承自 nn.Module
class RMSNorm(nn.Module):
    # 初始化函数
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        # 初始化缩放因子
        self.scale = dim ** -0.5
        # 初始化 eps
        self.eps = eps
        # 创建可学习参数 g
        self.g = nn.Parameter(torch.ones(dim))

    # 前向传播函数
    def forward(self, x):
        # 计算输入张量 x 的 L2 范数
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        # 返回归一化后的结果
        return x / norm.clamp(min = self.eps) * self.g

# AliBi

# 定义 AlibiPositionalBias 类，继承自 nn.Module
class AlibiPositionalBias(nn.Module):
    # 初始化函数
    def __init__(self, heads, **kwargs):
        super().__init__()
        # 初始化头数
        self.heads = heads
        # 计算斜率
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        # 注册缓冲区 slopes 和 bias
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    # 获取偏置
    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    # 静态方法，获取斜率
    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** floor(log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    # 前向传播函数
    def forward(self, qk_sim):
        h, i, j, device = *qk_sim.shape[-3:], qk_sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)

        return bias

# residual

# 定义 Residual 类，继承自 nn.Module
class Residual(nn.Module):
    # 初始化函数
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    # 前向传播函数
    def forward(self, x):
        return self.fn(x) + x

# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202

# 定义 SwiGLU 类，继承自 nn.Module
class SwiGLU(nn.Module):
    # 前向传播函数
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

# 定义 ParallelTransformerBlock 类，继承自 nn.Module
class ParallelTransformerBlock(nn.Module):
    # 初始化函数
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        # 初始化 RMSNorm 层
        self.norm = RMSNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5

        # 初始化 AlibiPositionalBias 层
        self.alibi_pos_biases = AlibiPositionalBias(heads = self.heads)

        # 初始化线性变换层
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask

        self.register_buffer("mask", None, persistent=False)

    # 获取掩码
    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.triu(torch.ones((n, n), device=device, dtype=torch.bool), 1)
        self.register_buffer("mask", mask, persistent=False)
        return mask
    # 定义前向传播函数，接受输入张量 x
    def forward(self, x):

        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        # 获取输入张量 x 的形状信息
        n, device, h = x.shape[1], x.device, self.heads

        # 对输入张量 x 进行预层归一化处理
        x = self.norm(x)

        # 获取注意力查询、键或值（共享键/值是我个人的发现）和前馈内部
        q, kv, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # 分割头部
        # 他们使用多查询单键值注意力，又一篇 Noam Shazeer 的论文
        # 他们发现在一定规模之后没有性能损失，而且解码更有效
        # https://arxiv.org/abs/1911.02150

        # 重新排列查询张量 q 的形状
        q = rearrange(q, "b n (h d) -> b h n d", h = h)

        # 缩放
        q = q * self.scale

        # 相似度计算
        sim = einsum("b h i d, b j d -> b h i j", q, kv)

        # 添加 alibi 偏置
        sim = sim + self.alibi_pos_biases(sim)

        # 因果掩码
        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力计算
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b j d -> b h i d", attn, kv)

        # 合并头部
        out = rearrange(out, "b h n d -> b n (h d)")

        # 合并头部并通过注意力输出和前馈输出层
        merge_heads = self.attn_out(out) + self.ff_out(ff)
        return merge_heads
# 定义一个函数PaLM，使用关键字参数，接受模型的维度dim、标记数量num_tokens、层数depth、头部维度dim_head、头部数量heads、前馈网络倍增ff_mult作为参数
def PaLM(*, dim, num_tokens, depth, dim_head=64, heads=8, ff_mult=4):

    # 创建一个神经网络模型，包括嵌入层、多个平行Transformer块、RMSNorm层和线性层
    net = nn.Sequential(
        nn.Embedding(num_tokens, dim), # 嵌入层，将标记映射到指定维度的向量
        *[Residual(ParallelTransformerBlock(dim, dim_head, heads, ff_mult)) for _ in range(depth)], # 多个平行Transformer块
        RMSNorm(dim), # RMSNorm层
        nn.Linear(dim, num_tokens, bias=False) # 线性层，将维度映射回标记数量
    )

    # 将最后一层的权重设置为与第一层嵌入层的权重相同，实现权重共享
    net[-1].weight = net[0].weight

    # 对第一层嵌入层的权重进行正态分布初始化
    nn.init.normal_(net[0].weight, std=0.02)
    
    # 返回神经网络模型
    return net

# 主函数，用于测试模型的功能
if __name__ == "__main__":

    # 创建一个PaLM模型实例
    palm = PaLM(
        num_tokens = 20000,
        dim = 512,
        depth = 1,
        heads = 8,
        dim_head = 64,
    )

    # 生成随机标记序列
    tokens = torch.randint(0, 20000, (1, 2048))
    # 输入标记序列到模型，得到预测结果logits
    logits = palm(tokens) # (1, 2048, 20000)

    # 统计模型中可训练参数的数量
    n_params_torch = sum(
        p.numel() for p in palm.parameters() if p.requires_grad
    )

    # 打印模型中可训练参数的数量
    print(f"Number of parameters in torch model: {n_params_torch}")
```