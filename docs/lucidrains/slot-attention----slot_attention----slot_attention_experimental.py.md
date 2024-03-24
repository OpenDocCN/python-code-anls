# `.\lucidrains\slot-attention\slot_attention\slot_attention_experimental.py`

```py
import torch
from torch import nn
from torch.nn import init

class WeightedAttention(nn.Module):
    def __init__(self, dim, eps = 1e-8, softmax_dim = 1, weighted_mean_dim = 2):
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)  # 对输入进行归一化
        self.norm_context = nn.LayerNorm(dim)  # 对上下文进行归一化

        self.to_q = nn.Linear(dim, dim)  # 线性变换，将输入转换为查询向量
        self.to_k = nn.Linear(dim, dim)  # 线性变换，将上下文转换为键向量
        self.to_v = nn.Linear(dim, dim)  # 线性变换，将上下文转换为值向量

        self.eps = eps  # 用于稳定softmax计算的小值
        self.scale = dim ** -0.5  # 缩放因子
        self.softmax_dim = softmax_dim  # softmax计算的维度
        self.weighted_mean_dim = weighted_mean_dim  # 加权平均的维度

    def forward(self, inputs, context):

        inputs = self.norm_input(inputs)  # 对输入进行归一化
        context = self.norm_context(context)  # 对上下文进行归一化

        q = self.to_q(inputs)  # 计算查询向量
        k = self.to_k(context)  # 计算键向量
        v = self.to_v(context)  # 计算值向量

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale  # 计算点积
        attn = dots.softmax(dim = self.softmax_dim) + self.eps  # 计算注意力权重
        attn = attn / attn.sum(dim = self.weighted_mean_dim, keepdim=True)  # 计算加权平均

        updates = torch.einsum('bjd,bij->bid', v, attn)  # 计算更新
        return updates

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class GatedResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)  # GRU单元
        self.fn = fn
    def forward(self, *args):
        inputs = args[0]
        b, _, d = inputs.shape

        updates = self.fn(*args)

        inputs = self.gru(
            updates.reshape(-1, d),
            inputs.reshape(-1, d)
        )
        return inputs.reshape(b, -1, d)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = max(dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 线性变换
            nn.ReLU(inplace = True),  # ReLU激活函数
            nn.Linear(hidden_dim, dim)  # 线性变换
        )
        self.norm = nn.LayerNorm(dim)  # 对输出进行归一化

    def forward(self, x):
        x = self.norm(x)  # 对输入进行归一化
        return self.net(x)

class SlotAttentionExperimental(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        scale = dim ** -0.5
        self.num_slots = num_slots
        self.iters = iters

        self.norm_inputs = nn.LayerNorm(dim)  # 对输入进行归一化

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))  # 槽的均值参数

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))  # 槽的对数标准差参数
        init.xavier_uniform_(self.slots_logsigma)  # 初始化槽的对数标准差参数

        self.slots_to_inputs_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps))  # 槽到输入的注意力机制
        self.slots_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))  # 槽的前馈网络

        self.inputs_to_slots_attn = GatedResidual(dim, WeightedAttention(dim, eps = eps, softmax_dim = 2, weighted_mean_dim = 1))  # 输入到槽的注意力机制
        self.inputs_ff = GatedResidual(dim, FeedForward(dim, hidden_dim))  # 输入的前馈网络

    def forward(self, inputs, num_slots = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)  # 扩展槽的均值参数
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)  # 扩展槽的对数标准差参数

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)  # 生成槽

        inputs = self.norm_inputs(inputs)  # 对输入进行归一化

        for _ in range(self.iters):
            slots = self.slots_to_inputs_attn(slots, inputs)  # 槽到输入的注意力机制
            slots = self.slots_ff(slots)  # 槽的前馈网络

            inputs = self.inputs_to_slots_attn(inputs, slots)  # 输入到槽的注意力机制
            inputs = self.inputs_ff(inputs)  # 输入的前馈网络

        return slots, inputs  # 返回槽和输入
```