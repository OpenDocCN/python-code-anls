# `.\lucidrains\slot-attention\slot_attention\slot_attention.py`

```py
import torch
from torch import nn
from torch.nn import init

class SlotAttention(nn.Module):
    # 定义 SlotAttention 类，继承自 nn.Module
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        # 初始化函数，接受 num_slots（槽的数量）、dim（维度）、iters（迭代次数，默认为3）、eps（小数值，默认为1e-8）、hidden_dim（隐藏层维度，默认为128）
        super().__init__()
        # 调用父类的初始化函数

        self.num_slots = num_slots
        # 设置槽的数量
        self.iters = iters
        # 设置迭代次数
        self.eps = eps
        # 设置小数值
        self.scale = dim ** -0.5
        # 计算缩放因子

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        # 初始化槽的均值参数
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        # 初始化槽的对数标准差参数
        init.xavier_uniform_(self.slots_logsigma)
        # 使用 Xavier 初始化方法初始化槽的对数标准差参数

        self.to_q = nn.Linear(dim, dim)
        # 创建线性层，用于将输入转换为查询向量
        self.to_k = nn.Linear(dim, dim)
        # 创建线性层，用于将输入转换为键向量
        self.to_v = nn.Linear(dim, dim)
        # 创建线性层，用于将输入转换为值向量

        self.gru = nn.GRUCell(dim, dim)
        # 创建 GRU 单元，用于更新槽的状态

        hidden_dim = max(dim, hidden_dim)
        # 计算隐藏层维度

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        # 创建多层感知机模型，用于更新槽的状态

        self.norm_input  = nn.LayerNorm(dim)
        # 创建 LayerNorm 层，用于对输入进行归一化
        self.norm_slots  = nn.LayerNorm(dim)
        # 创建 LayerNorm 层，用于对槽的状态进行归一化
        self.norm_pre_ff = nn.LayerNorm(dim)
        # 创建 LayerNorm 层，用于对前馈网络的输出进行归一化

    def forward(self, inputs, num_slots = None):
        # 前向传播函数，接受输入和槽的数量（可选）
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        # 获取输入的形状、设备和数据类型
        n_s = num_slots if num_slots is not None else self.num_slots
        # 设置槽的数量为给定值或默认值

        mu = self.slots_mu.expand(b, n_s, -1)
        # 复制槽的均值参数以匹配批次大小和槽的数��
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        # 计算槽的标准差并复制以匹配批次大小和槽的数量

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)
        # 生成服从正态分布的槽的状态

        inputs = self.norm_input(inputs)
        # 对输入进行归一化
        k, v = self.to_k(inputs), self.to_v(inputs)
        # 将输入转换为键和值

        for _ in range(self.iters):
            # 迭代更新槽的状态
            slots_prev = slots
            # 保存上一次的槽状态

            slots = self.norm_slots(slots)
            # 对槽的状态进行归一化
            q = self.to_q(slots)
            # 将槽的状态转换为查询向量

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            # 计算查询向量和键向量的点积，并乘以缩放因子
            attn = dots.softmax(dim=1) + self.eps
            # 对点积结果进行 softmax 操作，并加上小数值

            attn = attn / attn.sum(dim=-1, keepdim=True)
            # 归一化注意力权重

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # 根据注意力权重更新值向量

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            # 使用 GRU 单元更新槽的状态

            slots = slots.reshape(b, -1, d)
            # 重新调整槽的状态的形状
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            # 使用多层感知机更新槽的状态

        return slots
        # 返回更新后的槽的状态
```