# `.\lucidrains\ESBN-pytorch\esbn_pytorch\esbn_pytorch.py`

```
# 导入 torch 库
import torch
# 从 functools 库中导入 partial 函数
from functools import partial
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 einops 库中导入 repeat 和 rearrange 函数
from einops import repeat, rearrange

# 定义辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 安全地拼接张量的函数
def safe_cat(t, el, dim = 0):
    if not exists(t):
        return el
    return torch.cat((t, el), dim = dim)

# 映射函数的函数
def map_fn(fn, *args, **kwargs):
    def inner(*arr):
        return map(lambda t: fn(t, *args, **kwargs), arr)
    return inner

# 定义类

# 定义 ESBN 类，继承自 nn.Module 类
class ESBN(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        value_dim = 64,
        key_dim = 64,
        hidden_dim = 512,
        output_dim = 4,
        encoder = None
    ):
        super().__init__()
        # 初始化隐藏状态、细胞状态和键
        self.h0 = torch.zeros(hidden_dim)
        self.c0 = torch.zeros(hidden_dim)
        self.k0 = torch.zeros(key_dim + 1)

        # 定义 LSTMCell 层、线性层和全连接层
        self.rnn = nn.LSTMCell(key_dim + 1, hidden_dim)
        self.to_gate = nn.Linear(hidden_dim, 1)
        self.to_key = nn.Linear(hidden_dim, key_dim)
        self.to_output = nn.Linear(hidden_dim, output_dim)

        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 4, stride = 2),
            nn.Flatten(1),
            nn.Linear(4 * 64, value_dim)
        ) if not exists(encoder) else encoder

        # 定义置信度的线性层
        self.to_confidence = nn.Linear(1, 1)

    # 前向传播函数
    def forward(self, images):
        # 获取 batch 大小
        b = images.shape[1]
        Mk = None
        Mv = None

        # 将隐藏状态、细胞状态和键重复到 batch 维度
        hx, cx, kx, k0 = map_fn(repeat, 'd -> b d', b = b)(self.h0, self.c0, self.k0, self.k0)
        out = []

        # 遍历图像序列
        for ind, image in enumerate(images):
            is_first = ind == 0
            z = self.encoder(image)
            hx, cx = self.rnn(kx, (hx, cx))
            y, g, kw = self.to_output(hx), self.to_gate(hx), self.to_key(hx)

            if is_first:
                kx = k0
            else:
                # 注意力机制
                sim = einsum('b n d, b d -> b n', Mv, z)
                wk = sim.softmax(dim = -1)

                # 计算置信度
                sim, wk = map_fn(rearrange, 'b n -> b n ()')(sim, wk)
                ck = self.to_confidence(sim).sigmoid()

                # 拼接置信度到记忆键中，然后根据注意力对记忆值进行加权求和
                kx = g.sigmoid() * (wk * torch.cat((Mk, ck), dim = -1)).sum(dim = 1)

            kw, z = map_fn(rearrange, 'b d -> b () d')(kw, z)
            Mk = safe_cat(Mk, kw, dim = 1)
            Mv = safe_cat(Mv, z, dim = 1)
            out.append(y)

        # 将输出堆叠成张量
        return torch.stack(out)
```