# `.\lucidrains\se3-transformer-pytorch\se3_transformer_pytorch\rotary.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 定义 SinusoidalEmbeddings 类，继承自 nn.Module
class SinusoidalEmbeddings(nn.Module):
    # 初始化函数，接受维度参数 dim
    def __init__(self, dim):
        # 调用父类的初始化函数
        super().__init__()
        # 计算频率的倒数，用于生成正弦位置编码
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 将频率的倒数作为缓冲区注册到模型中
        self.register_buffer('inv_freq', inv_freq)

    # 前向传播函数，接受输入张量 t
    def forward(self, t):
        # 计算频率，用于生成正弦位置编码
        freqs = t[..., None].float() * self.inv_freq[None, :]
        # 将频率重复两次，用于位置编码
        return repeat(freqs, '... d -> ... (d r)', r = 2)

# 定义 rotate_half 函数，用于旋转输入张量的一半
def rotate_half(x):
    # 重新排列输入张量的维度
    x = rearrange(x, '... (d j) m -> ... d j m', j = 2)
    # 将输入张量按照最后一个维度拆分为两部分
    x1, x2 = x.unbind(dim = -2)
    # 将两部分张量进行旋转并拼接在一起
    return torch.cat((-x2, x1), dim = -2)

# 定义 apply_rotary_pos_emb 函数，用于应用旋转位置编码
def apply_rotary_pos_emb(t, freqs):
    # 获取旋转维度的大小
    rot_dim = freqs.shape[-2]
    # 将输入张量 t 拆分为旋转部分和非旋转部分
    t, t_pass = t[..., :rot_dim, :], t[..., rot_dim:, :]
    # 应用旋转位置编码到输入张量 t
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    # 将旋转部分和非旋转部分拼接在一起
    return torch.cat((t, t_pass), dim = -2)
```