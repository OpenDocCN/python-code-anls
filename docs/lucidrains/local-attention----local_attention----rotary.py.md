# `.\lucidrains\local-attention\local_attention\rotary.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum

# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 定义一个函数，用于检查变量是否存在
def exists(val):
    return val is not None

# 定义一个继承自 nn.Module 的类 SinusoidalEmbeddings
class SinusoidalEmbeddings(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        scale_base = None,
        use_xpos = False
    ):
        super().__init__()
        # 计算频率的倒数
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 将频率的倒数作为缓冲区注册到模型中
        self.register_buffer('inv_freq', inv_freq)

        # xpos 相关

        # 是否使用 xpos
        self.use_xpos = use_xpos
        # 缩放基数
        self.scale_base = scale_base

        # 断言，如果使用 xpos，则必须定义缩放基数
        assert not (use_xpos and not exists(scale_base)), 'scale base must be defined if using xpos'

        # 计算缩放值
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        # 将缩放值作为缓冲区注册到模型中，不持久化
        self.register_buffer('scale', scale, persistent = False)

    # 前向传播函数
    def forward(self, x):
        # 获取序列长度和设备信息
        seq_len, device = x.shape[-2], x.device

        # 生成时间步长
        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        # 计算频率
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        # 如果不使用 xpos，则返回频率和单位矩阵
        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        # 计算幂次
        power = (t - (seq_len // 2)) / self.scale_base
        # 计算缩放值
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

# 定义一个函数，用于将输入向量旋转 180 度
def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b ... r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(q, k, freqs, scale = 1):
    # 获取查询向量的长度
    q_len = q.shape[-2]
    # 获取查询向量的频率
    q_freqs = freqs[..., -q_len:, :]

    # 计算缩放的倒数
    inv_scale = scale ** -1

    # 如果缩放的维度为 2，则截取对应维度
    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    # 对查询向量��用旋转位置嵌入
    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k
```