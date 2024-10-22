# `.\diffusers\pipelines\wuerstchen\modeling_wuerstchen_common.py`

```py
# 导入 PyTorch 及其神经网络模块
import torch
import torch.nn as nn

# 从指定路径导入 Attention 处理模块
from ...models.attention_processor import Attention


# 定义自定义的层归一化类，继承自 nn.LayerNorm
class WuerstchenLayerNorm(nn.LayerNorm):
    # 初始化方法，接收可变参数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

    # 前向传播方法
    def forward(self, x):
        # 调整输入张量的维度顺序
        x = x.permute(0, 2, 3, 1)
        # 调用父类的前向传播方法进行归一化
        x = super().forward(x)
        # 恢复输入张量的维度顺序并返回
        return x.permute(0, 3, 1, 2)


# 定义时间步块类，继承自 nn.Module
class TimestepBlock(nn.Module):
    # 初始化方法，接收通道数和时间步数
    def __init__(self, c, c_timestep):
        # 调用父类的初始化方法
        super().__init__()
        # 定义线性映射层，将时间步数映射到两倍的通道数
        self.mapper = nn.Linear(c_timestep, c * 2)

    # 前向传播方法
    def forward(self, x, t):
        # 使用映射层处理时间步，并将结果分割为两个部分
        a, b = self.mapper(t)[:, :, None, None].chunk(2, dim=1)
        # 根据公式更新输入张量并返回
        return x * (1 + a) + b


# 定义残差块类，继承自 nn.Module
class ResBlock(nn.Module):
    # 初始化方法，接收通道数、跳过连接的通道数、卷积核大小和丢弃率
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        # 调用父类的初始化方法
        super().__init__()
        # 定义深度可分离卷积层
        self.depthwise = nn.Conv2d(c + c_skip, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        # 定义自定义层归一化
        self.norm = WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-6)
        # 定义通道处理的顺序模块
        self.channelwise = nn.Sequential(
            nn.Linear(c, c * 4), nn.GELU(), GlobalResponseNorm(c * 4), nn.Dropout(dropout), nn.Linear(c * 4, c)
        )

    # 前向传播方法
    def forward(self, x, x_skip=None):
        # 保存输入张量以便后续残差连接
        x_res = x
        # 如果有跳过连接的张量，则进行拼接
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        # 对输入张量进行深度卷积和归一化
        x = self.norm(self.depthwise(x)).permute(0, 2, 3, 1)
        # 通过通道处理模块
        x = self.channelwise(x).permute(0, 3, 1, 2)
        # 返回残差连接后的结果
        return x + x_res


# 从外部库导入的全局响应归一化类
class GlobalResponseNorm(nn.Module):
    # 初始化方法，接收特征维度
    def __init__(self, dim):
        # 调用父类的初始化方法
        super().__init__()
        # 定义可学习参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    # 前向传播方法
    def forward(self, x):
        # 计算输入张量的聚合范数
        agg_norm = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # 计算标准化范数
        stand_div_norm = agg_norm / (agg_norm.mean(dim=-1, keepdim=True) + 1e-6)
        # 返回经过归一化后的结果
        return self.gamma * (x * stand_div_norm) + self.beta + x


# 定义注意力块类，继承自 nn.Module
class AttnBlock(nn.Module):
    # 初始化方法，接收通道数、条件通道数、头数、是否自注意力及丢弃率
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        # 调用父类的初始化方法
        super().__init__()
        # 设置是否使用自注意力
        self.self_attn = self_attn
        # 定义归一化层
        self.norm = WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-6)
        # 定义注意力机制
        self.attention = Attention(query_dim=c, heads=nhead, dim_head=c // nhead, dropout=dropout, bias=True)
        # 定义键值映射层
        self.kv_mapper = nn.Sequential(nn.SiLU(), nn.Linear(c_cond, c))

    # 前向传播方法
    def forward(self, x, kv):
        # 使用键值映射层处理 kv
        kv = self.kv_mapper(kv)
        # 对输入张量进行归一化
        norm_x = self.norm(x)
        # 如果使用自注意力，则拼接归一化后的 x 和 kv
        if self.self_attn:
            batch_size, channel, _, _ = x.shape
            kv = torch.cat([norm_x.view(batch_size, channel, -1).transpose(1, 2), kv], dim=1)
        # 将注意力机制的输出与原输入相加
        x = x + self.attention(norm_x, encoder_hidden_states=kv)
        # 返回处理后的张量
        return x
```