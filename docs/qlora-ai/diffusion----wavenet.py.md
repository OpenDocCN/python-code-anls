# `so-vits-svc\diffusion\wavenet.py`

```
# 导入 math 模块
import math
# 从 math 模块中导入 sqrt 函数
from math import sqrt

# 导入 torch 模块
import torch
# 导入 torch.nn 模块
import torch.nn as nn
# 导入 torch.nn.functional 模块
import torch.nn.functional as F
# 从 torch.nn 模块中导入 Mish 类
from torch.nn import Mish

# 定义 Conv1d 类，继承自 torch.nn.Conv1d
class Conv1d(torch.nn.Conv1d):
    # 初始化函数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化函数
        super().__init__(*args, **kwargs)
        # 使用 kaiming_normal_ 方法初始化权重

# 定义 SinusoidalPosEmb 类，继承自 nn.Module
class SinusoidalPosEmb(nn.Module):
    # 初始化函数
    def __init__(self, dim):
        # 调用父类的初始化函数
        super().__init__()
        # 设置维度属性
        self.dim = dim

    # 前向传播函数
    def forward(self, x):
        # 获取输入张量的设备信息
        device = x.device
        # 计算一半维度
        half_dim = self.dim // 2
        # 计算位置编码的增长率
        emb = math.log(10000) / (half_dim - 1)
        # 计算位置编码
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 计算位置编码矩阵
        emb = x[:, None] * emb[None, :]
        # 拼接正弦和余弦位置编码
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # 返回位置编码张量

# 定义 ResidualBlock 类，继承自 nn.Module
class ResidualBlock(nn.Module):
    # 初始化函数
    def __init__(self, encoder_hidden, residual_channels, dilation):
        # 调用父类的初始化函数
        super().__init__()
        # 设置残差通道数属性
        self.residual_channels = residual_channels
        # 创建膨胀卷积层
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        # 创建扩散投影层
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        # 创建条件投影层
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        # 创建输出投影层
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
    # 定义一个前向传播函数，接受输入 x、条件器 conditioner 和扩散步数 diffusion_step
    def forward(self, x, conditioner, diffusion_step):
        # 对扩散步数进行投影，并在最后添加一个维度
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        # 对条件器进行投影
        conditioner = self.conditioner_projection(conditioner)
        # 将输入 x 和扩散步数相加得到 y
        y = x + diffusion_step

        # 对 y 进行扩张卷积操作，并加上条件器
        y = self.dilated_conv(y) + conditioner

        # 使用 torch.split 而不是 torch.chunk 来避免使用 onnx::Slice
        # 将 y 按照通道数分割成 gate 和 filter
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        # 对 gate 进行 sigmoid 操作，对 filter 进行 tanh 操作，然后相乘得到 y
        y = torch.sigmoid(gate) * torch.tanh(filter)

        # 对 y 进行输出投影
        y = self.output_projection(y)

        # 使用 torch.split 而不是 torch.chunk 来避免使用 onnx::Slice
        # 将 y 按照通道数分割成 residual 和 skip
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        # 返回 (x + residual) 除以 sqrt(2.0) 和 skip
        return (x + residual) / math.sqrt(2.0), skip
class WaveNet(nn.Module):
    # WaveNet 模型类
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        # 初始化函数
        super().__init__()
        # 输入投影层，将输入维度转换为 n_chans 维度
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        # 扩散嵌入层，用于处理扩散步数
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        # 多层感知机，用于学习非线性映射
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        # 残差层列表，包含 n_layers 个残差块
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=1
            )
            for i in range(n_layers)
        ])
        # 跳跃投影层，将残差层输出转换为 n_chans 维度
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        # 输出投影层，将 n_chans 维度转换为 in_dims 维度
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        # 输出投影层权重初始化为零
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        # 将 spec 的第二维度压缩，去掉维度为 1 的维度
        x = spec.squeeze(1)
        # 输入投影层处理输入数据，将其转换为 n_chans 维度
        x = self.input_projection(x)  # [B, residual_channel, T]

        # 使用 ReLU 激活函数
        x = F.relu(x)
        # 扩散步数嵌入处理扩散步数
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        # 遍历残差层列表
        for layer in self.residual_layers:
            # 每个残差层处理输入数据，得到输出和跳跃连接
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        # 对所有跳跃连接求和并除以根号下残差层的数量
        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        # 跳跃投影层处理输出数据
        x = self.skip_projection(x)
        # 使用 ReLU 激活函数
        x = F.relu(x)
        # 输出投影层处理输出数据，将其转换为 in_dims 维度
        x = self.output_projection(x)  # [B, mel_bins, T]
        # 在第二维度上添加一个维度
        return x[:, None, :, :]
```