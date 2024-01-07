# `Bert-VITS2\onnx_modules\V220_novq_dev\models_onnx.py`

```

import math  # 导入数学库
import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch库中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch库中导入神经网络函数模块

import commons  # 导入自定义的commons模块
import modules  # 导入自定义的modules模块
from . import attentions_onnx  # 从当前目录下的attentions_onnx模块中导入内容

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # 从PyTorch库中导入一维卷积、一维转置卷积和二维卷积
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # 从PyTorch库中导入权重归一化、移除权重归一化和谱归一化

from commons import init_weights, get_padding  # 从自定义的commons模块中导入初始化权重和获取填充的函数
from .text import symbols, num_tones, num_languages  # 从当前目录下的text模块中导入符号、音调数量和语言数量

# 定义DurationDiscriminator类，继承自nn.Module
class DurationDiscriminator(nn.Module):  # vits2
    # 初始化函数
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()  # 调用父类的初始化函数

        # 初始化各个参数
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # 定义Dropout层
        self.drop = nn.Dropout(p_dropout)
        # 定义一维卷积层
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 定义LayerNorm层
        self.norm_1 = modules.LayerNorm(filter_channels)
        # 定义一维卷积层
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 定义LayerNorm层
        self.norm_2 = modules.LayerNorm(filter_channels)
        # 定义一维卷积层
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        # 定义预输出的一维卷积层
        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 定义预输出的LayerNorm层
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        # 定义预输出的一维卷积层
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # 定义预输出的LayerNorm层
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        # 如果gin_channels不为0，定义条件的一维卷积层
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        # 定义输出层，包含线性层和Sigmoid激活函数
        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    # 前向传播函数，计算输出概率
    def forward_probability(self, x, x_mask, dur, g=None):
        # 对持续时间进行投影
        dur = self.dur_proj(dur)
        # 将输入和持续时间拼接起来
        x = torch.cat([x, dur], dim=1)
        # 进行预输出的一维卷积操作
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.drop(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = self.drop(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        # 经过输出层得到输出概率
        output_prob = self.output_layer(x)
        return output_prob

    # 前向传播函数，计算输出概率
    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        # 对输入进行detach操作
        x = torch.detach(x)
        # 如果g不为None，对g进行detach操作
        if g is not None:
            g = torch.detach(g)
            # 对输入进行条件操作
            x = x + self.cond(g)
        # 进行第一个一维卷积操作
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        # 进行第二个一维卷积操作
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        # 初始化输出概率列表
        output_probs = []
        # 遍历持续时间的真实值和预测值
        for dur in [dur_r, dur_hat]:
            # 计算输出概率
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append(output_prob)

        return output_probs

```