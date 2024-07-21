# `.\pytorch\test\onnx\model_defs\srresnet.py`

```
import math  # 导入数学库，用于数学运算

from torch import nn  # 从 torch 库中导入神经网络模块
from torch.nn import init  # 从 torch.nn 模块中导入初始化方法


def _initialize_orthogonal(conv):
    """
    初始化正交矩阵

    Args:
    conv: 卷积层对象

    Returns:
    None
    """
    prelu_gain = math.sqrt(2)  # 计算 PReLU 的增益值
    init.orthogonal(conv.weight, gain=prelu_gain)  # 使用正交初始化卷积层权重
    if conv.bias is not None:
        conv.bias.data.zero_()  # 如果有偏置项，则将其数据置零


class ResidualBlock(nn.Module):
    """
    残差块类，继承自 nn.Module

    Args:
    n_filters: 滤波器数量
    """
    def __init__(self, n_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1, bias=False
        )  # 第一个卷积层，无偏置项
        self.bn1 = nn.BatchNorm2d(n_filters)  # 第一个批归一化层
        self.prelu = nn.PReLU(n_filters)  # PReLU 激活函数
        self.conv2 = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1, bias=False
        )  # 第二个卷积层，无偏置项
        self.bn2 = nn.BatchNorm2d(n_filters)  # 第二个批归一化层

        # 正交初始化
        _initialize_orthogonal(self.conv1)
        _initialize_orthogonal(self.conv2)

    def forward(self, x):
        """
        前向传播函数

        Args:
        x: 输入张量

        Returns:
        输出张量
        """
        residual = self.prelu(self.bn1(self.conv1(x)))  # 第一个残差块操作
        residual = self.bn2(self.conv2(residual))  # 第二个残差块操作
        return x + residual  # 返回原始输入与残差的和


class UpscaleBlock(nn.Module):
    """
    上采样块类，继承自 nn.Module

    Args:
    n_filters: 滤波器数量
    """
    def __init__(self, n_filters):
        super().__init__()
        self.upscaling_conv = nn.Conv2d(
            n_filters, 4 * n_filters, kernel_size=3, padding=1
        )  # 上采样卷积层
        self.upscaling_shuffler = nn.PixelShuffle(2)  # 像素洗牌层，进行上采样
        self.upscaling = nn.PReLU(n_filters)  # PReLU 激活函数
        _initialize_orthogonal(self.upscaling_conv)  # 正交初始化

    def forward(self, x):
        """
        前向传播函数

        Args:
        x: 输入张量

        Returns:
        输出张量
        """
        return self.upscaling(self.upscaling_shuffler(self.upscaling_conv(x)))  # 上采样块的前向传播


class SRResNet(nn.Module):
    """
    超分辨率残差网络类，继承自 nn.Module

    Args:
    rescale_factor: 尺度因子
    n_filters: 滤波器数量
    n_blocks: 残差块数量
    """
    def __init__(self, rescale_factor, n_filters, n_blocks):
        super().__init__()
        self.rescale_levels = int(math.log(rescale_factor, 2))  # 计算尺度级别
        self.n_filters = n_filters  # 滤波器数量
        self.n_blocks = n_blocks  # 残差块数量

        self.conv1 = nn.Conv2d(3, n_filters, kernel_size=9, padding=4)  # 输入到第一个卷积层
        self.prelu1 = nn.PReLU(n_filters)  # 第一个 PReLU 激活函数

        # 添加残差块
        for residual_block_num in range(1, n_blocks + 1):
            residual_block = ResidualBlock(self.n_filters)  # 创建残差块
            self.add_module(
                "residual_block" + str(residual_block_num),
                nn.Sequential(residual_block),  # 将残差块添加到模型
            )

        self.skip_conv = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1, bias=False
        )  # 跳跃连接的卷积层，无偏置项
        self.skip_bn = nn.BatchNorm2d(n_filters)  # 跳跃连接的批归一化层

        # 添加上采样块
        for upscale_block_num in range(1, self.rescale_levels + 1):
            upscale_block = UpscaleBlock(self.n_filters)  # 创建上采样块
            self.add_module(
                "upscale_block" + str(upscale_block_num), nn.Sequential(upscale_block)
            )  # 将上采样块添加到模型

        self.output_conv = nn.Conv2d(n_filters, 3, kernel_size=9, padding=4)  # 输出卷积层

        # 正交初始化
        _initialize_orthogonal(self.conv1)
        _initialize_orthogonal(self.skip_conv)
        _initialize_orthogonal(self.output_conv)
    # 定义神经网络的前向传播方法，接收输入张量 x
    def forward(self, x):
        # 对输入张量 x 进行第一层卷积和 PReLU 激活处理，得到初始化后的特征张量 x_init
        x_init = self.prelu1(self.conv1(x))
        
        # 通过第一个残差块处理特征张量 x，并更新 x
        x = self.residual_block1(x_init)
        
        # 循环处理后续的残差块，self.n_blocks 是残差块的总数
        for residual_block_num in range(2, self.n_blocks + 1):
            # 根据残差块的编号动态获取并应用残差块的处理方法
            x = getattr(self, "residual_block" + str(residual_block_num))(x)
        
        # 对特征张量 x 进行跳跃连接的批量归一化、卷积操作，再加上初始的特征张量 x_init
        x = self.skip_bn(self.skip_conv(x)) + x_init
        
        # 循环处理上采样块，self.rescale_levels 是上采样块的总数
        for upscale_block_num in range(1, self.rescale_levels + 1):
            # 根据上采样块的编号动态获取并应用上采样块的处理方法
            x = getattr(self, "upscale_block" + str(upscale_block_num))(x)
        
        # 将最终处理后的特征张量 x 输入到输出卷积层中，得到神经网络的输出结果
        return self.output_conv(x)
```