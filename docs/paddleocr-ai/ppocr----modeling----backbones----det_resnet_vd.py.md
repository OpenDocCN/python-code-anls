# `.\PaddleOCR\ppocr\modeling\backbones\det_resnet_vd.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 PaddlePaddle 库
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

# 导入视觉操作相关的库
from paddle.vision.ops import DeformConv2D
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Normal, Constant, XavierUniform

# 定义导出的模块列表
__all__ = ["ResNet_vd", "ConvBNLayer", "DeformableConvV2"]

# 定义 DeformableConvV2 类，继承自 nn.Layer
class DeformableConvV2(nn.Layer):
    # 前向传播函数
    def forward(self, x):
        # 使用卷积层计算偏移和掩码
        offset_mask = self.conv_offset(x)
        # 将偏移和掩码分割成两部分
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1)
        # 对掩码进行 sigmoid 激活
        mask = F.sigmoid(mask)
        # 使用变形卷积计算输出
        y = self.conv_dcn(x, offset, mask=mask)
        return y

# 定义 ConvBNLayer 类，继承自 nn.Layer
class ConvBNLayer(nn.Layer):
    # 定义一个卷积层和批归一化层的类，继承自 nn.Module 类
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 dcn_groups=1,
                 is_vd_mode=False,
                 act=None,
                 is_dcn=False):
        # 调用父类的构造函数
        super(ConvBNLayer, self).__init__()
    
        # 初始化是否为 VD 模式的标志
        self.is_vd_mode = is_vd_mode
        # 创建一个 2x2 的平均池化层
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        # 如果不是使用 Deformable Convolution，则创建普通的卷积层
        if not is_dcn:
            self._conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias_attr=False)
        # 如果使用 Deformable Convolution，则创建 DeformableConvV2 层
        else:
            self._conv = DeformableConvV2(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=dcn_groups,  #groups,
                bias_attr=False)
        # 创建一个批归一化层
        self._batch_norm = nn.BatchNorm(out_channels, act=act)
    
    # 定义前向传播函数
    def forward(self, inputs):
        # 如果是 VD 模式，则对输入进行平均池化
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        # 经过卷积层
        y = self._conv(inputs)
        # 经过批归一化层
        y = self._batch_norm(y)
        # 返回结果
        return y
# 定义 BottleneckBlock 类，继承自 nn.Layer
class BottleneckBlock(nn.Layer):
    # 初始化方法，接受输入通道数、输出通道数、步长、是否使用快捷连接、是否为第一个块、是否使用 DCN 等参数
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            shortcut=True,
            if_first=False,
            is_dcn=False, ):
        # 调用父类的初始化方法
        super(BottleneckBlock, self).__init__()

        # 创建 1x1 卷积层和 BN 层，用于降维
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu')
        # 创建 3x3 卷积层和 BN 层，用于特征提取
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            is_dcn=is_dcn,
            dcn_groups=2)
        # 创建 1x1 卷积层和 BN 层，用于升维
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None)

        # 如果不使用快捷连接，则创建额外的卷积层和 BN 层
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True)

        # 记录是否使用快捷连接
        self.shortcut = shortcut

    # 前向传播方法，接受输入数据
    def forward(self, inputs):
        # 通过 1x1 卷积层和 BN 层进行特征提取
        y = self.conv0(inputs)
        # 通过 3x3 卷积层和 BN 层进行特征提取
        conv1 = self.conv1(y)
        # 通过 1x1 卷积层和 BN 层进行特征提取
        conv2 = self.conv2(conv1)

        # 如果使用快捷连接，则直接将输入作为快捷连接
        if self.shortcut:
            short = inputs
        # 否则通过额外的卷积层和 BN 层进行快捷连接
        else:
            short = self.short(inputs)
        # 将快捷连接和卷积结果相加
        y = paddle.add(x=short, y=conv2)
        # 使用 ReLU 激活函数
        y = F.relu(y)
        # 返回结果
        return y


# 定义 BasicBlock 类
class BasicBlock(nn.Layer):
    # 初始化 BasicBlock 类
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            shortcut=True,
            if_first=False, ):
        # 调用父类的初始化方法
        super(BasicBlock, self).__init__()
        # 设置步长
        self.stride = stride
        # 创建第一个卷积层和批归一化层
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu')
        # 创建第二个卷积层和批归一化层
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None)

        # 如果没有快捷连接，则创建一个短连接
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True)

        # 设置是否有快捷连接
        self.shortcut = shortcut

    # 前向传播函数
    def forward(self, inputs):
        # 第一个卷积层
        y = self.conv0(inputs)
        # 第二个卷积层
        conv1 = self.conv1(y)

        # 如果有快捷连接，则直接将输入作为短连接
        if self.shortcut:
            short = inputs
        # 如果没有快捷连接，则通过短连接层计算短连接
        else:
            short = self.short(inputs)
        # 将短连接和卷积结果相加
        y = paddle.add(x=short, y=conv1)
        # 使用 ReLU 激活函数
        y = F.relu(y)
        # 返回结果
        return y
# 定义一个名为 ResNet_vd 的类，继承自 nn.Layer 类
class ResNet_vd(nn.Layer):
    # 定义类的前向传播方法，接收输入参数 inputs
    def forward(self, inputs):
        # 对输入数据进行第一次卷积操作
        y = self.conv1_1(inputs)
        # 对第一次卷积后的数据进行第二次卷积操作
        y = self.conv1_2(y)
        # 对第二次卷积后的数据进行第三次卷积操作
        y = self.conv1_3(y)
        # 对经过三次卷积后的数据进行最大池化操作
        y = self.pool2d_max(y)
        # 初始化一个空列表 out 用于存储输出结果
        out = []
        # 遍历 self.stages 中的每个元素，i 为索引，block 为元素
        for i, block in enumerate(self.stages):
            # 将数据 y 传入当前 block 中进行处理
            y = block(y)
            # 如果当前索引 i 在 self.out_indices 中
            if i in self.out_indices:
                # 将处理后的数据 y 添加到输出列表 out 中
                out.append(y)
        # 返回输出列表 out
        return out
```