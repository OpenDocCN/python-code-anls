# `.\PaddleOCR\ppocr\modeling\backbones\rec_resnet_32.py`

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
# 均按“原样”分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用的代码来源于：
# https://github.com/hikopensource/DAVAR-Lab-OCR/davarocr/davar_rcg/models/backbones/ResNet32.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle.nn 模块
import paddle.nn as nn

# 定义导出的模块列表
__all__ = ["ResNet32"]

# 初始化卷积权重属性为 KaimingNormal
conv_weight_attr = nn.initializer.KaimingNormal()

# 定义 ResNet32 类
class ResNet32(nn.Layer):
    """
    Feature Extractor is proposed in  FAN Ref [1]

    Ref [1]: Focusing Attention: Towards Accurate Text Recognition in Neural Images ICCV-2017
    """

    # 初始化函数
    def __init__(self, in_channels, out_channels=512):
        """

        Args:
            in_channels (int): 输入通道数
            output_channel (int): 输出通道数
        """
        super(ResNet32, self).__init__()
        self.out_channels = out_channels
        # 创建 ResNet 对象
        self.ConvNet = ResNet(in_channels, out_channels, BasicBlock, [1, 2, 5, 3])

    # 前向传播函数
    def forward(self, inputs):
        """
        Args:
            inputs: 输入特征

        Returns:
            输出特征

        """
        return self.ConvNet(inputs)

# 定义 BasicBlock 类
class BasicBlock(nn.Layer):
    """Res-net Basic Block"""
    expansion = 1
    # 初始化函数，定义了 BasicBlock 类的初始化方法
    def __init__(self, inplanes, planes,
                 stride=1, downsample=None,
                 norm_type='BN', **kwargs):
        """
        Args:
            inplanes (int): 输入通道数
            planes (int): 中间特征的通道数
            stride (int): 卷积的步长
            downsample (int): 下采样类型
            norm_type (str): 归一化类型
            **kwargs (None): 备用参数
        """
        # 调用父类的初始化方法
        super(BasicBlock, self).__init__()
        # 创建第一个卷积层
        self.conv1 = self._conv3x3(inplanes, planes)
        # 创建第一个批归一化层
        self.bn1 = nn.BatchNorm2D(planes)
        # 创建第二个卷积层
        self.conv2 = self._conv3x3(planes, planes)
        # 创建第二个批归一化层
        self.bn2 = nn.BatchNorm2D(planes)
        # 创建 ReLU 激活函数
        self.relu = nn.ReLU()
        # 设置下采样类型
        self.downsample = downsample
        # 设置卷积的步长
        self.stride = stride

    # 定义一个私有方法，用于创建 3x3 卷积层
    def _conv3x3(self, in_planes, out_planes, stride=1):
        """
        Args:
            in_planes (int): 输入通道数
            out_planes (int): 中间特征的通道数
            stride (int): 卷积的步长
        Returns:
            nn.Layer: 3x3 卷积层
        """

        return nn.Conv2D(in_planes, out_planes,
                         kernel_size=3, stride=stride,
                         padding=1, weight_attr=conv_weight_attr,
                         bias_attr=False)

    # 前向传播函数，定义了 BasicBlock 类的前向传播过程
    def forward(self, x):
        # 保存输入的残差连接
        residual = x

        # 第一个卷积层
        out = self.conv1(x)
        # 第一个批归一化层
        out = self.bn1(out)
        # ReLU 激活函数
        out = self.relu(out)

        # 第二个卷积层
        out = self.conv2(out)
        # 第二个批归一化层
        out = self.bn2(out)

        # 如果存在下采样类型，对输入进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)
        # 残差连接
        out += residual
        # 再次经过 ReLU 激活函数
        out = self.relu(out)

        # 返回输出结果
        return out
class ResNet(nn.Layer):
    """Res-Net network structure"""
    # 定义 Res-Net 网络结构
    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建网络层

        Args:
            block (block): 卷积块
            planes (int): 输入通道数
            blocks (list): 块的层数
            stride (int): 卷积的步长

        Returns:
            nn.Sequential: 卷积块的组合

        """
        downsample = None
        # 如果步长不为1或输入通道数不等于 planes * block.expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 创建下采样层
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride,
                          weight_attr=conv_weight_attr, 
                          bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 循环创建块的层数
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        return x
```