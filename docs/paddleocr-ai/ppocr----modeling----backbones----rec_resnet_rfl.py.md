# `.\PaddleOCR\ppocr\modeling\backbones\rec_resnet_rfl.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码来源于：
# https://github.com/hikopensource/DAVAR-Lab-OCR/blob/main/davarocr/davar_rcg/models/backbones/ResNetRFL.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
# 导入 paddle 中的 nn 模块
import paddle.nn as nn

# 从 paddle.nn.initializer 中导入 TruncatedNormal, Constant, Normal, KaimingNormal 初始化器
from paddle.nn.initializer import TruncatedNormal, Constant, Normal, KaimingNormal

# 使用 Kaiming 初始化方法
kaiming_init_ = KaimingNormal()
# 使用常数初始化为 0
zeros_ = Constant(value=0.)
# 使用常数初始化为 1
ones_ = Constant(value=1.)

# 定义 BasicBlock 类，继承自 nn.Layer
class BasicBlock(nn.Layer):
    """Res-net Basic Block"""
    # 扩展系数为 1
    expansion = 1
    # 初始化函数，定义基本块的结构
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_type='BN',
                 **kwargs):
        """
        Args:
            inplanes (int): 输入通道数
            planes (int): 中间特征的通道数
            stride (int): 卷积的步长
            downsample (int): 下采样类型
            norm_type (str): 归一化类型
            **kwargs (None): 备用参数
        """
        # 调用父类的初始化函数
        super(BasicBlock, self).__init__()
        # 定义第一个卷积层
        self.conv1 = self._conv3x3(inplanes, planes)
        # 定义第一个批归一化层
        self.bn1 = nn.BatchNorm(planes)
        # 定义第二个卷积层
        self.conv2 = self._conv3x3(planes, planes)
        # 定义第二个批归一化层
        self.bn2 = nn.BatchNorm(planes)
        # 定义激活函数ReLU
        self.relu = nn.ReLU()
        # 定义下采样方式
        self.downsample = downsample
        # 定义步长
        self.stride = stride

    # 定义3x3卷积函数
    def _conv3x3(self, in_planes, out_planes, stride=1):
        # 返回一个3x3的卷积层
        return nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False)

    # 前向传播函数
    def forward(self, x):
        # 保存输入的残差
        residual = x

        # 第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果有下采样，对输入进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)
        # 将残差与输出相加
        out += residual
        out = self.relu(out)

        return out
# 定义 ResNetRFL 类，继承自 nn.Layer
class ResNetRFL(nn.Layer):
    # 定义_make_layer 方法，用于创建 ResNet 的每个层
    def _make_layer(self, block, planes, blocks, stride=1):
        
        # 初始化 downsample 变量
        downsample = None
        # 如果步长不为1或者输入通道数不等于输出通道数乘以 block 的扩展系数
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 创建 downsample，包含一个卷积层和一个批归一化层
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                nn.BatchNorm(planes * block.expansion), )
        
        # 初始化 layers 列表
        layers = list()
        # 将第一个 block 添加到 layers 中
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 更新输入通道数
        self.inplanes = planes * block.expansion
        # 循环创建剩余的 blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        # 返回包含所有 layers 的 Sequential 对象
        return nn.Sequential(*layers)

    # 定义前向传播方法
    def forward(self, inputs):
        # 使用 backbone 处理输入数据
        x_1 = self.backbone(inputs)

        # 如果使用视觉特征
        if self.use_cnt:
            # 处理视觉特征
            v_x = self.v_maxpool3(x_1)
            v_x = self.v_layer3(v_x)
            v_x = self.v_conv3(v_x)
            v_x = self.v_bn3(v_x)
            visual_feature_2 = self.relu(v_x)

            v_x = self.v_layer4(visual_feature_2)
            v_x = self.v_conv4_1(v_x)
            v_x = self.v_bn4_1(v_x)
            v_x = self.relu(v_x)
            v_x = self.v_conv4_2(v_x)
            v_x = self.v_bn4_2(v_x)
            visual_feature_3 = self.relu(v_x)
        else:
            visual_feature_3 = None
        
        # 如果使用序列特征
        if self.use_seq:
            x = self.maxpool3(x_1)
            x = self.layer3(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x_2 = self.relu(x)

            x = self.layer4(x_2)
            x = self.conv4_1(x)
            x = self.bn4_1(x)
            x = self.relu(x)
            x = self.conv4_2(x)
            x = self.bn4_2(x)
            x_3 = self.relu(x)
        else:
            x_3 = None
        
        # 返回视觉特征和序列特征
        return [visual_feature_3, x_3]


# 定义 ResNetBase 类，继承自 nn.Layer
class ResNetBase(nn.Layer):
    # 初始化函数，定义 ResNetBase 类，接受输入通道数、输出通道数、基本块、层数作为参数
    def __init__(self, in_channels, out_channels, block, layers):
        # 调用父类的初始化函数
        super(ResNetBase, self).__init__()

        # 定义每个块的输出通道数
        self.out_channels_block = [
            int(out_channels / 4), int(out_channels / 2), out_channels,
            out_channels
        ]

        # 初始化输入平面数为输出通道数的八分之一
        self.inplanes = int(out_channels / 8)
        # 创建第一个卷积层，输入通道数为 in_channels，输出通道数为输出通道数的十六分之一
        self.conv0_1 = nn.Conv2D(
            in_channels,
            int(out_channels / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        # 创建第一个批归一化层
        self.bn0_1 = nn.BatchNorm(int(out_channels / 16))
        # 创建第二个卷积层，输入通道数为输出通道数的十六分之一，输出通道数为输入平面数
        self.conv0_2 = nn.Conv2D(
            int(out_channels / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        # 创建第二个批归一化层
        self.bn0_2 = nn.BatchNorm(self.inplanes)
        # 创建 ReLU 激活函数
        self.relu = nn.ReLU()

        # 创建第一个最大池化层
        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        # 创建第一个残差块层
        self.layer1 = self._make_layer(block, self.out_channels_block[0],
                                       layers[0])
        # 创建第一个卷积层，输入输出通道数相同
        self.conv1 = nn.Conv2D(
            self.out_channels_block[0],
            self.out_channels_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        # 创建第一个批归一化层
        self.bn1 = nn.BatchNorm(self.out_channels_block[0])

        # 创建第二个最大池化层
        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        # 创建第二个残差块层，步长为1
        self.layer2 = self._make_layer(
            block, self.out_channels_block[1], layers[1], stride=1)
        # 创建第二个卷积层，输入输出通道数相同
        self.conv2 = nn.Conv2D(
            self.out_channels_block[1],
            self.out_channels_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        # 创建第二个批归一化层
        self.bn2 = nn.BatchNorm(self.out_channels_block[1])
    # 创建 ResNet 的一个层，包含多个 block
    def _make_layer(self, block, planes, blocks, stride=1):
        # 初始化下采样为 None
        downsample = None
        # 如果步长不为 1 或者输入通道数不等于 planes * block.expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 创建下采样模块，包含一个卷积层和一个批归一化层
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                nn.BatchNorm(planes * block.expansion), )

        # 初始化层列表
        layers = list()
        # 添加第一个 block 到层列表
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 更新输入通道数
        self.inplanes = planes * block.expansion
        # 循环创建剩余的 block 并添加到层列表
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # 返回包含所有 block 的序列
        return nn.Sequential(*layers)

    # 前向传播函数
    def forward(self, x):
        # 第一层卷积
        x = self.conv0_1(x)
        # 第一层批归一化
        x = self.bn0_1(x)
        # 激活函数
        x = self.relu(x)
        # 第二层卷积
        x = self.conv0_2(x)
        # 第二层批归一化
        x = self.bn0_2(x)
        # 激活函数
        x = self.relu(x)

        # 最大池化
        x = self.maxpool1(x)
        # 第一个层
        x = self.layer1(x)
        # 第一个卷积
        x = self.conv1(x)
        # 第一个批归一化
        x = self.bn1(x)
        # 激活函数
        x = self.relu(x)

        # 最大池化
        x = self.maxpool2(x)
        # 第二个层
        x = self.layer2(x)
        # 第二个卷积
        x = self.conv2(x)
        # 第二个批归一化
        x = self.bn2(x)
        # 激活函数
        x = self.relu(x)

        # 返回结果
        return x
# 定义一个名为 RFLBase 的类，用于实现共享骨干网络的互逆特征学习
class RFLBase(nn.Layer):
    """ Reciprocal feature learning share backbone network"""

    # 初始化方法，接受输入通道数和输出通道数，默认为512
    def __init__(self, in_channels, out_channels=512):
        # 调用父类的初始化方法
        super(RFLBase, self).__init__()
        # 创建一个名为 ConvNet 的属性，使用 ResNetBase 类来构建共享骨干网络
        self.ConvNet = ResNetBase(in_channels, out_channels, BasicBlock,
                                  [1, 2, 5, 3])

    # 前向传播方法，接受输入 inputs
    def forward(self, inputs):
        # 调用 ConvNet 属性的前向传播方法，返回结果
        return self.ConvNet(inputs)
```