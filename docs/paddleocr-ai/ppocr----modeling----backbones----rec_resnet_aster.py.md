# `.\PaddleOCR\ppocr\modeling\backbones\rec_resnet_aster.py`

```py
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
# 代码参考自:
# https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/resnet_aster.py
"""
# 导入 paddle 库
import paddle
# 导入 paddle 中的 nn 模块
import paddle.nn as nn

# 导入 sys 模块
import sys
# 导入 math 模块
import math

# 定义一个 3x3 卷积层函数，带有 padding
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=False)

# 定义一个 1x1 卷积层函数
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)

# 获取正弦编码
def get_sinusoid_encoding(n_position, feat_dim, wave_length=10000):
    # 生成位置信息 [n_position]
    positions = paddle.arange(0, n_position)
    # 生成特征维度信息 [feat_dim]
    dim_range = paddle.arange(0, feat_dim)
    dim_range = paddle.pow(wave_length, 2 * (dim_range // 2) / feat_dim)
    # 生成角度信息 [n_position, feat_dim]
    angles = paddle.unsqueeze(
        positions, axis=1) / paddle.unsqueeze(
            dim_range, axis=0)
    angles = paddle.cast(angles, "float32")
    # 计算正弦值和余弦值
    angles[:, 0::2] = paddle.sin(angles[:, 0::2])
    angles[:, 1::2] = paddle.cos(angles[:, 1::2])
    return angles

# 定义 AsterBlock 类，继承自 nn.Layer
class AsterBlock(nn.Layer):
    # 初始化函数，定义了AsterBlock类的初始化方法
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # 调用父类的初始化方法
        super(AsterBlock, self).__init__()
        # 创建1x1卷积层，输入通道数为inplanes，输出通道数为planes，步长为stride
        self.conv1 = conv1x1(inplanes, planes, stride)
        # 创建Batch Normalization层，输入通道数为planes
        self.bn1 = nn.BatchNorm2D(planes)
        # 创建ReLU激活函数层
        self.relu = nn.ReLU()
        # 创建3x3卷积层，输入通道数为planes，输出通道数为planes
        self.conv2 = conv3x3(planes, planes)
        # 创建Batch Normalization层，输入通道数为planes
        self.bn2 = nn.BatchNorm2D(planes)
        # 下采样函数，用于降低特征图的尺寸
        self.downsample = downsample
        # 步长
        self.stride = stride

    # 前向传播函数，定义了AsterBlock类的前向传播方法
    def forward(self, x):
        # 保存输入x作为残差
        residual = x
        # 第一个卷积层
        out = self.conv1(x)
        # Batch Normalization
        out = self.bn1(out)
        # ReLU激活函数
        out = self.relu(out)
        # 第二个卷积层
        out = self.conv2(out)
        # Batch Normalization
        out = self.bn2(out)

        # 如果存在下采样函数
        if self.downsample is not None:
            # 对输入x进行下采样
            residual = self.downsample(x)
        # 将残差与输出相加
        out += residual
        # 再次经过ReLU激活函数
        out = self.relu(out)
        # 返回输出
        return out
class ResNet_ASTER(nn.Layer):
    """For aster or crnn"""

    def __init__(self, with_lstm=True, n_group=1, in_channels=3):
        # 初始化 ResNet_ASTER 类
        super(ResNet_ASTER, self).__init__()
        # 设置是否包含 LSTM 和分组数
        self.with_lstm = with_lstm
        self.n_group = n_group

        # 定义网络的第一层
        self.layer0 = nn.Sequential(
            # 卷积层，输入通道数为 in_channels，输出通道数为 32，卷积核大小为 3x3，步长为 1，填充为 1，无偏置
            nn.Conv2D(
                in_channels,
                32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias_attr=False),
            # 批归一化层，通道数为 32
            nn.BatchNorm2D(32),
            # ReLU 激活函数
            nn.ReLU())

        # 初始化当前层的输入通道数
        self.inplanes = 32
        # 创建网络的第二层
        self.layer1 = self._make_layer(32, 3, [2, 2])  # [16, 50]
        # 创建网络的第三层
        self.layer2 = self._make_layer(64, 4, [2, 2])  # [8, 25]
        # 创建网络的第四层
        self.layer3 = self._make_layer(128, 6, [2, 1])  # [4, 25]
        # 创建网络的第五层
        self.layer4 = self._make_layer(256, 6, [2, 1])  # [2, 25]
        # 创建网络的第六层
        self.layer5 = self._make_layer(512, 3, [2, 1])  # [1, 25]

        # 如果包含 LSTM 层
        if with_lstm:
            # 创建 LSTM 层，输入通道数为 512，隐藏单元数为 256，双向，2 层
            self.rnn = nn.LSTM(512, 256, direction="bidirect", num_layers=2)
            # 设置输出通道数为 2 * 256
            self.out_channels = 2 * 256
        else:
            # 设置输出通道数为 512
            self.out_channels = 512

    def _make_layer(self, planes, blocks, stride):
        # 初始化下采样层
        downsample = None
        # 如果步长不为 [1, 1] 或者输入通道数不等于输出通道数
        if stride != [1, 1] or self.inplanes != planes:
            # 创建下采样层，包括 1x1 卷积层和批归一化层
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride), nn.BatchNorm2D(planes))

        # 初始化层列表
        layers = []
        # 添加 AsterBlock 模块到层列表
        layers.append(AsterBlock(self.inplanes, planes, stride, downsample))
        # 更新当前层的输入通道数
        self.inplanes = planes
        # 循环创建多个 AsterBlock 模块并添加到层列表
        for _ in range(1, blocks):
            layers.append(AsterBlock(self.inplanes, planes))
        # 返回包含所有层的序列
        return nn.Sequential(*layers)
    # 定义前向传播函数，接收输入 x
    def forward(self, x):
        # 通过 layer0 处理输入 x，得到 x0
        x0 = self.layer0(x)
        # 通过 layer1 处理 x0，得到 x1
        x1 = self.layer1(x0)
        # 通过 layer2 处理 x1，得到 x2
        x2 = self.layer2(x1)
        # 通过 layer3 处理 x2，得到 x3
        x3 = self.layer3(x2)
        # 通过 layer4 处理 x3，得到 x4
        x4 = self.layer4(x3)
        # 通过 layer5 处理 x4，得到 x5
        x5 = self.layer5(x4)

        # 压缩 x5 的第二维，得到 cnn_feat，形状为 [N, c, w]
        cnn_feat = x5.squeeze(2)
        # 调换 cnn_feat 的维度顺序，perm=[0, 2, 1] 表示将第二维和第三维交换
        cnn_feat = paddle.transpose(cnn_feat, perm=[0, 2, 1])
        # 如果模型包含 LSTM 层
        if self.with_lstm:
            # 通过 LSTM 处理 cnn_feat，得到 rnn_feat
            rnn_feat, _ = self.rnn(cnn_feat)
            # 返回 rnn_feat
            return rnn_feat
        else:
            # 如果模型不包含 LSTM 层，则直接返回 cnn_feat
            return cnn_feat
```