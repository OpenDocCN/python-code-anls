# `.\comic-translate\modules\ocr\pororo\pororo\models\brainOCR\modules\feature_extraction.py`

```py
import torch.nn as nn

# 定义一个特征提取器类，基于 VGG 网络结构
class VGGFeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    # 前向传播方法，接受输入 x，通过 ConvNet 处理后返回结果
    def forward(self, x):
        return self.ConvNet(x)


# 定义一个特征提取器类，基于 ResNet 网络结构
class ResNetFeatureExtractor(nn.Module):
    """
    FeatureExtractor of FAN
    (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)
    """

    # 初始化方法，设置输入通道数和输出通道数，默认为 1 输入通道和 512 输出通道
    def __init__(self, n_input_channels: int = 1, n_output_channels: int = 512):
        super(ResNetFeatureExtractor, self).__init__()
        # 使用 ResNet 网络构建 ConvNet，并设置不同的层数
        self.ConvNet = ResNet(n_input_channels, n_output_channels, BasicBlock,
                              [1, 2, 5, 3])

    # 前向传播方法，接受输入 inputs，通过 ConvNet 处理后返回结果
    def forward(self, inputs):
        return self.ConvNet(inputs)


# 定义基础的 ResNet 块
class BasicBlock(nn.Module):
    expansion = 1

    # 初始化方法，设置输入通道数、输出通道数、步长和下采样
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个 3x3 卷积层，不使用偏置
        self.conv1 = self._conv3x3(inplanes, planes, stride)
        # 批标准化层
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个 3x3 卷积层，不使用偏置
        self.conv2 = self._conv3x3(planes, planes)
        # 批标准化层
        self.bn2 = nn.BatchNorm2d(planes)
        # ReLU 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    # 定义一个 3x3 的卷积函数，带有填充
    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes,
                         out_planes,
                         kernel_size=3,
                         stride=stride,
                         padding=1,
                         bias=False)

    # 前向传播方法，接受输入 x，通过基础块的结构处理后返回结果
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在下采样，则对输入进行下采样操作
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


# 定义一个完整的 ResNet 网络
class ResNet(nn.Module):

    # 构建 ResNet 的层方法，包括块、通道数、块数和步长
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 如果步长不为 1 或者输入通道数不等于输出通道数乘块的扩展倍数，则进行下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 添加第一个块
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 添加额外的块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    # 前向传播函数，用于执行神经网络的前向计算过程
    def forward(self, x):
        # 第一层卷积操作，对输入 x 执行卷积运算
        x = self.conv0_1(x)
        # 对卷积结果执行批归一化操作
        x = self.bn0_1(x)
        # 对批归一化后的结果执行 ReLU 激活函数
        x = self.relu(x)
        
        # 第二层卷积操作，对前一层输出执行卷积运算
        x = self.conv0_2(x)
        # 对卷积结果执行批归一化操作
        x = self.bn0_2(x)
        # 对批归一化后的结果执行 ReLU 激活函数
        x = self.relu(x)

        # 执行最大池化操作，减少数据维度
        x = self.maxpool1(x)
        # 执行第一个残差块的计算
        x = self.layer1(x)
        # 执行额外的卷积操作，进一步提取特征
        x = self.conv1(x)
        # 对卷积结果执行批归一化操作
        x = self.bn1(x)
        # 对批归一化后的结果执行 ReLU 激活函数
        x = self.relu(x)

        # 执行第二次最大池化操作，再次减少数据维度
        x = self.maxpool2(x)
        # 执行第二个残差块的计算
        x = self.layer2(x)
        # 执行额外的卷积操作，进一步提取特征
        x = self.conv2(x)
        # 对卷积结果执行批归一化操作
        x = self.bn2(x)
        # 对批归一化后的结果执行 ReLU 激活函数
        x = self.relu(x)

        # 执行第三次最大池化操作，再次减少数据维度
        x = self.maxpool3(x)
        # 执行第三个残差块的计算
        x = self.layer3(x)
        # 执行额外的卷积操作，进一步提取特征
        x = self.conv3(x)
        # 对卷积结果执行批归一化操作
        x = self.bn3(x)
        # 对批归一化后的结果执行 ReLU 激活函数
        x = self.relu(x)

        # 执行最后一个残差块的计算
        x = self.layer4(x)
        # 执行最后的卷积操作，进一步提取特征
        x = self.conv4_1(x)
        # 对卷积结果执行批归一化操作
        x = self.bn4_1(x)
        # 对批归一化后的结果执行 ReLU 激活函数
        x = self.relu(x)
        # 执行额外的卷积操作，进一步提取特征
        x = self.conv4_2(x)
        # 对卷积结果执行批归一化操作
        x = self.bn4_2(x)
        # 对批归一化后的结果执行 ReLU 激活函数
        x = self.relu(x)

        # 返回经过所有层操作后的特征张量
        return x
```