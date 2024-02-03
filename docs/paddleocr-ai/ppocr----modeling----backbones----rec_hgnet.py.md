# `.\PaddleOCR\ppocr\modeling\backbones\rec_hgnet.py`

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

# 导入 PaddlePaddle 模块
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal, Constant
from paddle.nn import Conv2D, BatchNorm2D, ReLU, AdaptiveAvgPool2D, MaxPool2D
from paddle.regularizer import L2Decay
from paddle import ParamAttr

# 初始化权重和偏置
kaiming_normal_ = KaimingNormal()
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

# 定义一个包含卷积、批归一化和激活函数的模块
class ConvBNAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_act=True):
        super().__init__()
        self.use_act = use_act
        # 创建卷积层
        self.conv = Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias_attr=False)
        # 创建批归一化层
        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        # 如果需要激活函数，则创建 ReLU 激活函数层
        if self.use_act:
            self.act = ReLU()

    def forward(self, x):
        # 前向传播过程
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x

# 定义一个 ESE 模块
class ESEModule(nn.Layer):
    # 初始化函数，接受通道数作为参数
    def __init__(self, channels):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个自适应平均池化层，输出大小为1
        self.avg_pool = AdaptiveAvgPool2D(1)
        # 创建一个卷积层，输入通道数和输出通道数相同，卷积核大小为1，步长为1，填充为0
        self.conv = Conv2D(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0)
        # 创建一个 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 保存输入 x 作为恒等映射
        identity = x
        # 对输入 x 进行平均池化
        x = self.avg_pool(x)
        # 对平均池化后的结果进行卷积
        x = self.conv(x)
        # 对卷积后的结果应用 Sigmoid 激活函数
        x = self.sigmoid(x)
        # 返回恒等映射和经过处理后的结果的乘积
        return paddle.multiply(x=identity, y=x)
# 定义 HG_Block 类，用于构建 Hourglass 网络的一个模块
class HG_Block(nn.Layer):
    # 初始化函数，接受输入通道数、中间通道数、输出通道数、层数和是否使用残差连接作为参数
    def __init__(
            self,
            in_channels,
            mid_channels,
            out_channels,
            layer_num,
            identity=False, ):
        super().__init__()
        # 设置是否使用残差连接
        self.identity = identity

        # 创建一个包含所有层的 LayerList
        self.layers = nn.LayerList()
        # 添加第一个卷积层到 layers 中
        self.layers.append(
            ConvBNAct(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1))
        # 根据层数循环添加中间卷积层到 layers 中
        for _ in range(layer_num - 1):
            self.layers.append(
                ConvBNAct(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1))

        # 特征聚合
        total_channels = in_channels + layer_num * mid_channels
        # 添加聚合卷积层到 layers 中
        self.aggregation_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1)
        # 添加 ESEModule 模块到 layers 中
        self.att = ESEModule(out_channels)

    # 前向传播函数
    def forward(self, x):
        # 保存输入作为 identity
        identity = x
        # 初始化输出列表
        output = []
        output.append(x)
        # 遍历所有层，对输入进行卷积操作，并将结果添加到输出列表中
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        # 在通道维度上拼接所有输出
        x = paddle.concat(output, axis=1)
        # 对拼接后的特征进行聚合卷积
        x = self.aggregation_conv(x)
        # 使用 ESEModule 进行特征加权
        x = self.att(x)
        # 如果使用残差连接，则将原始输入与处理后的特征相加
        if self.identity:
            x += identity
        # 返回处理后的特征
        return x


class HG_Stage(nn.Layer):
    # 初始化函数，定义了一个神经网络模型
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 block_num,
                 layer_num,
                 downsample=True,
                 stride=[2, 1]):
        # 调用父类的初始化函数
        super().__init__()
        # 是否进行下采样
        self.downsample = downsample
        # 如果需要下采样
        if downsample:
            # 创建下采样层，使用ConvBNAct类
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                use_act=False)

        # 存储所有的HG_Block块
        blocks_list = []
        # 添加第一个HG_Block块，identity为False表示不使用残差连接
        blocks_list.append(
            HG_Block(
                in_channels,
                mid_channels,
                out_channels,
                layer_num,
                identity=False))
        # 添加剩余的HG_Block块，共block_num-1个，identity为True表示使用残差连接
        for _ in range(block_num - 1):
            blocks_list.append(
                HG_Block(
                    out_channels,
                    mid_channels,
                    out_channels,
                    layer_num,
                    identity=True))
        # 将所有的HG_Block块组合成一个序列
        self.blocks = nn.Sequential(*blocks_list)

    # 前向传播函数
    def forward(self, x):
        # 如果需要下采样，则对输入进行下采样
        if self.downsample:
            x = self.downsample(x)
        # 将输入通过所有的HG_Block块
        x = self.blocks(x)
        # 返回输出
        return x
# 定义 PPHGNet 类，用于构建 PPHGNet 模型
class PPHGNet(nn.Layer):
    """
    PPHGNet
    Args:
        stem_channels: list. Stem channel list of PPHGNet. PPHGNet 的 stem 通道列表
        stage_config: dict. The configuration of each stage of PPHGNet. such as the number of channels, stride, etc. PPHGNet 每个阶段的配置，如通道数、步长等
        layer_num: int. Number of layers of HG_Block. HG_Block 的层数
        use_last_conv: boolean. Whether to use a 1x1 convolutional layer before the classification layer. 是否在分类层之前使用 1x1 卷积层
        class_expand: int=2048. Number of channels for the last 1x1 convolutional layer. 最后一个 1x1 卷积层的通道数
        dropout_prob: float. Parameters of dropout, 0.0 means dropout is not used. dropout 参数，0.0 表示不使用 dropout
        class_num: int=1000. The number of classes. 类别数
    Returns:
        model: nn.Layer. Specific PPHGNet model depends on args. 返回 PPHGNet 模型
    """
    # 初始化函数，设置网络结构参数
    def __init__(
            self,
            stem_channels,
            stage_config,
            layer_num,
            in_channels=3,
            det=False,
            out_indices=None, ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置是否为目标检测模式
        self.det = det
        # 设置输出索引，如果未指定则默认为[0, 1, 2, 3]
        self.out_indices = out_indices if out_indices is not None else [
            0, 1, 2, 3
        ]

        # stem
        # 将输入通道数插入到通道列表的第一个位置
        stem_channels.insert(0, in_channels)
        # 创建 stem 层，包含多个 ConvBNAct 模块
        self.stem = nn.Sequential(* [
            ConvBNAct(
                in_channels=stem_channels[i],
                out_channels=stem_channels[i + 1],
                kernel_size=3,
                stride=2 if i == 0 else 1) for i in range(
                    len(stem_channels) - 1)
        ])

        # 如果是目标检测模式，添加最大池化层
        if self.det:
            self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        
        # 创建 stages 层，包含多个 HG_Stage 模块
        self.stages = nn.LayerList()
        self.out_channels = []
        # 遍历 stage_config 中的每个阶段
        for block_id, k in enumerate(stage_config):
            # 获取当前阶段的参数
            in_channels, mid_channels, out_channels, block_num, downsample, stride = stage_config[
                k]
            # 添加 HG_Stage 模块到 stages 层
            self.stages.append(
                HG_Stage(in_channels, mid_channels, out_channels, block_num,
                         layer_num, downsample, stride))
            # 如果当前阶段在输出索引中，记录输出通道数
            if block_id in self.out_indices:
                self.out_channels.append(out_channels)

        # 如果不是目标检测模式，设置输出通道数为 stage4 的输出通道数
        if not self.det:
            self.out_channels = stage_config["stage4"][2]

        # 初始化网络权重
        self._init_weights()

    # 初始化网络权重
    def _init_weights(self):
        # 遍历网络的子层
        for m in self.sublayers():
            # 如果是卷积层，使用 kaiming_normal_ 初始化权重
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            # 如果是 BatchNorm2D 层，将权重初始化为 1，偏置初始化为 0
            elif isinstance(m, (nn.BatchNorm2D)):
                ones_(m.weight)
                zeros_(m.bias)
            # 如果是全连接层，偏置初始化为 0
            elif isinstance(m, nn.Linear):
                zeros_(m.bias)
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 使用 stem 模块处理输入 x
        x = self.stem(x)
        # 如果是检测模式，对 x 进行池化操作
        if self.det:
            x = self.pool(x)

        # 初始化一个空列表 out 用于存储输出
        out = []
        # 遍历 self.stages 中的每个阶段
        for i, stage in enumerate(self.stages):
            # 对 x 应用当前阶段的处理
            x = stage(x)
            # 如果是检测模式且当前阶段在输出索引中
            if self.det and i in self.out_indices:
                # 将处理后的 x 添加到输出列表 out 中
                out.append(x)
        # 如果是检测模式，返回输出列表 out
        if self.det:
            return out

        # 如果是训练模式
        if self.training:
            # 对 x 进行自适应平均池化，输出大小为 [1, 40]
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            # 如果不是训练模式，对 x 进行平均池化，输出大小为 [3, 2]
            x = F.avg_pool2d(x, [3, 2])
        # 返回处理后的 x
        return x
def PPHGNet_tiny(pretrained=False, use_ssld=False, **kwargs):
    """
    PPHGNet_tiny
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPHGNet_tiny` model depends on args.
    """
    # 定义不同阶段的配置参数，包括输入通道数、中间通道数、输出通道数、块数、是否下采样等
    stage_config = {
        # in_channels, mid_channels, out_channels, blocks, downsample
        "stage1": [96, 96, 224, 1, False, [2, 1]],
        "stage2": [224, 128, 448, 1, True, [1, 2]],
        "stage3": [448, 160, 512, 2, True, [2, 1]],
        "stage4": [512, 192, 768, 1, True, [2, 1]],
    }

    # 创建 PPHGNet 模型，设置不同阶段的配置参数
    model = PPHGNet(
        stem_channels=[48, 48, 96],
        stage_config=stage_config,
        layer_num=5,
        **kwargs)
    return model


def PPHGNet_small(pretrained=False, use_ssld=False, det=False, **kwargs):
    """
    PPHGNet_small
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPHGNet_small` model depends on args.
    """
    # 目标检测模型的阶段配置参数
    stage_config_det = {
        # in_channels, mid_channels, out_channels, blocks, downsample
        "stage1": [128, 128, 256, 1, False, 2],
        "stage2": [256, 160, 512, 1, True, 2],
        "stage3": [512, 192, 768, 2, True, 2],
        "stage4": [768, 224, 1024, 1, True, 2],
    }

    # 人脸识别模型的阶段配置参数
    stage_config_rec = {
        # in_channels, mid_channels, out_channels, blocks, downsample
        "stage1": [128, 128, 256, 1, True, [2, 1]],
        "stage2": [256, 160, 512, 1, True, [1, 2]],
        "stage3": [512, 192, 768, 2, True, [2, 1]],
        "stage4": [768, 224, 1024, 1, True, [2, 1]],
    }
    # 创建 PPHGNet 模型对象，根据参数设置不同的参数值
    model = PPHGNet(
        stem_channels=[64, 64, 128],  # 设置模型的初始通道数
        stage_config=stage_config_det if det else stage_config_rec,  # 根据 det 参数选择不同的 stage_config
        layer_num=6,  # 设置模型的层数
        det=det,  # 根据 det 参数选择不同的模式
        **kwargs)  # 使用 kwargs 参数传递额外的参数
    # 返回创建的模型对象
    return model
def PPHGNet_base(pretrained=False, use_ssld=True, **kwargs):
    """
    PPhGNet_base 函数定义，用于创建 PPHGNet_base 模型
    Args:
        pretrained: bool=False or str. 如果为 `True`，则加载预训练参数，否则为 `False`。
                    如果为字符串，则表示预训练模型的路径。
        use_ssld: bool=False. 当 pretrained=True 时，是否使用蒸馏预训练模型。
    Returns:
        model: nn.Layer. 返回具体的 `PPHGNet_base` 模型，取决于参数。
    """
    # 定义不同阶段的配置信息，包括输入通道数、中间通道数、输出通道数、块数、是否下采样、下采样步长
    stage_config = {
        # in_channels, mid_channels, out_channels, blocks, downsample, downsample_stride
        "stage1": [160, 192, 320, 1, False, [2, 1]],
        "stage2": [320, 224, 640, 2, True, [1, 2]],
        "stage3": [640, 256, 960, 3, True, [2, 1]],
        "stage4": [960, 288, 1280, 2, True, [2, 1]],
    }

    # 创建 PPHGNet 模型，设置不同阶段的参数，包括干节点通道数、阶段配置、层级数、dropout 概率等
    model = PPHGNet(
        stem_channels=[96, 96, 160],
        stage_config=stage_config,
        layer_num=7,
        dropout_prob=0.2,
        **kwargs)
    # 返回创建的模型
    return model
```