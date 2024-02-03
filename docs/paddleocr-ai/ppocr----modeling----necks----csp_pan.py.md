# `.\PaddleOCR\ppocr\modeling\necks\csp_pan.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何担保或条件
# 请查看许可证以获取特定语言的权限和限制

# 代码基于：
# https://github.com/PaddlePaddle/PaddleDetection/blob/release%2F2.3/ppdet/modeling/necks/csp_pan.py

# 导入所需的库
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

# 定义模型中的所有公共接口
__all__ = ['CSPPAN']

# 定义一个卷积 + BN 层的类
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channel=96,
                 out_channel=96,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 act='leaky_relu'):
        super(ConvBNLayer, self).__init__()
        # 使用 KaimingUniform 初始化器
        initializer = nn.initializer.KaimingUniform()
        self.act = act
        assert self.act in ['leaky_relu', "hard_swish"]
        # 创建卷积层
        self.conv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            weight_attr=ParamAttr(initializer=initializer),
            bias_attr=False)
        # 创建 BN 层
        self.bn = nn.BatchNorm2D(out_channel)

    # 前向传播函数
    def forward(self, x):
        # 对输入进行卷积和 BN 处理
        x = self.bn(self.conv(x))
        # 根据激活函数类型进行激活
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        return x

# 定义一个 DPModule 类
class DPModule(nn.Layer):
    """
    Depth-wise and point-wise module.
     Args:
        in_channel (int): This Module的输入通道数.
        out_channel (int): This Module的输出通道数.
        kernel_size (int): This Module的卷积核大小.
        stride (int): This Module的卷积的步长.
        act (str): This Module的激活函数，支持 `leaky_relu` 和 `hard_swish`.
    """

    def __init__(self,
                 in_channel=96,
                 out_channel=96,
                 kernel_size=3,
                 stride=1,
                 act='leaky_relu'):
        super(DPModule, self).__init__()
        # 使用 KaimingUniform 初始化器
        initializer = nn.initializer.KaimingUniform()
        self.act = act
        # 深度可分离卷积层
        self.dwconv = nn.Conv2D(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            groups=out_channel,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            weight_attr=ParamAttr(initializer=initializer),
            bias_attr=False)
        # 批归一化层
        self.bn1 = nn.BatchNorm2D(out_channel)
        # 点卷积层
        self.pwconv = nn.Conv2D(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            groups=1,
            padding=0,
            weight_attr=ParamAttr(initializer=initializer),
            bias_attr=False)
        # 批归一化层
        self.bn2 = nn.BatchNorm2D(out_channel)

    def act_func(self, x):
        # 根据激活函数类型进行激活
        if self.act == "leaky_relu":
            x = F.leaky_relu(x)
        elif self.act == "hard_swish":
            x = F.hardswish(x)
        return x

    def forward(self, x):
        # 深度可分离卷积 + 批归一化 + 激活函数
        x = self.act_func(self.bn1(self.dwconv(x)))
        # 点卷积 + 批归一化 + 激活函数
        x = self.act_func(self.bn2(self.pwconv(x)))
        return x
class DarknetBottleneck(nn.Layer):
    """The basic bottleneck block used in Darknet.
    Each Block consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and act.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expansion=0.5,
                 add_identity=True,
                 use_depthwise=False,
                 act="leaky_relu"):
        super(DarknetBottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        conv_func = DPModule if use_depthwise else ConvBNLayer
        # 第一个卷积层，1x1卷积
        self.conv1 = ConvBNLayer(
            in_channel=in_channels,
            out_channel=hidden_channels,
            kernel_size=1,
            act=act)
        # 第二个卷积层，3x3卷积
        self.conv2 = conv_func(
            in_channel=hidden_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=1,
            act=act)
        # 是否添加恒等映射
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Layer):
    """Cross Stage Partial Layer.
    Args:
        in_channels (int): CSP层的输入通道数。
        out_channels (int): CSP层的输出通道数。
        expand_ratio (float): 调整隐藏层通道数的比例。默认值为0.5。
        num_blocks (int): 块的数量。默认值为1。
        add_identity (bool): 是否在块中添加身份连接。默认值为True。
        use_depthwise (bool): 是否在块中使用深度可分离卷积。默认值为False。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 act="leaky_relu"):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.short_conv = ConvBNLayer(in_channels, mid_channels, 1, act=act)
        self.final_conv = ConvBNLayer(
            2 * mid_channels, out_channels, 1, act=act)

        self.blocks = nn.Sequential(* [
            DarknetBottleneck(
                mid_channels,
                mid_channels,
                kernel_size,
                1.0,
                add_identity,
                use_depthwise,
                act=act) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)  # 对输入进行短连接卷积

        x_main = self.main_conv(x)  # 对输入进行主要卷积
        x_main = self.blocks(x_main)  # 对主要卷积结果进行多个块的处理

        x_final = paddle.concat((x_main, x_short), axis=1)  # 将主要卷积结果和短连接卷积结果拼接在一起
        return self.final_conv(x_final)  # 对拼接结果进行最终卷积
class Channel_T(nn.Layer):
    # Channel_T 类，用于定义通道转换模块
    def __init__(self,
                 in_channels=[116, 232, 464],
                 out_channels=96,
                 act="leaky_relu"):
        # 初始化函数，设置输入通道数、输出通道数和激活函数类型
        super(Channel_T, self).__init__()
        # 调用父类的初始化函数
        self.convs = nn.LayerList()
        # 创建一个空的 LayerList 对象用于存储卷积层
        for i in range(len(in_channels)):
            # 遍历输入通道数列表
            self.convs.append(
                ConvBNLayer(
                    in_channels[i], out_channels, 1, act=act))
            # 向 LayerList 中添加 ConvBNLayer 对象，设置输入通道数、输出通道数、卷积核大小和激活函数类型

    def forward(self, x):
        # 前向传播函数，接收输入 x
        outs = [self.convs[i](x[i]) for i in range(len(x))]
        # 对输入 x 的每个元素应用对应的卷积层，得到输出列表
        return outs
        # 返回输出列表


class CSPPAN(nn.Layer):
    # CSPPAN 类，Path Aggregation Network with CSP module
    """Path Aggregation Network with CSP module.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """
    # 文档字符串，描述 CSPPAN 类的参数信息
    # 初始化函数，定义 CSPPAN 模型的参数
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 num_csp_blocks=1,
                 use_depthwise=True,
                 act='hard_swish'):
        # 调用父类的初始化函数
        super(CSPPAN, self).__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置输出通道数列表，长度与输入通道数相同
        self.out_channels = [out_channels] * len(in_channels)
        # 根据是否使用深度可分离卷积选择不同的卷积函数
        conv_func = DPModule if use_depthwise else ConvBNLayer

        # 创建通道转换层
        self.conv_t = Channel_T(in_channels, out_channels, act=act)

        # 构建自顶向下的模块
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_blocks = nn.LayerList()
        # 遍历输入通道数列表，从最后一个通道开始到第一个通道
        for idx in range(len(in_channels) - 1, 0, -1):
            # 添加 CSPLayer 模块到自顶向下的模块列表
            self.top_down_blocks.append(
                CSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act))

        # 构建自底向上的模块
        self.downsamples = nn.LayerList()
        self.bottom_up_blocks = nn.LayerList()
        # 遍历输入通道数列表，从第一个通道开始到倒数第二个通道
        for idx in range(len(in_channels) - 1):
            # 添加下采样层到下采样列表
            self.downsamples.append(
                conv_func(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    act=act))
            # 添加 CSPLayer 模块到自底向上的模块列表
            self.bottom_up_blocks.append(
                CSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    act=act))
    # 前向传播函数，接收输入特征并返回 CSPPAN 特征
    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features. 输入特征的元组
        Returns:
            tuple[Tensor]: CSPPAN features. CSPPAN 特征的元组
        """
        # 确保输入特征的数量与通道数相同
        assert len(inputs) == len(self.in_channels)
        # 对输入特征进行卷积操作
        inputs = self.conv_t(inputs)

        # 自顶向下路径
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            # 上采样特征以匹配低级特征的大小
            upsample_feat = F.upsample(
                feat_heigh, size=paddle.shape(feat_low)[2:4], mode="nearest")

            # 使用拼接后的特征进行自顶向下块的操作
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # 自底向上路径
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            # 对低级特征进行下采样
            downsample_feat = self.downsamples[idx](feat_low)
            # 使用拼接后的特征进行自底向上块的操作
            out = self.bottom_up_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], 1))
            outs.append(out)

        return tuple(outs)
```