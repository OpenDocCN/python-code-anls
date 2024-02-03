# `.\PaddleOCR\StyleText\arch\base_module.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发在“按原样”基础上，
# 没有任何明示或暗示的保证或条件
# 有关特定语言的权限和限制，请参阅许可证
import paddle
import paddle.nn as nn

# 从 arch.spectral_norm 模块导入 spectral_norm 函数
from arch.spectral_norm import spectral_norm

# 定义 CBN 类，继承自 nn.Layer
class CBN(nn.Layer):
    # 定义 CBN 类，继承自 nn.Layer 类
    def __init__(self,
                 name,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 norm_layer=None,
                 act=None,
                 act_attr=None):
        super(CBN, self).__init__()
        # 如果使用偏置，则创建 ParamAttr 对象，否则为 None
        if use_bias:
            bias_attr = paddle.ParamAttr(name=name + "_bias")
        else:
            bias_attr = None
        # 创建 Conv2D 对象，设置各种参数
        self._conv = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight_attr=paddle.ParamAttr(name=name + "_weights"),
            bias_attr=bias_attr)
        # 如果指定了 norm_layer，则创建相应的层
        if norm_layer:
            self._norm_layer = getattr(paddle.nn, norm_layer)(
                num_features=out_channels, name=name + "_bn")
        else:
            self._norm_layer = None
        # 如果指定了激活函数，则创建相应的激活函数层
        if act:
            if act_attr:
                self._act = getattr(paddle.nn, act)(**act_attr,
                                                    name=name + "_" + act)
            else:
                self._act = getattr(paddle.nn, act)(name=name + "_" + act)
        else:
            self._act = None
    
    # 前向传播函数
    def forward(self, x):
        # 使用卷积层处理输入数据
        out = self._conv(x)
        # 如果有归一化层，则对输出进行归一化处理
        if self._norm_layer:
            out = self._norm_layer(out)
        # 如果有激活函数，则对输出进行激活函数处理
        if self._act:
            out = self._act(out)
        # 返回处理后的输出
        return out
# 定义一个带有参数的卷积层类 SNConv，继承自 nn.Layer
class SNConv(nn.Layer):
    # 初始化函数，接受多个参数
    def __init__(self,
                 name,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 norm_layer=None,
                 act=None,
                 act_attr=None):
        # 调用父类的初始化函数
        super(SNConv, self).__init__()
        # 如果使用偏置项，则设置偏置项属性
        if use_bias:
            bias_attr = paddle.ParamAttr(name=name + "_bias")
        else:
            bias_attr = None
        # 使用 spectral_norm 对 paddle.nn.Conv2D 进行封装，创建 SNConv 层
        self._sn_conv = spectral_norm(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                weight_attr=paddle.ParamAttr(name=name + "_weights"),
                bias_attr=bias_attr))
        # 如果指定了规范化层，则创建规范化层
        if norm_layer:
            self._norm_layer = getattr(paddle.nn, norm_layer)(
                num_features=out_channels, name=name + "_bn")
        else:
            self._norm_layer = None
        # 如果指定了激活函数，则创建激活函数
        if act:
            if act_attr:
                self._act = getattr(paddle.nn, act)(**act_attr,
                                                    name=name + "_" + act)
            else:
                self._act = getattr(paddle.nn, act)(name=name + "_" + act)
        else:
            self._act = None

    # 前向传播函数，接受输入 x，返回输出 out
    def forward(self, x):
        # 对输入 x 进行 SNConv 操作
        out = self._sn_conv(x)
        # 如果存在规范化层，则对输出进行规范化
        if self._norm_layer:
            out = self._norm_layer(out)
        # 如果存在激活函数，则对输出进行激活
        if self._act:
            out = self._act(out)
        # 返回处理后的输出
        return out


# 定义 SNConvTranspose 类，继承自 nn.Layer
class SNConvTranspose(nn.Layer):
    # 定义 SNConvTranspose 类，继承自 nn.Layer 类
    def __init__(self,
                 name,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 norm_layer=None,
                 act=None,
                 act_attr=None):
        # 调用父类的构造函数
        super(SNConvTranspose, self).__init__()
        # 如果使用偏置项，则设置偏置项属性
        if use_bias:
            bias_attr = paddle.ParamAttr(name=name + "_bias")
        else:
            bias_attr = None
        # 使用谱归一化函数对 Conv2DTranspose 进行谱归一化
        self._sn_conv_transpose = spectral_norm(
            paddle.nn.Conv2DTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
                weight_attr=paddle.ParamAttr(name=name + "_weights"),
                bias_attr=bias_attr))
        # 如果指定了规范化层，则创建规范化层
        if norm_layer:
            self._norm_layer = getattr(paddle.nn, norm_layer)(
                num_features=out_channels, name=name + "_bn")
        else:
            self._norm_layer = None
        # 如果指定了激活函数，则创建激活函数
        if act:
            if act_attr:
                self._act = getattr(paddle.nn, act)(**act_attr,
                                                    name=name + "_" + act)
            else:
                self._act = getattr(paddle.nn, act)(name=name + "_" + act)
        else:
            self._act = None
    
    # 前向传播函数
    def forward(self, x):
        # 对输入进行 SNConvTranspose 操作
        out = self._sn_conv_transpose(x)
        # 如果存在规范化层，则对输出进行规范化
        if self._norm_layer:
            out = self._norm_layer(out)
        # 如果存在激活函数，则对输出进行激活
        if self._act:
            out = self._act(out)
        # 返回处理后的输出
        return out
# 定义一个名为MiddleNet的类，继承自nn.Layer
class MiddleNet(nn.Layer):
    # 初始化方法，接受名称、输入通道数、中间通道数、输出通道数和是否使用偏置作为参数
    def __init__(self, name, in_channels, mid_channels, out_channels, use_bias):
        # 调用父类的初始化方法
        super(MiddleNet, self).__init__()
        # 创建一个名为_sn_conv1的SNConv对象，用于1x1卷积
        self._sn_conv1 = SNConv(
            name=name + "_sn_conv1",
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            use_bias=use_bias,
            norm_layer=None,
            act=None)
        # 创建一个2D填充层对象，用于填充
        self._pad2d = nn.Pad2D(padding=[1, 1, 1, 1], mode="replicate")
        # 创建一个名为_sn_conv2的SNConv对象，用于3x3卷积
        self._sn_conv2 = SNConv(
            name=name + "_sn_conv2",
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            use_bias=use_bias)
        # 创建一个名为_sn_conv3的SNConv对象，用于1x1卷积
        self._sn_conv3 = SNConv(
            name=name + "_sn_conv3",
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_bias=use_bias)

    # 前向传播方法，接受输入x
    def forward(self, x):
        # 对输入x进行1x1卷积操作
        sn_conv1 = self._sn_conv1.forward(x)
        # 对卷积结果进行2D填充操作
        pad_2d = self._pad2d.forward(sn_conv1)
        # 对填充结果进行3x3卷积操作
        sn_conv2 = self._sn_conv2.forward(pad_2d)
        # 对卷积结果进行1x1卷积操作
        sn_conv3 = self._sn_conv3.forward(sn_conv2)
        # 返回卷积结果
        return sn_conv3
    # 初始化 ResBlock 类
    def __init__(self, name, channels, norm_layer, use_dropout, use_dilation,
                 use_bias):
        # 调用父类的初始化方法
        super(ResBlock, self).__init__()
        # 根据是否使用 dilation 设置不同的 padding
        if use_dilation:
            padding_mat = [1, 1, 1, 1]
        else:
            padding_mat = [0, 0, 0, 0]
        # 创建 Pad2D 层，用于填充
        self._pad1 = nn.Pad2D(padding_mat, mode="replicate")

        # 创建 SNConv 层，用于卷积操作
        self._sn_conv1 = SNConv(
            name=name + "_sn_conv1",
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=0,
            norm_layer=norm_layer,
            use_bias=use_bias,
            act="ReLU",
            act_attr=None)
        # 根据是否使用 dropout 设置不同的操作
        if use_dropout:
            self._dropout = nn.Dropout(0.5)
        else:
            self._dropout = None
        # 创建第二个 Pad2D 层
        self._pad2 = nn.Pad2D([1, 1, 1, 1], mode="replicate")
        # 创建第二个 SNConv 层
        self._sn_conv2 = SNConv(
            name=name + "_sn_conv2",
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            norm_layer=norm_layer,
            use_bias=use_bias,
            act="ReLU",
            act_attr=None)

    # 前向传播函数
    def forward(self, x):
        # 对输入进行第一个填充操作
        pad1 = self._pad1.forward(x)
        # 对填充后的数据进行第一个卷积操作
        sn_conv1 = self._sn_conv1.forward(pad1)
        # 对第一个卷积结果进行第二个填充操作
        pad2 = self._pad2.forward(sn_conv1)
        # 对第二个填充结果进行第二个卷积操作
        sn_conv2 = self._sn_conv2.forward(pad2)
        # 返回最终结果，加上输入数据本身
        return sn_conv2 + x
```