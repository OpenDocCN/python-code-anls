# `.\PaddleOCR\test_tipc\supplementary\mv3.py`

```py
# 版权声明
# 2020年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发，基于“原样”分发，
# 没有任何形式的担保或条件，无论是明示的还是暗示的。
# 请查看许可证以获取特定语言的权限和
# 许可证下的限制。

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.functional import hardswish, hardsigmoid
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.regularizer import L2Decay
import math

# 从自定义操作中加载自定义操作
from paddle.utils.cpp_extension import load
custom_ops = load(
    name="custom_jit_ops",
    sources=["./custom_op/custom_relu_op.cc", "./custom_op/custom_relu_op.cu"])

# 定义一个函数，使输入值可被除数整除
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 定义MobileNetV3类，继承自nn.Layer
class MobileNetV3(nn.Layer):
    # 前向传播函数
    def forward(self, inputs, label=None):
        # 第一层卷积操作
        x = self.conv1(inputs)

        # 遍历所有的块并进行操作
        for block in self.block_list:
            x = block(x)

        # 倒数第二层卷积操作
        x = self.last_second_conv(x)
        # 池化操作
        x = self.pool(x)

        # 最后一层卷积操作
        x = self.last_conv(x)
        # 使用hardswish激活函数
        x = hardswish(x)
        # dropout操作
        x = self.dropout(x)
        # 将张量展平
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        # 输出层
        x = self.out(x)
        return x

# 定义卷积和批归一化层
class ConvBNLayer(nn.Layer):
    # 定义 ConvBNLayer 类，用于实现卷积、批归一化和激活函数的功能
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 use_cudnn=True,
                 name="",
                 use_custom_relu=False):
        # 调用父类的构造函数
        super(ConvBNLayer, self).__init__()
        # 是否使用激活函数
        self.if_act = if_act
        # 激活函数类型
        self.act = act
        # 创建卷积层对象
        self.conv = Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(),
            bias_attr=False)
        # 创建批归一化层对象
        self.bn = BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
            # moving_mean_name=name + "_bn_mean",
            # moving_variance_name=name + "_bn_variance")
        # 是否使用自定义的激活函数
        self.use_custom_relu = use_custom_relu
    
    # 前向传播函数
    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        # 批归一化操作
        x = self.bn(x)
        # 如果需要激活函数
        if self.if_act:
            # 如果激活函数为 relu
            if self.act == "relu":
                # 如果使用自定义的 relu 函数
                if self.use_custom_relu:
                    x = custom_ops.custom_relu(x)
                else:
                    x = F.relu(x)
            # 如果激活函数为 hardswish
            elif self.act == "hardswish":
                x = hardswish(x)
            else:
                # 打印错误信息并退出
                print("The activation function is selected incorrectly.")
                exit()
        # 返回结果
        return x
# 定义残差单元类，继承自 nn.Layer
class ResidualUnit(nn.Layer):
    # 初始化函数，定义残差单元的各个参数
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None,
                 name='',
                 use_custom_relu=False):
        super(ResidualUnit, self).__init__()
        # 判断是否需要添加快捷连接
        self.if_shortcut = stride == 1 and in_c == out_c
        # 判断是否使用 SE 模块
        self.if_se = use_se

        self.use_custom_relu = use_custom_relu

        # 扩展卷积层，用于将输入通道数扩展到中间通道数
        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + "_expand",
            use_custom_relu=self.use_custom_relu)
        # 瓶颈卷积层，包含深度卷积和分组卷积
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            if_act=True,
            act=act,
            name=name + "_depthwise",
            use_custom_relu=self.use_custom_relu)
        # 如果使用 SE 模块，则添加 SE 模块
        if self.if_se:
            self.mid_se = SEModule(mid_c, name=name + "_se")
        # 线性卷积层，将中间通道数转换为输出通道数
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name=name + "_linear",
            use_custom_relu=self.use_custom_relu)

    # 前向传播函数
    def forward(self, inputs):
        # 执行扩展卷积层
        x = self.expand_conv(inputs)
        # 执行瓶颈卷积层
        x = self.bottleneck_conv(x)
        # 如果使用 SE 模块，则执行 SE 模块
        if self.if_se:
            x = self.mid_se(x)
        # 执行线性卷积层
        x = self.linear_conv(x)
        # 如果需要添加快捷连接，则执行快捷连接
        if self.if_shortcut:
            x = paddle.add(inputs, x)
        # 返回结果
        return x


# 定义 SE 模块类，继承自 nn.Layer
class SEModule(nn.Layer):
    # 初始化方法，接受通道数、缩减比例和名称作为参数
    def __init__(self, channel, reduction=4, name=""):
        # 调用父类的初始化方法
        super(SEModule, self).__init__()
        # 创建一个自适应平均池化层，输出大小为1
        self.avg_pool = AdaptiveAvgPool2D(1)
        # 创建一个卷积层，用于通道压缩，输入通道数为channel，输出通道数为channel // reduction
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())
        # 创建一个卷积层，用于通道扩张，输入通道数为channel // reduction，输出通道数为channel
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())

    # 前向传播方法，接受输入数据inputs
    def forward(self, inputs):
        # 对输入数据进行平均池化
        outputs = self.avg_pool(inputs)
        # 通过第一个卷积层处理数据
        outputs = self.conv1(outputs)
        # 对输出数据进行ReLU激活函数处理
        outputs = F.relu(outputs)
        # 通过第二个卷积层处理数据
        outputs = self.conv2(outputs)
        # 对输出数据进行hardsigmoid激活函数处理，设置斜率为0.2，偏移为0.5
        outputs = hardsigmoid(outputs, slope=0.2, offset=0.5)
        # 返回输入数据与处理后的输出数据的乘积结果
        return paddle.multiply(x=inputs, y=outputs)
# 创建一个 MobileNetV3 模型，规模为 0.35
def MobileNetV3_small_x0_35(**args):
    model = MobileNetV3(model_name="small", scale=0.35, **args)
    return model

# 创建一个 MobileNetV3 模型，规模为 0.5
def MobileNetV3_small_x0_5(**args):
    model = MobileNetV3(model_name="small", scale=0.5, **args)
    return model

# 创建一个 MobileNetV3 模型，规模为 0.75
def MobileNetV3_small_x0_75(**args):
    model = MobileNetV3(model_name="small", scale=0.75, **args)
    return model

# 创建一个 MobileNetV3 模型，规模为 1.0
def MobileNetV3_small_x1_0(**args):
    model = MobileNetV3(model_name="small", scale=1.0, **args)
    return model

# 创建一个 MobileNetV3 模型，规模为 1.25
def MobileNetV3_small_x1_25(**args):
    model = MobileNetV3(model_name="small", scale=1.25, **args)
    return model

# 创建一个 MobileNetV3 模型，规模为 0.35
def MobileNetV3_large_x0_35(**args):
    model = MobileNetV3(model_name="large", scale=0.35, **args)
    return model

# 创建一个 MobileNetV3 模型，规模为 0.5
def MobileNetV3_large_x0_5(**args):
    model = MobileNetV3(model_name="large", scale=0.5, **args)
    return model

# 创建一个 MobileNetV3 模型，规模为 0.75
def MobileNetV3_large_x0_75(**args):
    model = MobileNetV3(model_name="large", scale=0.75, **args)
    return model

# 创建一个 MobileNetV3 模型，规模为 1.0
def MobileNetV3_large_x1_0(**args):
    model = MobileNetV3(model_name="large", scale=1.0, **args)
    return model

# 创建一个 MobileNetV3 模型，规模为 1.25
def MobileNetV3_large_x1_25(**args):
    model = MobileNetV3(model_name="large", scale=1.25, **args)
    return

# 创建一个 DistillMV3 类，包含两个 MobileNetV3 学生模型
class DistillMV3(nn.Layer):
    def __init__(self,
                 scale=1.0,
                 model_name="small",
                 dropout_prob=0.2,
                 class_dim=1000,
                 args=None,
                 use_custom_relu=False):
        super(DistillMV3, self).__init__()

        # 创建一个 MobileNetV3 学生模型
        self.student = MobileNetV3(
            model_name=model_name,
            scale=scale,
            class_dim=class_dim,
            use_custom_relu=use_custom_relu)

        # 创建另一个 MobileNetV3 学生模型
        self.student1 = MobileNetV3(
            model_name=model_name,
            scale=scale,
            class_dim=class_dim,
            use_custom_relu=use_custom_relu)
    # 定义一个前向传播函数，接受输入和标签作为参数
    def forward(self, inputs, label=None):
        # 创建一个空字典用于存储预测结果
        predicts = dict()
        # 调用self.student方法对输入进行预测，并将结果存储在字典中的'student'键下
        predicts['student'] = self.student(inputs, label)
        # 调用self.student1方法对输入进行预测，并将结果存储在字典中的'student1'键下
        predicts['student1'] = self.student1(inputs, label)
        # 返回包含预测结果的字典
        return predicts
# 定义一个函数，用于创建一个DistillMV3模型，参数为model_name="large"和scale=0.5
def distillmv3_large_x0_5(**args):
    model = DistillMV3(model_name="large", scale=0.5, **args)
    return model

# 定义一个SiameseMV3类，继承自nn.Layer
class SiameseMV3(nn.Layer):
    # 初始化函数，接受参数scale=1.0, model_name="small", dropout_prob=0.2, class_dim=1000, args=None, use_custom_relu=False
    def __init__(self,
                 scale=1.0,
                 model_name="small",
                 dropout_prob=0.2,
                 class_dim=1000,
                 args=None,
                 use_custom_relu=False):
        super(SiameseMV3, self).__init__()

        # 创建一个MobileNetV3网络，用于SiameseMV3的net部分
        self.net = MobileNetV3(
            model_name=model_name,
            scale=scale,
            class_dim=class_dim,
            use_custom_relu=use_custom_relu)
        
        # 创建一个MobileNetV3网络，用于SiameseMV3的net1部分
        self.net1 = MobileNetV3(
            model_name=model_name,
            scale=scale,
            class_dim=class_dim,
            use_custom_relu=use_custom_relu)

    # 前向传播函数，接受输入inputs和标签label
    def forward(self, inputs, label=None):
        # 对net部分进行前向传播
        x = self.net.conv1(inputs)
        for block in self.net.block_list:
            x = block(x)

        # 对net1部分进行前向传播
        x1 = self.net1.conv1(inputs)
        for block in self.net1.block_list:
            x1 = block(x1)
        
        # 将net和net1的输出相加
        x = x + x1

        x = self.net.last_second_conv(x)
        x = self.net.pool(x)

        x = self.net.last_conv(x)
        x = hardswish(x)
        x = self.net.dropout(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.net.out(x)
        return x

# 定义一个函数，用于创建一个SiameseMV3模型，参数为class_dim和use_custom_relu
def siamese_mv3(class_dim, use_custom_relu):
    model = SiameseMV3(
        scale=0.5,
        model_name="large",
        class_dim=class_dim,
        use_custom_relu=use_custom_relu)
    return model

# 定义一个函数，根据config中的'model_type'创建相应的模型
def build_model(config):
    model_type = config['model_type']
    # 如果模型类型是分类模型
    if model_type == "cls":
        # 从配置文件中获取类别数和是否使用自定义的激活函数
        class_dim = config['MODEL']['class_dim']
        use_custom_relu = config['MODEL']['use_custom_relu']
        # 如果配置文件中包含'siamese'并且值为True，则创建siamese_mv3模型，否则创建MobileNetV3_large_x0_5模型
        if 'siamese' in config['MODEL'] and config['MODEL']['siamese'] is True:
            model = siamese_mv3(
                class_dim=class_dim, use_custom_relu=use_custom_relu)
        else:
            model = MobileNetV3_large_x0_5(
                class_dim=class_dim, use_custom_relu=use_custom_relu)

    # 如果模型类型是蒸馏分类模型
    elif model_type == "cls_distill":
        # 从配置文件中获取类别数和是否使用自定义的激活函数
        class_dim = config['MODEL']['class_dim']
        use_custom_relu = config['MODEL']['use_custom_relu']
        # 创建distillmv3_large_x0_5模型
        model = distillmv3_large_x0_5(
            class_dim=class_dim, use_custom_relu=use_custom_relu)

    # 如果模型类型是多优化器蒸馏分类模型
    elif model_type == "cls_distill_multiopt":
        # 从配置文件中获取类别数和是否使用自定义的激活函数
        class_dim = config['MODEL']['class_dim']
        use_custom_relu = config['MODEL']['use_custom_relu']
        # 创建类别数为100的distillmv3_large_x0_5模型
        model = distillmv3_large_x0_5(
            class_dim=100, use_custom_relu=use_custom_relu)
    else:
        # 如果模型类型不在指定范围内，则抛出数值错误
        raise ValueError("model_type should be one of ['cls', 'cls_distill', 'cls_distill_multiopt']")

    # 返回创建的模型
    return model
```