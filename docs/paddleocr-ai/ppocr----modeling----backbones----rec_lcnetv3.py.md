# `.\PaddleOCR\ppocr\modeling\backbones\rec_lcnetv3.py`

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
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
# 导入 paddle 中的神经网络模块
import paddle.nn as nn
# 导入 paddle 中的神经网络函数模块
import paddle.nn.functional as F
# 从 paddle 中导入 ParamAttr
from paddle import ParamAttr
# 从 paddle.nn.initializer 中导入 Constant 和 KaimingNormal
from paddle.nn.initializer import Constant, KaimingNormal
# 从 paddle.nn 中导入 AdaptiveAvgPool2D, BatchNorm2D, Conv2D, Dropout, Hardsigmoid, Hardswish, Identity, Linear, ReLU
from paddle.nn import AdaptiveAvgPool2D, BatchNorm2D, Conv2D, Dropout, Hardsigmoid, Hardswish, Identity, Linear, ReLU
# 从 paddle.regularizer 中导入 L2Decay
from paddle.regularizer import L2Decay

# 目标检测网络配置
NET_CONFIG_det = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5":
    [[3, 128, 256, 2, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False],
     [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True],
                [5, 512, 512, 1, False], [5, 512, 512, 1, False]]
}

# 文本识别网络配置
NET_CONFIG_rec = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, (2, 1), False], [3, 128, 128, 1, False]],
    "blocks5":
    [[3, 128, 256, (1, 2), False], [5, 256, 256, 1, False],
     [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    # 定义一个包含多个子列表的列表，每个子列表表示一个卷积块的配置信息
    "blocks6": [[5, 256, 512, (2, 1), True],  # 第一个卷积块的配置信息：卷积核大小为5，输入通道数为256，输出通道数为512，步长为(2, 1)，是否使用激活函数为True
                [5, 512, 512, 1, True],  # 第二个卷积块的配置信息：卷积核大小为5，输入通道数为512，输出通道数为512，步长为1，是否使用激活函数为True
                [5, 512, 512, (2, 1), False],  # 第三个卷积块的配置信息：卷积核大小为5，输入通道数为512，输出通道数为512，步长为(2, 1)，是否使用激活函数为False
                [5, 512, 512, 1, False]]  # 第四个卷积块的配置信息：卷积核大小为5，输入通道数为512，输出通道数为512，步长为1，是否使用激活函数为False
# 定义一个函数，用于确保输入值是可被除数整除的，且大于等于给定最小值
def make_divisible(v, divisor=16, min_value=None):
    # 如果没有指定最小值，则默认为除数
    if min_value is None:
        min_value = divisor
    # 对输入值进行处理，确保是可被除数整除的
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 如果处理后的值小于原值的90%，则加上除数
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 定义一个可学习的仿射块
class LearnableAffineBlock(nn.Layer):
    def __init__(self, scale_value=1.0, bias_value=0.0, lr_mult=1.0,
                 lab_lr=0.1):
        super().__init__()
        # 创建可学习的缩放参数
        self.scale = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=scale_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        self.add_parameter("scale", self.scale)
        # 创建可学习的偏置参数
        self.bias = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=bias_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        # 返回经过缩放和偏置的结果
        return self.scale * x + self.bias

# 定义一个包含卷积和批归一化的层
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 lr_mult=1.0):
        super().__init__()
        # 创建卷积层
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), learning_rate=lr_mult),
            bias_attr=False)
        # 创建批归一化层
        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult))

    def forward(self, x):
        # 经过卷积和批归一化后返回结果
        x = self.conv(x)
        x = self.bn(x)
        return x

# 定义一个激活函数层
class Act(nn.Layer):
    # 初始化函数，设置激活函数类型、学习率倍数和标签学习率
    def __init__(self, act="hswish", lr_mult=1.0, lab_lr=0.1):
        # 调用父类的初始化函数
        super().__init__()
        # 根据激活函数类型选择不同的激活函数对象
        if act == "hswish":
            self.act = Hardswish()
        else:
            # 如果激活函数类型不是"hswish"，则必须是"relu"
            assert act == "relu"
            self.act = ReLU()
        # 创建可学习的仿射块对象，设置学习率倍数和标签学习率
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)

    # 前向传播函数，对输入数据进行激活函数处理和仿射变换
    def forward(self, x):
        # 先使用激活函数处理输入数据，再通过仿射块进行变换
        return self.lab(self.act(x))
# 定义一个可学习的表示层类
class LearnableRepLayer(nn.Layer):
    # 初始化函数，设置各种参数
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 num_conv_branches=1,
                 lr_mult=1.0,
                 lab_lr=0.1):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化一些属性
        self.is_repped = False
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        # 如果输出通道数等于输入通道数且步长为1，则设置identity为一个批归一化层，否则为None
        self.identity = BatchNorm2D(
            num_features=in_channels,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult)
        ) if out_channels == in_channels and stride == 1 else None

        # 创建一个包含多个卷积层的列表，每个卷积层都有相同的参数
        self.conv_kxk = nn.LayerList([
            ConvBNLayer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                groups=groups,
                lr_mult=lr_mult) for _ in range(self.num_conv_branches)
        ])

        # 如果卷积核大小大于1，则设置conv_1x1为一个卷积层，否则为None
        self.conv_1x1 = ConvBNLayer(
            in_channels,
            out_channels,
            1,
            stride,
            groups=groups,
            lr_mult=lr_mult) if kernel_size > 1 else None

        # 创建一个可学习的仿射块
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)
        # 创建一个激活函数
        self.act = Act(lr_mult=lr_mult, lab_lr=lab_lr)
    # 前向传播函数，接收输入 x
    def forward(self, x):
        # 用于导出模型
        if self.is_repped:
            # 对输入 x 进行重参数化卷积操作，并经过激活函数 lab
            out = self.lab(self.reparam_conv(x))
            # 如果步长不为 2，则再次经过激活函数 act
            if self.stride != 2:
                out = self.act(out)
            # 返回处理后的结果
            return out

        # 初始化输出为 0
        out = 0
        # 如果存在 identity 函数，则将输入 x 经过 identity 函数处理后加到输出中
        if self.identity is not None:
            out += self.identity(x)

        # 如果存在 conv_1x1 函数，则将输入 x 经过 conv_1x1 函数处理后加到输出中
        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)

        # 遍历 conv_kxk 列表中的卷积函数，将输入 x 经过每个卷积函数处理后加到输出中
        for conv in self.conv_kxk:
            out += conv(x)

        # 对输出进行激活函数 lab 处理
        out = self.lab(out)
        # 如果步长不为 2，则再次经过激活函数 act
        if self.stride != 2:
            out = self.act(out)
        # 返回处理后的结果
        return out

    # 重参数化函数
    def rep(self):
        # 如果已经重参数化过，则直接返回
        if self.is_repped:
            return
        # 获取卷积核和偏置
        kernel, bias = self._get_kernel_bias()
        # 创建重参数化卷积层
        self.reparam_conv = Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)
        # 设置卷积层的权重和偏置
        self.reparam_conv.weight.set_value(kernel)
        self.reparam_conv.bias.set_value(bias)
        # 标记已经重参数化
        self.is_repped = True

    # 将 1x1 卷积核填充为 kxk 大小的函数
    def _pad_kernel_1x1_to_kxk(self, kernel1x1, pad):
        # 如果输入不是 paddle.Tensor 类型，则返回 0
        if not isinstance(kernel1x1, paddle.Tensor):
            return 0
        else:
            # 对输入的 1x1 卷积核进行填充操作，返回填充后的结果
            return nn.functional.pad(kernel1x1, [pad, pad, pad, pad])
    # 获取卷积核和偏置项
    def _get_kernel_bias(self):
        # 融合卷积层1x1的卷积核和偏置项
        kernel_conv_1x1, bias_conv_1x1 = self._fuse_bn_tensor(self.conv_1x1)
        # 将卷积核1x1填充到kxk大小
        kernel_conv_1x1 = self._pad_kernel_1x1_to_kxk(kernel_conv_1x1,
                                                      self.kernel_size // 2)

        # 融合恒等映射的卷积核和偏置项
        kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        # 初始化卷积核和偏置项的和为0
        kernel_conv_kxk = 0
        bias_conv_kxk = 0
        # 遍历kxk卷积层，融合卷积核和偏置项并累加
        for conv in self.conv_kxk:
            kernel, bias = self._fuse_bn_tensor(conv)
            kernel_conv_kxk += kernel
            bias_conv_kxk += bias

        # 计算重参数化后的卷积核和偏置项
        kernel_reparam = kernel_conv_kxk + kernel_conv_1x1 + kernel_identity
        bias_reparam = bias_conv_kxk + bias_conv_1x1 + bias_identity
        # 返回重参数化后的卷积核和偏置项
        return kernel_reparam, bias_reparam
    # 合并 BatchNorm 层和卷积层的张量
    def _fuse_bn_tensor(self, branch):
        # 如果分支为空，则返回 0, 0
        if not branch:
            return 0, 0
        # 如果分支是 ConvBNLayer 类型
        elif isinstance(branch, ConvBNLayer):
            # 获取卷积核
            kernel = branch.conv.weight
            # 获取 BatchNorm 层的 running_mean
            running_mean = branch.bn._mean
            # 获取 BatchNorm 层的 running_var
            running_var = branch.bn._variance
            # 获取 BatchNorm 层的 gamma
            gamma = branch.bn.weight
            # 获取 BatchNorm 层的 beta
            beta = branch.bn.bias
            # 获取 BatchNorm 层的 epsilon
            eps = branch.bn._epsilon
        else:
            # 断言分支是 BatchNorm2D 类型
            assert isinstance(branch, BatchNorm2D)
            # 如果没有 id_tensor 属性，则创建一个
            if not hasattr(self, 'id_tensor'):
                # 计算输入维度
                input_dim = self.in_channels // self.groups
                # 创建一个全零张量作为卷积核
                kernel_value = paddle.zeros(
                    (self.in_channels, input_dim, self.kernel_size,
                     self.kernel_size),
                    dtype=branch.weight.dtype)
                # 将对角线元素设置为 1
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            # 获取卷积核
            kernel = self.id_tensor
            # 获取 BatchNorm 层的 running_mean
            running_mean = branch._mean
            # 获取 BatchNorm 层的 running_var
            running_var = branch._variance
            # 获取 BatchNorm 层的 gamma
            gamma = branch.weight
            # 获取 BatchNorm 层的 beta
            beta = branch.bias
            # 获取 BatchNorm 层的 epsilon
            eps = branch._epsilon
        # 计算标准差
        std = (running_var + eps).sqrt()
        # 计算 t
        t = (gamma / std).reshape((-1, 1, 1, 1))
        # 返回合并后的张量和偏置
        return kernel * t, beta - running_mean * gamma / std
# 定义一个 SELayer 类，用于实现 Squeeze-and-Excitation 模块
class SELayer(nn.Layer):
    # 初始化函数，接受通道数、缩减比例和学习率倍数作为参数
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        # 创建一个自适应平均池化层，将输入特征图池化为1x1大小
        self.avg_pool = AdaptiveAvgPool2D(1)
        # 创建一个卷积层，用于降维
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        # 创建一个 ReLU 激活函数
        self.relu = ReLU()
        # 创建一个卷积层，用于升维
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        # 创建一个 Hardsigmoid 激活函数
        self.hardsigmoid = Hardsigmoid()

    # 前向传播函数，接受输入 x，返回经过 SE 模块处理后的特征
    def forward(self, x):
        # 保存输入特征，用于残差连接
        identity = x
        # 平均池化
        x = self.avg_pool(x)
        # 第一个卷积层
        x = self.conv1(x)
        # ReLU 激活
        x = self.relu(x)
        # 第二个卷积层
        x = self.conv2(x)
        # Hardsigmoid 激活
        x = self.hardsigmoid(x)
        # 残差连接
        x = paddle.multiply(x=identity, y=x)
        # 返回处理后的特征
        return x


# 定义一个 LCNetV3Block 类
class LCNetV3Block(nn.Layer):
    # 初始化深度可分离卷积层
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dw_size,
                 use_se=False,
                 conv_kxk_num=4,
                 lr_mult=1.0,
                 lab_lr=0.1):
        # 调用父类的初始化方法
        super().__init__()
        # 是否使用 SE 模块
        self.use_se = use_se
        # 深度可分离卷积层
        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr)
        # 如果使用 SE 模块
        if use_se:
            # 添加 SE 模块
            self.se = SELayer(in_channels, lr_mult=lr_mult)
        # 点卷积层
        self.pw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr)
    
    # 前向传播函数
    def forward(self, x):
        # 深度可分离卷积
        x = self.dw_conv(x)
        # 如果使用 SE 模块
        if self.use_se:
            # 添加 SE 模块
            x = self.se(x)
        # 点卷积
        x = self.pw_conv(x)
        return x
class PPLCNetV3(nn.Layer):
    # 定义 PPLCNetV3 类，继承自 nn.Layer

    def forward(self, x):
        # 定义 forward 方法，接受输入 x

        out_list = []
        # 初始化一个空列表 out_list

        x = self.conv1(x)
        # 对输入 x 进行卷积操作，并将结果赋值给 x

        x = self.blocks2(x)
        # 对 x 进行 blocks2 操作，并将结果赋值给 x
        x = self.blocks3(x)
        # 对 x 进行 blocks3 操作，并将结果赋值给 x
        out_list.append(x)
        # 将 x 添加到 out_list 中
        x = self.blocks4(x)
        # 对 x 进行 blocks4 操作，并将结果赋值给 x
        out_list.append(x)
        # 将 x 添加到 out_list 中
        x = self.blocks5(x)
        # 对 x 进行 blocks5 操作，并将结果赋值给 x
        out_list.append(x)
        # 将 x 添加到 out_list 中
        x = self.blocks6(x)
        # 对 x 进行 blocks6 操作，并将结果赋值给 x
        out_list.append(x)
        # 将 x 添加到 out_list 中

        if self.det:
            # 如果 self.det 为真
            out_list[0] = self.layer_list[0](out_list[0])
            # 对 out_list 中的第一个元素进行 layer_list[0] 操作
            out_list[1] = self.layer_list[1](out_list[1])
            # 对 out_list 中的第二个元素进行 layer_list[1] 操作
            out_list[2] = self.layer_list[2](out_list[2])
            # 对 out_list 中的第三个元素进行 layer_list[2] 操作
            out_list[3] = self.layer_list[3](out_list[3])
            # 对 out_list 中的第四个元素进行 layer_list[3] 操作
            return out_list
            # 返回 out_list

        if self.training:
            # 如果处于训练状态
            x = F.adaptive_avg_pool2d(x, [1, 40])
            # 对 x 进行自适应平均池化操作
        else:
            x = F.avg_pool2d(x, [3, 2])
            # 对 x 进行平均池化操作
        return x
        # 返回 x
```