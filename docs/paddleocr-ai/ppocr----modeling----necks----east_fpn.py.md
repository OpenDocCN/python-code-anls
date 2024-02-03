# `.\PaddleOCR\ppocr\modeling\necks\east_fpn.py`

```
# 版权声明
#
# 版权所有 (c) 2019 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”基础分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

# 定义卷积和批归一化层
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        # 创建卷积层
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        # 创建批归一化层
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),
            moving_mean_name="bn_" + name + "_mean",
            moving_variance_name="bn_" + name + "_variance")

    def forward(self, x):
        # 前向传播：卷积 -> 批归一化
        x = self.conv(x)
        x = self.bn(x)
        return x

# 定义反卷积和批归一化层
class DeConvBNLayer(nn.Layer):
    # 定义 DeConvBNLayer 类，继承自 nn.Layer
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        # 调用父类的初始化方法
        super(DeConvBNLayer, self).__init__()
        # 初始化是否需要激活和激活函数
        self.if_act = if_act
        self.act = act
        # 创建反卷积层对象
        self.deconv = nn.Conv2DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        # 创建批归一化层对象
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),
            moving_mean_name="bn_" + name + "_mean",
            moving_variance_name="bn_" + name + "_variance")
    
    # 定义前向传播方法
    def forward(self, x):
        # 对输入进行反卷积操作
        x = self.deconv(x)
        # 对反卷积结果进行批归一化操作
        x = self.bn(x)
        # 返回处理后的结果
        return x
class EASTFPN(nn.Layer):
    # 定义一个名为EASTFPN的类，继承自nn.Layer类

    def forward(self, x):
        # 定义一个名为forward的方法，接受输入参数x

        f = x[::-1]
        # 将输入x进行逆序操作，赋值给变量f

        h = f[0]
        # 取f中的第一个元素，赋值给变量h

        g = self.g0_deconv(h)
        # 使用self.g0_deconv对h进行处理，得到结果赋值给变量g

        h = paddle.concat([g, f[1]], axis=1)
        # 将g和f中的第二个元素在axis=1的方向上拼接，赋值给变量h

        h = self.h1_conv(h)
        # 使用self.h1_conv对h进行处理，得到结果赋值给变量h

        g = self.g1_deconv(h)
        # 使用self.g1_deconv对h进行处理，得到结果赋值给变量g

        h = paddle.concat([g, f[2]], axis=1)
        # 将g和f中的第三个元素在axis=1的方向上拼接，赋值给变量h

        h = self.h2_conv(h)
        # 使用self.h2_conv对h进行处理，得到结果赋值给变量h

        g = self.g2_deconv(h)
        # 使用self.g2_deconv对h进行处理，得到结果赋值给变量g

        h = paddle.concat([g, f[3]], axis=1)
        # 将g和f中的第四个元素在axis=1的方向上拼接，赋值给变量h

        h = self.h3_conv(h)
        # 使用self.h3_conv对h进行处理，得到结果赋值给变量h

        g = self.g3_conv(h)
        # 使用self.g3_conv对h进行处理，得到结果赋值给变量g

        return g
        # 返回变量g作为方法的输出结果
```