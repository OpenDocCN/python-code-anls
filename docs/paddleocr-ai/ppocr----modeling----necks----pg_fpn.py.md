# `.\PaddleOCR\ppocr\modeling\necks\pg_fpn.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发
# 没有任何明示或暗示的担保或条件，无论是明示还是暗示
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 Paddle 库
import paddle
# 从 Paddle 库中导入 nn 模块
from paddle import nn
# 从 Paddle 库中导入 nn 模块下的 functional 模块
import paddle.nn.functional as F
# 从 Paddle 库中导入 ParamAttr 类
from paddle import ParamAttr

# 定义 ConvBNLayer 类，继承自 nn.Layer 类
class ConvBNLayer(nn.Layer):
    # 初始化函数，定义卷积层和批归一化层
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                 name=None):
        # 调用父类的初始化函数
        super(ConvBNLayer, self).__init__()

        # 是否使用 VD 模式
        self.is_vd_mode = is_vd_mode
        # 创建平均池化层对象
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        # 创建卷积层对象
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        # 根据卷积层名称确定批归一化层名称
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        # 创建批归一化层对象
        self._batch_norm = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            use_global_stats=False)

    # 前向传播函数，实现卷积和批归一化操作
    def forward(self, inputs):
        # 卷积操作
        y = self._conv(inputs)
        # 批归一化操作
        y = self._batch_norm(y)
        return y
# 定义反卷积层和批归一化层的类
class DeConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,  # 输入通道数
                 out_channels,  # 输出通道数
                 kernel_size=4,  # 卷积核大小
                 stride=2,  # 步长
                 padding=1,  # 填充
                 groups=1,  # 分组卷积
                 if_act=True,  # 是否使用激活函数
                 act=None,  # 激活函数类型
                 name=None):  # 层的名称
        super(DeConvBNLayer, self).__init__()

        self.if_act = if_act  # 是否使用激活函数
        self.act = act  # 激活函数类型
        # 创建反卷积层
        self.deconv = nn.Conv2DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),  # 权重参数属性
            bias_attr=False)  # 不使用偏置参数
        # 创建批归一化层
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),  # 参数属性
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),  # 偏置属性
            moving_mean_name="bn_" + name + "_mean",  # 移动平均值的名称
            moving_variance_name="bn_" + name + "_variance",  # 移动方差的名称
            use_global_stats=False)  # 不使用全局统计信息

    def forward(self, x):
        # 反卷积操作
        x = self.deconv(x)
        # 批归一化操作
        x = self.bn(x)
        return x


class PGFPN(nn.Layer):
    # 定义前向传播函数，接收输入 x
    def forward(self, x):
        # 将输入 x 拆分为 c0, c1, c2, c3, c4, c5, c6
        c0, c1, c2, c3, c4, c5, c6 = x
        # FPN_Down_Fusion 部分
        f = [c0, c1, c2]
        g = [None, None, None]
        h = [None, None, None]
        # 对 f 中的每个元素进行卷积和批归一化操作，存储到 h 中
        h[0] = self.conv_bn_layer_1(f[0])
        h[1] = self.conv_bn_layer_2(f[1])
        h[2] = self.conv_bn_layer_3(f[2])

        # 对 h 中的每个元素进行卷积和批归一化操作，存储到 g 中
        g[0] = self.conv_bn_layer_4(h[0])
        g[1] = paddle.add(g[0], h[1])
        g[1] = F.relu(g[1])
        g[1] = self.conv_bn_layer_5(g[1])
        g[1] = self.conv_bn_layer_6(g[1])

        g[2] = paddle.add(g[1], h[2])
        g[2] = F.relu(g[2])
        g[2] = self.conv_bn_layer_7(g[2])
        f_down = self.conv_bn_layer_8(g[2])

        # FPN UP Fusion 部分
        f1 = [c6, c5, c4, c3, c2]
        g = [None, None, None, None, None]
        h = [None, None, None, None, None]
        # 对 f1 中的每个元素进行卷积操作，存储到 h 中
        h[0] = self.conv_h0(f1[0])
        h[1] = self.conv_h1(f1[1])
        h[2] = self.conv_h2(f1[2])
        h[3] = self.conv_h3(f1[3])
        h[4] = self.conv_h4(f1[4])

        # 对 h 中的每个元素进行反卷积操作，存储到 g 中
        g[0] = self.dconv0(h[0])
        g[1] = paddle.add(g[0], h[1])
        g[1] = F.relu(g[1])
        g[1] = self.conv_g1(g[1])
        g[1] = self.dconv1(g[1])

        g[2] = paddle.add(g[1], h[2])
        g[2] = F.relu(g[2])
        g[2] = self.conv_g2(g[2])
        g[2] = self.dconv2(g[2])

        g[3] = paddle.add(g[2], h[3])
        g[3] = F.relu(g[3])
        g[3] = self.conv_g3(g[3])
        g[3] = self.dconv3(g[3])

        g[4] = paddle.add(x=g[3], y=h[4])
        g[4] = F.relu(g[4])
        g[4] = self.conv_g4(g[4])
        f_up = self.convf(g[4])
        f_common = paddle.add(f_down, f_up)
        f_common = F.relu(f_common)
        # 返回融合后的结果
        return f_common
```