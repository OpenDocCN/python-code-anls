# `.\PaddleOCR\ppocr\modeling\necks\sast_fpn.py`

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
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”提供的，没有任何明示或暗示的保证或条件。
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
            padding=(kernel_size - 1) // 2,
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

    # 前向传播函数
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# 定义反卷积和批归一化层
class DeConvBNLayer(nn.Layer):
    # 定义反卷积和批归一化层的类
    class DeConvBNLayer(nn.Layer):
        # 初始化函数，设置各种参数
        def __init__(self,
                     in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     groups=1,
                     if_act=True,
                     act=None,
                     name=None):
            # 调用父类的初始化函数
            super(DeConvBNLayer, self).__init__()
            # 是否需要激活函数
            self.if_act = if_act
            # 激活函数类型
            self.act = act
            # 创建反卷积层对象
            self.deconv = nn.Conv2DTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
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
        
        # 前向传播函数
        def forward(self, x):
            # 反卷积操作
            x = self.deconv(x)
            # 批归一化操作
            x = self.bn(x)
            # 返回结果
            return x
class FPN_Up_Fusion(nn.Layer):
    # 定义 FPN_Up_Fusion 类，继承自 nn.Layer
    def __init__(self, in_channels):
        # 初始化函数，接受输入通道数列表
        super(FPN_Up_Fusion, self).__init__()
        # 调用父类的初始化函数

        # 将输入通道数列表反转
        in_channels = in_channels[::-1]
        # 定义输出通道数列表
        out_channels = [256, 256, 192, 192, 128]
                
        # 定义不同层级的卷积层
        self.h0_conv = ConvBNLayer(in_channels[0], out_channels[0], 1, 1, act=None, name='fpn_up_h0')
        self.h1_conv = ConvBNLayer(in_channels[1], out_channels[1], 1, 1, act=None, name='fpn_up_h1')
        self.h2_conv = ConvBNLayer(in_channels[2], out_channels[2], 1, 1, act=None, name='fpn_up_h2')
        self.h3_conv = ConvBNLayer(in_channels[3], out_channels[3], 1, 1, act=None, name='fpn_up_h3')
        self.h4_conv = ConvBNLayer(in_channels[4], out_channels[4], 1, 1, act=None, name='fpn_up_h4')

        # 定义不同层级的反卷积层
        self.g0_conv = DeConvBNLayer(out_channels[0], out_channels[1], 4, 2, act=None, name='fpn_up_g0')

        self.g1_conv = nn.Sequential(
            ConvBNLayer(out_channels[1], out_channels[1], 3, 1, act='relu', name='fpn_up_g1_1'),
            DeConvBNLayer(out_channels[1], out_channels[2], 4, 2, act=None, name='fpn_up_g1_2')
        )
        self.g2_conv = nn.Sequential(
            ConvBNLayer(out_channels[2], out_channels[2], 3, 1, act='relu', name='fpn_up_g2_1'),
            DeConvBNLayer(out_channels[2], out_channels[3], 4, 2, act=None, name='fpn_up_g2_2')
        )
        self.g3_conv = nn.Sequential(
            ConvBNLayer(out_channels[3], out_channels[3], 3, 1, act='relu', name='fpn_up_g3_1'),
            DeConvBNLayer(out_channels[3], out_channels[4], 4, 2, act=None, name='fpn_up_g3_2')
        )

        # 定义最终的融合卷积层
        self.g4_conv = nn.Sequential(
            ConvBNLayer(out_channels[4], out_channels[4], 3, 1, act='relu', name='fpn_up_fusion_1'),
            ConvBNLayer(out_channels[4], out_channels[4], 1, 1, act=None, name='fpn_up_fusion_2')
        )

    def _add_relu(self, x1, x2):
        # 定义一个私有函数，实现两个张量的加法和 ReLU 激活
        x = paddle.add(x=x1, y=x2)
        x = F.relu(x)
        return x
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 从输入 x 中取出索引为 2 开始的元素，然后进行反转
        f = x[2:][::-1]
        # 使用第一个元素经过 h0_conv 进行卷积操作
        h0 = self.h0_conv(f[0])
        # 使用第二个元素经过 h1_conv 进行卷积操作
        h1 = self.h1_conv(f[1])
        # 使用第三个元素经过 h2_conv 进行卷积操作
        h2 = self.h2_conv(f[2])
        # 使用第四个元素经过 h3_conv 进行卷积操作
        h3 = self.h3_conv(f[3])
        # 使用第五个元素经过 h4_conv 进行卷积操作
        h4 = self.h4_conv(f[4])

        # 使用 h0 经过 g0_conv 进行卷积操作
        g0 = self.g0_conv(h0)
        # 将 g0 和 h1 经过 _add_relu 函数相加并进行 ReLU 激活，然后再经过 g1_conv 进行卷积操作
        g1 = self._add_relu(g0, h1)
        g1 = self.g1_conv(g1)
        # 将 g1 和 h2 经过 _add_relu 函数相加并进行 ReLU 激活，然后再经过 g2_conv 进行卷积操作
        g2 = self.g2_conv(self._add_relu(g1, h2))
        # 将 g2 和 h3 经过 _add_relu 函数相加并进行 ReLU 激活，然后再经过 g3_conv 进行卷积操作
        g3 = self.g3_conv(self._add_relu(g2, h3))
        # 将 g3 和 h4 经过 _add_relu 函数相加并进行 ReLU 激活，然后再经过 g4_conv 进行卷积操作
        g4 = self.g4_conv(self._add_relu(g3, h4))

        # 返回最终结果 g4
        return g4
# 定义 FPN 下采样融合模块类，继承自 nn.Layer
class FPN_Down_Fusion(nn.Layer):
    # 初始化函数，接受输入通道数列表，并定义输出通道数列表
    def __init__(self, in_channels):
        super(FPN_Down_Fusion, self).__init__()
        out_channels = [32, 64, 128]

        # 定义三个卷积层，分别处理不同输入通道的特征
        self.h0_conv = ConvBNLayer(in_channels[0], out_channels[0], 3, 1, act=None, name='fpn_down_h0')
        self.h1_conv = ConvBNLayer(in_channels[1], out_channels[1], 3, 1, act=None, name='fpn_down_h1')
        self.h2_conv = ConvBNLayer(in_channels[2], out_channels[2], 3, 1, act=None, name='fpn_down_h2')

        # 定义两个卷积层，用于融合不同层级的特征
        self.g0_conv = ConvBNLayer(out_channels[0], out_channels[1], 3, 2, act=None, name='fpn_down_g0')

        self.g1_conv = nn.Sequential(
            ConvBNLayer(out_channels[1], out_channels[1], 3, 1, act='relu', name='fpn_down_g1_1'),
            ConvBNLayer(out_channels[1], out_channels[2], 3, 2, act=None, name='fpn_down_g1_2')            
        )

        self.g2_conv = nn.Sequential(
            ConvBNLayer(out_channels[2], out_channels[2], 3, 1, act='relu', name='fpn_down_fusion_1'),
            ConvBNLayer(out_channels[2], out_channels[2], 1, 1, act=None, name='fpn_down_fusion_2')            
        )

    # 前向传播函数，接受输入特征 x，进行特征融合操作
    def forward(self, x):
        # 将输入特征切片为三部分
        f = x[:3]
        # 分别对三部分特征进行卷积操作
        h0 = self.h0_conv(f[0])
        h1 = self.h1_conv(f[1])
        h2 = self.h2_conv(f[2])
        # 对第一部分特征进行下采样
        g0 = self.g0_conv(h0)
        # 将下采样后的特征与第二部分特征相加，并进行激活函数处理
        g1 = paddle.add(x=g0, y=h1)
        g1 = F.relu(g1)
        # 对相加后的特征进行卷积操作
        g1 = self.g1_conv(g1)
        # 将卷积后的特征与第三部分特征相加，并进行激活函数处理
        g2 = paddle.add(x=g1, y=h2)
        g2 = F.relu(g2)
        # 对相加后的特征进行卷积操作
        g2 = self.g2_conv(g2)
        # 返回融合后的特征
        return g2


class Cross_Attention(nn.Layer):
    # 初始化交叉注意力模块，设置输入通道数
    def __init__(self, in_channels):
        # 调用父类的初始化方法
        super(Cross_Attention, self).__init__()
        # 创建 theta 卷积层，用于计算 theta
        self.theta_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_theta')
        # 创建 phi 卷积层，用于计算 phi
        self.phi_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_phi')
        # 创建 g 卷积层，用于计算 g
        self.g_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act='relu', name='f_g')

        # 创建 fh_weight 卷积层，用于计算 f_h 的权重
        self.fh_weight_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fh_weight')
        # 创建 fh_sc 卷积层，用于计算 f_h 的缩放
        self.fh_sc_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fh_sc')

        # 创建 fv_weight 卷积层，用于计算 f_v 的权重
        self.fv_weight_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fv_weight')
        # 创建 fv_sc 卷积层，用于计算 f_v 的缩放
        self.fv_sc_conv = ConvBNLayer(in_channels, in_channels, 1, 1, act=None, name='fv_sc')

        # 创建 f_attn 卷积层，用于计算注意力加权后的特征
        self.f_attn_conv = ConvBNLayer(in_channels * 2, in_channels, 1, 1, act='relu', name='f_attn')

    # 计算特征权重
    def _cal_fweight(self, f, shape):
        f_theta, f_phi, f_g = f
        # 将 f_theta 展平
        f_theta = paddle.transpose(f_theta, [0, 2, 3, 1])
        f_theta = paddle.reshape(f_theta, [shape[0] * shape[1], shape[2], 128])
        # 将 f_phi 展平
        f_phi = paddle.transpose(f_phi, [0, 2, 3, 1])
        f_phi = paddle.reshape(f_phi, [shape[0] * shape[1], shape[2], 128])
        # 将 f_g 展平
        f_g = paddle.transpose(f_g, [0, 2, 3, 1])
        f_g = paddle.reshape(f_g, [shape[0] * shape[1], shape[2], 128])
        # 计算相关性
        f_attn = paddle.matmul(f_theta, paddle.transpose(f_phi, [0, 2, 1]))
        # 缩放
        f_attn = f_attn / (128**0.5)
        f_attn = F.softmax(f_attn)
        # 加权求和
        f_weight = paddle.matmul(f_attn, f_g)
        f_weight = paddle.reshape(
            f_weight, [shape[0], shape[1], shape[2], 128])
        return f_weight
    # 前向传播函数，接受一个通用特征图作为输入
    def forward(self, f_common):
        # 获取通用特征图的形状
        f_shape = paddle.shape(f_common)
        # 打印特征图的形状
        # print('f_shape: ', f_shape)

        # 使用 theta 卷积层处理通用特征图
        f_theta = self.theta_conv(f_common)
        # 使用 phi 卷积层处理通用特征图
        f_phi = self.phi_conv(f_common)
        # 使用 g 卷积层处理通用特征图
        f_g = self.g_conv(f_common)

        ######## horizon ########
        # 计算水平方向的权重
        fh_weight = self._cal_fweight([f_theta, f_phi, f_g], 
                                        [f_shape[0], f_shape[2], f_shape[3]])
        # 调整权重的维度顺序
        fh_weight = paddle.transpose(fh_weight, [0, 3, 1, 2])
        # 使用水平方向权重卷积层处理权重
        fh_weight = self.fh_weight_conv(fh_weight)
        #short cut
        # 使用水平方向的短连接卷积层处理通用特征图
        fh_sc = self.fh_sc_conv(f_common)
        # 对水平方向的特征图进行 ReLU 激活
        f_h = F.relu(fh_weight + fh_sc)

        ######## vertical ########
        # 调整 theta 特征图的维度顺序
        fv_theta = paddle.transpose(f_theta, [0, 1, 3, 2])
        # 调整 phi 特征图的维度顺序
        fv_phi = paddle.transpose(f_phi, [0, 1, 3, 2])
        # 调整 g 特征图的维度顺序
        fv_g = paddle.transpose(f_g, [0, 1, 3, 2])
        # 计算垂直方向的权重
        fv_weight = self._cal_fweight([fv_theta, fv_phi, fv_g], 
                                        [f_shape[0], f_shape[3], f_shape[2]])
        # 调整权重的维度顺序
        fv_weight = paddle.transpose(fv_weight, [0, 3, 2, 1])
        # 使用垂直方向权重卷积层处理权重
        fv_weight = self.fv_weight_conv(fv_weight)
        #short cut
        # 使用垂直方向的短连接卷积层处理通用特征图
        fv_sc = self.fv_sc_conv(f_common)
        # 对垂直方向的特征图进行 ReLU 激活
        f_v = F.relu(fv_weight + fv_sc)

        ######## merge ########
        # 沿着通道维度拼接水平和垂直方向的特征图
        f_attn = paddle.concat([f_h, f_v], axis=1)
        # 使用注意力卷积层处理拼接后的特征图
        f_attn = self.f_attn_conv(f_attn)
        # 返回注意力特征图
        return f_attn
class SASTFPN(nn.Layer):
    # 定义 SASTFPN 类，继承自 nn.Layer
    def __init__(self, in_channels, with_cab=False, **kwargs):
        # 初始化函数，接受输入通道数和是否使用 CAB 参数
        super(SASTFPN, self).__init__()
        # 调用父类的初始化函数
        self.in_channels = in_channels
        # 设置输入通道数
        self.with_cab = with_cab
        # 设置是否使用 CAB
        self.FPN_Down_Fusion = FPN_Down_Fusion(self.in_channels)
        # 创建 FPN_Down_Fusion 对象
        self.FPN_Up_Fusion = FPN_Up_Fusion(self.in_channels)
        # 创建 FPN_Up_Fusion 对象
        self.out_channels = 128
        # 设置输出通道数为 128
        self.cross_attention = Cross_Attention(self.out_channels)
        # 创建 Cross_Attention 对象

    def forward(self, x):
        # 定义前向传播函数，接受输入 x
        #down fpn
        # 下采样 FPN
        f_down = self.FPN_Down_Fusion(x)

        #up fpn
        # 上采样 FPN
        f_up = self.FPN_Up_Fusion(x)

        #fusion
        # 融合下采样和上采样结果
        f_common = paddle.add(x=f_down, y=f_up)
        f_common = F.relu(f_common)

        if self.with_cab:
            # 如果使用 CAB
            # print('enhence f_common with CAB.')
            # 打印信息
            f_common = self.cross_attention(f_common)

        return f_common
        # 返回融合后的结果
```