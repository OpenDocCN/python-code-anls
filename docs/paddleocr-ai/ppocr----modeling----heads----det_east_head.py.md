# `.\PaddleOCR\ppocr\modeling\heads\det_east_head.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发在“按原样”基础上，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制
#
# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

# 定义 ConvBNLayer 类，继承自 nn.Layer
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
        # 创建卷积层对象
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        # 创建 BatchNorm 层对象
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(name="bn_" + name + "_scale"),
            bias_attr=ParamAttr(name="bn_" + name + "_offset"),
            moving_mean_name="bn_" + name + "_mean",
            moving_variance_name="bn_" + name + "_variance")

    # 前向传播函数
    def forward(self, x):
        # 卷积操作
        x = self.conv(x)
        # BatchNorm 操作
        x = self.bn(x)
        return x
class EASTHead(nn.Layer):
    """
    定义一个名为EASTHead的类，继承自nn.Layer
    """
    def __init__(self, in_channels, model_name, **kwargs):
        """
        初始化方法，接受输入通道数和模型名称作为参数
        """
        super(EASTHead, self).__init__()
        # 将模型名称保存到实例变量中
        self.model_name = model_name
        # 根据模型名称选择不同的输出通道数配置
        if self.model_name == "large":
            num_outputs = [128, 64, 1, 8]
        else:
            num_outputs = [64, 32, 1, 8]

        # 创建第一个检测卷积层
        self.det_conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=num_outputs[0],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="det_head1")
        # 创建第二个检测卷积层
        self.det_conv2 = ConvBNLayer(
            in_channels=num_outputs[0],
            out_channels=num_outputs[1],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="det_head2")
        # 创建得分卷积层
        self.score_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[2],
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name="f_score")
        # 创建几何信息卷积层
        self.geo_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[3],
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name="f_geo")

    def forward(self, x, targets=None):
        """
        前向传播方法，接受输入数据x和目标targets作为参数
        """
        # 通过第一个检测卷积层得到特征图
        f_det = self.det_conv1(x)
        # 通过第二个检测卷积层得到特征图
        f_det = self.det_conv2(f_det)
        # 通过得分卷积层得到得分图
        f_score = self.score_conv(f_det)
        # 对得分图进行sigmoid激活
        f_score = F.sigmoid(f_score)
        # 通过几何信息卷积层得到几何信息图
        f_geo = self.geo_conv(f_det)
        # 对几何信息图进行sigmoid激活，并进行后续处理
        f_geo = (F.sigmoid(f_geo) - 0.5) * 2 * 800

        # 将得分图和几何信息图组成预测结果字典
        pred = {'f_score': f_score, 'f_geo': f_geo}
        # 返回预测结果字典
        return pred
```