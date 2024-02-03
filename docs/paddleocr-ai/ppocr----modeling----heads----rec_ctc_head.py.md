# `.\PaddleOCR\ppocr\modeling\heads\rec_ctc_head.py`

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

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# 导入 Paddle 库
import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F

# 定义函数，用于获取参数和偏置的属性
def get_para_bias_attr(l2_decay, k):
    # 创建 L2 正则化器
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    # 计算初始化参数
    stdv = 1.0 / math.sqrt(k * 1.0)
    # 创建均匀分布初始化器
    initializer = nn.initializer.Uniform(-stdv, stdv)
    # 创建参数属性对象，包括正则化器和初始化器
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    # 返回参数和偏置属性列表
    return [weight_attr, bias_attr]

# 定义 CTCHead 类，继承自 nn.Layer
class CTCHead(nn.Layer):
    # 初始化 CTCHead 类
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        # 调用父类的初始化方法
        super(CTCHead, self).__init__()
        # 如果中间通道数未指定，则创建一个线性层
        if mid_channels is None:
            # 获取权重和偏置属性
            weight_attr, bias_attr = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels)
            # 创建线性层
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                weight_attr=weight_attr,
                bias_attr=bias_attr)
        else:
            # 获取权重和偏置属性
            weight_attr1, bias_attr1 = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels)
            # 创建第一个线性层
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                weight_attr=weight_attr1,
                bias_attr=bias_attr1)

            # 获取权重和偏置属性
            weight_attr2, bias_attr2 = get_para_bias_attr(
                l2_decay=fc_decay, k=mid_channels)
            # 创建第二个线性层
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                weight_attr=weight_attr2,
                bias_attr=bias_attr2)
        # 保存输出通道数和中间通道数
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    # 前向传播函数
    def forward(self, x, targets=None):
        # 如果中间通道数未指定，则使用第一个线性层
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            # 使用第一个线性层和第二个线性层
            x = self.fc1(x)
            predicts = self.fc2(x)

        # 如果需要返回特征，则返回特征和预测结果
        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        # 如果不是训练阶段，则对预测结果进行 softmax 处理
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
            result = predicts

        return result
```