# `.\PaddleOCR\ppocr\modeling\heads\cls_head.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, ParamAttr
import paddle.nn.functional as F

# 定义一个类，用于构建分类网络
class ClsHead(nn.Layer):
    """
    Class orientation

    Args:

        params(dict): super parameters for build Class network
    """

    def __init__(self, in_channels, class_dim, **kwargs):
        super(ClsHead, self).__init__()
        # 创建一个自适应平均池化层
        self.pool = nn.AdaptiveAvgPool2D(1)
        # 计算权重的标准差
        stdv = 1.0 / math.sqrt(in_channels * 1.0)
        # 创建一个全连接层，设置权重和偏置属性
        self.fc = nn.Linear(
            in_channels,
            class_dim,
            weight_attr=ParamAttr(
                name="fc_0.w_0",
                initializer=nn.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_0.b_0"), )

    def forward(self, x, targets=None):
        # 对输入进行池化操作
        x = self.pool(x)
        # 重新调整张量的形状
        x = paddle.reshape(x, shape=[x.shape[0], x.shape[1]])
        # 通过全连接层进行前向传播
        x = self.fc(x)
        # 如果不是训练阶段，则对输出进行 softmax 操作
        if not self.training:
            x = F.softmax(x, axis=1)
        return x
```