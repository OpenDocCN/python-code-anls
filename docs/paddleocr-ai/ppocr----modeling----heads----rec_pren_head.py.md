# `.\PaddleOCR\ppocr\modeling\heads\rec_pren_head.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 PaddlePaddle 的神经网络模块
from paddle import nn
# 导入 PaddlePaddle 的神经网络函数模块
from paddle.nn import functional as F

# 定义一个自定义的神经网络层 PRENHead
class PRENHead(nn.Layer):
    # 初始化函数，定义输入通道数和输出通道数
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PRENHead, self).__init__()
        # 创建一个线性层，将输入通道数映射到输出通道数
        self.linear = nn.Linear(in_channels, out_channels)

    # 前向传播函数，接收输入 x 和目标 targets
    def forward(self, x, targets=None):
        # 将输入 x 经过线性层得到预测结果
        predicts = self.linear(x)

        # 如果不是训练状态，则对预测结果进行 softmax 处理
        if not self.training:
            predicts = F.softmax(predicts, axis=2)

        # 返回预测结果
        return predicts
```