# `.\PaddleOCR\ppocr\losses\rec_srn_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
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

import paddle
from paddle import nn

# 定义 SRNLoss 类，继承自 nn.Layer
class SRNLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(SRNLoss, self).__init__()
        # 初始化损失函数为交叉熵损失函数，求和作为损失值
        self.loss_func = paddle.nn.loss.CrossEntropyLoss(reduction="sum")

    # 前向传播函数
    def forward(self, predicts, batch):
        # 获取预测结果
        predict = predicts['predict']
        word_predict = predicts['word_out']
        gsrm_predict = predicts['gsrm_out']
        # 获取标签
        label = batch[1]

        # 将标签转换为 int64 类型，并重塑形状
        casted_label = paddle.cast(x=label, dtype='int64')
        casted_label = paddle.reshape(x=casted_label, shape=[-1, 1])

        # 计算 word 损失、gsrm 损失和 vsfd 损失
        cost_word = self.loss_func(word_predict, label=casted_label)
        cost_gsrm = self.loss_func(gsrm_predict, label=casted_label)
        cost_vsfd = self.loss_func(predict, label=casted_label)

        # 将损失值求和并重塑形状
        cost_word = paddle.reshape(x=paddle.sum(cost_word), shape=[1])
        cost_gsrm = paddle.reshape(x=paddle.sum(cost_gsrm), shape=[1])
        cost_vsfd = paddle.reshape(x=paddle.sum(cost_vsfd), shape=[1])

        # 计算总损失值
        sum_cost = cost_word * 3.0 + cost_vsfd + cost_gsrm * 0.15

        # 返回损失值字典
        return {'loss': sum_cost, 'word_loss': cost_word, 'img_loss': cost_vsfd}
```