# `.\PaddleOCR\ppocr\losses\rec_att_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 PaddlePaddle 库
import paddle
from paddle import nn

# 定义 AttentionLoss 类，继承自 nn.Layer
class AttentionLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(AttentionLoss, self).__init__()
        # 初始化损失函数为交叉熵损失函数
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')

    # 前向传播函数
    def forward(self, predicts, batch):
        # 获取目标值和标签长度
        targets = batch[1].astype("int64")
        label_lengths = batch[2].astype('int64')
        # 获取预测结果的形状信息
        batch_size, num_steps, num_classes = predicts.shape[0], predicts.shape[1], predicts.shape[2]
        # 断言目标值的形状和预测结果的形状匹配
        assert len(targets.shape) == len(list(predicts.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        # 重塑预测结果和目标值的形状
        inputs = paddle.reshape(predicts, [-1, predicts.shape[-1]])
        targets = paddle.reshape(targets, [-1])

        # 返回损失值
        return {'loss': paddle.sum(self.loss_func(inputs, targets))}
```