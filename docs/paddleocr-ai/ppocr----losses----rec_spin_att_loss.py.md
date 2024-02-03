# `.\PaddleOCR\ppocr\losses\rec_spin_att_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
# 从 paddle 库中导入 nn 模块
from paddle import nn

'''This code is refer from:
https://github.com/hikopensource/DAVAR-Lab-OCR
'''

# 定义 SPINAttentionLoss 类，继承自 nn.Layer 类
class SPINAttentionLoss(nn.Layer):
    # 初始化函数，定义损失函数
    def __init__(self, reduction='mean', ignore_index=-100, **kwargs):
        super(SPINAttentionLoss, self).__init__()
        # 使用交叉熵损失函数
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction=reduction, ignore_index=ignore_index)

    # 前向传播函数，计算损失
    def forward(self, predicts, batch):
        # 获取标签数据
        targets = batch[1].astype("int64")
        # 移除标签中的 [eos]
        targets = targets[:, 1:]

        # 获取标签长度
        label_lengths = batch[2].astype('int64')
        # 获取预测结果的形状信息
        batch_size, num_steps, num_classes = predicts.shape[0], predicts.shape[1], predicts.shape[2]
        # 检查目标形状和输入形状是否匹配
        assert len(targets.shape) == len(list(predicts.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        # 重塑预测结果和目标标签的形状
        inputs = paddle.reshape(predicts, [-1, predicts.shape[-1]])
        targets = paddle.reshape(targets, [-1])

        # 返回损失值
        return {'loss': self.loss_func(inputs, targets)}
```