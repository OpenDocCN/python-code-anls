# `.\PaddleOCR\ppocr\losses\rec_satrn_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何担保或条件，无论是明示还是暗示的
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/models/textrecog/module_losses/ce_module_loss.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
# 从 paddle 中导入 nn 模块
from paddle import nn

# 定义 SATRNLoss 类，继承自 nn.Layer
class SATRNLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(SATRNLoss, self).__init__()
        # 获取 ignore_index 参数，默认值为 92
        ignore_index = kwargs.get('ignore_index', 92)  # 6626
        # 初始化交叉熵损失函数，设置忽略索引为 ignore_index
        self.loss_func = paddle.nn.loss.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index)

    # 前向传播函数
    def forward(self, predicts, batch):
        # 忽略预测的最后一个索引，使其与目标序列长度相同
        predict = predicts[:, :-1, :]
        # 忽略目标序列的第一个索引，用于损失计算
        label = batch[1].astype("int64")[:, 1:]
        # 获取预测张量的形状信息
        batch_size, num_steps, num_classes = predict.shape[0], predict.shape[1], predict.shape[2]
        # 断言目标张量的形状与输入张量的形状匹配
        assert len(label.shape) == len(list(predict.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        # 将预测张量重塑为二维张量
        inputs = paddle.reshape(predict, [-1, num_classes])
        # 将目标张量重塑为一维张量
        targets = paddle.reshape(label, [-1])
        # 计算损失值
        loss = self.loss_func(inputs, targets)
        # 返回损失值的平均值
        return {'loss': loss.mean()}
```