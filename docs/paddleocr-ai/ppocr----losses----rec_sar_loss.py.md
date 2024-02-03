# `.\PaddleOCR\ppocr\losses\rec_sar_loss.py`

```py
# 导入必要的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 模块
import paddle
# 从 paddle 模块中导入 nn 模块
from paddle import nn

# 定义 SARLoss 类，继承自 nn.Layer 类
class SARLoss(nn.Layer):
    # 初始化函数，接受关键字参数
    def __init__(self, **kwargs):
        # 调用父类的初始化函数
        super(SARLoss, self).__init__()
        # 获取 ignore_index 参数，默认值为 92
        ignore_index = kwargs.get('ignore_index', 92)  # 6626
        # 创建交叉熵损失函数对象，设置损失计算方式为平均，忽略指定索引
        self.loss_func = paddle.nn.loss.CrossEntropyLoss(
            reduction="mean", ignore_index=ignore_index)

    # 前向传播函数，接受 predicts 和 batch 两个参数
    def forward(self, predicts, batch):
        # 忽略输出的最后一个索引，使其与目标序列长度相同
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
        # 返回损失值字典
        return {'loss': loss}
```