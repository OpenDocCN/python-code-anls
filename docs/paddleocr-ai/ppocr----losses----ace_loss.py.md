# `.\PaddleOCR\ppocr\losses\ace_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证以“原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 许可证下的限制

# 代码来源：https://github.com/viig99/LS-ACELoss

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 PaddlePaddle 深度学习框架
import paddle
import paddle.nn as nn

# 定义 ACELoss 类，继承自 nn.Layer
class ACELoss(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        # 初始化交叉熵损失函数
        self.loss_func = nn.CrossEntropyLoss(
            weight=None,
            ignore_index=0,
            reduction='none',
            soft_label=True,
            axis=-1)

    # 定义 __call__ 方法，计算 ACE 损失
    def __call__(self, predicts, batch):
        # 如果 predicts 是列表或元组，则取最后一个元素
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]

        # 获取预测张量的形状
        B, N = predicts.shape[:2]
        # 创建一个张量 div，值为 N，数据类型为 float32
        div = paddle.to_tensor([N]).astype('float32')

        # 对预测张量进行 softmax 操作
        predicts = nn.functional.softmax(predicts, axis=-1)
        # 对预测张量在第一维度上求和
        aggregation_preds = paddle.sum(predicts, axis=1)
        # 将求和结果除以 div
        aggregation_preds = paddle.divide(aggregation_preds, div)

        # 获取 batch 的长度和数据
        length = batch[2].astype("float32")
        batch = batch[3].astype("float32")
        # 将 batch 的第一列减去 div，再除以 div
        batch[:, 0] = paddle.subtract(div, length)
        batch = paddle.divide(batch, div)

        # 计算 ACE 损失
        loss = self.loss_func(aggregation_preds, batch)
        # 返回损失值
        return {"loss_ace": loss}
```