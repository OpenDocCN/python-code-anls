# `.\PaddleOCR\ppocr\losses\rec_ce_loss.py`

```py
# 导入 paddle 模块
import paddle
# 从 paddle 模块中导入 nn 模块
from paddle import nn
# 从 paddle.nn 模块中导入 F 模块
import paddle.nn.functional as F

# 定义一个继承自 nn.Layer 的交叉熵损失类
class CELoss(nn.Layer):
    # 初始化函数，接受平滑参数、是否包含所有类别、忽略的索引等参数
    def __init__(self,
                 smoothing=False,
                 with_all=False,
                 ignore_index=-1,
                 **kwargs):
        # 调用父类的初始化函数
        super(CELoss, self).__init__()
        # 如果忽略的索引大于等于 0
        if ignore_index >= 0:
            # 使用指定的忽略索引创建交叉熵损失函数，设置为平均值，忽略指定索引
            self.loss_func = nn.CrossEntropyLoss(
                reduction='mean', ignore_index=ignore_index)
        else:
            # 使用默认设置创建交叉熵损失函数，设置为平均值
            self.loss_func = nn.CrossEntropyLoss(reduction='mean')
        # 保存是否进行平滑处理的参数
        self.smoothing = smoothing
        # 保存是否包含所有类别的参数
        self.with_all = with_all
```