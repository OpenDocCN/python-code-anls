# `.\PaddleOCR\ppocr\losses\vqa_token_layoutlm_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 PaddlePaddle 中的 nn 模块
from paddle import nn
# 导入自定义的 DMLLoss 损失函数
from ppocr.losses.basic_loss import DMLLoss

# 定义 VQASerTokenLayoutLMLoss 类，继承自 nn.Layer
class VQASerTokenLayoutLMLoss(nn.Layer):
    # 初始化函数，接受 num_classes 和 key 两个参数
    def __init__(self, num_classes, key=None):
        super().__init__()
        # 使用交叉熵损失函数作为分类损失
        self.loss_class = nn.CrossEntropyLoss()
        # 设置类别数量
        self.num_classes = num_classes
        # 获取损失函数的忽略索引
        self.ignore_index = self.loss_class.ignore_index
        # 设置关键字
        self.key = key

    # 前向传播函数，接受 predicts 和 batch 两个参数
    def forward(self, predicts, batch):
        # 如果 predicts 是字典且 key 不为空，则取出 key 对应的值
        if isinstance(predicts, dict) and self.key is not None:
            predicts = predicts[self.key]
        # 获取标签和注意力掩码
        labels = batch[5]
        attention_mask = batch[2]
        # 如果注意力掩码不为空
        if attention_mask is not None:
            # 获取有效的损失和输出
            active_loss = attention_mask.reshape([-1, ]) == 1
            active_output = predicts.reshape([-1, self.num_classes])[active_loss]
            active_label = labels.reshape([-1, ])[active_loss]
            # 计算损失
            loss = self.loss_class(active_output, active_label)
        else:
            # 计算损失
            loss = self.loss_class(
                predicts.reshape([-1, self.num_classes]),
                labels.reshape([-1, ]))
        # 返回损失值
        return {'loss': loss}
```