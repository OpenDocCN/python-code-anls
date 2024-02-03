# `.\PaddleOCR\ppocr\losses\cls_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 PaddlePaddle 的神经网络模块
from paddle import nn

# 定义一个名为 ClsLoss 的类，继承自 nn.Layer 类
class ClsLoss(nn.Layer):
    # 初始化方法
    def __init__(self, **kwargs):
        super(ClsLoss, self).__init__()
        # 初始化损失函数为交叉熵损失函数，reduction 参数设置为 'mean'
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    # 前向传播方法
    def forward(self, predicts, batch):
        # 获取标签数据并转换为 int64 类型
        label = batch[1].astype("int64")
        # 计算损失值
        loss = self.loss_func(input=predicts, label=label)
        # 返回损失值作为字典的值
        return {'loss': loss}
```