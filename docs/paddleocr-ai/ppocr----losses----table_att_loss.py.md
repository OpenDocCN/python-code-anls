# `.\PaddleOCR\ppocr\losses\table_att_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
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

# 导入 Paddle 库
import paddle
from paddle import nn
from paddle.nn import functional as F

# 定义一个名为 TableAttentionLoss 的类，继承自 nn.Layer 类
class TableAttentionLoss(nn.Layer):
    # 初始化函数，接受结构权重和位置权重作为参数
    def __init__(self, structure_weight, loc_weight, **kwargs):
        # 调用父类的初始化函数
        super(TableAttentionLoss, self).__init__()
        # 定义损失函数为交叉熵损失，权重为 None，减少方式为 'none'
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='none')
        # 设置结构权重
        self.structure_weight = structure_weight
        # 设置位置权重
        self.loc_weight = loc_weight
    # 定义前向传播函数，接受模型预测结果和批处理数据作为输入
    def forward(self, predicts, batch):
        # 获取预测的结构概率和目标结构
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1].astype("int64")
        # 从第二列开始获取结构目标
        structure_targets = structure_targets[:, 1:]
        # 重塑结构概率和目标结构的形状
        structure_probs = paddle.reshape(structure_probs, [-1, structure_probs.shape[-1]])
        structure_targets = paddle.reshape(structure_targets, [-1])
        # 计算结构损失
        structure_loss = self.loss_func(structure_probs, structure_targets)

        # 计算平均结构损失并乘以结构权重
        structure_loss = paddle.mean(structure_loss) * self.structure_weight

        # 获取位置预测和位置目标
        loc_preds = predicts['loc_preds']
        loc_targets = batch[2].astype("float32")
        loc_targets_mask = batch[3].astype("float32")
        # 从第二列开始获取位置目标和位置目标掩码
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]
        # 计算位置损失
        loc_loss = F.mse_loss(loc_preds * loc_targets_mask, loc_targets) * self.loc_weight

        # 计算总损失
        total_loss = structure_loss + loc_loss
        # 返回损失字典
        return {
            'loss': total_loss,
            "structure_loss": structure_loss,
            "loc_loss": loc_loss
        }
class SLALoss(nn.Layer):
    # 定义 SLA 损失函数类，包含结构权重、位置权重、位置损失类型等参数
    def __init__(self, structure_weight, loc_weight, loc_loss='mse', **kwargs):
        super(SLALoss, self).__init__()
        # 初始化交叉熵损失函数，不使用权重，采用均值作为损失值
        self.loss_func = nn.CrossEntropyLoss(weight=None, reduction='mean')
        # 设置结构权重
        self.structure_weight = structure_weight
        # 设置位置权重
        self.loc_weight = loc_weight
        # 设置位置损失类型
        self.loc_loss = loc_loss
        # 设置一个很小的数，用于防止除零错误
        self.eps = 1e-12

    def forward(self, predicts, batch):
        # 获取预测的结构概率
        structure_probs = predicts['structure_probs']
        # 获取结构目标值，并转换为整数类型
        structure_targets = batch[1].astype("int64")
        # 去掉第一列，因为第一列是起始标记
        structure_targets = structure_targets[:, 1:]

        # 计算结构损失
        structure_loss = self.loss_func(structure_probs, structure_targets)

        # 计算结构损失的均值，并乘以结构权重
        structure_loss = paddle.mean(structure_loss) * self.structure_weight

        # 获取位置预测值
        loc_preds = predicts['loc_preds']
        # 获取位置目标值，并转换为浮点数类型
        loc_targets = batch[2].astype("float32")
        # 获取位置目标值的掩码，并转换为浮点数类型
        loc_targets_mask = batch[3].astype("float32")
        # 去掉第一列，因为第一列是起始标记
        loc_targets = loc_targets[:, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]

        # 计算位置损失，采用平滑 L1 损失函数
        loc_loss = F.smooth_l1_loss(
            loc_preds * loc_targets_mask,
            loc_targets * loc_targets_mask,
            reduction='sum') * self.loc_weight

        # 将位置损失除以位置目标值的掩码和 eps，防止除零错误
        loc_loss = loc_loss / (loc_targets_mask.sum() + self.eps)
        # 计算总损失，包括结构损失和位置损失
        total_loss = structure_loss + loc_loss
        return {
            'loss': total_loss,
            "structure_loss": structure_loss,
            "loc_loss": loc_loss
        }
```