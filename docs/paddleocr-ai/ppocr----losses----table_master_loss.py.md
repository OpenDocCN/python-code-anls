# `.\PaddleOCR\ppocr\losses\table_master_loss.py`

```py
# 版权声明，告知代码版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""
# 代码来源于：
# https://github.com/JiaquanYe/TableMASTER-mmocr/tree/master/mmocr/models/textrecog/losses

# 导入 paddle 库
import paddle
# 从 paddle 库中导入 nn 模块
from paddle import nn

# 定义 TableMasterLoss 类，继承自 nn.Layer 类
class TableMasterLoss(nn.Layer):
    # 初始化函数，设置忽略索引为-1
    def __init__(self, ignore_index=-1):
        # 调用父类的初始化函数
        super(TableMasterLoss, self).__init__()
        # 定义结构损失为交叉熵损失，设置忽略索引和减少方式为均值
        self.structure_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='mean')
        # 定义框损失为 L1 损失，设置减少方式为总和
        self.box_loss = nn.L1Loss(reduction='sum')
        # 设置一个很小的值 eps
        self.eps = 1e-12
    # 前向传播函数，计算损失并返回
    def forward(self, predicts, batch):
        # 获取预测的结构概率和目标结构
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1]
        # 从第二个位置开始截取目标结构
        structure_targets = structure_targets[:, 1:]
        # 重塑结构概率和目标结构的形状
        structure_probs = structure_probs.reshape([-1, structure_probs.shape[-1]])
        structure_targets = structure_targets.reshape([-1])

        # 计算结构损失
        structure_loss = self.structure_loss(structure_probs, structure_targets)
        structure_loss = structure_loss.mean()
        losses = dict(structure_loss=structure_loss)

        # 边界框损失
        bboxes_preds = predicts['loc_preds']
        bboxes_targets = batch[2][:, 1:, :]
        bbox_masks = batch[3][:, 1:]
        # 掩码空边界框或非边界框结构令牌的边界框

        masked_bboxes_preds = bboxes_preds * bbox_masks
        masked_bboxes_targets = bboxes_targets * bbox_masks

        # 水平损失（x 和宽度）
        horizon_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 0::2],
                                         masked_bboxes_targets[:, :, 0::2])
        horizon_loss = horizon_sum_loss / (bbox_masks.sum() + self.eps)
        # 垂直损失（y 和高度）
        vertical_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 1::2],
                                          masked_bboxes_targets[:, :, 1::2])
        vertical_loss = vertical_sum_loss / (bbox_masks.sum() + self.eps)

        horizon_loss = horizon_loss.mean()
        vertical_loss = vertical_loss.mean()
        all_loss = structure_loss + horizon_loss + vertical_loss
        # 更新损失字典
        losses.update({
            'loss': all_loss,
            'horizon_bbox_loss': horizon_loss,
            'vertical_bbox_loss': vertical_loss
        })
        return losses
```