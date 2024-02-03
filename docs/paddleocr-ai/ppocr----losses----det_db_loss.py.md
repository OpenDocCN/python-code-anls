# `.\PaddleOCR\ppocr\losses\det_db_loss.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 代码来源于：
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/DB_loss.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入 paddle 库
import paddle
# 从 paddle 库中导入 nn 模块
from paddle import nn

# 从当前目录下的 det_basic_loss 文件中导入 BalanceLoss、MaskL1Loss、DiceLoss 类
from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss

# 定义 DBLoss 类，继承自 nn.Layer 类
class DBLoss(nn.Layer):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    # 初始化函数，接收一些参数
    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 **kwargs):
        # 调用父类的初始化函数
        super(DBLoss, self).__init__()
        # 设置一些参数
        self.alpha = alpha
        self.beta = beta
        # 创建 DiceLoss 对象
        self.dice_loss = DiceLoss(eps=eps)
        # 创建 MaskL1Loss 对象
        self.l1_loss = MaskL1Loss(eps=eps)
        # 创建 BalanceLoss 对象
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio)
    # 前向传播函数，计算损失并返回损失字典
    def forward(self, predicts, labels):
        # 获取预测结果中的预测地图
        predict_maps = predicts['maps']
        # 获取标签中的阈值地图、阈值掩码、收缩地图、收缩掩码
        label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[1:]
        # 获取预测地图中的收缩地图、阈值地图、二值地图
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]

        # 计算收缩地图的二元交叉熵损失
        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map, label_shrink_mask)
        # 计算阈值地图的 L1 损失
        loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map, label_threshold_mask)
        # 计算二值地图的 Dice 损失
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map, label_shrink_mask)
        # 对收缩地图损失乘以 alpha
        loss_shrink_maps = self.alpha * loss_shrink_maps
        # 对阈值地图损失乘以 beta
        loss_threshold_maps = self.beta * loss_threshold_maps
        
        # 如果预测结果中包含距离地图
        if 'distance_maps' in predicts.keys():
            # 获取距离地图和 CBN 地图
            distance_maps = predicts['distance_maps']
            cbn_maps = predicts['cbn_maps']
            # 计算 CBN 地图的二元交叉熵损失
            cbn_loss = self.bce_loss(cbn_maps[:, 0, :, :], label_shrink_map, label_shrink_mask)
        else:
            # 如果没有距离地图，则设置距离损失和 CBN 损失为0
            dis_loss = paddle.to_tensor([0.])
            cbn_loss = paddle.to_tensor([0.])

        # 计算总损失
        loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps
        # 构建损失字典
        losses = {'loss': loss_all + cbn_loss, \
                  "loss_shrink_maps": loss_shrink_maps, \
                  "loss_threshold_maps": loss_threshold_maps, \
                  "loss_binary_maps": loss_binary_maps, \
                  "loss_cbn": cbn_loss}
        # 返回损失字典
        return losses
```