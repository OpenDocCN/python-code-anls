# `.\YOLO-World\yolo_world\models\losses\dynamic_loss.py`

```py
# 导入必要的库
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.models.losses.mse_loss import mse_loss
from mmyolo.registry import MODELS

# 注册模型类为CoVMSELoss
@MODELS.register_module()
class CoVMSELoss(nn.Module):

    def __init__(self,
                 dim: int = 0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        # 初始化参数
        self.dim = dim
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self,
                pred: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function of loss."""
        # 确保重写的减少参数在合法范围内
        assert reduction_override in (None, 'none', 'mean', 'sum')
        # 根据重写的减少参数或者默认减少参数来确定减少方式
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # 计算协方差
        cov = pred.std(self.dim) / pred.mean(self.dim).clamp(min=self.eps)
        # 创建目标张量
        target = torch.zeros_like(cov)
        # 计算损失
        loss = self.loss_weight * mse_loss(
            cov, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
```