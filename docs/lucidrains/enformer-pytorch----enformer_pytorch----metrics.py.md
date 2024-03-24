# `.\lucidrains\enformer-pytorch\enformer_pytorch\metrics.py`

```
from torchmetrics import Metric
from typing import Optional
import torch

# 定义一个自定义的 Metric 类，用于计算每个通道的平均皮尔逊相关系数
class MeanPearsonCorrCoefPerChannel(Metric):
    # 是否可微分，默认为不可微分
    is_differentiable: Optional[bool] = False
    # 较高值是否更好，默认为是
    higher_is_better: Optional[bool] = True

    def __init__(self, n_channels:int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        # 调用父类的初始化方法
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # 设置要减少的维度
        self.reduce_dims=(0, 1)
        # 添加状态变量，用于存储乘积、真实值、真实值平方、预测值、预测值平方、计数
        self.add_state("product", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("true", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("true_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("pred_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 断言预测值和目标值的形状相同
        assert preds.shape == target.shape

        # 更新状态变量
        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        # 计算真实值和预测值的均值
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        # 计算协方差、真实值方差、预测值方差、真实值和预测值的平方根乘积、相关系数
        covariance = (self.product
                    - true_mean * self.pred
                    - pred_mean * self.true
                    + self.count * true_mean * pred_mean)

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation
```