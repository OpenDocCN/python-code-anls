# `.\pytorch\torch\distributions\kumaraswamy.py`

```py
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 nan 函数
from torch import nan
# 导入约束模块
from torch.distributions import constraints
# 导入 TransformedDistribution 类
from torch.distributions.transformed_distribution import TransformedDistribution
# 导入变换模块
from torch.distributions.transforms import AffineTransform, PowerTransform
# 导入均匀分布类
from torch.distributions.uniform import Uniform
# 导入广播函数和欧拉常数
from torch.distributions.utils import broadcast_all, euler_constant

# 导出的类和函数名
__all__ = ["Kumaraswamy"]


def _moments(a, b, n):
    """
    使用 torch.lgamma 计算 Kumaraswamy 分布的第 n 阶矩
    """
    # 计算参数 arg1 和对数值
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)


class Kumaraswamy(TransformedDistribution):
    """
    从 Kumaraswamy 分布中抽样。

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Kumaraswamy(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # 从集中度 alpha=1 和 beta=1 的 Kumaraswamy 分布中抽样
        tensor([ 0.1729])

    Args:
        concentration1 (float or Tensor): 分布的第一个集中度参数 (通常称为 alpha)
        concentration0 (float or Tensor): 分布的第二个集中度参数 (通常称为 beta)
    """
    # 参数约束
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    # 支持的分布范围
    support = constraints.unit_interval
    # 是否支持 rsample 方法
    has_rsample = True

    def __init__(self, concentration1, concentration0, validate_args=None):
        # 广播参数 concentration1 和 concentration0
        self.concentration1, self.concentration0 = broadcast_all(
            concentration1, concentration0
        )
        # 获取 concentration0 的数值精度信息
        finfo = torch.finfo(self.concentration0.dtype)
        # 创建基础分布 Uniform
        base_dist = Uniform(
            torch.full_like(self.concentration0, 0),
            torch.full_like(self.concentration0, 1),
            validate_args=validate_args,
        )
        # 定义变换序列
        transforms = [
            PowerTransform(exponent=self.concentration0.reciprocal()),
            AffineTransform(loc=1.0, scale=-1.0),
            PowerTransform(exponent=self.concentration1.reciprocal()),
        ]
        # 调用父类初始化方法
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # 拓展分布对象至指定的批次形状
        new = self._get_checked_instance(Kumaraswamy, _instance)
        new.concentration1 = self.concentration1.expand(batch_shape)
        new.concentration0 = self.concentration0.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self):
        # 返回 Kumaraswamy 分布的均值
        return _moments(self.concentration1, self.concentration0, 1)

    @property


这里是对给定代码的详细注释，按照要求每行代码都有对应的解释。
    def mode(self):
        """
        # 在对数空间中计算以确保数值稳定性。

        # 计算众数的对数值
        log_mode = (
            self.concentration0.reciprocal() * (-self.concentration0).log1p()
            - (-self.concentration0 * self.concentration1).log1p()
        )

        # 将那些浓度参数小于1的条件下的众数设为 NaN
        log_mode[(self.concentration0 < 1) | (self.concentration1 < 1)] = nan
        
        # 返回众数的指数值
        return log_mode.exp()
        """

    @property
    def variance(self):
        """
        # 计算分布的方差

        # 调用 _moments 函数计算第二中心矩
        return _moments(self.concentration1, self.concentration0, 2) - torch.pow(
            self.mean, 2
        )
        """

    def entropy(self):
        """
        # 计算分布的熵

        # 计算 t1 和 t0 用于熵的计算
        t1 = 1 - self.concentration1.reciprocal()
        t0 = 1 - self.concentration0.reciprocal()

        # 计算 H0，即对数 Gamma 函数的期望值
        H0 = torch.digamma(self.concentration0 + 1) + euler_constant

        # 返回熵的计算结果
        return (
            t0
            + t1 * H0
            - torch.log(self.concentration1)
            - torch.log(self.concentration0)
        )
        """
```