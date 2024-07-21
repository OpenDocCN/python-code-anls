# `.\pytorch\torch\distributions\pareto.py`

```py
# mypy: allow-untyped-defs
# 引入必要的约束条件模块
from torch.distributions import constraints
# 引入指数分布类
from torch.distributions.exponential import Exponential
# 引入变换后分布类
from torch.distributions.transformed_distribution import TransformedDistribution
# 引入仿射变换和指数变换
from torch.distributions.transforms import AffineTransform, ExpTransform
# 引入广播函数
from torch.distributions.utils import broadcast_all

# 指定外部可访问的类列表
__all__ = ["Pareto"]


# 定义 Pareto 类，继承自 TransformedDistribution 类
class Pareto(TransformedDistribution):
    r"""
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
        tensor([ 1.5623])

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    """
    
    # 参数约束，scale 和 alpha 必须为正数
    arg_constraints = {"alpha": constraints.positive, "scale": constraints.positive}

    # 初始化方法
    def __init__(self, scale, alpha, validate_args=None):
        # 使用广播函数确保 scale 和 alpha 具有相同的形状
        self.scale, self.alpha = broadcast_all(scale, alpha)
        # 创建指数分布对象作为基础分布
        base_dist = Exponential(self.alpha, validate_args=validate_args)
        # 定义变换列表，分别为指数变换和仿射变换
        transforms = [ExpTransform(), AffineTransform(loc=0, scale=self.scale)]
        # 调用父类的初始化方法
        super().__init__(base_dist, transforms, validate_args=validate_args)

    # 扩展方法，用于处理批次形状
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Pareto, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    # 返回 Pareto 分布的均值
    @property
    def mean(self):
        # 当 alpha <= 1 时，均值为无穷大
        a = self.alpha.clamp(min=1)
        return a * self.scale / (a - 1)

    # 返回 Pareto 分布的众数
    @property
    def mode(self):
        return self.scale

    # 返回 Pareto 分布的方差
    @property
    def variance(self):
        # 当 alpha <= 2 时，方差为无穷大
        a = self.alpha.clamp(min=2)
        return self.scale.pow(2) * a / ((a - 1).pow(2) * (a - 2))

    # 返回 Pareto 分布的支持集
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.greater_than_eq(self.scale)

    # 返回 Pareto 分布的熵
    def entropy(self):
        return (self.scale / self.alpha).log() + (1 + self.alpha.reciprocal())
```