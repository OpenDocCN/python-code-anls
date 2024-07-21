# `.\pytorch\torch\distributions\log_normal.py`

```
# 引入需要的模块和类
# mypy: allow-untyped-defs
from torch.distributions import constraints  # 导入约束模块
from torch.distributions.normal import Normal  # 导入正态分布类
from torch.distributions.transformed_distribution import TransformedDistribution  # 导入变换分布类
from torch.distributions.transforms import ExpTransform  # 导入指数变换类

# 定义导出的类列表
__all__ = ["LogNormal"]

# 定义对数正态分布类
class LogNormal(TransformedDistribution):
    r"""
    创建一个由 `loc` 和 `scale` 参数化的对数正态分布，其中::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # 服从均值=0，标准差=1的对数正态分布
        tensor([ 0.1046])

    参数:
        loc (float or Tensor): 对数分布的均值
        scale (float or Tensor): 对数分布的标准差
    """

    # 参数约束，loc为实数，scale为正数
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # 支持值的约束为正数
    support = constraints.positive
    # 标记该分布类有rsample方法
    has_rsample = True

    # 初始化方法
    def __init__(self, loc, scale, validate_args=None):
        # 使用正态分布作为基础分布
        base_dist = Normal(loc, scale, validate_args=validate_args)
        # 使用指数变换对基础分布进行变换
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    # 扩展方法，支持对实例进行批处理
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    # 返回loc属性，即基础分布的均值
    @property
    def loc(self):
        return self.base_dist.loc

    # 返回scale属性，即基础分布的标准差
    @property
    def scale(self):
        return self.base_dist.scale

    # 返回分布的均值
    @property
    def mean(self):
        return (self.loc + self.scale.pow(2) / 2).exp()

    # 返回分布的众数
    @property
    def mode(self):
        return (self.loc - self.scale.square()).exp()

    # 返回分布的方差
    @property
    def variance(self):
        scale_sq = self.scale.pow(2)
        return scale_sq.expm1() * (2 * self.loc + scale_sq).exp()

    # 返回分布的熵
    def entropy(self):
        return self.base_dist.entropy() + self.loc
```