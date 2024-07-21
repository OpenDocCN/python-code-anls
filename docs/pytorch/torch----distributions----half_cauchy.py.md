# `.\pytorch\torch\distributions\half_cauchy.py`

```py
# mypy: allow-untyped-defs
# 引入 math 库，用于数学运算
import math

# 引入 torch 库及相关模块
import torch
from torch import inf
from torch.distributions import constraints
from torch.distributions.cauchy import Cauchy
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform

# 模块内公开的类或函数列表
__all__ = ["HalfCauchy"]

# 定义 HalfCauchy 类，继承自 TransformedDistribution 类
class HalfCauchy(TransformedDistribution):
    r"""
    创建一个半-Cauchy分布，由参数 `scale` 控制，其中::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # 使用 scale=1 生成半-Cauchy分布
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): 完整Cauchy分布的尺度参数
    """
    
    # 参数约束，scale 必须为正数
    arg_constraints = {"scale": constraints.positive}
    
    # 分布的支持范围为非负数
    support = constraints.nonnegative
    
    # 是否支持 rsample 方法
    has_rsample = True

    # 构造函数，初始化 HalfCauchy 实例
    def __init__(self, scale, validate_args=None):
        # 创建基础分布 Cauchy(0, scale)，不验证参数
        base_dist = Cauchy(0, scale, validate_args=False)
        # 调用父类的构造函数，使用 AbsTransform() 进行变换
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    # 扩展函数，用于扩展到指定的批处理形状
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HalfCauchy, _instance)
        return super().expand(batch_shape, _instance=new)

    # 获取 scale 属性的方法
    @property
    def scale(self):
        return self.base_dist.scale

    # 获取均值属性的方法
    @property
    def mean(self):
        return torch.full(
            self._extended_shape(),
            math.inf,
            dtype=self.scale.dtype,
            device=self.scale.device,
        )

    # 获取众数属性的方法
    @property
    def mode(self):
        return torch.zeros_like(self.scale)

    # 获取方差属性的方法
    @property
    def variance(self):
        return self.base_dist.variance

    # 对数概率密度函数的计算
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 将 value 转换为张量，与 base_dist.scale 具有相同的 dtype 和 device
        value = torch.as_tensor(
            value, dtype=self.base_dist.scale.dtype, device=self.base_dist.scale.device
        )
        # 计算对数概率密度，加上 log(2)
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        # 对负值进行处理，返回 -inf
        log_prob = torch.where(value >= 0, log_prob, -inf)
        return log_prob

    # 累积分布函数的计算
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 返回 2 * base_dist.cdf(value) - 1
        return 2 * self.base_dist.cdf(value) - 1

    # 逆累积分布函数的计算
    def icdf(self, prob):
        # 返回 base_dist.icdf((prob + 1) / 2)
        return self.base_dist.icdf((prob + 1) / 2)

    # 熵的计算
    def entropy(self):
        # 返回 base_dist 的熵减去 log(2)
        return self.base_dist.entropy() - math.log(2)
```