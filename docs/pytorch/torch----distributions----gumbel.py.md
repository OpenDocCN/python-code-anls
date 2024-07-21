# `.\pytorch\torch\distributions\gumbel.py`

```
# 引入数学库和 torch 库
import math
import torch
# 从 torch.distributions 中引入需要的约束条件和分布类
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all, euler_constant

# 定义只有 Gumbel 类在当前模块中可以被导入的列表
__all__ = ["Gumbel"]

# 定义 Gumbel 类，继承自 TransformedDistribution 类
class Gumbel(TransformedDistribution):
    """
    Samples from a Gumbel Distribution.

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
        tensor([ 1.0124])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
    """
    # 约束条件定义，loc 是实数，scale 是正数
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # 支持条件定义为实数
    support = constraints.real

    # 初始化方法
    def __init__(self, loc, scale, validate_args=None):
        # 将 loc 和 scale 广播到相同的形状
        self.loc, self.scale = broadcast_all(loc, scale)
        # 获取浮点数类型的信息
        finfo = torch.finfo(self.loc.dtype)
        # 如果 loc 和 scale 都是数值类型
        if isinstance(loc, Number) and isinstance(scale, Number):
            # 创建一个均匀分布对象，范围从 finfo.tiny 到 1 - finfo.eps
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps, validate_args=validate_args)
        else:
            # 创建一个均匀分布对象，范围从与 loc 形状相同的 finfo.tiny 到与 loc 形状相同的 1 - finfo.eps
            base_dist = Uniform(
                torch.full_like(self.loc, finfo.tiny),
                torch.full_like(self.loc, 1 - finfo.eps),
                validate_args=validate_args,
            )
        # 定义变换列表，用于变换均匀分布为 Gumbel 分布
        transforms = [
            ExpTransform().inv,
            AffineTransform(loc=0, scale=-torch.ones_like(self.scale)),
            ExpTransform().inv,
            AffineTransform(loc=loc, scale=-self.scale),
        ]
        # 调用父类的初始化方法
        super().__init__(base_dist, transforms, validate_args=validate_args)

    # 扩展方法，用于创建新的 Gumbel 分布实例
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gumbel, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    # 显式定义 Gumbel 分布的对数概率密度函数，用于处理精度问题
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (self.loc - value) / self.scale
        return (y - y.exp()) - self.scale.log()

    # 计算均值的属性
    @property
    def mean(self):
        return self.loc + self.scale * euler_constant

    # 计算众数的属性
    @property
    def mode(self):
        return self.loc

    # 计算标准差的属性
    @property
    def stddev(self):
        return (math.pi / math.sqrt(6)) * self.scale

    # 计算方差的属性
    @property
    def variance(self):
        return self.stddev.pow(2)

    # 计算熵的方法
    def entropy(self):
        return self.scale.log() + (1 + euler_constant)
```