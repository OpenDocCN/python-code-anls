# `.\pytorch\torch\distributions\laplace.py`

```
# mypy: allow-untyped-defs
# 引入 Number 类型用于类型检查
from numbers import Number

# 引入 torch 库
import torch
# 从 torch.distributions 模块中导入 constraints
from torch.distributions import constraints
# 从 torch.distributions.distribution 模块中导入 Distribution 类
from torch.distributions.distribution import Distribution
# 从 torch.distributions.utils 模块中导入 broadcast_all 函数
from torch.distributions.utils import broadcast_all

# 模块中公开的类名列表
__all__ = ["Laplace"]

# Laplace 类，继承自 Distribution 类
class Laplace(Distribution):
    """
    创建由 loc 和 scale 参数化的 Laplace 分布。

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # Laplace distributed with loc=0, scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): 分布的均值
        scale (float or Tensor): 分布的尺度
    """
    
    # 参数约束，loc 是实数，scale 是正数
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # 分布的支持集为实数集
    support = constraints.real
    # 是否具有 rsample 方法
    has_rsample = True

    # 分布的均值属性
    @property
    def mean(self):
        return self.loc

    # 分布的众数属性
    @property
    def mode(self):
        return self.loc

    # 分布的方差属性
    @property
    def variance(self):
        return 2 * self.scale.pow(2)

    # 分布的标准差属性
    @property
    def stddev(self):
        return (2**0.5) * self.scale

    # Laplace 分布的初始化方法
    def __init__(self, loc, scale, validate_args=None):
        # 使用 broadcast_all 函数，将 loc 和 scale 广播到相同的形状
        self.loc, self.scale = broadcast_all(loc, scale)
        # 如果 loc 和 scale 都是数值类型，则批量形状为空
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        # 调用父类 Distribution 的初始化方法
        super().__init__(batch_shape, validate_args=validate_args)

    # 扩展 Laplace 分布的 batch_shape
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Laplace, _instance)
        batch_shape = torch.Size(batch_shape)
        # 扩展 loc 和 scale 到新的 batch_shape
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        # 调用父类 Distribution 的初始化方法
        super(Laplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    # 从 Laplace 分布中抽样
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        finfo = torch.finfo(self.loc.dtype)
        if torch._C._get_tracing_state():
            # [JIT WORKAROUND] 缺乏对 .uniform_() 方法的支持
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device) * 2 - 1
            return self.loc - self.scale * u.sign() * torch.log1p(
                -u.abs().clamp(min=finfo.tiny)
            )
        # 在指定形状下生成均匀分布的随机数 u
        u = self.loc.new(shape).uniform_(finfo.eps - 1, 1)
        # 返回 Laplace 分布的样本
        return self.loc - self.scale * u.sign() * torch.log1p(-u.abs())

    # Laplace 分布的对数概率密度函数
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # 返回给定值的对数概率密度
        return -torch.log(2 * self.scale) - torch.abs(value - self.loc) / self.scale
    # 计算累积分布函数（CDF）的值，给定一个数值作为输入
    def cdf(self, value):
        # 如果需要进行参数验证，则调用内部方法 _validate_sample 进行验证
        if self._validate_args:
            self._validate_sample(value)
        # 计算并返回累积分布函数的结果
        return 0.5 - 0.5 * (value - self.loc).sign() * torch.expm1(
            -(value - self.loc).abs() / self.scale
        )

    # 计算逆累积分布函数（ICDF）的值，给定一个数值作为输入
    def icdf(self, value):
        # 将 value 减去 0.5 存储在 term 变量中
        term = value - 0.5
        # 计算并返回逆累积分布函数的结果
        return self.loc - self.scale * (term).sign() * torch.log1p(-2 * term.abs())

    # 计算概率分布的熵
    def entropy(self):
        # 返回熵的计算结果，使用 PyTorch 的函数来计算对数
        return 1 + torch.log(2 * self.scale)


这些注释解释了每个方法在给定的类中的作用和计算过程，保持了代码结构和缩进不变。
```