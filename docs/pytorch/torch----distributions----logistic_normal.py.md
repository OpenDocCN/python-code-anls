# `.\pytorch\torch\distributions\logistic_normal.py`

```py
# mypy: allow-untyped-defs
# 导入需要的约束条件和分布类
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import StickBreakingTransform

# 模块内导出的符号列表
__all__ = ["LogisticNormal"]

# LogisticNormal 类，继承自 TransformedDistribution
class LogisticNormal(TransformedDistribution):
    r"""
    创建一个 logistic-normal 分布，由参数 :attr:`loc` 和 :attr:`scale` 定义
    这些参数定义了基础 `Normal` 分布，通过 `StickBreakingTransform` 进行变换，使得::

        X ~ LogisticNormal(loc, scale)
        Y = log(X / (1 - X.cumsum(-1)))[..., :-1] ~ Normal(loc, scale)

    Args:
        loc (float or Tensor): 基础分布的均值
        scale (float or Tensor): 基础分布的标准差

    Example::

        >>> # logistic-normal 分布，均值=(0, 0, 0)，标准差=(1, 1, 1)
        >>> # 基础 Normal 分布
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogisticNormal(torch.tensor([0.0] * 3), torch.tensor([1.0] * 3))
        >>> m.sample()
        tensor([ 0.7653,  0.0341,  0.0579,  0.1427])

    """
    # 参数约束
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # 分布的支持域
    support = constraints.simplex
    # 是否支持 rsample 方法
    has_rsample = True

    # 构造函数，初始化 LogisticNormal 对象
    def __init__(self, loc, scale, validate_args=None):
        # 创建基础分布 Normal 对象
        base_dist = Normal(loc, scale, validate_args=validate_args)
        # 如果基础分布的批次形状为空，则扩展为 [1]
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        # 调用父类的构造函数初始化 TransformedDistribution
        super().__init__(
            base_dist, StickBreakingTransform(), validate_args=validate_args
        )

    # 扩展方法，用于创建具有新批次形状的实例
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogisticNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    # loc 属性的 getter 方法，返回基础分布的均值
    @property
    def loc(self):
        return self.base_dist.base_dist.loc

    # scale 属性的 getter 方法，返回基础分布的标准差
    @property
    def scale(self):
        return self.base_dist.base_dist.scale
```