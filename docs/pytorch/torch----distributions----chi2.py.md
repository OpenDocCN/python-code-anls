# `.\pytorch\torch\distributions\chi2.py`

```
# 引入 torch 中的约束和 Gamma 分布
from torch.distributions import constraints
from torch.distributions.gamma import Gamma

# 定义导出的类列表，只导出 "Chi2"
__all__ = ["Chi2"]

# 定义 Chi2 类，继承自 Gamma 分布
class Chi2(Gamma):
    r"""
    创建由形状参数 :attr:`df` 参数化的卡方分布。
    这与 ``Gamma(alpha=0.5*df, beta=0.5)`` 完全等价。

    示例::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Chi2(torch.tensor([1.0]))
        >>> m.sample()  # 形状参数 df=1 的卡方分布
        tensor([ 0.1046])

    Args:
        df (float or Tensor): 分布的形状参数
    """
    # 参数约束，确保 df 是正数
    arg_constraints = {"df": constraints.positive}

    # 构造函数，初始化 Chi2 分布对象
    def __init__(self, df, validate_args=None):
        # 使用 Gamma 分布的构造函数初始化，设置 alpha=0.5*df, beta=0.5
        super().__init__(0.5 * df, 0.5, validate_args=validate_args)

    # 扩展方法，支持批次形状扩展
    def expand(self, batch_shape, _instance=None):
        # 获取已检查的实例，并调用 Gamma 的扩展方法
        new = self._get_checked_instance(Chi2, _instance)
        return super().expand(batch_shape, new)

    # 属性 df，返回分布的自由度
    @property
    def df(self):
        return self.concentration * 2
```