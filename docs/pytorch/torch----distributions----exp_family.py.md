# `.\pytorch\torch\distributions\exp_family.py`

```
# 引入 torch 库
import torch
# 从 torch.distributions.distribution 模块中引入 Distribution 类
from torch.distributions.distribution import Distribution

# 定义一个列表，包含当前模块中公开的类 ExponentialFamily
__all__ = ["ExponentialFamily"]

# 定义 ExponentialFamily 类，继承自 Distribution 类
class ExponentialFamily(Distribution):
    """
    ExponentialFamily 是指属于指数族的概率分布的抽象基类，其概率质量/密度函数的形式如下所定义：

    .. math::

        p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))

    其中 :math:`\theta` 表示自然参数，:math:`t(x)` 表示充分统计量，
    :math:`F(\theta)` 是给定族别的对数归一化函数，:math:`k(x)` 是载体测度。

    注意:
        这个类是 `Distribution` 类与属于指数族的分布之间的中介，主要用于检查 `.entropy()` 和解析 KL 散度方法的正确性。
        我们使用这个类使用自动微分框架和 Bregman 散度来计算熵和 KL 散度（感谢: Frank Nielsen 和 Richard Nock, Entropies and
        Cross-entropies of Exponential Families）。
    """

    @property
    def _natural_params(self):
        """
        抽象方法，用于获取自然参数的元组，基于分布返回张量
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        """
        抽象方法，用于获取对数归一化函数，基于分布和输入返回对数归一化函数
        """
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self):
        """
        抽象方法，用于获取期望的载体测度，用于计算熵
        """
        raise NotImplementedError

    def entropy(self):
        """
        方法，使用对数归一化函数的 Bregman 散度来计算熵
        """
        # 初始化结果为期望的载体测度的相反数
        result = -self._mean_carrier_measure
        # 对自然参数进行去梯度化并要求梯度
        nparams = [p.detach().requires_grad_() for p in self._natural_params]
        # 计算对数归一化函数并返回梯度
        lg_normal = self._log_normalizer(*nparams)
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        # 结果加上对数归一化函数
        result += lg_normal
        # 对于自然参数和梯度，按批次形状计算相应结果
        for np, g in zip(nparams, gradients):
            result -= (np * g).reshape(self._batch_shape + (-1,)).sum(-1)
        return result
```