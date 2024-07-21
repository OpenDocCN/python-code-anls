# `.\pytorch\torch\distributions\von_mises.py`

```py
# mypy: allow-untyped-defs
# 导入数学库
import math

# 导入 PyTorch 相关模块
import torch
import torch.jit
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property

# 模块的公开接口，仅包含 VonMises 类
__all__ = ["VonMises"]


# 多项式求值函数，用于计算多项式在给定点的值
def _eval_poly(y, coef):
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


# 以下是修改的贝塞尔函数 I0 和 I1 的系数
_I0_COEF_SMALL = [
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.360768e-1,
    0.45813e-2,
]
_I0_COEF_LARGE = [
    0.39894228,
    0.1328592e-1,
    0.225319e-2,
    -0.157565e-2,
    0.916281e-2,
    -0.2057706e-1,
    0.2635537e-1,
    -0.1647633e-1,
    0.392377e-2,
]
_I1_COEF_SMALL = [
    0.5,
    0.87890594,
    0.51498869,
    0.15084934,
    0.2658733e-1,
    0.301532e-2,
    0.32411e-3,
]
_I1_COEF_LARGE = [
    0.39894228,
    -0.3988024e-1,
    -0.362018e-2,
    0.163801e-2,
    -0.1031555e-1,
    0.2282967e-1,
    -0.2895312e-1,
    0.1787654e-1,
    -0.420059e-2,
]

# 小值和大值情况下的系数列表
_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0):
    """
    返回对数修正贝塞尔函数 log(I_order(x))，其中 x > 0，order 可为 0 或 1。
    """
    assert order == 0 or order == 1

    # 计算小值情况下的解
    y = x / 3.75
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # 计算大值情况下的解
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    # 根据 x 的大小选择返回结果
    result = torch.where(x < 3.75, small, large)
    return result


@torch.jit.script_if_tracing
def _rejection_sample(loc, concentration, proposal_r, x):
    """
    拒绝抽样函数，用于从 von Mises 分布中抽样。
    """
    # 初始化完成标志
    done = torch.zeros(x.shape, dtype=torch.bool, device=loc.device)
    while not done.all():
        # 生成随机数 u，shape 为 (3,) + x.shape
        u = torch.rand((3,) + x.shape, dtype=loc.dtype, device=loc.device)
        u1, u2, u3 = u.unbind()
        z = torch.cos(math.pi * u1)
        f = (1 + proposal_r * z) / (proposal_r + z)
        c = concentration * (proposal_r - f)
        accept = ((c * (2 - c) - u2) > 0) | ((c / u2).log() + 1 - c >= 0)
        if accept.any():
            x = torch.where(accept, (u3 - 0.5).sign() * f.acos(), x)
            done = done | accept
    return (x + math.pi + loc) % (2 * math.pi) - math.pi


class VonMises(Distribution):
    """
    一个圆形 von Mises 分布类。

    此实现使用极坐标。loc 和 value 参数可以是任意实数（以便进行无约束优化），
    但被解释为模 2 pi 的角度。

    例子::
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = VonMises(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # loc=1, concentration=1 的 von Mises 分布
        tensor([1.9777])

    :param torch.Tensor loc: 弧度角。
    :param torch.Tensor concentration: 集中参数
    """
    # 定义参数约束字典，loc 是实数，concentration 是正数
    arg_constraints = {"loc": constraints.real, "concentration": constraints.positive}
    # 支持实数约束
    support = constraints.real
    # 是否有 rsample 方法，默认为 False
    has_rsample = False

    def __init__(self, loc, concentration, validate_args=None):
        # 广播 loc 和 concentration 到相同的形状
        self.loc, self.concentration = broadcast_all(loc, concentration)
        # 确定批次形状和事件形状
        batch_shape = self.loc.shape
        event_shape = torch.Size()
        # 调用父类初始化方法
        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        # 如果开启参数验证，验证样本值是否符合分布要求
        if self._validate_args:
            self._validate_sample(value)
        # 计算对数概率密度函数
        log_prob = self.concentration * torch.cos(value - self.loc)
        log_prob = (
            log_prob
            - math.log(2 * math.pi)
            - _log_modified_bessel_fn(self.concentration, order=0)
        )
        return log_prob

    @lazy_property
    def _loc(self):
        # 将 loc 转换为双精度浮点数
        return self.loc.to(torch.double)

    @lazy_property
    def _concentration(self):
        # 将 concentration 转换为双精度浮点数
        return self.concentration.to(torch.double)

    @lazy_property
    def _proposal_r(self):
        # 计算提议分布的参数 _proposal_r
        kappa = self._concentration
        tau = 1 + (1 + 4 * kappa**2).sqrt()
        rho = (tau - (2 * tau).sqrt()) / (2 * kappa)
        _proposal_r = (1 + rho**2) / (2 * rho)
        # 对于小的 kappa，使用以 0 为中心的二阶泰勒展开
        _proposal_r_taylor = 1 / kappa + kappa
        return torch.where(kappa < 1e-5, _proposal_r_taylor, _proposal_r)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        """
        基于 von Mises 分布的抽样算法基于以下论文：
        D.J. Best 和 N.I. Fisher, "Efficient simulation of the
        von Mises distribution." Applied Statistics (1979): 152-157.

        为了避免在集中度较小时（约 1e-4）使用单精度时在 _rejection_sample() 中出现的卡住问题，
        抽样总是在内部使用双精度进行。
        """
        shape = self._extended_shape(sample_shape)
        x = torch.empty(shape, dtype=self._loc.dtype, device=self.loc.device)
        return _rejection_sample(
            self._loc, self._concentration, self._proposal_r, x
        ).to(self.loc.dtype)

    def expand(self, batch_shape):
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get("_validate_args")
            # 扩展 loc 和 concentration 到新的批次形状
            loc = self.loc.expand(batch_shape)
            concentration = self.concentration.expand(batch_shape)
            return type(self)(loc, concentration, validate_args=validate_args)

    @property
    def mean(self):
        """
        所提供的均值是循环的均值。
        """
        return self.loc

    @property
    def mode(self):
        # 返回分布的众数，即 loc
        return self.loc

    @lazy_property
    def variance(self):
        """
        计算方差，这里使用的是圆形方差。
        """
        # 计算方差公式的一部分，使用修正的贝塞尔函数计算
        return (
            1
            - (
                _log_modified_bessel_fn(self.concentration, order=1)
                - _log_modified_bessel_fn(self.concentration, order=0)
            ).exp()
        )
```