# `.\pytorch\torch\distributions\wishart.py`

```
# 导入需要的模块和函数
# mypy: allow-untyped-defs 表示允许未类型化的函数定义
import math  # 导入数学函数库
import warnings  # 导入警告模块，用于发出警告
from numbers import Number  # 导入 Number 类型，用于数值类型的判断
from typing import Optional, Union  # 导入类型注解相关的模块

import torch  # 导入 PyTorch 库
from torch import nan  # 导入 nan 函数
from torch.distributions import constraints  # 导入分布相关的约束条件
from torch.distributions.exp_family import ExponentialFamily  # 导入指数族分布的基类
from torch.distributions.multivariate_normal import _precision_to_scale_tril  # 导入用于将精度矩阵转换为 Cholesky 分解的函数
from torch.distributions.utils import lazy_property  # 导入 lazy_property 装饰器函数

__all__ = ["Wishart"]  # 模块公开的接口，只包含 Wishart 分布

_log_2 = math.log(2)  # 计算常数 log(2)

def _mvdigamma(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    计算多变量 digamma 函数的值

    Args:
        x (torch.Tensor): 输入张量
        p (int): 维度数

    Returns:
        torch.Tensor: digamma 函数的值
    """
    assert x.gt((p - 1) / 2).all(), "Wrong domain for multivariate digamma function."
    return torch.digamma(
        x.unsqueeze(-1)
        - torch.arange(p, dtype=x.dtype, device=x.device).div(2).expand(x.shape + (-1,))
    ).sum(-1)

def _clamp_above_eps(x: torch.Tensor) -> torch.Tensor:
    """
    将输入张量中小于机器精度的元素设置为机器精度

    Args:
        x (torch.Tensor): 输入张量（假定为正数）

    Returns:
        torch.Tensor: 处理后的张量
    """
    # 假设输入是正数，将小于机器精度的值设置为机器精度
    return x.clamp(min=torch.finfo(x.dtype).eps)

class Wishart(ExponentialFamily):
    """
    Wishart 分布类，参数化为对称正定矩阵 Sigma 或其 Cholesky 分解 Sigma = LL^T

    Args:
        df (float or Tensor): 自由度参数，大于 (矩阵维度 - 1)
        covariance_matrix (Tensor): 正定协方差矩阵
        precision_matrix (Tensor): 正定精度矩阵
        scale_tril (Tensor): 协方差的下三角 Cholesky 分解，对角线元素为正数

    Note:
        只能指定其中之一：covariance_matrix、precision_matrix 或 scale_tril。
        使用 scale_tril 将更高效：所有内部计算都基于 scale_tril。
        如果传入 covariance_matrix 或 precision_matrix，则仅用于使用 Cholesky 分解计算对应的下三角矩阵。
        'torch.distributions.LKJCholesky' 是受限制的 Wishart 分布的特例。

    Example:
        >>> # xdoctest: +SKIP("FIXME: scale_tril must be at least two-dimensional")
        >>> m = Wishart(torch.Tensor([2]), covariance_matrix=torch.eye(2))
        >>> m.sample()  # Wishart 分布，均值为 `df * I`，对角元素方差为 `df`，非对角元素方差为 `2 * df`

    References:
        [1] Wang, Z., Wu, Y. and Chu, H., 2018. `On equivalence of the LKJ distribution and the restricted Wishart distribution`.
        [2] Sawyer, S., 2007. `Wishart Distributions and Inverse-Wishart Sampling`.
        [3] Anderson, T. W., 2003. `An Introduction to Multivariate Statistical Analysis (3rd ed.)`.
        [4] Odell, P. L. & Feiveson, A. H., 1966. `A Numerical Procedure to Generate a SampleCovariance Matrix`. JASA, 61(313):199-203.
        [5] Ku, Y.-C. & Bloomfield, P., 2010. `Generating Random Wishart Matrices with Fractional Degrees of Freedom in OX`.
    """
    # 定义参数约束字典，指定每个参数的约束条件
    arg_constraints = {
        "covariance_matrix": constraints.positive_definite,  # 协方差矩阵必须是正定的约束
        "precision_matrix": constraints.positive_definite,   # 精度矩阵必须是正定的约束
        "scale_tril": constraints.lower_cholesky,            # 下三角矩阵必须是下三角的约束（用于 Cholesky 分解）
        "df": constraints.greater_than(0),                   # 自由度必须大于0的约束
    }
    
    # 支持正定约束
    support = constraints.positive_definite
    
    # 存在 rsample 方法的标志
    has_rsample = True
    
    # 平均载体测度（未注明其具体含义的变量名）
    _mean_carrier_measure = 0
    
    # 定义初始化方法
    def __init__(
        self,
        df: Union[torch.Tensor, Number],                     # 自由度参数，可以是张量或数值
        covariance_matrix: Optional[torch.Tensor] = None,    # 协方差矩阵，可选参数
        precision_matrix: Optional[torch.Tensor] = None,     # 精度矩阵，可选参数
        scale_tril: Optional[torch.Tensor] = None,           # 下三角矩阵，可选参数
        validate_args=None,
        ):
            # 确保仅有一个参数（协方差矩阵、精度矩阵或下三角矩阵）被指定
            assert (covariance_matrix is not None) + (scale_tril is not None) + (
                precision_matrix is not None
            ) == 1, "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."

            # 选择非空的参数赋值给param
            param = next(
                p
                for p in (covariance_matrix, precision_matrix, scale_tril)
                if p is not None
            )

            # 检查参数的维度，scale_tril至少是二维的，可以有额外的批次维度
            if param.dim() < 2:
                raise ValueError(
                    "scale_tril must be at least two-dimensional, with optional leading batch dimensions"
                )

            # 如果df是数字，创建一个与param形状相同的tensor
            if isinstance(df, Number):
                batch_shape = torch.Size(param.shape[:-2])
                self.df = torch.tensor(df, dtype=param.dtype, device=param.device)
            else:
                # 否则使用广播方式扩展df的形状与param的批次形状匹配
                batch_shape = torch.broadcast_shapes(param.shape[:-2], df.shape)
                self.df = df.expand(batch_shape)
            event_shape = param.shape[-2:]

            # 检查df是否小于event_shape[-1] - 1中的任何值，如果是则引发异常
            if self.df.le(event_shape[-1] - 1).any():
                raise ValueError(
                    f"Value of df={df} expected to be greater than ndim - 1 = {event_shape[-1]-1}."
                )

            # 根据参数类型，将参数扩展到相同的批次和事件形状
            if scale_tril is not None:
                self.scale_tril = param.expand(batch_shape + (-1, -1))
            elif covariance_matrix is not None:
                self.covariance_matrix = param.expand(batch_shape + (-1, -1))
            elif precision_matrix is not None:
                self.precision_matrix = param.expand(batch_shape + (-1, -1))

            # 约束df参数的最小值，使其大于等于event_shape[-1] - 1
            self.arg_constraints["df"] = constraints.greater_than(event_shape[-1] - 1)
            # 如果df的某些值小于event_shape[-1]，发出警告
            if self.df.lt(event_shape[-1]).any():
                warnings.warn(
                    "Low df values detected. Singular samples are highly likely to occur for ndim - 1 < df < ndim."
                )

            # 调用父类的初始化方法
            super().__init__(batch_shape, event_shape, validate_args=validate_args)
            # 初始化_batch_dims属性，用于指定批次维度
            self._batch_dims = [-(x + 1) for x in range(len(self._batch_shape))]

            # 根据不同的参数类型初始化_unbroadcasted_scale_tril属性
            if scale_tril is not None:
                self._unbroadcasted_scale_tril = scale_tril
            elif covariance_matrix is not None:
                self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
            else:  # precision_matrix is not None
                self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

            # 初始化_chi2分布对象，用于Bartlett分解采样
            self._dist_chi2 = torch.distributions.chi2.Chi2(
                df=(
                    self.df.unsqueeze(-1)
                    - torch.arange(
                        self._event_shape[-1],
                        dtype=self._unbroadcasted_scale_tril.dtype,
                        device=self._unbroadcasted_scale_tril.device,
                    ).expand(batch_shape + (-1,))
                )
            )
    # 扩展当前 Wishart 分布实例的批量形状和实例特性
    def expand(self, batch_shape, _instance=None):
        # 获取一个新的 Wishart 分布实例，确保类型和实例正确
        new = self._get_checked_instance(Wishart, _instance)
        # 将输入的批量形状转换为 torch.Size 对象
        batch_shape = torch.Size(batch_shape)
        # 计算扩展后的协方差矩阵形状
        cov_shape = batch_shape + self.event_shape
        # 扩展未广播的下三角矩阵
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril.expand(cov_shape)
        # 扩展自由度参数
        new.df = self.df.expand(batch_shape)

        # 设置批次维度的负数索引
        new._batch_dims = [-(x + 1) for x in range(len(batch_shape))]

        # 如果当前实例包含协方差矩阵属性，则进行扩展
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        # 如果当前实例包含下三角矩阵属性，则进行扩展
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        # 如果当前实例包含精度矩阵属性，则进行扩展
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)

        # 为 Bartlett 分解采样需要的 Chi2 分布
        new._dist_chi2 = torch.distributions.chi2.Chi2(
            df=(
                new.df.unsqueeze(-1)
                - torch.arange(
                    self.event_shape[-1],
                    dtype=new._unbroadcasted_scale_tril.dtype,
                    device=new._unbroadcasted_scale_tril.device,
                ).expand(batch_shape + (-1,))
            )
        )

        # 调用父类初始化方法，设定验证参数为 False
        super(Wishart, new).__init__(batch_shape, self.event_shape, validate_args=False)
        # 继承当前实例的参数验证属性
        new._validate_args = self._validate_args
        # 返回扩展后的新 Wishart 分布实例
        return new

    # 惰性属性：返回扩展后的下三角矩阵
    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape
        )

    # 惰性属性：返回扩展后的协方差矩阵
    @lazy_property
    def covariance_matrix(self):
        return (
            self._unbroadcasted_scale_tril
            @ self._unbroadcasted_scale_tril.transpose(-2, -1)
        ).expand(self._batch_shape + self._event_shape)

    # 惰性属性：返回扩展后的精度矩阵
    @lazy_property
    def precision_matrix(self):
        # 创建单位矩阵，并用 Cholesky 分解求解扩展后的精度矩阵
        identity = torch.eye(
            self._event_shape[-1],
            device=self._unbroadcasted_scale_tril.device,
            dtype=self._unbroadcasted_scale_tril.dtype,
        )
        return torch.cholesky_solve(identity, self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape
        )

    # 属性：返回均值矩阵
    @property
    def mean(self):
        return self.df.view(self._batch_shape + (1, 1)) * self.covariance_matrix

    # 属性：返回众数矩阵
    @property
    def mode(self):
        # 计算众数的因子，并视情况设置为 NaN
        factor = self.df - self.covariance_matrix.shape[-1] - 1
        factor[factor <= 0] = nan
        return factor.view(self._batch_shape + (1, 1)) * self.covariance_matrix

    # 属性：返回方差矩阵
    @property
    def variance(self):
        # 获取协方差矩阵的对角线元素
        V = self.covariance_matrix  # has shape (batch_shape x event_shape)
        diag_V = V.diagonal(dim1=-2, dim2=-1)
        # 计算并返回方差矩阵
        return self.df.view(self._batch_shape + (1, 1)) * (
            V.pow(2) + torch.einsum("...i,...j->...ij", diag_V, diag_V)
        )
    def _bartlett_sampling(self, sample_shape=torch.Size()):
        p = self._event_shape[-1]  # 获取事件形状的最后一个维度，通常为1

        # 使用Bartlett分解实现采样
        noise = _clamp_above_eps(
            self._dist_chi2.rsample(sample_shape).sqrt()
        ).diag_embed(dim1=-2, dim2=-1)  # 对协方差矩阵进行修正和对角化处理

        i, j = torch.tril_indices(p, p, offset=-1)  # 获取下三角矩阵的索引
        noise[..., i, j] = torch.randn(
            torch.Size(sample_shape) + self._batch_shape + (int(p * (p - 1) / 2),),
            dtype=noise.dtype,
            device=noise.device,
        )  # 生成符合正态分布的噪声，并将其填充到下三角矩阵的对应位置
        chol = self._unbroadcasted_scale_tril @ noise  # 计算Cholesky分解

        return chol @ chol.transpose(-2, -1)  # 返回Cholesky分解后的乘积

    def rsample(self, sample_shape=torch.Size(), max_try_correction=None):
        r"""
        .. warning::
            在某些情况下，基于Bartlett分解的采样算法可能返回奇异矩阵样本。
            默认情况下会尝试多次修正奇异样本，但可能仍然返回奇异矩阵样本。
            奇异样本可能会导致`.log_prob()`返回`-inf`值。
            在这些情况下，用户应验证样本，并修正`df`的值或根据需要调整`.rsample`的`max_try_correction`参数值。

        """

        if max_try_correction is None:
            max_try_correction = 3 if torch._C._get_tracing_state() else 10  # 设置最大尝试修正次数

        sample_shape = torch.Size(sample_shape)
        sample = self._bartlett_sampling(sample_shape)  # 调用Bartlett采样函数获取样本

        # 以下部分是为了提高数值稳定性而进行的临时处理，在将来应予以删除
        is_singular = self.support.check(sample)  # 检查样本是否奇异
        if self._batch_shape:
            is_singular = is_singular.amax(self._batch_dims)  # 如果有批次形状，则取批次维度中的最大值

        if torch._C._get_tracing_state():
            # JIT编译时的较少优化版本
            for _ in range(max_try_correction):
                sample_new = self._bartlett_sampling(sample_shape)  # 重新进行Bartlett采样
                sample = torch.where(is_singular, sample_new, sample)  # 如果样本奇异，则使用新样本替换旧样本

                is_singular = ~self.support.check(sample)  # 再次检查样本是否奇异
                if self._batch_shape:
                    is_singular = is_singular.amax(self._batch_dims)  # 如果有批次形状，则取批次维度中的最大值

        else:
            # 数据依赖控制流的优化版本
            if is_singular.any():
                warnings.warn("检测到奇异样本。")

                for _ in range(max_try_correction):
                    sample_new = self._bartlett_sampling(is_singular[is_singular].shape)  # 使用奇异样本的形状进行采样
                    sample[is_singular] = sample_new  # 替换奇异样本

                    is_singular_new = ~self.support.check(sample_new)  # 检查新样本是否奇异
                    if self._batch_shape:
                        is_singular_new = is_singular_new.amax(self._batch_dims)  # 如果有批次形状，则取批次维度中的最大值
                    is_singular[is_singular.clone()] = is_singular_new

                    if not is_singular.any():
                        break  # 如果没有奇异样本，则退出循环

        return sample  # 返回最终的样本
    # 计算对数概率密度函数（log probability）的方法，接受一个值作为参数
    def log_prob(self, value):
        # 如果启用参数验证，则验证样本值是否有效
        if self._validate_args:
            self._validate_sample(value)
        # 获取自由度（degrees of freedom），形状为 (batch_shape)
        nu = self.df
        # 获取事件形状中的最后一个维度大小，应为单例形状
        p = self._event_shape[-1]
        # 返回对数概率密度函数的计算结果
        return (
            # 主要公式部分，按照多元 t 分布的对数概率密度函数计算
            -nu
            * (
                p * _log_2 / 2
                + self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1)
                .log()
                .sum(-1)
            )
            - torch.mvlgamma(nu / 2, p=p)
            + (nu - p - 1) / 2 * torch.linalg.slogdet(value).logabsdet
            - torch.cholesky_solve(value, self._unbroadcasted_scale_tril)
            .diagonal(dim1=-2, dim2=-1)
            .sum(dim=-1)
            / 2
        )

    # 计算分布的熵（entropy）的方法
    def entropy(self):
        # 获取自由度（degrees of freedom），形状为 (batch_shape)
        nu = self.df
        # 获取事件形状中的最后一个维度大小，应为单例形状
        p = self._event_shape[-1]
        # 获取协方差矩阵（covariance matrix），形状为 (batch_shape x event_shape)
        V = self.covariance_matrix
        # 返回分布的熵的计算结果
        return (
            # 主要公式部分，按照多元 t 分布的熵计算
            (p + 1)
            * (
                p * _log_2 / 2
                + self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1)
                .log()
                .sum(-1)
            )
            + torch.mvlgamma(nu / 2, p=p)
            - (nu - p - 1) / 2 * _mvdigamma(nu / 2, p=p)
            + nu * p / 2
        )

    # 返回分布的自然参数（natural parameters）
    @property
    def _natural_params(self):
        # 获取自由度（degrees of freedom），形状为 (batch_shape)
        nu = self.df
        # 获取事件形状中的最后一个维度大小，应为单例形状
        p = self._event_shape[-1]
        # 返回自然参数的计算结果
        return -self.precision_matrix / 2, (nu - p - 1) / 2

    # 计算分布的对数归一化常数（log normalizer）
    def _log_normalizer(self, x, y):
        # 获取事件形状中的最后一个维度大小，应为单例形状
        p = self._event_shape[-1]
        # 返回对数归一化常数的计算结果
        return (y + (p + 1) / 2) * (
            -torch.linalg.slogdet(-2 * x).logabsdet + _log_2 * p
        ) + torch.mvlgamma(y + (p + 1) / 2, p=p)
```