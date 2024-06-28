# `.\time_series_utils.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明，声明代码的版权归属
# 版权声明，版权归 Amazon.com, Inc. 或其关联公司所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”提供的，没有任何明示或暗示的保证或条件
# 请参阅许可证以了解特定语言下的许可条件
"""
时间序列分布输出类和实用程序。
"""
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributions import (
    AffineTransform,
    Distribution,
    Independent,
    NegativeBinomial,
    Normal,
    StudentT,
    TransformedDistribution,
)

class AffineTransformed(TransformedDistribution):
    def __init__(self, base_distribution: Distribution, loc=None, scale=None, event_dim=0):
        # 如果 loc 未提供，默认为 0.0
        self.loc = 0.0 if loc is None else loc
        # 如果 scale 未提供，默认为 1.0
        self.scale = 1.0 if scale is None else scale

        # 调用父类的初始化方法，使用 AffineTransform 将 loc 和 scale 应用于基本分布
        super().__init__(base_distribution, [AffineTransform(loc=self.loc, scale=self.scale, event_dim=event_dim)])

    @property
    def mean(self):
        """
        返回分布的均值。
        """
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self):
        """
        返回分布的方差。
        """
        return self.base_dist.variance * self.scale**2

    @property
    def stddev(self):
        """
        返回分布的标准差。
        """
        return self.variance.sqrt()


class ParameterProjection(nn.Module):
    def __init__(
        self, in_features: int, args_dim: Dict[str, int], domain_map: Callable[..., Tuple[torch.Tensor]], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # 参数维度字典，映射输入特征维度到每个参数的维度
        self.args_dim = args_dim
        # 使用 nn.Linear 创建一系列线性映射模块，将输入特征映射到每个参数的维度
        self.proj = nn.ModuleList([nn.Linear(in_features, dim) for dim in args_dim.values()])
        # 域映射函数，将未限制的参数映射到定义域
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # 对输入 x 应用所有的线性映射，得到未限制的参数列表
        params_unbounded = [proj(x) for proj in self.proj]

        # 使用域映射函数将未限制的参数映射到定义域，返回映射后的参数元组
        return self.domain_map(*params_unbounded)


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super().__init__()
        # 初始化 LambdaLayer 类时传入的函数对象
        self.function = function

    def forward(self, x, *args):
        # 调用传入的函数对象，传入 x 和其他参数 args，返回结果
        return self.function(x, *args)


class DistributionOutput:
    # 分布输出的类别
    distribution_class: type
    # 输入特征的维度
    in_features: int
    # 参数维度的字典
    args_dim: Dict[str, int]

    def __init__(self, dim: int = 1) -> None:
        # 初始化分布输出对象的维度属性
        self.dim = dim
        # 参数维度的字典，每个参数的维度乘以 dim
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}
    # 根据给定的参数创建一个分布对象，如果维度为1，则直接创建分布对象；否则，创建一个独立分布对象
    def _base_distribution(self, distr_args):
        if self.dim == 1:
            return self.distribution_class(*distr_args)
        else:
            return Independent(self.distribution_class(*distr_args), 1)

    # 根据基本分布对象创建一个分布对象，并根据需要添加仿射变换
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        distr = self._base_distribution(distr_args)
        if loc is None and scale is None:
            return distr
        else:
            return AffineTransformed(distr, loc=loc, scale=scale, event_dim=self.event_dim)

    # 返回事件形状的元组，如果维度为1，则返回空元组；否则返回包含维度大小的元组
    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distributions that this object constructs.
        """
        return () if self.dim == 1 else (self.dim,)

    # 返回事件维度的整数值，即事件形状元组的长度
    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple, of the distributions that this object
        constructs.
        """
        return len(self.event_shape)

    # 返回支持域中的数值，用于计算对应分布的对数损失，默认为0.0，用于数据序列的填充
    @property
    def value_in_support(self) -> float:
        r"""
        A float that will have a valid numeric value when computing the log-loss of the corresponding distribution. By
        default 0.0. This value will be used when padding data series.
        """
        return 0.0

    # 返回将输入映射到分布参数的参数投影层
    def get_parameter_projection(self, in_features: int) -> nn.Module:
        r"""
        Return the parameter projection layer that maps the input to the appropriate parameters of the distribution.
        """
        return ParameterProjection(
            in_features=in_features,
            args_dim=self.args_dim,
            domain_map=LambdaLayer(self.domain_map),
        )

    # 将输入参数转换为正确的形状和域，具体形状取决于分布类型，需对末尾轴进行重塑以定义正确事件形状的分布
    def domain_map(self, *args: torch.Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends on the type of distribution, while the
        correct shape is obtained by reshaping the trailing axis in such a way that the returned tensors define a
        distribution of the right event_shape.
        """
        raise NotImplementedError()

    # 静态方法：通过应用square-plus操作将输入映射到正半轴上
    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0
# 学生 t 分布输出类，继承自 DistributionOutput 类
class StudentTOutput(DistributionOutput):
    """
    Student-T distribution output class.
    """

    # 定义参数维度的字典，包括自由度 df、位置 loc、标度 scale
    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    # 分布类为 StudentT
    distribution_class: type = StudentT

    @classmethod
    # 将域映射方法，处理输入的张量参数 df、loc、scale
    def domain_map(cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        # 对 scale 进行平方后加上正值（避免负数），并限制下限为浮点数精度的最小正数
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        # 对 df 进行平方后加上正值
        df = 2.0 + cls.squareplus(df)
        # 去除最后一个维度的squeeze操作，返回处理后的 df、loc、scale 参数
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


# 正态分布输出类，继承自 DistributionOutput 类
class NormalOutput(DistributionOutput):
    """
    Normal distribution output class.
    """

    # 定义参数维度的字典，包括位置 loc、标度 scale
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    # 分布类为 Normal
    distribution_class: type = Normal

    @classmethod
    # 将域映射方法，处理输入的张量参数 loc、scale
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):
        # 对 scale 进行平方后加上正值（避免负数），并限制下限为浮点数精度的最小正数
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        # 去除最后一个维度的squeeze操作，返回处理后的 loc、scale 参数
        return loc.squeeze(-1), scale.squeeze(-1)


# 负二项分布输出类，继承自 DistributionOutput 类
class NegativeBinomialOutput(DistributionOutput):
    """
    Negative Binomial distribution output class.
    """

    # 定义参数维度的字典，包括总数 total_count、logits
    args_dim: Dict[str, int] = {"total_count": 1, "logits": 1}
    # 分布类为 NegativeBinomial
    distribution_class: type = NegativeBinomial

    @classmethod
    # 将域映射方法，处理输入的张量参数 total_count、logits
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):
        # 对 total_count 进行平方后加上正值
        total_count = cls.squareplus(total_count)
        # 去除最后一个维度的squeeze操作，返回处理后的 total_count、logits 参数
        return total_count.squeeze(-1), logits.squeeze(-1)

    # 重写父类方法，根据维度返回对应的分布对象
    def _base_distribution(self, distr_args) -> Distribution:
        total_count, logits = distr_args
        # 如果维度为 1，则返回负二项分布对象
        if self.dim == 1:
            return self.distribution_class(total_count=total_count, logits=logits)
        else:
            # 否则返回独立分布对象
            return Independent(self.distribution_class(total_count=total_count, logits=logits), 1)

    # 覆盖父类方法，用于计算分布，根据需求调整 logits 参数
    def distribution(
        self, distr_args, loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        total_count, logits = distr_args

        if scale is not None:
            # 根据 Gamma 分布的缩放属性调整 logits 参数
            logits += scale.log()

        # 返回基础分布对象的计算结果
        return self._base_distribution((total_count, logits))
```