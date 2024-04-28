# `.\transformers\time_series_utils.py`

```py
# 定义编码格式为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有
# 版权声明，版权归 2018 Amazon.com, Inc. 或其关联公司所有
#
# 根据 Apache 许可证，版本 2.0 进行许可
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发
# 没有任何明示或暗示的担保或条件
# 请参阅许可证以获取特定语言下的权限和限制
"""
时间序列分布输出类和实用工具。
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
        # 初始化 AffineTransformed 类
        self.scale = 1.0 if scale is None else scale
        self.loc = 0.0 if loc is None else loc

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
        # 初始化 ParameterProjection 类
        super().__init__(**kwargs)
        self.args_dim = args_dim
        # 使用 Linear 层创建线性变换的模块列表
        self.proj = nn.ModuleList([nn.Linear(in_features, dim) for dim in args_dim.values()])
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # 执行前向传播，将输入映射到参数空间
        params_unbounded = [proj(x) for proj in self.proj]

        return self.domain_map(*params_unbounded)


class LambdaLayer(nn.Module):
    def __init__(self, function):
        # 初始化 LambdaLayer 类
        super().__init__()
        self.function = function

    def forward(self, x, *args):
        # 执行前向传播，应用给定的函数
        return self.function(x, *args)


class DistributionOutput:
    distribution_class: type
    in_features: int
    args_dim: Dict[str, int]

    def __init__(self, dim: int = 1) -> None:
        # 初始化 DistributionOutput 类
        self.dim = dim
        # 将参数的维度乘以输出的维度
        self.args_dim = {k: dim * self.args_dim[k] for k in self.args_dim}
    # 根据参数创建基本的分布对象
    def _base_distribution(self, distr_args):
        # 如果维度为1，则直接创建分布对象
        if self.dim == 1:
            return self.distribution_class(*distr_args)
        # 如果维度不为1，则创建独立分布对象
        else:
            return Independent(self.distribution_class(*distr_args), 1)

    # 创建分布对象
    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        # 基于给定参数创建分布对象
        distr = self._base_distribution(distr_args)
        # 如果 loc 和 scale 参数都为 None，则直接返回分布对象
        if loc is None and scale is None:
            return distr
        # 否则，返回经过仿射变换的分布对象
        else:
            return AffineTransformed(distr, loc=loc, scale=scale, event_dim=self.event_dim)

    # 返回事件形状，即分布对象构建的每个单独事件的形状
    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distributions that this object constructs.
        """
        return () if self.dim == 1 else (self.dim,)

    # 返回事件维度，即分布对象构建的分布的事件维度数量
    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple, of the distributions that this object
        constructs.
        """
        return len(self.event_shape)

    # 返回支持中的值，用于计算对应分布的对数损失
    @property
    def value_in_support(self) -> float:
        r"""
        A float that will have a valid numeric value when computing the log-loss of the corresponding distribution. By
        default 0.0. This value will be used when padding data series.
        """
        return 0.0

    # 返回参数投影层，将输入映射到分布的适当参数
    def get_parameter_projection(self, in_features: int) -> nn.Module:
        r"""
        Return the parameter projection layer that maps the input to the appropriate parameters of the distribution.
        """
        return ParameterProjection(
            in_features=in_features,
            args_dim=self.args_dim,
            domain_map=LambdaLayer(self.domain_map),
        )

    # 将输入参数映射到正确的形状和域
    def domain_map(self, *args: torch.Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends on the type of distribution, while the
        correct shape is obtained by reshaping the trailing axis in such a way that the returned tensors define a
        distribution of the right event_shape.
        """
        raise NotImplementedError()

    # 将输入映射到正半轴的辅助函数，通过应用平方加操作
    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        return (x + torch.sqrt(torch.square(x) + 4.0)) / 2.0
class StudentTOutput(DistributionOutput):
    """
    Student-T distribution output class.
    """

    # 定义参数维度字典，包括自由度(df)、位置(loc)和尺度(scale)
    args_dim: Dict[str, int] = {"df": 1, "loc": 1, "scale": 1}
    # 指定分布类为 Student-T
    distribution_class: type = StudentT

    @classmethod
    def domain_map(cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        # 对尺度进行平方加操作，并限制最小值为浮点数精度
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        # 对自由度进行平方加操作
        df = 2.0 + cls.squareplus(df)
        # 去除维度为1的维度
        return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


class NormalOutput(DistributionOutput):
    """
    Normal distribution output class.
    """

    # 定义参数维度字典，包括位置(loc)和尺度(scale)
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    # 指定分布类为 Normal
    distribution_class: type = Normal

    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor):
        # 对尺度进行平方加操作，并限制最小值为浮点数精度
        scale = cls.squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
        # 去除维度为1的维度
        return loc.squeeze(-1), scale.squeeze(-1)


class NegativeBinomialOutput(DistributionOutput):
    """
    Negative Binomial distribution output class.
    """

    # 定义参数维度字典，包括总数(total_count)和对数概率(logits)
    args_dim: Dict[str, int] = {"total_count": 1, "logits": 1}
    # 指定分布类为 Negative Binomial
    distribution_class: type = NegativeBinomial

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):
        # 对总数进行平方加操作
        total_count = cls.squareplus(total_count)
        # 去除维度为1的维度
        return total_count.squeeze(-1), logits.squeeze(-1)

    def _base_distribution(self, distr_args) -> Distribution:
        total_count, logits = distr_args
        if self.dim == 1:
            return self.distribution_class(total_count=total_count, logits=logits)
        else:
            return Independent(self.distribution_class(total_count=total_count, logits=logits), 1)

    # 覆盖父类方法。由于负二项分布应返回整数，因此不能使用仿射变换进行缩放。相反，我们对参数进行缩放。
    def distribution(
        self, distr_args, loc: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None
    ) -> Distribution:
        total_count, logits = distr_args

        if scale is not None:
            # 查看 Gamma 分布的缩放属性
            logits += scale.log()

        return self._base_distribution((total_count, logits))
```