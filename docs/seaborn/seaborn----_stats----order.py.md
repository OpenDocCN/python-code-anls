# `D:\src\scipysrc\seaborn\seaborn\_stats\order.py`

```
from __future__ import annotations
# 导入用于支持 Python 3.7+ 类型注解的模块

from dataclasses import dataclass
# 导入用于创建数据类的装饰器

from typing import ClassVar, cast
# 导入类型提示相关的模块和函数

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
# 尝试导入标准库中的 Literal 类型提示，如果失败则从 typing_extensions 中导入（仅在需要时忽略错误）

import numpy as np
# 导入 NumPy 库

from pandas import DataFrame
# 从 Pandas 库中导入 DataFrame 类

from seaborn._core.scales import Scale
# 从 Seaborn 库中导入 Scale 类

from seaborn._core.groupby import GroupBy
# 从 Seaborn 库中导入 GroupBy 类

from seaborn._stats.base import Stat
# 从 Seaborn 库中导入 Stat 类

from seaborn.utils import _version_predates
# 从 Seaborn 工具模块中导入版本检测函数


# 定义一个字面类型，包含不同的方法名，用于指定方法的插值方法
_MethodKind = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]


@dataclass
# 使用 dataclass 装饰器定义 Perc 类，继承自 Stat 类
class Perc(Stat):
    """
    Replace observations with percentile values.

    Parameters
    ----------
    k : list of numbers or int
        If a list of numbers, this gives the percentiles (in [0, 100]) to compute.
        If an integer, compute `k` evenly-spaced percentiles between 0 and 100.
        For example, `k=5` computes the 0, 25, 50, 75, and 100th percentiles.
    method : str
        Method for interpolating percentiles between observed datapoints.
        See :func:`numpy.percentile` for valid options and more information.

    Examples
    --------
    .. include:: ../docstrings/objects.Perc.rst

    """
    k: int | list[float] = 5
    # k 参数可以是整数或浮点数列表，默认为 5

    method: str = "linear"
    # method 参数指定插值方法，默认为 "linear"

    group_by_orient: ClassVar[bool] = True
    # 类变量 group_by_orient，默认为 True

    def _percentile(self, data: DataFrame, var: str) -> DataFrame:
        # 私有方法 _percentile 用于计算百分位数

        k = list(np.linspace(0, 100, self.k)) if isinstance(self.k, int) else self.k
        # 根据参数 k 的类型，生成百分位数的列表

        method = cast(_MethodKind, self.method)
        # 将 method 转换为 _MethodKind 类型

        values = data[var].dropna()
        # 获取数据中指定变量 var 的值，且去除缺失值

        if _version_predates(np, "1.22"):
            res = np.percentile(values, k, interpolation=method)  # type: ignore
            # 如果 NumPy 版本较旧，则使用指定插值方法计算百分位数
        else:
            res = np.percentile(data[var].dropna(), k, method=method)
            # 否则，使用指定插值方法计算百分位数

        return DataFrame({var: res, "percentile": k})
        # 返回包含计算结果的 DataFrame 对象

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        # 实现调用对象时的行为，接受数据框、分组对象、方向和刻度作为参数，返回 DataFrame 对象

        var = {"x": "y", "y": "x"}[orient]
        # 根据方向 orient 选择对应的变量名

        return groupby.apply(data, self._percentile, var)
        # 使用 groupby 对象将数据应用于 _percentile 方法，并返回结果 DataFrame
```