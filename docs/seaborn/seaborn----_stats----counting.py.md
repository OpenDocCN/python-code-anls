# `D:\src\scipysrc\seaborn\seaborn\_stats\counting.py`

```
# 导入未来版本的类型注解支持
from __future__ import annotations
# 导入用于数据类的装饰器
from dataclasses import dataclass
# 导入类型提示工具
from typing import ClassVar

# 导入科学计算库 NumPy
import numpy as np
# 导入数据处理和分析库 Pandas
import pandas as pd
# 从 Pandas 中导入 DataFrame 类型
from pandas import DataFrame

# 从 Seaborn 中导入用于分组操作的 GroupBy 类
from seaborn._core.groupby import GroupBy
# 从 Seaborn 中导入比例尺操作的 Scale 类
from seaborn._core.scales import Scale
# 从 Seaborn 中导入统计操作的基础类 Stat
from seaborn._stats.base import Stat

# 导入类型检查工具，用于静态类型检查
from typing import TYPE_CHECKING
# 如果处于类型检查模式下
if TYPE_CHECKING:
    # 从 NumPy 的类型提示中导入 ArrayLike 类型

@dataclass
# 统计类 Count 继承自统计基类 Stat
class Count(Stat):
    """
    Count distinct observations within groups.

    See Also
    --------
    Hist : A more fully-featured transform including binning and/or normalization.

    Examples
    --------
    .. include:: ../docstrings/objects.Count.rst

    """
    # 类变量，指示分组方向是否默认为 True
    group_by_orient: ClassVar[bool] = True

    # 定义类的调用方法，计算分组内的观察值数量并返回 DataFrame
    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:

        # 根据 orient 参数确定要聚合的变量名称
        var = {"x": "y", "y": "x"}[orient]
        # 执行聚合操作：根据分组对象 groupby，对数据进行赋值和长度计数
        res = (
            groupby
            .agg(data.assign(**{var: data[orient]}), {var: len})
            .dropna(subset=["x", "y"])  # 删除包含空值的行
            .reset_index(drop=True)    # 重置索引并丢弃原索引
        )
        # 返回聚合结果 DataFrame
        return res


@dataclass
# 统计类 Hist 继承自统计基类 Stat
class Hist(Stat):
    """
    Bin observations, count them, and optionally normalize or cumulate.

    Parameters
    ----------
    stat : str
        Aggregate statistic to compute in each bin:

        - `count`: the number of observations
        - `density`: normalize so that the total area of the histogram equals 1
        - `percent`: normalize so that bar heights sum to 100
        - `probability` or `proportion`: normalize so that bar heights sum to 1
        - `frequency`: divide the number of observations by the bin width

    bins : str, int, or ArrayLike
        Generic parameter that can be the name of a reference rule, the number
        of bins, or the bin breaks. Passed to :func:`numpy.histogram_bin_edges`.
    binwidth : float
        Width of each bin; overrides `bins` but can be used with `binrange`.
        Note that if `binwidth` does not evenly divide the bin range, the actual
        bin width used will be only approximately equal to the parameter value.
    binrange : (min, max)
        Lowest and highest value for bin edges; can be used with either
        `bins` (when a number) or `binwidth`. Defaults to data extremes.
    common_norm : bool or list of variables
        When not `False`, the normalization is applied across groups. Use
        `True` to normalize across all groups, or pass variable name(s) that
        define normalization groups.
    common_bins : bool or list of variables
        When not `False`, the same bins are used for all groups. Use `True` to
        share bins across all groups, or pass variable name(s) to share within.
    cumulative : bool
        If True, cumulate the bin values.
    discrete : bool
        If True, set `binwidth` and `binrange` so that bins have unit width and
        are centered on integer values

    Notes
    -----
    The choice of bins for computing and plotting a histogram can exert
    """
    stat: str = "count"
    bins: str | int | ArrayLike = "auto"
    binwidth: float | None = None
    binrange: tuple[float, float] | None = None
    common_norm: bool | list[str] = True
    common_bins: bool | list[str] = True
    cumulative: bool = False
    discrete: bool = False

    def __post_init__(self):
        # 定义统计参数的选项列表
        stat_options = [
            "count", "density", "percent", "probability", "proportion", "frequency"
        ]
        # 检查 stat 参数是否在合法选项中
        self._check_param_one_of("stat", stat_options)

    def _define_bin_edges(self, vals, weight, bins, binwidth, binrange, discrete):
        """Inner function that takes bin parameters as arguments."""
        # 处理无穷大和缺失值
        vals = vals.replace(-np.inf, np.nan).replace(np.inf, np.nan).dropna()

        if binrange is None:
            # 根据数据的最小值和最大值确定起始和结束点
            start, stop = vals.min(), vals.max()
        else:
            # 如果指定了 binrange，则使用指定的范围
            start, stop = binrange

        if discrete:
            # 如果是离散数据，使用半开区间进行边界定义
            bin_edges = np.arange(start - .5, stop + 1.5)
        else:
            if binwidth is not None:
                # 根据指定的 binwidth 计算 bins 的数量
                bins = int(round((stop - start) / binwidth))
            # 使用 numpy 的直方图函数确定 bin 的边界
            bin_edges = np.histogram_bin_edges(vals, bins, binrange, weight)

        # TODO warning or cap on too many bins?

        return bin_edges

    def _define_bin_params(self, data, orient, scale_type):
        """Given data, return numpy.histogram parameters to define bins."""
        vals = data[orient]
        weights = data.get("weight", None)

        # TODO We'll want this for ordinal / discrete scales too
        # (Do we need discrete as a parameter or just infer from scale?)
        discrete = self.discrete or scale_type == "nominal"

        # 使用内部函数 _define_bin_edges 定义 bin 的边界
        bin_edges = self._define_bin_edges(
            vals, weights, self.bins, self.binwidth, self.binrange, discrete,
        )

        if isinstance(self.bins, (str, int)):
            # 如果 bins 参数是字符串或整数，确定 bins 的数量和范围
            n_bins = len(bin_edges) - 1
            bin_range = bin_edges.min(), bin_edges.max()
            bin_kws = dict(bins=n_bins, range=bin_range)
        else:
            # 否则直接使用 bin_edges 参数
            bin_kws = dict(bins=bin_edges)

        return bin_kws
    ```
    # 定义内部方法 `_get_bins_and_eval`，接受数据、方向、分组、缩放类型作为参数
    def _get_bins_and_eval(self, data, orient, groupby, scale_type):
        # 调用 `_define_bin_params` 方法获取直方图参数
        bin_kws = self._define_bin_params(data, orient, scale_type)
        # 将数据应用到 `groupby` 函数，使用 `_eval` 方法评估数据，传入方向和直方图参数
        return groupby.apply(data, self._eval, orient, bin_kws)

    # 定义内部方法 `_eval`，接受数据、方向和直方图参数作为参数
    def _eval(self, data, orient, bin_kws):
        # 从数据中获取特定方向的值
        vals = data[orient]
        # 获取可能的权重数据
        weights = data.get("weight", None)

        # 判断是否为密度图
        density = self.stat == "density"
        # 使用 numpy 的直方图函数计算直方图 `hist` 和边界 `edges`
        hist, edges = np.histogram(vals, **bin_kws, weights=weights, density=density)

        # 计算直方图条的宽度和中心
        width = np.diff(edges)
        center = edges[:-1] + width / 2

        # 返回一个包含中心、计数和宽度的 pandas DataFrame
        return pd.DataFrame({orient: center, "count": hist, "space": width})

    # 定义内部方法 `_normalize`，接受数据作为参数
    def _normalize(self, data):
        # 获取直方图计数
        hist = data["count"]

        # 根据统计类型对直方图进行归一化处理
        if self.stat == "probability" or self.stat == "proportion":
            hist = hist.astype(float) / hist.sum()
        elif self.stat == "percent":
            hist = hist.astype(float) / hist.sum() * 100
        elif self.stat == "frequency":
            hist = hist.astype(float) / data["space"]

        # 如果设置了累积选项，根据统计类型进行累积处理
        if self.cumulative:
            if self.stat in ["density", "frequency"]:
                hist = (hist * data["space"]).cumsum()
            else:
                hist = hist.cumsum()

        # 将归一化后的结果添加回数据中并返回
        return data.assign(**{self.stat: hist})

    # 定义 `__call__` 方法，接受数据框、分组对象、方向和比例尺作为参数，返回数据框
    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        # 获取指定方向的比例尺类型的小写形式
        scale_type = scales[orient].__class__.__name__.lower()
        # 将数据列名转换为字符串形式，仅包含在分组顺序中的列
        grouping_vars = [str(v) for v in data if v in groupby.order]

        # 如果没有分组变量或者设置了使用公共直方图，调用 `_define_bin_params` 方法获取直方图参数
        if not grouping_vars or self.common_bins is True:
            bin_kws = self._define_bin_params(data, orient, scale_type)
            # 将数据应用到 `groupby` 函数，使用 `_eval` 方法评估数据，传入方向和直方图参数
            data = groupby.apply(data, self._eval, orient, bin_kws)
        else:
            # 如果不使用公共直方图，则创建新的分组对象
            if self.common_bins is False:
                bin_groupby = GroupBy(grouping_vars)
            else:
                bin_groupby = GroupBy(self.common_bins)
                # 检查分组变量是否符合预期
                self._check_grouping_vars("common_bins", grouping_vars)

            # 将数据应用到 `bin_groupby` 对象，使用 `_get_bins_and_eval` 方法获取直方图和评估结果
            data = bin_groupby.apply(
                data, self._get_bins_and_eval, orient, groupby, scale_type,
            )

        # 如果没有分组变量或者设置了使用公共归一化选项，对数据进行归一化处理
        if not grouping_vars or self.common_norm is True:
            data = self._normalize(data)
        else:
            # 如果不使用公共归一化选项，则创建新的分组对象
            if self.common_norm is False:
                norm_groupby = GroupBy(grouping_vars)
            else:
                norm_groupby = GroupBy(self.common_norm)
                # 检查分组变量是否符合预期
                self._check_grouping_vars("common_norm", grouping_vars)

            # 将数据应用到 `norm_groupby` 对象，使用 `_normalize` 方法进行归一化处理
            data = norm_groupby.apply(data, self._normalize)

        # 根据方向的不同，返回处理后的数据框，交换 `x` 和 `y` 列名对应的数据
        other = {"x": "y", "y": "x"}[orient]
        return data.assign(**{other: data[self.stat]})
```