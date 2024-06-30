# `D:\src\scipysrc\seaborn\seaborn\_stats\aggregation.py`

```
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable

import pandas as pd
from pandas import DataFrame

from seaborn._core.scales import Scale
from seaborn._core.groupby import GroupBy
from seaborn._stats.base import Stat
from seaborn._statistics import (
    EstimateAggregator,
    WeightedAggregator,
)
from seaborn._core.typing import Vector


@dataclass
class Agg(Stat):
    """
    Aggregate data along the value axis using given method.

    Parameters
    ----------
    func : str or callable
        Name of a :class:`pandas.Series` method or a vector -> scalar function.

    See Also
    --------
    objects.Est : Aggregation with error bars.

    Examples
    --------
    .. include:: ../docstrings/objects.Agg.rst

    """
    func: str | Callable[[Vector], float] = "mean"  # 默认聚合方法为平均值

    group_by_orient: ClassVar[bool] = True  # 类变量，指示是否支持按方向分组

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        """
        Perform aggregation along the specified axis.

        Parameters
        ----------
        data : DataFrame
            The input data to be aggregated.
        groupby : GroupBy
            Object representing the groups to apply aggregation within.
        orient : str
            Orientation of the aggregation ('x' or 'y').
        scales : dict[str, Scale]
            Mapping of scale names to Scale objects.

        Returns
        -------
        DataFrame
            Aggregated data along the specified axis.
        """
        var = {"x": "y", "y": "x"}.get(orient)  # 根据 orient 确定变量名
        res = (
            groupby
            .agg(data, {var: self.func})  # 聚合操作，使用指定的 func 函数
            .dropna(subset=[var])  # 删除缺失值
            .reset_index(drop=True)  # 重设索引
        )
        return res


@dataclass
class Est(Stat):
    """
    Calculate a point estimate and error bar interval.

    For more information about the various `errorbar` choices, see the
    :doc:`errorbar tutorial </tutorial/error_bars>`.

    Additional variables:

    - **weight**: When passed to a layer that uses this stat, a weighted estimate
      will be computed. Note that use of weights currently limits the choice of
      function and error bar method  to `"mean"` and `"ci"`, respectively.

    Parameters
    ----------
    func : str or callable
        Name of a :class:`numpy.ndarray` method or a vector -> scalar function.
    errorbar : str, (str, float) tuple, or callable
        Name of errorbar method (one of "ci", "pi", "se" or "sd"), or a tuple
        with a method name and a level parameter, or a function that maps from a
        vector to a (min, max) interval.
    n_boot : int
       Number of bootstrap samples to draw for "ci" errorbars.
    seed : int
        Seed for the PRNG used to draw bootstrap samples.

    Examples
    --------
    .. include:: ../docstrings/objects.Est.rst

    """
    func: str | Callable[[Vector], float] = "mean"  # 默认估算方法为平均值
    errorbar: str | tuple[str, float] = ("ci", 95)  # 默认误差条方法为 "ci"，置信水平为 95%
    n_boot: int = 1000  # Bootstrap 抽样次数
    seed: int | None = None  # Bootstrap 抽样的随机数种子，如果为 None 则不设定

    group_by_orient: ClassVar[bool] = True  # 类变量，指示是否支持按方向分组

    def _process(
        self, data: DataFrame, var: str, estimator: EstimateAggregator
    ) -> DataFrame:
        """
        Process the estimation for the given variable.

        Parameters
        ----------
        data : DataFrame
            The input data for estimation.
        var : str
            The variable for which estimation is performed.
        estimator : EstimateAggregator
            The aggregator object to perform the estimation.

        Returns
        -------
        DataFrame
            Resulting DataFrame containing the estimated values.
        """
        # Needed because GroupBy.apply assumes func is DataFrame -> DataFrame
        # which we could probably make more general to allow Series return
        res = estimator(data, var)  # 使用估算器进行估算
        return pd.DataFrame([res])

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],
    ) -> DataFrame:
        """
        Perform estimation and error bar calculation.

        Parameters
        ----------
        data : DataFrame
            The input data to estimate and calculate error bars for.
        groupby : GroupBy
            Object representing the groups to apply estimation within.
        orient : str
            Orientation of the estimation ('x' or 'y').
        scales : dict[str, Scale]
            Mapping of scale names to Scale objects.

        Returns
        -------
        DataFrame
            Estimated values and error bars.
        """
    ) -> DataFrame:
        # 定义函数签名，指定返回类型为 DataFrame

        # 根据 self.n_boot 和 self.seed 创建启动关键字参数字典
        boot_kws = {"n_boot": self.n_boot, "seed": self.seed}

        # 如果数据中包含 "weight" 列，则使用 WeightedAggregator 引擎
        if "weight" in data:
            engine = WeightedAggregator(self.func, self.errorbar, **boot_kws)
        else:
            # 否则使用 EstimateAggregator 引擎
            engine = EstimateAggregator(self.func, self.errorbar, **boot_kws)

        # 根据 orient 参数确定 var 变量
        var = {"x": "y", "y": "x"}[orient]

        # 对 groupby 对象应用 self._process 方法处理数据，使用 engine 引擎
        # 然后去除缺失值，并重置索引为连续整数
        res = (
            groupby
            .apply(data, self._process, var, engine)
            .dropna(subset=[var])
            .reset_index(drop=True)
        )

        # 使用 res 中 var 列的均值来填充缺失值，填充后列名为 f"{var}min" 和 f"{var}max"
        res = res.fillna({f"{var}min": res[var], f"{var}max": res[var]})

        # 返回处理后的结果 DataFrame
        return res
# 定义一个数据类 Rolling，继承自 Stat 类
@dataclass
class Rolling(Stat):
    # 这里省略了类的具体实现部分，用省略号表示

    # 定义类的调用方法，使其可以像函数一样被调用
    def __call__(self, data, groupby, orient, scales):
        # 这里省略了方法的具体实现部分，用省略号表示
```