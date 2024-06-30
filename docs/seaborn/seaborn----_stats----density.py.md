# `D:\src\scipysrc\seaborn\seaborn\_stats\density.py`

```
from __future__ import annotations
# 允许在类定义中使用类型注解（Python 3.7+），这样可以引用尚未定义的类

from dataclasses import dataclass
# 导入 dataclass 装饰器，用于轻松创建不可变的数据类

from typing import Any, Callable
# 导入类型相关的模块，用于类型提示

import numpy as np
# 导入 NumPy 库并重命名为 np

from numpy import ndarray
# 从 NumPy 中导入 ndarray 类型

import pandas as pd
# 导入 Pandas 库并重命名为 pd

from pandas import DataFrame
# 从 Pandas 中导入 DataFrame 类型

try:
    from scipy.stats import gaussian_kde
    _no_scipy = False
except ImportError:
    from seaborn.external.kde import gaussian_kde
    _no_scipy = True
# 尝试从 SciPy 中导入 gaussian_kde 函数；如果导入失败，从 seaborn 的外部依赖中导入

from seaborn._core.groupby import GroupBy
# 从 seaborn 内部模块导入 GroupBy 类

from seaborn._core.scales import Scale
# 从 seaborn 内部模块导入 Scale 类

from seaborn._stats.base import Stat
# 从 seaborn 内部模块导入 Stat 类

@dataclass
# 使用 dataclass 装饰器，用于定义数据类
class KDE(Stat):
    """
    Compute a univariate kernel density estimate.

    Parameters
    ----------
    bw_adjust : float
        Factor that multiplicatively scales the value chosen using
        `bw_method`. Increasing will make the curve smoother. See Notes.
    bw_method : string, scalar, or callable
        Method for determining the smoothing bandwidth to use. Passed directly
        to :class:`scipy.stats.gaussian_kde`; see there for options.
    common_norm : bool or list of variables
        If `True`, normalize so that the areas of all curves sums to 1.
        If `False`, normalize each curve independently. If a list, defines
        variable(s) to group by and normalize within.
    common_grid : bool or list of variables
        If `True`, all curves will share the same evaluation grid.
        If `False`, each evaluation grid is independent. If a list, defines
        variable(s) to group by and share a grid within.
    gridsize : int or None
        Number of points in the evaluation grid. If None, the density is
        evaluated at the original datapoints.
    cut : float
        Factor, multiplied by the kernel bandwidth, that determines how far
        the evaluation grid extends past the extreme datapoints. When set to 0,
        the curve is truncated at the data limits.
    cumulative : bool
        If True, estimate a cumulative distribution function. Requires scipy.

    Notes
    -----
    The *bandwidth*, or standard deviation of the smoothing kernel, is an
    important parameter. Much like histogram bin width, using the wrong
    bandwidth can produce a distorted representation. Over-smoothing can erase
    true features, while under-smoothing can create false ones. The default
    uses a rule-of-thumb that works best for distributions that are roughly
    bell-shaped. It is a good idea to check the default by varying `bw_adjust`.

    Because the smoothing is performed with a Gaussian kernel, the estimated
    density curve can extend to values that may not make sense. For example, the
    curve may be drawn over negative values when data that are naturally
    positive. The `cut` parameter can be used to control the evaluation range,
    but datasets that have many observations close to a natural boundary may be
    better served by a different method.

    Similar distortions may arise when a dataset is naturally discrete or "spiky"
    (containing many repeated observations of the same value). KDEs will always
    """
    # 定义 KDE 类，继承自 Stat 类，用于计算一元核密度估计
    pass
    # 占位符，因为示例代码中没有具体的类实现内容，只有类的文档字符串
    bw_adjust: float = 1
    bw_method: str | float | Callable[[gaussian_kde], float] = "scott"
    common_norm: bool | list[str] = True
    common_grid: bool | list[str] = True
    gridsize: int | None = 200
    cut: float = 3
    cumulative: bool = False

    def __post_init__(self):
        # 如果设置了 cumulative 并且没有安装 scipy，抛出运行时错误
        if self.cumulative and _no_scipy:
            raise RuntimeError("Cumulative KDE evaluation requires scipy")

    def _check_var_list_or_boolean(self, param: str, grouping_vars: Any) -> None:
        """Do input checks on grouping parameters."""
        # 获取参数值
        value = getattr(self, param)
        # 检查参数值是否是布尔类型或者是字符串列表且所有元素都是字符串
        if not (
            isinstance(value, bool)
            or (isinstance(value, list) and all(isinstance(v, str) for v in value))
        ):
            param_name = f"{self.__class__.__name__}.{param}"
            raise TypeError(f"{param_name} must be a boolean or list of strings.")
        # 检查参数与分组变量的对应关系
        self._check_grouping_vars(param, grouping_vars, stacklevel=3)

    def _fit(self, data: DataFrame, orient: str) -> gaussian_kde:
        """Fit and return a KDE object."""
        # TODO 需要处理单一数据的情况

        # 设置拟合参数
        fit_kws: dict[str, Any] = {"bw_method": self.bw_method}
        # 如果数据中有权重列，则使用权重
        if "weight" in data:
            fit_kws["weights"] = data["weight"]
        # 使用 gaussian_kde 对象拟合数据
        kde = gaussian_kde(data[orient], **fit_kws)
        # 根据 bw_adjust 调整带宽
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        return kde

    def _get_support(self, data: DataFrame, orient: str) -> ndarray:
        """Define the grid that the KDE will be evaluated on."""
        # 如果 gridsize 为 None，则直接使用数据的值作为支持数组
        if self.gridsize is None:
            return data[orient].to_numpy()

        # 使用 _fit 方法拟合数据并得到 KDE 对象
        kde = self._fit(data, orient)
        # 计算带宽
        bw = np.sqrt(kde.covariance.squeeze())
        # 计算网格的最小和最大值
        gridmin = data[orient].min() - bw * self.cut
        gridmax = data[orient].max() + bw * self.cut
        # 生成等间隔的支持数组
        return np.linspace(gridmin, gridmax, self.gridsize)

    def _fit_and_evaluate(
        self, data: DataFrame, orient: str, support: ndarray
    ) -> DataFrame:
        """Transform single group by fitting a KDE and evaluating on a support grid."""
        # 创建一个空的 DataFrame，用于存储 KDE 和评估结果
        empty = pd.DataFrame(columns=[orient, "weight", "density"], dtype=float)
        # 如果数据长度小于 2，则返回空的 DataFrame
        if len(data) < 2:
            return empty
        try:
            # 尝试拟合数据并生成 KDE
            kde = self._fit(data, orient)
        except np.linalg.LinAlgError:
            # 捕获线性代数错误，返回空的 DataFrame
            return empty

        if self.cumulative:
            # 如果是累积模式，计算累积密度估计值
            s_0 = support[0]
            density = np.array([kde.integrate_box_1d(s_0, s_i) for s_i in support])
        else:
            # 否则，计算支持网格上的密度估计
            density = kde(support)

        # 计算数据中权重列的总和
        weight = data["weight"].sum()
        # 返回结果作为 DataFrame，包括方向、权重和密度
        return pd.DataFrame({orient: support, "weight": weight, "density": density})

    def _transform(
        self, data: DataFrame, orient: str, grouping_vars: list[str]
    ) -> DataFrame:
        """Transform multiple groups by fitting KDEs and evaluating."""
        # 创建一个空的 DataFrame，用于存储 KDE 和评估结果，列名包括数据列和密度
        empty = pd.DataFrame(columns=[*data.columns, "density"], dtype=float)
        # 如果数据长度小于 2，则返回空的 DataFrame
        if len(data) < 2:
            return empty
        try:
            # 获取支持值
            support = self._get_support(data, orient)
        except np.linalg.LinAlgError:
            # 捕获线性代数错误，返回空的 DataFrame
            return empty

        # 过滤出具有多个唯一值的分组变量
        grouping_vars = [x for x in grouping_vars if data[x].nunique() > 1]
        if not grouping_vars:
            # 如果没有分组变量，直接调用 _fit_and_evaluate 处理数据并返回结果
            return self._fit_and_evaluate(data, orient, support)
        # 使用 GroupBy 对象分组并应用 _fit_and_evaluate 函数处理数据
        groupby = GroupBy(grouping_vars)
        return groupby.apply(data, self._fit_and_evaluate, orient, support)

    def __call__(
        self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale],


注释：这段代码定义了一个类的方法和函数，用于对数据进行 KDE 拟合和评估，支持单一组和多组数据的处理。
        ) -> DataFrame:
        # 定义函数签名，指定返回类型为 DataFrame

        if "weight" not in data:
            # 如果数据中不包含列名为 "weight" 的列，则添加该列并赋值为 1
            data = data.assign(weight=1)
        
        # 删除数据中指定列（orient 和 "weight"）中含有缺失值的行
        data = data.dropna(subset=[orient, "weight"])

        # 根据 groupby 对象的顺序，将数据中的变量转换为字符串
        grouping_vars = [str(v) for v in data if v in groupby.order]

        if not grouping_vars or self.common_grid is True:
            # 如果没有指定分组变量或者 self.common_grid 为 True，则调用 _transform 方法进行转换
            res = self._transform(data, orient, grouping_vars)
        else:
            if self.common_grid is False:
                # 如果 self.common_grid 为 False，则将分组变量设为 grouping_vars
                grid_vars = grouping_vars
            else:
                # 否则，检查 self.common_grid 是否为有效变量列表或布尔值，然后将其设为与 grouping_vars 相交的部分
                self._check_var_list_or_boolean("common_grid", grouping_vars)
                grid_vars = [v for v in self.common_grid if v in grouping_vars]

            # 使用 GroupBy 对象按 grid_vars 分组，并应用 _transform 方法进行转换
            res = (
                GroupBy(grid_vars)
                .apply(data, self._transform, orient, grouping_vars)
            )

        # 根据情况对结果进行归一化处理，可能是在组内进行
        if not grouping_vars or self.common_norm is True:
            # 如果没有指定分组变量或者 self.common_norm 为 True，则对 res 中的 "density" 列进行归一化处理
            res = res.assign(group_weight=data["weight"].sum())
        else:
            if self.common_norm is False:
                # 如果 self.common_norm 为 False，则将归一化变量设为 grouping_vars
                norm_vars = grouping_vars
            else:
                # 否则，检查 self.common_norm 是否为有效变量列表或布尔值，然后将其设为与 grouping_vars 相交的部分
                self._check_var_list_or_boolean("common_norm", grouping_vars)
                norm_vars = [v for v in self.common_norm if v in grouping_vars]

            # 使用 data 根据 norm_vars 进行分组，并计算每组中 "weight" 列的总和，并重命名为 "group_weight"，然后将其与 res 合并
            res = res.join(
                data.groupby(norm_vars)["weight"].sum().rename("group_weight"),
                on=norm_vars,
            )

        # 计算 res 中的 "density" 列乘以 "weight" 列除以 "group_weight" 列的结果，并存储在 "density" 列中
        res["density"] *= res.eval("weight / group_weight")

        # 根据 orient 变量的值，将 "x" 和 "y" 列进行交换
        value = {"x": "y", "y": "x"}[orient]
        res[value] = res["density"]

        # 返回 res 数据框，并丢弃 "weight" 和 "group_weight" 列
        return res.drop(["weight", "group_weight"], axis=1)
```