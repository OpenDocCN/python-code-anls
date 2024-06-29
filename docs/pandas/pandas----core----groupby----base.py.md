# `D:\src\scipysrc\pandas\pandas\core\groupby\base.py`

```
"""
Provide basic components for groupby.
"""

# 从未来导入类型注解允许，在Python 3.7之前的版本中使用
from __future__ import annotations

# 导入 dataclasses 模块，用于创建不可变数据类
import dataclasses
# 导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 如果在类型检查环境下
if TYPE_CHECKING:
    # 导入 Hashable 类型
    from collections.abc import Hashable


# 创建一个带有顺序和冻结属性的数据类 OutputKey
@dataclasses.dataclass(order=True, frozen=True)
class OutputKey:
    label: Hashable  # 标签，必须是可哈希的
    position: int    # 位置，整数类型


# 用于防止在从 NDFrames 转发方法时出现重复绘图的特殊情况
plotting_methods = frozenset(["plot", "hist"])

# 已经使用 Cython 进行优化的转换或预定义的 "agg+broadcast" 方法
cythonized_kernels = frozenset(["cumprod", "cumsum", "shift", "cummin", "cummax"])

# 聚合/减少函数的列表
reduction_kernels = frozenset(
    [
        "all",
        "any",
        "corrwith",
        "count",
        "first",
        "idxmax",
        "idxmin",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "nunique",
        "prod",
        # 只要 `quantile` 的签名接受单个分位数值，它就是一个减少函数
        # GH#27526 可能会改变这一点。
        "quantile",
        "sem",
        "size",
        "skew",
        "std",
        "sum",
        "var",
    ]
)

# 转换函数的列表
transformation_kernels = frozenset(
    [
        "bfill",
        "cumcount",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "diff",
        "ffill",
        "ngroup",
        "pct_change",
        "rank",
        "shift",
    ]
)

# Grouper 上的公共方法列表，不属于上述两个列表的方法
groupby_other_methods = frozenset(
    [
        "agg",
        "aggregate",
        "apply",
        "boxplot",
        # corr 和 cov 返回 ngroups*ncolumns 行，因此它们既不是转换函数也不是减少函数
        "corr",
        "cov",
        "describe",
        "expanding",
        "ewm",
        "filter",
        "get_group",
        "groups",
        "head",
        "hist",
        "indices",
        "ndim",
        "ngroups",
        "nth",
        "ohlc",
        "pipe",
        "plot",
        "resample",
        "rolling",
        "tail",
        "take",
        "transform",
        "sample",
        "value_counts",
    ]
)

# groupby.transform(name) 方法中允许的有效值列表
# 注意：不要直接编辑此列表。新添加的内容应插入到上面适当的列表中。
transform_kernel_allowlist = reduction_kernels | transformation_kernels
```