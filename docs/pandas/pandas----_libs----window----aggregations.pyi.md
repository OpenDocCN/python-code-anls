# `D:\src\scipysrc\pandas\pandas\_libs\window\aggregations.pyi`

```
# 引入必要的类型声明和模块
from typing import (
    Any,
    Callable,
    Literal,
)

import numpy as np  # 导入 NumPy 库

from pandas._typing import (
    WindowingRankType,  # 引入 Pandas 内部的类型声明
    npt,  # 引入 Pandas 内部的类型声明
)

# 定义一个函数 roll_sum，计算滚动窗口内数组值的总和
def roll_sum(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_mean，计算滚动窗口内数组值的均值
def roll_mean(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_var，计算滚动窗口内数组值的方差
def roll_var(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
    ddof: int = ...,  # 自由度（默认值为省略值）
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_skew，计算滚动窗口内数组值的偏度
def roll_skew(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_kurt，计算滚动窗口内数组值的峰度
def roll_kurt(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_median_c，计算滚动窗口内数组值的中位数
def roll_median_c(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_max，计算滚动窗口内数组值的最大值
def roll_max(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_min，计算滚动窗口内数组值的最小值
def roll_min(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_quantile，计算滚动窗口内数组值的分位数
def roll_quantile(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
    quantile: float,  # 分位数的浮点数值
    interpolation: Literal["linear", "lower", "higher", "nearest", "midpoint"],  # 插值方法的文字字面量
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_rank，计算滚动窗口内数组值的排名
def roll_rank(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
    percentile: bool,  # 是否返回百分位数值的布尔值
    method: WindowingRankType,  # 窗口排名类型
    ascending: bool,  # 排序顺序的布尔值
) -> np.ndarray: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_apply，应用自定义函数到滚动窗口内的数组值
def roll_apply(
    obj: object,  # 输入对象参数
    start: np.ndarray,  # 滚动窗口起始索引的 NumPy 整数数组
    end: np.ndarray,  # 滚动窗口结束索引的 NumPy 整数数组
    minp: int,  # 滚动窗口最小数量的整数值
    function: Callable[..., Any],  # 应用的函数参数
    raw: bool,  # 是否原始布尔值
    args: tuple[Any, ...],  # 函数的额外参数元组
    kwargs: dict[str, Any],  # 函数的关键字参数字典
) -> npt.NDArray[np.float64]: ...  # 返回值为包含浮点数的 NumPy 数组

# 定义一个函数 roll_weighted_sum，计算滚动窗口内数组值的加权总和
def roll_weighted_sum(
    values: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    weights: np.ndarray,  # 输入参数，包含浮点数的 NumPy 数组
    minp: int,  # 滚动窗口最小
    values: np.ndarray,  # 定义一个变量 values，类型为 numpy 的数组，存储浮点数数据，相当于 const float64_t[:]
    weights: np.ndarray,  # 定义一个变量 weights，类型为 numpy 的数组，存储浮点数数据，相当于 const float64_t[:]
    minp: int,  # 定义一个变量 minp，类型为整数，表示一个最小值
# 返回类型为 np.ndarray，包含 np.float64 类型的元素
def roll_weighted_var(
    values: np.ndarray,  # 输入的值数组，类型为 const float64_t[:]
    weights: np.ndarray,  # 输入的权重数组，类型为 const float64_t[:]
    minp: int,  # 最小数据点数，类型为 int64_t
    ddof: int,  # 自由度修正值，类型为 unsigned int
) -> np.ndarray:  # 返回 np.ndarray，包含 np.float64 类型的元素

# 返回类型为 np.ndarray，包含 np.float64 类型的元素
def ewm(
    vals: np.ndarray,  # 输入的值数组，类型为 const float64_t[:]
    start: np.ndarray,  # 起始时间数组，类型为 const int64_t[:]
    end: np.ndarray,  # 结束时间数组，类型为 const int64_t[:]
    minp: int,  # 最小数据点数，类型为 int64_t
    com: float,  # 指数加权移动平均的中心参数，类型为 float64_t
    adjust: bool,  # 是否进行指数加权移动平均调整，类型为 bool
    ignore_na: bool,  # 是否忽略 NaN 值，类型为 bool
    deltas: np.ndarray | None = None,  # 可选的增量数组，类型为 const float64_t[:]，默认为 None
    normalize: bool = True,  # 是否进行归一化，类型为 bool，默认为 True
) -> np.ndarray:  # 返回 np.ndarray，包含 np.float64 类型的元素

# 返回类型为 np.ndarray，包含 np.float64 类型的元素
def ewmcov(
    input_x: np.ndarray,  # 输入的 X 数组，类型为 const float64_t[:]
    start: np.ndarray,  # 起始时间数组，类型为 const int64_t[:]
    end: np.ndarray,  # 结束时间数组，类型为 const int64_t[:]
    minp: int,  # 最小数据点数，类型为 int64_t
    input_y: np.ndarray,  # 输入的 Y 数组，类型为 const float64_t[:]
    com: float,  # 指数加权移动协方差的中心参数，类型为 float64_t
    adjust: bool,  # 是否进行指数加权移动协方差调整，类型为 bool
    ignore_na: bool,  # 是否忽略 NaN 值，类型为 bool
    bias: bool,  # 是否进行偏差调整，类型为 bool
) -> np.ndarray:  # 返回 np.ndarray，包含 np.float64 类型的元素
```