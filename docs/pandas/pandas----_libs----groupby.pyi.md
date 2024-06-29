# `D:\src\scipysrc\pandas\pandas\_libs\groupby.pyi`

```
# 从 typing 模块导入 Literal 类型提示
from typing import Literal

# 导入 NumPy 库，并使用 np 别名
import numpy as np

# 从 pandas._typing 模块导入 npt 类型提示
from pandas._typing import npt

# 定义函数 group_median_float64，计算分组中每组的中位数并存储在 out 中
def group_median_float64(
    out: np.ndarray,  # 输出数组，存储中位数结果，数据类型为 float64，二维数组
    counts: npt.NDArray[np.int64],  # 数组，存储每组元素个数，数据类型为 int64
    values: np.ndarray,  # 数组，存储待计算中位数的数据，数据类型为 float64，二维数组
    labels: npt.NDArray[np.int64],  # 数组，存储每个元素所属的分组标签，数据类型为 int64
    min_count: int = ...,  # 最小计数，用于确定有效计算中位数的最小组成员数量，数据类型为 Py_ssize_t
    mask: np.ndarray | None = ...,  # 掩码数组，指示哪些数据是有效的，数据类型为 ndarray 或 None
    result_mask: np.ndarray | None = ...,  # 结果掩码数组，存储中位数计算结果的有效性，数据类型为 ndarray 或 None
    is_datetimelike: bool = ...,  # 布尔值，指示数据是否类似日期时间格式，数据类型为 bint
) -> None: ...
    
# 定义函数 group_cumprod，计算分组中每组数据的累积乘积并存储在 out 中
def group_cumprod(
    out: np.ndarray,  # 输出数组，存储累积乘积结果，数据类型为 float64，二维数组
    values: np.ndarray,  # 数组，存储待计算累积乘积的数据，数据类型为 float64，二维数组
    labels: np.ndarray,  # 数组，存储每个元素所属的分组标签，数据类型为 int64
    ngroups: int,  # 整数，指示分组的数量，数据类型为 int
    is_datetimelike: bool,  # 布尔值，指示数据是否类似日期时间格式，数据类型为布尔值
    skipna: bool = ...,  # 布尔值，指示是否跳过 NaN 值，数据类型为布尔值
    mask: np.ndarray | None = ...,  # 掩码数组，指示哪些数据是有效的，数据类型为 ndarray 或 None
    result_mask: np.ndarray | None = ...,  # 结果掩码数组，存储累积乘积计算结果的有效性，数据类型为 ndarray 或 None
) -> None: ...

# 定义函数 group_cumsum，计算分组中每组数据的累积和并存储在 out 中
def group_cumsum(
    out: np.ndarray,  # 输出数组，存储累积和结果，数据类型为 int64float_t，二维数组
    values: np.ndarray,  # 数组，存储待计算累积和的数据，数据类型为 int64float_t，二维数组
    labels: np.ndarray,  # 数组，存储每个元素所属的分组标签，数据类型为 int64
    ngroups: int,  # 整数，指示分组的数量，数据类型为 int
    is_datetimelike: bool,  # 布尔值，指示数据是否类似日期时间格式，数据类型为布尔值
    skipna: bool = ...,  # 布尔值，指示是否跳过 NaN 值，数据类型为布尔值
    mask: np.ndarray | None = ...,  # 掩码数组，指示哪些数据是有效的，数据类型为 ndarray 或 None
    result_mask: np.ndarray | None = ...,  # 结果掩码数组，存储累积和计算结果的有效性，数据类型为 ndarray 或 None
) -> None: ...

# 定义函数 group_shift_indexer，计算分组中每组数据的位移索引并存储在 out 中
def group_shift_indexer(
    out: np.ndarray,  # 输出数组，存储位移索引结果，数据类型为 int64_t，一维数组
    labels: np.ndarray,  # 数组，存储每个元素所属的分组标签，数据类型为 int64
    ngroups: int,  # 整数，指示分组的数量，数据类型为 int
    periods: int,  # 整数，指示位移的周期数，数据类型为 int
) -> None: ...

# 定义函数 group_fillna_indexer，计算分组中每组数据的填充索引并存储在 out 中
def group_fillna_indexer(
    out: np.ndarray,  # 输出数组，存储填充索引结果，数据类型为 intp_t，一维数组
    labels: np.ndarray,  # 数组，存储每个元素所属的分组标签，数据类型为 int64
    mask: npt.NDArray[np.uint8],  # 掩码数组，指示哪些数据是有效的，数据类型为 uint8_t
    limit: int,  # 整数，指示填充的限制数量，数据类型为 int64
    compute_ffill: bool,  # 布尔值，指示是否进行前向填充，数据类型为布尔值
    ngroups: int,  # 整数，指示分组的数量，数据类型为 int
) -> None: ...

# 定义函数 group_any_all，计算分组中每组数据的是否存在满足条件的情况并存储在 out 中
def group_any_all(
    out: np.ndarray,  # 输出数组，存储计算结果，数据类型为 uint8_t，一维数组
    values: np.ndarray,  # 数组，存储待计算数据，数据类型为 uint8_t，一维数组
    labels: np.ndarray,  # 数组，存储每个元素所属的分组标签，数据类型为 int64
    mask: np.ndarray,  # 掩码数组，指示哪些数据是有效的，数据类型为 uint8_t，一维数组
    val_test: Literal["any", "all"],  # 文字类型提示，指示计算中使用的测试类型
    skipna: bool,  # 布尔值，指示是否跳过 NaN 值，数据类型为布尔值
    result_mask: np.ndarray | None,  # 结果掩码数组，存储计算结果的有效性，数据类型为 ndarray 或 None
) -> None: ...

# 定义函数 group_sum，计算分组中每组数据的和并存储在 out 中
def group_sum(
    out: np.ndarray,  # 输出数组，存储和的结果，数据类型为 complexfloatingintuint_t，二维数组
    counts: np.ndarray,  # 数组，存储每组元素个数，数据类型为 int64，一维数组
    values: np.ndarray,  # 数组，存储待计算和的数据，数据类型为 complexfloatingintuint_t，二维数组
    labels: np.ndarray,  # 数组，存储每个元素所属的分组标签，数据类型为 intp_t，一维数组
    mask: np.ndarray | None,  # 掩码数组，指示哪些数据是有效的，数据类型为 ndarray 或 None
    result_mask: np.ndarray | None = ...,  # 结果掩码数组，存储和计算结果的有效性，数据类型为 ndarray 或 None
    min_count: int = ...,  # 最小计数，用于确定有效计算和的最小组成员数量，数据类型为 Py_ssize_t
    is_datetimelike: bool = ...,  # 布尔值，指示数据是否类似日期时间格式，数据类型为布尔值
) -> None: ...

# 定义函数 group_prod，计算分组中每组数据的乘积并存储在 out 中
def group_prod(
    out: np.ndarray,  # 输出数组，存
    values: np.ndarray,  # 定义变量 values，类型为 np.ndarray，存储浮点数数据的二维数组
    labels: np.ndarray,  # 定义变量 labels，类型为 np.ndarray，存储整数数据的一维数组
    mask: np.ndarray | None = ...,  # 可选变量 mask，类型为 np.ndarray 或 None，用于标记数据的掩码，默认为 ...
    result_mask: np.ndarray | None = ...,  # 可选变量 result_mask，类型为 np.ndarray 或 None，结果的掩码，默认为 ...
    skipna: bool = ...,  # 布尔类型变量 skipna，默认为 ...
def group_mean(
    out: np.ndarray,  # 输出数组，存储组平均值
    counts: np.ndarray,  # 数组，存储每个组中元素的数量
    values: np.ndarray,  # 二维数组，存储组的值
    labels: np.ndarray,  # 数组，存储元素所属的组标签
    min_count: int = ...,  # 最小元素数量要求
    is_datetimelike: bool = ...,  # 布尔值，指示是否为日期时间类型数据
    mask: np.ndarray | None = ...,  # 可选的布尔掩码数组，用于过滤数据
    result_mask: np.ndarray | None = ...,  # 可选的结果掩码数组，标记输出哪些位置有有效数据
) -> None: ...
def group_ohlc(
    out: np.ndarray,  # 输出数组，存储组的开盘、最高、最低和收盘值
    counts: np.ndarray,  # 数组，存储每个组中元素的数量
    values: np.ndarray,  # 二维数组，存储组的开盘、最高、最低和收盘值
    labels: np.ndarray,  # 数组，存储元素所属的组标签
    min_count: int = ...,  # 最小元素数量要求
    mask: np.ndarray | None = ...,  # 可选的布尔掩码数组，用于过滤数据
    result_mask: np.ndarray | None = ...,  # 可选的结果掩码数组，标记输出哪些位置有有效数据
) -> None: ...
def group_quantile(
    out: npt.NDArray[np.float64],  # 输出数组，存储组的分位数值
    values: np.ndarray,  # 数组，存储待计算分位数的数据
    labels: npt.NDArray[np.intp],  # 数组，存储元素所属的组标签
    mask: npt.NDArray[np.uint8],  # 数组，用于指示哪些数据点是有效的
    qs: npt.NDArray[np.float64],  # 数组，存储分位数的值
    starts: npt.NDArray[np.int64],  # 数组，存储每个组的起始索引
    ends: npt.NDArray[np.int64],  # 数组，存储每个组的结束索引
    interpolation: Literal["linear", "lower", "higher", "nearest", "midpoint"],  # 字面常量，指定分位数插值方法
    result_mask: np.ndarray | None,  # 可选的结果掩码数组，标记输出哪些位置有有效数据
    is_datetimelike: bool,  # 布尔值，指示是否为日期时间类型数据
) -> None: ...
def group_last(
    out: np.ndarray,  # 输出数组，存储每个组的最后一个元素值
    counts: np.ndarray,  # 数组，存储每个组中元素的数量
    values: np.ndarray,  # 二维数组，存储组的值
    labels: np.ndarray,  # 数组，存储元素所属的组标签
    mask: npt.NDArray[np.bool_] | None,  # 可选的布尔掩码数组，用于过滤数据
    result_mask: npt.NDArray[np.bool_] | None = ...,  # 可选的结果掩码数组，标记输出哪些位置有有效数据
    min_count: int = ...,  # 最小元素数量要求
    is_datetimelike: bool = ...,  # 布尔值，指示是否为日期时间类型数据
    skipna: bool = ...,  # 布尔值，指示是否跳过缺失值
) -> None: ...
def group_nth(
    out: np.ndarray,  # 输出数组，存储每个组的第n个元素值
    counts: np.ndarray,  # 数组，存储每个组中元素的数量
    values: np.ndarray,  # 二维数组，存储组的值
    labels: np.ndarray,  # 数组，存储元素所属的组标签
    mask: npt.NDArray[np.bool_] | None,  # 可选的布尔掩码数组，用于过滤数据
    result_mask: npt.NDArray[np.bool_] | None = ...,  # 可选的结果掩码数组，标记输出哪些位置有有效数据
    min_count: int = ...,  # 最小元素数量要求
    rank: int = ...,  # 指定的元素排名
    is_datetimelike: bool = ...,  # 布尔值，指示是否为日期时间类型数据
    skipna: bool = ...,  # 布尔值，指示是否跳过缺失值
) -> None: ...
def group_rank(
    out: np.ndarray,  # 输出数组，存储每个元素在其组内的排名
    values: np.ndarray,  # 二维数组，存储组的值
    labels: np.ndarray,  # 数组，存储元素所属的组标签
    ngroups: int,  # 整数，指示组的数量
    is_datetimelike: bool,  # 布尔值，指示是否为日期时间类型数据
    ties_method: Literal["average", "min", "max", "first", "dense"] = ...,  # 字面常量，指定处理并列值的方法
    ascending: bool = ...,  # 布尔值，指示是否升序排列
    pct: bool = ...,  # 布尔值，指示是否输出排名百分比
    na_option: Literal["keep", "top", "bottom"] = ...,  # 字面常量，指定处理缺失值的方法
    mask: npt.NDArray[np.bool_] | None = ...,  # 可选的布尔掩码数组，用于过滤数据
) -> None: ...
def group_max(
    out: np.ndarray,  # 输出数组，存储每个组的最大值
    counts: np.ndarray,  # 数组，存储每个组中元素的数量
    values: np.ndarray,  # 二维数组，存储组的值
    labels: np.ndarray,  # 数组，存储元素所属的组标签
    min_count: int = ...,  # 最小元素数量要求
    is_datetimelike: bool = ...,  # 布尔值，指示是否为日期时间类型数据
    mask: np.ndarray | None = ...,  # 可选的布尔掩码数组，用于过滤数据
    result_mask: np.ndarray | None = ...,  # 可选的结果掩码数组，标记输出哪些位置有有效数据
) -> None: ...
def group_min(
    out: np.ndarray,  # 输出数组，存储每个组的最小值
    counts: np.ndarray,  # 数组，存储每个组中元素的数量
    values: np.ndarray,  # 二维数组，存储组的值
    labels: np.ndarray,  # 数组，存储元素所属的组标签
    min_count: int = ...,  # 最小元素数量要求
    is_datetimelike: bool = ...,  # 定义一个布尔型变量 is_datetimelike，初始值为未指定
    mask: np.ndarray | None = ...,  # 定义一个 NumPy 数组或空值变量 mask，初始值为未指定
    result_mask: np.ndarray | None = ...,  # 定义一个 NumPy 数组或空值变量 result_mask，初始值为未指定
# 定义函数 group_idxmin_idxmax，计算每个分组内的最小和最大索引
def group_idxmin_idxmax(
    out: npt.NDArray[np.intp],  # 输出数组，用于存储结果索引
    counts: npt.NDArray[np.int64],  # 每个分组的元素计数
    values: np.ndarray,  # 值数组，形状为 [groupby_t, ndim=2]
    labels: npt.NDArray[np.intp],  # 标签数组，用于分组的索引
    min_count: int = ...,  # 最小计数阈值
    is_datetimelike: bool = ...,  # 是否是日期时间类型
    mask: np.ndarray | None = ...,  # 掩码数组，用于指示无效值
    name: str = ...,  # 名称字符串
    skipna: bool = ...,  # 是否跳过 NaN 值
    result_mask: np.ndarray | None = ...,  # 结果的掩码数组
) -> None: ...
# 定义函数 group_cummin，计算每个分组内的累积最小值
def group_cummin(
    out: np.ndarray,  # 输出数组，用于存储累积最小值
    values: np.ndarray,  # 值数组，形状为 [groupby_t, ndim=2]
    labels: np.ndarray,  # 标签数组，用于分组的索引
    ngroups: int,  # 分组的数量
    is_datetimelike: bool,  # 是否是日期时间类型
    mask: np.ndarray | None = ...,  # 掩码数组，用于指示无效值
    result_mask: np.ndarray | None = ...,  # 结果的掩码数组
    skipna: bool = ...,  # 是否跳过 NaN 值
) -> None: ...
# 定义函数 group_cummax，计算每个分组内的累积最大值
def group_cummax(
    out: np.ndarray,  # 输出数组，用于存储累积最大值
    values: np.ndarray,  # 值数组，形状为 [groupby_t, ndim=2]
    labels: np.ndarray,  # 标签数组，用于分组的索引
    ngroups: int,  # 分组的数量
    is_datetimelike: bool,  # 是否是日期时间类型
    mask: np.ndarray | None = ...,  # 掩码数组，用于指示无效值
    result_mask: np.ndarray | None = ...,  # 结果的掩码数组
    skipna: bool = ...,  # 是否跳过 NaN 值
) -> None: ...
```