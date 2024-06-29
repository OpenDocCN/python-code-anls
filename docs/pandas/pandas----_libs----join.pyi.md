# `D:\src\scipysrc\pandas\pandas\_libs\join.pyi`

```
# 导入 numpy 库，用于处理数组和数值计算
import numpy as np

# 导入 pandas 库中定义的类型别名 npt，用于类型注解
from pandas._typing import npt

# 定义内连接函数 inner_join，接受两个 np.ndarray 类型的数组参数，一个整数 max_groups 和一个布尔类型的 sort 参数
def inner_join(
    left: np.ndarray,  # 左侧数组，类型为 np.ndarray
    right: np.ndarray,  # 右侧数组，类型为 np.ndarray
    max_groups: int,    # 最大分组数，整数类型
    sort: bool = ...,   # 是否排序，默认值省略
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    # 函数未实现，直接使用省略号占位

# 定义左外连接函数 left_outer_join，参数和返回类型与 inner_join 类似
def left_outer_join(
    left: np.ndarray,  # 左侧数组，类型为 np.ndarray
    right: np.ndarray,  # 右侧数组，类型为 np.ndarray
    max_groups: int,    # 最大分组数，整数类型
    sort: bool = ...,   # 是否排序，默认值省略
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    # 函数未实现，直接使用省略号占位

# 定义全外连接函数 full_outer_join，接受两个 np.ndarray 类型的数组参数和一个整数 max_groups
def full_outer_join(
    left: np.ndarray,  # 左侧数组，类型为 np.ndarray
    right: np.ndarray,  # 右侧数组，类型为 np.ndarray
    max_groups: int,    # 最大分组数，整数类型
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    # 函数未实现，直接使用省略号占位

# 定义向前填充索引器函数 ffill_indexer，接受一个 np.ndarray 类型的数组参数
def ffill_indexer(
    indexer: np.ndarray,  # 索引器数组，类型为 np.ndarray
) -> npt.NDArray[np.intp]: ...
    # 函数未实现，直接使用省略号占位

# 定义左连接索引器函数 left_join_indexer_unique，接受两个 np.ndarray 类型的数组参数
def left_join_indexer_unique(
    left: np.ndarray,  # 左侧数组，类型为 np.ndarray
    right: np.ndarray,  # 右侧数组，类型为 np.ndarray
) -> npt.NDArray[np.intp]: ...
    # 函数未实现，直接使用省略号占位

# 定义左连接索引器函数 left_join_indexer，接受两个 np.ndarray 类型的数组参数，返回一个元组
def left_join_indexer(
    left: np.ndarray,  # 左侧数组，类型为 np.ndarray
    right: np.ndarray,  # 右侧数组，类型为 np.ndarray
) -> tuple[
    np.ndarray,  # 左连接结果数组，类型为 np.ndarray
    npt.NDArray[np.intp],  # 左连接的索引器数组，类型为 npt.NDArray[np.intp]
    npt.NDArray[np.intp],  # 未匹配的索引器数组，类型为 npt.NDArray[np.intp]
]: ...
    # 函数未实现，直接使用省略号占位

# 定义内连接索引器函数 inner_join_indexer，接受两个 np.ndarray 类型的数组参数，返回一个元组
def inner_join_indexer(
    left: np.ndarray,  # 左侧数组，类型为 np.ndarray
    right: np.ndarray,  # 右侧数组，类型为 np.ndarray
) -> tuple[
    np.ndarray,  # 内连接结果数组，类型为 np.ndarray
    npt.NDArray[np.intp],  # 内连接的索引器数组，类型为 npt.NDArray[np.intp]
    npt.NDArray[np.intp],  # 未匹配的索引器数组，类型为 npt.NDArray[np.intp]
]: ...
    # 函数未实现，直接使用省略号占位

# 定义外连接索引器函数 outer_join_indexer，接受两个 np.ndarray 类型的数组参数，返回一个元组
def outer_join_indexer(
    left: np.ndarray,  # 左侧数组，类型为 np.ndarray
    right: np.ndarray,  # 右侧数组，类型为 np.ndarray
) -> tuple[
    np.ndarray,  # 外连接结果数组，类型为 np.ndarray
    npt.NDArray[np.intp],  # 外连接的左侧索引器数组，类型为 npt.NDArray[np.intp]
    npt.NDArray[np.intp],  # 外连接的右侧索引器数组，类型为 npt.NDArray[np.intp]
]: ...
    # 函数未实现，直接使用省略号占位

# 定义基于 X 通过 Y 向后的 asof 连接函数 asof_join_backward_on_X_by_Y，
# 接受两个 ndarray[numeric_t] 类型的数值数组和两个 const int64_t[:] 类型的索引数组
def asof_join_backward_on_X_by_Y(
    left_values: np.ndarray,  # 左侧数值数组，类型为 np.ndarray[numeric_t]
    right_values: np.ndarray,  # 右侧数值数组，类型为 np.ndarray[numeric_t]
    left_by_values: np.ndarray,  # 左侧索引数组，类型为 const int64_t[:]
    right_by_values: np.ndarray,  # 右侧索引数组，类型为 const int64_t[:]
    allow_exact_matches: bool = ...,  # 允许精确匹配，默认值省略
    tolerance: np.number | float | None = ...,  # 容差值，类型为 np.number | float | None，默认值省略
    use_hashtable: bool = ...,  # 是否使用哈希表，默认值省略
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    # 函数未实现，直接使用省略号占位

# 定义基于 X 通过 Y 向前的 asof 连接函数 asof_join_forward_on_X_by_Y，
# 接受两个 ndarray[numeric_t] 类型的数值数组和两个 const int64_t[:] 类型的索引数组
def asof_join_forward_on_X_by_Y(
    left_values: np.ndarray,  # 左侧数值数组，类型为 np.ndarray[numeric_t]
    right_values: np.ndarray,  # 右侧数值数组，类型为 np.ndarray[numeric_t]
    left_by_values: np.ndarray,  # 左侧索引数组，类型为 const int64_t[:]
    right_by_values: np.ndarray,  # 右侧索引数组，类型为 const int64_t[:]
    allow_exact_matches: bool = ...,  # 允许精确匹配，默认值省略
    tolerance: np.number | float | None = ...,  # 容差值，类型为 np.number | float | None，默认值省略
    use_hashtable: bool = ...,  # 是否使用哈希表，默认值省略
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    # 函数未实现，直接使用省略号占位

# 定义基于 X 通过 Y 最近的 asof 连接函数 asof_join_nearest_on_X_by_Y，
# 接受两个 ndarray[numeric_t] 类型的数值数组和两个 const int64_t[:] 类型的索引数组
def asof_join_nearest_on_X_by_Y(
    left_values: np.ndarray,  # 左侧数值数组，类型为 np.ndarray[numeric_t]
    right_values: np.ndarray,  # 右侧数值数组，类型为 np.ndarray[numeric_t]
    left_by_values: np.ndarray,  # 左侧索引数组，类型为 const int64_t[:]
    right_by_values: np.ndarray,  # 右侧索引数组，类型为 const int64_t[:]
    allow_exact_matches: bool = ...,  # 允许精确匹配，默认值省略
    tolerance: np.number | float | None = ...,  # 容差值，类型
```