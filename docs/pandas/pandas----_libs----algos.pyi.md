# `D:\src\scipysrc\pandas\pandas\_libs\algos.pyi`

```
from typing import Any

import numpy as np

from pandas._typing import npt

# 定义 Infinity 类，实现比较运算符来处理无穷大值
class Infinity:
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...

# 定义 NegInfinity 类，实现比较运算符来处理负无穷大值
class NegInfinity:
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...

# 函数 unique_deltas：计算给定数组的唯一差分值
def unique_deltas(
    arr: np.ndarray,  # const int64_t[:]
) -> np.ndarray: ...  # np.ndarray[np.int64, ndim=1]

# 函数 is_lexsorted：检查数组列表是否按字典序排序
def is_lexsorted(list_of_arrays: list[npt.NDArray[np.int64]]) -> bool: ...

# 函数 groupsort_indexer：对索引进行分组排序
def groupsort_indexer(
    index: np.ndarray,  # const int64_t[:]
    ngroups: int,
) -> tuple[
    np.ndarray,  # ndarray[int64_t, ndim=1]
    np.ndarray,  # ndarray[int64_t, ndim=1]
]: ...

# 函数 kth_smallest：找到数组中第 k 小的元素
def kth_smallest(
    arr: np.ndarray,  # numeric[:]
    k: int,
) -> Any: ...  # numeric

# ----------------------------------------------------------------------
# 函数 nancorr：计算具有 NaN 值的数据集的相关系数或协方差
def nancorr(
    mat: npt.NDArray[np.float64],  # const float64_t[:, :]
    cov: bool = ...,
    minp: int | None = ...,
) -> npt.NDArray[np.float64]: ...  # ndarray[float64_t, ndim=2]

# 函数 nancorr_spearman：计算具有 NaN 值的数据集的 Spearman 相关系数
def nancorr_spearman(
    mat: npt.NDArray[np.float64],  # ndarray[float64_t, ndim=2]
    minp: int = ...,
) -> npt.NDArray[np.float64]: ...  # ndarray[float64_t, ndim=2]

# ----------------------------------------------------------------------
# 函数 validate_limit：验证观测数量的限制，如果为 None 则返回默认值
def validate_limit(nobs: int | None, limit=...) -> int: ...

# 函数 get_fill_indexer：获取填充索引，用于填充缺失值
def get_fill_indexer(
    mask: npt.NDArray[np.bool_],
    limit: int | None = None,
) -> npt.NDArray[np.intp]: ...

# 函数 pad：填充旧数组以匹配新数组的形状
def pad(
    old: np.ndarray,  # ndarray[numeric_object_t]
    new: np.ndarray,  # ndarray[numeric_object_t]
    limit=...,
) -> npt.NDArray[np.intp]: ...  # np.ndarray[np.intp, ndim=1]

# 函数 pad_inplace：就地填充数组以匹配掩码的形状
def pad_inplace(
    values: np.ndarray,  # numeric_object_t[:]
    mask: np.ndarray,  # uint8_t[:]
    limit=...,
) -> None: ...

# 函数 pad_2d_inplace：就地填充二维数组以匹配掩码的形状
def pad_2d_inplace(
    values: np.ndarray,  # numeric_object_t[:, :]
    mask: np.ndarray,  # const uint8_t[:, :]
    limit=...,
) -> None: ...

# 函数 backfill：回填旧数组以匹配新数组的形状
def backfill(
    old: np.ndarray,  # ndarray[numeric_object_t]
    new: np.ndarray,  # ndarray[numeric_object_t]
    limit=...,
) -> npt.NDArray[np.intp]: ...  # np.ndarray[np.intp, ndim=1]

# 函数 backfill_inplace：就地回填数组以匹配掩码的形状
def backfill_inplace(
    values: np.ndarray,  # numeric_object_t[:]
    mask: np.ndarray,  # uint8_t[:]
    limit=...,
) -> None: ...

# 函数 backfill_2d_inplace：就地回填二维数组以匹配掩码的形状
def backfill_2d_inplace(
    values: np.ndarray,  # numeric_object_t[:, :]
    mask: np.ndarray,  # const uint8_t[:, :]
    limit=...,
) -> None: ...

# 函数 is_monotonic：检查数组是否单调
def is_monotonic(
    arr: np.ndarray,  # ndarray[numeric_object_t, ndim=1]
    timelike: bool,
) -> tuple[bool, bool, bool]: ...
# ----------------------------------------------------------------------

# 排序一维数组，并返回排名结果
def rank_1d(
    values: np.ndarray,  # 一维数组，包含数值对象
    labels: np.ndarray | None = ...,  # 可选参数，整型数组，默认为None
    is_datetimelike: bool = ...,  # 布尔值，指示数组是否类似日期时间
    ties_method=...,  # 排名时遇到并列情况的处理方法
    ascending: bool = ...,  # 布尔值，指示排名顺序（升序或降序）
    pct: bool = ...,  # 布尔值，指示是否返回百分位排名
    na_option=...,  # 处理缺失值的选项
    mask: npt.NDArray[np.bool_] | None = ...,  # 可选参数，布尔掩码数组，用于屏蔽特定值
) -> np.ndarray:  # 返回一维数组，浮点数类型，包含排名结果
    ...

# 排序二维数组，并返回排名结果
def rank_2d(
    in_arr: np.ndarray,  # 二维数组，包含数值对象
    axis: int = ...,  # 整数，指示沿哪个轴排序
    is_datetimelike: bool = ...,  # 布尔值，指示数组是否类似日期时间
    ties_method=...,  # 排名时遇到并列情况的处理方法
    ascending: bool = ...,  # 布尔值，指示排名顺序（升序或降序）
    na_option=...,  # 处理缺失值的选项
    pct: bool = ...,  # 布尔值，指示是否返回百分位排名
) -> np.ndarray:  # 返回一维数组，浮点数类型，包含排名结果
    ...

# 计算二维数组的差分，并将结果存储到指定的输出数组
def diff_2d(
    arr: np.ndarray,  # 输入的二维数组，包含差分的源数据
    out: np.ndarray,  # 输出的二维数组，用于存储差分结果
    periods: int,  # 整数，指示进行差分的周期数
    axis: int,  # 整数，指示沿哪个轴进行差分计算
    datetimelike: bool = ...,  # 布尔值，指示数组是否类似日期时间
) -> None:  # 返回空值，直接修改输出数组out

# 确保输入数组转换为平台整数类型的 NumPy 数组
def ensure_platform_int(arr: object) -> npt.NDArray[np.intp]:  # 返回平台整数类型的 NumPy 数组
    ...

# 确保输入数组转换为对象类型的 NumPy 数组
def ensure_object(arr: object) -> npt.NDArray[np.object_]:  # 返回对象类型的 NumPy 数组
    ...

# 确保输入数组转换为双精度浮点数类型的 NumPy 数组
def ensure_float64(arr: object) -> npt.NDArray[np.float64]:  # 返回双精度浮点数类型的 NumPy 数组
    ...

# 确保输入数组转换为8位有符号整数类型的 NumPy 数组
def ensure_int8(arr: object) -> npt.NDArray[np.int8]:  # 返回8位有符号整数类型的 NumPy 数组
    ...

# 确保输入数组转换为16位有符号整数类型的 NumPy 数组
def ensure_int16(arr: object) -> npt.NDArray[np.int16]:  # 返回16位有符号整数类型的 NumPy 数组
    ...

# 确保输入数组转换为32位有符号整数类型的 NumPy 数组
def ensure_int32(arr: object) -> npt.NDArray[np.int32]:  # 返回32位有符号整数类型的 NumPy 数组
    ...

# 确保输入数组转换为64位有符号整数类型的 NumPy 数组
def ensure_int64(arr: object) -> npt.NDArray[np.int64]:  # 返回64位有符号整数类型的 NumPy 数组
    ...

# 确保输入数组转换为64位无符号整数类型的 NumPy 数组
def ensure_uint64(arr: object) -> npt.NDArray[np.uint64]:  # 返回64位无符号整数类型的 NumPy 数组
    ...

# 将一维数组的整数索引应用于另一个一维数组，并将结果存储到输出数组，输入数组和输出数组都是8位有符号整数类型
def take_1d_int8_int8(
    values: np.ndarray,  # 一维数组，包含数值对象
    indexer: npt.NDArray[np.intp],  # 整数索引数组
    out: np.ndarray,  # 输出的一维数组，用于存储结果
    fill_value=...  # 填充值，默认值未指定
) -> None:  # 返回空值，直接修改输出数组out
    ...

# 将一维数组的整数索引应用于另一个一维数组，并将结果存储到输出数组，输入数组是8位有符号整数类型，输出数组是32位有符号整数类型
def take_1d_int8_int32(
    values: np.ndarray,  # 一维数组，包含数值对象
    indexer: npt.NDArray[np.intp],  # 整数索引数组
    out: np.ndarray,  # 输出的一维数组，用于存储结果
    fill_value=...  # 填充值，默认值未指定
) -> None:  # 返回空值，直接修改输出数组out
    ...

# 其余的函数依此类推，参考具体函数名和参数注释进行理解和添加注释
    # 声明函数参数：
    # - values: np.ndarray，表示输入的数组数据类型为 NumPy 的 ndarray
    # - indexer: npt.NDArray[np.intp]，表示索引器，数据类型为 NumPy 的整数索引数组
    # - out: np.ndarray，表示输出数组的数据类型为 NumPy 的 ndarray
    # - fill_value: 可选参数，默认值为 ...
def take_1d_uint32_uint32(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从一维数组 `values` 中按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_1d_uint64_uint64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从一维数组 `values` 中按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_1d_int64_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从一维数组 `values` 中按照 `indexer` 指定的索引取值，转换为 float64 类型，并将结果存入 `out` 中
    ...

def take_1d_float32_float32(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从一维数组 `values` 中按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_1d_float32_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从一维数组 `values` 中按照 `indexer` 指定的索引取值，转换为 float64 类型，并将结果存入 `out` 中
    ...

def take_1d_float64_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从一维数组 `values` 中按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_1d_object_object(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从一维对象数组 `values` 中按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_1d_bool_bool(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从一维布尔数组 `values` 中按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_1d_bool_object(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从一维布尔数组 `values` 中按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int8_int8(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int8_int32(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int8_int64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int8_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int16_int16(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int16_int32(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int16_int64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int16_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int32_int32(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int32_int64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int32_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int64_int64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_int64_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...

def take_2d_axis0_uint16_uint16(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None:
    # 从二维数组 `values` 的第0轴（行）上按照 `indexer` 指定的索引取值，并将结果存入 `out` 中
    ...
    # 接收四个参数：values 是一个 NumPy 数组，indexer 是一个 NumPy 整数数组，
    # out 是一个 NumPy 数组，fill_value 是一个可选参数，默认值为 ...
# 定义一个函数，从二维数组中按照轴0（行）索引元素，并将结果写入指定的输出数组，支持输入和输出为 uint32 类型
def take_2d_axis0_uint32_uint32(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴0（行）索引元素，并将结果写入指定的输出数组，支持输入为 uint64 类型
def take_2d_axis0_uint64_uint64(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴0（行）索引元素，并将结果写入指定的输出数组，支持输入为 float32 和 float64 类型的混合情况
def take_2d_axis0_float32_float64(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴0（行）索引元素，并将结果写入指定的输出数组，支持输入和输出为 float32 类型
def take_2d_axis0_float32_float32(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴0（行）索引元素，并将结果写入指定的输出数组，支持输入和输出为 float64 类型
def take_2d_axis0_float64_float64(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴0（行）索引元素，并将结果写入指定的输出数组，支持输入和输出为 object 类型
def take_2d_axis0_object_object(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴0（行）索引元素，并将结果写入指定的输出数组，支持输入和输出为 bool 类型
def take_2d_axis0_bool_bool(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴0（行）索引元素，并将结果写入指定的输出数组，支持输入为 bool 类型，输出为 object 类型
def take_2d_axis0_bool_object(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴1（列）索引元素，并将结果写入指定的输出数组，支持输入和输出为 int8 类型
def take_2d_axis1_int8_int8(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴1（列）索引元素，并将结果写入指定的输出数组，支持输入为 int8 类型，输出为 int32 类型
def take_2d_axis1_int8_int32(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴1（列）索引元素，并将结果写入指定的输出数组，支持输入为 int8 类型，输出为 int64 类型
def take_2d_axis1_int8_int64(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...

# 定义一个函数，从二维数组中按照轴1（列）索引元素，并将结果写入指定的输出数组，支持输入为 int8 类型，输出为 float64 类型
def take_2d_axis1_int8_float64(
    values: np.ndarray,  # 输入的二维数组，数据类型为 np.ndarray
    indexer: npt.NDArray[np.intp],  # 索引器数组，数据类型为 npt.NDArray[np.intp]
    out: np.ndarray,  # 输出的二维数组，数据类型为 np.ndarray
    fill_value=...  # 填充值，默认未指定
) -> None:  # 函数没有返回值
    ...
    # 定义函数参数：values为numpy数组，indexer为整数类型的numpy数组，out为numpy数组，fill_value为可选参数
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
# 定义函数 take_2d_axis1_int64_float64，用于从 2D 数组 values 中按照整数索引 indexer 提取数据到输出数组 out
def take_2d_axis1_int64_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None: ...

# 定义函数 take_2d_axis1_float32_float32，用于从 2D 数组 values 中按照浮点数索引 indexer 提取数据到输出数组 out
def take_2d_axis1_float32_float32(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None: ...

# 定义函数 take_2d_axis1_float32_float64，用于从 2D 数组 values 中按照浮点数索引 indexer 提取数据到输出数组 out
def take_2d_axis1_float32_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None: ...

# 定义函数 take_2d_axis1_float64_float64，用于从 2D 数组 values 中按照浮点数索引 indexer 提取数据到输出数组 out
def take_2d_axis1_float64_float64(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None: ...

# 定义函数 take_2d_axis1_object_object，用于从 2D 数组 values 中按照对象索引 indexer 提取数据到输出数组 out
def take_2d_axis1_object_object(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None: ...

# 定义函数 take_2d_axis1_bool_bool，用于从 2D 数组 values 中按照布尔值索引 indexer 提取数据到输出数组 out
def take_2d_axis1_bool_bool(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None: ...

# 定义函数 take_2d_axis1_bool_object，用于从 2D 数组 values 中按照布尔值索引 indexer 提取数据到输出数组 out
def take_2d_axis1_bool_object(
    values: np.ndarray, indexer: npt.NDArray[np.intp], out: np.ndarray, fill_value=...
) -> None: ...

# 定义函数 take_2d_multi_int8_int8，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int8_int8(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int8_int32，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int8_int32(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int8_int64，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int8_int64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int8_float64，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int8_float64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int16_int16，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int16_int16(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int16_int32，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int16_int32(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int16_int64，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int16_int64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int16_float64，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int16_float64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int32_int32，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int32_int32(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int32_int64，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int32_int64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int32_float64，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int32_float64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...

# 定义函数 take_2d_multi_int64_float64，用于从 2D 数组 values 中按照多维整数索引 indexer 提取数据到输出数组 out
def take_2d_multi_int64_float64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
# 定义一个函数，该函数没有返回值（None），接受三个参数：values（NumPy 数组，存储浮点数数据）、indexer（元组，包含两个整数 NumPy 数组）、out（NumPy 数组，存储浮点数数据），以及可选的 fill_value 参数
def take_2d_multi_float32_float32(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...
# 定义一个函数，该函数没有返回值（None），接受三个参数：values（NumPy 数组，存储浮点数数据）、indexer（元组，包含两个整数 NumPy 数组）、out（NumPy 数组，存储浮点数数据），以及可选的 fill_value 参数
def take_2d_multi_float32_float64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...
# 定义一个函数，该函数没有返回值（None），接受三个参数：values（NumPy 数组，存储浮点数数据）、indexer（元组，包含两个整数 NumPy 数组）、out（NumPy 数组，存储浮点数数据），以及可选的 fill_value 参数
def take_2d_multi_float64_float64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...
# 定义一个函数，该函数没有返回值（None），接受三个参数：values（NumPy 数组，存储浮点数数据）、indexer（元组，包含两个整数 NumPy 数组）、out（NumPy 数组，存储对象数据），以及可选的 fill_value 参数
def take_2d_multi_object_object(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...
# 定义一个函数，该函数没有返回值（None），接受三个参数：values（NumPy 数组，存储布尔值数据）、indexer（元组，包含两个整数 NumPy 数组）、out（NumPy 数组，存储布尔值数据），以及可选的 fill_value 参数
def take_2d_multi_bool_bool(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...
# 定义一个函数，该函数没有返回值（None），接受三个参数：values（NumPy 数组，存储布尔值数据）、indexer（元组，包含两个整数 NumPy 数组）、out（NumPy 数组，存储对象数据），以及可选的 fill_value 参数
def take_2d_multi_bool_object(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...
# 定义一个函数，该函数没有返回值（None），接受三个参数：values（NumPy 数组，存储整数数据）、indexer（元组，包含两个整数 NumPy 数组）、out（NumPy 数组，存储整数数据），以及可选的 fill_value 参数
def take_2d_multi_int64_int64(
    values: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value=...,
) -> None: ...
```