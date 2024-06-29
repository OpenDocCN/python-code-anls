# `D:\src\scipysrc\pandas\pandas\_libs\lib.pyi`

```
# 导入 Decimal 和 typing 模块中的一些类型
# 部分类型具体的版本被注释在注释中
from decimal import Decimal
from typing import (
    Any,
    Callable,
    Final,               # 最终类型声明
    Generator,
    Hashable,
    Literal,             # 字面量类型
    TypeAlias,
    overload,
)

import numpy as np     # 导入 NumPy 库

from pandas._typing import (
    ArrayLike,          # 类似数组的类型
    DtypeObj,           # 数据类型对象
    TypeGuard,
    npt,
)

# 占位符，直到能够指定 np.ndarray[object, ndim=2] 为止
ndarray_obj_2d = np.ndarray

from enum import Enum  # 导入枚举类型

class _NoDefault(Enum):
    no_default = ...    # 表示没有默认值

no_default: Final = _NoDefault.no_default  # 使用枚举类型作为不可变的最终值
NoDefault: TypeAlias = Literal[_NoDefault.no_default]  # 使用字面量类型别名表示无默认值

i8max: int             # 最大值整数
u8max: int             # 最大无符号整数

# 检查是否为 NumPy 数据类型
def is_np_dtype(dtype: object, kinds: str | None = ...) -> TypeGuard[np.dtype]: ...

# 从零维数据中获取项
def item_from_zerodim(val: object) -> object: ...

# 推断数据类型
def infer_dtype(value: object, skipna: bool = ...) -> str: ...

# 检查对象是否为迭代器
def is_iterator(obj: object) -> bool: ...

# 检查对象是否为标量
def is_scalar(val: object) -> bool: ...

# 检查对象是否类似列表
def is_list_like(obj: object, allow_sets: bool = ...) -> bool: ...

# 检查对象是否为 pyarrow 数组
def is_pyarrow_array(obj: object) -> bool: ...

# 检查对象是否为 Decimal 类型
def is_decimal(obj: object) -> TypeGuard[Decimal]: ...

# 检查对象是否为复数类型
def is_complex(obj: object) -> TypeGuard[complex]: ...

# 检查对象是否为布尔类型（包括 NumPy 中的布尔类型）
def is_bool(obj: object) -> TypeGuard[bool | np.bool_]: ...

# 检查对象是否为整数类型（包括 NumPy 中的整数类型）
def is_integer(obj: object) -> TypeGuard[int | np.integer]: ...

# 检查对象是否为整数或 None
def is_int_or_none(obj) -> bool: ...

# 检查对象是否为浮点数类型
def is_float(obj: object) -> TypeGuard[float]: ...

# 检查 NumPy 数组是否为区间数组
def is_interval_array(values: np.ndarray) -> bool: ...

# 检查 NumPy 数组是否为 datetime64 类型数组
def is_datetime64_array(values: np.ndarray, skipna: bool = True) -> bool: ...

# 检查 NumPy 数组是否为 timedelta64 类型数组
def is_timedelta_or_timedelta64_array(
    values: np.ndarray, skipna: bool = True
) -> bool: ...

# 检查 NumPy 数组是否为带单一时区的 datetime 类型数组
def is_datetime_with_singletz_array(values: np.ndarray) -> bool: ...

# 检查 NumPy 数组是否为时间数组
def is_time_array(values: np.ndarray, skipna: bool = ...): ...

# 检查 NumPy 数组是否为日期数组
def is_date_array(values: np.ndarray, skipna: bool = ...): ...

# 检查 NumPy 数组是否为日期时间数组
def is_datetime_array(values: np.ndarray, skipna: bool = ...): ...

# 检查 NumPy 数组是否为字符串数组
def is_string_array(values: np.ndarray, skipna: bool = ...): ...

# 检查 NumPy 数组是否为浮点数数组
def is_float_array(values: np.ndarray): ...

# 检查 NumPy 数组是否为整数数组
def is_integer_array(values: np.ndarray, skipna: bool = ...): ...

# 检查 NumPy 数组是否为布尔数组
def is_bool_array(values: np.ndarray, skipna: bool = ...): ...

# 快速多重获取函数，从映射中获取给定键的值
def fast_multiget(
    mapping: dict,
    keys: np.ndarray,  # 对象类型的数组
    default=...,
) -> ArrayLike: ...

# 快速生成多个唯一值列表的生成器
def fast_unique_multiple_list_gen(gen: Generator, sort: bool = ...) -> list: ...

# 根据重载提供的参数映射推断函数
@overload
def map_infer(
    arr: np.ndarray,
    f: Callable[[Any], Any],
    *,
    convert: Literal[False],  # 不进行转换
    ignore_na: bool = ...,
) -> np.ndarray: ...

@overload
def map_infer(
    arr: np.ndarray,
    f: Callable[[Any], Any],
    *,
    convert: bool = ...,
    ignore_na: bool = ...,
) -> ArrayLike: ...

# 可能转换对象数组的函数
@overload
def maybe_convert_objects(
    objects: npt.NDArray[np.object_],
    *,
    try_float: bool = ...,
    safe: bool = ...,
    convert_numeric: bool = ...,
    convert_non_numeric: Literal[False] = ...,
    convert_to_nullable_dtype: Literal[False] = ...,
    dtype_if_all_nat: DtypeObj | None = ...,
) -> npt.NDArray[np.object_ | np.number]: ...
# 函数签名声明，用于将包含 np.object_ 类型对象的数组转换为 ArrayLike 类型
def maybe_convert_objects(
    objects: npt.NDArray[np.object_],
    *,
    try_float: bool = ...,
    safe: bool = ...,
    convert_numeric: bool = ...,
    convert_non_numeric: bool = ...,
    convert_to_nullable_dtype: Literal[True] = ...,
    dtype_if_all_nat: DtypeObj | None = ...,
) -> ArrayLike: ...

# 函数签名声明的重载，用于将包含 np.object_ 类型对象的数组转换为 ArrayLike 类型
@overload
def maybe_convert_objects(
    objects: npt.NDArray[np.object_],
    *,
    try_float: bool = ...,
    safe: bool = ...,
    convert_numeric: bool = ...,
    convert_non_numeric: bool = ...,
    convert_to_nullable_dtype: bool = ...,
    dtype_if_all_nat: DtypeObj | None = ...,
) -> ArrayLike: ...

# 函数签名声明，用于将包含 np.object_ 类型对象的数组转换为数值类型的数组
def maybe_convert_numeric(
    values: npt.NDArray[np.object_],
    na_values: set,
    convert_empty: bool = ...,
    coerce_numeric: bool = ...,
    convert_to_masked_nullable: Literal[False] = ...,
) -> tuple[np.ndarray, None]: ...

# 函数签名声明的重载，用于将包含 np.object_ 类型对象的数组转换为数值类型的数组
@overload
def maybe_convert_numeric(
    values: npt.NDArray[np.object_],
    na_values: set,
    convert_empty: bool = ...,
    coerce_numeric: bool = ...,
    *,
    convert_to_masked_nullable: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...

# TODO: 是否需要限制 `arr` 的类型？
def ensure_string_array(
    arr,
    na_value: object = ...,
    convert_na_value: bool = ...,
    copy: bool = ...,
    skipna: bool = ...,
) -> npt.NDArray[np.object_]: ...

# 函数签名声明，用于将包含 np.object_ 类型对象的数组中的 NaN 值转换为 NA
def convert_nans_to_NA(
    arr: npt.NDArray[np.object_],
) -> npt.NDArray[np.object_]: ...

# 函数签名声明，用于快速创建多个 ndarray 的笛卡尔积
def fast_zip(ndarrays: list) -> npt.NDArray[np.object_]: ...

# TODO: 是否可以对 `rows` 的类型进行更具体的描述？
def to_object_array_tuples(rows: object) -> ndarray_obj_2d: ...

# 函数签名声明，用于将包含元组的数组转换为对象数组的二维数组
def tuples_to_object_array(
    tuples: npt.NDArray[np.object_],
) -> ndarray_obj_2d: ...

# TODO: 是否可以对 `rows` 的类型进行更具体的描述？
def to_object_array(rows: object, min_width: int = ...) -> ndarray_obj_2d: ...

# 函数签名声明，用于将包含字典的列表转换为对象数组的二维数组
def dicts_to_array(dicts: list, columns: list) -> ndarray_obj_2d: ...

# 函数签名声明，用于将可能为布尔值的数组转换为切片或布尔数组
def maybe_booleans_to_slice(
    mask: npt.NDArray[np.uint8],
) -> slice | npt.NDArray[np.uint8]: ...

# 函数签名声明，用于将可能为索引的数组转换为切片或整数索引数组
def maybe_indices_to_slice(
    indices: npt.NDArray[np.intp],
    max_len: int,
) -> slice | npt.NDArray[np.intp]: ...

# 函数签名声明，用于判断给定对象是否都是数组
def is_all_arraylike(obj: list) -> bool: ...

# -----------------------------------------------------------------
# 下面的函数实际上接受内存视图作为参数

# 函数签名声明，用于获取包含对象的 ndarray 对象的内存使用量
def memory_usage_of_objects(arr: np.ndarray) -> int: ...  # object[:]  # np.int64

# 函数签名声明，用于根据给定的函数推断 mask，并返回处理后的数组
@overload
def map_infer_mask(
    arr: np.ndarray,
    f: Callable[[Any], Any],
    mask: np.ndarray,  # const uint8_t[:]
    *,
    convert: Literal[False],
    na_value: Any = ...,
    dtype: np.dtype = ...,
) -> np.ndarray: ...

# 函数签名声明的重载，用于根据给定的函数推断 mask，并返回处理后的数组
@overload
def map_infer_mask(
    arr: np.ndarray,
    f: Callable[[Any], Any],
    mask: np.ndarray,  # const uint8_t[:]
    *,
    convert: bool = ...,
    na_value: Any = ...,
    dtype: np.dtype = ...,
) -> ArrayLike: ...

# 函数签名声明，用于快速索引给定索引和标签
def indices_fast(
    index: npt.NDArray[np.intp],
    labels: np.ndarray,  # const int64_t[:]
    keys: list,
    sorted_labels: list[npt.NDArray[np.int64]],
) -> dict[Hashable, npt.NDArray[np.intp]]: ...
# 根据标签和组数生成切片
def generate_slices(
    labels: np.ndarray,
    ngroups: int,  # const intp_t[:]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...

# 计算二维掩码中每个等级的数量
def count_level_2d(
    mask: np.ndarray,  # ndarray[uint8_t, ndim=2, cast=True],
    labels: np.ndarray,  # const intp_t[:]
    max_bin: int,
) -> np.ndarray: ...  # np.ndarray[np.int64, ndim=2]

# 获取等级排序器
def get_level_sorter(
    codes: np.ndarray,  # const int64_t[:]
    starts: np.ndarray,  # const intp_t[:]
) -> np.ndarray: ...  # np.ndarray[np.intp, ndim=1]

# 生成 datetime64 类型的分箱
def generate_bins_dt64(
    values: npt.NDArray[np.int64],
    binner: np.ndarray,  # const int64_t[:]
    closed: object = ...,
    hasnans: bool = ...,
) -> np.ndarray: ...  # np.ndarray[np.int64, ndim=1]

# 检查两个对象数组是否相等
def array_equivalent_object(
    left: npt.NDArray[np.object_],
    right: npt.NDArray[np.object_],
) -> bool: ...

# 检查数组是否包含无穷大数值
def has_infs(arr: np.ndarray) -> bool: ...  # const floating[:]

# 检查数组是否只包含整数或 NaN 值
def has_only_ints_or_nan(arr: np.ndarray) -> bool: ...  # const floating[:]

# 获取逆向索引器
def get_reverse_indexer(
    indexer: np.ndarray,  # const intp_t[:]
    length: int,
) -> npt.NDArray[np.intp]: ...

# 判断对象是否为布尔列表
def is_bool_list(obj: list) -> bool: ...

# 检查所有数据类型是否相等
def dtypes_all_equal(types: list[DtypeObj]) -> bool: ...

# 判断数组是否为范围索引器
def is_range_indexer(
    left: np.ndarray,
    n: int,  # np.ndarray[np.int64, ndim=1]
) -> bool: ...

# 判断数组是否为序列范围
def is_sequence_range(
    sequence: np.ndarray,
    step: int,  # np.ndarray[np.int64, ndim=1]
) -> bool: ...
```