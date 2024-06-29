# `D:\src\scipysrc\pandas\pandas\core\_numba\executor.py`

```
# 导入必要的模块和类型声明
from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
)

# 如果在类型检查模式下，则导入特定模块
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import Scalar

# 导入 NumPy 库并重命名为 np
import numpy as np

# 导入 pandas 的可选依赖
from pandas.compat._optional import import_optional_dependency

# 导入 numba 库中的用户函数装饰器
from pandas.core.util.numba_ import jit_user_function

# 使用 functools.cache 装饰器缓存函数的结果
@functools.cache
# 定义生成应用循环器的函数
def generate_apply_looper(func, nopython=True, nogil=True, parallel=False):
    # 如果在类型检查模式下，则导入 numba 模块
    if TYPE_CHECKING:
        import numba
    else:
        # 否则导入可选依赖的 numba
        numba = import_optional_dependency("numba")
    
    # 使用 numba.jit 装饰器对 func 进行 JIT 编译
    nb_compat_func = jit_user_function(func)

    # 定义 numba.jit 装饰的循环函数 nb_looper
    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def nb_looper(values, axis, *args):
        # 如果 axis 为 0，则操作第一行/列以获取输出形状
        if axis == 0:
            first_elem = values[:, 0]
            dim0 = values.shape[1]
        else:
            first_elem = values[0]
            dim0 = values.shape[0]
        
        # 调用 JIT 编译后的 func 处理第一个元素
        res0 = nb_compat_func(first_elem, *args)
        
        # 使用 np.asarray 获取 res0 的形状，解决特定问题
        buf_shape = (dim0,) + np.atleast_1d(np.asarray(res0)).shape
        
        # 如果 axis 为 0，则反转 buf_shape
        if axis == 0:
            buf_shape = buf_shape[::-1]
        
        # 创建一个空的缓冲区 buff
        buff = np.empty(buf_shape)

        # 根据 axis 的不同，填充缓冲区 buff
        if axis == 1:
            buff[0] = res0
            for i in numba.prange(1, values.shape[0]):
                buff[i] = nb_compat_func(values[i], *args)
        else:
            buff[:, 0] = res0
            for j in numba.prange(1, values.shape[1]):
                buff[:, j] = nb_compat_func(values[:, j], *args)
        
        # 返回填充后的缓冲区 buff
        return buff

    # 返回 JIT 编译后的循环函数 nb_looper
    return nb_looper


# 使用 functools.cache 装饰器缓存函数的结果
@functools.cache
# 定义生成循环函数的函数
def make_looper(func, result_dtype, is_grouped_kernel, nopython, nogil, parallel):
    # 如果在类型检查模式下，则导入 numba 模块
    if TYPE_CHECKING:
        import numba
    else:
        # 否则导入可选依赖的 numba
        numba = import_optional_dependency("numba")

    # 如果是分组内核，则定义针对列的循环函数 column_looper
    if is_grouped_kernel:
        @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
        def column_looper(
            values: np.ndarray,
            labels: np.ndarray,
            ngroups: int,
            min_periods: int,
            *args,
        ):
            # 创建结果数组 result 和记录 NA 位置的字典 na_positions
            result = np.empty((values.shape[0], ngroups), dtype=result_dtype)
            na_positions = {}
            
            # 遍历 values 的每一行并应用 func 函数
            for i in numba.prange(values.shape[0]):
                # 调用 func 处理当前行，获取输出和 NA 位置
                output, na_pos = func(
                    values[i], result_dtype, labels, ngroups, min_periods, *args
                )
                # 将处理结果存入 result 数组的当前行
                result[i] = output
                # 如果 NA 位置不为空，则记录到 na_positions 中
                if len(na_pos) > 0:
                    na_positions[i] = np.array(na_pos)
            
            # 返回处理结果和 NA 位置字典
            return result, na_positions

        # 返回 column_looper 函数
        return column_looper
    # 如果条件不满足，则定义一个装饰了 Numba JIT 编译器优化的函数 column_looper
    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    # 函数 column_looper 接受多个参数：values 是一个 NumPy 数组，start 和 end 是 NumPy 数组，
    # min_periods 是一个整数，*args 是可变长度参数列表
    def column_looper(
        values: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        min_periods: int,
        *args,
    ):
        # 创建一个空的 NumPy 数组 result，用于存储计算结果，其形状为 (values.shape[0], len(start))，数据类型为 result_dtype
        result = np.empty((values.shape[0], len(start)), dtype=result_dtype)
        # 初始化一个空字典 na_positions，用于存储缺失值的位置信息
        na_positions = {}
        # 使用 Numba 提供的并行循环进行迭代，循环次数为 values 数组的第一个维度的大小
        for i in numba.prange(values.shape[0]):
            # 调用 func 函数处理 values[i]，并返回输出结果 output 和缺失值位置列表 na_pos
            output, na_pos = func(
                values[i], result_dtype, start, end, min_periods, *args
            )
            # 将 func 的输出结果 output 存储到 result 的第 i 行
            result[i] = output
            # 如果 na_pos 长度大于 0，则将其转换为 NumPy 数组并存储到 na_positions 字典中，键为 i
            if len(na_pos) > 0:
                na_positions[i] = np.array(na_pos)
        # 返回计算结果 result 和缺失值位置字典 na_positions
        return result, na_positions

    # 返回定义好的函数 column_looper
    return column_looper
# 定义默认的数据类型映射字典，将 NumPy 数据类型映射到对应的目标数据类型
default_dtype_mapping: dict[np.dtype, Any] = {
    np.dtype("int8"): np.int64,
    np.dtype("int16"): np.int64,
    np.dtype("int32"): np.int64,
    np.dtype("int64"): np.int64,
    np.dtype("uint8"): np.uint64,
    np.dtype("uint16"): np.uint64,
    np.dtype("uint32"): np.uint64,
    np.dtype("uint64"): np.uint64,
    np.dtype("float32"): np.float64,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.complex128,
    np.dtype("complex128"): np.complex128,
}

# TODO: 保留复杂数据类型的映射

# 定义浮点数据类型的映射字典，将 NumPy 数据类型映射到浮点数的目标数据类型（统一为 np.float64）
float_dtype_mapping: dict[np.dtype, Any] = {
    np.dtype("int8"): np.float64,
    np.dtype("int16"): np.float64,
    np.dtype("int32"): np.float64,
    np.dtype("int64"): np.float64,
    np.dtype("uint8"): np.float64,
    np.dtype("uint16"): np.float64,
    np.dtype("uint32"): np.float64,
    np.dtype("uint64"): np.float64,
    np.dtype("float32"): np.float64,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.float64,
    np.dtype("complex128"): np.float64,
}

# 定义恒等映射数据类型的字典，将 NumPy 数据类型映射到其自身
identity_dtype_mapping: dict[np.dtype, Any] = {
    np.dtype("int8"): np.int8,
    np.dtype("int16"): np.int16,
    np.dtype("int32"): np.int32,
    np.dtype("int64"): np.int64,
    np.dtype("uint8"): np.uint8,
    np.dtype("uint16"): np.uint16,
    np.dtype("uint32"): np.uint32,
    np.dtype("uint64"): np.uint64,
    np.dtype("float32"): np.float32,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.complex64,
    np.dtype("complex128"): np.complex128,
}


def generate_shared_aggregator(
    func: Callable[..., Scalar],
    dtype_mapping: dict[np.dtype, np.dtype],
    is_grouped_kernel: bool,
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    """
    Generate a Numba function that loops over the columns 2D object and applies
    a 1D numba kernel over each column.

    Parameters
    ----------
    func : function
        aggregation function to be applied to each column
    dtype_mapping: dict or None
        If not None, maps a dtype to a result dtype.
        Otherwise, will fall back to default mapping.
    is_grouped_kernel: bool, default False
        Whether func operates using the group labels (True)
        or using starts/ends arrays

        If true, you also need to pass the number of groups to this function
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """

    # 一个包装器函数，围绕 looper 函数，根据数据类型分发调用，因为 numba 无法在 nopython 模式下执行此操作
    # 这个函数还会在值少于 min_periods 时插入 NaN，不能在 numba 的 nopython 模式下执行
    # （会出现类型统一错误，将 int 转换为 float）
    # 定义一个装饰器函数 looper_wrapper，接受多个参数和关键字参数
    def looper_wrapper(
        values,
        start=None,
        end=None,
        labels=None,
        ngroups=None,
        min_periods: int = 0,
        **kwargs,
    ):
        # 根据 values 的数据类型从 dtype_mapping 中获取对应的结果数据类型
        result_dtype = dtype_mapping[values.dtype]
        # 根据传入的参数创建一个循环器函数 column_looper
        column_looper = make_looper(
            func, result_dtype, is_grouped_kernel, nopython, nogil, parallel
        )
        # 由于 numba 只支持 *args，需要拆开 kwargs
        # 如果是分组内核，则使用 labels、ngroups、min_periods 和 kwargs.values() 调用 column_looper
        if is_grouped_kernel:
            result, na_positions = column_looper(
                values, labels, ngroups, min_periods, *kwargs.values()
            )
        else:
            # 否则使用 start、end、min_periods 和 kwargs.values() 调用 column_looper
            result, na_positions = column_looper(
                values, start, end, min_periods, *kwargs.values()
            )
        # 如果结果的数据类型是整数类型
        if result.dtype.kind == "i":
            # 检查 na_positions 是否非空
            # 如果是，则将整个结果转换为浮点数类型
            # 这是因为整数类型无法容纳 NaN，因此如果某列不满足 min_periods，则所有该索引下的列都不满足
            for na_pos in na_positions.values():
                if len(na_pos) > 0:
                    result = result.astype("float64")
                    break
        # 遍历 na_positions 中的每个索引 i 和对应的 na_pos
        # 如果 na_pos 长度大于 0，则在结果的相应位置设置为 NaN
        for i, na_pos in na_positions.items():
            if len(na_pos) > 0:
                result[i, na_pos] = np.nan
        # 返回处理后的结果
        return result

    # 返回 looper_wrapper 装饰器函数
    return looper_wrapper
```