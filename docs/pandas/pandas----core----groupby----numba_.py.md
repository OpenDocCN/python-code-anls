# `D:\src\scipysrc\pandas\pandas\core\groupby\numba_.py`

```
"""Common utilities for Numba operations with groupby ops"""

from __future__ import annotations

import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.core.util.numba_ import (
    NumbaUtilError,
    jit_user_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import Scalar


def validate_udf(func: Callable) -> None:
    """
    Validate user defined function for ops when using Numba with groupby ops.

    The first signature arguments should include:

    def f(values, index, ...):
        ...

    Parameters
    ----------
    func : function, default False
        user defined function

    Returns
    -------
    None

    Raises
    ------
    NumbaUtilError
    """
    # 检查 func 是否为可调用对象，否则抛出异常
    if not callable(func):
        raise NotImplementedError(
            "Numba engine can only be used with a single function."
        )
    # 获取用户定义函数的参数签名列表
    udf_signature = list(inspect.signature(func).parameters.keys())
    expected_args = ["values", "index"]
    min_number_args = len(expected_args)
    # 检查函数签名是否符合预期
    if (
        len(udf_signature) < min_number_args
        or udf_signature[:min_number_args] != expected_args
    ):
        raise NumbaUtilError(
            f"The first {min_number_args} arguments to {func.__name__} must be "
            f"{expected_args}"
        )


@functools.cache
def generate_numba_agg_func(
    func: Callable[..., Scalar],
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any], np.ndarray]:
    """
    Generate a numba jitted agg function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a groupby agg function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the groupby evaluation loop.

    Parameters
    ----------
    func : function
        function to be applied to each group and will be JITed
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
    # 使用 jit_user_function 函数对用户定义的函数进行 JIT 编译
    numba_func = jit_user_function(func)
    # 如果是类型检查阶段，导入 numba 模块；否则，使用延迟加载导入 numba
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_agg(
        values: np.ndarray,
        index: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        num_columns: int,
        *args: Any,
    ) -> np.ndarray:
        # 定义函数，接收 begin 和 end 参数，返回类型为 np.ndarray
        assert len(begin) == len(end)
        # 断言确保 begin 和 end 的长度相同，用于分组操作

        num_groups = len(begin)
        # 获取分组的数量

        result = np.empty((num_groups, num_columns))
        # 创建一个空的 numpy 数组用于存放结果，形状为 (num_groups, num_columns)

        for i in numba.prange(num_groups):
            # 使用 numba.prange 迭代每个分组
            group_index = index[begin[i] : end[i]]
            # 获取当前分组的索引范围

            for j in numba.prange(num_columns):
                # 使用 numba.prange 迭代每个列
                group = values[begin[i] : end[i], j]
                # 获取当前分组中第 j 列的数据

                result[i, j] = numba_func(group, group_index, *args)
                # 调用 numba_func 处理当前分组第 j 列的数据，并将结果存入 result 数组

        return result
        # 返回处理完的结果数组

    return group_agg
    # 返回定义好的 group_agg 函数
@functools.cache
def generate_numba_transform_func(
    func: Callable[..., np.ndarray],
    nopython: bool,
    nogil: bool,
    parallel: bool,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any], np.ndarray]:
    """
    Generate a numba jitted transform function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a groupby transform function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the groupby evaluation loop.

    Parameters
    ----------
    func : function
        function to be applied to each window and will be JITed
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
    # JIT compile the user-provided function using numba
    numba_func = jit_user_function(func)

    # Import numba or mock if TYPE_CHECKING is enabled
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    # Define the group transform function with numba.jit compilation
    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def group_transform(
        values: np.ndarray,
        index: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        num_columns: int,
        *args: Any,
    ) -> np.ndarray:
        # Assert that begin and end arrays have the same length
        assert len(begin) == len(end)
        num_groups = len(begin)

        # Initialize result array to store transformed values
        result = np.empty((len(values), num_columns))

        # Iterate over groups using numba.prange for parallel execution
        for i in numba.prange(num_groups):
            group_index = index[begin[i] : end[i]]
            # Process each column of the group in parallel
            for j in numba.prange(num_columns):
                group = values[begin[i] : end[i], j]
                # Apply the jitted user function to the group and store the result
                result[begin[i] : end[i], j] = numba_func(group, group_index, *args)

        return result

    return group_transform
```