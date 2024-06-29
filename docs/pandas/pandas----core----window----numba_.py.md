# `D:\src\scipysrc\pandas\pandas\core\window\numba_.py`

```
# 从未来导入类型注解，用于支持类型提示
from __future__ import annotations

# 导入 functools 模块，用于函数装饰器的支持
import functools

# 导入 TYPE_CHECKING，用于类型检查时的条件判断
from typing import (
    TYPE_CHECKING,
    Any,
)

# 导入 numpy 库，用于处理数值计算
import numpy as np

# 导入 pandas 的可选依赖项导入函数
from pandas.compat._optional import import_optional_dependency

# 导入 pandas 的 numba_ 模块下的 jit_user_function 函数
from pandas.core.util.numba_ import jit_user_function

# 如果是类型检查状态，导入 Callable 和 Scalar 类型
if TYPE_CHECKING:
    from collections.abc import Callable
    from pandas._typing import Scalar

# 使用 functools.cache 装饰器缓存函数的调用结果
@functools.cache
def generate_numba_apply_func(
    func: Callable[..., Scalar],  # 接受一个函数作为参数，返回一个标量值
    nopython: bool,  # 是否启用 Numba 的 nopython 模式
    nogil: bool,  # 是否启用 Numba 的 nogil 模式
    parallel: bool,  # 是否启用 Numba 的 parallel 模式
):
    """
    Generate a numba jitted apply function specified by values from engine_kwargs.

    1. jit the user's function
    2. Return a rolling apply function with the jitted function inline

    Configurations specified in engine_kwargs apply to both the user's
    function _AND_ the rolling apply function.

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
    # 对用户函数进行 Numba JIT 编译
    numba_func = jit_user_function(func)
    
    # 在类型检查状态下，导入 numba 库；否则，导入可选依赖项 "numba"
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    # 定义一个 Numba JIT 编译后的滚动应用函数
    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def roll_apply(
        values: np.ndarray,  # 输入数据的数组
        begin: np.ndarray,  # 起始位置的数组
        end: np.ndarray,  # 结束位置的数组
        minimum_periods: int,  # 最小周期数
        *args: Any,  # 其他参数
    ) -> np.ndarray:  # 返回一个 numpy 数组
        result = np.empty(len(begin))  # 创建一个与 begin 数组长度相同的空数组
        for i in numba.prange(len(result)):  # 使用 Numba 并行循环
            start = begin[i]  # 获取起始位置
            stop = end[i]  # 获取结束位置
            window = values[start:stop]  # 根据起始和结束位置获取窗口数据
            count_nan = np.sum(np.isnan(window))  # 统计窗口中的 NaN 值数量
            if len(window) - count_nan >= minimum_periods:  # 如果窗口有效数据点数大于等于最小周期数
                result[i] = numba_func(window, *args)  # 调用 JIT 编译后的用户函数计算结果
            else:
                result[i] = np.nan  # 否则，结果为 NaN
        return result  # 返回计算结果数组

    return roll_apply  # 返回 JIT 编译后的滚动应用函数


# 使用 functools.cache 装饰器缓存函数的调用结果
@functools.cache
def generate_numba_ewm_func(
    nopython: bool,  # 是否启用 Numba 的 nopython 模式
    nogil: bool,  # 是否启用 Numba 的 nogil 模式
    parallel: bool,  # 是否启用 Numba 的 parallel 模式
    com: float,  # EWM 函数的 com 参数
    adjust: bool,  # EWM 函数的 adjust 参数
    ignore_na: bool,  # EWM 函数的 ignore_na 参数
    deltas: tuple,  # EWM 函数的 deltas 参数
    normalize: bool,  # EWM 函数的 normalize 参数
):
    """
    Generate a numba jitted ewm mean or sum function specified by values
    from engine_kwargs.

    Parameters
    ----------
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit
    com : float
    adjust : bool
    ignore_na : bool
    deltas : tuple
    normalize : bool

    Returns
    -------
    Numba function
    """
    # 在类型检查状态下，导入 numba 库；否则，导入可选依赖项 "numba"
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    # 定义一个 Numba JIT 编译后的 EWM 函数
    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def ewm(
        values: np.ndarray,  # 输入数据的数组
        begin: np.ndarray,  # 起始位置的数组
        end: np.ndarray,  # 结束位置的数组
        minimum_periods: int,  # 最小周期数
        *args: Any,  # 其他参数
    ) -> np.ndarray:  # 返回一个 numpy 数组
        # 函数体未完整提供，在此省略具体实现细节
        pass

    return ewm  # 返回 JIT 编译后的 EWM 函数
    # 定义一个函数 ewm，接收 values 数组和一些参数，并返回一个 NumPy 数组
    ) -> np.ndarray:
        # 创建一个空的 NumPy 数组，长度与 values 相同，用于存储计算结果
        result = np.empty(len(values))
        # 计算指数加权移动平均的 alpha 值，alpha 是一个权重因子
        alpha = 1.0 / (1.0 + com)
        # 计算旧权重因子，用于指数加权移动平均的计算
        old_wt_factor = 1.0 - alpha
        # 如果 adjust 参数为 True，则使用全新的权重，否则使用 alpha
        new_wt = 1.0 if adjust else alpha

        # 使用 numba.prange 并行迭代 begin 数组的每个元素
        for i in numba.prange(len(begin)):
            # 获取当前迭代的起始和结束位置
            start = begin[i]
            stop = end[i]
            # 获取 values 数组中 start 到 stop 范围内的子数组
            window = values[start:stop]
            # 创建一个空的 NumPy 数组，用于存储子结果
            sub_result = np.empty(len(window))

            # 初始加权值为 window 中的第一个元素
            weighted = window[0]
            # 判断第一个值是否为有效观测值，并计算有效观测值的数量
            nobs = int(not np.isnan(weighted))
            # 如果有效观测值数量不足 minimum_periods，则将第一个结果设为 NaN
            sub_result[0] = weighted if nobs >= minimum_periods else np.nan
            # 初始旧权重为 1.0
            old_wt = 1.0

            # 遍历 window 中的每个元素
            for j in range(1, len(window)):
                cur = window[j]
                is_observation = not np.isnan(cur)
                nobs += is_observation

                # 如果加权值不为 NaN，则根据情况调整加权值
                if not np.isnan(weighted):
                    if is_observation or not ignore_na:
                        # 如果 normalize 为 True，则根据 deltas 调整旧权重因子
                        if normalize:
                            # 注意 deltas 的长度为 vals 的长度减 1，deltas[i] 与 vals[i+1] 配合使用
                            old_wt *= old_wt_factor ** deltas[start + j - 1]
                        else:
                            # 否则直接按照旧权重因子调整加权值
                            weighted = old_wt_factor * weighted

                        # 如果当前值为有效观测值，则根据情况调整加权值
                        if is_observation:
                            if normalize:
                                # 避免在常数序列上的数值误差
                                if weighted != cur:
                                    weighted = old_wt * weighted + new_wt * cur
                                    if normalize:
                                        weighted = weighted / (old_wt + new_wt)
                                # 根据 adjust 参数调整旧权重
                                if adjust:
                                    old_wt += new_wt
                                else:
                                    old_wt = 1.0
                            else:
                                # 直接累加有效观测值到加权值上
                                weighted += cur
                elif is_observation:
                    # 如果加权值为 NaN 且当前值为有效观测值，则将加权值设为当前值
                    weighted = cur

                # 如果有效观测值数量不足 minimum_periods，则将当前结果设为 NaN
                sub_result[j] = weighted if nobs >= minimum_periods else np.nan

            # 将子结果 sub_result 复制到结果数组 result 的对应位置
            result[start:stop] = sub_result

        # 返回最终计算得到的结果数组 result
        return result

    # 返回定义的指数加权移动平均函数 ewm
    return ewm
# 使用 functools.cache 装饰器缓存生成的函数，以便后续调用
@functools.cache
def generate_numba_table_func(
    func: Callable[..., np.ndarray],
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    """
    Generate a numba jitted function to apply window calculations table-wise.

    Func will be passed a M window size x N number of columns array, and
    must return a 1 x N number of columns array.

    1. jit the user's function
    2. Return a rolling apply function with the jitted function inline

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
    # 对用户提供的函数进行 JIT 编译
    numba_func = jit_user_function(func)
    
    # 根据 TYPE_CHECKING 导入 numba 模块或者从依赖项中导入
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    # 定义一个 numba.jit 编译的函数，用于表格级别的滚动应用
    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def roll_table(
        values: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        minimum_periods: int,
        *args: Any,
    ):
        result = np.empty((len(begin), values.shape[1]))  # 创建一个空的结果数组
        min_periods_mask = np.empty(result.shape)         # 创建一个空的最小周期掩码数组
        # 使用 numba.prange 并行迭代结果数组的索引
        for i in numba.prange(len(result)):
            start = begin[i]                # 获取起始索引
            stop = end[i]                   # 获取结束索引
            window = values[start:stop]     # 提取窗口数据
            count_nan = np.sum(np.isnan(window), axis=0)  # 统计窗口中的 NaN 数量
            nan_mask = len(window) - count_nan >= minimum_periods  # 创建 NaN 掩码
            if nan_mask.any():
                result[i, :] = numba_func(window, *args)   # 如果有有效数据，则应用 numba_func 函数
            min_periods_mask[i, :] = nan_mask  # 更新最小周期掩码数组
        result = np.where(min_periods_mask, result, np.nan)  # 将不满足最小周期的值设置为 NaN
        return result

    return roll_table


# This function will no longer be needed once numba supports
# axis for all np.nan* agg functions
# https://github.com/numba/numba/issues/1269
# 使用 functools.cache 装饰器缓存生成的函数，以便后续调用
@functools.cache
def generate_manual_numpy_nan_agg_with_axis(nan_func):
    # 根据 TYPE_CHECKING 导入 numba 模块或者从依赖项中导入
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    # 定义一个 numba.jit 编译的函数，用于沿轴执行 NaN 聚合
    @numba.jit(nopython=True, nogil=True, parallel=True)
    def nan_agg_with_axis(table):
        result = np.empty(table.shape[1])   # 创建一个空结果数组
        # 使用 numba.prange 并行迭代表格的列索引
        for i in numba.prange(table.shape[1]):
            partition = table[:, i]         # 获取表格的每一列
            result[i] = nan_func(partition) # 应用给定的 NaN 聚合函数
        return result

    return nan_agg_with_axis


# 使用 functools.cache 装饰器缓存生成的函数，以便后续调用
@functools.cache
def generate_numba_ewm_table_func(
    nopython: bool,
    nogil: bool,
    parallel: bool,
    com: float,
    adjust: bool,
    ignore_na: bool,
    deltas: tuple,
    normalize: bool,
):
    """
    Generate a numba jitted ewm mean or sum function applied table wise specified
    by values from engine_kwargs.

    Parameters
    ----------
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit
    com : float
        ...
    """
    # 根据 TYPE_CHECKING 导入 numba 模块或者从依赖项中导入
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    # 定义一个 numba.jit 编译的 ewm（指数加权移动）均值或总和函数
    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def ewm_table_func(
        values: np.ndarray,
        span: int,
        adjust: bool,
        ignore_na: bool,
        deltas: tuple,
        normalize: bool,
    ):
        ...
        # 函数主体省略，根据需求填写

    return ewm_table_func
    adjust : bool
    # 是否调整加权系数的标志，类型为布尔值

    ignore_na : bool
    # 是否忽略缺失值的标志，类型为布尔值

    deltas : tuple
    # 用于存储增量值的元组，类型为元组

    normalize: bool
    # 是否进行归一化处理的标志，类型为布尔值

    Returns
    -------
    Numba function
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")
    # 根据是否在类型检查阶段导入 numba 库或使用可选依赖导入

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    # 使用 numba 装饰器 jit 对 ewm_table 函数进行加速优化
    def ewm_table(
        values: np.ndarray,
        begin: np.ndarray,
        end: np.ndarray,
        minimum_periods: int,
    ) -> np.ndarray:
        alpha = 1.0 / (1.0 + com)
        # 计算指数加权移动平均的衰减因子 alpha

        old_wt_factor = 1.0 - alpha
        # 计算旧权重因子

        new_wt = 1.0 if adjust else alpha
        # 根据 adjust 标志确定新权重因子

        old_wt = np.ones(values.shape[1])
        # 初始化旧权重数组，长度与数据列数相同

        result = np.empty(values.shape)
        # 创建与 values 形状相同的空数组用于存储结果

        weighted = values[0].copy()
        # 复制第一行数据作为初始加权数据

        nobs = (~np.isnan(weighted)).astype(np.int64)
        # 计算非缺失值的观测数，并转换为整型数组

        result[0] = np.where(nobs >= minimum_periods, weighted, np.nan)
        # 将第一行结果存储到 result 中，若观测数不足最小周期，则标记为 NaN

        for i in range(1, len(values)):
            cur = values[i]
            # 获取当前行数据

            is_observations = ~np.isnan(cur)
            # 判断当前行的观测值情况

            nobs += is_observations.astype(np.int64)
            # 更新观测数

            for j in numba.prange(len(cur)):
                if not np.isnan(weighted[j]):
                    # 如果加权值不是 NaN
                    if is_observations[j] or not ignore_na:
                        # 如果当前有观测值或不忽略缺失值
                        if normalize:
                            # 如果需要归一化
                            old_wt[j] *= old_wt_factor ** deltas[i - 1]
                            # 根据增量值更新旧权重
                        else:
                            weighted[j] = old_wt_factor * weighted[j]
                            # 根据旧权重因子更新加权值

                        if is_observations[j]:
                            # 如果有观测值
                            if normalize:
                                # 如果需要归一化
                                if weighted[j] != cur[j]:
                                    weighted[j] = (
                                        old_wt[j] * weighted[j] + new_wt * cur[j]
                                    )
                                    # 更新加权值
                                    if normalize:
                                        weighted[j] = weighted[j] / (old_wt[j] + new_wt)
                                    # 归一化加权值

                                if adjust:
                                    old_wt[j] += new_wt
                                else:
                                    old_wt[j] = 1.0
                                # 根据 adjust 标志更新旧权重
                            else:
                                weighted[j] += cur[j]
                                # 直接加上当前值
                elif is_observations[j]:
                    # 如果没有加权值但有观测值
                    weighted[j] = cur[j]
                    # 直接使用当前值作为加权值

            result[i] = np.where(nobs >= minimum_periods, weighted, np.nan)
            # 将当前行结果存储到 result 中，若观测数不足最小周期，则标记为 NaN

        return result
        # 返回计算结果数组

    return ewm_table
    # 返回指数加权移动平均计算函数
```