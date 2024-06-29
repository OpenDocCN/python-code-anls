# `D:\src\scipysrc\pandas\pandas\core\window\online.py`

```
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency

# 导入相关库和模块，包括未来的类型注解和依赖项导入


def generate_online_numba_ewma_func(
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    """
    Generate a numba jitted groupby ewma function specified by values
    from engine_kwargs.

    Parameters
    ----------
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
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")
    
    # 根据传入的参数创建一个使用 Numba JIT 编译的在线指数加权移动平均函数
    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def online_ewma(
        values: np.ndarray,
        deltas: np.ndarray,
        minimum_periods: int,
        old_wt_factor: float,
        new_wt: float,
        old_wt: np.ndarray,
        adjust: bool,
        ignore_na: bool,
    ):
        """
        Compute online exponentially weighted mean per column over 2D values.

        Takes the first observation as is, then computes the subsequent
        exponentially weighted mean accounting minimum periods.
        """
        # 初始化结果数组
        result = np.empty(values.shape)
        # 复制第一个观测值作为初始加权平均值
        weighted_avg = values[0].copy()
        # 记录非 NaN 的观测数量
        nobs = (~np.isnan(weighted_avg)).astype(np.int64)
        # 第一个结果值取决于观测数是否达到最小周期要求
        result[0] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)

        # 遍历 values 中的每一行数据
        for i in range(1, len(values)):
            cur = values[i]
            is_observations = ~np.isnan(cur)
            nobs += is_observations.astype(np.int64)
            # 并行处理每一行的数据
            for j in numba.prange(len(cur)):
                if not np.isnan(weighted_avg[j]):
                    if is_observations[j] or not ignore_na:
                        # 更新旧权重值
                        old_wt[j] *= old_wt_factor ** deltas[j - 1]
                        if is_observations[j]:
                            # 如果是观测到的数据点，更新加权平均值
                            if weighted_avg[j] != cur[j]:
                                weighted_avg[j] = (
                                    (old_wt[j] * weighted_avg[j]) + (new_wt * cur[j])
                                ) / (old_wt[j] + new_wt)
                            # 根据调整标志更新旧权重
                            if adjust:
                                old_wt[j] += new_wt
                            else:
                                old_wt[j] = 1.0
                elif is_observations[j]:
                    # 如果第一次观测到非 NaN 值，则直接更新加权平均值
                    weighted_avg[j] = cur[j]

            # 根据观测数量是否达到最小周期要求更新结果数组
            result[i] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)

        return result, old_wt

    return online_ewma

# 返回生成的在线指数加权移动平均函数


class EWMMeanState:
    # 定义类 EWMMeanState，用于存储在线指数加权移动平均的状态信息
    # 初始化方法，设置指数加权移动平均（EWMA）计算所需的参数
    def __init__(self, com, adjust, ignore_na, shape) -> None:
        # 计算平滑系数 alpha
        alpha = 1.0 / (1.0 + com)
        # 设置对象的形状 shape
        self.shape = shape
        # 设置是否调整参数 adjust
        self.adjust = adjust
        # 设置是否忽略缺失值 ignore_na
        self.ignore_na = ignore_na
        # 根据是否调整参数来确定新权重值
        self.new_wt = 1.0 if adjust else alpha
        # 计算旧权重系数
        self.old_wt_factor = 1.0 - alpha
        # 初始化旧权重数组
        self.old_wt = np.ones(self.shape[-1])
        # 初始化最近一次的指数加权移动平均结果为 None
        self.last_ewm = None

    # 执行指数加权移动平均计算
    def run_ewm(self, weighted_avg, deltas, min_periods, ewm_func):
        # 调用指定的指数加权移动平均函数进行计算
        result, old_wt = ewm_func(
            weighted_avg,
            deltas,
            min_periods,
            self.old_wt_factor,
            self.new_wt,
            self.old_wt,
            self.adjust,
            self.ignore_na,
        )
        # 更新对象的旧权重数组
        self.old_wt = old_wt
        # 记录最近一次的指数加权移动平均结果
        self.last_ewm = result[-1]
        # 返回计算结果
        return result

    # 重置对象的状态
    def reset(self) -> None:
        # 重置旧权重数组为全为1的数组
        self.old_wt = np.ones(self.shape[-1])
        # 将最近一次的指数加权移动平均结果设为 None
        self.last_ewm = None
```