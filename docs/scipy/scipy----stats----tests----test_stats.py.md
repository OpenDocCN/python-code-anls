# `D:\src\scipysrc\scipy\scipy\stats\tests\test_stats.py`

```
""" Test functions for stats module

    WRITTEN BY LOUIS LUANGKESORN <lluang@yahoo.com> FOR THE STATS MODULE
    BASED ON WILKINSON'S STATISTICS QUIZ
    https://www.stanford.edu/~clint/bench/wilk.txt

    Additional tests by a host of SciPy developers.
"""
# 导入必要的库和模块
import os  # 操作系统功能模块
import re  # 正则表达式模块
import warnings  # 警告模块
from collections import namedtuple  # 命名元组模块
from itertools import product  # 迭代工具模块
import hypothesis.extra.numpy as npst  # 假设测试的 NumPy 扩展
import hypothesis  # 假设测试库
import contextlib  # 上下文管理模块

# 导入 NumPy 测试断言
from numpy.testing import (assert_, assert_equal,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_approx_equal,
                           assert_allclose, suppress_warnings,
                           assert_array_less)
import pytest  # 测试框架
from pytest import raises as assert_raises  # 断言引发异常
import numpy.ma.testutils as mat  # NumPy 掩码数组测试工具
from numpy import array, arange, float32, power  # 导入 NumPy 相关函数
import numpy as np  # 导入 NumPy

# 导入 SciPy 统计模块及其子模块和特定函数
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize

# 导入共享测试函数
from .common_tests import check_named_results
# 导入 SciPy 统计模块中处理 NaN 的函数和警告
from scipy.stats._axis_nan_policy import (_broadcast_concatenate, SmallSampleWarning,
                                          too_small_nd_omit, too_small_nd_not_omit,
                                          too_small_1d_omit, too_small_1d_not_omit)
# 导入 SciPy 统计模块中的一些基础计算函数和类
from scipy.stats._stats_py import (_permutation_distribution_t, _chk_asarray, _moment,
                                   LinregressResult, _xp_mean, _xp_var)
# 导入 SciPy 内部工具和异常处理类
from scipy._lib._util import AxisError
# 导入 SciPy 测试配置相关函数
from scipy.conftest import array_api_compatible, skip_xp_invalid_arg
# 导入 SciPy 数组 API 相关函数和工具
from scipy._lib._array_api import (xp_assert_close, xp_assert_equal, array_namespace,
                                   copy, is_numpy, is_torch, SCIPY_ARRAY_API,
                                   size as xp_size, copy as xp_copy)

# 定义一个 pytest 标记，用于跳过某些后端测试
skip_xp_backends = pytest.mark.skip_xp_backends


""" Numbers in docstrings beginning with 'W' refer to the section numbers
    and headings found in the STATISTICS QUIZ of Leland Wilkinson.  These are
    considered to be essential functionality.  True testing and
    evaluation of a statistics package requires use of the
    NIST Statistical test data.  See McCoullough(1999) Assessing The Reliability
    of Statistical Software for a test methodology and its
    implementation in testing SAS, SPSS, and S-Plus
"""
# 文档字符串中以 'W' 开头的数字指向 Leland Wilkinson 的统计测验中的章节和标题，
# 这些内容被认为是核心功能的一部分。为了真正测试和评估一个统计软件包，需要使用 NIST 统计测试数据。
# 参见 McCoullough(1999) 的文章 Assessing The Reliability of Statistical Software，
# 了解一种测试方法及其在测试 SAS、SPSS 和 S-Plus 中的实施方式。

# 数据集定义
# 这些数据集来自 Wilkinson 使用的 nasty.dat 数据集
# 为了完整性，我应该编写相关测试，并将其视为失败的计数
# 尽管这仍然是 beta 软件的接受程度，但这将作为 1.0 版本状态的一个良好目标
X = array([1,2,3,4,5,6,7,8,9], float)  # 定义一个包含浮点数的 NumPy 数组
ZERO = array([0,0,0,0,0,0,0,0,0], float)  # 定义一个全零的 NumPy 数组
BIG = array([99999991,99999992,99999993,99999994,99999995,99999996,99999997,
             99999998,99999999], float)  # 定义一个包含大整数的 NumPy 数组
# 定义小数数组，用于数值计算，存储高精度浮点数
LITTLE = array([0.99999991, 0.99999992, 0.99999993, 0.99999994, 0.99999995, 0.99999996,
                0.99999997, 0.99999998, 0.99999999], float)
# 定义大数数组，用于数值计算，存储较大的浮点数
HUGE = array([1e+12, 2e+12, 3e+12, 4e+12, 5e+12, 6e+12, 7e+12, 8e+12, 9e+12], float)
# 定义极小数数组，用于数值计算，存储极小的浮点数
TINY = array([1e-12, 2e-12, 3e-12, 4e-12, 5e-12, 6e-12, 7e-12, 8e-12, 9e-12], float)
# 定义四舍五入数数组，用于数值计算，存储需要四舍五入的浮点数
ROUND = array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], float)

# 使用修饰器将该类标记为与数组API兼容，同时跳过指定的后端执行环境
@array_api_compatible
@skip_xp_backends('array_api_strict', 'jax.numpy',
                  reasons=["`array_api_strict.where` `fillvalue` doesn't "
                           "accept Python floats. See data-apis/array-api#807.",
                           "JAX doesn't allow item assignment."])
# 标记使用跳过后端执行环境的修饰器，以防止特定情况下的执行错误
@pytest.mark.usefixtures("skip_xp_backends")
# 定义测试类 TestTrimmedStats，用于测试修剪统计量的功能
class TestTrimmedStats:
    # TODO: write these tests to handle missing values properly
    # 设置双精度浮点数的精度，用于处理缺失值
    dprec = np.finfo(np.float64).precision

    # 定义测试方法 test_tmean，用于测试修剪平均值的计算
    def test_tmean(self, xp):
        # 将输入数据转换为特定后端（xp）的数组，默认使用后端的默认数据类型
        x = xp.asarray(X.tolist())

        # 计算修剪平均值，限制范围为 (2, 8)，包括端点，返回结果应为 5.0
        y = stats.tmean(x, (2, 8), (True, True))
        xp_assert_close(y, xp.asarray(5.0))

        # 使用不同的限制和包含性参数计算修剪平均值，并验证结果是否一致
        y1 = stats.tmean(x, limits=(2, 8), inclusive=(False, False))
        y2 = stats.tmean(x, limits=None)
        xp_assert_close(y1, y2)

        # 对于二维数组，计算全局平均值
        x_2d = xp.reshape(xp.arange(63.), (9, 7))
        y = stats.tmean(x_2d, axis=None)
        xp_assert_close(y, xp.mean(x_2d))

        # 按列计算平均值
        y = stats.tmean(x_2d, axis=0)
        xp_assert_close(y, xp.mean(x_2d, axis=0))

        # 按行计算平均值
        y = stats.tmean(x_2d, axis=1)
        xp_assert_close(y, xp.mean(x_2d, axis=1))

        # 在指定范围 (2, 61) 内计算全局平均值
        y = stats.tmean(x_2d, limits=(2, 61), axis=None)
        xp_assert_close(y, xp.asarray(31.5))

        # 在指定范围 (2, 21) 内按列计算平均值
        y = stats.tmean(x_2d, limits=(2, 21), axis=0)
        y_true = [14, 11.5, 9, 10, 11, 12, 13]
        xp_assert_close(y, xp.asarray(y_true))

        # 在指定范围 (2, 21) 内按列计算平均值，包含起始值，不包含结束值
        y = stats.tmean(x_2d, limits=(2, 21), inclusive=(True, False), axis=0)
        y_true = [10.5, 11.5, 9, 10, 11, 12, 13]
        xp_assert_close(y, xp.asarray(y_true))

        # 处理包含 NaN 值的二维数组，计算在指定范围 (1, 13) 内按列计算平均值
        x_2d_with_nan = xp_copy(x_2d)
        x_2d_with_nan[-1, -3:] = xp.nan
        y = stats.tmean(x_2d_with_nan, limits=(1, 13), axis=0)
        y_true = [7, 4.5, 5.5, 6.5, xp.nan, xp.nan, xp.nan]
        xp_assert_close(y, xp.asarray(y_true))

        # 在指定范围 (2, 21) 内按行计算平均值
        y = stats.tmean(x_2d, limits=(2, 21), axis=1)
        y_true = [4, 10, 17, 21, xp.nan, xp.nan, xp.nan, xp.nan, xp.nan]
        xp_assert_close(y, xp.asarray(y_true))

        # 在指定范围 (2, 21) 内按行计算平均值，不包含起始值，包含结束值
        y = stats.tmean(x_2d, limits=(2, 21),
                        inclusive=(False, True), axis=1)
        y_true = [4.5, 10, 17, 21, xp.nan, xp.nan, xp.nan, xp.nan, xp.nan]
        xp_assert_close(y, xp.asarray(y_true))
    # 定义测试函数 `test_tvar`，接受参数 `self` 和 `xp`
    def test_tvar(self, xp):
        # 将 X 转换为列表，并使用 xp 的默认数据类型创建数组 x
        x = xp.asarray(X.tolist())  # use default dtype of xp
        # 为了兼容 array-api，需要创建 xp_test 变量作为 `correction` 的替代
        xp_test = array_namespace(x)  # need array-api-compat var for `correction`

        # 计算 x 的样本方差，限定计算范围为 (2, 8)，计算结果赋给 y
        y = stats.tvar(x, limits=(2, 8), inclusive=(True, True))
        # 断言 y 与预期值 4.6666666666666661 在 xp 的数组表示上相近
        xp_assert_close(y, xp.asarray(4.6666666666666661))

        # 计算 x 的样本方差，不设定限制范围
        y = stats.tvar(x, limits=None)
        # 断言 y 与 xp_test.var(x, correction=1) 的结果在 xp 的数组表示上相近
        xp_assert_close(y, xp_test.var(x, correction=1))

        # 将 x 重塑为二维数组 x_2d
        x_2d = xp.reshape(xp.arange(63.), (9, 7))
        # 计算 x_2d 的样本方差，axis=None 表示对所有元素计算方差
        y = stats.tvar(x_2d, axis=None)
        # 断言 y 与 xp_test.var(x_2d, correction=1) 的结果在 xp 的数组表示上相近
        xp_assert_close(y, xp_test.var(x_2d, correction=1))

        # 计算 x_2d 沿 axis=0 方向的样本方差
        y = stats.tvar(x_2d, axis=0)
        # 断言 y 与 shape 为 (7,) 且值全为 367.5 的数组在 xp 的数组表示上相等
        xp_assert_close(y, xp.full((7,), 367.5))

        # 计算 x_2d 沿 axis=1 方向的样本方差
        y = stats.tvar(x_2d, axis=1)
        # 断言 y 与 shape 为 (9,) 且值全为 4.66666667 的数组在 xp 的数组表示上相等
        xp_assert_close(y, xp.full((9,), 4.66666667))

        # 在 axis=1 方向上限制部分值的计算
        y = stats.tvar(x_2d, limits=(1, 5), axis=1, inclusive=(True, True))
        # 断言 y[0] 与值为 2.5 的数组在 xp 的数组表示上相等
        xp_assert_close(y[0], xp.asarray(2.5))

        # 在 axis=1 方向上限制所有值的计算
        y = stats.tvar(x_2d, limits=(0, 6), axis=1, inclusive=(True, True))
        # 断言 y[0] 与值为 4.666666666666667 的数组在 xp 的数组表示上相等
        xp_assert_close(y[0], xp.asarray(4.666666666666667))
        # 断言 y[1] 与 xp.nan 在 xp 的数组表示上相等
        xp_assert_equal(y[1], xp.asarray(xp.nan))

    # 定义测试函数 `test_tstd`，接受参数 `self` 和 `xp`
    def test_tstd(self, xp):
        # 将 X 转换为列表，并使用 xp 的默认数据类型创建数组 x
        x = xp.asarray(X.tolist())  # use default dtype of xp
        # 为了兼容 array-api，需要创建 xp_test 变量作为 `correction` 的替代
        xp_test = array_namespace(x)  # need array-api-compat std for `correction`

        # 计算 x 的样本标准差，limits=(2, 8)，inclusive=(True, True)
        y = stats.tstd(x, (2, 8), (True, True))
        # 断言 y 与预期值 2.1602468994692865 在 xp 的数组表示上相近
        xp_assert_close(y, xp.asarray(2.1602468994692865))

        # 计算 x 的样本标准差，不设定限制范围
        y = stats.tstd(x, limits=None)
        # 断言 y 与 xp_test.std(x, correction=1) 的结果在 xp 的数组表示上相近
        xp_assert_close(y, xp_test.std(x, correction=1))

    # 定义测试函数 `test_tmin`，接受参数 `self` 和 `xp`
    def test_tmin(self, xp):
        # 创建一个长度为 10 的 xp 数组 x
        x = xp.arange(10)
        # 断言计算 x 的最小值为 0，在 xp 的数组表示上相等
        xp_assert_equal(stats.tmin(x), xp.asarray(0))
        # 断言计算 x 的最小值为 0，限制 lowerlimit=0，在 xp 的数组表示上相等
        xp_assert_equal(stats.tmin(x, lowerlimit=0), xp.asarray(0))
        # 断言计算 x 的最小值为 1，限制 lowerlimit=0，inclusive=False，在 xp 的数组表示上相等
        xp_assert_equal(stats.tmin(x, lowerlimit=0, inclusive=False), xp.asarray(1))

        # 将 x 重塑为形状为 (5, 2) 的二维数组
        x = xp.reshape(x, (5, 2))
        # 断言计算 x 沿 axis=1 方向的最小值，限制 lowerlimit=0，inclusive=False，在 xp 的数组表示上相等
        xp_assert_equal(stats.tmin(x, lowerlimit=0, inclusive=False),
                        xp.asarray([2, 1]))
        # 断言计算 x 沿 axis=1 方向的最小值，在 xp 的数组表示上相等
        xp_assert_equal(stats.tmin(x, axis=1), xp.asarray([0, 2, 4, 6, 8]))
        # 断言计算 x 所有元素的最小值，在 xp 的数组表示上相等
        xp_assert_equal(stats.tmin(x, axis=None), xp.asarray(0))

        # 创建一个长度为 10 的浮点数 xp 数组 x，并将最后一个元素设为 xp.nan
        x = xp.arange(10.)
        x[9] = xp.nan
        # 断言计算 x 的最小值，结果为 xp.nan，在 xp 的数组表示上相等
        xp_assert_equal(stats.tmin(x), xp.asarray(xp.nan))

        # 检查如果一个全切片被掩码，输出应返回 nan 而不是垃圾值
        x = xp.arange(16).reshape(4, 4)
        # 计算 x 沿 axis=1 方向的最小值，lowerlimit=4，在 xp 的数组表示上相等
        res = stats.tmin(x, lowerlimit=4, axis=1)
        xp_assert_equal(res, xp.asarray([np.nan, 4, 8, 12]))

    # 跳过只支持 NumPy 数组的后端，原因是 scalar input/`nan_policy` 仅在 NumPy 中支持
    @skip_xp_backends(np_only=True,
                      reasons=["Only NumPy arrays support scalar input/`nan_policy`."])
    # 测试 stats.tmin 函数对标量输入和 NaN 策略的处理
    def test_tmin_scalar_and_nanpolicy(self, xp):
        # 断言 stats.tmin(4) 返回 4
        assert_equal(stats.tmin(4), 4)

        # 创建一个包含 NaN 的 NumPy 数组
        x = np.arange(10.)
        x[9] = np.nan

        # 使用 suppress_warnings 上下文管理器捕获运行时警告
        with suppress_warnings() as sup:
            # 记录 RuntimeWarning 类型的 "invalid value*" 警告
            sup.record(RuntimeWarning, "invalid value*")
            # 断言使用 'omit' NaN 策略时 stats.tmin 返回 0.0
            assert_equal(stats.tmin(x, nan_policy='omit'), 0.)
            # 断言使用 'raise' NaN 策略时会引发 ValueError 异常，匹配给定的错误消息
            msg = "The input contains nan values"
            with assert_raises(ValueError, match=msg):
                stats.tmin(x, nan_policy='raise')
            # 断言使用无效的 'foobar' NaN 策略时会引发 ValueError 异常，匹配给定的错误消息
            msg = "nan_policy must be one of..."
            with assert_raises(ValueError, match=msg):
                stats.tmin(x, nan_policy='foobar')

    # 测试 stats.tmax 函数的多个参数组合
    def test_tmax(self, xp):
        # 创建一个 NumPy 数组 x 包含值 0 到 9
        x = xp.arange(10)
        # 断言 stats.tmax(x) 返回数组的最大值 9
        xp_assert_equal(stats.tmax(x), xp.asarray(9))
        # 断言 stats.tmax(x, upperlimit=9) 返回数组的最大值 9
        xp_assert_equal(stats.tmax(x, upperlimit=9), xp.asarray(9))
        # 断言 stats.tmax(x, upperlimit=9, inclusive=False) 返回数组的最大值 8
        xp_assert_equal(stats.tmax(x, upperlimit=9, inclusive=False), xp.asarray(8))

        # 将数组 x 重塑为形状为 (5, 2) 的数组
        x = xp.reshape(x, (5, 2))
        # 断言 stats.tmax(x, upperlimit=9, inclusive=False) 返回数组 [8, 7]
        xp_assert_equal(stats.tmax(x, upperlimit=9, inclusive=False),
                        xp.asarray([8, 7]))
        # 断言 stats.tmax(x, axis=1) 返回沿轴 1 的最大值数组 [1, 3, 5, 7, 9]
        xp_assert_equal(stats.tmax(x, axis=1), xp.asarray([1, 3, 5, 7, 9]))
        # 断言 stats.tmax(x, axis=None) 返回整个数组的最大值 9
        xp_assert_equal(stats.tmax(x, axis=None), xp.asarray(9))

        # 创建一个包含 NaN 的 NumPy 数组
        x = xp.arange(10.)
        x[9] = xp.nan
        # 断言 stats.tmax(x) 返回 NaN
        xp_assert_equal(stats.tmax(x), xp.asarray(xp.nan))

        # 使用 suppress_warnings 上下文管理器捕获运行时警告
        with suppress_warnings() as sup:
            # 过滤 RuntimeWarning 类型的 "All-NaN slice encountered" 警告
            sup.filter(RuntimeWarning, "All-NaN slice encountered")
            # 将数组 x 重塑为形状为 (4, 4) 的数组，沿轴 1 计算最大值
            x = xp.reshape(xp.arange(16), (4, 4))
            # 断言 stats.tmax(x, upperlimit=11, axis=1) 返回沿轴 1 的最大值数组 [3, 7, 11, np.nan]
            res = stats.tmax(x, upperlimit=11, axis=1)
            xp_assert_equal(res, xp.asarray([3, 7, 11, np.nan]))

    # 使用 skip_xp_backends 装饰器标记的测试，仅适用于 NumPy 数组，因为只有它们支持标量输入和 `nan_policy`
    @skip_xp_backends(np_only=True,
                      reasons=["Only NumPy arrays support scalar input/`nan_policy`."])
    def test_tax_scalar_and_nanpolicy(self, xp):
        # 断言 stats.tmax(4) 返回 4
        assert_equal(stats.tmax(4), 4)

        # 创建一个包含 NaN 的 NumPy 数组
        x = np.arange(10.)
        x[6] = np.nan

        # 使用 suppress_warnings 上下文管理器捕获运行时警告
        with suppress_warnings() as sup:
            # 记录 RuntimeWarning 类型的 "invalid value*" 警告
            sup.record(RuntimeWarning, "invalid value*")
            # 断言使用 'omit' NaN 策略时 stats.tmax 返回 9.0
            assert_equal(stats.tmax(x, nan_policy='omit'), 9.)
            # 断言使用 'raise' NaN 策略时会引发 ValueError 异常，匹配给定的错误消息
            msg = "The input contains nan values"
            with assert_raises(ValueError, match=msg):
                stats.tmax(x, nan_policy='raise')
            # 断言使用无效的 'foobar' NaN 策略时会引发 ValueError 异常，匹配给定的错误消息
            msg = "nan_policy must be one of..."
            with assert_raises(ValueError, match=msg):
                stats.tmax(x, nan_policy='foobar')

    # 测试 stats.tsem 函数的用法
    def test_tsem(self, xp):
        # 将 X 转换为数组，使用默认的 dtype
        x = xp.asarray(X.tolist())  # use default dtype of xp
        # 使用 array_namespace 处理 x，确保与 array-api-compat 标准兼容，需要 'correction' 参数
        xp_test = array_namespace(x)  # need array-api-compat std for `correction`

        # 调用 stats.tsem 函数，限制在 (3, 8) 之间，inclusive=(False, True)
        y = stats.tsem(x, limits=(3, 8), inclusive=(False, True))
        # 参考结果 y_ref，应为 [4., 5., 6., 7., 8.]
        y_ref = xp.asarray([4., 5., 6., 7., 8.])
        # 断言 stats.tsem 的计算结果与参考结果 y_ref 的标准误差相近
        xp_assert_close(y, xp_test.std(y_ref, correction=1) / xp_size(y_ref)**0.5)
        # 断言 stats.tsem(x, limits=[-1, 10]) 等同于无限制的调用
        xp_assert_close(stats.tsem(x, limits=[-1, 10]), stats.tsem(x, limits=None))
class TestPearsonrWilkinson:
    """ W.II.D. Compute a correlation matrix on all the variables.

        All the correlations, except for ZERO and MISS, should be exactly 1.
        ZERO and MISS should have undefined or missing correlations with the
        other variables.  The same should go for SPEARMAN correlations, if
        your program has them.
    """

    # 定义测试类 TestPearsonrWilkinson，用于计算所有变量之间的相关系数矩阵
    def test_pXX(self):
        # 计算变量 X 与自身的皮尔逊相关系数
        y = stats.pearsonr(X,X)
        # 获取相关系数的第一个值
        r = y[0]
        # 断言相关系数近似为 1.0
        assert_approx_equal(r,1.0)

    def test_pXBIG(self):
        # 计算变量 X 与 BIG 的皮尔逊相关系数
        y = stats.pearsonr(X,BIG)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pXLITTLE(self):
        # 计算变量 X 与 LITTLE 的皮尔逊相关系数
        y = stats.pearsonr(X,LITTLE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pXHUGE(self):
        # 计算变量 X 与 HUGE 的皮尔逊相关系数
        y = stats.pearsonr(X,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pXTINY(self):
        # 计算变量 X 与 TINY 的皮尔逊相关系数
        y = stats.pearsonr(X,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pXROUND(self):
        # 计算变量 X 与 ROUND 的皮尔逊相关系数
        y = stats.pearsonr(X,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGBIG(self):
        # 计算变量 BIG 与自身的皮尔逊相关系数
        y = stats.pearsonr(BIG,BIG)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGLITTLE(self):
        # 计算变量 BIG 与 LITTLE 的皮尔逊相关系数
        y = stats.pearsonr(BIG,LITTLE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGHUGE(self):
        # 计算变量 BIG 与 HUGE 的皮尔逊相关系数
        y = stats.pearsonr(BIG,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGTINY(self):
        # 计算变量 BIG 与 TINY 的皮尔逊相关系数
        y = stats.pearsonr(BIG,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pBIGROUND(self):
        # 计算变量 BIG 与 ROUND 的皮尔逊相关系数
        y = stats.pearsonr(BIG,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pLITTLELITTLE(self):
        # 计算变量 LITTLE 与自身的皮尔逊相关系数
        y = stats.pearsonr(LITTLE,LITTLE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pLITTLEHUGE(self):
        # 计算变量 LITTLE 与 HUGE 的皮尔逊相关系数
        y = stats.pearsonr(LITTLE,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pLITTLETINY(self):
        # 计算变量 LITTLE 与 TINY 的皮尔逊相关系数
        y = stats.pearsonr(LITTLE,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pLITTLEROUND(self):
        # 计算变量 LITTLE 与 ROUND 的皮尔逊相关系数
        y = stats.pearsonr(LITTLE,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pHUGEHUGE(self):
        # 计算变量 HUGE 与自身的皮尔逊相关系数
        y = stats.pearsonr(HUGE,HUGE)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pHUGETINY(self):
        # 计算变量 HUGE 与 TINY 的皮尔逊相关系数
        y = stats.pearsonr(HUGE,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pHUGEROUND(self):
        # 计算变量 HUGE 与 ROUND 的皮尔逊相关系数
        y = stats.pearsonr(HUGE,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pTINYTINY(self):
        # 计算变量 TINY 与自身的皮尔逊相关系数
        y = stats.pearsonr(TINY,TINY)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pTINYROUND(self):
        # 计算变量 TINY 与 ROUND 的皮尔逊相关系数
        y = stats.pearsonr(TINY,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)

    def test_pROUNDROUND(self):
        # 计算变量 ROUND 与自身的皮尔逊相关系数
        y = stats.pearsonr(ROUND,ROUND)
        r = y[0]
        assert_approx_equal(r,1.0)


@array_api_compatible
# 使用装饰器指定兼容数组 API
@pytest.mark.usefixtures("skip_xp_backends")
# 使用 pytest 的装饰器标记，跳过 XP 后端测试
@skip_xp_backends(cpu_only=True)
# 使用自定义装饰器跳过仅限 CPU 的 XP 后端测试
# 定义一个测试类 TestPearsonr，用于测试 Pearson 相关系数的计算
class TestPearsonr:
    
    # 用装饰器 @skip_xp_backends 标记，指定仅对 NumPy 运行测试
    @skip_xp_backends(np_only=True)
    # 测试 Pearson 相关系数结果的属性
    def test_pearsonr_result_attributes(self):
        # 计算输入数组 X 自身的 Pearson 相关系数
        res = stats.pearsonr(X, X)
        # 指定需要检查的结果属性
        attributes = ('correlation', 'pvalue')
        # 调用函数检查结果是否具有指定的命名属性
        check_named_results(res, attributes)
        # 断言结果的相关系数与统计量相等
        assert_equal(res.correlation, res.statistic)

    # 用装饰器 @skip_xp_backends 标记，指定在特定条件下不运行 JAX 的测试
    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    # 测试特定情况下 Pearson 相关系数几乎等于正1的情况
    def test_r_almost_exactly_pos1(self, xp):
        # 创建一个数组 a，其值为 [0.0, 1.0, 2.0]
        a = xp.arange(3.0)
        # 计算数组 a 与自身的 Pearson 相关系数及概率值
        r, prob = stats.pearsonr(a, a)

        # 断言计算得到的相关系数几乎等于 1.0
        xp_assert_close(r, xp.asarray(1.0), atol=1e-15)
        # 根据误差公式，断言计算得到的概率值几乎等于 0.0
        xp_assert_close(prob, xp.asarray(0.0), atol=np.sqrt(2*np.spacing(1.0)))

    # 用装饰器 @skip_xp_backends 标记，指定在特定条件下不运行 JAX 的测试
    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    # 测试特定情况下 Pearson 相关系数几乎等于负1的情况
    def test_r_almost_exactly_neg1(self, xp):
        # 创建一个数组 a，其值为 [0.0, -1.0, -2.0]
        a = xp.arange(3.0)
        # 计算数组 a 与其相反数的 Pearson 相关系数及概率值
        r, prob = stats.pearsonr(a, -a)

        # 断言计算得到的相关系数几乎等于 -1.0
        xp_assert_close(r, xp.asarray(-1.0), atol=1e-15)
        # 根据误差公式，断言计算得到的概率值几乎等于 0.0
        xp_assert_close(prob, xp.asarray(0.0), atol=np.sqrt(2*np.spacing(1.0)))

    # 用装饰器 @skip_xp_backends 标记，指定在特定条件下不运行 JAX 的测试
    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    # 执行基本测试，测试不是1或-1的相关系数情况
    def test_basic(self, xp):
        # 创建数组 a，其值为 [-1, 0, 1]
        a = xp.asarray([-1, 0, 1])
        # 创建数组 b，其值为 [0, 0, 3]
        b = xp.asarray([0, 0, 3])
        # 计算数组 a 与 b 的 Pearson 相关系数及概率值
        r, prob = stats.pearsonr(a, b)
        # 断言计算得到的相关系数几乎等于 sqrt(3)/2
        xp_assert_close(r, xp.asarray(3**0.5/2))
        # 断言计算得到的概率值几乎等于 1/3
        xp_assert_close(prob, xp.asarray(1/3))

    # 用装饰器 @skip_xp_backends 标记，指定在特定条件下不运行 JAX 的测试
    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    # 测试常量输入的情况
    def test_constant_input(self, xp):
        # 输入数组 x 是常量的情况，即所有元素都相同
        msg = "An input array is constant"
        # 使用 pytest 的 warns 函数来检查是否会发出警告
        with pytest.warns(stats.ConstantInputWarning, match=msg):
            # 创建数组 x，其值为 [0.667, 0.667, 0.667]
            x = xp.asarray([0.667, 0.667, 0.667])
            # 创建数组 y，其值为 [0.123, 0.456, 0.789]
            y = xp.asarray([0.123, 0.456, 0.789])
            # 计算数组 x 与 y 的 Pearson 相关系数及概率值
            r, p = stats.pearsonr(x, y)
            # 断言计算得到的相关系数为 NaN
            xp_assert_close(r, xp.asarray(xp.nan))
            # 断言计算得到的概率值为 NaN
            xp_assert_close(p, xp.asarray(xp.nan))

    # 用装饰器 @skip_xp_backends 标记，指定在特定条件下不运行 JAX 的测试
    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    # 对参数 dtype 进行参数化测试，支持 'float32' 和 'float64'
    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    def test_near_constant_input(self, xp, dtype):
        npdtype = getattr(np, dtype)
        dtype = getattr(xp, dtype)
        # Near constant input (but not constant):
        # 创建包含近似常数输入数据的数组 x 和 y，其中 x 中的最后一个元素略有偏差
        x = xp.asarray([2, 2, 2 + np.spacing(2, dtype=npdtype)], dtype=dtype)
        y = xp.asarray([3, 3, 3 + 6*np.spacing(3, dtype=npdtype)], dtype=dtype)
        msg = "An input array is nearly constant; the computed"
        # 断言会触发警告，警告消息包含在 msg 变量中
        with pytest.warns(stats.NearConstantInputWarning, match=msg):
            # 调用 stats.pearsonr 计算 x 和 y 的 Pearson 相关系数
            stats.pearsonr(x, y)

    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    def test_very_small_input_values(self, xp):
        # Very small values in an input.  A naive implementation will
        # suffer from underflow.
        # 创建包含非常小输入值的数组 x 和 y，使用 xp.float64 类型
        x = xp.asarray([0.004434375, 0.004756007, 0.003911996, 0.0038005, 0.003409971],
                       dtype=xp.float64)
        y = xp.asarray([2.48e-188, 7.41e-181, 4.09e-208, 2.08e-223, 2.66e-245],
                       dtype=xp.float64)
        # 计算 x 和 y 的 Pearson 相关系数 r 和 p
        r, p = stats.pearsonr(x, y)

        # The expected values were computed using mpmath with 80 digits
        # of precision.
        # 使用高精度计算结果作为预期值来验证计算得到的 r 和 p
        xp_assert_close(r, xp.asarray(0.7272930540750450, dtype=xp.float64))
        xp_assert_close(p, xp.asarray(0.1637805429533202, dtype=xp.float64))

    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    def test_very_large_input_values(self, xp):
        # Very large values in an input.  A naive implementation will
        # suffer from overflow.
        # 创建包含非常大输入值的数组 x 和 y，使用 xp.float64 类型
        x = 1e90*xp.asarray([0, 0, 0, 1, 1, 1, 1], dtype=xp.float64)
        y = 1e90*xp.arange(7, dtype=xp.float64)

        # 计算 x 和 y 的 Pearson 相关系数 r 和 p
        r, p = stats.pearsonr(x, y)

        # The expected values were computed using mpmath with 80 digits
        # of precision.
        # 使用高精度计算结果作为预期值来验证计算得到的 r 和 p
        xp_assert_close(r, xp.asarray(0.8660254037844386, dtype=xp.float64))
        xp_assert_close(p, xp.asarray(0.011724811003954638, dtype=xp.float64))

    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    def test_extremely_large_input_values(self, xp):
        # 测试极端大的输入值 x 和 y。
        # 如果这两个因子分别计算，它们的乘积 sigma_x * sigma_y 会溢出。
        x = xp.asarray([2.3e200, 4.5e200, 6.7e200, 8e200], dtype=xp.float64)
        y = xp.asarray([1.2e199, 5.5e200, 3.3e201, 1.0e200], dtype=xp.float64)
        # 计算 Pearson 相关系数 r 和 p 值
        r, p = stats.pearsonr(x, y)

        # 期望值使用 mpmath 计算，精确到 80 位数字
        xp_assert_close(r, xp.asarray(0.351312332103289, dtype=xp.float64))
        xp_assert_close(p, xp.asarray(0.648687667896711, dtype=xp.float64))

    def test_length_two_pos1(self, xp):
        # 长度为 2 的输入。
        # 参考 https://github.com/scipy/scipy/issues/7730
        x = xp.asarray([1., 2.])
        y = xp.asarray([3., 5.])
        # 计算 Pearson 相关系数
        res = stats.pearsonr(x, y)
        r, p = res
        one = xp.asarray(1.)
        xp_assert_equal(r, one)
        xp_assert_equal(p, one)
        # 计算置信区间
        low, high = res.confidence_interval()
        xp_assert_equal(low, -one)
        xp_assert_equal(high, one)

    def test_length_two_neg1(self, xp):
        # 长度为 2 的输入。
        # 参考 https://github.com/scipy/scipy/issues/7730
        x = xp.asarray([2., 1.])
        y = xp.asarray([3., 5.])
        # 计算 Pearson 相关系数
        res = stats.pearsonr(x, y)
        r, p = res
        one = xp.asarray(1.)
        xp_assert_equal(r, -one)
        xp_assert_equal(p, one)
        # 计算置信区间
        low, high = res.confidence_interval()
        xp_assert_equal(low, -one)
        xp_assert_equal(high, one)

    # 使用 R 3.6.2 cor.test 计算的期望值，例如
    # options(digits=16)
    # x <- c(1, 2, 3, 4)
    # y <- c(0, 1, 0.5, 1)
    # cor.test(x, y, method = "pearson", alternative = "g")
    # 在 mpmath 精确到 16 位小数的情况下，计算得到的相关系数和 p 值是一致的。
    @pytest.mark.skip_xp_backends(np_only=True)
    @pytest.mark.parametrize('alternative, pval, rlow, rhigh, sign',
            [('two-sided', 0.325800137536, -0.814938968841, 0.99230697523, 1),
             ('less', 0.8370999312316, -1, 0.985600937290653, 1),
             ('greater', 0.1629000687684, -0.6785654158217636, 1, 1),
             ('two-sided', 0.325800137536, -0.992306975236, 0.81493896884, -1),
             ('less', 0.1629000687684, -1.0, 0.6785654158217636, -1),
             ('greater', 0.8370999312316, -0.985600937290653, 1.0, -1)])
    def test_basic_example(self, alternative, pval, rlow, rhigh, sign):
        # 基本示例测试
        x = [1, 2, 3, 4]
        y = np.array([0, 1, 0.5, 1]) * sign
        # 计算 Pearson 相关系数，指定 alternative 参数
        result = stats.pearsonr(x, y, alternative=alternative)
        # 断言结果的 statistic 和 pvalue
        assert_allclose(result.statistic, 0.6741998624632421*sign, rtol=1e-12)
        assert_allclose(result.pvalue, pval, rtol=1e-6)
        # 获取置信区间
        ci = result.confidence_interval()
        # 断言置信区间的范围
        assert_allclose(ci, (rlow, rhigh), rtol=1e-6)
    # 使用装饰器跳过 JAX 的 numpy 后端，因为它不支持项目赋值
    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    # 测试负相关性的 p 值，针对给定的 xp（不同的 numpy 后端）
    def test_negative_correlation_pvalue_gh17795(self, xp):
        # 创建 numpy 数组 x 和 y，x = [0., 1., ..., 9.], y = [-0., -1., ..., -9.]
        x = xp.arange(10.)
        y = -x
        # 计算 Pearson 相关系数的 p 值，使用 alternative='greater' 进行测试
        test_greater = stats.pearsonr(x, y, alternative='greater')
        # 计算 Pearson 相关系数的 p 值，使用 alternative='less' 进行测试
        test_less = stats.pearsonr(x, y, alternative='less')
        # 断言 test_greater 的 p 值接近于 1.0，使用 xp.asarray 将数值转换为相应后端的数组
        xp_assert_close(test_greater.pvalue, xp.asarray(1.))
        # 断言 test_less 的 p 值接近于 0.0，设置容差为 1e-20
        xp_assert_close(test_less.pvalue, xp.asarray(0.), atol=1e-20)

    # 使用装饰器跳过 JAX 的 numpy 后端，因为它不支持项目赋值
    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    # 测试长度为 3 的 numpy 数组 x 和 y 的相关性
    def test_length3_r_exactly_negative_one(self, xp):
        # 创建 numpy 数组 x = [1., 2., 3.] 和 y = [5., -4., -13.]
        x = xp.asarray([1., 2., 3.])
        y = xp.asarray([5., -4., -13.])
        # 计算 Pearson 相关系数的 r 和 p 值
        res = stats.pearsonr(x, y)
        
        # 断言 r 值接近于 -1.0，设置容差为 1e-7
        r, p = res
        one = xp.asarray(1.0)
        xp_assert_close(r, -one)
        # 断言 p 值接近于 0.0，设置容差为 1e-7
        xp_assert_close(p, 0*one, atol=1e-7)
        
        # 计算相关系数的置信区间，断言置信区间的上下界分别为 -1.0 和 1.0
        low, high = res.confidence_interval()
        xp_assert_equal(low, -one)
        xp_assert_equal(high, one)

    # 使用 pytest 的装饰器，标记跳过所有的 numpy 后端测试
    @pytest.mark.skip_xp_backends(np_only=True)
    # 测试输入的验证情况
    def test_input_validation(self):
        # 测试当 x 和 y 的长度不相等时，引发 ValueError，匹配指定的错误信息
        x = [1, 2, 3]
        y = [4, 5]
        message = '`x` and `y` must have the same length along `axis`.'
        with pytest.raises(ValueError, match=message):
            stats.pearsonr(x, y)

        # 测试当 x 和 y 的长度小于 2 时，引发 ValueError，匹配指定的错误信息
        x = [1]
        y = [2]
        message = '`x` and `y` must have length at least 2.'
        with pytest.raises(ValueError, match=message):
            stats.pearsonr(x, y)

        # 测试当 x 和 y 包含复数时，引发 ValueError，匹配指定的错误信息
        x = [-1j, -2j, -3.0j]
        y = [-1j, -2j, -3.0j]
        message = 'This function does not support complex data'
        with pytest.raises(ValueError, match=message):
            stats.pearsonr(x, y)

        # 测试当 method 参数不正确时，引发 ValueError，匹配指定的错误信息
        message = "`method` must be an instance of..."
        with pytest.raises(ValueError, match=message):
            stats.pearsonr([1, 2], [3, 4], method="asymptotic")

        # 对已计算出的相关系数结果 res 进行更多的 method 参数测试，预期也会引发 ValueError
        res = stats.pearsonr([1, 2], [3, 4])
        with pytest.raises(ValueError, match=message):
            res.confidence_interval(method="exact")

    # 使用 pytest 的装饰器，标记当失败缓慢时继续执行 5 次
    @pytest.mark.fail_slow(5)
    # 使用 pytest 的装饰器，标记跳过所有的 numpy 后端测试
    @pytest.mark.skip_xp_backends(np_only=True)
    # 标记在 32 位系统上跳过，提供特定的失败信息
    @pytest.mark.xfail_on_32bit("Monte Carlo method needs > a few kB of memory")
    # 参数化测试，测试不同的 alternative 和 method 参数组合
    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    @pytest.mark.parametrize('method', ('permutation', 'monte_carlo'))
    # 定义一个测试方法，用于检验重新采样的 p-value 计算
    def test_resampling_pvalue(self, method, alternative):
        # 使用指定的随机种子创建一个 RNG 实例
        rng = np.random.default_rng(24623935790378923)
        # 根据不同方法设置不同的数据大小
        size = (2, 100) if method == 'permutation' else (2, 1000)
        # 从正态分布中生成数据 x 和 y
        x = rng.normal(size=size)
        y = rng.normal(size=size)
        # 根据方法名称选择合适的统计方法对象
        methods = {'permutation': stats.PermutationMethod(random_state=rng),
                   'monte_carlo': stats.MonteCarloMethod(rvs=(rng.normal,)*2)}
        method = methods[method]
        # 计算 Pearson 相关系数和 p-value，使用指定的方法和备择假设
        res = stats.pearsonr(x, y, alternative=alternative, method=method, axis=-1)
        # 计算参考的 Pearson 相关系数和 p-value，使用默认方法和备择假设
        ref = stats.pearsonr(x, y, alternative=alternative, axis=-1)
        # 检验计算结果的统计量是否在允许的相对误差范围内
        assert_allclose(res.statistic, ref.statistic, rtol=1e-15)
        # 检验计算结果的 p-value 是否在允许的相对误差和绝对误差范围内
        assert_allclose(res.pvalue, ref.pvalue, rtol=1e-2, atol=1e-3)

    # 使用 pytest 标记跳过某些 numpy 后端的测试，并对备择假设参数化测试
    @pytest.mark.skip_xp_backends(np_only=True)
    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    # 定义一个测试方法，用于检验 bootstrap 方法计算的置信区间
    def test_bootstrap_ci(self, alternative):
        # 使用指定的随机种子创建一个 RNG 实例
        rng = np.random.default_rng(2462935790378923)
        # 从正态分布中生成数据 x 和 y
        x = rng.normal(size=(2, 100))
        y = rng.normal(size=(2, 100))
        # 计算 Pearson 相关系数和 p-value，使用指定的备择假设和轴
        res = stats.pearsonr(x, y, alternative=alternative, axis=-1)

        # 使用 bootstrap 方法计算相关系数的置信区间
        method = stats.BootstrapMethod(random_state=rng)
        res_ci = res.confidence_interval(method=method)
        # 计算参考的相关系数的置信区间，使用默认方法
        ref_ci = res.confidence_interval()

        # 检验计算结果的置信区间是否在允许的绝对误差范围内
        assert_allclose(res_ci, ref_ci, atol=1.5e-2)

    # 使用 pytest 标记跳过某些 numpy 后端的测试，并对轴参数化测试
    @pytest.mark.skip_xp_backends(np_only=True)
    @pytest.mark.parametrize('axis', [0, 1])
    # 定义一个测试方法，用于检验不同轴向的 Pearson 相关系数计算
    def test_axis01(self, axis, xp):
        # 使用指定的随机种子创建一个 RNG 实例
        rng = np.random.default_rng(38572345825)
        shape = (9, 10)
        # 从正态分布中生成数据 x 和 y，根据轴的不同选择进行转置
        x, y = rng.normal(size=(2,) + shape)
        if axis == 0:
            x, y = x.T, y.T
        # 计算 Pearson 相关系数和 p-value，使用指定的轴
        res = stats.pearsonr(x, y, axis=axis)
        # 计算结果的置信区间
        ci = res.confidence_interval()
        for i in range(x.shape[0]):
            # 对每个轴向数据计算 Pearson 相关系数和 p-value
            res_i = stats.pearsonr(x[i], y[i])
            # 计算每个轴向结果的置信区间
            ci_i = res_i.confidence_interval()
            # 检验计算结果的统计量是否在允许的相对误差范围内
            assert_allclose(res.statistic[i], res_i.statistic)
            # 检验计算结果的 p-value 是否在允许的相对误差和绝对误差范围内
            assert_allclose(res.pvalue[i], res_i.pvalue)
            # 检验置信区间的下限是否在允许的绝对误差范围内
            assert_allclose(ci.low[i], ci_i.low)
            # 检验置信区间的上限是否在允许的绝对误差范围内
            assert_allclose(ci.high[i], ci_i.high)

    # 使用 pytest 标记跳过某些 numpy 后端的测试
    @pytest.mark.skip_xp_backends(np_only=True)
    # 定义一个测试方法，用于检验 axis=None 时的 Pearson 相关系数计算
    def test_axis_None(self, xp):
        # 使用指定的随机种子创建一个 RNG 实例
        rng = np.random.default_rng(38572345825)
        shape = (9, 10)
        # 从正态分布中生成数据 x 和 y
        x, y = rng.normal(size=(2,) + shape)
        # 计算 axis=None 时的 Pearson 相关系数和 p-value
        res = stats.pearsonr(x, y, axis=None)
        # 计算参考的 Pearson 相关系数和 p-value，对数据进行扁平化处理
        ref = stats.pearsonr(x.ravel(), y.ravel())
        # 计算结果的置信区间
        ci = res.confidence_interval()
        ci_ref = ref.confidence_interval()
        # 检验计算结果的统计量是否在允许的相对误差范围内
        assert_allclose(res.statistic, ref.statistic)
        # 检验计算结果的 p-value 是否在允许的相对误差和绝对误差范围内
        assert_allclose(res.pvalue, ref.pvalue)
        # 检验置信区间是否在允许的绝对误差范围内
        assert_allclose(ci, ci_ref)
    # 测试非法输入验证函数，验证 `stats.pearsonr` 的异常情况
    def test_nd_input_validation(self, xp):
        # 创建两个大小为 (2, 5) 的全1数组 x 和 y
        x = y = xp.ones((2, 5))
        # 设置错误消息，用于匹配 ValueError 异常
        message = '`axis` must be an integer.'
        # 验证在调用 `stats.pearsonr` 时是否会引发 ValueError 异常，且异常信息匹配预期消息
        with pytest.raises(ValueError, match=message):
            stats.pearsonr(x, y, axis=1.5)

        # 设置错误消息，用于匹配 ValueError 异常
        message = '`x` and `y` must have the same length along `axis`'
        # 验证在调用 `stats.pearsonr` 时是否会引发 ValueError 异常，且异常信息匹配预期消息
        with pytest.raises(ValueError, match=message):
            stats.pearsonr(x, xp.ones((2, 1)), axis=1)

        # 设置错误消息，用于匹配 ValueError 异常
        message = '`x` and `y` must have length at least 2.'
        # 验证在调用 `stats.pearsonr` 时是否会引发 ValueError 异常，且异常信息匹配预期消息
        with pytest.raises(ValueError, match=message):
            stats.pearsonr(xp.ones((2, 1)), xp.ones((2, 1)), axis=1)

        # 设置错误消息，用于匹配 ValueError 异常
        message = '`x` and `y` must be broadcastable.'
        # 验证在调用 `stats.pearsonr` 时是否会引发 ValueError 异常，且异常信息匹配预期消息
        with pytest.raises(ValueError, match=message):
            stats.pearsonr(x, xp.ones((3, 5)), axis=1)

        # 设置错误消息，用于匹配 ValueError 异常
        message = '`method` must be `None` if arguments are not NumPy arrays.'
        # 如果当前不是 NumPy 环境，则执行以下代码块
        if not is_numpy(xp):
            # 创建一个长度为 10 的数组 x
            x = xp.arange(10)
            # 验证在调用 `stats.pearsonr` 时是否会引发 ValueError 异常，且异常信息匹配预期消息
            with pytest.raises(ValueError, match=message):
                # 在使用自定义的 `stats.PermutationMethod()` 作为 method 参数调用 `stats.pearsonr` 时会引发异常
                stats.pearsonr(x, x, method=stats.PermutationMethod())

    # 跳过支持 JAX 数组的后端，因为其不支持项目赋值操作，测试特殊情况
    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    def test_nd_special_cases(self, xp):
        # 使用指定种子创建随机数生成器 rng
        rng = np.random.default_rng(34989235492245)
        # 将随机生成的 (3, 5) 数组转换为 xp 数组 x0 和 y0
        x0 = xp.asarray(rng.random((3, 5)))
        y0 = xp.asarray(rng.random((3, 5)))

        # 设置警告消息，用于匹配 ConstantInputWarning 警告
        message = 'An input array is constant'
        # 验证在调用 `stats.pearsonr` 时是否会引发 ConstantInputWarning 警告，且警告信息匹配预期消息
        with pytest.warns(stats.ConstantInputWarning, match=message):
            # 复制 x0 为 x，并将第一行所有元素设置为 1，然后计算 stats.pearsonr
            x = copy(x0)
            x[0, ...] = 1
            res = stats.pearsonr(x, y0, axis=1)
            ci = res.confidence_interval()
            # 将 xp.nan 转换为 xp.float64 类型的数组 nan
            nan = xp.asarray(xp.nan, dtype=xp.float64)
            # 验证 res.statistic[0] 是否与 nan 相等
            xp_assert_equal(res.statistic[0], nan)
            # 验证 res.pvalue[0] 是否与 nan 相等
            xp_assert_equal(res.pvalue[0], nan)
            # 验证 ci.low[0] 是否与 nan 相等
            xp_assert_equal(ci.low[0], nan)
            # 验证 ci.high[0] 是否与 nan 相等
            xp_assert_equal(ci.high[0], nan)
            # 断言 res.statistic[1:] 中是否没有 NaN 值
            assert not xp.any(xp.isnan(res.statistic[1:]))
            # 断言 res.pvalue[1:] 中是否没有 NaN 值
            assert not xp.any(xp.isnan(res.pvalue[1:]))
            # 断言 ci.low[1:] 中是否没有 NaN 值
            assert not xp.any(xp.isnan(ci.low[1:]))
            # 断言 ci.high[1:] 中是否没有 NaN 值
            assert not xp.any(xp.isnan(ci.high[1:]))

        # 设置警告消息，用于匹配 NearConstantInputWarning 警告
        message = 'An input array is nearly constant'
        # 验证在调用 `stats.pearsonr` 时是否会引发 NearConstantInputWarning 警告，且警告信息匹配预期消息
        with pytest.warns(stats.NearConstantInputWarning, match=message):
            # 将 x 的第一行第一列元素增加 1e-15，并计算 stats.pearsonr
            x[0, 0] = 1 + 1e-15
            stats.pearsonr(x, y0, axis=1)

        # 创建 4x2 的 xp 数组 x 和 y，全1 的 xp 数组 ones
        x = xp.asarray([[1, 2], [1, 2], [2, 1], [2, 1.]])
        y = xp.asarray([[1, 2], [2, 1], [1, 2], [2, 1.]])
        ones = xp.ones(4)
        # 计算 stats.pearsonr，并获取其置信区间
        res = stats.pearsonr(x, y, axis=-1)
        ci = res.confidence_interval()
        # 验证 res.statistic 是否与 [1, -1, -1, 1.] 相近
        xp_assert_close(res.statistic, xp.asarray([1, -1, -1, 1.]))
        # 验证 res.pvalue 是否与 ones 相近
        xp_assert_close(res.pvalue, ones)
        # 验证 ci.low 是否与 -ones 相近
        xp_assert_close(ci.low, -ones)
        # 验证 ci.high 是否与 ones 相近
        xp_assert_close(ci.high, ones)

    # 跳过支持 JAX 数组的后端，因为其不支持项目赋值操作，针对不同的轴参数进行参数化测试
    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'],
                      cpu_only=True)
    @pytest.mark.parametrize('axis', [0, 1, None])
    # 使用 pytest 的参数化功能，为 alternative 参数传入多个取值进行测试
    @pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
    # 定义测试函数 test_array_api，接受 xp、axis、alternative 三个参数
    def test_array_api(self, xp, axis, alternative):
        # 生成正态分布随机数，分别赋值给 x 和 y，形状为 (2, 10, 11)
        x, y = rng.normal(size=(2, 10, 11))
        # 使用 xp.asarray 将 x 和 y 转换为 xp（可能是 NumPy 或 Cupy）数组，计算相关系数和 p 值
        res = stats.pearsonr(xp.asarray(x), xp.asarray(y),
                             axis=axis, alternative=alternative)
        # 在参考实现上同样计算相关系数和 p 值
        ref = stats.pearsonr(x, y, axis=axis, alternative=alternative)
        # 使用 xp_assert_close 检查统计量的相似性
        xp_assert_close(res.statistic, xp.asarray(ref.statistic))
        # 使用 xp_assert_close 检查 p 值的相似性
        xp_assert_close(res.pvalue, xp.asarray(ref.pvalue))

        # 计算结果的置信区间
        res_ci = res.confidence_interval()
        # 参考实现的置信区间
        ref_ci = ref.confidence_interval()
        # 使用 xp_assert_close 检查置信区间下界的相似性
        xp_assert_close(res_ci.low, xp.asarray(ref_ci.low))
        # 使用 xp_assert_close 检查置信区间上界的相似性
        xp_assert_close(res_ci.high, xp.asarray(ref_ci.high))
class TestFisherExact:
    """Some tests to show that fisher_exact() works correctly.

    Note that in SciPy 0.9.0 this was not working well for large numbers due to
    inaccuracy of the hypergeom distribution (see #1218). Fixed now.

    Also note that R and SciPy have different argument formats for their
    hypergeometric distribution functions.

    R:
    > phyper(18999, 99000, 110000, 39000, lower.tail = FALSE)
    [1] 1.701815e-09
    """

    def test_basic(self):
        # 导入 fisher_exact 函数
        fisher_exact = stats.fisher_exact

        # 测试1: 使用 fisher_exact 函数计算 [[14500, 20000], [30000, 40000]] 的 p 值，并检查近似值是否等于 0.01106
        res = fisher_exact([[14500, 20000], [30000, 40000]])[1]
        assert_approx_equal(res, 0.01106, significant=4)

        # 测试2: 使用 fisher_exact 函数计算 [[100, 2], [1000, 5]] 的 p 值，并检查近似值是否等于 0.1301
        res = fisher_exact([[100, 2], [1000, 5]])[1]
        assert_approx_equal(res, 0.1301, significant=4)

        # 测试3: 使用 fisher_exact 函数计算 [[2, 7], [8, 2]] 的 p 值，并检查近似值是否等于 0.0230141
        res = fisher_exact([[2, 7], [8, 2]])[1]
        assert_approx_equal(res, 0.0230141, significant=6)

        # 测试4: 使用 fisher_exact 函数计算 [[5, 1], [10, 10]] 的 p 值，并检查近似值是否等于 0.1973244
        res = fisher_exact([[5, 1], [10, 10]])[1]
        assert_approx_equal(res, 0.1973244, significant=6)

        # 测试5: 使用 fisher_exact 函数计算 [[5, 15], [20, 20]] 的 p 值，并检查近似值是否等于 0.0958044
        res = fisher_exact([[5, 15], [20, 20]])[1]
        assert_approx_equal(res, 0.0958044, significant=6)

        # 测试6: 使用 fisher_exact 函数计算 [[5, 16], [20, 25]] 的 p 值，并检查近似值是否等于 0.1725862
        res = fisher_exact([[5, 16], [20, 25]])[1]
        assert_approx_equal(res, 0.1725862, significant=6)

        # 测试7: 使用 fisher_exact 函数计算 [[10, 5], [10, 1]] 的 p 值，并检查近似值是否等于 0.1973244
        res = fisher_exact([[10, 5], [10, 1]])[1]
        assert_approx_equal(res, 0.1973244, significant=6)

        # 测试8: 使用 fisher_exact 函数计算 [[5, 0], [1, 4]] 的 p 值，并检查近似值是否等于 0.04761904
        res = fisher_exact([[5, 0], [1, 4]])[1]
        assert_approx_equal(res, 0.04761904, significant=6)

        # 测试9: 使用 fisher_exact 函数计算 [[0, 1], [3, 2]] 的 p 值，并检查近似值是否等于 1.0
        res = fisher_exact([[0, 1], [3, 2]])[1]
        assert_approx_equal(res, 1.0)

        # 测试10: 使用 fisher_exact 函数计算 [[0, 2], [6, 4]] 的 p 值，并检查近似值是否等于 0.4545454545
        res = fisher_exact([[0, 2], [6, 4]])[1]
        assert_approx_equal(res, 0.4545454545)

        # 测试11: 使用 fisher_exact 函数计算 [[2, 7], [8, 2]] 的 p 值，并检查近似值是否等于 0.0230141
        res = fisher_exact([[2, 7], [8, 2]])
        assert_approx_equal(res[1], 0.0230141, significant=6)
        
        # 检查多重假设检验的第一个返回值是否等于预期的值
        assert_approx_equal(res[0], 4.0 / 56)
    def test_precise(self):
        # 从 R 中获取的结果
        #
        # R 中对 oddsratio 的定义不同（见 fisher_exact 文档字符串的注释部分），因此这些值不会匹配。
        # 我们仍然保留它们，以防它们以后可能会有用。我们只测试 p 值。
        tablist = [
            ([[100, 2], [1000, 5]], (2.505583993422285e-001, 1.300759363430016e-001)),
            ([[2, 7], [8, 2]], (8.586235135736206e-002, 2.301413756522114e-002)),
            ([[5, 1], [10, 10]], (4.725646047336584e+000, 1.973244147157190e-001)),
            ([[5, 15], [20, 20]], (3.394396617440852e-001, 9.580440012477637e-002)),
            ([[5, 16], [20, 25]], (3.960558326183334e-001, 1.725864953812994e-001)),
            ([[10, 5], [10, 1]], (2.116112781158483e-001, 1.973244147157190e-001)),
            ([[10, 5], [10, 0]], (0.000000000000000e+000, 6.126482213438734e-002)),
            ([[5, 0], [1, 4]], (np.inf, 4.761904761904762e-002)),
            ([[0, 5], [1, 4]], (0.000000000000000e+000, 1.000000000000000e+000)),
            ([[5, 1], [0, 4]], (np.inf, 4.761904761904758e-002)),
            ([[0, 1], [3, 2]], (0.000000000000000e+000, 1.000000000000000e+000))
            ]
        for table, res_r in tablist:
            # 使用 fisher_exact 计算统计结果
            res = stats.fisher_exact(np.asarray(table))
            # 使用 np.testing.assert_almost_equal 断言 p 值的近似相等性
            np.testing.assert_almost_equal(res[1], res_r[1], decimal=11,
                                           verbose=True)

    def test_gh4130(self):
        # 以前，用于区分理论上和数值上不同概率质量的误差修正因子是 1e-4；现在已经加紧以修复 gh4130。准确性已经与 R 的 fisher.test 进行检查。
        # options(digits=16)
        # table <- matrix(c(6, 108, 37, 200), nrow = 2)
        # fisher.test(table, alternative = "t")
        x = [[6, 37], [108, 200]]
        # 使用 fisher_exact 计算统计结果
        res = stats.fisher_exact(x)
        # 使用 assert_allclose 断言 p 值的接近程度
        assert_allclose(res[1], 0.005092697748126)

        # 来自 https://github.com/brentp/fishers_exact_test/issues/27 的案例
        # 该软件包有一个（绝对？）误差修正因子为 1e-6；太大了
        x = [[22, 0], [0, 102]]
        # 使用 fisher_exact 计算统计结果
        res = stats.fisher_exact(x)
        # 使用 assert_allclose 断言 p 值的接近程度
        assert_allclose(res[1], 7.175066786244549e-25)

        # 来自 https://github.com/brentp/fishers_exact_test/issues/1 的案例
        x = [[94, 48], [3577, 16988]]
        # 使用 fisher_exact 计算统计结果
        res = stats.fisher_exact(x)
        # 使用 assert_allclose 断言 p 值的接近程度
        assert_allclose(res[1], 2.069356340993818e-37)

    def test_gh9231(self):
        # 以前，对于这个表格，fisher_exact 的运行速度非常慢
        # 如 gh-9231 报告的，p 值应该非常接近零
        x = [[5829225, 5692693], [5760959, 5760959]]
        # 使用 fisher_exact 计算统计结果
        res = stats.fisher_exact(x)
        # 使用 assert_allclose 断言 p 值的接近程度，设置绝对容忍度为 1e-170
        assert_allclose(res[1], 0, atol=1e-170)

    @pytest.mark.slow
    def test_large_numbers(self):
        # 使用一些较大的数进行测试。这是对问题 #1401 的回归测试。
        pvals = [5.56e-11, 2.666e-11, 1.363e-11]  # 来自 R 语言
        for pval, num in zip(pvals, [75, 76, 77]):
            # 调用 stats 模块中的 fisher_exact 函数计算 Fisher 精确概率，并获取其返回结果中的第二个元素
            res = stats.fisher_exact([[17704, 496], [1065, num]])[1]
            # 使用 assert_approx_equal 函数检查 res 是否接近于 pval，精度为 4 位小数
            assert_approx_equal(res, pval, significant=4)

        # 再次调用 fisher_exact 函数，传入新的数据表格，并获取其返回结果中的第二个元素
        res = stats.fisher_exact([[18000, 80000], [20000, 90000]])[1]
        # 使用 assert_approx_equal 函数检查 res 是否接近于 0.2751，精度为 4 位小数
        assert_approx_equal(res, 0.2751, significant=4)

    def test_raises(self):
        # 测试当输入数据形状错误时是否会引发 ValueError 异常。
        assert_raises(ValueError, stats.fisher_exact,
                      np.arange(6).reshape(2, 3))

    def test_row_or_col_zero(self):
        tables = ([[0, 0], [5, 10]],
                  [[5, 10], [0, 0]],
                  [[0, 5], [0, 10]],
                  [[5, 0], [10, 0]])
        for table in tables:
            # 调用 fisher_exact 函数计算 Fisher 精确概率，并获取其返回结果的两个元素
            oddsratio, pval = stats.fisher_exact(table)
            # 使用 assert_equal 函数检查 pval 是否等于 1.0
            assert_equal(pval, 1.0)
            # 使用 assert_equal 函数检查 oddsratio 是否为 NaN
            assert_equal(oddsratio, np.nan)

    def test_less_greater(self):
        tables = (
            # 一些与 R 中进行比较的数据表格：
            [[2, 7], [8, 2]],
            [[200, 7], [8, 300]],
            [[28, 21], [6, 1957]],
            [[190, 800], [200, 900]],
            # 一些具有简单精确值的数据表格（包括问题 #1568 的回归测试）：
            [[0, 2], [3, 0]],
            [[1, 1], [2, 1]],
            [[2, 0], [1, 2]],
            [[0, 1], [2, 3]],
            [[1, 0], [1, 4]],
            )
        pvals = (
            # 来自 R：
            [0.018521725952066501, 0.9990149169715733],
            [1.0, 2.0056578803889148e-122],
            [1.0, 5.7284374608319831e-44],
            [0.7416227, 0.2959826],
            # 精确值：
            [0.1, 1.0],
            [0.7, 0.9],
            [1.0, 0.3],
            [2./3, 1.0],
            [1.0, 1./3],
            )
        for table, pval in zip(tables, pvals):
            res = []
            # 调用 fisher_exact 函数，使用 "less" 和 "greater" 作为 alternative 参数，获取返回结果中的第二个元素
            res.append(stats.fisher_exact(table, alternative="less")[1])
            res.append(stats.fisher_exact(table, alternative="greater")[1])
            # 使用 assert_allclose 函数检查 res 是否接近于 pval，绝对误差为 0，相对误差为 1e-7
            assert_allclose(res, pval, atol=0, rtol=1e-7)

    def test_gh3014(self):
        # 检查是否修复了问题 #3014。
        # 以前，这会引发 ValueError 异常。
        odds, pvalue = stats.fisher_exact([[1, 2], [9, 84419233]])

    @pytest.mark.parametrize("alternative", ['two-sided', 'less', 'greater'])
    def test_result(self, alternative):
        table = np.array([[14500, 20000], [30000, 40000]])
        # 调用 fisher_exact 函数，使用 alternative 参数作为参数，获取返回结果
        res = stats.fisher_exact(table, alternative=alternative)
        # 使用 assert_equal 函数检查结果的统计量和 p 值是否与返回结果相等
        assert_equal((res.statistic, res.pvalue), res)
class TestCorrSpearmanr:
    """ W.II.D. Compute a correlation matrix on all the variables.

        All the correlations, except for ZERO and MISS, should be exactly 1.
        ZERO and MISS should have undefined or missing correlations with the
        other variables.  The same should go for SPEARMAN correlations, if
        your program has them.
    """

    # 定义测试方法，用于测试单个标量输入的情况
    def test_scalar(self):
        # 计算两个标量的Spearman相关性，预期结果应该是包含NaN的数组
        y = stats.spearmanr(4., 2.)
        assert_(np.isnan(y).all())

    # 定义测试方法，用于测试输入序列长度不一致的情况
    def test_uneven_lengths(self):
        # 当输入序列长度不一致时，预期会抛出ValueError异常
        assert_raises(ValueError, stats.spearmanr, [1, 2, 1], [8, 9])
        assert_raises(ValueError, stats.spearmanr, [1, 2, 1], 8)

    # 定义测试方法，用于测试输入二维数组形状不一致的情况
    def test_uneven_2d_shapes(self):
        # 生成随机数据，并测试Spearman相关性的统计量和p值的形状
        np.random.seed(232324)
        x = np.random.randn(4, 3)
        y = np.random.randn(4, 2)
        assert stats.spearmanr(x, y).statistic.shape == (5, 5)
        assert stats.spearmanr(x.T, y.T, axis=1).pvalue.shape == (5, 5)

        # 当输入数组的列数不一致时，预期会抛出ValueError异常
        assert_raises(ValueError, stats.spearmanr, x, y, axis=1)
        assert_raises(ValueError, stats.spearmanr, x.T, y.T)

    # 定义测试方法，用于测试输入数组维度过高的情况
    def test_ndim_too_high(self):
        # 生成高维随机数据，并预期会抛出ValueError异常
        np.random.seed(232324)
        x = np.random.randn(4, 3, 2)
        assert_raises(ValueError, stats.spearmanr, x)
        assert_raises(ValueError, stats.spearmanr, x, x)
        assert_raises(ValueError, stats.spearmanr, x, None, None)
        # 使用axis=None（扁平化轴）应该可以处理两个输入数组
        assert_allclose(stats.spearmanr(x, x, axis=None),
                        stats.spearmanr(x.flatten(), x.flatten(), axis=0))

    # 定义测试方法，用于测试NaN处理策略的情况
    def test_nan_policy(self):
        # 创建包含NaN的序列，并测试不同的NaN处理策略下的Spearman相关性
        x = np.arange(10.)
        x[9] = np.nan
        assert_array_equal(stats.spearmanr(x, x), (np.nan, np.nan))
        assert_array_equal(stats.spearmanr(x, x, nan_policy='omit'),
                           (1.0, 0.0))
        assert_raises(ValueError, stats.spearmanr, x, x, nan_policy='raise')
        assert_raises(ValueError, stats.spearmanr, x, x, nan_policy='foobar')

    # 定义测试方法，用于测试NaN处理策略在Bug #12458下的情况
    def test_nan_policy_bug_12458(self):
        # 创建包含NaN的随机数据，并进行NaN策略测试
        np.random.seed(5)
        x = np.random.rand(5, 10)
        k = 6
        x[:, k] = np.nan
        y = np.delete(x, k, axis=1)
        corx, px = stats.spearmanr(x, nan_policy='omit')
        cory, py = stats.spearmanr(y)
        corx = np.delete(np.delete(corx, k, axis=1), k, axis=0)
        px = np.delete(np.delete(px, k, axis=1), k, axis=0)
        assert_allclose(corx, cory, atol=1e-14)
        assert_allclose(px, py, atol=1e-14)

    # 定义测试方法，用于测试NaN处理策略在Bug #12411下的情况
    def test_nan_policy_bug_12411(self):
        # 创建包含NaN的随机数据，并进行NaN策略测试
        np.random.seed(5)
        m = 5
        n = 10
        x = np.random.randn(m, n)
        x[1, 0] = np.nan
        x[3, -1] = np.nan
        corr, pvalue = stats.spearmanr(x, axis=1, nan_policy="propagate")
        res = [[stats.spearmanr(x[i, :], x[j, :]).statistic for i in range(m)]
               for j in range(m)]
        assert_allclose(corr, res)
    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sXX(self):
        # 计算 X 与 X 的 Spearman 相关系数
        y = stats.spearmanr(X, X)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sXBIG(self):
        # 计算 X 与 BIG 的 Spearman 相关系数
        y = stats.spearmanr(X, BIG)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sXLITTLE(self):
        # 计算 X 与 LITTLE 的 Spearman 相关系数
        y = stats.spearmanr(X, LITTLE)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sXHUGE(self):
        # 计算 X 与 HUGE 的 Spearman 相关系数
        y = stats.spearmanr(X, HUGE)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sXTINY(self):
        # 计算 X 与 TINY 的 Spearman 相关系数
        y = stats.spearmanr(X, TINY)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sXROUND(self):
        # 计算 X 与 ROUND 的 Spearman 相关系数
        y = stats.spearmanr(X, ROUND)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sBIGBIG(self):
        # 计算 BIG 与 BIG 的 Spearman 相关系数
        y = stats.spearmanr(BIG, BIG)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sBIGLITTLE(self):
        # 计算 BIG 与 LITTLE 的 Spearman 相关系数
        y = stats.spearmanr(BIG, LITTLE)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sBIGHUGE(self):
        # 计算 BIG 与 HUGE 的 Spearman 相关系数
        y = stats.spearmanr(BIG, HUGE)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sBIGTINY(self):
        # 计算 BIG 与 TINY 的 Spearman 相关系数
        y = stats.spearmanr(BIG, TINY)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sBIGROUND(self):
        # 计算 BIG 与 ROUND 的 Spearman 相关系数
        y = stats.spearmanr(BIG, ROUND)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sLITTLELITTLE(self):
        # 计算 LITTLE 与 LITTLE 的 Spearman 相关系数
        y = stats.spearmanr(LITTLE, LITTLE)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sLITTLEHUGE(self):
        # 计算 LITTLE 与 HUGE 的 Spearman 相关系数
        y = stats.spearmanr(LITTLE, HUGE)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sLITTLETINY(self):
        # 计算 LITTLE 与 TINY 的 Spearman 相关系数
        y = stats.spearmanr(LITTLE, TINY)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sLITTLEROUND(self):
        # 计算 LITTLE 与 ROUND 的 Spearman 相关系数
        y = stats.spearmanr(LITTLE, ROUND)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sHUGEHUGE(self):
        # 计算 HUGE 与 HUGE 的 Spearman 相关系数
        y = stats.spearmanr(HUGE, HUGE)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sHUGETINY(self):
        # 计算 HUGE 与 TINY 的 Spearman 相关系数
        y = stats.spearmanr(HUGE, TINY)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sHUGEROUND(self):
        # 计算 HUGE 与 ROUND 的 Spearman 相关系数
        y = stats.spearmanr(HUGE, ROUND)
        # 获取相关系数的值
        r = y[0]
        # 断言相关系数近似等于 1.0
        assert_approx_equal(r, 1.0)

    # 定义用于测试的函数，计算 Spearman 相关系数并进行断言
    def test_sTINYTINY(self):
        # 计算 TINY 与 TINY 的 Spearman 相关系
    def test_1d_vs_2d(self):
        # 定义两个一维数组
        x1 = [1, 2, 3, 4, 5, 6]
        x2 = [1, 2, 3, 4, 6, 5]
        # 计算 x1 和 x2 的斯皮尔曼相关系数
        res1 = stats.spearmanr(x1, x2)
        # 将 x1 和 x2 转换为二维数组，然后计算其斯皮尔曼相关系数
        res2 = stats.spearmanr(np.asarray([x1, x2]).T)
        # 断言两种方式计算得到的结果应该非常接近
        assert_allclose(res1, res2)

    def test_1d_vs_2d_nans(self):
        # 现在考虑包含 NaN 值的情况，用于 gh-9103 的回归测试
        for nan_policy in ['propagate', 'omit']:
            # 定义包含 NaN 值的 x1 和 x2 数组
            x1 = [1, np.nan, 3, 4, 5, 6]
            x2 = [1, 2, 3, 4, 6, np.nan]
            # 分别计算带有不同 NaN 策略的 x1 和 x2 的斯皮尔曼相关系数
            res1 = stats.spearmanr(x1, x2, nan_policy=nan_policy)
            res2 = stats.spearmanr(np.asarray([x1, x2]).T, nan_policy=nan_policy)
            # 断言两种方式计算得到的结果应该非常接近
            assert_allclose(res1, res2)

    def test_3cols(self):
        # 定义三列数组 x1, x2, x3
        x1 = np.arange(6)
        x2 = -x1
        x3 = np.array([0, 1, 2, 3, 5, 4])
        # 将 x1, x2, x3 组成一个三列的二维数组 x
        x = np.asarray([x1, x2, x3]).T
        # 计算 x 的斯皮尔曼相关系数
        actual = stats.spearmanr(x)
        # 定义预期的相关系数矩阵
        expected_corr = np.array([[1, -1, 0.94285714],
                                  [-1, 1, -0.94285714],
                                  [0.94285714, -0.94285714, 1]])
        # 定义预期的 p 值矩阵
        expected_pvalue = np.zeros((3, 3), dtype=float)
        expected_pvalue[2, 0:2] = 0.00480466472
        expected_pvalue[0:2, 2] = 0.00480466472
        # 断言实际计算的相关系数和预期值非常接近
        assert_allclose(actual.statistic, expected_corr)
        # 断言实际计算的 p 值和预期值非常接近
        assert_allclose(actual.pvalue, expected_pvalue)

    def test_gh_9103(self):
        # gh-9103 的回归测试
        x = np.array([[np.nan, 3.0, 4.0, 5.0, 5.1, 6.0, 9.2],
                      [5.0, np.nan, 4.1, 4.8, 4.9, 5.0, 4.1],
                      [0.5, 4.0, 7.1, 3.8, 8.0, 5.1, 7.6]]).T
        # 定义包含 NaN 的相关系数矩阵
        corr = np.array([[np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan],
                         [np.nan, np.nan, 1.]])
        # 断言使用不同的 NaN 策略计算得到的相关系数矩阵应该非常接近
        assert_allclose(stats.spearmanr(x, nan_policy='propagate').statistic,
                        corr)
        # 计算使用 'omit' 策略得到的相关系数，断言结果与预期非常接近
        res = stats.spearmanr(x, nan_policy='omit').statistic
        assert_allclose((res[0][1], res[0][2], res[1][2]),
                        (0.2051957, 0.4857143, -0.4707919), rtol=1e-6)

    def test_gh_8111(self):
        # gh-8111 的回归测试（float/int/bool 有不同结果的情况）
        n = 100
        np.random.seed(234568)
        x = np.random.rand(n)
        m = np.random.rand(n) > 0.7

        # bool 类型与 float 类型的斯皮尔曼相关系数比较，无 NaN
        a = (x > .5)
        b = np.array(x)
        res1 = stats.spearmanr(a, b, nan_policy='omit').statistic

        # bool 类型与 float 类型的斯皮尔曼相关系数比较，含 NaN
        b[m] = np.nan
        res2 = stats.spearmanr(a, b, nan_policy='omit').statistic

        # int 类型与 float 类型的斯皮尔曼相关系数比较，含 NaN
        a = a.astype(np.int32)
        res3 = stats.spearmanr(a, b, nan_policy='omit').statistic

        # 断言不同情况下计算得到的斯皮尔曼相关系数应该与预期非常接近
        expected = [0.865895477, 0.866100381, 0.866100381]
        assert_allclose([res1, res2, res3], expected)
class TestCorrSpearmanr2:
    """Some further tests of the spearmanr function."""

    def test_spearmanr_vs_r(self):
        # Cross-check with R:
        # cor.test(c(1,2,3,4,5),c(5,6,7,8,7),method="spearmanr")
        # 定义两个数组用于测试
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]
        # 预期的相关系数和 p 值
        expected = (0.82078268166812329, 0.088587005313543798)
        # 调用 scipy.stats 中的 spearmanr 函数进行计算
        res = stats.spearmanr(x1, x2)
        # 使用 assert_approx_equal 断言检查计算结果是否与预期接近
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    def test_empty_arrays(self):
        # 对空数组进行测试，期望返回 (NaN, NaN)
        assert_equal(stats.spearmanr([], []), (np.nan, np.nan))

    def test_normal_draws(self):
        # 使用随机数生成正态分布的数组进行测试
        np.random.seed(7546)
        x = np.array([np.random.normal(loc=1, scale=1, size=500),
                      np.random.normal(loc=1, scale=1, size=500)])
        # 定义一个指定的相关系数矩阵
        corr = [[1.0, 0.3],
                [0.3, 1.0]]
        # 对 x 进行线性变换，生成相关的数据
        x = np.dot(np.linalg.cholesky(corr), x)
        # 预期的相关系数和 p 值
        expected = (0.28659685838743354, 6.579862219051161e-11)
        # 调用 scipy.stats 中的 spearmanr 函数进行计算
        res = stats.spearmanr(x[0], x[1])
        # 使用 assert_approx_equal 断言检查计算结果是否与预期接近
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    def test_corr_1(self):
        # 对完全相关的情况进行测试，预期相关系数应为 1.0
        assert_approx_equal(stats.spearmanr([1, 1, 2], [1, 1, 2])[0], 1.0)

    def test_nan_policies(self):
        # 对包含 NaN 值的数组进行测试
        x = np.arange(10.)
        x[9] = np.nan
        # 检查默认情况下处理 NaN 值的返回结果
        assert_array_equal(stats.spearmanr(x, x), (np.nan, np.nan))
        # 检查忽略 NaN 值后的相关系数和 p 值
        assert_allclose(stats.spearmanr(x, x, nan_policy='omit'),
                        (1.0, 0))
        # 检查使用未知策略 'foobar' 时是否会引发 ValueError
        assert_raises(ValueError, stats.spearmanr, x, x, nan_policy='foobar')

    def test_unequal_lengths(self):
        # 对长度不相等的数组进行测试，预期会引发 ValueError
        x = np.arange(10.)
        y = np.arange(20.)
        assert_raises(ValueError, stats.spearmanr, x, y)

    def test_omit_paired_value(self):
        # 对包含 NaN 值的数组进行测试，检查忽略部分值的结果
        x1 = [1, 2, 3, 4]
        x2 = [8, 7, 6, np.nan]
        res1 = stats.spearmanr(x1, x2, nan_policy='omit')
        res2 = stats.spearmanr(x1[:3], x2[:3], nan_policy='omit')
        # 断言两种处理方式下的结果应该一致
        assert_equal(res1, res2)

    def test_gh_issue_6061_windows_overflow(self):
        # 对特定问题进行测试，预期的相关系数应接近 0.998
        x = list(range(2000))
        y = list(range(2000))
        y[0], y[9] = y[9], y[0]
        y[10], y[434] = y[434], y[10]
        y[435], y[1509] = y[1509], y[435]
        # 计算相关系数的预期值
        # rho = 1 - 6 * (2 * (9^2 + 424^2 + 1074^2))/(2000 * (2000^2 - 1))
        #     = 1 - (1 / 500)
        #     = 0.998
        x.append(np.nan)
        y.append(3.0)
        # 断言计算结果与预期值接近
        assert_almost_equal(stats.spearmanr(x, y, nan_policy='omit')[0], 0.998)
    def test_tie0(self):
        # with only ties in one or both inputs
        # 设置警告消息内容
        warn_msg = "An input array is constant"
        # 捕获 ConstantInputWarning 类型的警告，并验证警告消息是否匹配
        with pytest.warns(stats.ConstantInputWarning, match=warn_msg):
            # 计算 Spearman 相关系数 r 和 p 值，输入均为相同值的情况
            r, p = stats.spearmanr([2, 2, 2], [2, 2, 2])
            # 验证 r 是否为 NaN
            assert_equal(r, np.nan)
            # 验证 p 是否为 NaN
            assert_equal(p, np.nan)
            # 计算 Spearman 相关系数，其中一个输入有变化
            r, p = stats.spearmanr([2, 0, 2], [2, 2, 2])
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)
            # 计算 Spearman 相关系数，另一个输入有变化
            r, p = stats.spearmanr([2, 2, 2], [2, 0, 2])
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)

    def test_tie1(self):
        # Data
        # 定义数据 x 和 y
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 2.0, 2.0, 3.0]
        # Ranks of the data, with tie-handling.
        # 数据的排名，处理并列情况
        xr = [1.0, 2.0, 3.0, 4.0]
        yr = [1.0, 2.5, 2.5, 4.0]
        # 计算 Spearman 相关系数，与对排名应用 pearsonr 得到的结果应相同
        sr = stats.spearmanr(x, y)
        pr = stats.pearsonr(xr, yr)
        # 验证两种方法计算得到的结果是否近似相等
        assert_almost_equal(sr, pr)

    def test_tie2(self):
        # Test tie-handling if inputs contain nan's
        # 测试输入包含 NaN 时的并列处理
        # Data without nan's
        # 不包含 NaN 的数据
        x1 = [1, 2, 2.5, 2]
        y1 = [1, 3, 2.5, 4]
        # Same data with nan's
        # 包含 NaN 的相同数据
        x2 = [1, 2, 2.5, 2, np.nan]
        y2 = [1, 3, 2.5, 4, np.nan]

        # Results for two data sets should be the same if nan's are ignored
        # 如果忽略 NaN，两个数据集的结果应该相同
        sr1 = stats.spearmanr(x1, y1)
        sr2 = stats.spearmanr(x2, y2, nan_policy='omit')
        # 验证两个数据集的结果是否近似相等
        assert_almost_equal(sr1, sr2)

    def test_ties_axis_1(self):
        z1 = np.array([[1, 1, 1, 1], [1, 2, 3, 4]])
        z2 = np.array([[1, 2, 3, 4], [1, 1, 1, 1]])
        z3 = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        warn_msg = "An input array is constant"
        # 捕获 ConstantInputWarning 类型的警告，并验证警告消息是否匹配
        with pytest.warns(stats.ConstantInputWarning, match=warn_msg):
            # 沿着 axis=1 计算 Spearman 相关系数 r 和 p 值
            r, p = stats.spearmanr(z1, axis=1)
            # 验证 r 是否为 NaN
            assert_equal(r, np.nan)
            # 验证 p 是否为 NaN
            assert_equal(p, np.nan)
            # 沿着 axis=1 计算 Spearman 相关系数，对第二个数组 z2 进行相同操作
            r, p = stats.spearmanr(z2, axis=1)
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)
            # 沿着 axis=1 计算 Spearman 相关系数，对第三个数组 z3 进行相同操作
            r, p = stats.spearmanr(z3, axis=1)
            assert_equal(r, np.nan)
            assert_equal(p, np.nan)

    def test_gh_11111(self):
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([0, 0.009783728115345005, 0, 0, 0.0019759230121848587,
                      0.0007535430349118562, 0.0002661781514710257, 0, 0,
                      0.0007835762419683435])
        warn_msg = "An input array is constant"
        # 捕获 ConstantInputWarning 类型的警告，并验证警告消息是否匹配
        with pytest.warns(stats.ConstantInputWarning, match=warn_msg):
            # 计算 Spearman 相关系数 r 和 p 值
            r, p = stats.spearmanr(x, y)
            # 验证 r 是否为 NaN
            assert_equal(r, np.nan)
            # 验证 p 是否为 NaN
            assert_equal(p, np.nan)
    def test_index_error(self):
        # 创建包含数值的 NumPy 数组 x 和 y
        x = np.array([1.0, 7.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        y = np.array([0, 0.009783728115345005, 0, 0, 0.0019759230121848587,
                      0.0007535430349118562, 0.0002661781514710257, 0, 0,
                      0.0007835762419683435])
        # 断言 ValueError 异常在 stats.spearmanr 调用时被引发，期望 axis 参数为 2
        assert_raises(ValueError, stats.spearmanr, x, y, axis=2)

    def test_alternative(self):
        # 测试 alternative 参数

        # 简单测试 - 基于上述的 ``test_spearmanr_vs_r``
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]

        # 强正相关
        expected = (0.82078268166812329, 0.088587005313543798)

        # correlation > 0 -> 大的 "less" p 值
        res = stats.spearmanr(x1, x2, alternative="less")
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], 1 - (expected[1] / 2))

        # correlation > 0 -> 小的 "less" p 值
        res = stats.spearmanr(x1, x2, alternative="greater")
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1] / 2)

        # 使用 pytest 引发 ValueError，匹配特定错误消息
        with pytest.raises(ValueError, match="`alternative` must be 'less'..."):
            stats.spearmanr(x1, x2, alternative="ekki-ekki")

    @pytest.mark.parametrize("alternative", ('two-sided', 'less', 'greater'))
    def test_alternative_nan_policy(self, alternative):
        # 测试 nan 策略
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 7]
        x1nan = x1 + [np.nan]
        x2nan = x2 + [np.nan]

        # 测试 nan_policy="propagate"
        assert_array_equal(stats.spearmanr(x1nan, x2nan), (np.nan, np.nan))

        # 测试 nan_policy="omit"
        res_actual = stats.spearmanr(x1nan, x2nan, nan_policy='omit',
                                     alternative=alternative)
        res_expected = stats.spearmanr(x1, x2, alternative=alternative)
        assert_allclose(res_actual, res_expected)

        # 测试 nan_policy="raise"
        message = 'The input contains nan values'
        with pytest.raises(ValueError, match=message):
            stats.spearmanr(x1nan, x2nan, nan_policy='raise',
                            alternative=alternative)

        # 测试无效的 nan_policy
        message = "nan_policy must be one of..."
        with pytest.raises(ValueError, match=message):
            stats.spearmanr(x1nan, x2nan, nan_policy='ekki-ekki',
                            alternative=alternative)
#    W.II.E.  Tabulate X against X, using BIG as a case weight.  The values
#    should appear on the diagonal and the total should be 899999955.
#    If the table cannot hold these values, forget about working with
#    census data.  You can also tabulate HUGE against TINY.  There is no
#    reason a tabulation program should not be able to distinguish
#    different values regardless of their magnitude.

# I need to figure out how to do this one.


def test_kendalltau():
    # For the cases without ties, both variants should give the same
    # result.
    variants = ('b', 'c')

    # case without ties, con-dis equal zero
    x = [5, 2, 1, 3, 6, 4, 7, 8]
    y = [5, 2, 6, 3, 1, 8, 7, 4]
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (0.0, 1.0)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # case without ties, con-dis equal zero
    x = [0, 5, 2, 1, 3, 6, 4, 7, 8]
    y = [5, 2, 0, 6, 3, 1, 8, 7, 4]
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (0.0, 1.0)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # case without ties, con-dis close to zero
    x = [5, 2, 1, 3, 6, 4, 7]
    y = [5, 2, 6, 3, 1, 7, 4]
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (-0.14285714286, 0.77261904762)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # case without ties, con-dis close to zero
    x = [2, 1, 3, 6, 4, 7, 8]
    y = [2, 6, 3, 1, 8, 7, 4]
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (0.047619047619, 1.0)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # simple case without ties
    x = np.arange(10)
    y = np.arange(10)
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (1.0, 5.511463844797e-07)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # swap a couple of values
    b = y[1]
    y[1] = y[2]
    y[2] = b
    # Cross-check with exact result from R:
    # cor.test(x,y,method="kendall",exact=1)
    expected = (0.9555555555555556, 5.511463844797e-06)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])

    # swap a couple more
    b = y[5]
    y[5] = y[6]
    y[6] = b
    # 将变量 b 赋值给数组 y 的第 6 个元素

    # Cross-check with exact result from R:
    # 使用 R 的确切结果进行交叉验证：
    # cor.test(x,y,method="kendall",exact=1)
    # 在 x 和 y 上执行 Kendall 相关性检验，方法为 Kendall，exact 参数设置为 1

    expected = (0.9111111111111111, 2.976190476190e-05)
    # 预期的 Kendall 相关性系数和 p 值

    for taux in variants:
        # 对于每种变体 taux
        res = stats.kendalltau(x, y, variant=taux)
        # 使用 stats 模块中的 kendalltau 函数计算 x 和 y 的 Kendall 相关性系数和 p 值
        assert_approx_equal(res[0], expected[0])
        # 断言 Kendall 相关性系数接近预期值
        assert_approx_equal(res[1], expected[1])
        # 断言 p 值接近预期值

    # same in opposite direction
    # 在相反的方向上进行相同的操作

    x = np.arange(10)
    # 创建一个从 0 到 9 的数组 x
    y = np.arange(10)[::-1]
    # 创建一个从 9 到 0 的数组 y

    # Cross-check with exact result from R:
    # 使用 R 的确切结果进行交叉验证：
    # cor.test(x,y,method="kendall",exact=1)

    expected = (-1.0, 5.511463844797e-07)
    # 预期的 Kendall 相关性系数和 p 值

    for taux in variants:
        # 对于每种变体 taux
        res = stats.kendalltau(x, y, variant=taux)
        # 使用 stats 模块中的 kendalltau 函数计算 x 和 y 的 Kendall 相关性系数和 p 值
        assert_approx_equal(res[0], expected[0])
        # 断言 Kendall 相关性系数接近预期值
        assert_approx_equal(res[1], expected[1])
        # 断言 p 值接近预期值

    # swap a couple of values
    # 交换几个值

    b = y[1]
    # 将 y 的第 1 个元素赋值给变量 b
    y[1] = y[2]
    # 将 y 的第 2 个元素赋值给 y 的第 1 个元素
    y[2] = b
    # 将变量 b 的值赋值给 y 的第 2 个元素

    # Cross-check with exact result from R:
    # 使用 R 的确切结果进行交叉验证：
    # cor.test(x,y,method="kendall",exact=1)

    expected = (-0.9555555555555556, 5.511463844797e-06)
    # 预期的 Kendall 相关性系数和 p 值

    for taux in variants:
        # 对于每种变体 taux
        res = stats.kendalltau(x, y, variant=taux)
        # 使用 stats 模块中的 kendalltau 函数计算 x 和 y 的 Kendall 相关性系数和 p 值
        assert_approx_equal(res[0], expected[0])
        # 断言 Kendall 相关性系数接近预期值
        assert_approx_equal(res[1], expected[1])
        # 断言 p 值接近预期值

    # swap a couple more
    # 再交换几个值

    b = y[5]
    # 将 y 的第 5 个元素赋值给变量 b
    y[5] = y[6]
    # 将 y 的第 6 个元素赋值给 y 的第 5 个元素
    y[6] = b
    # 将变量 b 的值赋值给 y 的第 6 个元素

    # Cross-check with exact result from R:
    # 使用 R 的确切结果进行交叉验证：
    # cor.test(x,y,method="kendall",exact=1)

    expected = (-0.9111111111111111, 2.976190476190e-05)
    # 预期的 Kendall 相关性系数和 p 值

    for taux in variants:
        # 对于每种变体 taux
        res = stats.kendalltau(x, y, variant=taux)
        # 使用 stats 模块中的 kendalltau 函数计算 x 和 y 的 Kendall 相关性系数和 p 值
        assert_approx_equal(res[0], expected[0])
        # 断言 Kendall 相关性系数接近预期值
        assert_approx_equal(res[1], expected[1])
        # 断言 p 值接近预期值

    # Check a case where variants are different
    # 检查变体不同的情况

    # Example values found from Kendall (1970).
    # 从 Kendall (1970) 找到的示例值
    x = array([1, 2, 2, 4, 4, 6, 6, 8, 9, 9])
    # 创建数组 x
    y = array([1, 2, 4, 4, 4, 4, 8, 8, 8, 10])
    # 创建数组 y

    expected = 0.85895569
    # 预期的 Kendall 相关性系数

    assert_approx_equal(stats.kendalltau(x, y, variant='b')[0], expected)
    # 断言使用 variant='b' 计算的 Kendall 相关性系数接近预期值

    expected = 0.825
    # 预期的 Kendall 相关性系数

    assert_approx_equal(stats.kendalltau(x, y, variant='c')[0], expected)
    # 断言使用 variant='c' 计算的 Kendall 相关性系数接近预期值

    # check exception in case of ties and method='exact' requested
    # 检查绑定（ties）和请求 method='exact' 时的异常情况

    y[2] = y[1]
    # 将 y 的第 1 个元素赋值给 y 的第 2 个元素

    assert_raises(ValueError, stats.kendalltau, x, y, method='exact')
    # 断言调用 kendalltau 函数时会引发 ValueError 异常，因为出现了绑定并且 method='exact'

    # check exception in case of invalid method keyword
    # 检查无效的 method 关键字时的异常情况

    assert_raises(ValueError, stats.kendalltau, x, y, method='banana')
    # 断言调用 kendalltau 函数时会引发 ValueError 异常，因为 method 关键字是无效的

    # check exception in case of invalid variant keyword
    # 检查无效的 variant 关键字时的异常情况

    assert_raises(ValueError, stats.kendalltau, x, y, variant='rms')
    # 断言调用 kendalltau 函数时会引发 ValueError 异常，因为 variant 关键字是无效的

    # tau-b with some ties
    # 包含一些绑定（ties）的 tau-b 检验

    # Cross-check with R:
    # 使用 R 进行交叉验证：
    # cor.test(c(12,2,1,12,2),c(1,4,7,1,0),method="kendall",exact=FALSE)

    x1 = [12, 2, 1, 12, 2]
    # 创建数组 x1
    x2 = [1, 4, 7, 1, 0]
    # 创建数组 x2

    expected = (-0.47140452079103173, 0.28274545993277478)
    # 预期的 Kendall 相关性系数和 p 值

    res = stats.kendalltau(x1, x2)
    # 使用 stats 模块中的 kendalltau 函数计算 x1 和 x2 的 Kendall 相关性系数和 p 值

    assert_approx_equal(res[0], expected[0])
    # 断言 Kendall 相关性系数接近预期值
    assert_approx_equal(res[1], expected[
    # 对于每种变体执行以下操作：输入中只有一个或两个输入中有tau-b或tau-c的联系
    for taux in variants:
        # 断言：检查两个相同的输入数组使用Kendall's tau-b或tau-c时的结果是否为NaN
        assert_equal(stats.kendalltau([2, 2, 2], [2, 2, 2], variant=taux),
                     (np.nan, np.nan))
        assert_equal(stats.kendalltau([2, 0, 2], [2, 2, 2], variant=taux),
                     (np.nan, np.nan))
        assert_equal(stats.kendalltau([2, 2, 2], [2, 0, 2], variant=taux),
                     (np.nan, np.nan))

    # 空数组作为输入
    assert_equal(stats.kendalltau([], []), (np.nan, np.nan))

    # 使用较大的数组进行检查
    np.random.seed(7546)
    x = np.array([np.random.normal(loc=1, scale=1, size=500),
                  np.random.normal(loc=1, scale=1, size=500)])
    # 创建一个相关矩阵并应用Cholesky分解
    corr = [[1.0, 0.3],
            [0.3, 1.0]]
    x = np.dot(np.linalg.cholesky(corr), x)
    # 期望的结果
    expected = (0.19291382765531062, 1.1337095377742629e-10)
    # 计算Kendall's tau并断言结果近似等于预期值
    res = stats.kendalltau(x[0], x[1])
    assert_approx_equal(res[0], expected[0])
    assert_approx_equal(res[1], expected[1])

    # 对于tau-b应该结果为1但tau-c不应该
    assert_approx_equal(stats.kendalltau([1, 1, 2], [1, 1, 2], variant='b')[0],
                        1.0)
    assert_approx_equal(stats.kendalltau([1, 1, 2], [1, 1, 2], variant='c')[0],
                        0.88888888)

    # 测试nan_policy
    x = np.arange(10.)
    x[9] = np.nan
    # 断言：检查包含NaN值的输入数组使用Kendall's tau时的结果是否为NaN
    assert_array_equal(stats.kendalltau(x, x), (np.nan, np.nan))
    # 断言：检查忽略NaN值后的结果是否接近给定的值
    assert_allclose(stats.kendalltau(x, x, nan_policy='omit'),
                    (1.0, 5.5114638e-6), rtol=1e-06)
    assert_allclose(stats.kendalltau(x, x, nan_policy='omit', method='asymptotic'),
                    (1.0, 0.00017455009626808976), rtol=1e-06)
    # 断言：检查在指定了未知的nan_policy时是否引发了ValueError
    assert_raises(ValueError, stats.kendalltau, x, x, nan_policy='raise')
    assert_raises(ValueError, stats.kendalltau, x, x, nan_policy='foobar')

    # 断言：检查长度不相等的输入数组是否引发了ValueError
    x = np.arange(10.)
    y = np.arange(20.)
    assert_raises(ValueError, stats.kendalltau, x, y)

    # 断言：检查所有输入均为ties的情况下的结果是否为NaN
    tau, p_value = stats.kendalltau([], [])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.kendalltau([0], [0])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)

    # GitHub问题＃6061的回归测试 - 在Windows上的溢出问题
    x = np.arange(2000, dtype=float)
    x = np.ma.masked_greater(x, 1995)
    y = np.arange(2000, dtype=float)
    y = np.concatenate((y[1000:], y[:1000]))
    # 断言：检查Kendall's tau的结果是否为有限值
    assert_(np.isfinite(stats.kendalltau(x,y)[1]))
# 定义一个测试函数，用于测试 kendalltau 函数与 mstats_basic 模块的基本功能
def test_kendalltau_vs_mstats_basic():
    # 设置随机种子为 42，保证可重复性
    np.random.seed(42)
    # 循环测试排名长度从 2 到 9
    for s in range(2,10):
        a = []
        # 生成带有并列的排名
        for i in range(s):
            a += [i]*i
        b = list(a)
        # 打乱生成的排名顺序，模拟随机排名
        np.random.shuffle(a)
        np.random.shuffle(b)
        # 使用 mstats_basic 模块中的 kendalltau 计算期望值
        expected = mstats_basic.kendalltau(a, b)
        # 使用 stats 模块中的 kendalltau 计算实际值
        actual = stats.kendalltau(a, b)
        # 断言实际值的第一个返回值近似等于期望值的第一个返回值
        assert_approx_equal(actual[0], expected[0])
        # 断言实际值的第二个返回值近似等于期望值的第二个返回值
        assert_approx_equal(actual[1], expected[1])


# 定义一个测试函数，用于测试 kendalltau 函数对第二个参数含有 NaN 值的处理
def test_kendalltau_nan_2nd_arg():
    # 用于回归测试 gh-6134：第二个参数中含有 NaN 值时未被正确处理的情况
    x = [1., 2., 3., 4.]
    y = [np.nan, 2.4, 3.4, 3.4]

    # 使用 kendalltau 函数，设定 nan_policy='omit' 来处理 NaN 值
    r1 = stats.kendalltau(x, y, nan_policy='omit')
    # 对去除了第一个元素后的 x 和 y 运行 kendalltau 函数
    r2 = stats.kendalltau(x[1:], y[1:])
    # 断言两个统计量的值近似相等，容差为 1e-15
    assert_allclose(r1.statistic, r2.statistic, atol=1e-15)


# 定义一个测试函数，用于测试 kendalltau 函数是否解决了 gh-18139 报告的整数溢出问题
def test_kendalltau_gh18139_overflow():
    # gh-18139 报告在 SciPy 0.15.1 之后版本的 kendalltau 存在整数溢出问题
    # 设置随机种子为 6272161
    import random
    random.seed(6272161)
    classes = [1, 2, 3, 4, 5, 6, 7]
    n_samples = 2 * 10 ** 5
    # 从 classes 中随机选择 n_samples 个样本作为 x 和 y 的值
    x = random.choices(classes, k=n_samples)
    y = random.choices(classes, k=n_samples)
    # 使用 kendalltau 函数计算 x 和 y 的相关性
    res = stats.kendalltau(x, y)
    # 使用 SciPy 0.15.1 中的参考值作为统计量的基准值
    assert_allclose(res.statistic, 0.0011816493905730343)
    # 使用 permutation_test 函数的默认参数进行置换测试，期望 p 值的精确度至少达到两位小数
    assert_allclose(res.pvalue, 0.4894, atol=2e-3)
    # 定义一个测试方法，测试 Kendall Tau 相关系数的不同参数和方法

    # 第一个测试用例：强正相关
    x1 = [1, 2, 3, 4, 5]
    x2 = [5, 6, 7, 8, 7]

    # 使用两边检验法计算 Kendall Tau 相关系数，期望相关系数大于0
    expected = stats.kendalltau(x1, x2, alternative="two-sided")
    assert expected[0] > 0

    # 使用小于检验法计算 Kendall Tau 相关系数，确保相关系数等于期望值，
    # 并且 p 值与期望值的一半的差值足够小
    res = stats.kendalltau(x1, x2, alternative="less")
    assert_equal(res[0], expected[0])
    assert_allclose(res[1], 1 - (expected[1] / 2))

    # 使用大于检验法计算 Kendall Tau 相关系数，确保相关系数等于期望值，
    # 并且 p 值与期望值的一半的差值足够小
    res = stats.kendalltau(x1, x2, alternative="greater")
    assert_equal(res[0], expected[0])
    assert_allclose(res[1], expected[1] / 2)

    # 第二个测试用例：强负相关
    x2.reverse()  # 反转 x2，改变相关方向

    # 使用两边检验法计算 Kendall Tau 相关系数，期望相关系数小于0
    expected = stats.kendalltau(x1, x2, alternative="two-sided")
    assert expected[0] < 0

    # 使用大于检验法计算 Kendall Tau 相关系数，确保相关系数等于期望值，
    # 并且 p 值与期望值的一半的差值足够小
    res = stats.kendalltau(x1, x2, alternative="greater")
    assert_equal(res[0], expected[0])
    assert_allclose(res[1], 1 - (expected[1] / 2))

    # 使用小于检验法计算 Kendall Tau 相关系数，确保相关系数等于期望值，
    # 并且 p 值与期望值的一半的差值足够小
    res = stats.kendalltau(x1, x2, alternative="less")
    assert_equal(res[0], expected[0])
    assert_allclose(res[1], expected[1] / 2)

    # 引发 ValueError，测试非法的 alternative 参数是否触发异常
    with pytest.raises(ValueError, match="`alternative` must be 'less'..."):
        stats.kendalltau(x1, x2, alternative="ekki-ekki")

    # 备注部分：这里列举了在计算确切 p 值时所考虑的各种特殊情况，
    # 包括观察统计量位于左尾部分和右尾部分的情况，因为代码利用了零分布的对称性；
    # 我们使用相同的测试用例，但是对一个样本进行取反来分别测试。
    # 参考值是使用 R 中的 cor.test 计算得到的，例如：
    # options(digits=16)
    # x <- c(44.4, 45.9, 41.9, 53.3, 44.7, 44.1, 50.7, 45.2, 60.1)
    # y <- c( 2.6,  3.1,  2.5,  5.0,  3.6,  4.0,  5.2,  2.8,  3.8)
    # cor.test(x, y, method = "kendall", alternative = "g")

    # 预定义了一组 alternative 参数，用于后续测试
    alternatives = ('less', 'two-sided', 'greater')

    # 不同情况下的 p 值列表
    p_n1 = [np.nan, np.nan, np.nan]
    p_n2 = [1, 1, 0.5]
    p_c0 = [1, 0.3333333333333, 0.1666666666667]
    p_c1 = [0.9583333333333, 0.3333333333333, 0.1666666666667]
    p_no_correlation = [0.5916666666667, 1, 0.5916666666667]
    p_no_correlationb = [0.5475694444444, 1, 0.5475694444444]
    p_n_lt_171 = [0.9624118165785, 0.1194389329806, 0.0597194664903]
    p_n_lt_171b = [0.246236925303, 0.4924738506059, 0.755634083327]
    p_n_lt_171c = [0.9847475308925, 0.03071385306533, 0.01535692653267]
    # 定义一个方法来进行精确性测试，根据参数决定是否反转 y 数组，并调整期望的统计值和 p 值
    def exact_test(self, x, y, alternative, rev, stat_expected, p_expected):
        if rev:
            # 如果 rev 为 True，则将 y 数组转换为其相反数的数组
            y = -np.asarray(y)
            # 同时调整期望的统计值为其相反数
            stat_expected *= -1
        # 使用 scipy.stats 库中的 kendalltau 方法进行 Kendall tau 相关性测试
        res = stats.kendalltau(x, y, method='exact', alternative=alternative)
        # 将计算得到的结果与期望的结果组成元组
        res_expected = stat_expected, p_expected
        # 使用 assert_allclose 方法断言计算结果与期望结果的接近程度
        assert_allclose(res, res_expected)

    # 定义多组参数化测试用例，每组参数包括 alternative（替代假设类型）、p_expected（期望的 p 值）、rev（是否反转 y 数组）
    case_R_n1 = (list(zip(alternatives, p_n1, [False]*3))
                 + list(zip(alternatives, reversed(p_n1), [True]*3)))
    
    # 使用 pytest.mark.parametrize 注解，指定参数化测试用例，测试针对 R_n1 的情况
    @pytest.mark.parametrize("alternative, p_expected, rev", case_R_n1)
    def test_against_R_n1(self, alternative, p_expected, rev):
        # 定义 x 和 y 数组，分别为 [1] 和 [2]
        x, y = [1], [2]
        # 设置期望的统计值为 NaN
        stat_expected = np.nan
        # 调用 exact_test 方法进行测试
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)

    # 类似地定义测试用例针对 R_n2、R_c0、R_c1、no_correlation 和 no_correlationb 的情况，具体细节略同上
    # 生成包含了三个元素的元组列表，每个元组包含三个元素：alternative, p_n_lt_171, False
    # 以及包含了相同三个元素的元组列表，但是第三个元素为 True，然后将两个列表连接起来，形成 case_R_lt_171 变量
    case_R_lt_171 = (list(zip(alternatives, p_n_lt_171, [False]*3))
                     + list(zip(alternatives, reversed(p_n_lt_171), [True]*3)))
    
    @pytest.mark.parametrize("alternative, p_expected, rev", case_R_lt_171)
    # 定义一个参数化测试函数，使用 case_R_lt_171 中的参数来运行多个测试实例
    def test_against_R_lt_171(self, alternative, p_expected, rev):
        # 数据来自 Hollander & Wolfe (1973)，p. 187f.
        # 使用自 https://rdrr.io/r/stats/cor.test.html 获取的数据
        x = [44.4, 45.9, 41.9, 53.3, 44.7, 44.1, 50.7, 45.2, 60.1]
        y = [2.6, 3.1, 2.5, 5.0, 3.6, 4.0, 5.2, 2.8, 3.8]
        stat_expected = 0.4444444444444445
        # 调用自定义的 exact_test 方法进行测试，传入相关参数
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    
    # 生成包含了三个元素的元组列表，每个元组包含三个元素：alternative, p_n_lt_171b, False
    # 以及包含了相同三个元素的元组列表，但是第三个元素为 True，然后将两个列表连接起来，形成 case_R_lt_171b 变量
    case_R_lt_171b = (list(zip(alternatives, p_n_lt_171b, [False]*3))
                      + list(zip(alternatives, reversed(p_n_lt_171b),
                                 [True]*3)))
    
    @pytest.mark.parametrize("alternative, p_expected, rev", case_R_lt_171b)
    # 定义一个参数化测试函数，使用 case_R_lt_171b 中的参数来运行多个测试实例
    def test_against_R_lt_171b(self, alternative, p_expected, rev):
        np.random.seed(0)
        x = np.random.rand(100)
        y = np.random.rand(100)
        stat_expected = -0.04686868686868687
        # 调用自定义的 exact_test 方法进行测试，传入相关参数
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    
    # 生成包含了三个元素的元组列表，每个元组包含三个元素：alternative, p_n_lt_171c, False
    # 以及包含了相同三个元素的元组列表，但是第三个元素为 True，然后将两个列表连接起来，形成 case_R_lt_171c 变量
    case_R_lt_171c = (list(zip(alternatives, p_n_lt_171c, [False]*3))
                      + list(zip(alternatives, reversed(p_n_lt_171c),
                                 [True]*3)))
    
    @pytest.mark.parametrize("alternative, p_expected, rev", case_R_lt_171c)
    # 定义一个参数化测试函数，使用 case_R_lt_171c 中的参数来运行多个测试实例
    def test_against_R_lt_171c(self, alternative, p_expected, rev):
        np.random.seed(0)
        x = np.random.rand(170)
        y = np.random.rand(170)
        stat_expected = 0.1115906717716673
        # 调用自定义的 exact_test 方法进行测试，传入相关参数
        self.exact_test(x, y, alternative, rev, stat_expected, p_expected)
    
    # 生成包含了两个元素的元组列表，每个元组包含两个元素：alternative, False
    # 以及包含了相同两个元素的元组列表，但是第二个元素为 True，然后将两个列表连接起来，形成 case_gt_171 变量
    case_gt_171 = (list(zip(alternatives, [False]*3)) +
                   list(zip(alternatives, [True]*3)))
    
    @pytest.mark.parametrize("alternative, rev", case_gt_171)
    # 定义一个参数化测试函数，使用 case_gt_171 中的参数来运行多个测试实例
    def test_gt_171(self, alternative, rev):
        np.random.seed(0)
        x = np.random.rand(400)
        y = np.random.rand(400)
        # 使用 Kendall tau 相关系数进行计算，分别使用 exact 方法和 asymptotic 方法
        res0 = stats.kendalltau(x, y, method='exact', alternative=alternative)
        res1 = stats.kendalltau(x, y, method='asymptotic', alternative=alternative)
        # 断言两种方法的相关系数（第一个元素）应该相等
        assert_equal(res0[0], res1[0])
        # 断言两种方法的 p 值（第二个元素）应该在相对误差容忍范围内相等
        assert_allclose(res0[1], res1[1], rtol=1e-3)
    
    @pytest.mark.parametrize("method", ('exact', 'asymptotic'))
    @pytest.mark.parametrize("alternative", ('two-sided', 'less', 'greater'))
    # 定义一个参数化测试函数，将 method 和 alternative 参数化，使其能够组合生成多个测试实例
    # 定义一个测试方法，用于测试不同的 NaN 策略
    def test_nan_policy(self, method, alternative):
        # 创建包含 NaN 值的两个列表
        x1 = [1, 2, 3, 4, 5]
        x2 = [5, 6, 7, 8, 9]
        x1nan = x1 + [np.nan]
        x2nan = x2 + [np.nan]

        # 测试 nan_policy="propagate"
        # 调用 stats 模块中的 kendalltau 函数，计算相关系数，预期结果是 (NaN, NaN)
        res_actual = stats.kendalltau(x1nan, x2nan,
                                      method=method, alternative=alternative)
        res_expected = (np.nan, np.nan)
        # 使用 assert_allclose 函数验证实际结果与预期结果的近似性
        assert_allclose(res_actual, res_expected)

        # 测试 nan_policy="omit"
        # 调用 kendalltau 函数，设置 nan_policy='omit'，对比去除 NaN 后的结果
        res_actual = stats.kendalltau(x1nan, x2nan, nan_policy='omit',
                                      method=method, alternative=alternative)
        # 计算不含 NaN 的情况下的 Kendall tau 相关系数的预期结果
        res_expected = stats.kendalltau(x1, x2, method=method,
                                        alternative=alternative)
        assert_allclose(res_actual, res_expected)

        # 测试 nan_policy="raise"
        # 检验当 nan_policy='raise' 时，函数是否会抛出 ValueError 异常
        message = 'The input contains nan values'
        with pytest.raises(ValueError, match=message):
            stats.kendalltau(x1nan, x2nan, nan_policy='raise',
                             method=method, alternative=alternative)

        # 测试无效的 nan_policy
        # 检验当提供无效的 nan_policy 时，是否会抛出 ValueError 异常
        message = "nan_policy must be one of..."
        with pytest.raises(ValueError, match=message):
            stats.kendalltau(x1nan, x2nan, nan_policy='ekki-ekki',
                             method=method, alternative=alternative)
# 定义一个函数用于测试加权 Kendall's tau 相关性的统计功能
def test_weightedtau():
    # 准备测试数据
    x = [12, 2, 1, 12, 2]
    y = [1, 4, 7, 1, 0]
    
    # 计算加权 Kendall's tau 相关性及其 p 值
    tau, p_value = stats.weightedtau(x, y)
    # 断言计算结果与预期相近
    assert_approx_equal(tau, -0.56694968153682723)
    # 断言 p 值为 NaN
    assert_equal(np.nan, p_value)
    
    # 再次计算加权 Kendall's tau 相关性及其 p 值，不采用加法权重
    tau, p_value = stats.weightedtau(x, y, additive=False)
    # 断言计算结果与预期相近
    assert_approx_equal(tau, -0.62205716951801038)
    # 断言 p 值为 NaN
    assert_equal(np.nan, p_value)
    
    # 以 lambda 函数作为权重函数，计算加权 Kendall's tau 相关性及其 p 值
    tau, p_value = stats.weightedtau(x, y, weigher=lambda x: 1)
    # 断言计算结果与预期相近
    assert_approx_equal(tau, -0.47140452079103173)
    # 断言 p 值为 NaN
    assert_equal(np.nan, p_value)

    # 测试 namedtuple 属性结果
    res = stats.weightedtau(x, y)
    attributes = ('correlation', 'pvalue')
    # 检查结果是否包含指定的命名元组属性
    check_named_results(res, attributes)
    # 断言结果的 correlation 属性与 statistic 属性相等
    assert_equal(res.correlation, res.statistic)

    # 测试非对称、排名版本
    tau, p_value = stats.weightedtau(x, y, rank=None)
    assert_approx_equal(tau, -0.4157652301037516)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(y, x, rank=None)
    assert_approx_equal(tau, -0.7181341329699029)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, rank=None, additive=False)
    assert_approx_equal(tau, -0.40644850966246893)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(y, x, rank=None, additive=False)
    assert_approx_equal(tau, -0.83766582937355172)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(x, y, rank=False)
    assert_approx_equal(tau, -0.51604397940261848)
    assert_equal(np.nan, p_value)
    
    # 再次以 lambda 函数作为权重函数，测试加权 Kendall's tau 相关性及其 p 值
    tau, p_value = stats.weightedtau(x, y, rank=True, weigher=lambda x: 1)
    # 断言计算结果与预期相近
    assert_approx_equal(tau, -0.47140452079103173)
    # 断言 p 值为 NaN
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau(y, x, rank=True, weigher=lambda x: 1)
    assert_approx_equal(tau, -0.47140452079103173)
    assert_equal(np.nan, p_value)
    
    # 测试参数转换
    tau, p_value = stats.weightedtau(np.asarray(x, dtype=np.float64), y)
    assert_approx_equal(tau, -0.56694968153682723)
    tau, p_value = stats.weightedtau(np.asarray(x, dtype=np.int16), y)
    assert_approx_equal(tau, -0.56694968153682723)
    tau, p_value = stats.weightedtau(np.asarray(x, dtype=np.float64),
                                     np.asarray(y, dtype=np.float64))
    assert_approx_equal(tau, -0.56694968153682723)
    
    # 所有值都是 ties（平局）的情况
    tau, p_value = stats.weightedtau([], [])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.weightedtau([0], [0])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    
    # 大小不匹配的情况
    assert_raises(ValueError, stats.weightedtau, [0, 1], [0, 1, 2])
    assert_raises(ValueError, stats.weightedtau, [0, 1], [0, 1], [0])
    
    # 包含 NaN 的情况
    x = [12, 2, 1, 12, 2]
    y = [1, 4, 7, 1, np.nan]
    tau, p_value = stats.weightedtau(x, y)
    assert_approx_equal(tau, -0.56694968153682723)
    x = [12, 2, np.nan, 12, 2]
    tau, p_value = stats.weightedtau(x, y)
    # 断言：检查 tau 的近似值是否等于 -0.56694968153682723
    assert_approx_equal(tau, -0.56694968153682723)
    
    # 创建两个列表 x 和 y，包含了浮点数和 NaN 值
    x = [12.0, 2.0, 1.0, 12.0, 2.0]
    y = [1.0, 4.0, 7.0, 1.0, np.nan]
    
    # 使用 stats 模块中的 weightedtau 函数计算 x 和 y 的 tau 值和 p 值
    tau, p_value = stats.weightedtau(x, y)
    
    # 再次断言：检查 tau 的近似值是否等于 -0.56694968153682723
    assert_approx_equal(tau, -0.56694968153682723)
    
    # 更改 x 中的一个元素为 np.nan，重新计算 tau 和 p 值
    x = [12.0, 2.0, np.nan, 12.0, 2.0]
    tau, p_value = stats.weightedtau(x, y)
    
    # 再次断言：检查 tau 的近似值是否等于 -0.56694968153682723
    assert_approx_equal(tau, -0.56694968153682723)
    
    # 创建新的 x 和 y 列表，其中 x 包含一个 np.nan 值
    x = [12.0, 2.0, 1.0, 12.0, 1.0]
    y = [1.0, 4.0, 7.0, 1.0, 1.0]
    
    # 计算这些列表的 weightedtau，并获得 tau 和 p 值
    tau, p_value = stats.weightedtau(x, y)
    
    # 断言：检查 tau 的近似值是否等于 -0.6615242347139803
    assert_approx_equal(tau, -0.6615242347139803)
    
    # 更新 x，其中包含两个 np.nan 值
    x = [12.0, 2.0, np.nan, 12.0, np.nan]
    
    # 计算新的 weightedtau 值
    tau, p_value = stats.weightedtau(x, y)
    
    # 断言：检查 tau 的近似值是否等于 -0.6615242347139803
    assert_approx_equal(tau, -0.6615242347139803)
    
    # 更新 y，将其包含多个 np.nan 值
    y = [np.nan, 4.0, 7.0, np.nan, np.nan]
    
    # 重新计算 weightedtau
    tau, p_value = stats.weightedtau(x, y)
    
    # 断言：检查 tau 的近似值是否等于 -0.6615242347139803
    assert_approx_equal(tau, -0.6615242347139803)
def test_segfault_issue_9710():
    # https://github.com/scipy/scipy/issues/9710
    # 用于检查段错误的测试
    # 在优化后的构建中调用两次函数后才能复制问题SEGFAULT
    stats.weightedtau([1], [1.0])
    stats.weightedtau([1], [1.0])
    # 下面的代码也导致了段错误
    stats.weightedtau([np.nan], [52])


def test_kendall_tau_large():
    n = 172
    # 测试省略策略
    x = np.arange(n + 1).astype(float)
    y = np.arange(n + 1).astype(float)
    y[-1] = np.nan
    _, pval = stats.kendalltau(x, y, method='exact', nan_policy='omit')
    assert_equal(pval, 0.0)


def test_weightedtau_vs_quadratic():
    # 简单的二次实现，所有参数都是必需的
    def wkq(x, y, rank, weigher, add):
        tot = conc = disc = u = v = 0
        for (i, j) in product(range(len(x)), range(len(x))):
            w = weigher(rank[i]) + weigher(rank[j]) if add \
                else weigher(rank[i]) * weigher(rank[j])
            tot += w
            if x[i] == x[j]:
                u += w
            if y[i] == y[j]:
                v += w
            if x[i] < x[j] and y[i] < y[j] or x[i] > x[j] and y[i] > y[j]:
                conc += w
            elif x[i] < x[j] and y[i] > y[j] or x[i] > x[j] and y[i] < y[j]:
                disc += w
        return (conc - disc) / np.sqrt(tot - u) / np.sqrt(tot - v)

    def weigher(x):
        return 1. / (x + 1)

    np.random.seed(42)
    for s in range(3,10):
        a = []
        # 生成带有重复项的排名
        for i in range(s):
            a += [i]*i
        b = list(a)
        np.random.shuffle(a)
        np.random.shuffle(b)
        # 第一次通过：使用元素索引作为排名
        rank = np.arange(len(a), dtype=np.intp)
        for _ in range(2):
            for add in [True, False]:
                expected = wkq(a, b, rank, weigher, add)
                actual = stats.weightedtau(a, b, rank, weigher, add).statistic
                assert_approx_equal(expected, actual)
            # 第二次通过：使用随机排名
            np.random.shuffle(rank)


class TestFindRepeats:

    def test_basic(self):
        a = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 5]
        res, nums = stats.find_repeats(a)
        assert_array_equal(res, [1, 2, 3, 4])
        assert_array_equal(nums, [3, 3, 2, 2])

    def test_empty_result(self):
        # 检查在没有重复项时返回空数组
        for a in [[10, 20, 50, 30, 40], []]:
            repeated, counts = stats.find_repeats(a)
            assert_array_equal(repeated, [])
            assert_array_equal(counts, [])


class TestRegression:

    def test_one_arg_deprecation(self):
        x = np.arange(20).reshape((2, 10))
        message = "Inference of the two sets..."
        with pytest.deprecated_call(match=message):
            stats.linregress(x)
        stats.linregress(x[0], x[1])
    # 定义一个测试函数，用于对大数据集 X 进行线性回归测试
    def test_linregressBIGX(self):
        # W.II.F.  Regress BIG on X.
        # 对 X 和 BIG 进行线性回归分析，返回回归结果
        result = stats.linregress(X, BIG)
        # 断言拟合结果的截距接近于 99999990
        assert_almost_equal(result.intercept, 99999990)
        # 断言相关系数接近于 1.0
        assert_almost_equal(result.rvalue, 1.0)
        # 由于所有数据点都在一条直线上，预期回归标准误差几乎为零
        assert_almost_equal(result.stderr, 0.0)
        # 截距的标准误差应该几乎为零
        assert_almost_equal(result.intercept_stderr, 0.0)

    # 定义一个测试函数，用于对 X 自身进行线性回归测试
    def test_regressXX(self):
        # W.IV.B.  Regress X on X.
        # 对 X 自身进行线性回归分析，计算回归结果
        # 截距应该完全为 0，回归系数应为 1。这是一个完全有效的回归，程序不应报错。
        result = stats.linregress(X, X)
        # 断言截距接近于 0.0
        assert_almost_equal(result.intercept, 0.0)
        # 断言相关系数接近于 1.0
        assert_almost_equal(result.rvalue, 1.0)
        # 两个点的回归标准误差应该为 0
        assert_almost_equal(result.stderr, 0.0)
        # 截距的标准误差应该为 0
        assert_almost_equal(result.intercept_stderr, 0.0)

        # W.IV.C. Regress X on BIG and LITTLE (two predictors).  The program
        # should tell you that this model is "singular" because BIG and
        # LITTLE are linear combinations of each other.  Cryptic error
        # messages are unacceptable here.  Singularity is the most
        # fundamental regression error.
        #
        # 需要找出如何处理多元线性回归。这并不明显

    # 定义一个测试函数，用于对 ZERO 和 X 进行线性回归测试
    def test_regressZEROX(self):
        # W.IV.D. Regress ZERO on X.
        # 程序应告知 ZERO 没有方差，或者继续计算回归并报告精确的相关性和总平方和为 0
        result = stats.linregress(X, ZERO)
        # 断言截距接近于 0.0
        assert_almost_equal(result.intercept, 0.0)
        # 断言相关系数接近于 0.0
        assert_almost_equal(result.rvalue, 0.0)

    # 定义一个简单的线性回归测试函数，用于对带有正弦噪声的数据进行回归测试
    def test_regress_simple(self):
        # Regress a line with sinusoidal noise.
        # 创建一个包含正弦噪声的简单线性回归测试数据
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))

        # 对数据 x 和 y 进行线性回归分析，返回回归结果
        result = stats.linregress(x, y)
        lr = LinregressResult
        # 断言结果是 LinregressResult 类型的实例
        assert_(isinstance(result, lr))
        # 断言回归的标准误差接近于给定值
        assert_almost_equal(result.stderr, 2.3957814497838803e-3)
    def test_regress_alternative(self):
        # 测试 alternative 参数
        x = np.linspace(0, 100, 100)  # 创建从 0 到 100 的间隔为 100 的数组 x
        y = 0.2 * np.linspace(0, 100, 100) + 10  # 斜率大于零的线性关系
        y += np.sin(np.linspace(0, 20, 100))  # 添加正弦波噪声

        with pytest.raises(ValueError, match="`alternative` must be 'less'..."):
            # 使用 pytest 检查是否会引发 ValueError，并匹配特定错误信息
            stats.linregress(x, y, alternative="ekki-ekki")

        res1 = stats.linregress(x, y, alternative="two-sided")

        # 斜率大于零，因此 "less" 的 p 值应该很大
        res2 = stats.linregress(x, y, alternative="less")
        assert_allclose(res2.pvalue, 1 - (res1.pvalue / 2))

        # 斜率大于零，因此 "greater" 的 p 值应该很小
        res3 = stats.linregress(x, y, alternative="greater")
        assert_allclose(res3.pvalue, res1.pvalue / 2)

        assert res1.rvalue == res2.rvalue == res3.rvalue

    def test_regress_against_R(self):
        # 与 R 的 `lm` 函数进行比较测试
        # options(digits=16)
        # x <- c(151, 174, 138, 186, 128, 136, 179, 163, 152, 131)
        # y <- c(63, 81, 56, 91, 47, 57, 76, 72, 62, 48)
        # relation <- lm(y~x)
        # print(summary(relation))

        x = [151, 174, 138, 186, 128, 136, 179, 163, 152, 131]
        y = [63, 81, 56, 91, 47, 57, 76, 72, 62, 48]
        res = stats.linregress(x, y, alternative="two-sided")
        # 使用 R 的 `lm` 函数给出的期望值进行断言比较
        assert_allclose(res.slope, 0.6746104491292)
        assert_allclose(res.intercept, -38.4550870760770)
        assert_allclose(res.rvalue, np.sqrt(0.95478224775))
        assert_allclose(res.pvalue, 1.16440531074e-06)
        assert_allclose(res.stderr, 0.0519051424731)
        assert_allclose(res.intercept_stderr, 8.0490133029927)

    # TODO: remove this test once single-arg support is dropped;
    # deprecation warning tested in `test_one_arg_deprecation`
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_regress_simple_onearg_rows(self):
        # 回归带有正弦噪声的直线，
        # 使用形状为 (2, N) 的单个输入
        x = np.linspace(0, 100, 100)
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        rows = np.vstack((x, y))

        result = stats.linregress(rows)
        assert_almost_equal(result.stderr, 2.3957814497838803e-3)
        assert_almost_equal(result.intercept_stderr, 1.3866936078570702e-1)

    # TODO: remove this test once single-arg support is dropped;
    # deprecation warning tested in `test_one_arg_deprecation`
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    # 测试简单线性回归，使用一个参数（列）进行测试
    def test_regress_simple_onearg_cols(self):
        # 生成从0到100的等间距数字序列，包含100个点
        x = np.linspace(0, 100, 100)
        # 生成 y = 0.2 * x + 10 的线性关系，并加上正弦波干扰
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        # 将 x 和 y 组合成二维数组，即两列数据
        columns = np.hstack((np.expand_dims(x, 1), np.expand_dims(y, 1)))

        # 对 columns 进行线性回归分析
        result = stats.linregress(columns)
        # 断言标准误差接近特定值
        assert_almost_equal(result.stderr, 2.3957814497838803e-3)
        # 断言截距标准误差接近特定值
        assert_almost_equal(result.intercept_stderr, 1.3866936078570702e-1)

    # TODO: 在单参数支持被移除时删除此测试；在 `test_one_arg_deprecation` 中测试弃用警告
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_regress_shape_error(self):
        # 检查对于形状错误的单个输入参数，linregress 应该引发 ValueError
        assert_raises(ValueError, stats.linregress, np.ones((3, 3)))

    def test_linregress(self):
        # 与使用 pinv 的多元最小二乘法进行比较
        x = np.arange(11)
        y = np.arange(5, 16)
        y[[(1), (-2)]] -= 1
        y[[(0), (-1)]] += 1

        # 对 x 和 y 进行线性回归分析
        result = stats.linregress(x, y)

        # 此测试曾经使用 'assert_array_almost_equal'，但由于 LinregressResult 变为
        # _lib._bunch._make_tuple_bunch，其形式变得令人困惑
        # （为了向后兼容性，请参见 PR #12983）
        def assert_ae(x, y):
            return assert_almost_equal(x, y, decimal=14)
        # 断言斜率接近1.0
        assert_ae(result.slope, 1.0)
        # 断言截距接近5.0
        assert_ae(result.intercept, 5.0)
        # 断言相关系数接近0.98229948625750
        assert_ae(result.rvalue, 0.98229948625750)
        # 断言 p 值接近7.45259691e-008
        assert_ae(result.pvalue, 7.45259691e-008)
        # 断言斜率标准误差接近0.063564172616372733
        assert_ae(result.stderr, 0.063564172616372733)
        # 断言截距标准误差接近0.37605071654517686
        assert_ae(result.intercept_stderr, 0.37605071654517686)

    def test_regress_simple_negative_cor(self):
        # 如果回归的斜率为负数，相关系数 R 倾向于 -1 而不是 1。有时由于四舍五入误差，R 可能 < -1，
        # 导致标准误差为 NaN。
        a, n = 1e-71, 100000
        # 生成从 a 到 2a 的等间距数字序列，包含 n 个点
        x = np.linspace(a, 2 * a, n)
        # 生成从 2a 到 a 的等间距数字序列，包含 n 个点
        y = np.linspace(2 * a, a, n)
        # 对 x 和 y 进行线性回归分析
        result = stats.linregress(x, y)

        # 确保传播的数值误差没有使得 rvalue 低于 -1（或被强制转换）
        assert_(result.rvalue >= -1)
        # 断言相关系数接近 -1
        assert_almost_equal(result.rvalue, -1)

        # 斜率和截距的标准误差应保持为数值型
        assert_(not np.isnan(result.stderr))
        assert_(not np.isnan(result.intercept_stderr))
    def test_linregress_result_attributes(self):
        # 生成一个包含 100 个等间隔点的数组作为 x 值
        x = np.linspace(0, 100, 100)
        # 创建一个线性关系的 y 值数组，加上正弦波干扰
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        # 对 x, y 进行线性回归分析
        result = stats.linregress(x, y)

        # 断言结果属于正确的类 LinregressResult
        lr = LinregressResult
        assert_(isinstance(result, lr))

        # 检查 LinregressResult 对象的元素是否具有正确的命名
        attributes = ('slope', 'intercept', 'rvalue', 'pvalue', 'stderr')
        check_named_results(result, attributes)
        # 同时检查是否存在额外的属性 intercept_stderr
        assert 'intercept_stderr' in dir(result)

    def test_regress_two_inputs(self):
        # 对由两个点形成的简单直线进行回归分析
        x = np.arange(2)
        y = np.arange(3, 5)
        result = stats.linregress(x, y)

        # 断言结果不是水平线
        assert_almost_equal(result.pvalue, 0.0)

        # 通过两个点拟合时，误差为零
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

    def test_regress_two_inputs_horizontal_line(self):
        # 对由两个点形成的水平线进行回归分析
        x = np.arange(2)
        y = np.ones(2)
        result = stats.linregress(x, y)

        # 断言结果是水平线
        assert_almost_equal(result.pvalue, 1.0)

        # 通过两个点拟合时，误差为零
        assert_almost_equal(result.stderr, 0.0)
        assert_almost_equal(result.intercept_stderr, 0.0)

    def test_nist_norris(self):
        # 使用 NIST Norris 数据集进行回归分析
        x = [0.2, 337.4, 118.2, 884.6, 10.1, 226.5, 666.3, 996.3, 448.6, 777.0,
             558.2, 0.4, 0.6, 775.5, 666.9, 338.0, 447.5, 11.6, 556.0, 228.1,
             995.8, 887.6, 120.2, 0.3, 0.3, 556.8, 339.1, 887.2, 999.0, 779.0,
             11.1, 118.3, 229.2, 669.1, 448.9, 0.5]
        y = [0.1, 338.8, 118.1, 888.0, 9.2, 228.1, 668.5, 998.5, 449.1, 778.9,
             559.2, 0.3, 0.1, 778.1, 668.8, 339.3, 448.9, 10.8, 557.7, 228.3,
             998.0, 888.8, 119.6, 0.3, 0.6, 557.6, 339.3, 888.0, 998.5, 778.9,
             10.2, 117.6, 228.9, 668.4, 449.2, 0.2]
        # 对数据进行线性回归分析
        result = stats.linregress(x, y)

        # 断言结果的斜率接近于已知值
        assert_almost_equal(result.slope, 1.00211681802045)
        # 断言结果的截距接近于已知值
        assert_almost_equal(result.intercept, -0.262323073774029)
        # 断言结果的相关系数平方接近于已知值
        assert_almost_equal(result.rvalue**2, 0.999993745883712)
        # 断言结果的 p 值接近于零
        assert_almost_equal(result.pvalue, 0.0)
        # 断言结果的标准误差接近于已知值
        assert_almost_equal(result.stderr, 0.00042979684820)
        # 断言结果的截距标准误差接近于已知值
        assert_almost_equal(result.intercept_stderr, 0.23281823430153)

    def test_compare_to_polyfit(self):
        # 生成一个包含 100 个等间隔点的数组作为 x 值
        x = np.linspace(0, 100, 100)
        # 创建一个线性关系的 y 值数组，加上正弦波干扰
        y = 0.2 * np.linspace(0, 100, 100) + 10
        y += np.sin(np.linspace(0, 20, 100))
        # 对 x, y 进行线性回归分析
        result = stats.linregress(x, y)
        # 使用 polyfit 函数拟合一次多项式
        poly = np.polyfit(x, y, 1)

        # 确保线性回归的斜率与 numpy polyfit 的结果匹配
        assert_almost_equal(result.slope, poly[0])
        # 确保线性回归的截距与 numpy polyfit 的结果匹配
        assert_almost_equal(result.intercept, poly[1])
    # 定义一个测试函数，测试空输入情况
    def test_empty_input(self):
        # 断言调用 `stats.linregress([], [])` 会抛出 ValueError 异常
        assert_raises(ValueError, stats.linregress, [], [])

    # 定义一个测试函数，测试输入包含 NaN 值的情况
    def test_nan_input(self):
        # 创建包含 0 到 9 的一维数组，并将第九个元素设置为 NaN
        x = np.arange(10.)
        x[9] = np.nan

        # 忽略 NaN 值的影响，计算线性回归结果
        with np.errstate(invalid="ignore"):
            result = stats.linregress(x, x)

        # 确保结果仍然是 `LinregressResult` 类型
        lr = LinregressResult
        assert_(isinstance(result, lr))
        
        # 断言结果数组中的所有元素都为 NaN
        assert_array_equal(result, (np.nan,)*5)
        
        # 断言截距的标准误差为 NaN
        assert_equal(result.intercept_stderr, np.nan)

    # 定义一个测试函数，测试输入 x 变量完全相同的情况
    def test_identical_x(self):
        # 创建一个长度为 10 的全零数组和一个随机数组作为输入
        x = np.zeros(10)
        y = np.random.random(10)
        
        # 设置错误消息
        msg = "Cannot calculate a linear regression"
        
        # 断言调用 `stats.linregress(x, y)` 会抛出 ValueError 异常，并匹配指定的错误消息
        with assert_raises(ValueError, match=msg):
            stats.linregress(x, y)
# 定义名为 test_theilslopes 的测试函数
def test_theilslopes():
    # 对 stats.theilslopes 函数进行基本斜率测试
    slope, intercept, lower, upper = stats.theilslopes([0,1,1])
    # 断言斜率接近于 0.5
    assert_almost_equal(slope, 0.5)
    # 断言截距接近于 0.5
    assert_almost_equal(intercept, 0.5)

    # 测试方法参数为 'joint_separate' 时是否引发 ValueError 异常
    msg = ("method must be either 'joint' or 'separate'."
           "'joint_separate' is invalid.")
    with pytest.raises(ValueError, match=msg):
        stats.theilslopes([0, 1, 1], method='joint_separate')

    # 使用 'joint' 方法对 stats.theilslopes 进行斜率测试
    slope, intercept, lower, upper = stats.theilslopes([0, 1, 1],
                                                       method='joint')
    # 断言斜率接近于 0.5
    assert_almost_equal(slope, 0.5)
    # 断言截距接近于 0.0
    assert_almost_equal(intercept, 0.0)

    # 对置信区间进行测试
    x = [1, 2, 3, 4, 10, 12, 18]
    y = [9, 15, 19, 20, 45, 55, 78]
    # 使用 'separate' 方法计算斜率、截距及置信区间
    slope, intercept, lower, upper = stats.theilslopes(y, x, 0.07,
                                                       method='separate')
    # 断言斜率接近于 4
    assert_almost_equal(slope, 4)
    # 断言截距接近于 4.0
    assert_almost_equal(intercept, 4.0)
    # 断言上置信区间接近于 4.38，精度为两位小数
    assert_almost_equal(upper, 4.38, decimal=2)
    # 断言下置信区间接近于 3.71，精度为两位小数
    assert_almost_equal(lower, 3.71, decimal=2)

    # 使用 'joint' 方法计算斜率、截距及置信区间
    slope, intercept, lower, upper = stats.theilslopes(y, x, 0.07,
                                                       method='joint')
    # 断言斜率接近于 4
    assert_almost_equal(slope, 4)
    # 断言截距接近于 6.0
    assert_almost_equal(intercept, 6.0)
    # 断言上置信区间接近于 4.38，精度为两位小数
    assert_almost_equal(upper, 4.38, decimal=2)
    # 断言下置信区间接近于 3.71，精度为两位小数
    assert_almost_equal(lower, 3.71, decimal=2)


# 定义名为 test_cumfreq 的测试函数
def test_cumfreq():
    # 测试累计频数函数 stats.cumfreq
    x = [1, 4, 2, 1, 3, 1]
    # 使用 numbins=4 计算累计频数
    cumfreqs, lowlim, binsize, extrapoints = stats.cumfreq(x, numbins=4)
    # 断言累计频数数组接近于给定数组
    assert_array_almost_equal(cumfreqs, np.array([3., 4., 5., 6.]))
    # 使用 numbins=4 和 defaultreallimits=(1.5, 5) 计算累计频数
    cumfreqs, lowlim, binsize, extrapoints = stats.cumfreq(
        x, numbins=4, defaultreallimits=(1.5, 5))
    # 断言 extrapoints 等于 3
    assert_(extrapoints == 3)

    # 测试 namedtuple 属性结果是否正确
    attributes = ('cumcount', 'lowerlimit', 'binsize', 'extrapoints')
    # 检查 stats.cumfreq 返回结果的 namedtuple 属性
    res = stats.cumfreq(x, numbins=4, defaultreallimits=(1.5, 5))
    check_named_results(res, attributes)


# 定义名为 test_relfreq 的测试函数
def test_relfreq():
    # 测试相对频数函数 stats.relfreq
    a = np.array([1, 4, 2, 1, 3, 1])
    # 使用 numbins=4 计算相对频数
    relfreqs, lowlim, binsize, extrapoints = stats.relfreq(a, numbins=4)
    # 断言相对频数数组接近于给定数组
    assert_array_almost_equal(relfreqs,
                              array([0.5, 0.16666667, 0.16666667, 0.16666667]))

    # 测试 namedtuple 属性结果是否正确
    attributes = ('frequency', 'lowerlimit', 'binsize', 'extrapoints')
    # 检查 stats.relfreq 返回结果的 namedtuple 属性
    res = stats.relfreq(a, numbins=4)
    check_named_results(res, attributes)

    # 检查接受 array_like 输入
    relfreqs2, lowlim, binsize, extrapoints = stats.relfreq([1, 4, 2, 1, 3, 1],
                                                            numbins=4)
    # 断言相对频数数组接近于 relfreqs2
    assert_array_almost_equal(relfreqs, relfreqs2)


# 定义名为 TestScoreatpercentile 的测试类
class TestScoreatpercentile:
    # 设置每个测试方法的初始化
    def setup_method(self):
        self.a1 = [3, 4, 5, 10, -3, -5, 6]  # 初始化测试数组 a1
        self.a2 = [3, -6, -2, 8, 7, 4, 2, 1]  # 初始化测试数组 a2
        self.a3 = [3., 4, 5, 10, -3, -5, -6, 7.0]  # 初始化测试数组 a3
    # 定义一个测试方法，用于测试 scoreatpercentile 函数的基本功能
    def test_basic(self):
        # 创建一个长度为8的数组，并每个元素乘以0.5，生成新数组x
        x = arange(8) * 0.5
        # 断言调用 scoreatpercentile 函数，期望得到 x 数组的第0百分位数为0.0
        assert_equal(stats.scoreatpercentile(x, 0), 0.)
        # 断言调用 scoreatpercentile 函数，期望得到 x 数组的第100百分位数为3.5
        assert_equal(stats.scoreatpercentile(x, 100), 3.5)
        # 断言调用 scoreatpercentile 函数，期望得到 x 数组的第50百分位数为1.75

    # 定义一个测试方法，用于测试 scoreatpercentile 函数在指定分位数和插值方法下的行为
    def test_fraction(self):
        # 创建 scoreatperc 变量，指向 stats.scoreatpercentile 函数的引用
        scoreatperc = stats.scoreatpercentile

        # 测试默认情况下 scoreatperc 函数的行为
        assert_equal(scoreatperc(list(range(10)), 50), 4.5)
        assert_equal(scoreatperc(list(range(10)), 50, (2,7)), 4.5)
        assert_equal(scoreatperc(list(range(100)), 50, limit=(1, 8)), 4.5)
        assert_equal(scoreatperc(np.array([1, 10,100]), 50, (10,100)), 55)
        assert_equal(scoreatperc(np.array([1, 10,100]), 50, (1,10)), 5.5)

        # 明确指定插值方法为 'fraction'（默认值）
        assert_equal(scoreatperc(list(range(10)), 50, interpolation_method='fraction'),
                     4.5)
        assert_equal(scoreatperc(list(range(10)), 50, limit=(2, 7),
                                 interpolation_method='fraction'),
                     4.5)
        assert_equal(scoreatperc(list(range(100)), 50, limit=(1, 8),
                                 interpolation_method='fraction'),
                     4.5)
        assert_equal(scoreatperc(np.array([1, 10,100]), 50, (10, 100),
                                 interpolation_method='fraction'),
                     55)
        assert_equal(scoreatperc(np.array([1, 10,100]), 50, (1,10),
                                 interpolation_method='fraction'),
                     5.5)

    # 定义一个测试方法，用于测试 scoreatpercentile 函数在不同插值方法下的行为
    def test_lower_higher(self):
        # 创建 scoreatperc 变量，指向 stats.scoreatpercentile 函数的引用
        scoreatperc = stats.scoreatpercentile

        # 测试插值方法为 'lower'/'higher' 的情况
        assert_equal(scoreatperc(list(range(10)), 50,
                                 interpolation_method='lower'), 4)
        assert_equal(scoreatperc(list(range(10)), 50,
                                 interpolation_method='higher'), 5)
        assert_equal(scoreatperc(list(range(10)), 50, (2,7),
                                 interpolation_method='lower'), 4)
        assert_equal(scoreatperc(list(range(10)), 50, limit=(2,7),
                                 interpolation_method='higher'), 5)
        assert_equal(scoreatperc(list(range(100)), 50, (1,8),
                                 interpolation_method='lower'), 4)
        assert_equal(scoreatperc(list(range(100)), 50, (1,8),
                                 interpolation_method='higher'), 5)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (10, 100),
                                 interpolation_method='lower'), 10)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, limit=(10, 100),
                                 interpolation_method='higher'), 100)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, (1, 10),
                                 interpolation_method='lower'), 1)
        assert_equal(scoreatperc(np.array([1, 10, 100]), 50, limit=(1, 10),
                                 interpolation_method='higher'), 10)
    # 定义测试方法：测试 stats.scoreatpercentile 函数在序列上的表现
    def test_sequence_per(self):
        # 创建长度为 8 的数组 x，元素为 [0.0, 0.5, 1.0, ..., 3.5]
        x = arange(8) * 0.5
        # 预期结果是一个 numpy 数组 [0, 3.5, 1.75]
        expected = np.array([0, 3.5, 1.75])
        # 调用 stats.scoreatpercentile 函数，计算 x 在百分位 [0, 100, 50] 处的分位数
        res = stats.scoreatpercentile(x, [0, 100, 50])
        # 断言 res 与预期结果 expected 的各项近似相等
        assert_allclose(res, expected)
        # 断言 res 是 numpy 数组的实例
        assert_(isinstance(res, np.ndarray))
        
        # Regression 测试，使用 ndarray 作为输入。检测 gh-2861 的回归问题
        # 再次调用 stats.scoreatpercentile 函数，传入 numpy 数组 [0, 100, 50] 作为百分位数
        assert_allclose(stats.scoreatpercentile(x, np.array([0, 100, 50])),
                        expected)
        
        # 同时测试 2-D 数组、指定轴和类数组的情况
        # 调用 stats.scoreatpercentile 函数，计算 12 个元素的数组在每行上百分位 [0, 1, 100, 100] 处的分位数
        res2 = stats.scoreatpercentile(np.arange(12).reshape((3,4)),
                                       np.array([0, 1, 100, 100]), axis=1)
        # 预期结果是一个 2-D 数组
        expected2 = array([[0, 4, 8],
                           [0.03, 4.03, 8.03],
                           [3, 7, 11],
                           [3, 7, 11]])
        # 断言 res2 与 expected2 的各项近似相等
        assert_allclose(res2, expected2)

    # 定义测试方法：测试 stats.scoreatpercentile 函数在不同轴上的表现
    def test_axis(self):
        # 将 stats.scoreatpercentile 函数赋给 scoreatperc 变量
        scoreatperc = stats.scoreatpercentile
        # 创建一个 3x4 的数组 x
        x = arange(12).reshape(3, 4)

        # 断言 x 在整体上百分位 [25, 50, 100] 处的分位数
        assert_equal(scoreatperc(x, (25, 50, 100)), [2.75, 5.5, 11.0])

        # 预期在轴 0 上的结果 r0
        r0 = [[2, 3, 4, 5], [4, 5, 6, 7], [8, 9, 10, 11]]
        # 断言 x 在轴 0 上百分位 [25, 50, 100] 处的分位数
        assert_equal(scoreatperc(x, (25, 50, 100), axis=0), r0)

        # 预期在轴 1 上的结果 r1
        r1 = [[0.75, 4.75, 8.75], [1.5, 5.5, 9.5], [3, 7, 11]]
        # 断言 x 在轴 1 上百分位 [25, 50, 100] 处的分位数
        assert_equal(scoreatperc(x, (25, 50, 100), axis=1), r1)

        # 创建一个新的数组 x
        x = array([[1, 1, 1],
                   [1, 1, 1],
                   [4, 4, 3],
                   [1, 1, 1],
                   [1, 1, 1]])
        # 计算 x 在百分位 50 处的分位数
        score = stats.scoreatpercentile(x, 50)
        # 断言 score 的形状为空
        assert_equal(score.shape, ())
        # 断言 score 等于 1.0
        assert_equal(score, 1.0)
        
        # 计算 x 在轴 0 上百分位 50 处的分位数
        score = stats.scoreatpercentile(x, 50, axis=0)
        # 断言 score 的形状为 (3,)
        assert_equal(score.shape, (3,))
        # 断言 score 等于 [1, 1, 1]
        assert_equal(score, [1, 1, 1])

    # 定义测试方法：测试 stats.scoreatpercentile 函数在异常情况下的行为
    def test_exception(self):
        # 断言调用 stats.scoreatpercentile 函数时遇到 ValueError 异常
        assert_raises(ValueError, stats.scoreatpercentile, [1, 2], 56,
                      interpolation_method='foobar')
        # 断言调用 stats.scoreatpercentile 函数时遇到 ValueError 异常
        assert_raises(ValueError, stats.scoreatpercentile, [1], 101)
        # 断言调用 stats.scoreatpercentile 函数时遇到 ValueError 异常
        assert_raises(ValueError, stats.scoreatpercentile, [1], -1)

    # 定义测试方法：测试 stats.scoreatpercentile 函数在空输入下的行为
    def test_empty(self):
        # 断言 stats.scoreatpercentile 函数对空数组返回 np.nan
        assert_equal(stats.scoreatpercentile([], 50), np.nan)
        # 断言 stats.scoreatpercentile 函数对空的 2-D 数组返回 np.nan
        assert_equal(stats.scoreatpercentile(np.array([[], []]), 50), np.nan)
        # 断言 stats.scoreatpercentile 函数对空数组列表返回 [np.nan, np.nan]
        assert_equal(stats.scoreatpercentile([], [50, 99]), [np.nan, np.nan])
# 定义一个名为 TestMode 的测试类
class TestMode:

    # 测试空列表作为输入时的情况
    def test_empty(self):
        # 使用 pytest 的 warns 方法检查是否会引发 SmallSampleWarning，匹配给定的警告消息
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            # 调用 stats.mode 方法计算空列表的众数和计数
            vals, counts = stats.mode([])
        # 断言返回的众数是空的 NumPy 数组
        assert_equal(vals, np.array([]))
        # 断言返回的计数也是空的 NumPy 数组
        assert_equal(counts, np.array([]))

    # 测试输入为标量时的情况
    def test_scalar(self):
        # 调用 stats.mode 方法计算标量值的众数和计数
        vals, counts = stats.mode(4.)
        # 断言返回的众数是包含单个元素 4.0 的 NumPy 数组
        assert_equal(vals, np.array([4.]))
        # 断言返回的计数是包含数字 1 的 NumPy 数组
        assert_equal(counts, np.array([1]))

    # 测试基本的数据列表输入
    def test_basic(self):
        # 定义一个包含整数的列表
        data1 = [3, 5, 1, 10, 23, 3, 2, 6, 8, 6, 10, 6]
        # 调用 stats.mode 方法计算 data1 的众数和计数
        vals = stats.mode(data1)
        # 断言返回的众数是 6
        assert_equal(vals[0], 6)
        # 断言返回的计数是 3
        assert_equal(vals[1], 3)

    # 测试指定轴的多维数组输入
    def test_axes(self):
        # 定义几个包含整数的列表
        data1 = [10, 10, 30, 40]
        data2 = [10, 10, 10, 10]
        data3 = [20, 10, 20, 20]
        data4 = [30, 30, 30, 30]
        data5 = [40, 30, 30, 30]
        # 将这些列表组成 NumPy 数组 arr
        arr = np.array([data1, data2, data3, data4, data5])

        # 在 axis=None 的情况下计算 arr 的众数和计数，保持维度
        vals = stats.mode(arr, axis=None, keepdims=True)
        assert_equal(vals[0], np.array([[30]]))
        assert_equal(vals[1], np.array([[8]]))

        # 在 axis=0 的情况下计算 arr 的众数和计数，保持维度
        vals = stats.mode(arr, axis=0, keepdims=True)
        assert_equal(vals[0], np.array([[10, 10, 30, 30]]))
        assert_equal(vals[1], np.array([[2, 3, 3, 2]]))

        # 在 axis=1 的情况下计算 arr 的众数和计数，保持维度
        vals = stats.mode(arr, axis=1, keepdims=True)
        assert_equal(vals[0], np.array([[10], [10], [20], [30], [30]]))
        assert_equal(vals[1], np.array([[2], [4], [3], [4], [3]]))

    # 使用参数化测试对负轴的情况进行测试
    @pytest.mark.parametrize('axis', np.arange(-4, 0))
    def test_negative_axes_gh_15375(self, axis):
        # 设置随机种子确保结果可重复
        np.random.seed(984213899)
        # 生成一个随机数组 a
        a = np.random.rand(10, 11, 12, 13)
        # 对于 axis=a.ndim+axis 的负轴，计算数组 a 的众数
        res0 = stats.mode(a, axis=a.ndim+axis)
        # 对于 axis 的负值，计算数组 a 的众数
        res1 = stats.mode(a, axis=axis)
        # 使用 np.testing.assert_array_equal 断言两次计算结果相等
        np.testing.assert_array_equal(res0, res1)

    # 测试 mode 函数返回结果的属性
    def test_mode_result_attributes(self):
        # 定义一个包含整数的列表
        data1 = [3, 5, 1, 10, 23, 3, 2, 6, 8, 6, 10, 6]
        # 定义一个空列表
        data2 = []
        # 计算 data1 的众数和计数，并检查返回结果的命名属性
        actual = stats.mode(data1)
        attributes = ('mode', 'count')
        check_named_results(actual, attributes)
        # 检查对空列表 data2 调用 mode 函数会引发 SmallSampleWarning 警告
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            actual2 = stats.mode(data2)
        # 再次检查 mode 函数返回结果的命名属性
        check_named_results(actual2, attributes)

    # 测试包含 NaN 值的列表输入
    def test_mode_nan(self):
        # 定义一个包含 NaN 值的列表
        data1 = [3, np.nan, 5, 1, 10, 23, 3, 2, 6, 8, 6, 10, 6]
        # 计算包含 NaN 值的 data1 的众数和计数
        actual = stats.mode(data1)
        assert_equal(actual, (6, 3))

        # 使用 nan_policy='omit' 参数计算 data1 的众数和计数
        actual = stats.mode(data1, nan_policy='omit')
        assert_equal(actual, (6, 3))

        # 使用 nan_policy='raise' 参数调用 mode 函数应该引发 ValueError
        assert_raises(ValueError, stats.mode, data1, nan_policy='raise')

        # 使用未知的 nan_policy='foobar' 参数调用 mode 函数应该引发 ValueError
        assert_raises(ValueError, stats.mode, data1, nan_policy='foobar')

    # 对一系列输入数据进行参数化测试
    @pytest.mark.parametrize("data", [
        [3, 5, 1, 1, 3],
        [3, np.nan, 5, 1, 1, 3],
        [3, 5, 1],
        [3, np.nan, 5, 1],
    ])
    # 参数化 keepdims 参数测试
    @pytest.mark.parametrize('keepdims', [False, True])
    def test_smallest_equal(self, data, keepdims):
        # 使用 nan_policy='omit' 和指定的 keepdims 参数计算 data 的众数
        result = stats.mode(data, nan_policy='omit', keepdims=keepdims)
        # 如果 keepdims=True，则断言结果的众数是包含单个元素 1 的 NumPy 数组
        if keepdims:
            assert_equal(result[0][0], 1)
        else:
            # 否则，断言结果的众数是 1
            assert_equal(result[0], 1)

    # 参数化轴的负值进行测试
    @pytest.mark.parametrize('axis', np.arange(-3, 3))
    # 定义一个测试方法，用于检查 stats 模块中 mode 函数的行为
    def test_mode_shape_gh_9955(self, axis, dtype=np.float64):
        # 使用指定种子创建随机数生成器 rng
        rng = np.random.default_rng(984213899)
        # 创建一个指定形状和数据类型的随机数组 a
        a = rng.uniform(size=(3, 4, 5)).astype(dtype)
        # 调用 stats 模块中的 mode 函数计算数组 a 沿指定轴的众数及其出现次数
        res = stats.mode(a, axis=axis, keepdims=False)
        # 生成参考形状，去除指定轴的长度
        reference_shape = list(a.shape)
        reference_shape.pop(axis)
        # 使用 np.testing.assert_array_equal 断言 res.mode 的形状与参考形状相同
        np.testing.assert_array_equal(res.mode.shape, reference_shape)
        # 使用 np.testing.assert_array_equal 断言 res.count 的形状与参考形状相同
        np.testing.assert_array_equal(res.count.shape, reference_shape)

    # 定义一个测试方法，验证 mode 函数在 nan_policy='propagate' 时对 np.nan 的处理
    def test_nan_policy_propagate_gh_9815(self):
        # 创建包含 NaN 值的列表 a
        # mode 函数应该将 np.nan 看作普通对象处理，当 nan_policy='propagate' 时
        a = [2, np.nan, 1, np.nan]
        # 调用 stats 模块中的 mode 函数计算数组 a 的众数及其出现次数
        res = stats.mode(a)
        # 使用断言检查 res.mode 是否为 NaN，且 res.count 是否等于 2
        assert np.isnan(res.mode) and res.count == 2
    def test_keepdims(self):
        # test empty arrays (handled by `np.mean`)
        # 创建一个形状为 (1, 2, 3, 0) 的全零数组
        a = np.zeros((1, 2, 3, 0))

        # 计算沿着 axis=1 的众数，不保持维度
        res = stats.mode(a, axis=1, keepdims=False)
        # 断言结果的众数和计数的形状为 (1, 3, 0)
        assert res.mode.shape == res.count.shape == (1, 3, 0)

        # 计算沿着 axis=1 的众数，保持维度
        res = stats.mode(a, axis=1, keepdims=True)
        # 断言结果的众数和计数的形状为 (1, 1, 3, 0)
        assert res.mode.shape == res.count.shape == (1, 1, 3, 0)

        # test nan_policy='propagate'
        # 创建包含 NaN 的二维列表
        a = [[1, 3, 3, np.nan], [1, 1, np.nan, 1]]

        # 计算沿着 axis=1 的众数，不保持维度
        res = stats.mode(a, axis=1, keepdims=False)
        # 断言结果的众数为 [3, 1]，计数为 [2, 3]
        assert_array_equal(res.mode, [3, 1])
        assert_array_equal(res.count, [2, 3])

        # 计算沿着 axis=1 的众数，保持维度
        res = stats.mode(a, axis=1, keepdims=True)
        # 断言结果的众数为 [[3], [1]]，计数为 [[2], [3]]
        assert_array_equal(res.mode, [[3], [1]])
        assert_array_equal(res.count, [[2], [3]])

        # 将列表转换为 NumPy 数组
        a = np.array(a)
        # 计算沿着全局（flatten）的众数，不保持维度
        res = stats.mode(a, axis=None, keepdims=False)
        ref = stats.mode(a.ravel(), keepdims=False)
        # 断言结果与参考结果相等
        assert_array_equal(res, ref)
        # 断言结果的众数的形状为 ()，与参考结果一致
        assert res.mode.shape == ref.mode.shape == ()

        # 计算沿着全局（flatten）的众数，保持维度
        res = stats.mode(a, axis=None, keepdims=True)
        ref = stats.mode(a.ravel(), keepdims=True)
        # 断言结果的众数与参考结果相等
        assert_equal(res.mode.ravel(), ref.mode.ravel())
        # 断言结果的众数形状为 (1, 1)，与参考结果一致
        assert res.mode.shape == (1, 1)
        # 断言结果的计数与参考结果相等
        assert_equal(res.count.ravel(), ref.count.ravel())
        # 断言结果的计数形状为 (1, 1)，与参考结果一致
        assert res.count.shape == (1, 1)

        # test nan_policy='omit'
        # 创建包含 NaN 的二维列表
        a = [[1, np.nan, np.nan, np.nan, 1],
             [np.nan, np.nan, np.nan, np.nan, 2],
             [1, 2, np.nan, 5, 5]]

        # 计算沿着 axis=1 的众数，不保持维度，NaN 策略为 'omit'
        res = stats.mode(a, axis=1, keepdims=False, nan_policy='omit')
        # 断言结果的众数为 [1, 2, 5]，计数为 [2, 1, 2]
        assert_array_equal(res.mode, [1, 2, 5])
        assert_array_equal(res.count, [2, 1, 2])

        # 计算沿着 axis=1 的众数，保持维度，NaN 策略为 'omit'
        res = stats.mode(a, axis=1, keepdims=True, nan_policy='omit')
        # 断言结果的众数为 [[1], [2], [5]]，计数为 [[2], [1], [2]]
        assert_array_equal(res.mode, [[1], [2], [5]])
        assert_array_equal(res.count, [[2], [1], [2]])

        # 将列表转换为 NumPy 数组
        a = np.array(a)
        # 计算沿着全局（flatten）的众数，不保持维度，NaN 策略为 'omit'
        res = stats.mode(a, axis=None, keepdims=False, nan_policy='omit')
        ref = stats.mode(a.ravel(), keepdims=False, nan_policy='omit')
        # 断言结果与参考结果相等
        assert_array_equal(res, ref)
        # 断言结果的众数形状为 ()，与参考结果一致
        assert res.mode.shape == ref.mode.shape == ()

        # 计算沿着全局（flatten）的众数，保持维度，NaN 策略为 'omit'
        res = stats.mode(a, axis=None, keepdims=True, nan_policy='omit')
        ref = stats.mode(a.ravel(), keepdims=True, nan_policy='omit')
        # 断言结果的众数与参考结果相等
        assert_equal(res.mode.ravel(), ref.mode.ravel())
        # 断言结果的众数形状为 (1, 1)，与参考结果一致
        assert res.mode.shape == (1, 1)
        # 断言结果的计数与参考结果相等
        assert_equal(res.count.ravel(), ref.count.ravel())
        # 断言结果的计数形状为 (1, 1)，与参考结果一致
        assert res.count.shape == (1, 1)
    def test_gh16955(self, nan_policy):
        # Check that bug reported in gh-16955 is resolved
        # 定义一个形状为 (4, 3) 的数组
        shape = (4, 3)
        # 创建一个所有元素为 1 的数组
        data = np.ones(shape)
        # 将数组中第一个元素设为 NaN
        data[0, 0] = np.nan
        # 使用 stats.mode 函数计算数组的众数，指定轴为 1，保持维度为 False，nan 策略使用传入的参数
        res = stats.mode(a=data, axis=1, keepdims=False, nan_policy=nan_policy)
        # 断言计算得到的众数为 [1, 1, 1, 1]
        assert_array_equal(res.mode, [1, 1, 1, 1])
        # 断言计算得到的众数的数量为 [2, 3, 3, 3]
        assert_array_equal(res.count, [2, 3, 3, 3])

        # 测试来自 gh-16595 的输入。不支持非数值输入已被弃用，检查相应的错误。
        # 定义一个结构化数据类型 my_dtype
        my_dtype = np.dtype([('asdf', np.uint8), ('qwer', np.float64, (3,))])
        # 创建一个 dtype 为 my_dtype 的长度为 10 的零数组
        test = np.zeros(10, dtype=my_dtype)
        # 定义一个错误消息，用于检查 TypeError
        message = "Argument `a` is not....|An argument has dtype..."
        # 使用 pytest 来断言 stats.mode 函数对 test 的调用会抛出 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=message):
            stats.mode(test, nan_policy=nan_policy)

    def test_gh9955(self):
        # The behavior of mode with empty slices (whether the input was empty
        # or all elements were omitted) was inconsistent. Test that this is
        # resolved: the mode of an empty slice is NaN and the count is zero.
        # 使用 pytest 来测试 stats.mode 对空数组的行为是否正确处理
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            res = stats.mode([])
        # 定义参考值 ref，空数组的众数应为 NaN，计数应为 0
        ref = (np.nan, 0)
        # 断言计算得到的结果 res 与参考值 ref 相等
        assert_equal(res, ref)

        # 测试 nan_policy='omit' 情况下对包含 np.nan 的数组的处理
        with pytest.warns(SmallSampleWarning, match=too_small_1d_omit):
            res = stats.mode([np.nan], nan_policy='omit')
        # 断言计算得到的结果 res 与参考值 ref 相等
        assert_equal(res, ref)

        # 定义一个二维数组 a
        a = [[10., 20., 20.], [np.nan, np.nan, np.nan]]
        # 测试 nan_policy='omit' 情况下对二维数组的处理
        with pytest.warns(SmallSampleWarning, match=too_small_nd_omit):
            res = stats.mode(a, axis=1, nan_policy='omit')
        # 定义参考值 ref，众数应为 [20, np.nan]，计数应为 [2, 0]
        ref = ([20, np.nan], [2, 0])
        # 断言计算得到的结果 res 与参考值 ref 相等
        assert_equal(res, ref)

        # 测试 nan_policy='propagate' 情况下对二维数组的处理
        res = stats.mode(a, axis=1, nan_policy='propagate')
        # 定义参考值 ref，众数应为 [20, np.nan]，计数应为 [2, 3]
        ref = ([20, np.nan], [2, 3])
        # 断言计算得到的结果 res 与参考值 ref 相等
        assert_equal(res, ref)

        # 定义一个空的二维数组 z
        z = np.array([[], []])
        # 测试对空的二维数组 z 的处理
        with pytest.warns(SmallSampleWarning, match=too_small_nd_not_omit):
            res = stats.mode(z, axis=1)
        # 定义参考值 ref，众数应为 [np.nan, np.nan]，计数应为 [0, 0]
        ref = ([np.nan, np.nan], [0, 0])
        # 断言计算得到的结果 res 与参考值 ref 相等
        assert_equal(res, ref)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')  # np.mean warns
    @pytest.mark.parametrize('z', [np.empty((0, 1, 2)), np.empty((1, 1, 2))])
    def test_gh17214(self, z):
        # 如果 z 的大小为 0
        if z.size == 0:
            # 使用 pytest 来测试对空数组的处理，预期会产生 SmallSampleWarning 警告
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                res = stats.mode(z, axis=None, keepdims=True)
        else:
            # 对非空数组 z 进行 stats.mode 计算，不产生警告
            res = stats.mode(z, axis=None, keepdims=True)
        # 计算数组 z 的均值，用作参考值
        ref = np.mean(z, axis=None, keepdims=True)
        # 断言计算得到的结果 res[0] 的形状与 res[1] 的形状与参考值 ref 的形状都为 (1, 1, 1)
        assert res[0].shape == res[1].shape == ref.shape == (1, 1, 1)
    # 定义一个测试函数，用于测试引发非数值类型的异常（GitHub Issue #18254）
    def test_raise_non_numeric_gh18254(self):
        # 根据 SCIPY_ARRAY_API 变量选择不同的错误消息
        message = ("...only boolean and numerical dtypes..." if SCIPY_ARRAY_API
                   else "Argument `a` is not recognized as numeric.")

        # 定义一个类 ArrLike，模拟类似数组的对象
        class ArrLike:
            def __init__(self, x):
                self._x = x

            # 定义 __array__ 方法，使得该类实例能够表现为一个数组
            def __array__(self, dtype=None, copy=None):
                return self._x.astype(object)

        # 使用 pytest 的 assertRaises 来检测是否引发了 TypeError 异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=message):
            # 调用 stats.mode 函数，传入 ArrLike 类的实例，期望引发 TypeError 异常
            stats.mode(ArrLike(np.arange(3)))

        # 再次使用 pytest 的 assertRaises 来检测是否引发了 TypeError 异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=message):
            # 调用 stats.mode 函数，传入一个 dtype=object 的 numpy 数组，期望引发 TypeError 异常
            stats.mode(np.arange(3, dtype=object)))
@array_api_compatible
class TestSEM:
    # 定义类变量 `testcase`，包含一个测试用例数组
    testcase = [1., 2., 3., 4.]
    # 定义类变量 `scalar_testcase`，包含一个标量测试用例
    scalar_testcase = 4.

    def test_sem_scalar(self, xp):
        # 对标量进行 SEM（标准误）计算测试

        # 将标量测试用例转换为对应的数组形式
        scalar_testcase = xp.asarray(self.scalar_testcase)[()]

        # 如果使用的是 NumPy，预期会发出 `SmallSampleWarning` 警告，匹配特定的警告消息
        if is_numpy(xp):
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                y = stats.sem(scalar_testcase)
        else:
            # 对于其他类型的数组，会忽略特定类型的警告
            with np.testing.suppress_warnings() as sup:
                sup.filter(UserWarning)
                sup.filter(RuntimeWarning)
                y = stats.sem(scalar_testcase)

        # 断言结果应为 NaN
        assert xp.isnan(y)

    def test_sem(self, xp):
        # 对数组进行 SEM 计算测试

        # 将测试用例转换为对应的数组形式
        testcase = xp.asarray(self.testcase)

        # 计算 SEM，并断言结果与预期接近
        y = stats.sem(testcase)
        xp_assert_close(y, xp.asarray(0.6454972244))

        # 验证 SEM 的计算公式
        n = len(self.testcase)
        xp_assert_close(stats.sem(testcase, ddof=0) * (n/(n-2))**0.5,
                        stats.sem(testcase, ddof=2))

        # 创建包含 NaN 的数组，并验证 SEM 计算结果应为 NaN
        x = xp.arange(10.)
        x = xp.where(x == 9, xp.asarray(xp.nan), x)
        xp_assert_equal(stats.sem(x), xp.asarray(xp.nan))

    @skip_xp_backends(np_only=True,
                      reasons=['`nan_policy` only supports NumPy backend'])
    def test_sem_nan_policy(self, xp):
        # 测试 SEM 计算中的 NaN 策略

        # 创建包含 NaN 的数组
        x = np.arange(10.)
        x[9] = np.nan

        # 使用 'omit' 策略计算 SEM，并断言结果
        assert_equal(stats.sem(x, nan_policy='omit'), 0.9128709291752769)

        # 使用不支持的 NaN 策略应引发 ValueError 异常
        assert_raises(ValueError, stats.sem, x, nan_policy='raise')
        assert_raises(ValueError, stats.sem, x, nan_policy='foobar')


class TestZmapZscore:

    @pytest.mark.parametrize(
        'x, y',
        [([1, 2, 3, 4], [1, 2, 3, 4]),
         ([1, 2, 3], [0, 1, 2, 3, 4])]
    )
    def test_zmap(self, x, y):
        # 测试 zmap 函数

        # 调用 zmap 函数计算结果
        z = stats.zmap(x, y)

        # 对于简单的情况，直接计算预期结果（z-score）
        expected = (x - np.mean(y))/np.std(y)

        # 断言计算结果与预期接近
        assert_allclose(z, expected, rtol=1e-12)

    def test_zmap_axis(self):
        # 测试 zmap 函数中的 'axis' 关键字参数

        # 创建二维数组作为输入
        x = np.array([[0.0, 0.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 2.0],
                      [2.0, 0.0, 2.0, 0.0]])

        # 预期的结果值
        t1 = 1.0/np.sqrt(2.0/3)
        t2 = np.sqrt(3.)/3
        t3 = np.sqrt(2.)

        # 在不同的轴上调用 zmap 函数，并断言结果数组与预期接近
        z0 = stats.zmap(x, x, axis=0)
        z1 = stats.zmap(x, x, axis=1)

        z0_expected = [[-t1, -t3/2, -t3/2, 0.0],
                       [0.0, t3, -t3/2, t1],
                       [t1, -t3/2, t3, -t1]]
        z1_expected = [[-1.0, -1.0, 1.0, 1.0],
                       [-t2, -t2, -t2, np.sqrt(3.)],
                       [1.0, -1.0, 1.0, -1.0]]

        assert_array_almost_equal(z0, z0_expected)
        assert_array_almost_equal(z1, z1_expected)
    def test_zmap_ddof(self):
        # 测试在 zmap 中使用 'ddof' 关键字。
        x = np.array([[0.0, 0.0, 1.0, 1.0],
                      [0.0, 1.0, 2.0, 3.0]])

        # 调用 zmap 函数计算 z 值，指定计算轴为 1，并设定 ddof=1
        z = stats.zmap(x, x, axis=1, ddof=1)

        # 计算预期的 z0 和 z1 值
        z0_expected = np.array([-0.5, -0.5, 0.5, 0.5])/(1.0/np.sqrt(3))
        z1_expected = np.array([-1.5, -0.5, 0.5, 1.5])/(np.sqrt(5./3))

        # 断言 z[0] 和 z[1] 与预期值 z0_expected 和 z1_expected 接近
        assert_array_almost_equal(z[0], z0_expected)
        assert_array_almost_equal(z[1], z1_expected)

    @pytest.mark.parametrize('ddof', [0, 2])
    def test_zmap_nan_policy_omit(self, ddof):
        # 在 zmap 中测试 nan_policy='omit'，此时忽略 scores 中的 NaN 值
        scores = np.array([-3, -1, 2, np.nan])
        compare = np.array([-8, -3, 2, 7, 12, np.nan])

        # 调用 zmap 函数，比较 scores 和 compare，设定 ddof=ddof 和 nan_policy='omit'
        z = stats.zmap(scores, compare, ddof=ddof, nan_policy='omit')

        # 断言 z 与忽略了 compare 中 NaN 值后的 zmap 计算结果接近
        assert_allclose(z, stats.zmap(scores, compare[~np.isnan(compare)],
                                      ddof=ddof))

    @pytest.mark.parametrize('ddof', [0, 2])
    def test_zmap_nan_policy_omit_with_axis(self, ddof):
        # 在 zmap 中测试 nan_policy='omit' 和 axis=1，忽略 compare 中的 NaN 值
        scores = np.arange(-5.0, 9.0).reshape(2, -1)
        compare = np.linspace(-8, 6, 24).reshape(2, -1)
        compare[0, 4] = np.nan
        compare[0, 6] = np.nan
        compare[1, 1] = np.nan

        # 调用 zmap 函数，比较 scores 和 compare，设定 nan_policy='omit'、axis=1 和 ddof=ddof
        z = stats.zmap(scores, compare, nan_policy='omit', axis=1, ddof=ddof)

        # 计算预期的 z 值数组
        expected = np.array([stats.zmap(scores[0],
                                        compare[0][~np.isnan(compare[0])],
                                        ddof=ddof),
                             stats.zmap(scores[1],
                                        compare[1][~np.isnan(compare[1])],
                                        ddof=ddof)])

        # 断言 z 与预期的 z 值数组接近
        assert_allclose(z, expected, rtol=1e-14)

    def test_zmap_nan_policy_raise(self):
        # 测试在 zmap 中使用 nan_policy='raise'，当输入包含 NaN 时抛出 ValueError 异常
        scores = np.array([1, 2, 3])
        compare = np.array([-8, -3, 2, 7, 12, np.nan])

        # 使用 pytest 的断言，验证调用 zmap 函数时抛出 ValueError 异常，异常信息匹配 'input contains nan'
        with pytest.raises(ValueError, match='input contains nan'):
            stats.zmap(scores, compare, nan_policy='raise')

    def test_zscore(self):
        # 在 zscore 中进行测试，不同于 R 中的实现，使用以下方式进行验证：
        #    (testcase[i] - mean(testcase, axis=0)) / sqrt(var(testcase) * 3/4)
        y = stats.zscore([1, 2, 3, 4])

        # 预期的标准化结果
        desired = ([-1.3416407864999, -0.44721359549996, 0.44721359549996,
                    1.3416407864999])

        # 断言计算得到的 y 与预期的标准化结果接近
        assert_array_almost_equal(desired, y, decimal=12)
    def test_zscore_axis(self):
        # 测试 zscore 中 'axis' 关键字的使用。
        # 创建一个二维 NumPy 数组
        x = np.array([[0.0, 0.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 2.0],
                      [2.0, 0.0, 2.0, 0.0]])

        # 计算预期的标准化结果的常数
        t1 = 1.0 / np.sqrt(2.0 / 3)
        t2 = np.sqrt(3.) / 3
        t3 = np.sqrt(2.)

        # 在 axis=0 上计算 zscore
        z0 = stats.zscore(x, axis=0)
        # 在 axis=1 上计算 zscore
        z1 = stats.zscore(x, axis=1)

        # 预期的标准化结果
        z0_expected = [[-t1, -t3/2, -t3/2, 0.0],
                       [0.0, t3, -t3/2, t1],
                       [t1, -t3/2, t3, -t1]]
        z1_expected = [[-1.0, -1.0, 1.0, 1.0],
                       [-t2, -t2, -t2, np.sqrt(3.)],
                       [1.0, -1.0, 1.0, -1.0]]

        # 断言数组几乎相等
        assert_array_almost_equal(z0, z0_expected)
        assert_array_almost_equal(z1, z1_expected)

    def test_zscore_ddof(self):
        # 测试 zscore 中 'ddof' 关键字的使用。
        # 创建一个二维 NumPy 数组
        x = np.array([[0.0, 0.0, 1.0, 1.0],
                      [0.0, 1.0, 2.0, 3.0]])

        # 在 axis=1 上计算 zscore，并指定 ddof=1
        z = stats.zscore(x, axis=1, ddof=1)

        # 预期的标准化结果
        z0_expected = np.array([-0.5, -0.5, 0.5, 0.5]) / (1.0 / np.sqrt(3))
        z1_expected = np.array([-1.5, -0.5, 0.5, 1.5]) / (np.sqrt(5. / 3))

        # 断言数组几乎相等
        assert_array_almost_equal(z[0], z0_expected)
        assert_array_almost_equal(z[1], z1_expected)

    def test_zscore_nan_propagate(self):
        # 测试 zscore 中 nan_policy='propagate' 的使用。
        # 创建一个包含 NaN 的 NumPy 数组
        x = np.array([1, 2, np.nan, 4, 5])

        # 对数组进行标准化，保持 NaN 的策略
        z = stats.zscore(x, nan_policy='propagate')

        # 断言所有结果为 NaN
        assert all(np.isnan(z))

    def test_zscore_nan_omit(self):
        # 测试 zscore 中 nan_policy='omit' 的使用。
        # 创建一个包含 NaN 的 NumPy 数组
        x = np.array([1, 2, np.nan, 4, 5])

        # 对数组进行标准化，省略 NaN 的策略
        z = stats.zscore(x, nan_policy='omit')

        # 预期的标准化结果
        expected = np.array([-1.2649110640673518,
                             -0.6324555320336759,
                             np.nan,
                             0.6324555320336759,
                             1.2649110640673518
                             ])
        # 断言数组几乎相等
        assert_array_almost_equal(z, expected)

    def test_zscore_nan_omit_with_ddof(self):
        # 测试 zscore 中 nan_policy='omit' 和 ddof 的结合使用。
        # 创建一个包含 NaN 的 NumPy 数组
        x = np.array([np.nan, 1.0, 3.0, 5.0, 7.0, 9.0])

        # 对数组进行标准化，省略 NaN 的策略，并指定 ddof=1
        z = stats.zscore(x, ddof=1, nan_policy='omit')

        # 预期的标准化结果
        expected = np.r_[np.nan, stats.zscore(x[1:], ddof=1)]

        # 断言所有元素几乎相等
        assert_allclose(z, expected, rtol=1e-13)

    def test_zscore_nan_raise(self):
        # 测试 zscore 中 nan_policy='raise' 的使用。
        # 创建一个包含 NaN 的 NumPy 数组
        x = np.array([1, 2, np.nan, 4, 5])

        # 断言当出现 NaN 时，会引发 ValueError 异常
        assert_raises(ValueError, stats.zscore, x, nan_policy='raise')

    def test_zscore_constant_input_1d(self):
        # 测试 zscore 处理一维常数输入的情况。
        # 创建一个一维列表
        x = [-0.087] * 3

        # 对列表进行标准化
        z = stats.zscore(x)

        # 断言结果数组与期望数组相等
        assert_equal(z, np.full(len(x), np.nan))
    # 定义测试函数，验证 zscore 函数在二维常量输入下的行为
    def test_zscore_constant_input_2d(self):
        # 创建一个二维 NumPy 数组 x
        x = np.array([[10.0, 10.0, 10.0, 10.0],
                      [10.0, 11.0, 12.0, 13.0]])
        # 计算沿着 axis=0 方向的 zscore
        z0 = stats.zscore(x, axis=0)
        # 断言 z0 的计算结果与预期值的一致性
        assert_equal(z0, np.array([[np.nan, -1.0, -1.0, -1.0],
                                   [np.nan, 1.0, 1.0, 1.0]]))
        # 计算沿着 axis=1 方向的 zscore
        z1 = stats.zscore(x, axis=1)
        # 断言 z1 的计算结果与预期值的一致性
        assert_equal(z1, np.array([[np.nan, np.nan, np.nan, np.nan],
                                   stats.zscore(x[1])]))
        # 计算沿着 axis=None 方向的 zscore
        z = stats.zscore(x, axis=None)
        # 断言 z 的计算结果与预期值的一致性
        assert_equal(z, stats.zscore(x.ravel()).reshape(x.shape))

        # 创建一个新的 3x6 的全为 1 的数组 y
        y = np.ones((3, 6))
        # 计算沿着 axis=None 方向的 zscore
        z = stats.zscore(y, axis=None)
        # 断言 z 的计算结果为一个与 y 形状相同且全为 NaN 的数组
        assert_equal(z, np.full(y.shape, np.nan))

    # 定义测试函数，验证 zscore 函数在二维常量输入且忽略 NaN 值的策略下的行为
    def test_zscore_constant_input_2d_nan_policy_omit(self):
        # 创建一个包含 NaN 值的二维 NumPy 数组 x
        x = np.array([[10.0, 10.0, 10.0, 10.0],
                      [10.0, 11.0, 12.0, np.nan],
                      [10.0, 12.0, np.nan, 10.0]])
        # 计算沿着 axis=0 方向且忽略 NaN 值的 zscore
        z0 = stats.zscore(x, nan_policy='omit', axis=0)
        # 预先计算的标准差值
        s = np.sqrt(3/2)
        s2 = np.sqrt(2)
        # 断言 z0 的计算结果与预期值的一致性
        assert_allclose(z0, np.array([[np.nan, -s, -1.0, np.nan],
                                      [np.nan, 0, 1.0, np.nan],
                                      [np.nan, s, np.nan, np.nan]]))
        # 计算沿着 axis=1 方向且忽略 NaN 值的 zscore
        z1 = stats.zscore(x, nan_policy='omit', axis=1)
        # 断言 z1 的计算结果与预期值的一致性
        assert_allclose(z1, np.array([[np.nan, np.nan, np.nan, np.nan],
                                      [-s, 0, s, np.nan],
                                      [-s2/2, s2, np.nan, -s2/2]]))

    # 定义测试函数，验证 zscore 函数在二维数组中某一行全为 NaN 时的行为
    def test_zscore_2d_all_nan_row(self):
        # 创建一个二维 NumPy 数组 x，其中一行全部为 NaN，并使用 axis=1
        x = np.array([[np.nan, np.nan, np.nan, np.nan],
                      [10.0, 10.0, 12.0, 12.0]])
        # 计算沿着 axis=1 方向且忽略 NaN 值的 zscore
        z = stats.zscore(x, nan_policy='omit', axis=1)
        # 断言 z 的计算结果与预期值的一致性
        assert_equal(z, np.array([[np.nan, np.nan, np.nan, np.nan],
                                  [-1.0, -1.0, 1.0, 1.0]]))

    # 定义测试函数，验证 zscore 函数在整个二维数组全部为 NaN 时的行为
    def test_zscore_2d_all_nan(self):
        # 创建一个全为 NaN 的 2x3 的二维 NumPy 数组 y，并使用 axis=None
        y = np.full((2, 3), np.nan)
        # 计算沿着 axis=None 方向且忽略 NaN 值的 zscore
        z = stats.zscore(y, nan_policy='omit', axis=None)
        # 断言 z 的计算结果与 y 的一致性
        assert_equal(z, y)

    # 标记化参数化测试函数，测试空输入的 zscore 行为
    @pytest.mark.parametrize('x', [np.array([]), np.zeros((3, 0, 5))])
    def test_zscore_empty_input(self, x):
        # 计算空输入 x 的 zscore
        z = stats.zscore(x)
        # 断言 z 的计算结果与 x 的一致性
        assert_equal(z, x)

    # 定义测试函数，验证 gzscore 函数在正常数组输入时的行为
    def test_gzscore_normal_array(self):
        # 创建一个正常的一维 NumPy 数组 x
        x = np.array([1, 2, 3, 4])
        # 计算 x 的 gzscore
        z = stats.gzscore(x)
        # 计算预期的 gzscore 值
        desired = np.log(x / stats.gmean(x)) / np.log(stats.gstd(x, ddof=0))
        # 断言 z 的计算结果与预期值的一致性
        assert_allclose(desired, z)

    # 标记化参数化测试函数，测试带有遮罩数组输入的 gzscore 行为
    @skip_xp_invalid_arg
    def test_gzscore_masked_array(self):
        # 创建一个带有遮罩的一维 NumPy 数组 x
        x = np.array([1, 2, -1, 3, 4])
        mx = np.ma.masked_array(x, mask=[0, 0, 1, 0, 0])
        # 计算带有遮罩数组 mx 的 gzscore
        z = stats.gzscore(mx)
        # 预期的 gzscore 值
        desired = ([-1.526072095151, -0.194700599824, np.inf, 0.584101799472,
                    1.136670895503])
        # 断言 z 的计算结果与预期值的一致性
        assert_allclose(desired, z)
    def test_zscore_masked_element_0_gh19039(self):
        # 当第0个元素被屏蔽时，zscore 返回所有 NaN。参见 gh-19039。
        rng = np.random.default_rng(8675309)
        # 使用随机数生成器创建正态分布的数组 x
        x = rng.standard_normal(10)
        # 创建一个与 x 相同形状的全零掩码数组 mask
        mask = np.zeros_like(x)
        # 使用掩码数组创建屏蔽数组 y
        y = np.ma.masked_array(x, mask)
        # 将 y 的第一个元素设为屏蔽状态
        y.mask[0] = True

        # 计算非屏蔽元素的 zscore 作为参考
        ref = stats.zscore(x[1:])  # compute reference from non-masked elements
        # 断言参考值中不存在 NaN
        assert not np.any(np.isnan(ref))
        # 计算屏蔽数组 y 的 zscore，并断言其与参考值接近
        res = stats.zscore(y)
        assert_allclose(res[1:], ref)
        # 对屏蔽数组 y 进行全局（展平） zscore 计算，并断言其与参考值接近
        res = stats.zscore(y, axis=None)
        assert_allclose(res[1:], ref)

        # 当非屏蔽元素完全相同时，屏蔽数组 y 的 zscore 结果为 NaN
        y[1:] = y[1]  # when non-masked elements are identical, result is nan
        # 计算屏蔽数组 y 的 zscore，并断言其结果为 NaN
        res = stats.zscore(y)
        assert_equal(res[1:], np.nan)
        # 对屏蔽数组 y 进行全局 zscore 计算，并断言其结果为 NaN
        res = stats.zscore(y, axis=None)
        assert_equal(res[1:], np.nan)
class TestMedianAbsDeviation:
    # 定义测试类 TestMedianAbsDeviation

    def setup_class(self):
        # 设置测试类的初始化方法
        self.dat_nan = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9,
                                 3.03, 3.03, 3.10, 3.37, 3.4, 3.4, 3.4, 3.5,
                                 3.6, 3.7, 3.7, 3.7, 3.7, 3.77, 5.28, np.nan])
        # 创建包含 NaN 的 NumPy 数组 dat_nan
        self.dat = np.array([2.20, 2.20, 2.4, 2.4, 2.5, 2.7, 2.8, 2.9, 3.03,
                             3.03, 3.10, 3.37, 3.4, 3.4, 3.4, 3.5, 3.6, 3.7,
                             3.7, 3.7, 3.7, 3.77, 5.28, 28.95])
        # 创建 NumPy 数组 dat

    def test_median_abs_deviation(self):
        # 定义测试方法 test_median_abs_deviation
        assert_almost_equal(stats.median_abs_deviation(self.dat, axis=None),
                            0.355)
        # 断言计算 dat 的全局中位数绝对偏差是否接近 0.355
        dat = self.dat.reshape(6, 4)
        # 将 dat 重塑为 6 行 4 列的数组
        mad = stats.median_abs_deviation(dat, axis=0)
        # 计算 dat 沿轴 0 的中位数绝对偏差
        mad_expected = np.asarray([0.435, 0.5, 0.45, 0.4])
        # 创建预期的中位数绝对偏差数组 mad_expected
        assert_array_almost_equal(mad, mad_expected)
        # 断言计算得到的 mad 与预期的 mad_expected 数组几乎相等

    def test_mad_nan_omit(self):
        # 定义测试方法 test_mad_nan_omit
        mad = stats.median_abs_deviation(self.dat_nan, nan_policy='omit')
        # 计算在忽略 NaN 的情况下，dat_nan 的中位数绝对偏差 mad
        assert_almost_equal(mad, 0.34)
        # 断言计算得到的 mad 是否接近 0.34

    def test_axis_and_nan(self):
        # 定义测试方法 test_axis_and_nan
        x = np.array([[1.0, 2.0, 3.0, 4.0, np.nan],
                      [1.0, 4.0, 5.0, 8.0, 9.0]])
        # 创建包含 NaN 的 NumPy 二维数组 x
        mad = stats.median_abs_deviation(x, axis=1)
        # 计算 x 沿轴 1 的中位数绝对偏差
        assert_equal(mad, np.array([np.nan, 3.0]))
        # 断言计算得到的 mad 是否与预期的数组 np.array([np.nan, 3.0]) 相等

    def test_nan_policy_omit_with_inf(self):
        # 定义测试方法 test_nan_policy_omit_with_inf
        z = np.array([1, 3, 4, 6, 99, np.nan, np.inf])
        # 创建包含 NaN 和 Inf 的 NumPy 数组 z
        mad = stats.median_abs_deviation(z, nan_policy='omit')
        # 计算在忽略 NaN 的情况下，z 的中位数绝对偏差 mad
        assert_equal(mad, 3.0)
        # 断言计算得到的 mad 是否等于 3.0

    @pytest.mark.parametrize('axis', [0, 1, 2, None])
    def test_size_zero_with_axis(self, axis):
        # 使用参数化测试装饰器定义测试方法 test_size_zero_with_axis
        x = np.zeros((3, 0, 4))
        # 创建一个形状为 (3, 0, 4) 的全零 NumPy 数组 x
        mad = stats.median_abs_deviation(x, axis=axis)
        # 计算 x 沿指定轴的中位数绝对偏差 mad
        assert_equal(mad, np.full_like(x.sum(axis=axis), fill_value=np.nan))
        # 断言计算得到的 mad 是否与预期的全零数组 np.full_like(x.sum(axis=axis), fill_value=np.nan) 相等

    @pytest.mark.parametrize('nan_policy, expected',
                             [('omit', np.array([np.nan, 1.5, 1.5])),
                              ('propagate', np.array([np.nan, np.nan, 1.5]))])
    def test_nan_policy_with_axis(self, nan_policy, expected):
        # 使用参数化测试装饰器定义测试方法 test_nan_policy_with_axis
        x = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                      [1, 5, 3, 6, np.nan, np.nan],
                      [5, 6, 7, 9, 9, 10]])
        # 创建包含 NaN 的 NumPy 二维数组 x
        mad = stats.median_abs_deviation(x, nan_policy=nan_policy, axis=1)
        # 根据指定的 NaN 策略计算 x 沿轴 1 的中位数绝对偏差 mad
        assert_equal(mad, expected)
        # 断言计算得到的 mad 是否与预期的 expected 数组相等

    @pytest.mark.parametrize('axis, expected',
                             [(1, [2.5, 2.0, 12.0]), (None, 4.5)])
    def test_center_mean_with_nan(self, axis, expected):
        # 使用参数化测试装饰器定义测试方法 test_center_mean_with_nan
        x = np.array([[1, 2, 4, 9, np.nan],
                      [0, 1, 1, 1, 12],
                      [-10, -10, -10, 20, 20]])
        # 创建包含 NaN 的 NumPy 二维数组 x
        mad = stats.median_abs_deviation(x, center=np.mean, nan_policy='omit',
                                         axis=axis)
        # 根据指定的中心值和 NaN 策略计算 x 沿指定轴的中位数绝对偏差 mad
        assert_allclose(mad, expected, rtol=1e-15, atol=1e-15)
        # 断言计算得到的 mad 是否与预期的 expected 数组非常接近

    def test_center_not_callable(self):
        # 定义测试方法 test_center_not_callable
        with pytest.raises(TypeError, match='callable'):
            # 使用 pytest 的断言捕获预期的 TypeError 异常，确保其错误信息包含 'callable'
            stats.median_abs_deviation([1, 2, 3, 5], center=99)
            # 调用 median_abs_deviation 函数，并传入一个不可调用的 center 参数
    Checks that all of the warnings from a list returned by
    `warnings.catch_all(record=True)` are of the required type and that the list
    contains expected number of warnings.
    """
    # 使用 assert_equal 函数检查 warn_list 的长度是否等于期望长度，用于验证警告数量是否正确
    assert_equal(len(warn_list), expected_len, "number of warnings")
    # 遍历 warn_list 中的每个警告对象 warn_，使用 assert_ 函数验证其类别是否与期望类型 expected_type 相符
    for warn_ in warn_list:
        assert_(warn_.category is expected_type)
class TestIQR:
    
    # 测试基本的情况
    def test_basic(self):
        # 创建一个包含8个元素的数组，并乘以0.5
        x = np.arange(8) * 0.5
        # 随机打乱数组顺序
        np.random.shuffle(x)
        # 断言计算出的四分位距等于1.75
        assert_equal(stats.iqr(x), 1.75)

    # 测试 API 的不同参数组合
    def test_api(self):
        # 创建一个全为1的5x5数组
        d = np.ones((5, 5))
        stats.iqr(d)  # 计算数组的四分位距
        stats.iqr(d, None)  # 使用默认参数计算四分位距
        stats.iqr(d, 1)  # 沿指定轴计算四分位距
        stats.iqr(d, (0, 1))  # 沿指定轴计算四分位距
        stats.iqr(d, None, (10, 90))  # 指定百分位数计算四分位距
        stats.iqr(d, None, (30, 20), 1.0)  # 指定插值方法计算四分位距
        stats.iqr(d, None, (25, 75), 1.5, 'propagate')  # 指定参数计算四分位距
        stats.iqr(d, None, (50, 50), 'normal', 'raise', 'linear')  # 指定多个参数计算四分位距
        stats.iqr(d, None, (25, 75), -0.4, 'omit', 'lower', True)  # 指定多个参数计算四分位距

    # 测试空数组的情况，期望触发警告
    @pytest.mark.parametrize('x', [[], np.arange(0)])
    def test_empty(self, x):
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            # 断言空数组计算的四分位距为 NaN
            assert_equal(stats.iqr(x), np.nan)

    # 测试全为常数的情况
    def test_constant(self):
        # 创建一个全为1的7x4数组
        x = np.ones((7, 4))
        # 断言常数数组计算的四分位距为0.0
        assert_equal(stats.iqr(x), 0.0)
        # 断言按指定轴计算的四分位距为全0数组
        assert_array_equal(stats.iqr(x, axis=0), np.zeros(4))
        assert_array_equal(stats.iqr(x, axis=1), np.zeros(7))
        # 断言按不同插值方法计算的四分位距为0.0
        assert_equal(stats.iqr(x, interpolation='linear'), 0.0)
        assert_equal(stats.iqr(x, interpolation='midpoint'), 0.0)
        assert_equal(stats.iqr(x, interpolation='nearest'), 0.0)
        assert_equal(stats.iqr(x, interpolation='lower'), 0.0)
        assert_equal(stats.iqr(x, interpolation='higher'), 0.0)

        # 在常数维度上为0，测试多轴情况
        y = np.ones((4, 5, 6)) * np.arange(6)
        assert_array_equal(stats.iqr(y, axis=0), np.zeros((5, 6)))
        assert_array_equal(stats.iqr(y, axis=1), np.zeros((4, 6)))
        # 断言按指定轴计算的四分位距为全1数组
        assert_array_equal(stats.iqr(y, axis=2), np.full((4, 5), 2.5))
        assert_array_equal(stats.iqr(y, axis=(0, 1)), np.zeros(6))
        assert_array_equal(stats.iqr(y, axis=(0, 2)), np.full(5, 3.))
        assert_array_equal(stats.iqr(y, axis=(1, 2)), np.full(4, 3.))

    # 测试标量数组的情况
    def test_scalarlike(self):
        # 创建一个包含一个标量的数组，并断言其四分位距为0.0
        x = np.arange(1) + 7.0
        assert_equal(stats.iqr(x[0]), 0.0)
        assert_equal(stats.iqr(x), 0.0)
        assert_array_equal(stats.iqr(x, keepdims=True), [0.0])

    # 测试二维数组的情况
    def test_2D(self):
        # 创建一个15个元素的数组，并reshape成3x5数组
        x = np.arange(15).reshape((3, 5))
        # 断言二维数组计算的四分位距为7.0
        assert_equal(stats.iqr(x), 7.0)
        # 断言按指定轴计算的四分位距为全5数组
        assert_array_equal(stats.iqr(x, axis=0), np.full(5, 5.))
        assert_array_equal(stats.iqr(x, axis=1), np.full(3, 2.))
        assert_array_equal(stats.iqr(x, axis=(0, 1)), 7.0)
        assert_array_equal(stats.iqr(x, axis=(1, 0)), 7.0)
    def test_axis(self):
        # `axis` 关键字在 `test_keepdims` 中也被测试过。
        
        # 创建一个形状为 (71, 23) 的正态分布随机数组成的矩阵
        o = np.random.normal(size=(71, 23))
        
        # 沿着第三个维度堆叠 o 10 次，形状变为 (71, 23, 10)
        x = np.dstack([o] * 10)
        
        # 计算 o 的四分位距
        q = stats.iqr(o)

        # 测试沿着指定轴 (0, 1) 的四分位距是否等于 q
        assert_equal(stats.iqr(x, axis=(0, 1)), q)
        
        # 将 x 的最后一个维度移到第一个位置，形状变为 (10, 71, 23)
        x = np.moveaxis(x, -1, 0)
        
        # 测试沿着指定轴 (2, 1) 的四分位距是否等于 q
        assert_equal(stats.iqr(x, axis=(2, 1)), q)
        
        # 交换 x 的第一个和第二个维度，形状变为 (71, 10, 23)
        x = x.swapaxes(0, 1)
        
        # 测试沿着指定轴 (0, 2) 的四分位距是否等于 q
        assert_equal(stats.iqr(x, axis=(0, 2)), q)
        
        # 再次交换 x 的第一个和第二个维度，形状恢复为 (10, 71, 23)
        x = x.swapaxes(0, 1)

        # 测试沿着所有轴 (0, 1, 2) 的四分位距是否等于沿着 None 轴的四分位距
        assert_equal(stats.iqr(x, axis=(0, 1, 2)),
                     stats.iqr(x, axis=None))
        
        # 测试沿着指定轴 (0,) 的四分位距是否等于沿着第 0 轴的四分位距
        assert_equal(stats.iqr(x, axis=(0,)),
                     stats.iqr(x, axis=0))

        # 创建一个形状为 (3, 5, 7, 11) 的数组 d，用顺序数填充并打乱
        d = np.arange(3 * 5 * 7 * 11)
        np.random.shuffle(d)
        d = d.reshape((3, 5, 7, 11))
        
        # 测试沿着指定轴 (0, 1, 2) 的四分位距第一个元素是否等于扁平化第一个轴切片的四分位距
        assert_equal(stats.iqr(d, axis=(0, 1, 2))[0],
                     stats.iqr(d[:,:,:, 0].ravel()))
        
        # 测试沿着指定轴 (0, 1, 3) 的四分位距第二个元素是否等于扁平化第二个轴切片的四分位距
        assert_equal(stats.iqr(d, axis=(0, 1, 3))[1],
                     stats.iqr(d[:,:, 1,:].ravel()))
        
        # 测试沿着指定轴 (3, 1, -4) 的四分位距第三个元素是否等于扁平化第三个轴切片的四分位距
        assert_equal(stats.iqr(d, axis=(3, 1, -4))[2],
                     stats.iqr(d[:,:, 2,:].ravel()))
        
        # 测试沿着指定轴 (3, 1, 2) 的四分位距第三个元素是否等于切片 (2, :, :, :) 的四分位距
        assert_equal(stats.iqr(d, axis=(3, 1, 2))[2],
                     stats.iqr(d[2,:,:,:].ravel()))
        
        # 测试沿着指定轴 (3, 2) 的四分位距第三行第二列元素是否等于切片 (2, 1, :,:) 的四分位距
        assert_equal(stats.iqr(d, axis=(3, 2))[2, 1],
                     stats.iqr(d[2, 1,:,:].ravel()))
        
        # 测试沿着指定轴 (1, -2) 的四分位距第三行第二列元素是否等于切片 (2, :, :, 1) 的四分位距
        assert_equal(stats.iqr(d, axis=(1, -2))[2, 1],
                     stats.iqr(d[2, :, :, 1].ravel()))
        
        # 测试沿着指定轴 (1, 3) 的四分位距第三行第三列元素是否等于切片 (2, :, 2,:) 的四分位距
        assert_equal(stats.iqr(d, axis=(1, 3))[2, 2],
                     stats.iqr(d[2, :, 2,:].ravel()))

        # 测试在轴超出数组维度时是否抛出 AxisError 异常
        assert_raises(AxisError, stats.iqr, d, axis=4)
        
        # 测试重复的轴索引是否抛出 ValueError 异常
        assert_raises(ValueError, stats.iqr, d, axis=(0, 0))

    def test_rng(self):
        # 创建一个长度为 5 的数组 x
        x = np.arange(5)
        
        # 测试不同范围 rng 参数下的四分位距是否计算正确
        assert_equal(stats.iqr(x), 2)
        assert_equal(stats.iqr(x, rng=(25, 87.5)), 2.5)
        assert_equal(stats.iqr(x, rng=(12.5, 75)), 2.5)
        assert_almost_equal(stats.iqr(x, rng=(10, 50)), 1.6)  # 3-1.4
        
        # 测试超出百分位数范围的 ValueError 异常
        assert_raises(ValueError, stats.iqr, x, rng=(0, 101))
        
        # 测试 NaN 参数的 ValueError 异常
        assert_raises(ValueError, stats.iqr, x, rng=(np.nan, 25))
        
        # 测试错误的参数类型是否引发 TypeError 异常
        assert_raises(TypeError, stats.iqr, x, rng=(0, 50, 60))
    def test_interpolation(self):
        # 创建一个包含0到4的数组
        x = np.arange(5)
        # 创建一个包含0到3的数组
        y = np.arange(4)
        
        # 默认插值方式
        assert_equal(stats.iqr(x), 2)
        assert_equal(stats.iqr(y), 1.5)
        
        # 线性插值方式
        assert_equal(stats.iqr(x, interpolation='linear'), 2)
        assert_equal(stats.iqr(y, interpolation='linear'), 1.5)
        
        # 更高插值方式
        assert_equal(stats.iqr(x, interpolation='higher'), 2)
        assert_equal(stats.iqr(x, rng=(25, 80), interpolation='higher'), 3)
        assert_equal(stats.iqr(y, interpolation='higher'), 2)
        
        # 较低插值方式（通常与higher方式相同，但不总是）
        assert_equal(stats.iqr(x, interpolation='lower'), 2)
        assert_equal(stats.iqr(x, rng=(25, 80), interpolation='lower'), 2)
        assert_equal(stats.iqr(y, interpolation='lower'), 2)
        
        # 最近邻插值方式
        assert_equal(stats.iqr(x, interpolation='nearest'), 2)
        assert_equal(stats.iqr(y, interpolation='nearest'), 1)
        
        # 中点插值方式
        assert_equal(stats.iqr(x, interpolation='midpoint'), 2)
        assert_equal(stats.iqr(x, rng=(25, 80), interpolation='midpoint'), 2.5)
        assert_equal(stats.iqr(y, interpolation='midpoint'), 2)

        # 检查所有新的 numpy 1.22.0 版本中新增的 method= 值是否被接受
        for method in ('inverted_cdf', 'averaged_inverted_cdf',
                       'closest_observation', 'interpolated_inverted_cdf',
                       'hazen', 'weibull', 'median_unbiased',
                       'normal_unbiased'):
            stats.iqr(y, interpolation=method)

        # 断言错误，期望引发 ValueError 异常
        assert_raises(ValueError, stats.iqr, x, interpolation='foobar')

    def test_keepdims(self):
        # 同时测试大多数的 `axis`
        x = np.ones((3, 5, 7, 11))
        
        # keepdims=False 的情况
        assert_equal(stats.iqr(x, axis=None, keepdims=False).shape, ())
        assert_equal(stats.iqr(x, axis=2, keepdims=False).shape, (3, 5, 11))
        assert_equal(stats.iqr(x, axis=(0, 1), keepdims=False).shape, (7, 11))
        assert_equal(stats.iqr(x, axis=(0, 3), keepdims=False).shape, (5, 7))
        assert_equal(stats.iqr(x, axis=(1,), keepdims=False).shape, (3, 7, 11))
        assert_equal(stats.iqr(x, (0, 1, 2, 3), keepdims=False).shape, ())
        assert_equal(stats.iqr(x, axis=(0, 1, 3), keepdims=False).shape, (7,))
        
        # keepdims=True 的情况
        assert_equal(stats.iqr(x, axis=None, keepdims=True).shape, (1, 1, 1, 1))
        assert_equal(stats.iqr(x, axis=2, keepdims=True).shape, (3, 5, 1, 11))
        assert_equal(stats.iqr(x, axis=(0, 1), keepdims=True).shape, (1, 1, 7, 11))
        assert_equal(stats.iqr(x, axis=(0, 3), keepdims=True).shape, (1, 5, 7, 1))
        assert_equal(stats.iqr(x, axis=(1,), keepdims=True).shape, (3, 1, 7, 11))
        assert_equal(stats.iqr(x, (0, 1, 2, 3), keepdims=True).shape, (1, 1, 1, 1))
        assert_equal(stats.iqr(x, axis=(0, 1, 3), keepdims=True).shape, (1, 1, 7, 1))
    # 定义一个测试函数 test_nanpolicy，用于测试 stats 模块中的 iqr 函数对不同 NaN 策略的处理

    # 创建一个 3x5 的 NumPy 数组 x，包含值从 0 到 14，用于后续测试
    x = np.arange(15.0).reshape((3, 5))

    # 测试当数组中没有 NaN 值时，iqr 函数的返回结果是否正确
    assert_equal(stats.iqr(x, nan_policy='propagate'), 7)
    assert_equal(stats.iqr(x, nan_policy='omit'), 7)
    assert_equal(stats.iqr(x, nan_policy='raise'), 7)

    # 将数组 x 中的某个元素设为 NaN，模拟存在 NaN 值的情况
    x[1, 2] = np.nan

    # 使用 catch_warnings 来捕获警告，确保 NaN 策略为 'propagate' 时，iqr 函数正确处理 NaN 的情况
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert_equal(stats.iqr(x, nan_policy='propagate'), np.nan)
        assert_equal(stats.iqr(x, axis=0, nan_policy='propagate'), [5, 5, np.nan, 5, 5])
        assert_equal(stats.iqr(x, axis=1, nan_policy='propagate'), [2, np.nan, 2])

    # 测试 NaN 策略为 'omit' 时，iqr 函数对数组中 NaN 值的处理是否正确
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert_equal(stats.iqr(x, nan_policy='omit'), 7.5)
        assert_equal(stats.iqr(x, axis=0, nan_policy='omit'), np.full(5, 5))
        assert_equal(stats.iqr(x, axis=1, nan_policy='omit'), [2, 2.5, 2])

    # 测试 NaN 策略为 'raise' 时，iqr 函数是否会抛出 ValueError 异常
    assert_raises(ValueError, stats.iqr, x, nan_policy='raise')
    assert_raises(ValueError, stats.iqr, x, axis=0, nan_policy='raise')
    assert_raises(ValueError, stats.iqr, x, axis=1, nan_policy='raise')

    # 测试 NaN 策略为无效值 'barfood' 时，是否会抛出 ValueError 异常
    assert_raises(ValueError, stats.iqr, x, nan_policy='barfood')
    def test_scale(self):
        # 创建一个 3x5 的 NumPy 数组，元素为从 0 到 14
        x = np.arange(15.0).reshape((3, 5))

        # 测试在没有 NaN 的情况下，使用不同的缩放参数计算 IQR
        assert_equal(stats.iqr(x, scale=1.0), 7)
        assert_almost_equal(stats.iqr(x, scale='normal'), 7 / 1.3489795)
        assert_equal(stats.iqr(x, scale=2.0), 3.5)

        # 改变数组中的一个值为 NaN，测试 NaN 策略为 propagate 时的 IQR 计算
        x[1, 2] = np.nan
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert_equal(stats.iqr(x, scale=1.0, nan_policy='propagate'), np.nan)
            assert_equal(stats.iqr(x, scale='normal', nan_policy='propagate'), np.nan)
            assert_equal(stats.iqr(x, scale=2.0, nan_policy='propagate'), np.nan)
            # axis=1 选定用于展示处理带有和不带有 NaN 的行为
            assert_equal(stats.iqr(x, axis=1, scale=1.0,
                                   nan_policy='propagate'), [2, np.nan, 2])
            assert_almost_equal(stats.iqr(x, axis=1, scale='normal',
                                          nan_policy='propagate'),
                                np.array([2, np.nan, 2]) / 1.3489795)
            assert_equal(stats.iqr(x, axis=1, scale=2.0, nan_policy='propagate'),
                         [1, np.nan, 1])
            # 自 NumPy 1.17.0.dev 起，np.percentile 不再对带 NaN 的情况产生警告
            # 所以在这里不检查警告的数量。参见 https://github.com/numpy/numpy/pull/12679.

        # 测试 NaN 策略为 omit 时的 IQR 计算
        assert_equal(stats.iqr(x, scale=1.0, nan_policy='omit'), 7.5)
        assert_almost_equal(stats.iqr(x, scale='normal', nan_policy='omit'),
                            7.5 / 1.3489795)
        assert_equal(stats.iqr(x, scale=2.0, nan_policy='omit'), 3.75)

        # 测试错误的缩放参数时是否会引发 ValueError
        assert_raises(ValueError, stats.iqr, x, scale='foobar')
class TestMoments:
    """
    Comparison numbers are found using R v.1.5.1
    note that length(testcase) = 4
    testmathworks comes from documentation for the
    Statistics Toolbox for Matlab and can be found at both
    https://www.mathworks.com/help/stats/kurtosis.html
    https://www.mathworks.com/help/stats/skewness.html
    Note that both test cases came from here.
    """

    # 定义测试用例，包含四个数字
    testcase = [1., 2., 3., 4.]

    # 定义一个单独的数值测试用例
    scalar_testcase = 4.

    # 使用随机数种子初始化测试用例的精度
    np.random.seed(1234)
    testcase_moment_accuracy = np.random.rand(42)

    def _assert_equal(self, actual, expect, *, shape=None, dtype=None):
        """
        比较函数，用来断言两个数组是否相等
        """
        expect = np.asarray(expect)
        if shape is not None:
            expect = np.broadcast_to(expect, shape)
        assert_array_equal(actual, expect)
        if dtype is None:
            dtype = expect.dtype
        assert actual.dtype == dtype

    @array_api_compatible
    @pytest.mark.parametrize('size', [10, (10, 2)])
    @pytest.mark.parametrize('m, c', product((0, 1, 2, 3), (None, 0, 1)))
    def test_moment_center_scalar_moment(self, size, m, c, xp):
        """
        测试单一数值和中心化的矩
        """
        rng = np.random.default_rng(6581432544381372042)
        x = xp.asarray(rng.random(size=size))
        res = stats.moment(x, m, center=c)
        c = xp.mean(x, axis=0) if c is None else c
        ref = xp.sum((x - c)**m, axis=0)/x.shape[0]
        xp_assert_close(res, ref, atol=1e-16)

    @array_api_compatible
    @pytest.mark.parametrize('size', [10, (10, 2)])
    @pytest.mark.parametrize('c', (None, 0, 1))
    def test_moment_center_array_moment(self, size, c, xp):
        """
        测试数组和中心化的矩
        """
        rng = np.random.default_rng(1706828300224046506)
        x = xp.asarray(rng.random(size=size))
        m = [0, 1, 2, 3]
        res = stats.moment(x, m, center=c)
        xp_test = array_namespace(x)  # no `concat` in np < 2.0; no `newaxis` in torch
        ref = xp_test.concat([stats.moment(x, i, center=c)[xp_test.newaxis, ...]
                              for i in m])
        xp_assert_equal(res, ref)

    @array_api_compatible
    # 定义一个测试方法，用于测试统计学中的矩(moment)计算
    def test_moment(self, xp):
        # 使用 xp.asarray 将 self.testcase 转换为对应的数组表示
        testcase = xp.asarray(self.testcase)

        # 计算标量数组 self.scalar_testcase 的矩
        y = stats.moment(xp.asarray(self.scalar_testcase))
        xp_assert_close(y, xp.asarray(0.0))

        # 计算 testcase 数组的零阶矩，即均值的幂为0次方
        y = stats.moment(testcase, 0)
        xp_assert_close(y, xp.asarray(1.0))

        # 计算 testcase 数组的一阶矩，即均值的幂为1次方
        y = stats.moment(testcase, 1)
        xp_assert_close(y, xp.asarray(0.0))

        # 计算 testcase 数组的二阶矩，即均值的幂为2次方
        y = stats.moment(testcase, 2)
        xp_assert_close(y, xp.asarray(1.25))

        # 计算 testcase 数组的三阶矩，即均值的幂为3次方
        y = stats.moment(testcase, 3)
        xp_assert_close(y, xp.asarray(0.0))

        # 计算 testcase 数组的四阶矩，即均值的幂为4次方
        y = stats.moment(testcase, 4)
        xp_assert_close(y, xp.asarray(2.5625))

        # 检查 moment 函数接受 array_like 的 order 输入
        y = stats.moment(testcase, [1, 2, 3, 4])
        xp_assert_close(y, xp.asarray([0, 1.25, 0, 2.5625]))

        # 检查 moment 函数只接受整数的 order 输入
        y = stats.moment(testcase, 0.0)
        xp_assert_close(y, xp.asarray(1.0))

        # 用 pytest 检查 moment 函数在 order 输入非整数时抛出 ValueError 异常
        message = 'All elements of `order` must be integral.'
        with pytest.raises(ValueError, match=message):
            stats.moment(testcase, 1.2)

        # 用 pytest 检查 moment 函数在 order 输入为浮点数列表时抛出 ValueError 异常
        y = stats.moment(testcase, [1.0, 2, 3, 4.0])
        xp_assert_close(y, xp.asarray([0, 1.25, 0, 2.5625]))

        # 定义内部测试函数 test_cases，用于测试各种特殊情况
        def test_cases():
            # 测试空数组的矩，期望结果为 NaN
            y = stats.moment(xp.asarray([]))
            xp_assert_equal(y, xp.asarray(xp.nan))

            # 测试空数组的矩，指定数据类型为 float32，期望结果为 NaN
            y = stats.moment(xp.asarray([], dtype=xp.float32))
            xp_assert_equal(y, xp.asarray(xp.nan, dtype=xp.float32))

            # 测试形状为 (1, 0) 的零数组的矩，期望结果为空数组
            y = stats.moment(xp.zeros((1, 0)), axis=0)
            xp_assert_equal(y, xp.empty((0,)))

            # 测试形状为 (1, 1) 的空数组的矩，期望结果为 [NaN]
            y = stats.moment(xp.asarray([[]]), axis=1)
            xp_assert_equal(y, xp.asarray([xp.nan]))

            # 测试形状为 (2, 0) 的空数组的矩，期望结果为空数组
            y = stats.moment(xp.asarray([[]]), order=[0, 1], axis=0)
            xp_assert_equal(y, xp.empty((2, 0)))

        # 测试空输入情况下的特殊情况处理
        if is_numpy(xp):
            # 使用 pytest.warns 检查 SmallSampleWarning 警告信息
            with pytest.warns(SmallSampleWarning, match="See documentation for..."):
                test_cases()
        else:
            # 使用 np.testing.suppress_warnings 确保在 array_api_strict 模式下正确处理警告
            with np.testing.suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "Mean of empty slice.")
                sup.filter(RuntimeWarning, "invalid value")
                test_cases()

    # 测试 moment 函数的 NaN 处理策略
    def test_nan_policy(self):
        # 创建包含 NaN 的数组 x
        x = np.arange(10.)
        x[9] = np.nan

        # 使用 assert_equal 检查使用默认 NaN 策略计算的二阶矩
        assert_equal(stats.moment(x, 2), np.nan)

        # 使用 assert_almost_equal 检查使用 'omit' 策略计算的二阶矩，期望结果接近 0.0
        assert_almost_equal(stats.moment(x, nan_policy='omit'), 0.0)

        # 使用 assert_raises 检查使用无效 NaN 策略 'raise' 时是否引发 ValueError 异常
        assert_raises(ValueError, stats.moment, x, nan_policy='raise')

        # 使用 assert_raises 检查使用未知 NaN 策略 'foobar' 时是否引发 ValueError 异常
        assert_raises(ValueError, stats.moment, x, nan_policy='foobar')

    # 使用 array API 兼容修饰符，对不同数据类型和 order 进行参数化测试
    @array_api_compatible
    @pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex128'])
    @pytest.mark.parametrize('expect, order', [(0, 1), (1, 0)])
    # 测试给定数据类型、期望结果、阶数和数组库的常数时刻函数
    def test_constant_moments(self, dtype, expect, order, xp):
        # 如果数据类型为'complex128'并且使用的是 Torch，则跳过测试
        if dtype=='complex128' and is_torch(xp):
            pytest.skip()
        # 将数据类型转换为指定库的相应类型
        dtype = getattr(xp, dtype)
        # 创建一个指定数据类型的随机数组
        x = xp.asarray(np.random.rand(5), dtype=dtype)
        # 计算数组 x 的矩阵的 order 阶时刻，并将结果赋给 y
        y = stats.moment(x, order=order)
        xp_assert_equal(y, xp.asarray(expect, dtype=dtype))

        # 对广播后形状为 (6, 5) 的数组计算 order 阶时刻，并进行比较
        y = stats.moment(xp.broadcast_to(x, (6, 5)), axis=0, order=order)
        xp_assert_equal(y, xp.full((5,), expect, dtype=dtype))

        # 对广播后形状为 (1, 2, 3, 4, 5) 的数组沿 axis=2 计算 order 阶时刻，并进行比较
        y = stats.moment(xp.broadcast_to(x, (1, 2, 3, 4, 5)), axis=2,
                         order=order)
        xp_assert_equal(y, xp.full((1, 2, 4, 5), expect, dtype=dtype))

        # 对广播后形状为 (1, 2, 3, 4, 5) 的数组计算 order 阶时刻（全局），并进行比较
        y = stats.moment(xp.broadcast_to(x, (1, 2, 3, 4, 5)), axis=None,
                         order=order)
        xp_assert_equal(y, xp.full((), expect, dtype=dtype))

    # 跳过支持 JAX 数组的 numpy 后端
    @skip_xp_backends('jax.numpy',
                      reasons=["JAX arrays do not support item assignment"])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible
    def test_moment_propagate_nan(self, xp):
        # 检查输入带有和不带有 NaN 值时结果的形状是否相同，参见 gh-5817
        a = xp.reshape(xp.arange(8.), (2, -1))
        a[1, 0] = np.nan
        # 计算数组 a 沿 axis=1 的二阶时刻，并进行比较
        mm = stats.moment(a, 2, axis=1)
        xp_assert_close(mm, xp.asarray([1.25, np.nan]), atol=1e-15)

    @array_api_compatible
    def test_moment_empty_order(self, xp):
        # 测试空的 'order' 列表时的 moment 函数
        with pytest.raises(ValueError, match=r"'order' must be a scalar or a"
                                             r" non-empty 1D list/array."):
            # 调用 moment 函数，传入空的 'order' 列表，预期引发 ValueError 异常
            stats.moment(xp.asarray([1, 2, 3, 4]), order=[])

    @array_api_compatible
    def test_rename_moment_order(self, xp):
        # 参数 'order' 曾经称为 'moment'，旧名称仍然可用，因此必须继续支持
        x = xp.arange(10)
        # 使用 'moment' 参数调用 moment 函数，并与 'order' 参数调用结果进行比较
        res = stats.moment(x, moment=3)
        ref = stats.moment(x, order=3)
        xp_assert_equal(res, ref)

    # 测试 moment 函数的准确性
    def test_moment_accuracy(self):
        # 'moment' 必须与更慢但非常准确的 numpy.power() 实现的误差足够小
        tc_no_mean = (self.testcase_moment_accuracy
                      - np.mean(self.testcase_moment_accuracy))
        # 断言 np.power(tc_no_mean, 42).mean() 与 stats.moment 函数的结果在误差范围内相等
        assert_allclose(np.power(tc_no_mean, 42).mean(),
                        stats.moment(self.testcase_moment_accuracy, 42))

    @array_api_compatible
    @pytest.mark.parametrize('order', [0, 1, 2, 3])
    @pytest.mark.parametrize('axis', [-1, 0, 1])
    @pytest.mark.parametrize('center', [None, 0])
    def test_moment_array_api(self, xp, order, axis, center):
        # 使用随机数生成器创建形状为 (5, 6, 7) 的随机数组 x
        rng = np.random.default_rng(34823589259425)
        x = rng.random(size=(5, 6, 7))
        # 调用 moment 函数，传入指定的 order、axis 和 center 参数，与参考结果进行比较
        res = stats.moment(xp.asarray(x), order, axis=axis, center=center)
        ref = xp.asarray(_moment(x, order, axis, mean=center))
        xp_assert_close(res, ref)
class SkewKurtosisTest:
    # 定义一个标量测试用例
    scalar_testcase = 4.
    # 定义一个包含浮点数的测试用例列表
    testcase = [1., 2., 3., 4.]
    # 定义一个用于数学计算的测试用例列表
    testmathworks = [1.165, 0.6268, 0.0751, 0.3516, -0.6965]


class TestSkew(SkewKurtosisTest):
    @array_api_compatible
    @pytest.mark.parametrize('stat_fun', [stats.skew, stats.kurtosis])
    def test_empty_1d(self, stat_fun, xp):
        # 创建一个空的1维数组x
        x = xp.asarray([])
        if is_numpy(xp):
            # 如果使用的是NumPy，则测试警告是否正常产生
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                res = stat_fun(x)
        else:
            # 对于其他后端，使用suppress_warnings来过滤特定的警告
            with np.testing.suppress_warnings() as sup:
                # array_api_strict会产生这些警告
                sup.filter(RuntimeWarning, "Mean of empty slice")
                sup.filter(RuntimeWarning, "invalid value encountered")
                res = stat_fun(x)
        # 检查计算结果res是否为NaN
        xp_assert_equal(res, xp.asarray(xp.nan))

    @skip_xp_backends('jax.numpy',
                      reasons=["JAX arrays do not support item assignment"])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible
    def test_skewness(self, xp):
        # 标量测试用例的偏度计算
        y = stats.skew(xp.asarray(self.scalar_testcase))
        xp_assert_close(y, xp.asarray(xp.nan))
        # 对testmathworks列表进行偏度计算，并检查结果
        y = stats.skew(xp.asarray(self.testmathworks))
        xp_assert_close(y, xp.asarray(-0.29322304336607), atol=1e-10)
        # 使用bias参数进行偏度计算，并检查结果
        y = stats.skew(xp.asarray(self.testmathworks), bias=0)
        xp_assert_close(y, xp.asarray(-0.437111105023940), atol=1e-10)
        # 对测试用例列表self.testcase进行偏度计算，并检查结果
        y = stats.skew(xp.asarray(self.testcase))
        xp_assert_close(y, xp.asarray(0.0), atol=1e-10)

    def test_nan_policy(self):
        # 初始情况下，nan_policy在使用替代后端时会被忽略
        x = np.arange(10.)
        x[9] = np.nan
        with np.errstate(invalid='ignore'):
            # 检查包含NaN的数组x的偏度计算结果
            assert_equal(stats.skew(x), np.nan)
        # 使用nan_policy='omit'来处理包含NaN的数组x，检查偏度计算结果
        assert_equal(stats.skew(x, nan_policy='omit'), 0.)
        # 检查使用不支持的nan_policy值时是否引发了ValueError
        assert_raises(ValueError, stats.skew, x, nan_policy='raise')
        assert_raises(ValueError, stats.skew, x, nan_policy='foobar')

    def test_skewness_scalar(self):
        # `skew`必须对1维输入返回标量结果（仅适用于NumPy数组）
        assert_equal(stats.skew(arange(10)), 0.0)

    @skip_xp_backends('jax.numpy',
                      reasons=["JAX arrays do not support item assignment"])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible
    def test_skew_propagate_nan(self, xp):
        # 检查带有NaN和没有NaN输入的结果形状是否相同，参见gh-5817
        a = xp.arange(8.)
        a = xp.reshape(a, (2, -1))
        a[1, 0] = xp.nan
        with np.errstate(invalid='ignore'):
            # 对数组a按轴1进行偏度计算，并检查结果
            s = stats.skew(a, axis=1)
        xp_assert_equal(s, xp.asarray([0, xp.nan]))

    @array_api_compatible
    def test_skew_constant_value(self, xp):
        # 测试常数输入的偏度应该是 NaN (gh-16061)
        with pytest.warns(RuntimeWarning, match="Precision loss occurred"):
            # 创建一个长度为10的常数数组
            a = xp.asarray([-0.27829495]*10)  # xp.repeat not currently available
            # 断言计算得到的偏度应该是 NaN
            xp_assert_equal(stats.skew(a), xp.asarray(xp.nan))
            # 断言计算得到的偏度应该是 NaN，即使乘以一个非常大的数
            xp_assert_equal(stats.skew(a*2.**50), xp.asarray(xp.nan))
            # 断言计算得到的偏度应该是 NaN，即使除以一个非常大的数
            xp_assert_equal(stats.skew(a/2.**50), xp.asarray(xp.nan))
            # 断言计算得到的偏度应该是 NaN，关闭偏差校正
            xp_assert_equal(stats.skew(a, bias=False), xp.asarray(xp.nan))

            # 同样地，来自 gh-11086 的例子：
            # 创建一个长度为7的常数数组
            a = xp.asarray([14.3]*7)
            # 断言计算得到的偏度应该是 NaN
            xp_assert_equal(stats.skew(a), xp.asarray(xp.nan))
            # 创建一个接近常数的数组
            a = 1. + xp.arange(-3., 4)*1e-16
            # 断言计算得到的偏度应该是 NaN
            xp_assert_equal(stats.skew(a), xp.asarray(xp.nan))

    @skip_xp_backends('jax.numpy',
                      reasons=["JAX arrays do not support item assignment"])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible
    def test_precision_loss_gh15554(self, xp):
        # gh-15554 是关于常数或接近常数输入的问题之一。我们不能总是修复这些问题，但是确保有警告。
        with pytest.warns(RuntimeWarning, match="Precision loss occurred"):
            # 使用随机数生成器创建一个数组，大小为 (100, 10)
            rng = np.random.default_rng(34095309370)
            a = xp.asarray(rng.random(size=(100, 10)))
            # 将数组的第一列设为常数 1.01
            a[:, 0] = 1.01
            # 计算数组的偏度
            stats.skew(a)

    @skip_xp_backends('jax.numpy',
                      reasons=["JAX arrays do not support item assignment"])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible
    @pytest.mark.parametrize('axis', [-1, 0, 2, None])
    @pytest.mark.parametrize('bias', [False, True])
    def test_vectorization(self, xp, axis, bias):
        # 上面几乎没有测试数组输入的行为。与简单的实现进行比较。
        # 使用随机数生成器创建一个大小为 (3, 4, 5) 的数组
        rng = np.random.default_rng(1283413549926)
        x = xp.asarray(rng.random((3, 4, 5)))

        def skewness(a, axis, bias):
            # 偏度的简单实现
            if axis is None:
                # 将数组展平为一维数组
                a = xp.reshape(a, (-1,))
                axis = 0
            xp_test = array_namespace(a)  # 默认情况下是普通的 torch ddof=1
            # 计算均值
            mean = xp_test.mean(a, axis=axis, keepdims=True)
            # 计算三阶中心矩
            mu3 = xp_test.mean((a - mean)**3, axis=axis)
            # 计算标准差
            std = xp_test.std(a, axis=axis)
            # 计算偏度
            res = mu3 / std ** 3
            # 如果不使用偏差校正
            if not bias:
                n = a.shape[axis]
                res *= ((n - 1.0) * n) ** 0.5 / (n - 2.0)
            return res

        # 计算数组 x 的偏度
        res = stats.skew(x, axis=axis, bias=bias)
        # 使用 skewness 函数计算的参考值
        ref = skewness(x, axis=axis, bias=bias)
        # 断言两者非常接近
        xp_assert_close(res, ref)
class TestKurtosis(SkewKurtosisTest):

    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible
    def test_kurtosis(self, xp):
        # Scalar test case
        y = stats.kurtosis(xp.asarray(self.scalar_testcase))
        assert xp.isnan(y)

        # 计算峰度的公式1:
        # sum((testcase-mean(testcase,axis=0))**4,axis=0)
        # / ((sqrt(var(testcase)*3/4))**4)
        # / 4

        # 计算峰度的公式2:
        # sum((test2-mean(testmathworks,axis=0))**4,axis=0)
        # / ((sqrt(var(testmathworks)*4/5))**4)
        # / 5

        # 设置轴向标志为0，fisher=0（使用Pearson定义的峰度，与Matlab兼容）
        y = stats.kurtosis(xp.asarray(self.testmathworks), 0, fisher=0, bias=1)
        xp_assert_close(y, xp.asarray(2.1658856802973))

        # 注意：MATLAB对以下情况的文档不够清晰
        # kurtosis(x,0) 给出Pearson偏度的无偏估计
        # kurtosis(x) 给出Fisher偏度的有偏估计（Pearson-3）
        # MATLAB文档暗示两者都应该给出Fisher偏度
        y = stats.kurtosis(xp.asarray(self.testmathworks), fisher=0, bias=0)
        xp_assert_close(y, xp.asarray(3.663542721189047))
        y = stats.kurtosis(xp.asarray(self.testcase), 0, 0)
        xp_assert_close(y, xp.asarray(1.64))

        x = xp.arange(10.)
        x = xp.where(x == 8, xp.asarray(xp.nan), x)
        xp_assert_equal(stats.kurtosis(x), xp.asarray(xp.nan))

    def test_kurtosis_nan_policy(self):
        # nan_policy 目前仅适用于NumPy
        x = np.arange(10.)
        x[9] = np.nan
        assert_almost_equal(stats.kurtosis(x, nan_policy='omit'), -1.230000)
        assert_raises(ValueError, stats.kurtosis, x, nan_policy='raise')
        assert_raises(ValueError, stats.kurtosis, x, nan_policy='foobar')

    def test_kurtosis_array_scalar(self):
        # "array scalars" 在其他后端中不存在
        assert_equal(type(stats.kurtosis([1, 2, 3])), np.float64)

    def test_kurtosis_propagate_nan(self):
        # nan_policy 目前仅适用于NumPy
        # 检查结果的形状对带有NaN和不带NaN的输入是相同的，参见gh-5817
        a = np.arange(8).reshape(2, -1).astype(float)
        a[1, 0] = np.nan
        k = stats.kurtosis(a, axis=1, nan_policy="propagate")
        np.testing.assert_allclose(k, [-1.36, np.nan], atol=1e-15)

    @array_api_compatible


这些注释对每个方法和相关代码片段进行了详细解释，确保了代码的每个部分都得到了适当的说明和理解。
    def test_kurtosis_constant_value(self, xp):
        # Kurtosis of a constant input should be NaN (gh-16061)
        # 创建一个包含10个重复数值的数组
        a = xp.asarray([-0.27829495]*10)
        # 使用 pytest 来检查是否会发出 RuntimeWarning，且匹配特定的警告信息
        with pytest.warns(RuntimeWarning, match="Precision loss occurred"):
            # 检查未进行修正的峰度是否为 NaN
            assert xp.isnan(stats.kurtosis(a, fisher=False))
            # 检查乘以大数后的未进行修正的峰度是否为 NaN
            assert xp.isnan(stats.kurtosis(a * float(2**50), fisher=False))
            # 检查除以小数后的未进行修正的峰度是否为 NaN
            assert xp.isnan(stats.kurtosis(a / float(2**50), fisher=False))
            # 检查未进行修正且偏差为 False 的峰度是否为 NaN
            assert xp.isnan(stats.kurtosis(a, fisher=False, bias=False))

    @skip_xp_backends('jax.numpy',
                      reasons=['JAX arrays do not support item assignment'])
    @pytest.mark.usefixtures("skip_xp_backends")
    @array_api_compatible
    @pytest.mark.parametrize('axis', [-1, 0, 2, None])
    @pytest.mark.parametrize('bias', [False, True])
    @pytest.mark.parametrize('fisher', [False, True])
    def test_vectorization(self, xp, axis, bias, fisher):
        # Behavior with array input is not tested above. Compare
        # against naive implementation.
        # 使用 numpy 的默认随机数生成器创建一个4x5x6的数组 x
        rng = np.random.default_rng(1283413549926)
        x = xp.asarray(rng.random((4, 5, 6)))

        def kurtosis(a, axis, bias, fisher):
            # Simple implementation of kurtosis
            # 如果 axis 为 None，则将数组展平，并将 axis 设为 0
            if axis is None:
                a = xp.reshape(a, (-1,))
                axis = 0
            # 根据数组类型进行相应的数学运算
            xp_test = array_namespace(a)  # plain torch ddof=1 by default
            # 计算均值，keepdims=True 保持维度
            mean = xp_test.mean(a, axis=axis, keepdims=True)
            # 计算四阶中心矩
            mu4 = xp_test.mean((a - mean)**4, axis=axis)
            # 计算二阶中心矩，correction=0 为方差修正项
            mu2 = xp_test.var(a, axis=axis, correction=0)
            # 根据偏差选项计算峰度
            if bias:
                res = mu4 / mu2**2 - 3
            else:
                n = a.shape[axis]
                # 根据无偏修正公式计算峰度
                res = (n-1) / ((n-2) * (n-3)) * ((n + 1) * mu4/mu2**2 - 3*(n-1))

            # 尽管减去再加回3看起来有些奇怪，但这比其他替代方案更简单
            return res if fisher else res + 3

        # 计算使用库函数 stats.kurtosis 得到的峰度值
        res = stats.kurtosis(x, axis=axis, bias=bias, fisher=fisher)
        # 计算自定义函数 kurtosis 得到的峰度值作为参考
        ref = kurtosis(x, axis=axis, bias=bias, fisher=fisher)
        # 检查两种方法得到的结果是否足够接近
        xp_assert_close(res, ref)
@hypothesis.strategies.composite
# 定义一个 hypothesis 的策略装饰器，用于生成复合数据类型
def ttest_data_axis_strategy(draw):
    # 从生成器 draw 中获取符合 shape 和 value 约束的数组
    elements = dict(allow_nan=False, allow_infinity=False)
    shape = npst.array_shapes(min_dims=1, min_side=2)
    
    # `test_pvalue_ci` 使用 `float64` 类型进行测试极端的 `alpha` 值。
    # 如果需要该策略生成其他类型的浮点数，则可以相应地调整。
    data = draw(npst.arrays(dtype=np.float64, elements=elements, shape=shape))

    # 确定能够准确计算非零方差的轴
    ok_axes = []

    # 在本地环境中，不需要使用 catch_warnings 或 simplefilter 来抑制 RuntimeWarning。
    # 这里包含这些设置是为了在 CI 中获得相同的行为。
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for axis in range(len(data.shape)):
            with contextlib.suppress(Exception):
                # 计算数据的二阶矩，沿指定轴
                var = stats.moment(data, order=2, axis=axis)
                # 如果所有的方差大于零且是有限的，则将该轴加入到有效轴列表中
                if np.all(var > 0) and np.all(np.isfinite(var)):
                    ok_axes.append(axis)
    
    # 如果没有有效的轴，通知 hypothesis 尝试不同的示例
    hypothesis.assume(ok_axes)

    # 从有效轴列表中随机选择一个轴
    axis = draw(hypothesis.strategies.sampled_from(ok_axes))

    return data, axis


@pytest.mark.skip_xp_backends(cpu_only=True,
                              reasons=['Uses NumPy for pvalue, CI'])
@pytest.mark.usefixtures("skip_xp_backends")
@array_api_compatible
# 测试类 TestStudentTest，兼容数组 API
class TestStudentTest:
    # 保留原始的测试用例。
    # 使用 R 中的 t.test 重新计算统计量和 p 值，例如：
    # options(digits=16)
    # t.test(c(-1., 0., 1.), mu=2)
    X1 = [-1., 0., 1.]
    X2 = [0., 1., 2.]
    T1_0 = 0.
    P1_0 = 1.
    T1_1 = -1.7320508075689
    P1_1 = 0.2254033307585
    T1_2 = -3.4641016151378
    P1_2 = 0.07417990022745
    T2_0 = 1.7320508075689
    P2_0 = 0.2254033307585
    P1_1_l = P1_1 / 2
    P1_1_g = 1 - (P1_1 / 2)
    # 定义一个单样本 t 检验的测试方法，参数 xp 是一个表示数组库的变量
    def test_onesample(self, xp):
        # 使用 suppress_warnings 上下文管理器来忽略特定的警告
        with suppress_warnings() as sup, \
                np.errstate(invalid="ignore", divide="ignore"):
            # 过滤运行时警告 "Degrees of freedom <= 0 for slice"
            sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
            # 根据是否为 NumPy 数组，创建不同的数组对象 a
            a = xp.asarray(4.) if not is_numpy(xp) else 4.
            # 进行单样本 t 检验，计算 t 值和 p 值
            t, p = stats.ttest_1samp(a, 3.)
        # 断言 t 的值为 NaN
        xp_assert_equal(t, xp.asarray(xp.nan))
        # 断言 p 的值为 NaN
        xp_assert_equal(p, xp.asarray(xp.nan))

        # 对 self.X1 执行单样本 t 检验，计算 t 值和 p 值
        t, p = stats.ttest_1samp(xp.asarray(self.X1), 0.)
        # 断言计算得到的 t 值与预期的 self.T1_0 数值非常接近
        xp_assert_close(t, xp.asarray(self.T1_0))
        # 断言计算得到的 p 值与预期的 self.P1_0 数值非常接近
        xp_assert_close(p, xp.asarray(self.P1_0))

        # 对 self.X1 执行单样本 t 检验，计算 t 值和 p 值
        res = stats.ttest_1samp(xp.asarray(self.X1), 0.)
        # 检查返回结果的指定属性（statistic 和 pvalue），使用自定义的检查函数
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes, xp=xp)

        # 对 self.X2 执行单样本 t 检验，计算 t 值和 p 值
        t, p = stats.ttest_1samp(xp.asarray(self.X2), 0.)
        # 断言计算得到的 t 值与预期的 self.T2_0 数值非常接近
        xp_assert_close(t, xp.asarray(self.T2_0))
        # 断言计算得到的 p 值与预期的 self.P2_0 数值非常接近
        xp_assert_close(p, xp.asarray(self.P2_0))

        # 对 self.X1 执行单样本 t 检验，计算 t 值和 p 值
        t, p = stats.ttest_1samp(xp.asarray(self.X1), 1.)
        # 断言计算得到的 t 值与预期的 self.T1_1 数值非常接近
        xp_assert_close(t, xp.asarray(self.T1_1))
        # 断言计算得到的 p 值与预期的 self.P1_1 数值非常接近
        xp_assert_close(p, xp.asarray(self.P1_1))

        # 对 self.X1 执行单样本 t 检验，计算 t 值和 p 值
        t, p = stats.ttest_1samp(xp.asarray(self.X1), 2.)
        # 断言计算得到的 t 值与预期的 self.T1_2 数值非常接近
        xp_assert_close(t, xp.asarray(self.T1_2))
        # 断言计算得到的 p 值与预期的 self.P1_2 数值非常接近
        xp_assert_close(p, xp.asarray(self.P1_2))

    # 定义一个测试单样本 t 检验的 NaN 策略方法，参数 xp 是一个表示数组库的变量
    def test_onesample_nan_policy(self, xp):
        # 检查 NaN 策略是否生效
        if not is_numpy(xp):
            # 如果不是 NumPy 数组，创建包含 NaN 的数组 x
            x = xp.asarray([1., 2., 3., xp.nan])
            # 定义错误消息
            message = "Use of `nan_policy` and `keepdims`..."
            # 断言使用 nan_policy='omit' 时抛出 NotImplementedError 异常
            with pytest.raises(NotImplementedError, match=message):
                stats.ttest_1samp(x, 1., nan_policy='omit')
            return

        # 使用正态分布生成包含 NaN 的数组 x
        x = stats.norm.rvs(loc=5, scale=10, size=51, random_state=7654567)
        x[50] = np.nan
        # 忽略无效值错误，执行单样本 t 检验，检验均值为 5.0
        with np.errstate(invalid="ignore"):
            # 断言执行 t 检验得到的结果是 (NaN, NaN)
            assert_array_equal(stats.ttest_1samp(x, 5.0), (np.nan, np.nan))

            # 使用 nan_policy='omit' 执行 t 检验，检验均值为 5.0
            assert_array_almost_equal(stats.ttest_1samp(x, 5.0, nan_policy='omit'),
                                      (-1.6412624074367159, 0.107147027334048005))
            # 断言使用不支持的 nan_policy='raise' 时抛出 ValueError 异常
            assert_raises(ValueError, stats.ttest_1samp, x, 5.0, nan_policy='raise')
            # 断言使用不支持的 nan_policy='foobar' 时抛出 ValueError 异常
            assert_raises(ValueError, stats.ttest_1samp, x, 5.0,
                          nan_policy='foobar')

    # 定义一个测试单样本 t 检验的 alternative 参数方法，参数 xp 是一个表示数组库的变量
    def test_1samp_alternative(self, xp):
        # 定义错误消息
        message = "`alternative` must be 'less', 'greater', or 'two-sided'."
        # 断言当 alternative 参数不合法时抛出 ValueError 异常
        with pytest.raises(ValueError, match=message):
            stats.ttest_1samp(xp.asarray(self.X1), 0., alternative="error")

        # 使用 alternative='less' 执行单样本 t 检验，检验均值为 1.0
        t, p = stats.ttest_1samp(xp.asarray(self.X1), 1., alternative="less")
        # 断言计算得到的 p 值与预期的 self.P1_1_l 数值非常接近
        xp_assert_close(p, xp.asarray(self.P1_1_l))
        # 断言计算得到的 t 值与预期的 self.T1_1 数值非常接近
        xp_assert_close(t, xp.asarray(self.T1_1))

        # 使用 alternative='greater' 执行单样本 t 检验，检验均值为 1.0
        t, p = stats.ttest_1samp(xp.asarray(self.X1), 1., alternative="greater")
        # 断言计算得到的 p 值与预期的 self.P1_1_g 数值非常接近
        xp_assert_close(p, xp.asarray(self.P1_1_g))
        # 断言计算得到的 t 值与预期的 self.T1_1 数值非常接近
        xp_assert_close(t, xp.asarray(self.T1_1))

    # 使用参数化测试，测试单样本 t 检验的 alternative 参数
    @pytest.mark.parametrize("alternative", ['two-sided', 'less', 'greater'])
    # 对于给定的一维数据进行单样本 t 检验的置信区间测试
    def test_1samp_ci_1d(self, xp, alternative):
        # 使用特定种子生成随机数发生器
        rng = np.random.default_rng(8066178009154342972)
        # 设定样本数量为 10，从正态分布中生成数据，均值为 1.5，标准差为 2
        n = 10
        x = rng.normal(size=n, loc=1.5, scale=2)
        # 生成一个额外的总体均值，但不应影响置信区间计算
        popmean = rng.normal()
        # 参考值是使用 R 的 t.test 生成的：
        # options(digits=16)
        # x = c(2.75532884,  0.93892217,  0.94835861,  1.49489446, -0.62396595,
        #      -1.88019867, -1.55684465,  4.88777104,  5.15310979,  4.34656348)
        # t.test(x, conf.level=0.85, alternative='l')
        # 如果是 PyTorch 数组，则使用 float32 类型；否则使用 float64 类型
        dtype = xp.float32 if is_torch(xp) else xp.float64  # 使用默认的数据类型
        x = xp.asarray(x, dtype=dtype)
        popmean = xp.asarray(popmean, dtype=dtype)

        # 参考值字典，包含了两侧、大于和小于三种替代方案的置信区间
        ref = {'two-sided': [0.3594423211709136, 2.9333455028290860],
               'greater': [0.7470806207371626, np.inf],
               'less': [-np.inf, 2.545707203262837]}
        # 计算单样本 t 检验，并获取置信区间
        res = stats.ttest_1samp(x, popmean=popmean, alternative=alternative)
        ci = res.confidence_interval(confidence_level=0.85)
        # 使用测试框架断言置信区间的下限和上限与参考值相近
        xp_assert_close(ci.low, xp.asarray(ref[alternative][0]))
        xp_assert_close(ci.high, xp.asarray(ref[alternative][1]))
        # 使用测试框架断言自由度与 n-1 相等
        xp_assert_equal(res.df, xp.asarray(n-1))

    # 测试 `confidence_interval` 方法的输入验证
    def test_1samp_ci_iv(self, xp):
        res = stats.ttest_1samp(xp.arange(10.), 0.)
        # 验证在 `confidence_level` 超出范围时会抛出 ValueError 异常
        message = '`confidence_level` must be a number between 0 and 1.'
        with pytest.raises(ValueError, match=message):
            res.confidence_interval(confidence_level=10)

    # 使用假设推断测试框架来测试单侧 p 值与置信区间之间的关系
    @pytest.mark.xslow
    @hypothesis.given(alpha=hypothesis.strategies.floats(1e-15, 1-1e-15),
                      data_axis=ttest_data_axis_strategy())
    @pytest.mark.parametrize('alternative', ['less', 'greater'])
    def test_pvalue_ci(self, alpha, data_axis, alternative, xp):
        # 获取数据和轴信息，并将数据转换为指定的数组类型
        data, axis = data_axis
        data = xp.asarray(data)
        # 进行单样本 t 检验，计算 p 值和置信区间
        res = stats.ttest_1samp(data, 0.,
                                alternative=alternative, axis=axis)
        l, u = res.confidence_interval(confidence_level=alpha)
        # 根据替代方案确定总体均值
        popmean = l if alternative == 'greater' else u
        # 根据测试框架要求，调整数据维度以匹配 PyTorch 的需求
        xp_test = array_namespace(l)  # torch 需要 `expand_dims`
        popmean = xp_test.expand_dims(popmean, axis=axis)
        # 再次进行单样本 t 检验，此时使用调整后的总体均值和替代方案
        res = stats.ttest_1samp(data, popmean, alternative=alternative, axis=axis)
        shape = list(data.shape)
        shape.pop(axis)
        # 生成一个参考值，以匹配极端范围的 `alpha`
        ref = xp.broadcast_to(xp.asarray(1-alpha, dtype=xp.float64), shape)
        # 使用测试框架断言计算出的 p 值与参考值相近
        xp_assert_close(res.pvalue, ref)
# 定义一个名为 TestPercentileOfScore 的测试类，用于测试统计函数的百分位数计算

class TestPercentileOfScore:

    # 定义一个方法 f，接受任意位置参数和关键字参数，并调用 stats 模块的 percentileofscore 函数进行计算
    def f(self, *args, **kwargs):
        return stats.percentileofscore(*args, **kwargs)

    # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试数据
    @pytest.mark.parametrize("kind, result", [("rank", 40),
                                              ("mean", 35),
                                              ("strict", 30),
                                              ("weak", 40)])
    # 定义一个测试方法 test_unique，测试给定列表 a 中特定百分位数计算的准确性
    def test_unique(self, kind, result):
        # 创建一个列表 a
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # 断言调用方法 f 计算出的特定百分位数等于预期结果 result
        assert_equal(self.f(a, 4, kind=kind), result)

    @pytest.mark.parametrize("kind, result", [("rank", 45),
                                              ("mean", 40),
                                              ("strict", 30),
                                              ("weak", 50)])
    # 定义一个测试方法 test_multiple2，测试给定列表 a 中特定百分位数计算的准确性（包含重复元素）
    def test_multiple2(self, kind, result):
        # 创建一个列表 a
        a = [1, 2, 3, 4, 4, 5, 6, 7, 8, 9]
        # 断言调用方法 f 计算出的特定百分位数等于预期结果 result
        assert_equal(self.f(a, 4, kind=kind), result)

    @pytest.mark.parametrize("kind, result", [("rank", 50),
                                              ("mean", 45),
                                              ("strict", 30),
                                              ("weak", 60)])
    # 定义一个测试方法 test_multiple3，测试给定列表 a 中特定百分位数计算的准确性（包含更多重复元素）
    def test_multiple3(self, kind, result):
        # 创建一个列表 a
        a = [1, 2, 3, 4, 4, 4, 5, 6, 7, 8]
        # 断言调用方法 f 计算出的特定百分位数等于预期结果 result
        assert_equal(self.f(a, 4, kind=kind), result)

    @pytest.mark.parametrize("kind, result", [("rank", 30),
                                              ("mean", 30),
                                              ("strict", 30),
                                              ("weak", 30)])
    # 定义一个测试方法 test_missing，测试给定列表 a 中特定百分位数计算的准确性（目标值缺失）
    def test_missing(self, kind, result):
        # 创建一个列表 a
        a = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        # 断言调用方法 f 计算出的特定百分位数等于预期结果 result
        assert_equal(self.f(a, 4, kind=kind), result)

    @pytest.mark.parametrize("kind, result", [("rank", 40),
                                              ("mean", 35),
                                              ("strict", 30),
                                              ("weak", 40)])
    # 定义一个测试方法 test_large_numbers，测试给定列表 a 中特定百分位数计算的准确性（大数字集合）
    def test_large_numbers(self, kind, result):
        # 创建一个列表 a
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # 断言调用方法 f 计算出的特定百分位数等于预期结果 result
        assert_equal(self.f(a, 40, kind=kind), result)

    @pytest.mark.parametrize("kind, result", [("rank", 50),
                                              ("mean", 45),
                                              ("strict", 30),
                                              ("weak", 60)])
    # 定义一个测试方法 test_large_numbers_multiple3，测试给定列表 a 中特定百分位数计算的准确性（大数字集合，包含更多重复元素）
    def test_large_numbers_multiple3(self, kind, result):
        # 创建一个列表 a
        a = [10, 20, 30, 40, 40, 40, 50, 60, 70, 80]
        # 断言调用方法 f 计算出的特定百分位数等于预期结果 result
        assert_equal(self.f(a, 40, kind=kind), result)

    @pytest.mark.parametrize("kind, result", [("rank", 30),
                                              ("mean", 30),
                                              ("strict", 30),
                                              ("weak", 30)])
    # 定义一个测试方法 test_large_numbers_missing，测试给定列表 a 中特定百分位数计算的准确性（大数字集合，目标值缺失）
    def test_large_numbers_missing(self, kind, result):
        # 创建一个列表 a
        a = [10, 20, 30, 50, 60, 70, 80, 90, 100, 110]
        # 断言调用方法 f 计算出的特定百分位数等于预期结果 result
        assert_equal(self.f(a, 40, kind=kind), result)
    # 使用 pytest 模块的 parametrize 装饰器，定义多组测试参数，每组参数包括 kind 和 result
    @pytest.mark.parametrize("kind, result", [("rank", [0, 10, 100, 100]),
                                              ("mean", [0, 5, 95, 100]),
                                              ("strict", [0, 0, 90, 100]),
                                              ("weak", [0, 10, 100, 100])])
    # 定义测试方法 test_boundaries，对输入列表 a 进行测试，并断言其结果等于指定的 result
    def test_boundaries(self, kind, result):
        a = [10, 20, 30, 50, 60, 70, 80, 90, 100, 110]
        assert_equal(self.f(a, [0, 10, 110, 200], kind=kind), result)
    
    # 使用 pytest 模块的 parametrize 装饰器，定义多组测试参数，每组参数包括 kind 和 result
    @pytest.mark.parametrize("kind, result", [("rank", [0, 10, 100]),
                                              ("mean", [0, 5, 95]),
                                              ("strict", [0, 0, 90]),
                                              ("weak", [0, 10, 100])])
    # 定义测试方法 test_inf，对输入列表 a 进行测试，并断言其结果等于指定的 result
    def test_inf(self, kind, result):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, +np.inf]
        assert_equal(self.f(a, [-np.inf, 1, +np.inf], kind=kind), result)
    
    # 定义测试用例列表 cases，包含多组测试参数，每组参数为 policy, a, score, result
    cases = [("propagate", [], 1, np.nan),
             ("propagate", [np.nan], 1, np.nan),
             ("propagate", [np.nan], [0, 1, 2], [np.nan, np.nan, np.nan]),
             ("propagate", [1, 2], [1, 2, np.nan], [50, 100, np.nan]),
             ("omit", [1, 2, np.nan], [0, 1, 2], [0, 50, 100]),
             ("omit", [1, 2], [0, 1, np.nan], [0, 50, np.nan]),
             ("omit", [np.nan, np.nan], [0, 1, 2], [np.nan, np.nan, np.nan])]
    
    # 使用 pytest 模块的 parametrize 装饰器，定义多组测试参数，每组参数为 policy, a, score, result
    @pytest.mark.parametrize("policy, a, score, result", cases)
    # 定义测试方法 test_nans_ok，对输入列表 a 进行测试，并断言其结果等于指定的 result
    def test_nans_ok(self, policy, a, score, result):
        assert_equal(self.f(a, score, nan_policy=policy), result)
    
    # 定义测试用例列表 cases，包含多组测试参数，每组参数为 policy, a, score, message
    cases = [
        ("raise", [1, 2, 3, np.nan], [1, 2, 3],
         "The input contains nan values"),
        ("raise", [1, 2, 3], [1, 2, 3, np.nan],
         "The input contains nan values"),
    ]
    
    # 使用 pytest 模块的 parametrize 装饰器，定义多组测试参数，每组参数为 policy, a, score, message
    @pytest.mark.parametrize("policy, a, score, message", cases)
    # 定义测试方法 test_nans_fail，对输入列表 a 进行测试，预期引发 ValueError 异常，并匹配特定的错误信息 message
    def test_nans_fail(self, policy, a, score, message):
        with assert_raises(ValueError, match=message):
            self.f(a, score, nan_policy=policy)
    
    # 使用 pytest 模块的 parametrize 装饰器，定义多组测试参数，每组参数为 shape，即不同的数组形状
    @pytest.mark.parametrize("shape", [
        (6, ),
        (2, 3),
        (2, 1, 3),
        (2, 1, 1, 3),
    ])
    # 定义测试方法 test_nd，对输入列表 a 和 scores 进行测试，并断言其结果等于预期的 results
    def test_nd(self, shape):
        a = np.array([0, 1, 2, 3, 4, 5])
        scores = a.reshape(shape)
        results = scores * 10
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert_equal(self.f(a, scores, kind="rank"), results)
# 命名元组的定义，用于表示统计检验用例的结构
PowerDivCase = namedtuple('Case',  # type: ignore[name-match]
                          ['f_obs', 'f_exp', 'ddof', 'axis',
                           'chi2',     # Pearson's
                           'log',      # G-test (log-likelihood)
                           'mod_log',  # Modified log-likelihood
                           'cr',       # Cressie-Read (lambda=2/3)
                           ])

# power_div_1d_cases 是一个包含 PowerDivCase 元组对象的列表，每个对象代表一个统计检验用例
# 每个用例包含不同的 f_obs（观测值）、f_exp（期望值）、自由度（ddof）、轴（axis）等信息
power_div_1d_cases = [
    # 第一个用例，使用默认的 f_exp
    PowerDivCase(f_obs=[4, 8, 12, 8], f_exp=None, ddof=0, axis=None,
                 chi2=4,
                 log=2*(4*np.log(4/8) + 12*np.log(12/8)),
                 mod_log=2*(8*np.log(8/4) + 8*np.log(8/12)),
                 cr=(4*((4/8)**(2/3) - 1) + 12*((12/8)**(2/3) - 1))/(5/9)),
    # 第二个用例，给出非均匀的 f_exp
    PowerDivCase(f_obs=[4, 8, 12, 8], f_exp=[2, 16, 12, 2], ddof=0, axis=None,
                 chi2=24,
                 log=2*(4*np.log(4/2) + 8*np.log(8/16) + 8*np.log(8/2)),
                 mod_log=2*(2*np.log(2/4) + 16*np.log(16/8) + 2*np.log(2/8)),
                 cr=(4*((4/2)**(2/3) - 1) + 8*((8/16)**(2/3) - 1) +
                     8*((8/2)**(2/3) - 1))/(5/9)),
    # 第三个用例，f_exp 是一个标量
    PowerDivCase(f_obs=[4, 8, 12, 8], f_exp=8, ddof=0, axis=None,
                 chi2=4,
                 log=2*(4*np.log(4/8) + 12*np.log(12/8)),
                 mod_log=2*(8*np.log(8/4) + 8*np.log(8/12)),
                 cr=(4*((4/8)**(2/3) - 1) + 12*((12/8)**(2/3) - 1))/(5/9)),
    # 第四个用例，f_exp 等于 f_obs
    PowerDivCase(f_obs=[3, 5, 7, 9], f_exp=[3, 5, 7, 9], ddof=0, axis=0,
                 chi2=0, log=0, mod_log=0, cr=0),
]

# power_div_empty_cases 是另一个包含 PowerDivCase 元组对象的列表，用于测试空数据集的情况
# 每个用例中，f_obs 是空的，期望值和其他参数保持不变，用于检验统计值是否为 0
power_div_empty_cases = [
    # 形状为 (0,)，长度为 0 的数据集，计算出的测试统计量应为 0
    PowerDivCase(f_obs=[],
                 f_exp=None, ddof=0, axis=0,
                 chi2=0, log=0, mod_log=0, cr=0),
    # 形状为 (0, 3)，包含 3 个长度为 0 的数据集，计算出的测试统计量应为 [0, 0, 0]
    PowerDivCase(f_obs=np.array([[],[],[]]).T,
                 f_exp=None, ddof=0, axis=0,
                 chi2=[0, 0, 0],
                 log=[0, 0, 0],
                 mod_log=[0, 0, 0],
                 cr=[0, 0, 0]),
    # 形状为 (3, 0)，表示包含 3 个长度为 3 的空数据集，测试统计量应该是一个空数组
    PowerDivCase(f_obs=np.array([[],[],[]]),
                 f_exp=None, ddof=0, axis=0,
                 chi2=[],
                 log=[],
                 mod_log=[],
                 cr=[]),
]

# TestPowerDivergence 是一个类装饰器，用于标记下面的类是与数组 API 兼容的统计检验类
@array_api_compatible
class TestPowerDivergence:
    def check_power_divergence(self, f_obs, f_exp, ddof, axis, lambda_,
                               expected_stat, xp):
        # 获取数据类型，xp 是外部传入的数值计算库（例如 numpy 或 torch）的命名空间
        dtype = xp.asarray(1.).dtype

        # 将 f_obs 转换为 xp 数组，使用指定的数据类型
        f_obs = xp.asarray(f_obs, dtype=dtype)
        # 如果 f_exp 不为 None，则将其转换为 xp 数组，使用指定的数据类型
        f_exp = xp.asarray(f_exp, dtype=dtype) if f_exp is not None else f_exp

        # 如果 axis 为 None，则计算 f_obs 的大小
        if axis is None:
            num_obs = xp_size(f_obs)
        else:
            # array_namespace 是一个函数或对象，用于处理数组的广播操作（例如 torch 需要这样的操作）
            xp_test = array_namespace(f_obs)
            # 使用 xp_test 对象进行数组的广播操作，返回广播后的数组元组
            arrays = (xp_test.broadcast_arrays(f_obs, f_exp) if f_exp is not None
                      else (f_obs,))
            # 获取指定 axis 维度上的数组形状
            num_obs = arrays[0].shape[axis]

        # 使用 suppress_warnings 上下文管理器，过滤 RuntimeWarning 类型的警告信息
        with suppress_warnings() as sup:
            # 在上下文中调用 stats.power_divergence 函数，计算统计量 stat 和 p 值
            stat, p = stats.power_divergence(
                                f_obs=f_obs, f_exp=f_exp, ddof=ddof,
                                axis=axis, lambda_=lambda_)
            # 使用 xp_assert_close 函数断言 stat 的值等于预期的统计量 expected_stat
            xp_assert_close(stat, xp.asarray(expected_stat, dtype=dtype))

            # 如果 lambda_ 的值为 1 或 "pearson"，则进行额外的 chi-square 检验
            if lambda_ == 1 or lambda_ == "pearson":
                # 调用 stats.chisquare 函数，计算统计量 stat 和 p 值
                stat, p = stats.chisquare(f_obs=f_obs, f_exp=f_exp, ddof=ddof,
                                          axis=axis)
                # 使用 xp_assert_close 函数断言 stat 的值等于预期的统计量 expected_stat
                xp_assert_close(stat, xp.asarray(expected_stat, dtype=dtype))

        # 将 ddof 转换为 numpy 数组
        ddof = np.asarray(ddof)
        # 计算预期的 p 值，使用 stats.distributions.chi2.sf 函数
        expected_p = stats.distributions.chi2.sf(expected_stat,
                                                 num_obs - 1 - ddof)
        # 使用 xp_assert_close 函数断言 p 的值等于预期的 p 值
        xp_assert_close(p, xp.asarray(expected_p, dtype=dtype))

    # 使用 pytest.mark.parametrize 装饰器，将 power_div_1d_cases 中的每个 case 作为测试参数化
    @pytest.mark.parametrize('case', power_div_1d_cases)
    # 使用 pytest.mark.parametrize 装饰器，将 lambda_stat 中的每个元组作为测试参数化
    @pytest.mark.parametrize('lambda_stat',
        [(None, 'chi2'), ('pearson', 'chi2'), (1, 'chi2'),
         ('log-likelihood', 'log'), ('mod-log-likelihood', 'mod_log'),
         ('cressie-read', 'cr'), (2/3, 'cr')])
    # 测试方法，使用 xp 作为数值计算库的命名空间
    def test_basic(self, case, lambda_stat, xp):
        # 解包 lambda_stat 元组，获取 lambda_ 和 attr 值
        lambda_, attr = lambda_stat
        # 获取 case 对象中与 attr 名称对应的属性值，作为预期统计量
        expected_stat = getattr(case, attr)
        # 调用 check_power_divergence 方法，进行统计量检验
        self.check_power_divergence(case.f_obs, case.f_exp, case.ddof, case.axis,
                                    lambda_, expected_stat, xp)
    # 定义一个测试方法，用于测试不同的计算路径在 power_divergence 函数中的表现，使用了 xp 作为数组操作的命名空间。
    def test_axis(self, xp):
        # 从 power_div_1d_cases 中取出两个测试用例
        case0 = power_div_1d_cases[0]
        case1 = power_div_1d_cases[1]
        # 将两个测试用例的观察频数按行堆叠成一个 2D 数组
        f_obs = np.vstack((case0.f_obs, case1.f_obs))
        # 构建期望频数的数组，使用 case0 的均值填充第一行，case1 的期望频数作为第二行
        f_exp = np.vstack((np.ones_like(case0.f_obs)*np.mean(case0.f_obs),
                           case1.f_exp))
        # 检查 power_divergence 函数在 axis=1 的四个计算路径
        self.check_power_divergence(
               f_obs, f_exp, 0, 1,
               "pearson", [case0.chi2, case1.chi2], xp=xp)
        self.check_power_divergence(
               f_obs, f_exp, 0, 1,
               "log-likelihood", [case0.log, case1.log], xp=xp)
        self.check_power_divergence(
               f_obs, f_exp, 0, 1,
               "mod-log-likelihood", [case0.mod_log, case1.mod_log], xp=xp)
        self.check_power_divergence(
               f_obs, f_exp, 0, 1,
               "cressie-read", [case0.cr, case1.cr], xp=xp)
        # 将 case0.f_obs 重塑为形状为 (2,2) 的数组，并使用 axis=None 进行检查
        f_obs_reshape = xp.reshape(xp.asarray(case0.f_obs), (2, 2))
        self.check_power_divergence(
               f_obs_reshape, None, 0, None,
               "pearson", case0.chi2, xp=xp)

    # 定义一个测试方法，测试 ddof 的广播功能
    def test_ddof_broadcasting(self, xp):
        # 测试 ddof 是否正确广播
        # ddof 不影响检验统计量，它会与计算出的检验统计量广播以计算 p 值

        # 从 power_div_1d_cases 中取出两个测试用例
        case0 = power_div_1d_cases[0]
        case1 = power_div_1d_cases[1]
        # 创建 4x2 的观察频数和期望频数的数组，转置后进行堆叠
        f_obs = np.vstack((case0.f_obs, case1.f_obs)).T
        f_exp = np.vstack((np.ones_like(case0.f_obs)*np.mean(case0.f_obs),
                           case1.f_exp)).T

        # 期望的卡方值列表
        expected_chi2 = [case0.chi2, case1.chi2]

        # 将 f_obs、f_exp 和 expected_chi2 转换为 xp 的数组类型
        dtype = xp.asarray(1.).dtype
        f_obs = xp.asarray(f_obs, dtype=dtype)
        f_exp = xp.asarray(f_exp, dtype=dtype)
        expected_chi2 = xp.asarray(expected_chi2, dtype=dtype)

        # ddof 的形状为 (2, 1)，会与计算出的统计量广播，因此 p 的形状为 (2,2)
        ddof = xp.asarray([[0], [1]])

        # 计算统计量 stat 和 p 值 p
        stat, p = stats.power_divergence(f_obs, f_exp, ddof=ddof)
        xp_assert_close(stat, expected_chi2)

        # 分别计算 p 值，传入标量作为 ddof
        stat0, p0 = stats.power_divergence(f_obs, f_exp, ddof=ddof[0, 0])
        stat1, p1 = stats.power_divergence(f_obs, f_exp, ddof=ddof[1, 0])

        # 构建预期的 p 值数组 expected_p
        xp_test = array_namespace(f_obs)  # 需要 `concat`, `newaxis`
        expected_p = xp_test.concat((p0[xp_test.newaxis, :],
                                     p1[xp_test.newaxis, :]),
                                    axis=0)
        xp_assert_close(p, expected_p)

    # 使用 power_div_empty_cases 参数化 pytest 测试
    @pytest.mark.parametrize('case', power_div_empty_cases)
    @pytest.mark.parametrize('lambda_stat',
        [('pearson', 'chi2'), ('log-likelihood', 'log'),
         ('mod-log-likelihood', 'mod_log'),
         ('cressie-read', 'cr'), (2/3, 'cr')])
    # 使用 pytest 的参数化功能，为 lambda_stat 参数化不同的测试参数组合
    def test_empty_cases(self, case, lambda_stat, xp):
        lambda_, attr = lambda_stat
        # 从 lambda_stat 元组中获取 lambda_ 和 attr
        expected_stat = getattr(case, attr)
        # 使用 getattr 获取 case 对象中 attr 所指定的属性值，存储在 expected_stat 中
        with warnings.catch_warnings():
            # 捕获警告信息
            self.check_power_divergence(
                case.f_obs, case.f_exp, case.ddof, case.axis,
                lambda_, expected_stat, xp)
                # 调用 check_power_divergence 方法，传入相关参数进行测试

    def test_power_divergence_result_attributes(self, xp):
        f_obs = power_div_1d_cases[0].f_obs
        f_exp = power_div_1d_cases[0].f_exp
        ddof = power_div_1d_cases[0].ddof
        axis = power_div_1d_cases[0].axis
        dtype = xp.asarray(1.).dtype
        f_obs = xp.asarray(f_obs, dtype=dtype)
        # 将 f_obs 转换为 xp 的数组，并指定 dtype
        # f_exp is None

        res = stats.power_divergence(f_obs=f_obs, f_exp=f_exp, ddof=ddof,
                                     axis=axis, lambda_="pearson")
        # 调用 stats 中的 power_divergence 函数进行统计测试，传入相关参数
        attributes = ('statistic', 'pvalue')
        # 定义需要检查的结果属性列表
        check_named_results(res, attributes, xp=xp)
        # 调用 check_named_results 函数，检查 res 中是否包含指定的属性，使用 xp 进行验证

    def test_power_divergence_gh_12282(self, xp):
        # The sums of observed and expected frequencies must match
        # 观察频数和期望频数的总和必须匹配
        f_obs = xp.asarray([[10., 20.], [30., 20.]])
        f_exp = xp.asarray([[5., 15.], [35., 25.]])
        message = 'For each axis slice...'
        # 定义错误消息
        with pytest.raises(ValueError, match=message):
            # 使用 pytest 的 raises 断言，期望捕获 ValueError 异常，并匹配特定消息
            stats.power_divergence(f_obs=f_obs, f_exp=xp.asarray([30., 60.]))
            # 调用 stats 中的 power_divergence 函数，传入不匹配的 f_exp 进行测试
        with pytest.raises(ValueError, match=message):
            # 同上，针对不同的条件进行测试
            stats.power_divergence(f_obs=f_obs, f_exp=f_exp, axis=1)
            # 调用 stats 中的 power_divergence 函数，传入 axis=1 进行测试
        stat, pval = stats.power_divergence(f_obs=f_obs, f_exp=f_exp)
        # 调用 stats 中的 power_divergence 函数，传入 f_obs 和 f_exp 进行统计计算
        xp_assert_close(stat, xp.asarray([5.71428571, 2.66666667]))
        # 使用 xp_assert_close 函数验证 stat 的计算结果是否与给定值接近
        xp_assert_close(pval, xp.asarray([0.01682741, 0.10247043]))
        # 使用 xp_assert_close 函数验证 pval 的计算结果是否与给定值接近
    def test_power_divergence_against_cressie_read_data(self, xp):
        # 测试 stats.power_divergence 函数与 Cressie 和 Read 的数据表4和表5进行比较
        # 参考文献："Multimonial Goodness-of-Fit Tests", J. R. Statist. Soc. B (1984), Vol 46, No. 3, pp. 440-464.
        # 这个测试计算了几个 lambda 值的统计量。

        # 表4的数据根据更高精度重新计算，参考文献：
        # Shelby J. Haberman, Analysis of Qualitative Data: Volume 1
        # Introductory Topics, Academic Press, New York, USA (1978).
        # obs 数组包含了观察到的频数
        obs = xp.asarray([15., 11., 14., 17., 5., 11., 10., 4., 8.,
                          10., 7., 9., 11., 3., 6., 1., 1., 4.])
        beta = -0.083769  # Haberman (1978), p. 15
        # i 数组包含了从1到观测次数的序号
        i = xp.arange(1., obs.shape[0] + 1.)
        # 计算 alpha 值
        alpha = xp.log(xp.sum(obs) / xp.sum(xp.exp(beta*i)))
        # 计算期望频数
        expected_counts = xp.exp(alpha + beta*i)

        # `table4` 包含了表4的第二列和第三列数据
        xp_test = array_namespace(obs)  # NumPy 需要 concat，torch 需要 newaxis
        # 拼接 obs 和 expected_counts 成为 table4 的格式
        table4 = xp_test.concat((obs[xp_test.newaxis, :],
                                 expected_counts[xp_test.newaxis, :])).T

        # table5 包含了表5的数据，lambda 和统计量
        table5 = xp.asarray([
            # lambda, statistic
            -10.0, 72.2e3,
            -5.0, 28.9e1,
            -3.0, 65.6,
            -2.0, 40.6,
            -1.5, 34.0,
            -1.0, 29.5,
            -0.5, 26.5,
            0.0, 24.6,
            0.5, 23.4,
            0.67, 23.1,
            1.0, 22.7,
            1.5, 22.6,
            2.0, 22.9,
            3.0, 24.8,
            5.0, 35.5,
            10.0, 21.4e1,
            ])
        # 将 table5 重新整形为 (n, 2) 形状的数组
        table5 = xp.reshape(table5, (-1, 2))

        # 对于每一行数据，计算统计量和 p 值，然后进行断言比较
        for i in range(table5.shape[0]):
            lambda_, expected_stat = table5[i, 0], table5[i, 1]
            stat, p = stats.power_divergence(table4[:,0], table4[:,1],
                                             lambda_=lambda_)
            xp_assert_close(stat, expected_stat, rtol=5e-3)
@array_api_compatible
class TestChisquare:
    # 定义测试类 TestChisquare，用于检验卡方检验函数的功能

    def test_gh_chisquare_12282(self, xp):
        # 测试函数，用于验证 chisquare 是否通过 power_divergence 实现
        # 如果实现方式改变，可以使用类似 test_power_divergence_gh_12282 的基本测试
        with assert_raises(ValueError, match='For each axis slice...'):
            # 使用 assert_raises 检查 ValueError 是否会被引发，并且错误信息匹配 'For each axis slice...'
            f_obs = xp.asarray([10., 20.])
            # 使用 xp.asarray 将列表转换为数组 f_obs
            f_exp = xp.asarray([30., 60.])
            # 使用 xp.asarray 将列表转换为数组 f_exp
            stats.chisquare(f_obs=f_obs, f_exp=f_exp)
            # 调用 stats.chisquare 进行卡方检验

    @pytest.mark.parametrize("n, dtype", [(200, 'uint8'), (1000000, 'int32')])
    def test_chiquare_data_types_attributes(self, n, dtype, xp):
        # 参数化测试，用于测试不同的数据类型和属性
        # 回归测试 gh-10159 和 gh-18368
        dtype = getattr(xp, dtype)
        # 获取 xp 对象中对应的数据类型
        obs = xp.asarray([n, 0], dtype=dtype)
        # 使用 xp.asarray 创建数组 obs，并指定数据类型为 dtype
        exp = xp.asarray([n // 2, n // 2], dtype=dtype)
        # 使用 xp.asarray 创建数组 exp，并指定数据类型为 dtype
        res = stats.chisquare(obs, exp)
        # 调用 stats.chisquare 进行卡方检验，并将结果保存在 res 中
        stat, p = res
        # 将 res 的结果分别解包给 stat 和 p
        xp_assert_close(stat, xp.asarray(n, dtype=xp.asarray(1.).dtype), rtol=1e-13)
        # 使用 xp_assert_close 检查 stat 是否接近于 xp.asarray(n, dtype=xp.asarray(1.).dtype)，相对误差为 1e-13
        # xp.asarray(1.).dtype 获取 1 的数据类型，用于比较 stat 的数据类型
        xp_assert_equal(res.statistic, stat)
        # 使用 xp_assert_equal 检查 res.statistic 是否等于 stat
        xp_assert_equal(res.pvalue, p)
        # 使用 xp_assert_equal 检查 res.pvalue 是否等于 p

@skip_xp_invalid_arg
class TestChisquareMA:
    # 跳过无效参数的测试类 TestChisquareMA

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_deprecation_warning(self):
        # 测试函数，用于验证 DeprecationWarning 是否被正确引发
        a = np.asarray([1., 2., 3.])
        # 创建数组 a
        ma = np.ma.masked_array(a)
        # 创建掩码数组 ma
        message = "`power_divergence` and `chisquare` support for masked..."
        # 提示信息字符串
        with pytest.warns(DeprecationWarning, match=message):
            # 使用 pytest.warns 检查是否引发 DeprecationWarning，并且错误信息匹配 message
            stats.chisquare(ma)
            # 调用 stats.chisquare 进行卡方检验
        with pytest.warns(DeprecationWarning, match=message):
            # 使用 pytest.warns 检查是否引发 DeprecationWarning，并且错误信息匹配 message
            stats.chisquare(a, ma)
            # 再次调用 stats.chisquare 进行卡方检验，使用掩码数组 ma

def test_friedmanchisquare():
    # 测试 Friedman 卡方检验函数

    # see ticket:113
    # 参考票号 113

    # verified with matlab and R
    # 使用 matlab 和 R 进行验证

    # From Demsar "Statistical Comparisons of Classifiers over Multiple Data Sets"
    # 2006, Xf=9.28 (no tie handling, tie corrected Xf >=9.28)
    # Demsar 的论文中提到，对多个数据集的分类器进行统计比较，确认 Xf=9.28（未处理平局情况，修正后 Xf >= 9.28）

    x1 = [array([0.763, 0.599, 0.954, 0.628, 0.882, 0.936, 0.661, 0.583,
                 0.775, 1.0, 0.94, 0.619, 0.972, 0.957]),
          array([0.768, 0.591, 0.971, 0.661, 0.888, 0.931, 0.668, 0.583,
                 0.838, 1.0, 0.962, 0.666, 0.981, 0.978]),
          array([0.771, 0.590, 0.968, 0.654, 0.886, 0.916, 0.609, 0.563,
                 0.866, 1.0, 0.965, 0.614, 0.9751, 0.946]),
          array([0.798, 0.569, 0.967, 0.657, 0.898, 0.931, 0.685, 0.625,
                 0.875, 1.0, 0.962, 0.669, 0.975, 0.970])]
    # 定义数据集 x1，包含四个数组，每个数组代表一组观测值

    # From "Bioestadistica para las ciencias de la salud" Xf=18.95 p<0.001:
    # 从健康科学的生物统计学中，得到 Xf=18.95，p < 0.001

    x2 = [array([4,3,5,3,5,3,2,5,4,4,4,3]),
          array([2,2,1,2,3,1,2,3,2,1,1,3]),
          array([2,4,3,3,4,3,3,4,4,1,2,1]),
          array([3,5,4,3,4,4,3,3,3,4,4,4])]
    # 定义数据集 x2，包含四个数组，每个数组代表一组观测值

    # From Jerrorl H. Zar, "Biostatistical Analysis"(example 12.6),
    # Xf=10.68, 0.005 < p < 0.01:
    # 从 Zar 的生物统计分析中得到 Xf=10.68，0.005 < p < 0.01

    # Probability from this example is inexact
    # using Chisquare approximation of Friedman Chisquare.
    # 此例中的概率是不精确的，使用 Friedman 卡方检验的卡方逼近方法。

    x3 = [array([7.0,9.9,8.5,5.1,10.3]),
          array([5.3,5.7,4.7,3.5,7.7]),
          array([4.9,7.6,5.5,2.8,8.4]),
          array([8.8,8.9,8.1,3.3,9.1])]
    # 定义数据集 x3，包含四个数组，每个数组代表一组观测值
    # 对 x1 中的四组数据执行 Friedman 卡方检验，并断言检验结果与期望值几乎相等
    assert_array_almost_equal(stats.friedmanchisquare(x1[0], x1[1], x1[2], x1[3]),
                              (10.2283464566929, 0.0167215803284414))
    
    # 对 x2 中的四组数据执行 Friedman 卡方检验，并断言检验结果与期望值几乎相等
    assert_array_almost_equal(stats.friedmanchisquare(x2[0], x2[1], x2[2], x2[3]),
                              (18.9428571428571, 0.000280938375189499))
    
    # 对 x3 中的四组数据执行 Friedman 卡方检验，并断言检验结果与期望值几乎相等
    assert_array_almost_equal(stats.friedmanchisquare(x3[0], x3[1], x3[2], x3[3]),
                              (10.68, 0.0135882729582176))
    
    # 对不合法的输入 x3[0], x3[1] 执行 Friedman 卡方检验，断言会引发 ValueError 异常
    assert_raises(ValueError, stats.friedmanchisquare, x3[0], x3[1])

    # 测试 namedtuple 属性结果
    attributes = ('statistic', 'pvalue')
    # 对 x1 执行 Friedman 卡方检验，检查返回结果的属性是否符合预期
    res = stats.friedmanchisquare(*x1)
    check_named_results(res, attributes)

    # 使用 mstats 执行 Friedman 卡方检验，断言检验结果与期望值几乎相等
    assert_array_almost_equal(mstats.friedmanchisquare(x1[0], x1[1], x1[2], x1[3]),
                              (10.2283464566929, 0.0167215803284414))
    
    # mstats 版本的 Friedman 卡方检验，由于结果与期望值不匹配，以下断言会失败
    # assert_array_almost_equal(mstats.friedmanchisquare(x2[0], x2[1], x2[2], x2[3]),
    #                           (18.9428571428571, 0.000280938375189499))
    
    # 对 x3 使用 mstats 的 Friedman 卡方检验，断言检验结果与期望值几乎相等
    assert_array_almost_equal(mstats.friedmanchisquare(x3[0], x3[1], x3[2], x3[3]),
                              (10.68, 0.0135882729582176))
    
    # 对不合法的输入 x3[0], x3[1] 使用 mstats 的 Friedman 卡方检验，断言会引发 ValueError 异常
    assert_raises(ValueError, mstats.friedmanchisquare, x3[0], x3[1])
class TestKSTest:
    """Tests kstest and ks_1samp agree with K-S various sizes, alternatives, modes."""

    # 定义测试单个样本的方法
    def _testOne(self, x, alternative, expected_statistic, expected_prob,
                 mode='auto', decimal=14):
        # 使用 scipy.stats 中的 kstest 函数进行假设检验，检验样本 x 是否符合正态分布
        result = stats.kstest(x, 'norm', alternative=alternative, mode=mode)
        # 构造期望的检验统计量和 p 值的数组
        expected = np.array([expected_statistic, expected_prob])
        # 断言结果与期望的近似相等
        assert_array_almost_equal(np.array(result), expected, decimal=decimal)

    # 定义测试 kstest 和 ks_1samp 两个函数结果一致的方法
    def _test_kstest_and_ks1samp(self, x, alternative, mode='auto', decimal=14):
        # 使用 scipy.stats 中的 kstest 函数检验样本 x 是否符合正态分布
        result = stats.kstest(x, 'norm', alternative=alternative, mode=mode)
        # 使用 scipy.stats 中的 ks_1samp 函数检验样本 x 是否符合给定的累积分布函数
        result_1samp = stats.ks_1samp(x, stats.norm.cdf,
                                      alternative=alternative, mode=mode)
        # 断言 kstest 和 ks_1samp 的结果近似相等
        assert_array_almost_equal(np.array(result), result_1samp, decimal=decimal)

    # 测试返回的结果是否为命名元组，并检查其属性值
    def test_namedtuple_attributes(self):
        x = np.linspace(-1, 1, 9)
        # 检查命名元组属性的测试结果
        attributes = ('statistic', 'pvalue')
        res = stats.kstest(x, 'norm')
        # 调用函数检查结果是否符合预期的命名元组属性
        check_named_results(res, attributes)

    # 测试 kstest 和 ks_1samp 是否一致
    def test_agree_with_ks_1samp(self):
        x = np.linspace(-1, 1, 9)
        # 测试样本是否符合正态分布
        self._test_kstest_and_ks1samp(x, 'two-sided')

        x = np.linspace(-15, 15, 9)
        # 测试样本是否符合正态分布
        self._test_kstest_and_ks1samp(x, 'two-sided')

        x = [-1.23, 0.06, -0.60, 0.17, 0.66, -0.17, -0.08, 0.27, -0.98, -0.99]
        # 测试样本是否符合正态分布
        self._test_kstest_and_ks1samp(x, 'two-sided')
        # 测试样本是否符合大于给定分布的 ks_1samp 函数
        self._test_kstest_and_ks1samp(x, 'greater', mode='exact')
        # 测试样本是否符合小于给定分布的 ks_1samp 函数
        self._test_kstest_and_ks1samp(x, 'less', mode='exact')

    # 测试 kstest 函数对于包含正负无穷值的样本是否正常工作
    def test_pm_inf_gh20386(self):
        # 检查 gh-20386 的问题是否已解决：kstest 在样本中同时包含 -inf 和 inf 时不返回 NaN
        vals = [-np.inf, 0, 1, np.inf]
        # 对样本使用柯西分布的累积分布函数进行 kstest 检验
        res = stats.kstest(vals, stats.cauchy.cdf)
        # 使用 _no_deco=True 的 kstest 进行检验
        ref = stats.kstest(vals, stats.cauchy.cdf, _no_deco=True)
        # 断言检验结果全部为有限值
        assert np.all(np.isfinite(res))
        # 断言结果与参考值 ref 相等
        assert_equal(res, ref)
        # 断言统计量和 p 值均不是 NaN
        assert not np.isnan(res.statistic)
        assert not np.isnan(res.pvalue)

    # 缺失的部分：没有使用 *args 的测试
    def test_agree_with_r(self):
        # 比较结果与 R 中的一些数值
        x = np.linspace(-1, 1, 9)
        self._testOne(x, 'two-sided', 0.15865525393145705, 0.95164069201518386)

        x = np.linspace(-15, 15, 9)
        self._testOne(x, 'two-sided', 0.44435602715924361, 0.038850140086788665)

        x = [-1.23, 0.06, -0.60, 0.17, 0.66, -0.17, -0.08, 0.27, -0.98, -0.99]
        self._testOne(x, 'two-sided', 0.293580126801961, 0.293408463684361)
        self._testOne(x, 'greater', 0.293580126801961, 0.146988835042376, mode='exact')
        self._testOne(x, 'less', 0.109348552425692, 0.732768892470675, mode='exact')

    def test_known_examples(self):
        # 以下测试基于确定性复制的随机变量
        x = stats.norm.rvs(loc=0.2, size=100, random_state=987654321)
        self._testOne(x, 'two-sided', 0.12464329735846891, 0.089444888711820769,
                      mode='asymp')
        self._testOne(x, 'less', 0.12464329735846891, 0.040989164077641749)
        self._testOne(x, 'greater', 0.0072115233216310994, 0.98531158590396228)

    def test_ks1samp_allpaths(self):
        # 检查 NaN 输入和输出
        assert_(np.isnan(kolmogn(np.nan, 1, True)))
        with assert_raises(ValueError, match='n is not integral: 1.5'):
            kolmogn(1.5, 1, True)
        assert_(np.isnan(kolmogn(-1, 1, True)))

        dataset = np.asarray([
            # 检查 x 超出范围
            (101, 1, True, 1.0),
            (101, 1.1, True, 1.0),
            (101, 0, True, 0.0),
            (101, -0.1, True, 0.0),

            (32, 1.0 / 64, True, 0.0),  # Ruben-Gambino
            (32, 1.0 / 64, False, 1.0),  # Ruben-Gambino

            # Miller
            (32, 0.5, True, 0.9999999363163307),
            # Miller 2 * special.smirnov(32, 0.5)
            (32, 0.5, False, 6.368366937916623e-08),

            # 检查一些其他路径
            (32, 1.0 / 8, True, 0.34624229979775223),
            (32, 1.0 / 4, True, 0.9699508336558085),
            (1600, 0.49, False, 0.0),
            # 2 * special.smirnov(1600, 1/16.0)
            (1600, 1 / 16.0, False, 7.0837876229702195e-06),
            # _kolmogn_DMTW
            (1600, 14 / 1600, False, 0.99962357317602),
            # _kolmogn_PelzGood
            (1600, 1 / 32, False, 0.08603386296651416),
        ])
        FuncData(kolmogn, dataset, (0, 1, 2), 3).check(dtypes=[int, float, bool])

    @pytest.mark.parametrize("ksfunc", [stats.kstest, stats.ks_1samp])
    @pytest.mark.parametrize("alternative, x6val, ref_location, ref_sign",
                             [('greater', 6, 6, +1),
                              ('less', 7, 7, -1),
                              ('two-sided', 6, 6, +1),
                              ('two-sided', 7, 7, -1)])
    # 定义一个测试函数，用于检查统计量的位置和符号是否符合预期
    def test_location_sign(self, ksfunc, alternative,
                           x6val, ref_location, ref_sign):
        # 创建一个包含十个元素的一维数组，值为 0.5 到 9.5
        x = np.arange(10) + 0.5
        # 将数组中索引为 6 的元素设置为 x6val
        x[6] = x6val
        # 创建一个以均匀分布为基础的累积分布函数对象
        cdf = stats.uniform(scale=10).cdf
        # 调用给定的统计函数 ksfunc，计算统计量
        res = ksfunc(x, cdf, alternative=alternative)
        # 断言检查统计量的值接近 0.1，相对误差容忍度为 1e-15
        assert_allclose(res.statistic, 0.1, rtol=1e-15)
        # 断言检查计算得到的统计量位置是否与参考位置 ref_location 相符
        assert res.statistic_location == ref_location
        # 断言检查计算得到的统计量符号是否与参考符号 ref_sign 相符
        assert res.statistic_sign == ref_sign

    # 缺失的注释：没有使用 *args 进行的测试
class TestKSTwoSamples:
    """Tests 2-samples with K-S various sizes, alternatives, modes."""

    def _testOne(self, x1, x2, alternative, expected_statistic, expected_prob,
                 mode='auto'):
        # 调用 scipy.stats 中的 ks_2samp 函数进行两样本 Kolmogorov-Smirnov 检验
        result = stats.ks_2samp(x1, x2, alternative, mode=mode)
        # 期望的统计量和概率作为一个数组
        expected = np.array([expected_statistic, expected_prob])
        # 使用 numpy 的 assert_array_almost_equal 函数比较结果和期望值的接近程度
        assert_array_almost_equal(np.array(result), expected)

    def testSmall(self):
        # 测试两个小样本的不同情况
        self._testOne([0], [1], 'two-sided', 1.0/1, 1.0)
        self._testOne([0], [1], 'greater', 1.0/1, 0.5)
        self._testOne([0], [1], 'less', 0.0/1, 1.0)
        self._testOne([1], [0], 'two-sided', 1.0/1, 1.0)
        self._testOne([1], [0], 'greater', 0.0/1, 1.0)
        self._testOne([1], [0], 'less', 1.0/1, 0.5)

    def testTwoVsThree(self):
        # 测试一个样本有两个元素，另一个样本有三个元素的不同情况
        data1 = np.array([1.0, 2.0])
        data1p = data1 + 0.01
        data1m = data1 - 0.01
        data2 = np.array([1.0, 2.0, 3.0])
        self._testOne(data1p, data2, 'two-sided', 1.0 / 3, 1.0)
        self._testOne(data1p, data2, 'greater', 1.0 / 3, 0.7)
        self._testOne(data1p, data2, 'less', 1.0 / 3, 0.7)
        self._testOne(data1m, data2, 'two-sided', 2.0 / 3, 0.6)
        self._testOne(data1m, data2, 'greater', 2.0 / 3, 0.3)
        self._testOne(data1m, data2, 'less', 0, 1.0)

    def testTwoVsFour(self):
        # 测试一个样本有两个元素，另一个样本有四个元素的不同情况
        data1 = np.array([1.0, 2.0])
        data1p = data1 + 0.01
        data1m = data1 - 0.01
        data2 = np.array([1.0, 2.0, 3.0, 4.0])
        self._testOne(data1p, data2, 'two-sided', 2.0 / 4, 14.0/15)
        self._testOne(data1p, data2, 'greater', 2.0 / 4, 8.0/15)
        self._testOne(data1p, data2, 'less', 1.0 / 4, 12.0/15)

        self._testOne(data1m, data2, 'two-sided', 3.0 / 4, 6.0/15)
        self._testOne(data1m, data2, 'greater', 3.0 / 4, 3.0/15)
        self._testOne(data1m, data2, 'less', 0, 1.0)

    def test100_100(self):
        # 测试两个包含100个元素的样本的不同情况
        x100 = np.linspace(1, 100, 100)
        x100_2_p1 = x100 + 2 + 0.1
        x100_2_m1 = x100 + 2 - 0.1
        self._testOne(x100, x100_2_p1, 'two-sided', 3.0 / 100, 0.9999999999962055)
        self._testOne(x100, x100_2_p1, 'greater', 3.0 / 100, 0.9143290114276248)
        self._testOne(x100, x100_2_p1, 'less', 0, 1.0)
        self._testOne(x100, x100_2_m1, 'two-sided', 2.0 / 100, 1.0)
        self._testOne(x100, x100_2_m1, 'greater', 2.0 / 100, 0.960978450786184)
        self._testOne(x100, x100_2_m1, 'less', 0, 1.0)
    # 定义一个测试方法，用于测试一些数值计算
    def test100_110(self):
        # 生成一个包含100个元素的等间距数组
        x100 = np.linspace(1, 100, 100)
        # 生成一个包含110个元素的等间距数组
        x110 = np.linspace(1, 100, 110)
        # 对 x110 数组的每个元素加上20.1
        x110_20_p1 = x110 + 20 + 0.1
        # 对 x110 数组的每个元素减去20.1
        x110_20_m1 = x110 + 20 - 0.1
        # 调用 _testOne 方法，传入 x100, x110_20_p1 数组，执行两侧检验，期望值为 232.0 / 1100，显著性水平为 0.015739183865607353
        self._testOne(x100, x110_20_p1, 'two-sided', 232.0 / 1100, 0.015739183865607353)
        # 调用 _testOne 方法，传入 x100, x110_20_p1 数组，执行右侧检验，期望值为 232.0 / 1100，显著性水平为 0.007869594319053203
        self._testOne(x100, x110_20_p1, 'greater', 232.0 / 1100, 0.007869594319053203)
        # 调用 _testOne 方法，传入 x100, x110_20_p1 数组，执行左侧检验，期望值为 0，显著性水平为 1
        self._testOne(x100, x110_20_p1, 'less', 0, 1)
        # 调用 _testOne 方法，传入 x100, x110_20_m1 数组，执行两侧检验，期望值为 229.0 / 1100，显著性水平为 0.017803803861026313
        self._testOne(x100, x110_20_m1, 'two-sided', 229.0 / 1100, 0.017803803861026313)
        # 调用 _testOne 方法，传入 x100, x110_20_m1 数组，执行右侧检验，期望值为 229.0 / 1100，显著性水平为 0.008901905958245056
        self._testOne(x100, x110_20_m1, 'greater', 229.0 / 1100, 0.008901905958245056)
        # 调用 _testOne 方法，传入 x100, x110_20_m1 数组，执行左侧检验，期望值为 0.0，显著性水平为 1.0
        self._testOne(x100, x110_20_m1, 'less', 0.0, 1.0)

    # 定义一个测试方法，用于测试包含重复值的数组
    def testRepeatedValues(self):
        # 创建包含多个重复值的数组 x2233
        x2233 = np.array([2] * 3 + [3] * 4 + [5] * 5 + [6] * 4, dtype=int)
        # 创建 x3344 数组，每个元素比 x2233 对应位置的元素加1
        x3344 = x2233 + 1
        # 创建包含不同重复值的数组 x2356
        x2356 = np.array([2] * 3 + [3] * 4 + [5] * 10 + [6] * 4, dtype=int)
        # 创建包含不同重复值的数组 x3467
        x3467 = np.array([3] * 10 + [4] * 2 + [6] * 10 + [7] * 4, dtype=int)
        # 调用 _testOne 方法，传入 x2233, x3344 数组，执行两侧检验，期望值为 5.0/16，显著性水平为 0.4262934613454952
        self._testOne(x2233, x3344, 'two-sided', 5.0/16, 0.4262934613454952)
        # 调用 _testOne 方法，传入 x2233, x3344 数组，执行右侧检验，期望值为 5.0/16，显著性水平为 0.21465428276573786
        self._testOne(x2233, x3344, 'greater', 5.0/16, 0.21465428276573786)
        # 调用 _testOne 方法，传入 x2233, x3344 数组，执行左侧检验，期望值为 0.0/16，显著性水平为 1.0
        self._testOne(x2233, x3344, 'less', 0.0/16, 1.0)
        # 调用 _testOne 方法，传入 x2356, x3467 数组，执行两侧检验，期望值为 190.0/21/26，显著性水平为 0.0919245790168125
        self._testOne(x2356, x3467, 'two-sided', 190.0/21/26, 0.0919245790168125)
        # 调用 _testOne 方法，传入 x2356, x3467 数组，执行右侧检验，期望值为 190.0/21/26，显著性水平为 0.0459633806858544
        self._testOne(x2356, x3467, 'greater', 190.0/21/26, 0.0459633806858544)
        # 调用 _testOne 方法，传入 x2356, x3467 数组，执行左侧检验，期望值为 70.0/21/26，显著性水平为 0.6121593130022775
        self._testOne(x2356, x3467, 'less', 70.0/21/26, 0.6121593130022775)

    # 定义一个测试方法，用于测试等大小的数组
    @pytest.mark.slow
    def testEqualSizes(self):
        # 创建一个包含数值的数组 data2
        data2 = np.array([1.0, 2.0, 3.0])
        # 调用 _testOne 方法，传入 data2, data2+1 数组，执行两侧检验，期望值为 1.0/3，显著性水平为 1.0
        self._testOne(data2, data2+1, 'two-sided', 1.0/3, 1.0)
        # 调用 _testOne 方法，传入 data2, data2+1 数组，执行右侧检验，期望值为 1.0/3，显著性水平为 0.75
        self._testOne(data2, data2+1, 'greater', 1.0/3, 0.75)
        # 调用 _testOne 方法，传入 data2, data2+1 数组，执行左侧检验，期望值为 0.0/3，显著性水平为 1.0
        self._testOne(data2, data2+1, 'less', 0.0/3, 1.0)
        # 调用 _testOne 方法，传入 data2, data2+0.5 数组，执行两侧检验，期望值为 1.0/3，显著性水平为 1.0
        self._testOne(data2, data2+0.5, 'two-sided', 1.0/3, 1.0)
        # 调用 _testOne 方法，传入 data2, data2+0.5 数组，执行右侧检验，期望值为 1.0/3，显著性水平为 0.75
        self._testOne(data2, data2+0.5, 'greater', 1.0/3, 0.75)
        # 调用 _testOne 方法，传入 data2, data
    # 定义一个测试方法，测试两组数据的中位数及其他统计量是否符合预期
    def testMiddlingBoth(self):
        # 设置两组数据的大小为500和600
        n1, n2 = 500, 600
        # 计算步长
        delta = 1.0/n1/n2/2/2
        # 生成第一组数据x，从1到200，总共500个数据点，步长为delta
        x = np.linspace(1, 200, n1) - delta
        # 生成第二组数据y，从2到200，总共600个数据点
        y = np.linspace(2, 200, n2)
        # 进行两组数据的两样本Kolmogorov-Smirnov检验，检验双边分布
        self._testOne(x, y, 'two-sided', 2000.0 / n1 / n2, 1.0,
                      mode='auto')
        self._testOne(x, y, 'two-sided', 2000.0 / n1 / n2, 1.0,
                      mode='asymp')
        # 检验第一组数据大于第二组数据的分布
        self._testOne(x, y, 'greater', 2000.0 / n1 / n2, 0.9697596024683929,
                      mode='asymp')
        # 检验第一组数据小于第二组数据的分布
        self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021,
                      mode='asymp')
        # 忽略特定的运行时警告信息
        with suppress_warnings() as sup:
            message = "ks_2samp: Exact calculation unsuccessful."
            sup.filter(RuntimeWarning, message)
            # 对两组数据进行精确的Kolmogorov-Smirnov检验，检验大于分布
            self._testOne(x, y, 'greater', 2000.0 / n1 / n2, 0.9697596024683929,
                          mode='exact')
            # 对两组数据进行精确的Kolmogorov-Smirnov检验，检验小于分布
            self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021,
                          mode='exact')
        # 捕获并记录所有警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 对两组数据进行精确的Kolmogorov-Smirnov检验，检验小于分布
            self._testOne(x, y, 'less', 500.0 / n1 / n2, 0.9968735843165021,
                          mode='exact')
            # 检查是否出现特定的运行时警告
            _check_warnings(w, RuntimeWarning, 1)

    @pytest.mark.slow
    # 标记为慢速测试
    def testMediumBoth(self):
        # 设置两组数据的大小为1000和1100
        n1, n2 = 1000, 1100
        # 计算步长
        delta = 1.0/n1/n2/2/2
        # 生成第一组数据x，从1到200，总共1000个数据点，步长为delta
        x = np.linspace(1, 200, n1) - delta
        # 生成第二组数据y，从2到200，总共1100个数据点
        y = np.linspace(2, 200, n2)
        # 进行两组数据的两样本Kolmogorov-Smirnov检验，检验双边分布
        self._testOne(x, y, 'two-sided', 6600.0 / n1 / n2, 1.0,
                      mode='asymp')
        self._testOne(x, y, 'two-sided', 6600.0 / n1 / n2, 1.0,
                      mode='auto')
        # 检验第一组数据大于第二组数据的分布
        self._testOne(x, y, 'greater', 6600.0 / n1 / n2, 0.9573185808092622,
                      mode='asymp')
        # 检验第一组数据小于第二组数据的分布
        self._testOne(x, y, 'less', 1000.0 / n1 / n2, 0.9982410869433984,
                      mode='asymp')
        # 忽略特定的运行时警告信息
        with suppress_warnings() as sup:
            message = "ks_2samp: Exact calculation unsuccessful."
            sup.filter(RuntimeWarning, message)
            # 对两组数据进行精确的Kolmogorov-Smirnov检验，检验大于分布
            self._testOne(x, y, 'greater', 6600.0 / n1 / n2, 0.9573185808092622,
                          mode='exact')
            # 对两组数据进行精确的Kolmogorov-Smirnov检验，检验小于分布
            self._testOne(x, y, 'less', 1000.0 / n1 / n2, 0.9982410869433984,
                          mode='exact')
        # 捕获并记录所有警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 对两组数据进行精确的Kolmogorov-Smirnov检验，检验小于分布
            self._testOne(x, y, 'less', 1000.0 / n1 / n2, 0.9982410869433984,
                          mode='exact')
            # 检查是否出现特定的运行时警告
            _check_warnings(w, RuntimeWarning, 1)
    def testLarge(self):
        # 定义测试参数 n1 和 n2
        n1, n2 = 10000, 110
        # 计算 lcm（最小公倍数）
        lcm = n1*11.0
        # 计算 delta（增量）
        delta = 1.0/n1/n2/2/2
        # 生成包含 n1 个元素的数组 x，从1到200，减去 delta
        x = np.linspace(1, 200, n1) - delta
        # 生成包含 n2 个元素的数组 y，从2到100
        y = np.linspace(2, 100, n2)
        # 调用 _testOne 方法进行测试，用 'two-sided' 方式，比较结果与预期值
        self._testOne(x, y, 'two-sided', 55275.0 / lcm, 4.2188474935755949e-15)
        # 调用 _testOne 方法进行测试，用 'greater' 方式，比较结果与预期值
        self._testOne(x, y, 'greater', 561.0 / lcm, 0.99115454582047591)
        # 调用 _testOne 方法进行测试，用 'less' 方式，比较结果与预期值
        self._testOne(x, y, 'less', 55275.0 / lcm, 3.1317328311518713e-26)

    def test_gh11184(self):
        # 设置随机数种子
        np.random.seed(123456)
        # 生成服从正态分布的长度为 3000 的数组 x
        x = np.random.normal(size=3000)
        # 生成服从正态分布的长度为 3001 的数组 y，乘以 1.5
        y = np.random.normal(size=3001) * 1.5
        # 调用 _testOne 方法进行测试，用 'two-sided' 方式，比较结果与预期值
        self._testOne(x, y, 'two-sided', 0.11292880151060758, 2.7755575615628914e-15,
                      mode='asymp')
        # 调用 _testOne 方法进行测试，用 'two-sided' 方式，比较结果与预期值
        self._testOne(x, y, 'two-sided', 0.11292880151060758, 2.7755575615628914e-15,
                      mode='exact')

    @pytest.mark.xslow
    def test_gh11184_bigger(self):
        # 设置随机数种子
        np.random.seed(123456)
        # 生成服从正态分布的长度为 10000 的数组 x
        x = np.random.normal(size=10000)
        # 生成服从正态分布的长度为 10001 的数组 y，乘以 1.5
        y = np.random.normal(size=10001) * 1.5
        # 调用 _testOne 方法进行测试，用 'two-sided' 方式，比较结果与预期值
        self._testOne(x, y, 'two-sided', 0.10597913208679133, 3.3149311398483503e-49,
                      mode='asymp')
        # 调用 _testOne 方法进行测试，用 'two-sided' 方式，比较结果与预期值
        self._testOne(x, y, 'two-sided', 0.10597913208679133, 2.7755575615628914e-15,
                      mode='exact')
        # 调用 _testOne 方法进行测试，用 'greater' 方式，比较结果与预期值
        self._testOne(x, y, 'greater', 0.10597913208679133, 2.7947433906389253e-41,
                      mode='asymp')
        # 调用 _testOne 方法进行测试，用 'less' 方式，比较结果与预期值
        self._testOne(x, y, 'less', 0.09658002199780022, 2.7947433906389253e-41,
                      mode='asymp')

    @pytest.mark.xslow
    def test_gh12999(self):
        # 设置随机数种子
        np.random.seed(123456)
        # 遍历从 1000 到 12000，步长为 1000 的范围
        for x in range(1000, 12000, 1000):
            # 生成服从正态分布的长度为 x 的数组 vals1
            vals1 = np.random.normal(size=(x))
            # 生成服从正态分布的长度为 x+10 的数组 vals2，均值为 0.5
            vals2 = np.random.normal(size=(x + 10), loc=0.5)
            # 使用 'exact' 模式计算两组样本 vals1 和 vals2 的 Kolmogorov-Smirnov 检验的 p 值
            exact = stats.ks_2samp(vals1, vals2, mode='exact').pvalue
            # 使用 'asymp' 模式计算两组样本 vals1 和 vals2 的 Kolmogorov-Smirnov 检验的 p 值
            asymp = stats.ks_2samp(vals1, vals2, mode='asymp').pvalue
            # 断言 exact 和 asymp 两个 p 值应该相对应
            assert_array_less(exact, 3 * asymp)
            assert_array_less(asymp, 3 * exact)
    def testLargeBoth(self):
        # 定义两个整数变量，用于生成一组数据
        n1, n2 = 10000, 11000
        # 计算最小公倍数
        lcm = n1 * 11.0
        # 计算数值精度
        delta = 1.0 / n1 / n2 / 2 / 2
        # 生成长度为 n1 的数值序列 x
        x = np.linspace(1, 200, n1) - delta
        # 生成长度为 n2 的数值序列 y
        y = np.linspace(2, 200, n2)
        # 调用 _testOne 方法进行多个测试
        self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990660108966576,
                      mode='asymp')
        self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990456491488628,
                      mode='exact')
        self._testOne(x, y, 'two-sided', 563.0 / lcm, 0.9990660108966576,
                      mode='auto')
        self._testOne(x, y, 'greater', 563.0 / lcm, 0.7561851877420673)
        self._testOne(x, y, 'less', 10.0 / lcm, 0.9998239693191724)
        # 使用 suppress_warnings 上下文管理器抑制特定警告信息
        with suppress_warnings() as sup:
            message = "ks_2samp: Exact calculation unsuccessful."
            sup.filter(RuntimeWarning, message)
            # 在警告被抑制的情况下再次调用 _testOne 方法
            self._testOne(x, y, 'greater', 563.0 / lcm, 0.7561851877420673,
                          mode='exact')
            self._testOne(x, y, 'less', 10.0 / lcm, 0.9998239693191724,
                          mode='exact')

    def testNamedAttributes(self):
        # 测试 namedtuple 的属性结果
        attributes = ('statistic', 'pvalue')
        # 调用 stats.ks_2samp 方法进行两样本 Kolmogorov-Smirnov 检验
        res = stats.ks_2samp([1, 2], [3])
        # 检查返回结果是否符合命名属性的预期
        check_named_results(res, attributes)

    @pytest.mark.slow
    def test_some_code_paths(self):
        # 检查是否执行了部分代码路径
        from scipy.stats._stats_py import (
            _count_paths_outside_method,
            _compute_outer_prob_inside_method
        )
        # 调用 _compute_outer_prob_inside_method 函数执行计算
        _compute_outer_prob_inside_method(1, 1, 1, 1)
        # 调用 _count_paths_outside_method 函数执行计数
        _count_paths_outside_method(1000, 1, 1, 1001)
        
        # 在特定错误状态下进行断言测试
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, _count_paths_outside_method,
                          1100, 1099, 1, 1)
            assert_raises(FloatingPointError, _count_paths_outside_method,
                          2000, 1000, 1, 1)

    @pytest.mark.parametrize('case', (([], [1]), ([1], []), ([], [])))
    def test_argument_checking(self, case):
        # 检查空数组是否会触发警告
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            # 调用 stats.ks_2samp 方法进行两样本 Kolmogorov-Smirnov 检验
            res = stats.ks_2samp(*case)
            # 断言返回结果的 statistic 属性为 NaN
            assert_equal(res.statistic, np.nan)
            # 断言返回结果的 pvalue 属性为 NaN
            assert_equal(res.pvalue, np.nan)

    @pytest.mark.xslow
    def test_gh12218(self):
        """确保修复了 gh-12218。"""
        # gh-12218 引发了一个 TypeError，计算 sqrt(n1*n2*(n1+n2)) 时的问题。
        # n1 和 n2 都是大整数，它们的乘积超过了 2^64
        np.random.seed(12345678)
        n1 = 2097152  # 2*^21
        rvs1 = stats.uniform.rvs(size=n1, loc=0., scale=1)
        rvs2 = rvs1 + 1  # rvs2 的确切值并不重要。
        # 使用 mode='asymp' 参数调用 stats.ks_2samp 方法进行两样本 Kolmogorov-Smirnov 检验
        stats.ks_2samp(rvs1, rvs2, alternative='greater', mode='asymp')
        stats.ks_2samp(rvs1, rvs2, alternative='less', mode='asymp')
        stats.ks_2samp(rvs1, rvs2, alternative='two-sided', mode='asymp')
    # 定义测试方法，用于检查当 method='auto' 且精确的 p 值计算失败时是否引发 RuntimeWarning。详见 gh-14019。
    def test_warnings_gh_14019(self):
        # 使用种子 23493549 初始化随机数生成器
        rng = np.random.RandomState(seed=23493549)
        # 创建与问题中相同大小的随机样本数据
        data1 = rng.random(size=881) + 0.5
        data2 = rng.random(size=369)
        # 定义预期的警告消息
        message = "ks_2samp: Exact calculation unsuccessful"
        # 使用 pytest 的 warn 检查，确保在条件满足时引发 RuntimeWarning，且警告消息匹配预期消息
        with pytest.warns(RuntimeWarning, match=message):
            # 调用 stats.ks_2samp 进行 Kolmogorov-Smirnov 检验
            res = stats.ks_2samp(data1, data2, alternative='less')
            # 断言检验结果的 p 值接近于 0，允许的误差为 1e-14
            assert_allclose(res.pvalue, 0, atol=1e-14)

    # 使用 pytest 的参数化功能标记测试函数，测试 stats.kstest 和 stats.ks_2samp 方法
    @pytest.mark.parametrize("ksfunc", [stats.kstest, stats.ks_2samp])
    # 使用 pytest 的参数化功能，参数化 alternative, x6val, ref_location, ref_sign 四个参数
    @pytest.mark.parametrize("alternative, x6val, ref_location, ref_sign",
                             [('greater', 5.9, 5.9, +1),
                              ('less', 6.1, 6.0, -1),
                              ('two-sided', 5.9, 5.9, +1),
                              ('two-sided', 6.1, 6.0, -1)])
    # 定义测试方法，测试统计量的位置和符号是否与预期一致
    def test_location_sign(self, ksfunc, alternative,
                           x6val, ref_location, ref_sign):
        # 创建长度为 10 的浮点数数组 x 和 y
        x = np.arange(10, dtype=np.float64)
        y = x.copy()
        # 修改数组 x 的第 6 个元素为 x6val
        x[6] = x6val
        # 调用 stats.ks_2samp 进行 Kolmogorov-Smirnov 检验
        res = stats.ks_2samp(x, y, alternative=alternative)
        # 断言检验结果的统计量为 0.1
        assert res.statistic == 0.1
        # 断言检验结果的统计量位置与预期值 ref_location 一致
        assert res.statistic_location == ref_location
        # 断言检验结果的统计量符号与预期值 ref_sign 一致
        assert res.statistic_sign == ref_sign
def test_ttest_rel():
    # regression test
    # 设定预期的 t 值和 p 值
    tr, pr = 0.81248591389165692, 0.41846234511362157
    # 构造 tpr 变量，包含预期 t 值和 p 值的组合
    tpr = ([tr, -tr], [pr, pr])

    # 创建一维数组 rvs1 和 rvs2，分别存储从 1 到 100 和从 1.01 到 99.989 的等间隔数值
    rvs1 = np.linspace(1, 100, 100)
    rvs2 = np.linspace(1.01, 99.989, 100)
    # 创建二维数组 rvs1_2D 和 rvs2_2D，分别包含 rvs1 和 rvs2 的转置和原始形式
    rvs1_2D = np.array([np.linspace(1, 100, 100), np.linspace(1.01, 99.989, 100)])
    rvs2_2D = np.array([np.linspace(1.01, 99.989, 100), np.linspace(1, 100, 100)])

    # 对 rvs1 和 rvs2 进行相关 t 检验，沿 axis=0 方向
    t, p = stats.ttest_rel(rvs1, rvs2, axis=0)
    # 断言计算结果与预期结果（tr, pr）非常接近
    assert_array_almost_equal([t, p], (tr, pr))

    # 对 rvs1_2D 和 rvs2_2D 进行相关 t 检验，沿 axis=0 方向
    t, p = stats.ttest_rel(rvs1_2D.T, rvs2_2D.T, axis=0)
    # 断言计算结果与预期结果 tpr 非常接近
    assert_array_almost_equal([t, p], tpr)

    # 对 rvs1_2D 和 rvs2_2D 进行相关 t 检验，沿 axis=1 方向
    t, p = stats.ttest_rel(rvs1_2D, rvs2_2D, axis=1)
    # 断言计算结果与预期结果 tpr 非常接近
    assert_array_almost_equal([t, p], tpr)

    # 测试标量输入的 t 检验
    with suppress_warnings() as sup, \
            np.errstate(invalid="ignore", divide="ignore"):
        # 忽略特定的运行时警告，如自由度 <= 0 的警告
        sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
        # 对标量 4. 和 3. 进行相关 t 检验
        t, p = stats.ttest_rel(4., 3.)
    # 断言 t 和 p 的结果为 NaN
    assert_(np.isnan(t))
    assert_(np.isnan(p))

    # 测试返回命名元组属性的 t 检验结果
    attributes = ('statistic', 'pvalue')
    res = stats.ttest_rel(rvs1, rvs2, axis=0)
    # 检查 t 检验结果的命名元组属性
    check_named_results(res, attributes)

    # 对三维数组 rvs1_3D 和 rvs2_3D 进行相关 t 检验，沿 axis=1 方向
    rvs1_3D = np.dstack([rvs1_2D, rvs1_2D, rvs1_2D])
    rvs2_3D = np.dstack([rvs2_2D, rvs2_2D, rvs2_2D])
    t, p = stats.ttest_rel(rvs1_3D, rvs2_3D, axis=1)
    # 断言 t 的绝对值与预期的 tr 非常接近
    assert_array_almost_equal(np.abs(t), tr)
    # 断言 p 的绝对值与预期的 pr 非常接近
    assert_array_almost_equal(np.abs(p), pr)
    # 断言 t 的形状为 (2, 3)
    assert_equal(t.shape, (2, 3))

    # 对三维数组 rvs1_3D 和 rvs2_3D 进行相关 t 检验，沿 axis=2 方向
    t, p = stats.ttest_rel(np.moveaxis(rvs1_3D, 2, 0),
                           np.moveaxis(rvs2_3D, 2, 0),
                           axis=2)
    # 断言 t 的绝对值与预期的 tr 非常接近
    assert_array_almost_equal(np.abs(t), tr)
    # 断言 p 的绝对值与预期的 pr 非常接近
    assert_array_almost_equal(np.abs(p), pr)
    # 断言 t 的形状为 (3, 2)
    assert_equal(t.shape, (3, 2))

    # 测试 alternative 参数为非法值时引发 ValueError
    assert_raises(ValueError, stats.ttest_rel, rvs1, rvs2, alternative="error")

    # 对 rvs1 和 rvs2 进行相关 t 检验，使用 alternative="less" 参数
    t, p = stats.ttest_rel(rvs1, rvs2, axis=0, alternative="less")
    # 断言 p 的结果与预期值 1 - pr/2 非常接近
    assert_allclose(p, 1 - pr/2)
    # 断言 t 的结果与预期值 tr 非常接近
    assert_allclose(t, tr)

    # 对 rvs1 和 rvs2 进行相关 t 检验，使用 alternative="greater" 参数
    t, p = stats.ttest_rel(rvs1, rvs2, axis=0, alternative="greater")
    # 断言 p 的结果与预期值 pr/2 非常接近
    assert_allclose(p, pr/2)
    # 断言 t 的结果与预期值 tr 非常接近
    assert_allclose(t, tr)

    # 检查 NaN 策略
    rng = np.random.RandomState(12345678)
    x = stats.norm.rvs(loc=5, scale=10, size=501, random_state=rng)
    x[500] = np.nan
    y = (stats.norm.rvs(loc=5, scale=10, size=501, random_state=rng) +
         stats.norm.rvs(scale=0.2, size=501, random_state=rng))
    y[500] = np.nan

    with np.errstate(invalid="ignore"):
        # 断言忽略包含 NaN 的输入时 t 检验的结果为 NaN
        assert_array_equal(stats.ttest_rel(x, x), (np.nan, np.nan))

    # 断言使用 nan_policy='omit' 参数时的 t 检验结果与预期结果非常接近
    assert_array_almost_equal(stats.ttest_rel(x, y, nan_policy='omit'),
                              (0.25299925303978066, 0.8003729814201519))
    # 断言使用不支持的 nan_policy 参数值时引发 ValueError
    assert_raises(ValueError, stats.ttest_rel, x, y, nan_policy='raise')
    assert_raises(ValueError, stats.ttest_rel, x, y, nan_policy='foobar')

    # 检验除以零的问题
    with pytest.warns(RuntimeWarning, match="Precision loss occurred"):
        # 对包含 [0, 0, 0] 和 [1, 1, 1] 的数组进行 t 检验
        t, p = stats.ttest_rel([0, 0, 0], [1, 1, 1])
    # 断言 t 的绝对值为无穷大，p 的值为 0
    assert_equal((np.abs(t), p), (np.inf, 0))
    # 在忽略无效值的错误状态下执行以下代码块
    with np.errstate(invalid="ignore"):
        # 使用 t 检验检查两个相等数组的关系，期望结果为 (nan, nan)
        assert_equal(stats.ttest_rel([0, 0, 0], [0, 0, 0]), (np.nan, np.nan))

        # 检查输入数组中的 nan 是否会导致输出结果为 nan
        anan = np.array([[1, np.nan], [-1, 1]])
        assert_equal(stats.ttest_rel(anan, np.zeros((2, 2))),
                     ([0, np.nan], [1, np.nan]))

    # 测试不正确的输入形状是否会引发 ValueError 错误
    x = np.arange(24)
    assert_raises(ValueError, stats.ttest_rel, x.reshape((8, 3)),
                  x.reshape((2, 3, 4)))

    # 定义一个函数 convert，用于将双侧 p 值转换为单侧 p 值，根据 T 结果数据进行判断
    def convert(t, p, alt):
        if (t < 0 and alt == "less") or (t > 0 and alt == "greater"):
            return p / 2
        return 1 - (p / 2)
    # 创建一个向量化的 convert 函数
    converter = np.vectorize(convert)

    # 将 rvs1_2D 数组中的列 20-29 设置为 nan
    rvs1_2D[:, 20:30] = np.nan
    # 将 rvs2_2D 数组中的列 15-24 设置为 nan
    rvs2_2D[:, 15:25] = np.nan

    # 使用 t 检验函数计算 rvs1_2D 和 rvs2_2D 之间的 t 值和 p 值，忽略 nan 值，同时产生 SmallSampleWarning 警告
    with pytest.warns(SmallSampleWarning, match=too_small_nd_omit):
        tr, pr = stats.ttest_rel(rvs1_2D, rvs2_2D, 0, nan_policy='omit')

    # 使用 t 检验函数计算 rvs1_2D 和 rvs2_2D 之间的 t 值和 p 值，忽略 nan 值，同时产生 SmallSampleWarning 警告，
    # 并将双侧检验的结果转换为单侧检验的结果（alternative='less'）
    with pytest.warns(SmallSampleWarning, match=too_small_nd_omit):
        t, p = stats.ttest_rel(rvs1_2D, rvs2_2D, 0,
                               nan_policy='omit', alternative='less')
    # 断言 t 值在数值上接近预期的 tr 值，相对误差为 1e-14
    assert_allclose(t, tr, rtol=1e-14)
    # 在忽略无效值的错误状态下，断言 p 值在数值上接近预期的单侧 p 值，相对误差为 1e-14
    with np.errstate(invalid='ignore'):
        assert_allclose(p, converter(tr, pr, 'less'), rtol=1e-14)

    # 使用 t 检验函数计算 rvs1_2D 和 rvs2_2D 之间的 t 值和 p 值，忽略 nan 值，同时产生 SmallSampleWarning 警告，
    # 并将双侧检验的结果转换为单侧检验的结果（alternative='greater'）
    with pytest.warns(SmallSampleWarning, match=too_small_nd_omit):
        t, p = stats.ttest_rel(rvs1_2D, rvs2_2D, 0,
                               nan_policy='omit', alternative='greater')
    # 断言 t 值在数值上接近预期的 tr 值，相对误差为 1e-14
    assert_allclose(t, tr, rtol=1e-14)
    # 在忽略无效值的错误状态下，断言 p 值在数值上接近预期的单侧 p 值，相对误差为 1e-14
    with np.errstate(invalid='ignore'):
        assert_allclose(p, converter(tr, pr, 'greater'), rtol=1e-14)
# 定义一个用于测试相关 T 检验函数的单元测试函数，针对第二个参数中的 NaN 进行回归测试
def test_ttest_rel_nan_2nd_arg():
    # 创建包含 NaN 的列表 x 和普通列表 y 作为输入
    x = [np.nan, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 1.0, 2.0]

    # 对 x 和 y 进行相关 T 检验，忽略 NaN 值
    r1 = stats.ttest_rel(x, y, nan_policy='omit')
    r2 = stats.ttest_rel(y, x, nan_policy='omit')

    # 断言 r2 的统计量与 r1 的统计量相反
    assert_allclose(r2.statistic, -r1.statistic, atol=1e-15)
    # 断言 r2 的 p 值与 r1 的 p 值相等
    assert_allclose(r2.pvalue, r1.pvalue, atol=1e-15)

    # 注意事项：当 NaN 被丢弃时，参数是成对的
    r3 = stats.ttest_rel(y[1:], x[1:])
    assert_allclose(r2, r3, atol=1e-15)

    # 与 R 语言的一致性检查，R 代码提供了参考值
    assert_allclose(r2, (-2, 0.1835), atol=1e-4)


# 定义一个测试函数，验证当传入的一维空数组时，相关 T 检验函数返回包含 NaN 的 TtestResult
def test_ttest_rel_empty_1d_returns_nan():
    # 两个空输入应该返回包含 nan 值的 TtestResult
    with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
        result = stats.ttest_rel([], [])
    assert isinstance(result, stats._stats_py.TtestResult)
    assert_equal(result, (np.nan, np.nan))


# 使用参数化测试装饰器定义一个测试函数，验证当轴的长度为零时，相关 T 检验函数返回正确的形状与 nan 值的数组
@pytest.mark.parametrize('b, expected_shape',
                         [(np.empty((1, 5, 0)), (3, 5)),
                          (np.empty((1, 0, 0)), (3, 0))])
def test_ttest_rel_axis_size_zero(b, expected_shape):
    # 在这个测试中，轴维度的长度为零，结果应该是包含 nan 的数组，形状由非轴维度的广播决定
    a = np.empty((3, 1, 0))
    with np.testing.suppress_warnings() as sup:
        # 第一个情况应该警告，第二个情况不应该？
        sup.filter(SmallSampleWarning, too_small_nd_not_omit)
        result = stats.ttest_rel(a, b, axis=-1)
    assert isinstance(result, stats._stats_py.TtestResult)
    expected_value = np.full(expected_shape, fill_value=np.nan)
    assert_equal(result.statistic, expected_value)
    assert_equal(result.pvalue, expected_value)


# 定义一个测试函数，验证非轴维度长度为零时，相关 T 检验函数返回正确的广播形状 (5, 0)
def test_ttest_rel_nonaxis_size_zero():
    # 在这个测试中，轴维度的长度为非零，但是其中一个非轴维度的长度为零。检查是否依然能够得到正确的广播形状 (5, 0)
    a = np.empty((1, 8, 0))
    b = np.empty((5, 8, 1))
    result = stats.ttest_rel(a, b, axis=1)
    assert isinstance(result, stats._stats_py.TtestResult)
    assert_equal(result.statistic.shape, (5, 0))
    assert_equal(result.pvalue.shape, (5, 0))


# 使用参数化测试装饰器定义一个测试函数，验证一维相关 T 检验函数的置信区间方法与参考值的一致性
@pytest.mark.parametrize("alternative", ['two-sided', 'less', 'greater'])
def test_ttest_rel_ci_1d(alternative):
    # 针对一维数组 x 和 y 进行相关 T 检验，测试置信区间方法与参考值的一致性
    rng = np.random.default_rng(3749065329432213059)
    n = 10
    x = rng.normal(size=n, loc=1.5, scale=2)
    y = rng.normal(size=n, loc=2, scale=2)
    # 使用 R 的 t.test 生成参考值
    # options(digits=16)
    # x = c(1.22825792,  1.63950485,  4.39025641,  0.68609437,  2.03813481,
    #       -1.20040109,  1.81997937,  1.86854636,  2.94694282,  3.94291373)
    # 定义参考值字典，包含不同假设检验类型的置信区间上下限
    ref = {'two-sided': [-1.912194489914035, 0.400169725914035],
           'greater': [-1.563944820311475, np.inf],
           'less': [-np.inf, 0.05192005631147523]}
    
    # 进行配对 t 检验，返回检验结果对象
    res = stats.ttest_rel(x, y, alternative=alternative)
    
    # 根据置信水平计算检验结果的置信区间
    ci = res.confidence_interval(confidence_level=0.85)
    
    # 断言检验结果的置信区间与参考值中对应类型的置信区间相近
    assert_allclose(ci, ref[alternative])
    
    # 断言检验结果的自由度与样本量 n 减一相等
    assert_equal(res.df, n-1)
@pytest.mark.parametrize("test_fun, args",
                         [(stats.ttest_1samp, (np.arange(10), 0)),
                          (stats.ttest_rel, (np.arange(10), np.arange(10)))])
# 使用 pytest 的 parametrize 标记，为 test_ttest_ci_iv 函数定义多个参数化测试用例
def test_ttest_ci_iv(test_fun, args):
    # 测试 `confidence_interval` 方法的输入验证
    res = test_fun(*args)
    message = '`confidence_level` must be a number between 0 and 1.'
    # 使用 pytest.raises 检查 ValueError 异常是否被正确抛出，并验证异常信息
    with pytest.raises(ValueError, match=message):
        res.confidence_interval(confidence_level=10)


def _desc_stats(x1, x2, axis=0, *, xp=None):
    xp = array_namespace(x1, x2) if xp is None else xp

    def _stats(x, axis=0):
        x = xp.asarray(x)
        mu = xp.mean(x, axis=axis)
        std = xp.std(x, axis=axis, correction=1)
        nobs = x.shape[axis]
        return mu, std, nobs

    return _stats(x1, axis) + _stats(x2, axis)


@array_api_compatible
@pytest.mark.skip_xp_backends(cpu_only=True,
                              reasons=['Uses NumPy for pvalue, CI'])
# 使用 pytest 的标记：标记此测试用例需要跳过特定的计算后端，并给出原因
@pytest.mark.usefixtures("skip_xp_backends")
# 使用 pytest 的 usefixtures 标记，指定需要在测试运行之前执行的 fixture
def test_ttest_ind(xp):
    # 回归测试
    tr = xp.asarray(1.0912746897927283)
    pr = xp.asarray(0.27647818616351882)
    tr_2D = xp.asarray([tr, -tr])
    pr_2D = xp.asarray([pr, pr])

    rvs1 = xp.linspace(5, 105, 100)
    rvs2 = xp.linspace(1, 100, 100)
    rvs1_2D = xp.stack([rvs1, rvs2])
    rvs2_2D = xp.stack([rvs2, rvs1])

    res = stats.ttest_ind(rvs1, rvs2, axis=0)
    t, p = res  # 检查结果对象是否可以正确解包
    xp_assert_close(t, tr)
    xp_assert_close(p, pr)

    res = stats.ttest_ind_from_stats(*_desc_stats(rvs1, rvs2))
    t, p = res  # 检查结果对象是否可以正确解包
    xp_assert_close(t, tr)
    xp_assert_close(p, pr)

    res = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0)
    xp_assert_close(res.statistic, tr_2D)
    xp_assert_close(res.pvalue, pr_2D)

    res = stats.ttest_ind_from_stats(*_desc_stats(rvs1_2D.T, rvs2_2D.T))
    xp_assert_close(res.statistic, tr_2D)
    xp_assert_close(res.pvalue, pr_2D)

    res = stats.ttest_ind(rvs1_2D, rvs2_2D, axis=1)
    xp_assert_close(res.statistic, tr_2D)
    xp_assert_close(res.pvalue, pr_2D)

    res = stats.ttest_ind_from_stats(*_desc_stats(rvs1_2D, rvs2_2D, axis=1))
    xp_assert_close(res.statistic, tr_2D)
    xp_assert_close(res.pvalue, pr_2D)

    # 测试在三维数据上已删除，因为在 test_axis_nan_policy 中的通用测试更为强大

    # 测试 alternative 参数
    message = "`alternative` must be 'less', 'greater', or 'two-sided'."
    with pytest.raises(ValueError, match=message):
        stats.ttest_ind(rvs1, rvs2, alternative="error")

    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    with pytest.raises(ValueError, match=message):
        stats.ttest_ind_from_stats(*args, alternative="error")

    t, p = stats.ttest_ind(rvs1, rvs2, alternative="less")
    xp_assert_close(p, 1 - (pr / 2))
    xp_assert_close(t, tr)

    t, p = stats.ttest_ind(rvs1, rvs2, alternative="greater")
    xp_assert_close(p, pr / 2)
    xp_assert_close(t, tr)
    # 使用自定义的断言函数 xp_assert_close 检查 t 和 tr 的接近程度

    # Check that ttest_ind_from_stats agrees with ttest_ind
    # 检查 ttest_ind_from_stats 函数与 ttest_ind 函数的一致性
    res1 = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0, alternative="less")
    # 使用 ttest_ind 计算两组数据 rvs1_2D.T 和 rvs2_2D.T 的 t 检验结果，设置单尾检验
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    # 使用 _desc_stats 函数获取描述性统计信息的参数
    res2 = stats.ttest_ind_from_stats(*args, alternative="less")
    # 使用 ttest_ind_from_stats 函数基于描述性统计信息的参数计算 t 检验结果，设置单尾检验
    xp_assert_close(res1.statistic, res2.statistic)
    # 使用自定义的断言函数 xp_assert_close 检查 res1 和 res2 的统计量的接近程度
    xp_assert_close(res1.pvalue, res2.pvalue)
    # 使用自定义的断言函数 xp_assert_close 检查 res1 和 res2 的 p 值的接近程度

    res1 = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0, alternative="less")
    # 再次使用 ttest_ind 计算两组数据 rvs1_2D.T 和 rvs2_2D.T 的 t 检验结果，设置单尾检验
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    # 使用 _desc_stats 函数获取描述性统计信息的参数
    res2 = stats.ttest_ind_from_stats(*args, alternative="less")
    # 再次使用 ttest_ind_from_stats 函数基于描述性统计信息的参数计算 t 检验结果，设置单尾检验
    xp_assert_close(res1.statistic, res2.statistic)
    # 使用自定义的断言函数 xp_assert_close 检查 res1 和 res2 的统计量的接近程度
    xp_assert_close(res1.pvalue, res2.pvalue)
    # 使用自定义的断言函数 xp_assert_close 检查 res1 和 res2 的 p 值的接近程度

    # test NaNs
    # 测试 NaN 值情况
    NaN = xp.asarray(xp.nan)
    # 创建一个包含 NaN 的数组
    rvs1 = xp.where(xp.arange(rvs1.shape[0]) == 0, NaN, rvs1)
    # 将 rvs1 数组中第一个元素替换为 NaN，其余元素保持不变

    res = stats.ttest_ind(rvs1, rvs2, axis=0)
    # 使用 ttest_ind 计算修改后的 rvs1 和 rvs2 的 t 检验结果
    xp_assert_equal(res.statistic, NaN)
    # 使用自定义的断言函数 xp_assert_equal 检查结果统计量是否等于 NaN
    xp_assert_equal(res.pvalue, NaN)
    # 使用自定义的断言函数 xp_assert_equal 检查结果 p 值是否等于 NaN

    res = stats.ttest_ind_from_stats(*_desc_stats(rvs1, rvs2))
    # 使用 ttest_ind_from_stats 函数基于描述性统计信息的参数计算 rvs1 和 rvs2 的 t 检验结果
    xp_assert_equal(res.statistic, NaN)
    # 使用自定义的断言函数 xp_assert_equal 检查结果统计量是否等于 NaN
    xp_assert_equal(res.pvalue, NaN)
    # 使用自定义的断言函数 xp_assert_equal 检查结果 p 值是否等于 NaN
def test_ttest_ind_nan_policy():
    # 生成从5到105的等间隔数列，包含100个元素
    rvs1 = np.linspace(5, 105, 100)
    # 生成从1到100的等间隔数列，包含100个元素
    rvs2 = np.linspace(1, 100, 100)
    # 将rvs1和rvs2堆叠成2维数组
    rvs1_2D = np.array([rvs1, rvs2])
    # 将rvs2和rvs1堆叠成2维数组
    rvs2_2D = np.array([rvs2, rvs1])
    # 将rvs1_2D堆叠三次形成3维数组
    rvs1_3D = np.dstack([rvs1_2D, rvs1_2D, rvs1_2D])
    # 将rvs2_2D堆叠三次形成3维数组
    rvs2_3D = np.dstack([rvs2_2D, rvs2_2D, rvs2_2D])

    # 检查 NaN 处理策略
    rng = np.random.RandomState(12345678)
    x = stats.norm.rvs(loc=5, scale=10, size=501, random_state=rng)
    x[500] = np.nan
    y = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)

    with np.errstate(invalid="ignore"):
        # 使用 np.errstate 忽略无效操作警告，验证 ttest_ind 对包含 NaN 的数组返回 NaN
        assert_array_equal(stats.ttest_ind(x, y), (np.nan, np.nan))

    # 测试忽略 NaN 策略下 ttest_ind 的结果
    assert_array_almost_equal(stats.ttest_ind(x, y, nan_policy='omit'),
                              (0.24779670949091914, 0.80434267337517906))
    # 测试 ValueError 异常抛出情况
    assert_raises(ValueError, stats.ttest_ind, x, y, nan_policy='raise')
    assert_raises(ValueError, stats.ttest_ind, x, y, nan_policy='foobar')

    # 测试除零问题
    with pytest.warns(RuntimeWarning, match="Precision loss occurred"):
        t, p = stats.ttest_ind([0, 0, 0], [1, 1, 1])
    assert_equal((np.abs(t), p), (np.inf, 0))

    with np.errstate(invalid="ignore"):
        # 验证 ttest_ind 对全为零的输入数组返回 NaN
        assert_equal(stats.ttest_ind([0, 0, 0], [0, 0, 0]), (np.nan, np.nan))

        # 验证输入数组中包含 NaN 时，ttest_ind 返回 NaN
        anan = np.array([[1, np.nan], [-1, 1]])
        assert_equal(stats.ttest_ind(anan, np.zeros((2, 2))),
                     ([0, np.nan], [1, np.nan]))

    # 将rvs1_3D和rvs2_3D的部分元素设置为 NaN
    rvs1_3D[:, :, 10:15] = np.nan
    rvs2_3D[:, :, 6:12] = np.nan

    # 将双边 p 值转换为单边 p 值，使用 T 结果数据进行转换
    def convert(t, p, alt):
        if (t < 0 and alt == "less") or (t > 0 and alt == "greater"):
            return p / 2
        return 1 - (p / 2)
    # 创建矢量化版本的 convert 函数
    converter = np.vectorize(convert)

    # 测试忽略 NaN 策略下的 ttest_ind 结果
    tr, pr = stats.ttest_ind(rvs1_3D, rvs2_3D, 0, nan_policy='omit')

    # 验证 ttest_ind 结果与转换后的单边检验结果是否近似
    t, p = stats.ttest_ind(rvs1_3D, rvs2_3D, 0, nan_policy='omit',
                           alternative='less')
    assert_allclose(t, tr, rtol=1e-14)
    assert_allclose(p, converter(tr, pr, 'less'), rtol=1e-14)

    t, p = stats.ttest_ind(rvs1_3D, rvs2_3D, 0, nan_policy='omit',
                           alternative='greater')
    assert_allclose(t, tr, rtol=1e-14)
    assert_allclose(p, converter(tr, pr, 'greater'), rtol=1e-14)
    ```python`
    # 生成服从正态分布的随机样本，设置均值为5，标准差为10，生成大小为500的一维数组，再转换为100行5列的二维数组
    rvs1 = stats.norm.rvs(loc=5, scale=10, size=500).reshape(100, 5).T
    
    # 生成服从正态分布的随机样本，设置均值为8，标准差为20，生成大小为100的一维数组
    rvs2 = stats.norm.rvs(loc=8, scale=20, size=100)
    
    # 设定期望的 p 值
    p_d = [1/1001, (676+1)/1001]
    
    # 设定期望的 p 值，针对生成器种子
    p_d_gen = [1/1001, (672 + 1)/1001]
    
    # 设定更大的期望的 p 值列表
    p_d_big = [(993+1)/1001, (685+1)/1001, (840+1)/1001,
               (955+1)/1001, (255+1)/1001]
    
    # 设定参数组合列表
    params = [
        (a, b, {"axis": 1}, p_d),                     # 基本测试
        (a.T, b.T, {'axis': 0}, p_d),                 # 沿轴 0
        (a[0, :], b[0, :], {'axis': None}, p_d[0]),   # 一维数据
        (a[0, :].tolist(), b[0, :].tolist(), {'axis': None}, p_d[0]),  # 转换为列表
        # 不同的种子
        (a, b, {'random_state': 0, "axis": 1}, p_d),
        (a, b, {'random_state': np.random.RandomState(0), "axis": 1}, p_d),
        (a2, b2, {'equal_var': True}, 1/1001),  # 方差相等
        (rvs1, rvs2, {'axis': -1, 'random_state': 0}, p_d_big),  # 更大的测试
        (a3, b3, {}, 1/3),  # 精确测试
        (a, b, {'random_state': np.random.default_rng(0), "axis": 1}, p_d_gen),
    ]
    
    # 使用 pytest 的 parametrize 装饰器进行参数化测试
    @pytest.mark.parametrize("a,b,update,p_d", params)
    def test_ttest_ind_permutations(self, a, b, update, p_d):
        # 设定选项字典
        options_a = {'axis': None, 'equal_var': False}
        options_p = {'axis': None, 'equal_var': False,
                     'permutations': 1000, 'random_state': 0}
        # 更新选项字典
        options_a.update(update)
        options_p.update(update)
    
        # 执行独立双样本 t 检验，返回统计量和 p 值
        stat_a, _ = stats.ttest_ind(a, b, **options_a)
        stat_p, pvalue = stats.ttest_ind(a, b, **options_p)
        
        # 断言近似相等
        assert_array_almost_equal(stat_a, stat_p, 5)
        assert_array_almost_equal(pvalue, p_d)
    def test_ttest_ind_exact_alternative(self):
        # 设置随机数种子为0，确保结果可重现
        np.random.seed(0)
        # 定义数组维度参数N
        N = 3
        # 生成两个形状为(2, N, 2)的随机数组a和b
        a = np.random.rand(2, N, 2)
        b = np.random.rand(2, N, 2)

        # 定义参数字典options_p，包括轴向和排列次数的设定
        options_p = {'axis': 1, 'permutations': 1000}

        # 更新参数字典，设定 alternative 为 "greater"
        options_p.update(alternative="greater")
        # 执行 t 检验，返回结果给 res_g_ab
        res_g_ab = stats.ttest_ind(a, b, **options_p)
        # 执行 t 检验，返回结果给 res_g_ba，参数相同
        res_g_ba = stats.ttest_ind(b, a, **options_p)

        # 更新参数字典，设定 alternative 为 "less"
        options_p.update(alternative="less")
        # 执行 t 检验，返回结果给 res_l_ab
        res_l_ab = stats.ttest_ind(a, b, **options_p)
        # 执行 t 检验，返回结果给 res_l_ba，参数相同
        res_l_ba = stats.ttest_ind(b, a, **options_p)

        # 更新参数字典，设定 alternative 为 "two-sided"
        options_p.update(alternative="two-sided")
        # 执行 t 检验，返回结果给 res_2_ab
        res_2_ab = stats.ttest_ind(a, b, **options_p)
        # 执行 t 检验，返回结果给 res_2_ba，参数相同
        res_2_ba = stats.ttest_ind(b, a, **options_p)

        # 断言：alternative 不影响统计量的结果
        assert_equal(res_g_ab.statistic, res_l_ab.statistic)
        assert_equal(res_g_ab.statistic, res_2_ab.statistic)

        # 断言：反转输入的顺序会使统计量取反
        assert_equal(res_g_ab.statistic, -res_g_ba.statistic)
        assert_equal(res_l_ab.statistic, -res_l_ba.statistic)
        assert_equal(res_2_ab.statistic, -res_2_ba.statistic)

        # 断言：反转输入顺序不会影响两样本 t 检验的双尾 p 值
        assert_equal(res_2_ab.pvalue, res_2_ba.pvalue)

        # 在精确检验中，分布完全对称，因此以下等式完全成立。
        assert_equal(res_g_ab.pvalue, res_l_ba.pvalue)
        assert_equal(res_l_ab.pvalue, res_g_ba.pvalue)

        # 创建布尔掩码，用于过滤 p 值小于等于0.5的情况
        mask = res_g_ab.pvalue <= 0.5
        # 断言：部分 p 值之和等于两样本 t 检验的双尾 p 值
        assert_equal(res_g_ab.pvalue[mask] + res_l_ba.pvalue[mask],
                     res_2_ab.pvalue[mask])
        assert_equal(res_l_ab.pvalue[~mask] + res_g_ba.pvalue[~mask],
                     res_2_ab.pvalue[~mask])

    def test_ttest_ind_exact_selection(self):
        # 测试激活精确检验的各种方式
        np.random.seed(0)
        # 定义数组维度参数N
        N = 3
        # 生成长度为N的随机数组a和b
        a = np.random.rand(N)
        b = np.random.rand(N)
        # 执行 t 检验，返回结果给 res0
        res0 = stats.ttest_ind(a, b)
        # 执行 t 检验，设定 permutations=1000，返回结果给 res1
        res1 = stats.ttest_ind(a, b, permutations=1000)
        # 执行 t 检验，设定 permutations=0，返回结果给 res2
        res2 = stats.ttest_ind(a, b, permutations=0)
        # 执行 t 检验，设定 permutations=np.inf，返回结果给 res3
        res3 = stats.ttest_ind(a, b, permutations=np.inf)
        # 断言：res1 的 p 值与 res0 的 p 值不相等
        assert res1.pvalue != res0.pvalue
        # 断言：res2 的 p 值与 res0 的 p 值相等
        assert res2.pvalue == res0.pvalue
        # 断言：res3 的 p 值与 res1 的 p 值相等
        assert res3.pvalue == res1.pvalue

    def test_ttest_ind_exact_distribution(self):
        # 精确检验的统计量应具有二项分布(binom(na + nb, na))的元素，且全部唯一。
        # 这在 gh-4824 中不总是成立；gh-13661 修复了这个问题。
        np.random.seed(0)
        # 生成长度为3的随机数组a和长度为4的随机数组b
        a = np.random.rand(3)
        b = np.random.rand(4)

        # 将数组a和b连接成一个新的数据数组
        data = np.concatenate((a, b))
        na, nb = len(a), len(b)

        # 设定排列次数为100000
        permutations = 100000
        # 计算置换分布 t 的统计量，返回结果给 t_stat
        t_stat, _, _ = _permutation_distribution_t(data, permutations, na,
                                                   True)

        # 计算 t_stat 中唯一元素的数量
        n_unique = len(set(t_stat))
        # 断言：唯一元素的数量应等于二项式分布的元素个数
        assert n_unique == binom(na + nb, na)
        # 断言：t_stat 的长度应等于唯一元素的数量
        assert len(t_stat) == n_unique
    # 定义一个用于测试 t 检验的函数，使用随机种子以确保结果可复现
    def test_ttest_ind_randperm_alternative(self):
        # 设置随机种子为0，确保随机数生成的一致性
        np.random.seed(0)
        # 设置样本大小 N
        N = 50
        # 生成随机数组 a，形状为 (2, 3, N)
        a = np.random.rand(2, 3, N)
        # 生成随机数组 b，形状为 (3, N)
        b = np.random.rand(3, N)
        # 定义 t 检验的参数选项，包括轴、置换次数和随机种子
        options_p = {'axis': -1, 'permutations': 1000, "random_state": 0}

        # 更新参数选项，设置 alternative 为 "greater"
        options_p.update(alternative="greater")
        # 执行 t 检验，将结果存储在 res_g_ab 中
        res_g_ab = stats.ttest_ind(a, b, **options_p)
        # 再次执行 t 检验，交换 a 和 b 的位置，将结果存储在 res_g_ba 中
        res_g_ba = stats.ttest_ind(b, a, **options_p)

        # 更新参数选项，设置 alternative 为 "less"
        options_p.update(alternative="less")
        # 执行 t 检验，将结果存储在 res_l_ab 中
        res_l_ab = stats.ttest_ind(a, b, **options_p)
        # 再次执行 t 检验，交换 a 和 b 的位置，将结果存储在 res_l_ba 中
        res_l_ba = stats.ttest_ind(b, a, **options_p)

        # 断言：不同的 alternative 对统计量没有影响
        assert_equal(res_g_ab.statistic, res_l_ab.statistic)

        # 断言：交换输入顺序会使统计量的符号取反
        assert_equal(res_g_ab.statistic, -res_g_ba.statistic)
        assert_equal(res_l_ab.statistic, -res_l_ba.statistic)

        # 对于随机置换，观察到的检验统计量与总体之间的重合几率很小，因此：
        assert_equal(res_g_ab.pvalue + res_l_ab.pvalue,
                     1 + 1/(options_p['permutations'] + 1))
        assert_equal(res_g_ba.pvalue + res_l_ba.pvalue,
                     1 + 1/(options_p['permutations'] + 1))

    # 标记为慢速测试的函数，测试 t 检验的不同 alternative 设置
    @pytest.mark.slow()
    def test_ttest_ind_randperm_alternative2(self):
        # 设置随机种子为0，确保随机数生成的一致性
        np.random.seed(0)
        # 设置样本大小 N
        N = 50
        # 生成随机数组 a，形状为 (N, 4)
        a = np.random.rand(N, 4)
        # 生成随机数组 b，形状为 (N, 4)
        b = np.random.rand(N, 4)
        # 定义 t 检验的参数选项，包括置换次数和随机种子
        options_p = {'permutations': 20000, "random_state": 0}

        # 更新参数选项，设置 alternative 为 "greater"
        options_p.update(alternative="greater")
        # 执行 t 检验，将结果存储在 res_g_ab 中
        res_g_ab = stats.ttest_ind(a, b, **options_p)

        # 更新参数选项，设置 alternative 为 "less"
        options_p.update(alternative="less")
        # 执行 t 检验，将结果存储在 res_l_ab 中
        res_l_ab = stats.ttest_ind(a, b, **options_p)

        # 更新参数选项，设置 alternative 为 "two-sided"
        options_p.update(alternative="two-sided")
        # 执行 t 检验，将结果存储在 res_2_ab 中
        res_2_ab = stats.ttest_ind(a, b, **options_p)

        # 对于随机置换，观察到的检验统计量与总体之间的重合几率很小，因此：
        assert_equal(res_g_ab.pvalue + res_l_ab.pvalue,
                     1 + 1/(options_p['permutations'] + 1))

        # 对于大样本量，分布应当大致对称，因此这些身份应当近似满足
        mask = res_g_ab.pvalue <= 0.5
        assert_allclose(2 * res_g_ab.pvalue[mask],
                        res_2_ab.pvalue[mask], atol=2e-2)
        assert_allclose(2 * (1-res_g_ab.pvalue[~mask]),
                        res_2_ab.pvalue[~mask], atol=2e-2)
        assert_allclose(2 * res_l_ab.pvalue[~mask],
                        res_2_ab.pvalue[~mask], atol=2e-2)
        assert_allclose(2 * (1-res_l_ab.pvalue[mask]),
                        res_2_ab.pvalue[mask], atol=2e-2)
    def test_ttest_ind_permutation_nanpolicy(self):
        # 设置随机种子确保结果可重复
        np.random.seed(0)
        # 创建两个大小为 (50, 5) 的随机数组
        N = 50
        a = np.random.rand(N, 5)
        b = np.random.rand(N, 5)
        # 在数组 a 和 b 中插入 NaN 值
        a[5, 1] = np.nan
        b[8, 2] = np.nan
        a[9, 3] = np.nan
        b[9, 3] = np.nan
        # 设置 t 检验的选项，包括置换次数和随机状态
        options_p = {'permutations': 1000, "random_state": 0}

        # 当输入数据中包含 NaN 时，使用 'raise' 策略应抛出 ValueError 异常
        options_p.update(nan_policy="raise")
        with assert_raises(ValueError, match="The input contains nan values"):
            res = stats.ttest_ind(a, b, **options_p)

        # 当输入数据中包含 NaN 时，使用 'propagate' 策略应传播 NaN 值
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, "invalid value*")
            options_p.update(nan_policy="propagate")
            res = stats.ttest_ind(a, b, **options_p)

            # 创建用于过滤 NaN 值的掩码
            mask = np.isnan(a).any(axis=0) | np.isnan(b).any(axis=0)
            res2 = stats.ttest_ind(a[:, ~mask], b[:, ~mask], **options_p)

            # 检查传播策略下 NaN 值的处理情况
            assert_equal(res.pvalue[mask], np.nan)
            assert_equal(res.statistic[mask], np.nan)

            # 对非 NaN 值的统计量和 p 值进行比较
            assert_allclose(res.pvalue[~mask], res2.pvalue)
            assert_allclose(res.statistic[~mask], res2.statistic)

            # 对一维数据进行 t 检验时，检查返回的 p 值和统计量是否为 NaN
            res = stats.ttest_ind(a.ravel(), b.ravel(), **options_p)
            assert np.isnan(res.pvalue)  # assert makes sure it's a scalar
            assert np.isnan(res.statistic)

    def test_ttest_ind_permutation_check_inputs(self):
        # 当传入的置换次数不合法时，应该抛出 ValueError 异常
        with assert_raises(ValueError, match="Permutations must be"):
            stats.ttest_ind(self.a2, self.b2, permutations=-3)
        with assert_raises(ValueError, match="Permutations must be"):
            stats.ttest_ind(self.a2, self.b2, permutations=1.5)
        # 当 random_state 参数不合法时，应该抛出 ValueError 异常
        with assert_raises(ValueError, match="'hello' cannot be used"):
            stats.ttest_ind(self.a, self.b, permutations=1,
                            random_state='hello', axis=1)

    def test_ttest_ind_permutation_check_p_values(self):
        # 检查 t 检验的 p 值是否会出现精确为零的情况
        N = 10
        a = np.random.rand(N, 20)
        b = np.random.rand(N, 20)
        p_values = stats.ttest_ind(a, b, permutations=1).pvalue
        print(0.0 not in p_values)
        assert 0.0 not in p_values

    @array_api_compatible
    @pytest.mark.skip_xp_backends(cpu_only=True,
                                  reasons=['Uses NumPy for pvalue, CI'])
    @pytest.mark.usefixtures("skip_xp_backends")
    def test_permutation_not_implement_for_xp(self, xp):
        # 当使用的数组库不是 NumPy 时，应该抛出 NotImplementedError 异常
        message = "Use of `permutations` is compatible only with NumPy arrays."
        a2, b2 = xp.asarray(self.a2), xp.asarray(self.b2)
        if is_numpy(xp):  # no error
            stats.ttest_ind(a2, b2, permutations=10)
        else:  # NotImplementedError
            with pytest.raises(NotImplementedError, match=message):
                stats.ttest_ind(a2, b2, permutations=10)
# 定义测试类 Test_ttest_ind_common，用于测试 t-检验的多维数组变体，如置换和修剪等
class Test_ttest_ind_common:

    # 声明一个标记，表示这是一个运行较慢的测试
    @pytest.mark.xslow()

    # 参数化装饰器，用于多个测试参数的组合
    @pytest.mark.parametrize("kwds", [{'permutations': 200, 'random_state': 0},
                                      {'trim': .2}, {}],
                             ids=["permutations", "trim", "basic"])

    # 参数化装饰器，测试是否相等的两种变量条件
    @pytest.mark.parametrize('equal_var', [True, False],
                             ids=['equal_var', 'unequal_var'])

    # 定义测试方法 test_ttest_many_dims，接受参数 kwds 和 equal_var
    def test_ttest_many_dims(self, kwds, equal_var):
        # 设置随机种子为0，生成随机数据数组 a 和 b
        np.random.seed(0)
        a = np.random.rand(5, 4, 4, 7, 1, 6)
        b = np.random.rand(4, 1, 8, 2, 6)

        # 进行 t-检验，计算统计量和 p 值，axis=-3 表示沿第三个轴进行计算
        res = stats.ttest_ind(a, b, axis=-3, **kwds)

        # 比较完全向量化的 t-检验结果和较小切片的 t-检验结果
        i, j, k = 2, 3, 1
        a2 = a[i, :, j, :, 0, :]
        b2 = b[:, 0, :, k, :]
        res2 = stats.ttest_ind(a2, b2, axis=-2, **kwds)

        # 断言两组结果的统计量和 p 值相等
        assert_equal(res.statistic[i, :, j, k, :],
                     res2.statistic)
        assert_equal(res.pvalue[i, :, j, k, :],
                     res2.pvalue)

        # 按一维轴依次进行 t-检验比较

        # 使用 tile 手动广播数据；将轴移至末尾以简化操作
        x = np.moveaxis(np.tile(a, (1, 1, 1, 1, 2, 1)), -3, -1)
        y = np.moveaxis(np.tile(b, (5, 1, 4, 1, 1, 1)), -3, -1)
        shape = x.shape[:-1]
        statistics = np.zeros(shape)
        pvalues = np.zeros(shape)

        # 遍历所有可能的索引组合，对每个轴切片执行 t-检验
        for indices in product(*(range(i) for i in shape)):
            xi = x[indices]  # 使用元组索引单个轴切片
            yi = y[indices]
            res3 = stats.ttest_ind(xi, yi, axis=-1, **kwds)
            statistics[indices] = res3.statistic
            pvalues[indices] = res3.pvalue

        # 断言所有 t-检验的统计量和 p 值都接近于整体 t-检验的结果
        assert_allclose(statistics, res.statistic)
        assert_allclose(pvalues, res.pvalue)

    # 参数化装饰器，用于测试不同的 kwds 参数组合
    @pytest.mark.parametrize("kwds", [{'permutations': 200, 'random_state': 0},
                                      {'trim': .2}, {}],
                             ids=["trim", "permutations", "basic"])

    # 参数化装饰器，测试不同的 axis 参数
    @pytest.mark.parametrize("axis", [-1, 0])
    # 定义一个测试方法，用于测试在指定轴上使用给定关键字参数时的 NaN 处理情况
    def test_nans_on_axis(self, kwds, axis):
        # 确保在 `nan_policy='propagate'` 下，正确位置返回 NaN 结果
        a = np.random.randint(10, size=(5, 3, 10)).astype('float')
        b = np.random.randint(10, size=(5, 3, 10)).astype('float')
        
        # 将 `a` 和 `b` 中的部分索引设置为 `np.nan`
        a[0][2][3] = np.nan
        b[2][0][6] = np.nan

        # 将 `np.sum` 任意用作确定应该是 NaN 的索引的基准
        expected = np.isnan(np.sum(a + b, axis=axis))
        
        # 多维输入到 `t.sf(np.abs(t), df)`，在某些索引上有 NaN 会抛出警告。参见问题 gh-13844
        with suppress_warnings() as sup, np.errstate(invalid="ignore"):
            sup.filter(RuntimeWarning,
                       "invalid value encountered in less_equal")
            sup.filter(RuntimeWarning, "Precision loss occurred")
            # 执行 t 检验，检查在给定轴上的两组数据 `a` 和 `b`
            res = stats.ttest_ind(a, b, axis=axis, **kwds)
        
        # 检查 t 检验结果的 p 值是否为 NaN，并与预期的结果进行比较
        p_nans = np.isnan(res.pvalue)
        assert_array_equal(p_nans, expected)
        
        # 检查 t 检验结果的统计量是否为 NaN，并与预期的结果进行比较
        statistic_nans = np.isnan(res.statistic)
        assert_array_equal(statistic_nans, expected)
# 定义一个测试类 Test_ttest_trim，用于测试 ttest_trim 方法
class Test_ttest_trim:
    # 参数化测试数据，每组参数包含两个列表 a 和 b，预期的 p 值 pr、统计量 tr，以及修剪比例 trim
    params = [
        [[1, 2, 3], [1.1, 2.9, 4.2], 0.53619490753126731, -0.6864951273557258, .2],
        [[56, 128.6, 12, 123.8, 64.34, 78, 763.3], [1.1, 2.9, 4.2], 0.00998909252078421, 4.591598691181999, .2],
        [[56, 128.6, 12, 123.8, 64.34, 78, 763.3], [1.1, 2.9, 4.2], 0.10512380092302633, 2.832256715395378, .32],
        [[2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9], [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1], 0.002878909511344, -4.2461168970325, .2],
        [[-0.84504783, 0.13366078, 3.53601757, -0.62908581, 0.54119466, -1.16511574, -0.08836614, 1.18495416, 2.48028757, -1.58925028, -1.6706357, 0.3090472, -2.12258305, 0.3697304, -1.0415207, -0.57783497, -0.90997008, 1.09850192, 0.41270579, -1.4927376], [1.2725522, 1.1657899, 2.7509041, 1.2389013, -0.9490494, -1.0752459, 1.1038576, 2.9912821, 3.5349111, 0.4171922, 1.0168959, -0.7625041, -0.4300008, 3.0431921, 1.6035947, 0.5285634, -0.7649405, 1.5575896, 1.3670797, 1.1726023], 0.005293305834235, -3.0983317739483, .2]
    ]

    # 使用 pytest.mark.parametrize 装饰器，将参数化数据应用到 test_ttest_compare_r 方法
    @pytest.mark.parametrize("a,b,pr,tr,trim", params)
    def test_ttest_compare_r(self, a, b, pr, tr, trim):
        '''
        使用 PairedData 的 yuen.t.test 方法。需要注意的是，至少有三个 R 包含了修剪 t 检验方法，
        并且对它们进行了比较。发现 PairedData 方法的结果与此方法、SAS 和其中一个其他 R 方法一致。
        值得注意的是，DescTools 实现的功能存在显著差异，它只在某些情况下与 SAS、WRS2、PairedData 和
        这个实现一致。因此，在 R 中大多数比较是针对 PairedData 方法进行的。

        而不是提供所有评估的输入和输出，这里提供一个代表性示例：
        > library(PairedData)
        > a <- c(1, 2, 3)
        > b <- c(1.1, 2.9, 4.2)
        > options(digits=16)
        > yuen.t.test(a, b, tr=.2)

            两样本 Yuen 检验，修剪比例为 0.2

        数据：x 和 y
        t = -0.68649512735573, df = 3.4104431643464, p-value = 0.5361949075313
        备择假设：修剪后均值的真实差异不等于 0
        95% 置信区间：
         -3.912777195645217  2.446110528978550
        样本估计：
        x 的修剪均值 y 的修剪均值
        2.000000000000000 2.73333333333333
        '''
        # 使用 scipy 的 stats.ttest_ind 方法进行 t 检验，设定 equal_var=False 表示不假定方差相等
        statistic, pvalue = stats.ttest_ind(a, b, trim=trim, equal_var=False)
        # 使用 assert_allclose 进行数值比较，验证统计量和 p 值是否符合预期
        assert_allclose(statistic, tr, atol=1e-15)
        assert_allclose(pvalue, pr, atol=1e-15)
    def test_compare_SAS(self):
        # 数据来源：https://support.sas.com/resources/papers/proceedings14/1660-2014.pdf
        # 定义两个数组 a 和 b，用于进行 t 检验
        a = [12, 14, 18, 25, 32, 44, 12, 14, 18, 25, 32, 44]
        b = [17, 22, 14, 12, 30, 29, 19, 17, 22, 14, 12, 30, 29, 19]
        # 执行 t 检验，设置 trim 参数为 0.09，不使用等方差性，返回统计量和 p 值
        statistic, pvalue = stats.ttest_ind(a, b, trim=.09, equal_var=False)
        # 使用 assert_allclose 函数断言 p 值接近 0.514522，允许误差为 1e-6
        assert_allclose(pvalue, 0.514522, atol=1e-6)
        # 使用 assert_allclose 函数断言统计量接近 0.669169，允许误差为 1e-6
        assert_allclose(statistic, 0.669169, atol=1e-6)

    def test_equal_var(self):
        '''
        PairedData 库仅支持不等方差的 t 检验。若要比较等方差的样本，需要使用 multicon 库。
        > library(multicon)
        > a <- c(2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9)
        > b <- c(6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1)
        > dv = c(a,b)
        > iv = c(rep('a', length(a)), rep('b', length(b)))
        > yuenContrast(dv~ iv, EQVAR = TRUE)
        $Ms
           N                 M wgt
        a 11 2.442857142857143   1
        b 11 5.385714285714286  -1

        $test
                              stat df              crit                   p
        results -4.246116897032513 12 2.178812829667228 0.00113508833897713
        '''
        # 定义两个数组 a 和 b，用于进行 t 检验
        a = [2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9]
        b = [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1]
        # 默认情况下，使用 equal_var=True 进行 t 检验
        statistic, pvalue = stats.ttest_ind(a, b, trim=.2)
        # 使用 assert_allclose 函数断言 p 值接近 0.00113508833897713，允许误差为 1e-10
        assert_allclose(pvalue, 0.00113508833897713, atol=1e-10)
        # 使用 assert_allclose 函数断言统计量接近 -4.246116897032513，允许误差为 1e-10
        assert_allclose(statistic, -4.246116897032513, atol=1e-10)

    @pytest.mark.parametrize('alt,pr,tr',
                             (('greater', 0.9985605452443, -4.2461168970325),
                              ('less', 0.001439454755672, -4.2461168970325),),
                             )
    def test_alternatives(self, alt, pr, tr):
        '''
        > library(PairedData)
        > a <- c(2.7,2.7,1.1,3.0,1.9,3.0,3.8,3.8,0.3,1.9,1.9)
        > b <- c(6.5,5.4,8.1,3.5,0.5,3.8,6.8,4.9,9.5,6.2,4.1)
        > options(digits=16)
        > yuen.t.test(a, b, alternative = 'greater')
        '''
        # 定义两个数组 a 和 b，用于进行 t 检验
        a = [2.7, 2.7, 1.1, 3.0, 1.9, 3.0, 3.8, 3.8, 0.3, 1.9, 1.9]
        b = [6.5, 5.4, 8.1, 3.5, 0.5, 3.8, 6.8, 4.9, 9.5, 6.2, 4.1]
        # 执行 t 检验，设置 trim 参数为 0.2，不使用等方差性，指定备择假设
        statistic, pvalue = stats.ttest_ind(a, b, trim=.2, equal_var=False,
                                            alternative=alt)
        # 使用 assert_allclose 函数断言 p 值接近预期的 pr 值，允许误差为 1e-10
        assert_allclose(pvalue, pr, atol=1e-10)
        # 使用 assert_allclose 函数断言统计量接近预期的 tr 值，允许误差为 1e-10
        assert_allclose(statistic, tr, atol=1e-10)
    # 定义一个测试方法，用于验证在使用排列时尝试修剪会引发错误
    def test_errors_unsupported(self):
        # 错误匹配信息，确认尝试同时使用 `trim` 和 `permutations` 会引发错误
        match = "Use of `permutations` is incompatible with with use of `trim`."
        # 使用 assert_raises 断言期望抛出 NotImplementedError 异常，并匹配特定错误信息
        with assert_raises(NotImplementedError, match=match):
            # 调用 stats.ttest_ind 方法，传入参数并指定 trim 和 permutations
            stats.ttest_ind([1, 2], [2, 3], trim=.2, permutations=2)

    # 使用装饰器标记此方法与数组 API 兼容，并跳过特定后端的测试
    @array_api_compatible
    @pytest.mark.skip_xp_backends(cpu_only=True,
                                  reasons=['Uses NumPy for pvalue, CI'])
    @pytest.mark.usefixtures("skip_xp_backends")
    # 定义一个测试方法，用于验证在非 NumPy 数组上使用排列会抛出错误
    def test_permutation_not_implement_for_xp(self, xp):
        # 错误信息，指出只有在 NumPy 数组上使用 `trim` 才兼容
        message = "Use of `trim` is compatible only with NumPy arrays."
        # 创建两个数组 a 和 b，使用 xp.arange 方法，根据不同后端选择不同实现
        a, b = xp.arange(10), xp.arange(10)+1
        # 如果当前使用的是 NumPy，则不应该抛出错误
        if is_numpy(xp):  # no error
            # 在 NumPy 数组上调用 stats.ttest_ind 方法，指定 trim 参数
            stats.ttest_ind(a, b, trim=0.1)
        else:  # 如果不是 NumPy，则期望抛出 NotImplementedError 异常
            with pytest.raises(NotImplementedError, match=message):
                # 在非 NumPy 数组上调用 stats.ttest_ind 方法，指定 trim 参数
                stats.ttest_ind(a, b, trim=0.1)

    # 使用 pytest.mark.parametrize 装饰器指定参数范围，对 trim 进行边界测试
    @pytest.mark.parametrize("trim", [-.2, .5, 1])
    # 定义一个测试方法，用于验证 trim 参数超出范围会引发 ValueError
    def test_trim_bounds_error(self, trim):
        # 错误匹配信息，指出 trim 参数应在 0 到 0.5 之间
        match = "Trimming percentage should be 0 <= `trim` < .5."
        # 使用 assert_raises 断言期望抛出 ValueError 异常，并匹配特定错误信息
        with assert_raises(ValueError, match=match):
            # 调用 stats.ttest_ind 方法，传入参数并指定 trim
            stats.ttest_ind([1, 2], [2, 1], trim=trim)
@array_api_compatible
# 添加 NumPy 数组兼容性装饰器
@pytest.mark.skip_xp_backends(cpu_only=True,
                              reasons=['Uses NumPy for pvalue, CI'])
# 标记为跳过 XP 后端，仅限 CPU，并给出跳过原因
@pytest.mark.usefixtures("skip_xp_backends")
# 使用 pytest 的 usefixtures 装饰器，跳过 XP 后端

class Test_ttest_CI:
    # T 检验置信区间测试类

    # indices in order [alternative={two-sided, less, greater},
    #                   equal_var={False, True}, trim={0, 0.2}]
    # 索引顺序 [假设类型={双侧, 小于, 大于}, 方差齐性={False, True}, 修剪={0, 0.2}]

    # reference values in order `statistic, df, pvalue, low, high`
    # 参考值顺序 `统计量, 自由度, P 值, 下限, 上限`

    # equal_var=False reference values computed with R PairedData yuen.t.test:
    #
    # library(PairedData)
    # options(digits=16)
    # a < - c(0.88236329, 0.97318744, 0.4549262, 0.97893335, 0.0606677,
    #         0.44013366, 0.55806018, 0.40151434, 0.14453315, 0.25860601,
    #         0.20202162)
    # b < - c(0.93455277, 0.42680603, 0.49751939, 0.14152846, 0.711435,
    #         0.77669667, 0.20507578, 0.78702772, 0.94691855, 0.32464958,
    #         0.3873582, 0.35187468, 0.21731811)
    # yuen.t.test(a, b, tr=0, conf.level = 0.9, alternative = 'l')
    #
    # 使用 R 中的 PairedData 库计算 equal_var=False 的参考值

    # equal_var=True reference values computed with R multicon yuenContrast:
    #
    # library(multicon)
    # options(digits=16)
    # a < - c(0.88236329, 0.97318744, 0.4549262, 0.97893335, 0.0606677,
    #         0.44013366, 0.55806018, 0.40151434, 0.14453315, 0.25860601,
    #         0.20202162)
    # b < - c(0.93455277, 0.42680603, 0.49751939, 0.14152846, 0.711435,
    #         0.77669667, 0.20507578, 0.78702772, 0.94691855, 0.32464958,
    #         0.3873582, 0.35187468, 0.21731811)
    # dv = c(a, b)
    # iv = c(rep('a', length(a)), rep('b', length(b)))
    # yuenContrast(dv~iv, EQVAR = FALSE, alternative = 'unequal', tr = 0.2)
    #
    # 使用 R 中的 multicon 库计算 equal_var=True 的参考值

    r = np.empty(shape=(3, 2, 2, 5))
    # 创建一个空的 NumPy 数组 r，形状为 (3, 2, 2, 5)

    r[0, 0, 0] = [-0.2314607, 19.894435, 0.8193209, -0.247220294, 0.188729943]
    r[1, 0, 0] = [-0.2314607, 19.894435, 0.40966045, -np.inf, 0.1382426469]
    r[2, 0, 0] = [-0.2314607, 19.894435, 0.5903395, -0.1967329982, np.inf]
    # 填充 r 数组的第一个组合条件下的值

    r[0, 0, 1] = [-0.2452886, 11.427896, 0.8105823, -0.34057446, 0.25847383]
    r[1, 0, 1] = [-0.2452886, 11.427896, 0.40529115, -np.inf, 0.1865829074]
    r[2, 0, 1] = [-0.2452886, 11.427896, 0.5947089, -0.268683541, np.inf]
    # 填充 r 数组的第二个组合条件下的值

    # confidence interval not available for equal_var=True
    # equal_var=True 时置信区间不可用

    r[0, 1, 0] = [-0.2345625322555006, 22, 0.8167175905643815, None, None]
    r[1, 1, 0] = [-0.2345625322555006, 22, 0.4083587952821908, None, None]
    r[2, 1, 0] = [-0.2345625322555006, 22, 0.5916412047178092, None, None]
    # 填充 r 数组的第三个组合条件下的值，equal_var=True 时没有置信区间

    r[0, 1, 1] = [-0.2505369406507428, 14, 0.8058115135702835, None, None]
    r[1, 1, 1] = [-0.2505369406507428, 14, 0.4029057567851417, None, None]
    r[2, 1, 1] = [-0.2505369406507428, 14, 0.5970942432148583, None, None]
    # 填充 r 数组的第四个组合条件下的值，equal_var=True 时没有置信区间

    @pytest.mark.parametrize('alternative', ['two-sided', 'less', 'greater'])
    # 使用 pytest 的 parametrize 装饰器，参数化 alternative 变量
    @pytest.mark.parametrize('equal_var', [False, True])
    # 参数化 equal_var 变量
    @pytest.mark.parametrize('trim', [0, 0.2])
    # 参数化 trim 变量
    # 定义一个测试函数，用于计算两组数据的 t 检验及置信区间
    def test_confidence_interval(self, alternative, equal_var, trim, xp):
        # 如果 equal_var 和 trim 都为 True，则标记测试为失败，需要进一步调查
        if equal_var and trim:
            pytest.xfail('Discrepancy in `main`; needs further investigation.')

        # 如果 trim 为 True 且输入数据不是 NumPy 数组，则跳过测试
        if trim and not is_numpy(xp):
            pytest.skip('`trim` is only compatible with NumPy input')

        # 使用指定种子创建随机数生成器对象
        rng = np.random.default_rng(3810954496107292580)
        # 从随机数生成器中生成长度为 11 的随机数组，并转换为 xp 对应的数组类型
        x = xp.asarray(rng.random(11))
        # 从随机数生成器中生成长度为 13 的随机数组，并转换为 xp 对应的数组类型
        y = xp.asarray(rng.random(13))

        # 执行两组数据的 t 检验，并返回结果对象 res
        res = stats.ttest_ind(x, y, alternative=alternative,
                              equal_var=equal_var, trim=trim)

        # 定义 t 检验的三种备择假设，及其对应的参考值
        alternatives = {'two-sided': 0, 'less': 1, 'greater': 2}
        ref = self.r[alternatives[alternative], int(equal_var), int(np.ceil(trim))]
        statistic, df, pvalue, low, high = ref

        # 定义相对误差容限为 1e-7，用于数值比较
        rtol = 1e-7  # 参考值精度为 7 位有效数字
        # 检查计算结果的统计量与参考值的统计量是否接近
        xp_assert_close(res.statistic, xp.asarray(statistic), rtol=rtol)
        # 检查计算结果的自由度与参考值的自由度是否接近
        xp_assert_close(res.df, xp.asarray(df), rtol=rtol)
        # 检查计算结果的 p 值与参考值的 p 值是否接近
        xp_assert_close(res.pvalue, xp.asarray(pvalue), rtol=rtol)

        # 如果 equal_var 不为 True，则计算置信区间并进行检查
        if not equal_var:  # 当 `equal_var` 不为 True 时，无法计算置信区间
            # 计算置信水平为 0.9 的置信区间
            ci = res.confidence_interval(0.9)
            # 检查计算结果的下限与参考值的下限是否接近
            xp_assert_close(ci.low, xp.asarray(low), rtol=rtol)
            # 检查计算结果的上限与参考值的上限是否接近
            xp_assert_close(ci.high, xp.asarray(high), rtol=rtol)
# 定义测试函数 test__broadcast_concatenate
def test__broadcast_concatenate():
    # 测试 _broadcast_concatenate 函数是否正确地在除了指定轴（axis）以外的所有轴上广播数组，然后沿指定轴连接
    np.random.seed(0)
    # 创建随机数组 a 和 b，分别具有不同的形状
    a = np.random.rand(5, 4, 4, 3, 1, 6)
    b = np.random.rand(4, 1, 8, 2, 6)
    # 调用 _broadcast_concatenate 函数，沿 axis=-3 连接 a 和 b
    c = _broadcast_concatenate((a, b), axis=-3)
    
    # 手动广播作为独立检查
    # 将数组 a 沿着第 5 轴复制 2 次
    a = np.tile(a, (1, 1, 1, 1, 2, 1))
    # 将数组 b 在第 1 轴上添加一个维度，并在其他轴上复制 5 次
    b = np.tile(b[None, ...], (5, 1, 4, 1, 1, 1))
    
    # 遍历 c 数组的所有索引
    for index in product(*(range(i) for i in c.shape)):
        i, j, k, l, m, n = index
        # 如果 l < a.shape[-3]，则断言 c[i, j, k, l, m, n] 等于 a[i, j, k, l, m, n]
        if l < a.shape[-3]:
            assert a[i, j, k, l, m, n] == c[i, j, k, l, m, n]
        # 否则，断言 c[i, j, k, l, m, n] 等于 b[i, j, k, l - a.shape[-3], m, n]
        else:
            assert b[i, j, k, l - a.shape[-3], m, n] == c[i, j, k, l, m, n]


# 使用 array_api_compatible 装饰器装饰的测试函数
@array_api_compatible
# 使用 pytest.mark.skip_xp_backends 装饰器，跳过 CPU 后端的测试
@pytest.mark.skip_xp_backends(cpu_only=True,
                              reasons=['Uses NumPy for pvalue, CI'])
# 使用 pytest.mark.usefixtures 装饰器，跳过特定后端的测试
@pytest.mark.usefixtures("skip_xp_backends")
# 定义测试函数 test_ttest_ind_with_uneq_var
def test_ttest_ind_with_uneq_var(xp):
    # 检查与 R 中的 t.test 函数的比较，例如：
    # options(digits=20)
    # a = c(1., 2., 3.)
    # b = c(1.1, 2.9, 4.2)
    # t.test(a, b, equal.var=FALSE)

    # 创建 xp 数组 a 和 b
    a = xp.asarray([1., 2., 3.])
    b = xp.asarray([1.1, 2.9, 4.2])
    # 创建预期的 t 和 p 值
    pr = xp.asarray(0.53619490753126686)
    tr = xp.asarray(-0.686495127355726265)

    # 使用 scipy.stats.ttest_ind 计算 t 和 p 值，假设方差不相等
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # 断言计算得到的 t 值等于预期值 tr
    xp_assert_close(t, tr)
    # 断言计算得到的 p 值等于预期值 pr

    # 使用 scipy.stats.ttest_ind_from_stats 函数，从描述性统计信息计算 t 和 p 值，假设方差不相等
    t, p = stats.ttest_ind_from_stats(*_desc_stats(a, b), equal_var=False)
    # 断言计算得到的 t 值等于预期值 tr
    xp_assert_close(t, tr)
    # 断言计算得到的 p 值等于预期值 pr

    # 修改数组 a 的内容，并更新预期的 t 和 p 值
    a = xp.asarray([1., 2., 3., 4.])
    pr = xp.asarray(0.84354139131608252)
    tr = xp.asarray(-0.210866331595072315)

    # 重新计算 t 和 p 值，假设方差不相等
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # 断言计算得到的 t 值等于预期值 tr
    xp_assert_close(t, tr)
    # 断言计算得到的 p 值等于预期值 pr

    # 再次使用 scipy.stats.ttest_ind_from_stats 函数，从描述性统计信息计算 t 和 p 值，假设方差不相等
    t, p = stats.ttest_ind_from_stats(*_desc_stats(a, b), equal_var=False)
    # 断言计算得到的 t 值等于预期值 tr
    xp_assert_close(t, tr)
    # 断言计算得到的 p 值等于预期值 pr

    # 回归测试
    # 更新预期的 t 和 p 值
    tr = xp.asarray(1.0912746897927283)
    tr_uneq_n = xp.asarray(0.66745638708050492)
    pr = xp.asarray(0.27647831993021388)
    pr_uneq_n = xp.asarray(0.50873585065616544)
    tr_2D = xp.asarray([tr, -tr])
    pr_2D = xp.asarray([pr, pr])

    # 创建不同的随机数值
    rvs3 = xp.linspace(1, 100, 25)
    rvs2 = xp.linspace(1, 100, 100)
    rvs1 = xp.linspace(5, 105, 100)
    rvs1_2D = xp.stack([rvs1, rvs2])
    rvs2_2D = xp.stack([rvs2, rvs1])

    # 计算 t 和 p 值，假设方差不相等
    t, p = stats.ttest_ind(rvs1, rvs2, axis=0, equal_var=False)
    # 断言计算得到的 t 值等于预期值 tr
    xp_assert_close(t, tr)
    # 断言计算得到的 p 值等于预期值 pr

    # 再次使用 scipy.stats.ttest_ind_from_stats 函数，从描述性统计信息计算 t 和 p 值，假设方差不相等
    t, p = stats.ttest_ind_from_stats(*_desc_stats(rvs1, rvs2), equal_var=False)
    # 断言计算得到的 t 值等于预期值 tr
    xp_assert_close(t, tr)
    # 断言计算得到的 p 值等于预期值 pr

    # 计算 t 和 p 值，假设方差不相等
    t, p = stats.ttest_ind(rvs1, rvs3, axis=0, equal_var=False)
    # 断言计算得到的 t 值等于预期值 tr_uneq_n
    xp_assert_close(t, tr_uneq_n)
    # 断言计算得到的 p 值等于预期值 pr_uneq_n

    # 再次使用 scipy.stats.ttest_ind_from_stats 函数，从描述性统计信息计算 t 和 p 值，假设方差不相等
    t, p = stats.ttest_ind_from_stats(*_desc_stats(rvs1, rvs3), equal_var=False)
    # 断言计算得到的 t 值等于预期值 tr_uneq_n
    xp_assert_close(t, tr_uneq_n)
    # 断言计算得到的 p 值等于预期值 pr_uneq_n

    # 计算 t 和 p 值，假设方差不相等
    res = stats.ttest_ind(rvs1_2D.T, rvs2_2D.T, axis=0, equal_var=False)
    # 断言计算得到的统计量等于预期值 tr_2D
    xp_assert_close(res.stat
    # 使用_desc_stats函数计算rvs1_2D.T和rvs2_2D.T的描述统计信息，并进行独立双样本t检验，要求方差不相等
    args = _desc_stats(rvs1_2D.T, rvs2_2D.T)
    # 根据给定的描述统计信息进行独立双样本t检验，要求方差不相等，返回统计量和p值
    res = stats.ttest_ind_from_stats(*args, equal_var=False)
    # 使用xp_assert_close断言检验结果的统计量与预期的tr_2D的接近程度
    xp_assert_close(res.statistic, tr_2D)
    # 使用xp_assert_close断言检验结果的p值与预期的pr_2D的接近程度
    xp_assert_close(res.pvalue, pr_2D)
    
    # 对rvs1_2D和rvs2_2D进行独立双样本t检验，沿指定轴(axis=1)计算
    res = stats.ttest_ind(rvs1_2D, rvs2_2D, axis=1, equal_var=False)
    # 使用xp_assert_close断言检验结果的统计量与预期的tr_2D的接近程度
    xp_assert_close(res.statistic, tr_2D)
    # 使用xp_assert_close断言检验结果的p值与预期的pr_2D的接近程度
    xp_assert_close(res.pvalue, pr_2D)
    
    # 使用_desc_stats函数计算rvs1_2D和rvs2_2D沿指定轴(axis=1)的描述统计信息，并进行独立双样本t检验，要求方差不相等
    args = _desc_stats(rvs1_2D, rvs2_2D, axis=1)
    # 根据给定的描述统计信息进行独立双样本t检验，要求方差不相等，返回统计量和p值
    res = stats.ttest_ind_from_stats(*args, equal_var=False)
    # 使用xp_assert_close断言检验结果的统计量与预期的tr_2D的接近程度
    xp_assert_close(res.statistic, tr_2D)
    # 使用xp_assert_close断言检验结果的p值与预期的pr_2D的接近程度
    xp_assert_close(res.pvalue, pr_2D)
# 标记此函数兼容数组 API
@array_api_compatible
# 使用 pytest.mark.skip_xp_backends 装饰器跳过 CPU 测试，并添加原因说明
@pytest.mark.skip_xp_backends(cpu_only=True,
                              reasons=['Uses NumPy for pvalue, CI'])
# 使用 pytest.mark.usefixtures 装饰器确保在测试之前跳过特定后端的设置
@pytest.mark.usefixtures("skip_xp_backends")
# 定义 ttest_ind_zero_division 测试函数，使用 xp 作为参数
def test_ttest_ind_zero_division(xp):
    # 创建长度为 3 的零数组 x
    x = xp.zeros(3)
    # 创建长度为 3 的全一数组 y
    y = xp.ones(3)
    # 断言会触发 RuntimeWarning 并匹配给定的警告消息
    with pytest.warns(RuntimeWarning, match="Precision loss occurred"):
        # 执行 t 检验，不假设方差相等
        t, p = stats.ttest_ind(x, y, equal_var=False)
    # 断言 t 的值与负无穷相等
    xp_assert_equal(t, xp.asarray(-xp.inf))
    # 断言 p 的值与零相等
    xp_assert_equal(p, xp.asarray(0.))

    # 忽略所有的数值错误警告
    with np.errstate(all='ignore'):
        # 执行 t 检验，不假设方差相等，输入为相同数组 x
        t, p = stats.ttest_ind(x, x, equal_var=False)
        # 断言 t 的值与 NaN 相等
        xp_assert_equal(t, xp.asarray(xp.nan))
        # 断言 p 的值与 NaN 相等
        xp_assert_equal(p, xp.asarray(xp.nan))

        # 创建包含 NaN 的输入数组 anan
        anan = xp.asarray([[1, xp.nan], [-1, 1]])
        # 执行 t 检验，比较 anan 与全零数组的差异，不假设方差相等
        t, p = stats.ttest_ind(anan, xp.zeros((2, 2)), equal_var=False)
        # 断言 t 的值与 [0., NaN] 数组相等
        xp_assert_equal(t, xp.asarray([0., np.nan]))
        # 断言 p 的值与 [1., NaN] 数组相等
        xp_assert_equal(p, xp.asarray([1., np.nan]))


# 定义 test_ttest_ind_nan_2nd_arg 测试函数
def test_ttest_ind_nan_2nd_arg():
    # 回归测试 gh-6134: 第二个参数中的 NaN 未被处理的情况
    x = [np.nan, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 1.0, 2.0]

    # 执行 t 检验，使用 nan_policy='omit' 忽略 NaN
    r1 = stats.ttest_ind(x, y, nan_policy='omit')
    r2 = stats.ttest_ind(y, x, nan_policy='omit')
    # 断言 r2 的统计值与 r1 的统计值相反
    assert_allclose(r2.statistic, -r1.statistic, atol=1e-15)
    # 断言 r2 的 p 值与 r1 的 p 值相等
    assert_allclose(r2.pvalue, r1.pvalue, atol=1e-15)

    # NB: 当丢弃 NaN 时，参数不会成对处理
    # 执行 t 检验，y 与 x 的子数组（去除第一个元素 NaN）比较
    r3 = stats.ttest_ind(y, x[1:])
    # 断言 r2 与 r3 的结果相等
    assert_allclose(r2, r3, atol=1e-15)

    # .. 这与 R 一致。R 代码：
    # x = c(NA, 2.0, 3.0, 4.0)
    # y = c(1.0, 2.0, 1.0, 2.0)
    # t.test(x, y, var.equal=TRUE)
    # 断言 r2 与 (-2.5354627641855498, 0.052181400457057901) 数组相等
    assert_allclose(r2, (-2.5354627641855498, 0.052181400457057901),
                    atol=1e-15)


# 标记此函数兼容数组 API
@array_api_compatible
# 使用 pytest.mark.parametrize 装饰器定义参数化测试
@pytest.mark.parametrize('b, expected_shape',
                         # 参数化测试参数列表
                         [(np.empty((1, 5, 0)), (3, 5)),
                          (np.empty((1, 0, 0)), (3, 0))])
# 定义 test_ttest_ind_axis_size_zero 测试函数，接受 b 和 expected_shape 作为参数
def test_ttest_ind_axis_size_zero(b, expected_shape, xp):
    # 在此测试中，轴维度的长度为零。
    # 结果应该是包含 NaN 的数组，其形状由广播非轴维度给定。
    # 创建 shape 为 (3, 1, 0) 的空数组 a
    a = xp.empty((3, 1, 0))
    # 将 b 转换为 xp 数组
    b = xp.asarray(b)
    # 使用 np.testing.suppress_warnings() 上下文来忽略警告
    with np.testing.suppress_warnings() as sup:
        # 过滤 SmallSampleWarning，以匹配 too_small_nd_not_omit
        sup.filter(SmallSampleWarning, too_small_nd_not_omit)
        # 执行 t 检验，比较 a 与 b 在最后一个轴上的差异
        res = stats.ttest_ind(a, b, axis=-1)
    # 断言检查返回的结果类型是否为 stats._stats_py.TtestResult 类型
    assert isinstance(res, stats._stats_py.TtestResult)
    
    # 使用 xp.full 函数创建一个期望值数组，形状与 expected_shape 相同，填充值为 NaN
    expected_value = xp.full(expected_shape, fill_value=xp.nan)
    
    # 使用 xp_assert_equal 函数断言 res.statistic 的值与 expected_value 相等
    xp_assert_equal(res.statistic, expected_value)
    
    # 使用 xp_assert_equal 函数断言 res.pvalue 的值与 expected_value 相等
    xp_assert_equal(res.pvalue, expected_value)
@array_api_compatible
def test_ttest_ind_nonaxis_size_zero(xp):
    # 使用 xp 调用的空数组函数创建数组 a，形状为 (1, 8, 0)
    a = xp.empty((1, 8, 0))
    # 使用 xp 调用的空数组函数创建数组 b，形状为 (5, 8, 1)
    b = xp.empty((5, 8, 1))
    # 对数组 a 和 b 执行 t 检验，沿着 axis=1 进行计算
    res = stats.ttest_ind(a, b, axis=1)
    # 断言 res 类型为 TtestResult 对象
    assert isinstance(res, stats._stats_py.TtestResult)
    # 断言 res.statistic 的形状为 (5, 0)
    assert res.statistic.shape == (5, 0)
    # 断言 res.pvalue 的形状为 (5, 0)
    assert res.pvalue.shape == (5, 0)


@array_api_compatible
def test_ttest_ind_nonaxis_size_zero_different_lengths(xp):
    # 使用 xp 调用的空数组函数创建数组 a，形状为 (1, 7, 0)
    a = xp.empty((1, 7, 0))
    # 使用 xp 调用的空数组函数创建数组 b，形状为 (5, 8, 1)
    b = xp.empty((5, 8, 1))
    # 对数组 a 和 b 执行 t 检验，沿着 axis=1 进行计算
    res = stats.ttest_ind(a, b, axis=1)
    # 断言 res 类型为 TtestResult 对象
    assert isinstance(res, stats._stats_py.TtestResult)
    # 断言 res.statistic 的形状为 (5, 0)
    assert res.statistic.shape == (5, 0)
    # 断言 res.pvalue 的形状为 (5, 0)
    assert res.pvalue.shape == (5, 0)


@array_api_compatible
@pytest.mark.skip_xp_backends(np_only=True,
                              reasons=["Other backends don't like integers"])
@pytest.mark.usefixtures("skip_xp_backends")
def test_gh5686(xp):
    # 使用 xp 调用的 asarray 函数创建数组 mean1 和 mean2
    mean1, mean2 = xp.asarray([1, 2]), xp.asarray([3, 4])
    # 使用 xp 调用的 asarray 函数创建数组 std1 和 std2
    std1, std2 = xp.asarray([5, 3]), xp.asarray([4, 5])
    # 使用 xp 调用的 asarray 函数创建数组 nobs1 和 nobs2
    nobs1, nobs2 = xp.asarray([130, 140]), xp.asarray([100, 150])
    # 调用 stats.ttest_ind_from_stats 函数进行 t 检验
    # 如果 gh-5686 未修复，这会引发 TypeError
    stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)


@array_api_compatible
@pytest.mark.skip_xp_backends(cpu_only=True,
                              reasons=['Uses NumPy for pvalue, CI'])
@pytest.mark.usefixtures("skip_xp_backends")
def test_ttest_ind_from_stats_inputs_zero(xp):
    # 使用 xp 调用的 asarray 函数创建数组 zero, six 和 NaN
    zero = xp.asarray(0.)
    six = xp.asarray(6.)
    NaN = xp.asarray(xp.nan)
    # 调用 stats.ttest_ind_from_stats 函数进行 t 检验
    res = stats.ttest_ind_from_stats(zero, zero, six, zero, zero, six, equal_var=False)
    # 断言 res.statistic 和 res.pvalue 都与 NaN 相等
    xp_assert_equal(res.statistic, NaN)
    xp_assert_equal(res.pvalue, NaN)


@array_api_compatible
@pytest.mark.skip_xp_backends(cpu_only=True,
                              reasons=['Uses NumPy for pvalue, CI'])
@pytest.mark.usefixtures("skip_xp_backends")
def test_ttest_uniform_pvalues(xp):
    # 测试在零假设下，p 值是否均匀分布
    # 使用 np.random.default_rng 生成随机数生成器 rng
    rng = np.random.default_rng(246834602926842)
    # 使用 rng 创建数组 x，形状为 (10000, 2)
    x = xp.asarray(rng.normal(size=(10000, 2)))
    # 使用 rng 创建数组 y，形状为 (10000, 1)
    y = xp.asarray(rng.normal(size=(10000, 1)))
    # 使用 rng 创建数组 q，形状为 (100,)
    q = rng.uniform(size=100)

    # 对 x 和 y 执行 t 检验，equal_var=True，沿着 axis=-1 进行计算
    res = stats.ttest_ind(x, y, equal_var=True, axis=-1)
    # 将 res.pvalue 转换为 numpy 数组
    pvalue = np.asarray(res.pvalue)
    # 使用 stats.ks_1samp 检验 p 值是否服从均匀分布的分布函数
    assert stats.ks_1samp(pvalue, stats.uniform().cdf).pvalue > 0.1
    # 断言 pvalue 在 quantile(q) 处的值与 q 接近，容忍度为 1e-2
    assert_allclose(np.quantile(pvalue, q), q, atol=1e-2)

    # 对 y 和 x 执行 t 检验，equal_var=True，沿着 axis=-1 进行计算
    res = stats.ttest_ind(y, x, equal_var=True, axis=-1)
    # 将 res.pvalue 转换为 numpy 数组
    pvalue = np.asarray(res.pvalue)
    # 使用 stats.ks_1samp 检验 p 值是否服从均匀分布的分布函数
    assert stats.ks_1samp(pvalue, stats.uniform().cdf).pvalue > 0.1
    # 使用 numpy.testing.assert_allclose 函数检查 pvalue 数组的分位数是否与给定的量化水平 q 相匹配，容差为 1e-2
    assert_allclose(np.quantile(pvalue, q), q, atol=1e-2)
    
    # 使用 xp.asarray 将列表 [2, 3, 5] 和 [1.5] 转换为对应的数组 x 和 y
    # xp 可能是 numpy 或 torch，具体取决于环境
    x, y = xp.asarray([2, 3, 5]), xp.asarray([1.5])
    
    # 使用 scipy.stats.ttest_ind 函数进行两组样本 x 和 y 的独立 t 检验，假设方差相等
    res = stats.ttest_ind(x, y, equal_var=True)
    
    # 如果 xp 是 torch，rtol 设为 1e-6，否则设为 1e-10
    rtol = 1e-6 if is_torch(xp) else 1e-10
    
    # 使用 xp.testing.assert_close 函数检查 t 统计量和 p 值是否与预期值接近
    xp_assert_close(res.statistic, xp.asarray(1.0394023007754), rtol=rtol)
    xp_assert_close(res.pvalue, xp.asarray(0.407779907736), rtol=rtol)
# 定义一个私有函数 _convert_pvalue_alternative，用于处理 t 检验的单样本检验结果
def _convert_pvalue_alternative(t, p, alt, xp):
    # 将双侧 p 值转换为单侧 p 值，根据 alt 参数确定转换方向
    less = xp.asarray(alt == "less")  # 检查 alt 是否为 "less"，返回布尔数组
    greater = xp.asarray(alt == "greater")  # 检查 alt 是否为 "greater"，返回布尔数组
    # 根据 t 的正负和 alt 的值，确定使用 p/2 或 1 - p/2 作为单侧 p 值
    i = ((t < 0) & less) | ((t > 0) & greater)
    return xp.where(i, p/2, 1 - p/2)  # 根据 i 的值选择 p/2 或 1 - p/2 作为返回值


# 使用 pytest 的标记进行测试函数的装饰，标记为慢速测试，且仅在 CPU 上运行
# 同时使用 skip_xp_backends 标记跳过特定的 xp 后端（例如 GPU）
@pytest.mark.slow
@pytest.mark.skip_xp_backends(cpu_only=True,
                              reasons=['Uses NumPy for pvalue, CI'])
@pytest.mark.usefixtures("skip_xp_backends")  # 使用 pytest 的 usefixtures 装饰器，跳过 xp 后端
@array_api_compatible  # 声明测试函数与数组 API 兼容
def test_ttest_1samp_new(xp):
    n1, n2, n3 = (10, 15, 20)
    rvn1 = stats.norm.rvs(loc=5, scale=10, size=(n1, n2, n3))
    rvn1 = xp.asarray(rvn1)  # 将随机变量 rvn1 转换为 xp 的数组表示

    # 检查多维数组及正确的轴处理
    # popmean 为全为 1 的数组，与 rvn1 做单样本 t 检验
    popmean = xp.ones((1, n2, n3))
    t1, p1 = stats.ttest_1samp(rvn1, popmean, axis=0)  # 沿着 axis=0 方向进行 t 检验
    t2, p2 = stats.ttest_1samp(rvn1, 1., axis=0)  # 对 rvn1 沿 axis=0 方向做均值为 1 的 t 检验
    t3, p3 = stats.ttest_1samp(rvn1[:, 0, 0], 1.)  # 对 rvn1 的部分切片进行单样本 t 检验
    xp_assert_close(t1, t2, rtol=1e-14)  # 断言 t1 与 t2 接近
    xp_assert_close(t1[0, 0], t3, rtol=1e-14)  # 断言 t1[0, 0] 与 t3 接近
    assert_equal(t1.shape, (n2, n3))  # 断言 t1 的形状为 (n2, n3)

    popmean = xp.ones((n1, 1, n3))
    t1, p1 = stats.ttest_1samp(rvn1, popmean, axis=1)  # 沿着 axis=1 方向进行 t 检验
    t2, p2 = stats.ttest_1samp(rvn1, 1., axis=1)  # 对 rvn1 沿 axis=1 方向做均值为 1 的 t 检验
    t3, p3 = stats.ttest_1samp(rvn1[0, :, 0], 1.)  # 对 rvn1 的部分切片进行单样本 t 检验
    xp_assert_close(t1, t2, rtol=1e-14)  # 断言 t1 与 t2 接近
    xp_assert_close(t1[0, 0], t3, rtol=1e-14)  # 断言 t1[0, 0] 与 t3 接近
    assert_equal(t1.shape, (n1, n3))  # 断言 t1 的形状为 (n1, n3)

    popmean = xp.ones((n1, n2, 1))
    t1, p1 = stats.ttest_1samp(rvn1, popmean, axis=2)  # 沿着 axis=2 方向进行 t 检验
    t2, p2 = stats.ttest_1samp(rvn1, 1., axis=2)  # 对 rvn1 沿 axis=2 方向做均值为 1 的 t 检验
    t3, p3 = stats.ttest_1samp(rvn1[0, 0, :], 1.)  # 对 rvn1 的部分切片进行单样本 t 检验
    xp_assert_close(t1, t2, rtol=1e-14)  # 断言 t1 与 t2 接近
    xp_assert_close(t1[0, 0], t3, rtol=1e-14)  # 断言 t1[0, 0] 与 t3 接近
    assert_equal(t1.shape, (n1, n2))  # 断言 t1 的形状为 (n1, n2)

    # 测试除零问题
    t, p = stats.ttest_1samp(xp.asarray([0., 0., 0.]), 1.)  # 对全为 0 的数组进行单样本 t 检验
    xp_assert_equal(xp.abs(t), xp.asarray(xp.inf))  # 断言 t 的绝对值为 xp 的无穷大
    xp_assert_equal(p, xp.asarray(0.))  # 断言 p 值为 xp 的 0

    tr, pr = stats.ttest_1samp(rvn1[:, :, :], 1.)

    t, p = stats.ttest_1samp(rvn1[:, :, :], 1., alternative="greater")
    pc = _convert_pvalue_alternative(tr, pr, "greater", xp)  # 使用 _convert_pvalue_alternative 处理单侧检验
    xp_assert_close(p, pc)  # 断言计算得到的 p 值与转换后的 p 值接近
    xp_assert_close(t, tr)  # 断言计算得到的 t 值与原始 t 值接近

    t, p = stats.ttest_1samp(rvn1[:, :, :], 1., alternative="less")
    pc = _convert_pvalue_alternative(tr, pr, "less", xp)  # 使用 _convert_pvalue_alternative 处理单侧检验
    xp_assert_close(p, pc)  # 断言计算得到的 p 值与转换后的 p 值接近
    xp_assert_close(t, tr)  # 断言计算得到的 t 值与原始 t 值接近

    with np.errstate(all='ignore'):
        res = stats.ttest_1samp(xp.asarray([0., 0., 0.]), 0.)  # 对全为 0 的数组进行单样本 t 检验
        xp_assert_equal(res.statistic, xp.asarray(xp.nan))  # 断言检验结果的统计量为 xp 的 NaN
        xp_assert_equal(res.pvalue, xp.asarray(xp.nan))  # 断言检验结果的 p 值为 xp 的 NaN

        # 检查输入数组中存在 NaN 值时，检验结果是否正确处理 NaN
        anan = xp.asarray([[1., np.nan], [-1., 1.]])
        res = stats.ttest_1samp(anan, 0.)  # 对包含 NaN 的数组进行单样本 t 检验
        xp_assert_equal(res.statistic, xp.asarray([0., xp.nan]))  # 断言检验结果的统计量正确处理了 NaN
        xp_assert_equal(res.pvalue, xp.asarray([1., xp.nan]))  # 断言检验结果的 p 值正确处理了 NaN
@pytest.mark.usefixtures("skip_xp_backends")
@array_api_compatible
def test_ttest_1samp_new_omit(xp):
    # 定义三个维度的大小
    n1, n2, n3 = (5, 10, 15)
    # 从正态分布生成随机数
    rvn1 = stats.norm.rvs(loc=5, scale=10, size=(n1, n2, n3))
    # 将生成的随机数转换为数组
    rvn1 = xp.asarray(rvn1)

    # 将部分元素设置为 NaN
    rvn1[0:2, 1:3, 4:8] = xp.nan

    # 进行 t 检验，忽略 NaN 值
    tr, pr = stats.ttest_1samp(rvn1[:, :, :], 1., nan_policy='omit')

    # 进行单侧检验，检验是否大于给定值
    t, p = stats.ttest_1samp(rvn1[:, :, :], 1., nan_policy='omit',
                             alternative="greater")
    # 将双侧检验结果转换为单侧检验结果
    pc = _convert_pvalue_alternative(tr, pr, "greater", xp)
    # 断言单侧检验结果的一致性
    xp_assert_close(p, pc)
    xp_assert_close(t, tr)

    # 进行单侧检验，检验是否小于给定值
    t, p = stats.ttest_1samp(rvn1[:, :, :], 1., nan_policy='omit',
                             alternative="less")
    # 将双侧检验结果转换为单侧检验结果
    pc = _convert_pvalue_alternative(tr, pr, "less", xp)
    # 断言单侧检验结果的一致性
    xp_assert_close(p, pc)
    xp_assert_close(t, tr)


@pytest.mark.skip_xp_backends(cpu_only=True,
                              reasons=['Uses NumPy for pvalue, CI'])
@pytest.mark.usefixtures("skip_xp_backends")
@array_api_compatible
def test_ttest_1samp_popmean_array(xp):
    # 当 popmean.shape[axis] != 1 时，抛出错误
    # 如果用户想同时测试多个零假设，则使用标准的广播规则
    rng = np.random.default_rng(2913300596553337193)
    x = rng.random(size=(1, 15, 20))
    x = xp.asarray(x)

    message = r"`popmean.shape\[axis\]` must equal 1."
    # 创建不符合要求的 popmean 数组，预期抛出 ValueError
    popmean = xp.asarray(rng.random(size=(5, 2, 20)))
    with pytest.raises(ValueError, match=message):
        stats.ttest_1samp(x, popmean=popmean, axis=-2)

    # 创建符合要求的 popmean 数组
    popmean = xp.asarray(rng.random(size=(5, 1, 20)))
    # 执行 t 检验，对指定轴进行计算
    res = stats.ttest_1samp(x, popmean=popmean, axis=-2)
    # 断言统计量的形状是否正确
    assert res.statistic.shape == (5, 20)

    # 创建数组命名空间对象
    xp_test = array_namespace(x)  # torch needs expand_dims
    # 获取置信区间的下限和上限
    l, u = res.confidence_interval()
    # 在指定轴上扩展数组的维度
    l = xp_test.expand_dims(l, axis=-2)
    u = xp_test.expand_dims(u, axis=-2)

    # 使用下限进行 t 检验
    res = stats.ttest_1samp(x, popmean=l, axis=-2)
    # 创建参考的 p 值数组
    ref = xp.broadcast_to(xp.asarray(0.05, dtype=xp.float64), res.pvalue.shape)
    # 断言得到的 p 值与参考值的接近程度
    xp_assert_close(res.pvalue, ref)

    # 使用上限进行 t 检验
    res = stats.ttest_1samp(x, popmean=u, axis=-2)
    # 断言得到的 p 值与参考值的接近程度
    xp_assert_close(res.pvalue, ref)


class TestDescribe:
    @array_api_compatible
    def test_describe_scalar(self, xp):
        # 忽略警告，并设置特定的错误状态
        with suppress_warnings() as sup, \
              np.errstate(invalid="ignore", divide="ignore"):
            # 过滤特定的运行时警告信息
            sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
            # 描述标量值的统计特性
            n, mm, m, v, sk, kurt = stats.describe(xp.asarray(4.)[()])
        # 断言标量数量为1
        assert n == 1
        # 断言均值的一致性
        xp_assert_equal(mm[0], xp.asarray(4.0))
        xp_assert_equal(mm[1], xp.asarray(4.0))
        # 断言均值的一致性
        xp_assert_equal(m, xp.asarray(4.0))
        # 断言方差为 NaN
        xp_assert_equal(v ,xp.asarray(xp.nan))
        # 断言偏度为 NaN
        xp_assert_equal(sk, xp.asarray(xp.nan))
        # 断言峰度为 NaN
        xp_assert_equal(kurt, xp.asarray(xp.nan))
    # 定义一个测试函数，用于描述数字的统计特征
    def test_describe_numbers(self, xp):
        # 使用 array_namespace 函数将数组转换为特定命名空间的对象，并包装为数组
        xp_test = array_namespace(xp.asarray(1.))  # numpy 需要 `concat`
        # 创建一个新数组，连接两个由全 1 和全 2 组成的数组
        x = xp_test.concat((xp.ones((3, 4)), xp.full((2, 4), 2.)))
        # 设置数组的元素数量
        nc = 5
        # 定义包含多个数组的元组
        mmc = (xp.asarray([1., 1., 1., 1.]), xp.asarray([2., 2., 2., 2.]))
        # 创建一个一维数组
        mc = xp.asarray([1.4, 1.4, 1.4, 1.4])
        # 创建一个一维数组
        vc = xp.asarray([0.3, 0.3, 0.3, 0.3])
        # 创建一个一维数组
        skc = xp.asarray([0.40824829046386357] * 4)
        # 创建一个一维数组
        kurtc = xp.asarray([-1.833333333333333] * 4)
        # 对数组 x 进行描述统计，并返回计算结果的元组
        n, mm, m, v, sk, kurt = stats.describe(x)
        # 断言 n 的值等于 nc
        assert n == nc
        # 断言 mm[0] 等于 mmc[0]
        xp_assert_equal(mm[0], mmc[0])
        # 断言 mm[1] 等于 mmc[1]
        xp_assert_equal(mm[1], mmc[1])
        # 断言 m 与 mc 的值在指定的相对误差下接近
        xp_assert_close(m, mc, rtol=4 * xp.finfo(m.dtype).eps)
        # 断言 v 与 vc 的值在指定的相对误差下接近
        xp_assert_close(v, vc, rtol=4 * xp.finfo(m.dtype).eps)
        # 断言 sk 与 skc 的值接近
        xp_assert_close(sk, skc)
        # 断言 kurt 与 kurtc 的值接近

        # 对数组 x 的转置进行描述统计，并返回计算结果的元组
        n, mm, m, v, sk, kurt = stats.describe(x.T, axis=1)
        # 断言 n 的值等于 nc
        assert n == nc
        # 断言 mm[0] 等于 mmc[0]
        xp_assert_equal(mm[0], mmc[0])
        # 断言 mm[1] 等于 mmc[1]
        xp_assert_equal(mm[1], mmc[1])
        # 断言 m 与 mc 的值在指定的相对误差下接近
        xp_assert_close(m, mc, rtol=4 * xp.finfo(m.dtype).eps)
        # 断言 v 与 vc 的值在指定的相对误差下接近
        xp_assert_close(v, vc, rtol=4 * xp.finfo(m.dtype).eps)
        # 断言 sk 与 skc 的值接近
        xp_assert_close(sk, skc)
        # 断言 kurt 与 kurtc 的值接近

    # 定义测试 NaN 策略为忽略的函数
    def describe_nan_policy_omit_test(self):
        # 创建一个包含 NaN 值的一维数组
        x = np.arange(10.)
        x[9] = np.nan

        # 定义预期的 n 和 mmc
        nc, mmc = (9, (0.0, 8.0))
        # 定义预期的 m、v、sk 和 kurt
        mc = 4.0
        vc = 7.5
        skc = 0.0
        kurtc = -1.2300000000000002
        # 对数组 x 进行描述统计，使用忽略 NaN 值的策略，并返回计算结果的元组
        n, mm, m, v, sk, kurt = stats.describe(x, nan_policy='omit')
        # 断言 n 的值等于 nc
        assert_equal(n, nc)
        # 断言 mm 的值等于 mmc
        assert_equal(mm, mmc)
        # 断言 m 的值等于 mc
        assert_equal(m, mc)
        # 断言 v 的值等于 vc
        assert_equal(v, vc)
        # 断言 sk 的值与 skc 接近
        assert_array_almost_equal(sk, skc)
        # 断言 kurt 的值与 kurtc 接近，指定小数位数为 13
        assert_array_almost_equal(kurt, kurtc, decimal=13)

    # 根据数组 API 兼容性定义描述统计数字的测试函数
    @array_api_compatible
    def test_describe_nan_policy_other(self, xp):
        # 创建一个包含 NaN 值的一维数组，并将其中的一个值替换为 NaN
        x = xp.arange(10.)
        x = xp.where(x==9, xp.asarray(xp.nan), x)

        # 定义预期的错误信息
        message = 'The input contains nan values'
        # 使用 pytest 的 raises 函数，验证当数组中存在 NaN 值时会引发 ValueError
        with pytest.raises(ValueError, match=message):
            stats.describe(x, nan_policy='raise')

        # 对数组 x 进行描述统计，使用传播 NaN 值的策略，并返回计算结果的元组
        n, mm, m, v, sk, kurt = stats.describe(x, nan_policy='propagate')
        # 创建一个参考值 ref，为 NaN
        ref = xp.asarray(xp.nan)[()]
        # 断言 n 的值等于 10
        assert n == 10
        # 断言 mm[0] 的值等于 ref
        xp_assert_equal(mm[0], ref)
        # 断言 mm[1] 的值等于 ref
        xp_assert_equal(mm[1], ref)
        # 断言 m 的值等于 ref
        xp_assert_equal(m, ref)
        # 断言 v 的值等于 ref
        xp_assert_equal(v, ref)
        # 断言 sk 的值等于 ref
        xp_assert_equal(sk, ref)
        # 断言 kurt 的值等于 ref
        xp_assert_equal(kurt, ref)

        # 如果当前数组库是 NumPy，则执行 describe_nan_policy_omit_test 函数
        if is_numpy(xp):
            self.describe_nan_policy_omit_test()
        else:
            # 如果不是 NumPy 数组库，则验证 'omit' 策略与非 NumPy 数组不兼容时会引发 ValueError
            message = "`nan_policy='omit' is incompatible with non-NumPy arrays."
            with pytest.raises(ValueError, match=message):
                stats.describe(x, nan_policy='omit')

        # 定义预期的错误信息
        message = 'nan_policy must be one of...'
        # 验证当传入的 nan_policy 不在预期的策略列表中时会引发 ValueError
        with pytest.raises(ValueError, match=message):
            stats.describe(x, nan_policy='foobar')
    # 定义一个测试函数，用于测试描述统计结果的属性
    def test_describe_result_attributes(self):
        # 一些结果属性是元组，不适合与 `xp_assert_close` 进行比较
        actual = stats.describe(np.arange(5.))
        # 预期的结果属性
        attributes = ('nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis')
        # 调用函数检查实际结果的命名属性
        check_named_results(actual, attributes)

    @array_api_compatible
    # 定义一个测试函数，测试描述统计结果中的自由度修正
    def test_describe_ddof(self, xp):
        xp_test = array_namespace(xp.asarray(1.))  # numpy 需要 `concat`
        # 创建测试数组 x
        x = xp_test.concat((xp.ones((3, 4)), xp.full((2, 4), 2.)))
        # 预期的结果值
        nc = 5
        mmc = (xp.asarray([1., 1., 1., 1.]), xp.asarray([2., 2., 2., 2.]))
        mc = xp.asarray([1.4, 1.4, 1.4, 1.4])
        vc = xp.asarray([0.24, 0.24, 0.24, 0.24])
        skc = xp.asarray([0.40824829046386357] * 4)
        kurtc = xp.asarray([-1.833333333333333] * 4)
        # 调用描述统计函数，并返回相应的结果
        n, mm, m, v, sk, kurt = stats.describe(x, ddof=0)
        # 断言各个结果与预期值相等
        assert n == nc
        xp_assert_equal(mm[0], mmc[0])
        xp_assert_equal(mm[1], mmc[1])
        xp_assert_close(m, mc)
        xp_assert_close(v, vc)
        xp_assert_close(sk, skc)
        xp_assert_close(kurt, kurtc)

    @array_api_compatible
    # 定义一个测试函数，测试描述统计结果中的轴为 None 的情况
    def test_describe_axis_none(self, xp):
        xp_test = array_namespace(xp.asarray(1.))  # numpy 需要 `concat`
        # 创建测试数组 x
        x = xp_test.concat((xp.ones((3, 4)), xp.full((2, 4), 2.)))
        
        # 预期的结果值
        nc = 20
        mmc = (xp.asarray(1.0), xp.asarray(2.0))
        mc = xp.asarray(1.3999999999999999)
        vc = xp.asarray(0.25263157894736848)
        skc = xp.asarray(0.4082482904638634)
        kurtc = xp.asarray(-1.8333333333333333)
        
        # 调用描述统计函数，并返回相应的结果
        n, mm, m, v, sk, kurt = stats.describe(x, axis=None)
        
        # 断言各个结果与预期值相等
        assert n == nc
        xp_assert_equal(mm[0], mmc[0])
        xp_assert_equal(mm[1], mmc[1])
        xp_assert_close(m, mc)
        xp_assert_close(v, vc)
        xp_assert_close(sk, skc)
        xp_assert_close(kurt, kurtc)

    @array_api_compatible
    # 定义一个测试函数，测试空输入的情况
    def test_describe_empty(self, xp):
        # 预期的错误信息
        message = "The input must not be empty."
        # 使用 pytest 断言捕获 ValueError 异常，并匹配预期错误信息
        with pytest.raises(ValueError, match=message):
            stats.describe(xp.asarray([]))
@array_api_compatible
class NormalityTests:
    # 定义一个用于正态性测试的类，并标记为与数组 API 兼容

    def test_too_small(self, xp):
        # 检测样本过小的情况（一维样本观测值过少）-> 警告/错误
        test_fun = getattr(stats, self.test_name)
        # 获取与测试名称对应的测试函数
        x = xp.asarray(4.)
        # 将标量值 4 转换为数组，使用 xp.asarray() 可确保与测试库的统一接口

        if is_numpy(xp):
            # 如果使用的是 NumPy 库
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                # 用 pytest 检测小样本警告，匹配特定的警告信息
                res = test_fun(x)
                # 进行测试函数的调用
                NaN = xp.asarray(xp.nan)
                # 创建一个 NaN 数组
                xp_assert_equal(res.statistic, NaN)
                xp_assert_equal(res.pvalue, NaN)
                # 断言测试结果的统计量和 p 值为 NaN
        else:
            # 如果不是使用 NumPy 库
            message = "...requires at least..."
            # 错误消息内容
            with pytest.raises(ValueError, match=message):
                # 使用 pytest 检测值错误，匹配特定的错误消息
                test_fun(x)
                # 执行测试函数

    @pytest.mark.parametrize("alternative", ['two-sided', 'less', 'greater'])
    def test_against_R(self, alternative, xp):
        # 与 R 中的 `dagoTest` 测试对比
        # library(fBasics)
        # options(digits=16)
        # x = c(-2, -1, 0, 1, 2, 3)**2
        # x = rep(x, times=4)
        # test_result <- dagoTest(x)
        # test_result@test$statistic
        # test_result@test$p.value
        test_name = self.test_name
        # 获取测试名称
        test_fun = getattr(stats, test_name)
        # 获取与测试名称对应的测试函数
        ref_statistic, ref_pvalue = xp.asarray(self.case_ref)
        # 将参考统计量和 p 值转换为数组

        kwargs = {}
        if alternative in {'less', 'greater'}:
            # 如果选择了 'less' 或 'greater' 作为替代方案
            if test_name in {'skewtest', 'kurtosistest'}:
                # 如果测试名称为 'skewtest' 或 'kurtosistest'
                ref_pvalue = ref_pvalue/2 if alternative == "less" else 1-ref_pvalue/2
                # 根据选择的替代方案更新参考 p 值
                ref_pvalue = 1-ref_pvalue if test_name == 'skewtest' else ref_pvalue
                # 如果是 'skewtest' 则取 1 减去更新后的 p 值，否则保持不变
                kwargs['alternative'] = alternative
                # 更新关键字参数 'alternative'
            else:
                pytest.skip('`alternative` not available for `normaltest`')
                # 如果替代方案不适用于 'normaltest'，则跳过测试

        x = xp.asarray((-2, -1, 0, 1, 2, 3.)*4)**2
        # 生成测试数据 x，并将其转换为数组
        res = test_fun(x, **kwargs)
        # 执行测试函数，并传入额外的关键字参数
        res_statistic, res_pvalue = res
        # 分解测试结果为统计量和 p 值
        xp_assert_close(res_statistic, ref_statistic)
        # 断言测试结果的统计量接近于参考统计量
        xp_assert_close(res_pvalue, ref_pvalue)
        # 断言测试结果的 p 值接近于参考 p 值
        check_named_results(res, ('statistic', 'pvalue'), xp=xp)
        # 检查测试结果的命名结果是否符合预期，使用 xp 对象进行断言

    def test_nan(self, xp):
        # 输入中包含 NaN -> 输出也包含 NaN（默认的 NaN 策略是 'propagate'）
        test_fun = getattr(stats, self.test_name)
        # 获取与测试名称对应的测试函数
        x = xp.arange(30.)
        # 生成一个从 0 到 29 的数组
        NaN = xp.asarray(xp.nan, dtype=x.dtype)
        # 创建一个与 x 相同数据类型的 NaN 数组
        x = xp.where(x == 29, NaN, x)
        # 将 x 中值为 29 的元素替换为 NaN
        with np.errstate(invalid="ignore"):
            # 忽略无效值错误
            res = test_fun(x)
            # 执行测试函数
            xp_assert_equal(res.statistic, NaN)
            xp_assert_equal(res.pvalue, NaN)

class TestSkewTest(NormalityTests):
    # 继承自 NormalityTests 类的 SkewTest 测试类
    test_name = 'skewtest'
    # 测试名称为 'skewtest'
    case_ref = (1.98078826090875881, 0.04761502382843208)  # statistic, pvalue
    # 参考的统计量和 p 值

    def test_intuitive(self, xp):
        # 直观测试; 参见 gh-13549. skewnorm 参数为 1 时具有正偏态
        a1 = stats.skewnorm.rvs(a=1, size=10000, random_state=123)
        # 生成 skewnorm 分布的随机变量
        a1_xp = xp.asarray(a1)
        # 将随机变量转换为数组
        pval = stats.skewtest(a1_xp, alternative='greater').pvalue
        # 使用 'greater' 替代方案进行 skewtest 测试，获取 p 值
        xp_assert_close(pval, xp.asarray(0.0, dtype=a1_xp.dtype), atol=9e-6)
        # 断言 p 值接近于预期的 0.0，使用给定的容差值
    # 定义测试函数，用于检查 skewtest 在观测数据过少时的行为
    def test_skewtest_too_few_observations(self, xp):
        # 回归测试，针对问题票号 #1492。
        # skewtest 至少需要 8 个观测数据；7 个应该会引发 ValueError。
        # 使用 xp.arange(8.0) 生成一组数据，传递给 skewtest 进行检验
        stats.skewtest(xp.arange(8.0))

        # 创建一个长度为 7 的数据序列 xp.arange(7.0)
        x = xp.arange(7.0)
        # 如果 xp 是 numpy 库的话，预期会发出 SmallSampleWarning 警告，匹配 too_small_1d_not_omit 字符串
        if is_numpy(xp):
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                # 对 x 使用 skewtest 进行检验
                res = stats.skewtest(x)
                # 创建 NaN 数组 xp.asarray(xp.nan)
                NaN = xp.asarray(xp.nan)
                # 断言结果的 statistic 和 pvalue 都应该是 NaN
                xp_assert_equal(res.statistic, NaN)
                xp_assert_equal(res.pvalue, NaN)
        else:
            # 如果不是 numpy 库，预期会抛出 ValueError 异常，异常信息是 "`skewtest` requires at least 8 observations"
            message = "`skewtest` requires at least 8 observations"
            with pytest.raises(ValueError, match=message):
                # 对 x 使用 skewtest 进行检验，应该会引发 ValueError
                stats.skewtest(x)
class TestKurtosisTest(NormalityTests):
    # 定义测试类 TestKurtosisTest，继承自 NormalityTests
    test_name = 'kurtosistest'
    case_ref = (-0.01403734404759738, 0.98880018772590561)  # statistic, pvalue

    def test_intuitive(self, xp):
        # 直观测试；参见 gh-13549。拉普拉斯分布的过量峰度为3大于0
        a2 = stats.laplace.rvs(size=10000, random_state=123)
        a2_xp = xp.asarray(a2)
        # 使用 `stats.kurtosistest` 对 a2_xp 进行峰度检验，检验方向为“大于”
        pval = stats.kurtosistest(a2_xp, alternative='greater').pvalue
        # 断言 pval 接近于 0，使用 a2_xp 的数据类型，允许误差为 1e-15
        xp_assert_close(pval, xp.asarray(0.0, dtype=a2_xp.dtype), atol=1e-15)

    def test_gh9033_regression(self, xp):
        # issue gh-9033 的回归测试：x 明显非正态，但负面分母的幂需要正确处理以拒绝正态性
        counts = [128, 0, 58, 7, 0, 41, 16, 0, 0, 167]
        # 创建具有指定计数的数字列表，用于构造 x
        x = np.hstack([np.full(c, i) for i, c in enumerate(counts)])
        x = xp.asarray(x, dtype=xp.float64)
        # 断言 stats.kurtosistest(x) 的 p 值小于 0.01
        assert stats.kurtosistest(x)[1] < 0.01

    def test_kurtosistest_too_few_observations(self, xp):
        # kurtosistest 需要至少 5 个观测值；4 个应该引发 ValueError。
        # 至少需要 20 个以避免警告
        # ticket #1425 的回归测试。
        # 执行 stats.kurtosistest(xp.arange(20.0))，验证是否会引发警告
        stats.kurtosistest(xp.arange(20.0))

        # 准备捕获警告消息
        message = "`kurtosistest` p-value may be inaccurate..."
        with pytest.warns(UserWarning, match=message):
            # 执行 stats.kurtosistest(xp.arange(5.0))，验证是否会引发 UserWarning
            stats.kurtosistest(xp.arange(5.0))
        with pytest.warns(UserWarning, match=message):
            # 执行 stats.kurtosistest(xp.arange(19.0))，验证是否会引发 UserWarning
            stats.kurtosistest(xp.arange(19.0))

        x = xp.arange(4.0)
        if is_numpy(xp):
            # 在 NumPy 环境中，执行 stats.skewtest(x)，验证是否会引发 SmallSampleWarning
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                res = stats.skewtest(x)
                NaN = xp.asarray(xp.nan)
                xp_assert_equal(res.statistic, NaN)
                xp_assert_equal(res.pvalue, NaN)
        else:
            # 在非 NumPy 环境中，执行 stats.kurtosistest(x)，验证是否会引发 ValueError
            message = "`kurtosistest` requires at least 5 observations"
            with pytest.raises(ValueError, match=message):
                stats.kurtosistest(x)


class TestNormalTest(NormalityTests):
    # 定义测试类 TestNormalTest，继承自 NormalityTests
    test_name = 'normaltest'
    case_ref = (3.92371918158185551, 0.14059672529747502)  # statistic, pvalue


class TestRankSums:
    # 定义测试类 TestRankSums

    np.random.seed(0)
    x, y = np.random.rand(2, 10)

    @pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
    def test_ranksums_result_attributes(self, alternative):
        # ranksums pval = mannwhitneyu pval w/out continuity or tie correction
        # 执行 stats.ranksums(self.x, self.y, alternative=alternative).pvalue，获取结果
        res1 = stats.ranksums(self.x, self.y,
                              alternative=alternative).pvalue
        # 执行 stats.mannwhitneyu(self.x, self.y, use_continuity=False, alternative=alternative).pvalue，获取结果
        res2 = stats.mannwhitneyu(self.x, self.y, use_continuity=False,
                                  alternative=alternative).pvalue
        # 断言 res1 与 res2 接近
        assert_allclose(res1, res2)

    def test_ranksums_named_results(self):
        # 执行 stats.ranksums(self.x, self.y)，检查结果的命名属性
        res = stats.ranksums(self.x, self.y)
        check_named_results(res, ('statistic', 'pvalue'))
    # 定义一个测试方法，用于验证输入是否有效
    def test_input_validation(self):
        # 使用 assert_raises 断言，期望捕获 ValueError 异常，并匹配错误信息 "alternative must be 'less'"
        with assert_raises(ValueError, match="`alternative` must be 'less'"):
            # 调用 stats.ranksums 方法，传入 self.x 和 self.y 作为参数，并指定 alternative='foobar'
            # 该方法用于比较两组数据的秩和检验，但此处 alternative 参数值错误，期望触发 ValueError 异常
            stats.ranksums(self.x, self.y, alternative='foobar')
# 使用装饰器标记该类兼容数组 API
@array_api_compatible
# 定义 Jarque-Bera 测试类
class TestJarqueBera:

    # 测试 Jarque-Bera 统计量与 R 的比较
    def test_jarque_bera_against_R(self, xp):
        # 定义测试数据 x
        x = [-0.160104223201523288,  1.131262000934478040, -0.001235254523709458,
             -0.776440091309490987, -2.072959999533182884]
        # 将 x 转换为适合数组 API 的类型
        x = xp.asarray(x)
        # 参考值 ref
        ref = xp.asarray([0.17651605223752, 0.9155246169805])
        # 计算 Jarque-Bera 测试的结果 res
        res = stats.jarque_bera(x)
        # 断言 Jarque-Bera 统计量的近似性
        xp_assert_close(res.statistic, ref[0])
        # 断言 p 值的近似性
        xp_assert_close(res.pvalue, ref[1])

    # 标记为仅适用于 NumPy，跳过其他数组后端
    @skip_xp_backends(np_only=True)
    # 使用 pytest fixture 标记测试，跳过其他数组后端
    @pytest.mark.usefixtures("skip_xp_backends")
    # 测试 array-like 数据的 Jarque-Bera 检验
    def test_jarque_bera_array_like(self):
        # 生成随机数据 x
        np.random.seed(987654321)
        x = np.random.normal(0, 1, 100000)

        # 执行 Jarque-Bera 检验，并获取结果 JB1, p1
        jb_test1 = JB1, p1 = stats.jarque_bera(list(x))
        # 执行 Jarque-Bera 检验，并获取结果 JB2, p2
        jb_test2 = JB2, p2 = stats.jarque_bera(tuple(x))
        # 执行 Jarque-Bera 检验，并获取结果 JB3, p3
        jb_test3 = JB3, p3 = stats.jarque_bera(x.reshape(2, 50000))

        # 断言统计量 JB1, JB2, JB3 的一致性，以及与测试结果的一致性
        assert JB1 == JB2 == JB3 == jb_test1.statistic == jb_test2.statistic == jb_test3.statistic  # noqa: E501
        # 断言 p 值 p1, p2, p3 的一致性，以及与测试结果的一致性
        assert p1 == p2 == p3 == jb_test1.pvalue == jb_test2.pvalue == jb_test3.pvalue

    # 测试 Jarque-Bera 在空数组或单元素数组时的行为
    def test_jarque_bera_size(self, xp):
        # 创建空数组 x
        x = xp.asarray([])
        # 如果是 NumPy 数组，则测试空数组情况下的警告
        if is_numpy(xp):
            with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
                res = stats.jarque_bera(x)
            # 定义 NaN 值
            NaN = xp.asarray(xp.nan)
            # 断言统计量的结果为 NaN
            xp_assert_equal(res.statistic, NaN)
            # 断言 p 值的结果为 NaN
            xp_assert_equal(res.pvalue, NaN)
        else:
            # 如果不是 NumPy 数组，测试至少需要一个观测值的情况
            message = "At least one observation is required."
            with pytest.raises(ValueError, match=message):
                res = stats.jarque_bera(x)

    # 测试 Jarque-Bera 在指定轴上的计算结果
    def test_axis(self, xp):
        # 使用随机数生成器创建随机数据 x
        rng = np.random.RandomState(seed=122398129)
        x = xp.asarray(rng.random(size=(2, 45)))

        # 测试在 axis=None 时的 Jarque-Bera 计算结果
        res = stats.jarque_bera(x, axis=None)
        ref = stats.jarque_bera(xp.reshape(x, (-1,)))
        # 断言统计量的一致性
        xp_assert_equal(res.statistic, ref.statistic)
        # 断言 p 值的一致性
        xp_assert_equal(res.pvalue, ref.pvalue)

        # 测试在 axis=1 时的 Jarque-Bera 计算结果
        res = stats.jarque_bera(x, axis=1)
        # 计算每行的统计量 s0, s1 和 p 值 p0, p1
        s0, p0 = stats.jarque_bera(x[0, :])
        s1, p1 = stats.jarque_bera(x[1, :])
        # 断言统计量的近似性
        xp_assert_close(res.statistic, xp.asarray([s0, s1]))
        # 断言 p 值的近似性
        xp_assert_close(res.pvalue, xp.asarray([p0, p1]))

        # 测试在 axis=0 时的 Jarque-Bera 计算结果（通过转置来实现）
        resT = stats.jarque_bera(x.T, axis=0)
        # 断言统计量的近似性
        xp_assert_close(res.statistic, resT.statistic)
        # 断言 p 值的近似性
        xp_assert_close(res.pvalue, resT.pvalue)
    # X 列表，包含一系列数值
    X = [19.8958398126694, 19.5452691647182, 19.0577309166425, 21.716543054589,
         20.3269502208702, 20.0009273294025, 19.3440043632957, 20.4216806548105,
         19.0649894736528, 18.7808043120398, 19.3680942943298, 19.4848044069953,
         20.7514611265663, 19.0894948874598, 19.4975522356628, 18.9971170734274,
         20.3239606288208, 20.6921298083835, 19.0724259532507, 18.9825187935021,
         19.5144462609601, 19.8256857844223, 20.5174677102032, 21.1122407995892,
         17.9490854922535, 18.2847521114727, 20.1072217648826, 18.6439891962179,
         20.4970638083542, 19.5567594734914]

    # Y 列表，包含一系列数值
    Y = [19.2790668029091, 16.993808441865, 18.5416338448258, 17.2634018833575,
         19.1577183624616, 18.5119655377495, 18.6068455037221, 18.8358343362655,
         19.0366413269742, 18.1135025515417, 19.2201873866958, 17.8344909022841,
         18.2894380745856, 18.6661374133922, 19.9688601693252, 16.0672254617636,
         19.00596360572, 19.201561539032, 19.0487501090183, 19.0847908674356]

    # 显著性水平
    significant = 14

    # 使用 Mann-Whitney U 检验进行单尾检验
    def test_mannwhitneyu_one_sided(self):
        # 计算并返回两组样本的 U 值和单尾检验的 p 值（小于关系）
        u1, p1 = stats.mannwhitneyu(self.X, self.Y, alternative='less')
        # 计算并返回两组样本的 U 值和单尾检验的 p 值（大于关系）
        u2, p2 = stats.mannwhitneyu(self.Y, self.X, alternative='greater')
        # 再次计算并返回两组样本的 U 值和单尾检验的 p 值（大于关系，另一方向）
        u3, p3 = stats.mannwhitneyu(self.X, self.Y, alternative='greater')
        # 再次计算并返回两组样本的 U 值和单尾检验的 p 值（小于关系，另一方向）
        u4, p4 = stats.mannwhitneyu(self.Y, self.X, alternative='less')

        # 断言检验结果的一致性
        assert_equal(p1, p2)
        assert_equal(p3, p4)
        # 断言不同方向的检验结果 p 值不相等
        assert_(p1 != p3)
        # 断言 U 值的正确性
        assert_equal(u1, 498)
        assert_equal(u2, 102)
        assert_equal(u3, 498)
        assert_equal(u4, 102)
        # 断言 p 值的近似正确性，使用显著性水平作为参数
        assert_approx_equal(p1, 0.999957683256589, significant=self.significant)
        assert_approx_equal(p3, 4.5941632666275e-05, significant=self.significant)

    # 使用 Mann-Whitney U 检验进行双尾检验
    def test_mannwhitneyu_two_sided(self):
        # 计算并返回两组样本的 U 值和双尾检验的 p 值
        u1, p1 = stats.mannwhitneyu(self.X, self.Y, alternative='two-sided')
        # 计算并返回两组样本的 U 值和双尾检验的 p 值
        u2, p2 = stats.mannwhitneyu(self.Y, self.X, alternative='two-sided')

        # 断言双尾检验的 p 值相等
        assert_equal(p1, p2)
        # 断言 U 值的正确性
        assert_equal(u1, 498)
        assert_equal(u2, 102)
        # 断言 p 值的近似正确性，使用显著性水平作为参数
        assert_approx_equal(p1, 9.188326533255e-05,
                            significant=self.significant)
    # 执行 Mann-Whitney U 检验，单侧检验，比较 self.X 和 self.Y 的分布
    def test_mannwhitneyu_no_correct_one_sided(self):
        # 计算 Mann-Whitney U 统计量和单侧检验的 p 值，self.X < self.Y
        u1, p1 = stats.mannwhitneyu(self.X, self.Y, False,
                                    alternative='less')
        # 计算 Mann-Whitney U 统计量和单侧检验的 p 值，self.Y < self.X
        u2, p2 = stats.mannwhitneyu(self.Y, self.X, False,
                                    alternative='greater')
        # 计算 Mann-Whitney U 统计量和单侧检验的 p 值，self.X > self.Y
        u3, p3 = stats.mannwhitneyu(self.X, self.Y, False,
                                    alternative='greater')
        # 计算 Mann-Whitney U 统计量和单侧检验的 p 值，self.Y < self.X
        u4, p4 = stats.mannwhitneyu(self.Y, self.X, False,
                                    alternative='less')

        # 断言单侧检验时两个方向的 p 值相等
        assert_equal(p1, p2)
        # 断言两个不同方向的单侧检验 p 值不相等
        assert_equal(p3, p4)
        # 断言不同方向的单侧检验 p 值不相等
        assert_(p1 != p3)
        # 断言 Mann-Whitney U 统计量等于 498
        assert_equal(u1, 498)
        # 断言 Mann-Whitney U 统计量等于 102
        assert_equal(u2, 102)
        # 断言 Mann-Whitney U 统计量等于 498
        assert_equal(u3, 498)
        # 断言 Mann-Whitney U 统计量等于 102
        assert_equal(u4, 102)
        # 断言 p 值接近于 0.999955905990004，使用指定的显著性水平检验
        assert_approx_equal(p1, 0.999955905990004, significant=self.significant)
        # 断言 p 值接近于 4.40940099958089e-05，使用指定的显著性水平检验
        assert_approx_equal(p3, 4.40940099958089e-05, significant=self.significant)

    # 执行 Mann-Whitney U 检验，双侧检验，比较 self.X 和 self.Y 的分布
    def test_mannwhitneyu_no_correct_two_sided(self):
        # 计算 Mann-Whitney U 统计量和双侧检验的 p 值
        u1, p1 = stats.mannwhitneyu(self.X, self.Y, False,
                                    alternative='two-sided')
        # 计算 Mann-Whitney U 统计量和双侧检验的 p 值
        u2, p2 = stats.mannwhitneyu(self.Y, self.X, False,
                                    alternative='two-sided')

        # 断言双侧检验时两个方向的 p 值相等
        assert_equal(p1, p2)
        # 断言 Mann-Whitney U 统计量等于 498
        assert_equal(u1, 498)
        # 断言 Mann-Whitney U 统计量等于 102
        assert_equal(u2, 102)
        # 断言 p 值接近于 8.81880199916178e-05，使用指定的显著性水平检验
        assert_approx_equal(p1, 8.81880199916178e-05,
                            significant=self.significant)
    # 定义单元测试方法，测试 mannwhitneyu 函数对 gh-1428 的支持
    def test_mannwhitneyu_ones(self):
        # 创建包含大量相同值的 NumPy 数组 x 和 y
        x = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 2., 1., 1., 1., 1., 2., 1., 1., 2., 1., 1., 2.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 3., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1.])

        y = np.array([1., 1., 1., 1., 1., 1., 1., 2., 1., 2., 1., 1., 1., 1.,
                      2., 1., 1., 1., 2., 1., 1., 1., 1., 1., 2., 1., 1., 3.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1., 2., 1.,
                      1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 2.,
                      2., 1., 1., 2., 1., 1., 2., 1., 2., 1., 1., 1., 1., 2.,
                      2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 2., 1., 1., 1., 1., 1., 2., 2., 2., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      2., 1., 1., 2., 1., 1., 1., 1., 2., 1., 1., 1., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 2., 1., 1., 1., 2., 1., 1.,
                      1., 1., 1., 1.])

        # 使用 R 中的 wilcox.test 进行检验，断言 mannwhitneyu 函数的 less 方式的输出值
        assert_allclose(stats.mannwhitneyu(x, y, alternative='less'),
                        (16980.5, 2.8214327656317373e-005))
        # 使用 R 中的 wilcox.test 进行检验，断言 mannwhitneyu 函数的 greater 方式的输出值
        assert_allclose(stats.mannwhitneyu(x, y, alternative='greater'),
                        (16980.5, 0.9999719954296))
        # 使用 R 中的 wilcox.test 进行检验，断言 mannwhitneyu 函数的 two-sided 方式的输出值
        assert_allclose(stats.mannwhitneyu(x, y, alternative='two-sided'),
                        (16980.5, 5.642865531266e-05))
    def test_mannwhitneyu_result_attributes(self):
        # 定义一个测试函数，用于验证 Mann-Whitney U 检验返回的命名元组属性
        # 定义需要检查的命名元组属性
        attributes = ('statistic', 'pvalue')
        # 调用 Mann-Whitney U 检验函数，传入参数 self.X 和 self.Y，选择 alternative="less" 方式进行检验
        res = stats.mannwhitneyu(self.X, self.Y, alternative="less")
        # 调用自定义函数 check_named_results，检查返回的结果 res 是否包含指定的命名元组属性
        check_named_results(res, attributes)
def test_pointbiserial():
    # same as mstats test except for the nan
    # Test data: https://web.archive.org/web/20060504220742/https://support.sas.com/ctx/samples/index.jsp?sid=490&tab=output
    # 定义测试数据 x 和 y
    x = [1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,
         0,0,0,0,1]
    y = [14.8,13.8,12.4,10.1,7.1,6.1,5.8,4.6,4.3,3.5,3.3,3.2,3.0,
         2.8,2.8,2.5,2.4,2.3,2.1,1.7,1.7,1.5,1.3,1.3,1.2,1.2,1.1,
         0.8,0.7,0.6,0.5,0.2,0.2,0.1]
    # 断言检查 pointbiserialr 函数计算的相关系数是否接近预期值 0.36149
    assert_almost_equal(stats.pointbiserialr(x, y)[0], 0.36149, 5)

    # test for namedtuple attribute results
    # 检查返回结果是否符合命名元组的属性
    attributes = ('correlation', 'pvalue')
    res = stats.pointbiserialr(x, y)
    check_named_results(res, attributes)
    # 断言检查相关系数与 statistic 属性是否相等
    assert_equal(res.correlation, res.statistic)


def test_obrientransform():
    # A couple tests calculated by hand.
    # 手工计算的几个测试用例
    x1 = np.array([0, 2, 4])
    # 使用 obrientransform 函数变换数据 x1
    t1 = stats.obrientransform(x1)
    expected = [7, -2, 7]
    # 断言检查变换后的结果是否与预期值 expected 接近
    assert_allclose(t1[0], expected)

    x2 = np.array([0, 3, 6, 9])
    # 使用 obrientransform 函数变换数据 x2
    t2 = stats.obrientransform(x2)
    expected = np.array([30, 0, 0, 30])
    # 断言检查变换后的结果是否与预期值 expected 接近
    assert_allclose(t2[0], expected)

    # Test two arguments.
    # 测试两个参数的情况
    a, b = stats.obrientransform(x1, x2)
    # 断言检查 obrientransform 函数返回的两个结果是否分别与 t1 和 t2 的第一个元素相等
    assert_equal(a, t1[0])
    assert_equal(b, t2[0])

    # Test three arguments.
    # 测试三个参数的情况
    a, b, c = stats.obrientransform(x1, x2, x1)
    # 断言检查 obrientransform 函数返回的三个结果是否分别与 t1 和 t2 的第一个元素相等
    assert_equal(a, t1[0])
    assert_equal(b, t2[0])
    assert_equal(c, t1[0])

    # This is a regression test to check np.var replacement.
    # The author of this test didn't separately verify the numbers.
    # 这是一个回归测试，用于检查替代 np.var 的功能
    x1 = np.arange(5)
    result = np.array(
      [[5.41666667, 1.04166667, -0.41666667, 1.04166667, 5.41666667],
       [21.66666667, 4.16666667, -1.66666667, 4.16666667, 21.66666667]])
    # 断言检查 obrientransform 函数计算结果是否与预期值 result 接近
    assert_array_almost_equal(stats.obrientransform(x1, 2*x1), result, decimal=8)

    # Example from "O'Brien Test for Homogeneity of Variance"
    # by Herve Abdi.
    # 来自 Herve Abdi 的方差齐性检验的示例
    values = range(5, 11)
    reps = np.array([5, 11, 9, 3, 2, 2])
    data = np.repeat(values, reps)
    transformed_values = np.array([3.1828, 0.5591, 0.0344,
                                   1.6086, 5.2817, 11.0538])
    expected = np.repeat(transformed_values, reps)
    # 断言检查 obrientransform 函数计算结果是否与预期值 expected 接近
    assert_array_almost_equal(stats.obrientransform(data)[0], expected, decimal=4)


def check_equal_xmean(*args, xp, mean_fun, axis=None, dtype=None,
                      rtol=1e-7, weights=None):
    # Note this doesn't test when axis is not specified
    # 注意：这个函数没有测试 axis 未指定的情况
    dtype = dtype or xp.float64
    # 如果参数长度为 2，则将其分别赋值给 array_like 和 desired；否则分别赋值给 array_like, p, desired
    if len(args) == 2:
        array_like, desired = args
    else:
        array_like, p, desired = args
    # 将 array_like 和 desired 转换为指定的数据类型
    array_like = xp.asarray(array_like, dtype=dtype)
    desired = xp.asarray(desired, dtype=dtype)
    # 如果 weights 不为 None，则将其转换为指定的数据类型
    weights = xp.asarray(weights, dtype=dtype) if weights is not None else weights
    # 根据参数调用 mean_fun 函数计算均值，并传递指定的参数
    args = (array_like,) if len(args) == 2 else (array_like, p)
    x = mean_fun(*args, axis=axis, dtype=dtype, weights=weights)
    # 使用 xp_assert_close 函数断言检查计算的均值 x 是否与期望值 desired 接近
    xp_assert_close(x, desired, rtol=rtol)


def check_equal_gmean(*args, **kwargs):
    # 调用 check_equal_xmean 函数，传递 mean_fun=stats.gmean 的参数
    return check_equal_xmean(*args, mean_fun=stats.gmean, **kwargs)
# 使用 `check_equal_xmean` 函数来比较传入参数的调和平均数是否等于期望值，其中使用了 `stats.hmean` 函数
def check_equal_hmean(*args, **kwargs):
    return check_equal_xmean(*args, mean_fun=stats.hmean, **kwargs)

# 使用 `check_equal_xmean` 函数来比较传入参数的平均数是否等于期望值，其中使用了 `stats.pmean` 函数
def check_equal_pmean(*args, **kwargs):
    return check_equal_xmean(*args, mean_fun=stats.pmean, **kwargs)

# 用于测试调和平均数计算的类装饰器，用于兼容数组 API
@array_api_compatible
class TestHMean:
    # 测试在给定的数组 `a` 和期望值 `desired` 下，调用 `check_equal_hmean` 函数
    def test_0(self, xp):
        a = [1, 0, 2]
        desired = 0
        check_equal_hmean(a, desired, xp=xp)

    # 测试在给定的数组 `a` 和期望值 `desired` 下，调用 `check_equal_hmean` 函数
    def test_1d(self, xp):
        # 测试一维数组情况
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        desired = 34.1417152147
        check_equal_hmean(a, desired, xp=xp)

        a = [1, 2, 3, 4]
        desired = 4. / (1. / 1 + 1. / 2 + 1. / 3 + 1. / 4)
        check_equal_hmean(a, desired, xp=xp)

    # 测试在给定的数组 `a` 和期望值 `desired` 下，调用 `check_equal_hmean` 函数，包含零值情况
    def test_1d_with_zero(self, xp):
        a = np.array([1, 0])
        desired = 0.0
        check_equal_hmean(a, desired, xp=xp, rtol=0.0)

    # 使用 pytest 标记的测试函数，测试在给定的数组 `a` 和期望值 `xp.nan` 下，调用 `check_equal_hmean` 函数，并期望抛出 RuntimeWarning 异常
    @skip_xp_backends('array_api_strict',
                      reasons=["`array_api_strict.where` `fillvalue` doesn't "
                               "accept Python scalars. See data-apis/array-api#807."])
    @pytest.mark.usefixtures("skip_xp_backends")
    def test_1d_with_negative_value(self, xp):
        # 对于包含负值的数组 `a`，预期会抛出 RuntimeWarning 异常
        a = np.array([1, 0, -1])
        message = "The harmonic mean is only defined..."
        with pytest.warns(RuntimeWarning, match=message):
            check_equal_hmean(a, xp.nan, xp=xp, rtol=0.0)

    # 注意：下面的测试使用默认参数 `axis=None`，而不是 `axis=0`
    def test_2d(self, xp):
        # 测试二维数组情况
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = 38.6696271841
        check_equal_hmean(np.array(a), desired, xp=xp)

    # 测试二维数组情况，指定 `axis=0`
    def test_2d_axis0(self, xp):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([22.88135593, 39.13043478, 52.90076336, 65.45454545])
        check_equal_hmean(a, desired, axis=0, xp=xp)

    # 测试二维数组情况，指定 `axis=0`，包含零值情况
    def test_2d_axis0_with_zero(self, xp):
        a = [[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([22.88135593, 0.0, 52.90076336, 65.45454545])
        check_equal_hmean(a, desired, axis=0, xp=xp)

    # 测试二维数组情况，指定 `axis=1`
    def test_2d_axis1(self, xp):
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([19.2, 63.03939962, 103.80078637])
        check_equal_hmean(a, desired, axis=1, xp=xp)

    # 测试二维数组情况，指定 `axis=1`，包含零值情况
    def test_2d_axis1_with_zero(self, xp):
        a = [[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        desired = np.array([0.0, 63.03939962, 103.80078637])
        check_equal_hmean(a, desired, axis=1, xp=xp)

    # 使用 pytest 标记，跳过某些 XP 后端的测试
    @pytest.mark.skip_xp_backends(
        np_only=True,
        reasons=['array-likes only supported for NumPy backend'],
    )
    @pytest.mark.usefixtures("skip_xp_backends")
    # 定义一个测试函数，用于测试一维列表输入情况
    def test_weights_1d_list(self, xp):
        # 预期结果来自于指定链接的数学问题解答
        a = [2, 10, 6]
        weights = [10, 5, 3]
        desired = 3.
        # 所有其他测试使用 `check_equal_hmean`，此处检查函数仍接受整数列表输入
        res = stats.hmean(a, weights=weights)
        xp_assert_close(res, np.asarray(desired), rtol=1e-5)

    # 定义一个测试函数，用于测试一维数组输入情况
    def test_weights_1d(self, xp):
        # 预期结果来自于指定链接的数学问题解答
        a = np.asarray([2, 10, 6])
        weights = np.asarray([10, 5, 3])
        desired = 3
        # 调用 `check_equal_hmean` 检查均值相等性，传入一维数组和权重
        check_equal_hmean(a, desired, weights=weights, rtol=1e-5, xp=xp)

    # 定义一个测试函数，用于测试二维数组在轴0上的输入情况
    def test_weights_2d_axis0(self, xp):
        # 预期结果来自于指定链接的数学问题解答
        a = np.array([[2, 5], [10, 5], [6, 5]])
        weights = np.array([[10, 1], [5, 1], [3, 1]])
        desired = np.array([3, 5])
        # 调用 `check_equal_hmean` 检查均值相等性，传入二维数组和权重，并指定轴为0
        check_equal_hmean(a, desired, axis=0, weights=weights, rtol=1e-5, xp=xp)

    # 定义一个测试函数，用于测试二维数组在轴1上的输入情况
    def test_weights_2d_axis1(self, xp):
        # 预期结果来自于指定链接的数学问题解答
        a = np.array([[2, 10, 6], [7, 7, 7]])
        weights = np.array([[10, 5, 3], [1, 1, 1]])
        desired = np.array([3, 7])
        # 调用 `check_equal_hmean` 检查均值相等性，传入二维数组和权重，并指定轴为1
        check_equal_hmean(a, desired, axis=1, weights=weights, rtol=1e-5, xp=xp)

    @skip_xp_invalid_arg
    # 定义一个带有装饰器的测试函数，用于测试一维数组和掩码的输入情况
    def test_weights_masked_1d_array(self, xp):
        # 预期结果来自于指定链接的数学问题解答
        a = np.array([2, 10, 6, 42])
        weights = np.ma.array([10, 5, 3, 42], mask=[0, 0, 0, 1])
        desired = 3
        # 设置 xp 为 np.ma，以保留掩码，调用 `check_equal_hmean` 检查均值相等性，传入掩码数组和权重
        xp = np.ma
        check_equal_hmean(a, desired, weights=weights, rtol=1e-5,
                          dtype=np.float64, xp=xp)
# 使用装饰器标记这个类兼容多个数组API
@array_api_compatible
# 定义一个测试类 TestGMean
class TestGMean:
    # 定义一个测试方法 test_0，接受一个参数 xp
    def test_0(self, xp):
        # 定义一个列表 a
        a = [1, 0, 2]
        # 期望的结果是 0
        desired = 0
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired
        check_equal_gmean(a, desired, xp=xp)

    # 定义一个测试方法 test_1d，接受一个参数 xp
    def test_1d(self, xp):
        # 测试一个一维情况
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # 期望的结果是约等于 45.2872868812
        desired = 45.2872868812
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired
        check_equal_gmean(a, desired, xp=xp)

        # 定义另一个列表 a
        a = [1, 2, 3, 4]
        # 根据公式计算期望的结果
        desired = power(1 * 2 * 3 * 4, 1. / 4.)
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired，允许相对误差为 1e-14
        check_equal_gmean(a, desired, rtol=1e-14, xp=xp)

        # 使用 array 函数创建一个浮点数类型的数组 a
        a = array([1, 2, 3, 4], float32)
        # 根据公式计算期望的结果
        desired = power(1 * 2 * 3 * 4, 1. / 4.)
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired，指定数据类型为 xp.float32
        check_equal_gmean(a, desired, dtype=xp.float32, xp=xp)

    # 注意：下面的测试默认使用 axis=None，而不是 axis=0
    # 定义一个测试方法 test_2d，接受一个参数 xp
    def test_2d(self, xp):
        # 测试一个二维情况
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        # 期望的结果是约等于 52.8885199
        desired = 52.8885199
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired
        check_equal_gmean(a, desired, xp=xp)

    # 定义一个测试方法 test_2d_axis0，接受一个参数 xp
    def test_2d_axis0(self, xp):
        # 测试一个二维情况，指定 axis=0
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        # 期望的结果是一个 NumPy 数组
        desired = np.array([35.56893304, 49.32424149, 61.3579244, 72.68482371])
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired，计算每列的平均值
        check_equal_gmean(a, desired, axis=0, xp=xp)

        # 使用 array 函数创建一个数组 a
        a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        # 期望的结果是一个数组，每列的值为 1, 2, 3, 4
        desired = array([1, 2, 3, 4])
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired，计算每列的平均值，允许相对误差为 1e-14
        check_equal_gmean(a, desired, axis=0, rtol=1e-14, xp=xp)

    # 定义一个测试方法 test_2d_axis1，接受一个参数 xp
    def test_2d_axis1(self, xp):
        # 测试一个二维情况，指定 axis=1
        a = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        # 期望的结果是一个 NumPy 数组
        desired = np.array([22.13363839, 64.02171746, 104.40086817])
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired，计算每行的平均值
        check_equal_gmean(a, desired, axis=1, xp=xp)

        # 使用 array 函数创建一个数组 a
        a = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        # 根据公式计算期望的结果
        v = power(1 * 2 * 3 * 4, 1. / 4.)
        # 期望的结果是一个数组，每行的值均为 v
        desired = array([v, v, v])
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired，计算每行的平均值，允许相对误差为 1e-14
        check_equal_gmean(a, desired, axis=1, rtol=1e-14, xp=xp)

    # 定义一个测试方法 test_large_values，接受一个参数 xp
    def test_large_values(self, xp):
        # 定义一个数组 a，包含极大的数值
        a = array([1e100, 1e200, 1e300])
        # 期望的结果是 1e200
        desired = 1e200
        # 调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired，允许相对误差为 1e-13
        check_equal_gmean(a, desired, rtol=1e-13, xp=xp)

    # 定义一个测试方法 test_1d_with_0，接受一个参数 xp
    def test_1d_with_0(self, xp):
        # 测试一个包含零元素的一维情况
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 0]
        # 期望的结果是 0.0，因为 exp(-inf)=0
        desired = 0.0
        # 使用 np.errstate(ignore) 忽略错误，调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired
        with np.errstate(all='ignore'):
            check_equal_gmean(a, desired, xp=xp)

    # 定义一个测试方法 test_1d_neg，接受一个参数 xp
    def test_1d_neg(self, xp):
        # 测试一个包含负数元素的一维情况
        a = [10, 20, 30, 40, 50, 60, 70, 80, 90, -1]
        # 期望的结果是 NaN，因为 log(-1) = nan
        # 使用 np.errstate(ignore) 忽略无效操作错误，调用 check_equal_gmean 函数，检查 a 的几何平均值是否等于 desired
        with np.errstate(invalid='ignore'):
            check_equal_gmean(a, desired=np.nan, xp=xp)

    # 使用 pytest.mark.skip_xp_backends 装饰器标记这个测试，指定仅在 NumPy 后端运行，并提供跳过原因
    @pytest.mark
    # 定义一个测试方法，用于测试处理一维列表的加权几何平均值计算
    def test_weights_1d_list(self, xp):
        # 样例来源：
        # https://www.dummies.com/education/math/business-statistics/how-to-find-the-weighted-geometric-mean-of-a-data-set/
        
        # 定义输入数据列表a和对应的权重列表weights
        a = [1, 2, 3, 4, 5]
        weights = [2, 5, 6, 4, 3]
        # 预期的加权几何平均值
        desired = 2.77748

        # 使用统计模块中的gmean函数计算加权几何平均值，验证结果
        res = stats.gmean(a, weights=weights)
        xp_assert_close(res, np.asarray(desired), rtol=1e-5)

    # 定义另一个测试方法，用于测试处理一维NumPy数组的加权几何平均值计算
    def test_weights_1d(self, xp):
        # 样例来源：
        # https://www.dummies.com/education/math/business-statistics/how-to-find-the-weighted-geometric-mean-of-a-data-set/
        
        # 定义输入数据NumPy数组a和对应的权重NumPy数组weights
        a = np.array([1, 2, 3, 4, 5])
        weights = np.array([2, 5, 6, 4, 3])
        # 预期的加权几何平均值
        desired = 2.77748
        
        # 调用check_equal_gmean函数，验证加权几何平均值计算结果
        check_equal_gmean(a, desired, weights=weights, rtol=1e-5, xp=xp)

    # 标记为不支持某些参数的测试方法，用于测试处理带掩码的一维NumPy数组的加权几何平均值计算
    @skip_xp_invalid_arg
    def test_weights_masked_1d_array(self, xp):
        # 样例来源：
        # https://www.dummies.com/education/math/business-statistics/how-to-find-the-weighted-geometric-mean-of-a-data-set/
        
        # 定义输入数据NumPy数组a和带掩码的权重数组weights
        a = np.array([1, 2, 3, 4, 5, 6])
        weights = np.ma.array([2, 5, 6, 4, 3, 5], mask=[0, 0, 0, 0, 0, 1])
        # 预期的加权几何平均值
        desired = 2.77748
        
        # 设定xp为np.ma，以保留掩码，调用check_equal_gmean函数，验证加权几何平均值计算结果
        xp = np.ma
        check_equal_gmean(a, desired, weights=weights, rtol=1e-5,
                          dtype=np.float64, xp=xp)
# 装饰器，用于确保类的方法与数组 API 兼容
@array_api_compatible
# 定义 TestPMean 类
class TestPMean:

    # 计算给定数组 a 和指数 p 的普通均值的参考实现
    def pmean_reference(a, p):
        return (np.sum(a**p) / a.size)**(1/p)

    # 计算给定数组 a、指数 p 和权重 weights 的加权均值的参考实现
    def wpmean_reference(a, p, weights):
        return (np.sum(weights * a**p) / np.sum(weights))**(1/p)

    # 测试当指数 p 为非法值时是否引发 ValueError 异常
    def test_bad_exponent(self, xp):
        with pytest.raises(ValueError, match='Power mean only defined for'):
            stats.pmean(xp.asarray([1, 2, 3]), xp.asarray([0]))
        with pytest.raises(ValueError, match='Power mean only defined for'):
            stats.pmean(xp.asarray([1, 2, 3]), xp.asarray([0]))

    # 测试一维数组情况下的 pmean 函数
    def test_1d(self, xp):
        # 定义测试用例中的数组 a 和指数 p
        a, p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 3.5
        # 计算期望结果，调用 pmean_reference 方法
        desired = TestPMean.pmean_reference(np.array(a), p)
        # 调用 check_equal_pmean 函数，检查实现是否正确
        check_equal_pmean(a, p, desired, xp=xp)

        a, p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], -2.5
        desired = TestPMean.pmean_reference(np.array(a), p)
        check_equal_pmean(a, p, desired, xp=xp)

        a, p = [1, 2, 3, 4], 2
        # 预期结果是 1 到 4 平方和的平均的平方根
        desired = np.sqrt((1**2 + 2**2 + 3**2 + 4**2) / 4)
        check_equal_pmean(a, p, desired, xp=xp)

    # 测试包含零值的一维数组情况下的 pmean 函数
    def test_1d_with_zero(self, xp):
        a, p = np.array([1, 0]), -1
        desired = 0.0
        check_equal_pmean(a, p, desired, rtol=0.0, xp=xp)

    # 根据条件跳过特定的 xp 后端测试
    @skip_xp_backends('array_api_strict',
                      reasons=["`array_api_strict.where` `fillvalue` doesn't "
                               "accept Python scalars. See data-apis/array-api#807."])
    @pytest.mark.usefixtures("skip_xp_backends")
    # 测试包含负值的一维数组情况下的 pmean 函数
    def test_1d_with_negative_value(self, xp):
        a, p = np.array([1, 0, -1]), 1.23
        message = "The power mean is only defined..."
        # 检查是否触发 RuntimeWarning 并匹配给定消息
        with pytest.warns(RuntimeWarning, match=message):
            check_equal_pmean(a, p, xp.nan, xp=xp)

    # 参数化测试，测试二维数组情况下 axis=None 的 pmean 函数
    @pytest.mark.parametrize(
        ("a", "p"),
        [([[10, 20], [50, 60], [90, 100]], -0.5),
         (np.array([[10, 20], [50, 60], [90, 100]]), 0.5)]
    )
    def test_2d_axisnone(self, a, p, xp):
        desired = TestPMean.pmean_reference(np.array(a), p)
        check_equal_pmean(a, p, desired, xp=xp)

    # 参数化测试，测试二维数组情况下 axis=0 的 pmean 函数
    @pytest.mark.parametrize(
        ("a", "p"),
        [([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], -0.5),
         ([[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], 0.5)]
    )
    def test_2d_axis0(self, a, p, xp):
        # 计算每列的期望结果
        desired = [
            TestPMean.pmean_reference(
                np.array([a[i][j] for i in range(len(a))]), p
            )
            for j in range(len(a[0]))
        ]
        check_equal_pmean(a, p, desired, axis=0, xp=xp)

    # 参数化测试，测试二维数组情况下 axis=1 的 pmean 函数
    @pytest.mark.parametrize(
        ("a", "p"),
        [([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], -0.5),
         ([[10, 0, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]], 0.5)]
    )
    def test_2d_axis1(self, a, p, xp):
        # 计算每行的期望结果
        desired = [TestPMean.pmean_reference(np.array(a_), p) for a_ in a]
        check_equal_pmean(a, p, desired, axis=1, xp=xp)
    # 定义一个测试函数，测试一维情况下的加权均值计算
    def test_weights_1d(self, xp):
        # 设定测试数据
        a, p = [2, 10, 6], -1.23456789
        weights = [10, 5, 3]
        # 使用参考函数计算加权均值的期望结果
        desired = TestPMean.wpmean_reference(np.array(a), p, weights)
        # 调用检查函数，验证计算结果是否符合期望值
        check_equal_pmean(a, p, desired, weights=weights, rtol=1e-5, xp=xp)

    # 标记此测试函数只在 NumPy 后端运行，跳过其他后端
    @pytest.mark.skip_xp_backends(
        np_only=True,
        reasons=['array-likes only supported for NumPy backend'],
    )
    # 使用装饰器指定跳过 XP 后端的测试配置
    @pytest.mark.usefixtures("skip_xp_backends")
    # 定义一个测试函数，测试一维列表情况下的加权均值计算
    def test_weights_1d_list(self, xp):
        # 设定测试数据
        a, p = [2, 10, 6], -1.23456789
        weights = [10, 5, 3]
        # 使用参考函数计算加权均值的期望结果
        desired = TestPMean.wpmean_reference(np.array(a), p, weights)
        # 在调用 `pmean` 之前，检查输入是否正确转换为 XP 数组
        # 此时，验证函数仍然接受整数列表作为输入。
        res = stats.pmean(a, p, weights=weights)
        # 使用 XP 断言来检查结果是否与期望值非常接近
        xp_assert_close(res, np.asarray(desired), rtol=1e-5)

    # 标记跳过包含无效参数的 XP 测试
    @skip_xp_invalid_arg
    # 定义一个测试函数，测试掩码数组在一维情况下的加权平均计算
    def test_weights_masked_1d_array(self, xp):
        # 设定测试数据
        a, p = np.array([2, 10, 6, 42]), 1
        weights = np.ma.array([10, 5, 3, 42], mask=[0, 0, 0, 1])
        # 使用 NumPy 的 `average` 函数计算加权平均值的期望结果
        desired = np.average(a, weights=weights)
        # 将 XP 设置为 `np.ma`，以保留掩码，因为 `check_equal_pmean` 使用 `xp.asarray`
        xp = np.ma  # check_equal_pmean uses xp.asarray; this will preserve the mask
        # 调用检查函数，验证计算结果是否符合期望值
        check_equal_pmean(a, p, desired, weights=weights, rtol=1e-5,
                          dtype=np.float64, xp=xp)

    # 使用参数化装饰器定义多个测试情况，测试二维情况下的加权均值计算
    @pytest.mark.parametrize(
        ("axis", "fun_name", "p"),
        [(None, "wpmean_reference", 9.87654321),
         (0, "gmean", 0),
         (1, "hmean", -1)]
    )
    # 定义一个测试函数，测试二维情况下的加权均值计算
    def test_weights_2d(self, axis, fun_name, p, xp):
        # 如果函数名为 'wpmean_reference'，则定义一个特定的函数
        if fun_name == 'wpmean_reference':
            def fun(a, axis, weights):
                return TestPMean.wpmean_reference(a, p, weights)
        else:
            # 否则，获取对应名称的统计函数
            fun = getattr(stats, fun_name)
        # 设定测试数据
        a = np.array([[2, 5], [10, 5], [6, 5]])
        weights = np.array([[10, 1], [5, 1], [3, 1]])
        # 使用对应函数计算加权均值的期望结果
        desired = fun(a, axis=axis, weights=weights)
        # 调用检查函数，验证计算结果是否符合期望值
        check_equal_pmean(a, p, desired, axis=axis, weights=weights, rtol=1e-5, xp=xp)
class TestGSTD:
    # 定义一个一维数组，包含从1到24的整数，用于测试
    array_1d = np.arange(2 * 3 * 4) + 1
    # 预计的一维数组的几何标准差
    gstd_array_1d = 2.294407613602
    # 将一维数组重塑为三维数组
    array_3d = array_1d.reshape(2, 3, 4)

    # 测试计算一维数组的几何标准差
    def test_1d_array(self):
        gstd_actual = stats.gstd(self.array_1d)
        assert_allclose(gstd_actual, self.gstd_array_1d)

    # 测试计算类数组输入的一维数组的几何标准差
    def test_1d_numeric_array_like_input(self):
        gstd_actual = stats.gstd(tuple(self.array_1d))
        assert_allclose(gstd_actual, self.gstd_array_1d)

    # 测试当输入非数值类型时，是否会引发值错误
    def test_raises_value_error_non_numeric_input(self):
        # 此错误由 NumPy 抛出，但是很容易解释
        with pytest.raises(TypeError, match="ufunc 'log' not supported"):
            stats.gstd('You cannot take the logarithm of a string.')

    # 使用参数化测试，测试在输入无效值时返回 NaN 的情况
    @pytest.mark.parametrize('bad_value', (0, -1, np.inf, np.nan))
    def test_returns_nan_invalid_value(self, bad_value):
        x = np.append(self.array_1d, [bad_value])
        if np.isfinite(bad_value):
            message = "The geometric standard deviation is only defined..."
            with pytest.warns(RuntimeWarning, match=message):
                res = stats.gstd(x)
        else:
            res = stats.gstd(x)
        assert_equal(res, np.nan)

    # 测试是否正确传播 NaN 值
    def test_propagates_nan_values(self):
        a = array([[1, 1, 1, 16], [np.nan, 1, 2, 3]])
        gstd_actual = stats.gstd(a, axis=1)
        assert_allclose(gstd_actual, np.array([4, np.nan]))

    # 测试当自由度等于观察值数目时，是否会引发运行时警告
    def test_ddof_equal_to_number_of_observations(self):
        with pytest.warns(RuntimeWarning, match='Degrees of freedom <= 0'):
            assert_equal(stats.gstd(self.array_1d, ddof=self.array_1d.size), np.inf)

    # 测试计算三维数组的几何标准差，无指定轴
    def test_3d_array(self):
        gstd_actual = stats.gstd(self.array_3d, axis=None)
        assert_allclose(gstd_actual, self.gstd_array_1d)

    # 测试计算三维数组的几何标准差，指定轴为元组 (1, 2)
    def test_3d_array_axis_type_tuple(self):
        gstd_actual = stats.gstd(self.array_3d, axis=(1,2))
        assert_allclose(gstd_actual, [2.12939215, 1.22120169])

    # 测试计算三维数组的几何标准差，指定轴为 0
    def test_3d_array_axis_0(self):
        gstd_actual = stats.gstd(self.array_3d, axis=0)
        gstd_desired = np.array([
            [6.1330555493918, 3.958900210120, 3.1206598248344, 2.6651441426902],
            [2.3758135028411, 2.174581428192, 2.0260062829505, 1.9115518327308],
            [1.8205343606803, 1.746342404566, 1.6846557065742, 1.6325269194382]
        ])
        assert_allclose(gstd_actual, gstd_desired)

    # 测试计算三维数组的几何标准差，指定轴为 1
    def test_3d_array_axis_1(self):
        gstd_actual = stats.gstd(self.array_3d, axis=1)
        gstd_desired = np.array([
            [3.118993630946, 2.275985934063, 1.933995977619, 1.742896469724],
            [1.271693593916, 1.254158641801, 1.238774141609, 1.225164057869]
        ])
        assert_allclose(gstd_actual, gstd_desired)
    # 定义一个测试方法，用于测试在第三维度上计算标准差
    def test_3d_array_axis_2(self):
        # 计算实际的标准差，axis=2 表示在第三维度上进行计算
        gstd_actual = stats.gstd(self.array_3d, axis=2)
        # 期望的标准差结果数组
        gstd_desired = np.array([
            [1.8242475707664, 1.2243686572447, 1.1318311657788],
            [1.0934830582351, 1.0724479791887, 1.0591498540749]
        ])
        # 使用 assert_allclose 检查实际结果与期望结果的接近程度
        assert_allclose(gstd_actual, gstd_desired)

    # 定义一个测试方法，用于测试带掩码的三维数组输入情况
    def test_masked_3d_array(self):
        # 创建一个掩码数组，标记大于16的元素为掩码值
        ma = np.ma.masked_where(self.array_3d > 16, self.array_3d)
        # 设置警告消息内容
        message = "`gstd` support for masked array input was deprecated in..."
        # 使用 pytest.warns 检查是否有 DeprecationWarning 警告，并匹配警告消息
        with pytest.warns(DeprecationWarning, match=message):
            # 计算带掩码数组的实际标准差，axis=2 表示在第三维度上计算
            gstd_actual = stats.gstd(ma, axis=2)
        # 计算未掩码的原始数组在第三维度上的标准差
        gstd_desired = stats.gstd(self.array_3d, axis=2)
        # 期望的掩码数组，表示哪些元素被掩码
        mask = [[0, 0, 0], [0, 1, 1]]
        # 使用 assert_allclose 检查实际结果与期望结果的接近程度
        assert_allclose(gstd_actual, gstd_desired)
        # 使用 assert_equal 检查实际掩码数组是否与期望的掩码数组一致
        assert_equal(gstd_actual.mask, mask)
def test_binomtest():
    # precision tests compared to R for ticket:986
    # 定义概率数组，用于测试，分为三段
    pp = np.concatenate((np.linspace(0.1, 0.2, 5),
                         np.linspace(0.45, 0.65, 5),
                         np.linspace(0.85, 0.95, 5)))
    # 总体样本量
    n = 501
    # 成功次数
    x = 450
    # 预期的测试结果列表
    results = [0.0, 0.0, 1.0159969301994141e-304,
               2.9752418572150531e-275, 7.7668382922535275e-250,
               2.3381250925167094e-99, 7.8284591587323951e-81,
               9.9155947819961383e-65, 2.8729390725176308e-50,
               1.7175066298388421e-37, 0.0021070691951093692,
               0.12044570587262322, 0.88154763174802508, 0.027120993063129286,
               2.6102587134694721e-6]

    # 遍历每个概率和预期结果，进行断言比较
    for p, res in zip(pp, results):
        assert_approx_equal(stats.binomtest(x, n, p).pvalue, res,
                            significant=12, err_msg=f'fail forp={p}')
    
    # 单独的断言，用于验证特定参数的测试结果
    assert_approx_equal(stats.binomtest(50, 100, 0.1).pvalue,
                        5.8320387857343647e-24,
                        significant=12)


def test_binomtest2():
    # test added for issue #2384
    # 预期的测试结果列表，包含不同的概率组合
    res2 = [
        [1.0, 1.0],
        [0.5, 1.0, 0.5],
        [0.25, 1.0, 1.0, 0.25],
        [0.125, 0.625, 1.0, 0.625, 0.125],
        [0.0625, 0.375, 1.0, 1.0, 0.375, 0.0625],
        [0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875, 0.03125],
        [0.015625, 0.125, 0.453125, 1.0, 1.0, 0.453125, 0.125, 0.015625],
        [0.0078125, 0.0703125, 0.2890625, 0.7265625, 1.0, 0.7265625,
         0.2890625, 0.0703125, 0.0078125],
        [0.00390625, 0.0390625, 0.1796875, 0.5078125, 1.0, 1.0, 0.5078125,
         0.1796875, 0.0390625, 0.00390625],
        [0.001953125, 0.021484375, 0.109375, 0.34375, 0.75390625,
         1.0, 0.75390625, 0.34375, 0.109375, 0.021484375, 0.001953125]
    ]
    # 遍历每个参数组合，进行断言比较
    for k in range(1, 11):
        res1 = [stats.binomtest(v, k, 0.5).pvalue for v in range(k + 1)]
        assert_almost_equal(res1, res2[k - 1], decimal=10)


def test_binomtest3():
    # test added for issue #2384
    # 当 x == n*p 以及相邻情况的测试结果
    res3 = [stats.binomtest(v, v * k, 1. / k).pvalue
            for v in range(1, 11) for k in range(2, 11)]
    # 断言所有结果都等于1
    assert_equal(res3, np.ones(len(res3), int))

    # > bt=c()
    # > for(i in as.single(1:10)) {
    # +     for(k in as.single(2:10)) {
    # +         bt = c(bt, binom.test(i-1, k*i,(1/k))$p.value);
    # +         print(c(i+1, k*i,(1/k)))
    # +     }
    # + }
    # 创建一个 NumPy 数组，包含一系列预定义的浮点数
    binom_testm1 = np.array([
         0.5, 0.5555555555555556, 0.578125, 0.5904000000000003,
         0.5981224279835393, 0.603430543396034, 0.607304096221924,
         0.610255656871054, 0.612579511000001, 0.625, 0.670781893004115,
         0.68853759765625, 0.6980101120000006, 0.703906431368616,
         0.70793209416498, 0.7108561134173507, 0.713076544331419,
         0.714820192935702, 0.6875, 0.7268709038256367, 0.7418963909149174,
         0.74986110468096, 0.7548015520398076, 0.7581671424768577,
         0.760607984787832, 0.762459425024199, 0.7639120677676575, 0.7265625,
         0.761553963657302, 0.774800934828818, 0.7818005980538996,
         0.78613491480358, 0.789084353140195, 0.7912217659828884,
         0.79284214559524, 0.794112956558801, 0.75390625, 0.7856929451142176,
         0.7976688481430754, 0.8039848974727624, 0.807891868948366,
         0.8105487660137676, 0.812473307174702, 0.8139318233591120,
         0.815075399104785, 0.7744140625, 0.8037322594985427,
         0.814742863657656, 0.8205425178645808, 0.8241275984172285,
         0.8265645374416, 0.8283292196088257, 0.829666291102775,
         0.8307144686362666, 0.7905273437499996, 0.8178712053954738,
         0.828116983756619, 0.833508948940494, 0.8368403871552892,
         0.839104213210105, 0.840743186196171, 0.84198481438049,
         0.8429580531563676, 0.803619384765625, 0.829338573944648,
         0.8389591907548646, 0.84401876783902, 0.84714369697889,
         0.8492667010581667, 0.850803474598719, 0.851967542858308,
         0.8528799045949524, 0.8145294189453126, 0.838881732845347,
         0.847979024541911, 0.852760894015685, 0.8557134656773457,
         0.8577190131799202, 0.85917058278431, 0.860270010472127,
         0.861131648404582, 0.823802947998047, 0.846984756807511,
         0.855635653643743, 0.860180994825685, 0.86298688573253,
         0.864892525675245, 0.866271647085603, 0.867316125625004,
         0.8681346531755114
        ])
    
    # 下面是 R 语言的代码片段，这里注释显示了用 R 语言如何计算一系列值，并将结果存储在 bt 变量中
    # > bt=c()
    # > for(i in as.single(1:10)) {
    # +     for(k in as.single(2:10)) {
    # +         bt = c(bt, binom.test(i+1, k*i,(1/k))$p.value);
    # +         print(c(i+1, k*i,(1/k)))
    # +     }
    # + }
    # 创建包含多个浮点数的 NumPy 数组，表示二项分布检验中的概率值
    binom_testp1 = np.array([
         0.5, 0.259259259259259, 0.26171875, 0.26272, 0.2632244513031551,
         0.2635138663069203, 0.2636951804161073, 0.2638162407564354,
         0.2639010709000002, 0.625, 0.4074074074074074, 0.42156982421875,
         0.4295746560000003, 0.43473045988554, 0.4383309503172684,
         0.4409884859402103, 0.4430309389962837, 0.444649849401104, 0.6875,
         0.4927602499618962, 0.5096031427383425, 0.5189636628480,
         0.5249280070771274, 0.5290623300865124, 0.5320974248125793,
         0.5344204730474308, 0.536255847400756, 0.7265625, 0.5496019313526808,
         0.5669248746708034, 0.576436455045805, 0.5824538812831795,
         0.5866053321547824, 0.589642781414643, 0.5919618019300193,
         0.593790427805202, 0.75390625, 0.590868349763505, 0.607983393277209,
         0.617303847446822, 0.623172512167948, 0.627208862156123,
         0.6301556891501057, 0.632401894928977, 0.6341708982290303,
         0.7744140625, 0.622562037497196, 0.639236102912278, 0.648263335014579,
         0.65392850011132, 0.657816519817211, 0.660650782947676,
         0.662808780346311, 0.6645068560246006, 0.7905273437499996,
         0.6478843304312477, 0.6640468318879372, 0.6727589686071775,
         0.6782129857784873, 0.681950188903695, 0.684671508668418,
         0.686741824999918, 0.688369886732168, 0.803619384765625,
         0.668716055304315, 0.684360013879534, 0.6927642396829181,
         0.6980155964704895, 0.701609591890657, 0.7042244320992127,
         0.7062125081341817, 0.707775152962577, 0.8145294189453126,
         0.686243374488305, 0.7013873696358975, 0.709501223328243,
         0.714563595144314, 0.718024953392931, 0.7205416252126137,
         0.722454130389843, 0.723956813292035, 0.823802947998047,
         0.701255953767043, 0.715928221686075, 0.723772209289768,
         0.7286603031173616, 0.7319999279787631, 0.7344267920995765,
         0.736270323773157, 0.737718376096348
    ])
    
    # 使用嵌套列表推导式生成 res4_p1，其中包含对每个 (v, k) 组合应用二项分布检验并获取其 p 值
    res4_p1 = [stats.binomtest(v+1, v*k, 1./k).pvalue
               for v in range(1, 11) for k in range(2, 11)]
    
    # 使用嵌套列表推导式生成 res4_m1，其中包含对每个 (v, k) 组合应用二项分布检验并获取其 p 值
    res4_m1 = [stats.binomtest(v-1, v*k, 1./k).pvalue
               for v in range(1, 11) for k in range(2, 11)]
    
    # 断言：验证 res4_p1 数组与预期的 binom_testp1 数组在小数点后 13 位的精度上几乎相等
    assert_almost_equal(res4_p1, binom_testp1, decimal=13)
    # 断言：验证 res4_m1 数组与预期的 binom_testm1 数组在小数点后 13 位的精度上几乎相等
    assert_almost_equal(res4_m1, binom_testm1, decimal=13)
class TestTrim:
    # test trim functions

    # 测试 trim1 函数
    def test_trim1(self):
        # 创建一个长度为 11 的 NumPy 数组
        a = np.arange(11)
        # 断言：对数组 a 进行 10% 的左右修剪后，排序结果应为 [0, 1, ..., 9]
        assert_equal(np.sort(stats.trim1(a, 0.1)), np.arange(10))
        # 断言：对数组 a 进行 20% 的左右修剪后，排序结果应为 [0, 1, ..., 8]
        assert_equal(np.sort(stats.trim1(a, 0.2)), np.arange(9))
        # 断言：对数组 a 进行 20% 的左修剪后，排序结果应为 [2, 3, ..., 10]
        assert_equal(np.sort(stats.trim1(a, 0.2, tail='left')),
                     np.arange(2, 11))
        # 断言：对数组 a 进行 3/11 的左修剪后，排序结果应为 [3, 4, ..., 10]
        assert_equal(np.sort(stats.trim1(a, 3/11., tail='left')),
                     np.arange(3, 11))
        # 断言：对数组 a 进行完全的修剪后，返回空数组
        assert_equal(stats.trim1(a, 1.0), [])
        # 断言：对数组 a 进行完全的左修剪后，返回空数组
        assert_equal(stats.trim1(a, 1.0, tail='left'), [])

        # 空输入
        # 断言：对空数组进行 10% 的修剪后，返回空数组
        assert_equal(stats.trim1([], 0.1), [])
        # 断言：对空数组进行 3/11 的左修剪后，返回空数组
        assert_equal(stats.trim1([], 3/11., tail='left'), [])
        # 断言：对空数组进行 4/6 的修剪后，返回空数组
        assert_equal(stats.trim1([], 4/6.), [])

        # 测试 axis 参数
        # 创建一个形状为 (6, 4) 的数组 a
        a = np.arange(24).reshape(6, 4)
        # 创建参考数组 ref，将第一行进行修剪
        ref = np.arange(4, 24).reshape(5, 4)

        axis = 0
        # 在 axis=0 的维度上进行 20% 的左修剪
        trimmed = stats.trim1(a, 0.2, tail='left', axis=axis)
        # 断言：按 axis=0 维度排序后，修剪结果应与 ref 相同
        assert_equal(np.sort(trimmed, axis=axis), ref)

        axis = 1
        # 在 axis=1 的维度上进行 20% 的左修剪
        trimmed = stats.trim1(a.T, 0.2, tail='left', axis=axis)
        # 断言：按 axis=1 维度排序后，修剪结果应与 ref 的转置相同
        assert_equal(np.sort(trimmed, axis=axis), ref.T)

    # 测试 trimboth 函数
    def test_trimboth(self):
        # 创建一个长度为 11 的 NumPy 数组
        a = np.arange(11)
        # 断言：对数组 a 进行 3/11 的双侧修剪后，排序结果应为 [3, 4, 5, 6, 7]
        assert_equal(np.sort(stats.trimboth(a, 3/11.)), np.arange(3, 8))
        # 断言：对数组 a 进行 20% 的双侧修剪后，排序结果应为 [2, 3, ..., 8]
        assert_equal(np.sort(stats.trimboth(a, 0.2)),
                     np.array([2, 3, 4, 5, 6, 7, 8]))
        # 断言：对形状为 (6, 4) 的数组进行 20% 的双侧修剪后，排序结果应为 [4, 5, ..., 19]
        assert_equal(np.sort(stats.trimboth(np.arange(24).reshape(6, 4), 0.2)),
                     np.arange(4, 20).reshape(4, 4))
        # 断言：对转置后形状为 (4, 6) 的数组进行 2/6 的双侧修剪后，排序结果应为 [[2, 8, 14, 20], [3, 9, 15, 21]]
        assert_equal(np.sort(stats.trimboth(np.arange(24).reshape(4, 6).T,
                                            2/6.)),
                     np.array([[2, 8, 14, 20], [3, 9, 15, 21]]))
        # 断言：对转置后形状为 (4, 6) 的数组进行大于 4/6 的修剪时，应引发 ValueError 异常
        assert_raises(ValueError, stats.trimboth,
                      np.arange(24).reshape(4, 6).T, 4/6.)

        # 空输入
        # 断言：对空数组进行 10% 的双侧修剪后，返回空数组
        assert_equal(stats.trimboth([], 0.1), [])
        # 断言：对空数组进行 3/11 的双侧修剪后，返回空数组
        assert_equal(stats.trimboth([], 3/11.), [])
        # 断言：对空数组进行 4/6 的双侧修剪后，返回空数组
        assert_equal(stats.trimboth([], 4/6.), [])
    def test_trim_mean(self):
        # don't use pre-sorted arrays
        # 创建包含整数的 NumPy 数组 a
        a = np.array([4, 8, 2, 0, 9, 5, 10, 1, 7, 3, 6])
        # 创建索引数组 idx
        idx = np.array([3, 5, 0, 1, 2, 4])
        # 使用索引 idx 重新排列 NumPy 数组 a 的行
        a2 = np.arange(24).reshape(6, 4)[idx, :]
        # 使用列序（Fortran order）使用索引 idx 重新排列 NumPy 数组 a 的行
        a3 = np.arange(24).reshape(6, 4, order='F')[idx, :]
        # 调用 stats.trim_mean 函数，检查结果是否与预期值相等
        assert_equal(stats.trim_mean(a3, 2/6.),
                     np.array([2.5, 8.5, 14.5, 20.5]))
        # 调用 stats.trim_mean 函数，检查结果是否与预期值相等
        assert_equal(stats.trim_mean(a2, 2/6.),
                     np.array([10., 11., 12., 13.]))
        # 创建索引数组 idx4
        idx4 = np.array([1, 0, 3, 2])
        # 使用索引 idx4 重新排列 NumPy 数组 a4 的行
        a4 = np.arange(24).reshape(4, 6)[idx4, :]
        # 调用 stats.trim_mean 函数，检查结果是否与预期值相等
        assert_equal(stats.trim_mean(a4, 2/6.),
                     np.array([9., 10., 11., 12., 13., 14.]))
        # 创建混洗过的包含整数的列表 a
        a = [7, 11, 12, 21, 16, 6, 22, 1, 5, 0, 18, 10, 17, 9, 19, 15, 23,
             20, 2, 14, 4, 13, 8, 3]
        # 调用 stats.trim_mean 函数，检查结果是否与预期值相等
        assert_equal(stats.trim_mean(a, 2/6.), 11.5)
        # 调用 stats.trim_mean 函数，检查结果是否与预期值相等
        assert_equal(stats.trim_mean([5,4,3,1,2,0], 2/6.), 2.5)

        # check axis argument
        # 设置随机数种子
        np.random.seed(1234)
        # 创建随机整数数组 a，形状为 (5, 6, 4, 7)
        a = np.random.randint(20, size=(5, 6, 4, 7))
        # 对每一个轴进行迭代
        for axis in [0, 1, 2, 3, -1]:
            # 调用 stats.trim_mean 函数，检查结果是否与沿指定轴移动后的结果相等
            res1 = stats.trim_mean(a, 2/6., axis=axis)
            res2 = stats.trim_mean(np.moveaxis(a, axis, 0), 2/6.)
            assert_equal(res1, res2)

        # 调用 stats.trim_mean 函数，检查结果是否与沿所有轴展平后的结果相等
        res1 = stats.trim_mean(a, 2/6., axis=None)
        res2 = stats.trim_mean(a.ravel(), 2/6.)
        assert_equal(res1, res2)

        # 检查 ValueError 是否会被抛出
        assert_raises(ValueError, stats.trim_mean, a, 0.6)

        # empty input
        # 调用 stats.trim_mean 函数，检查空输入时返回值是否为 NaN
        assert_equal(stats.trim_mean([], 0.0), np.nan)
        # 调用 stats.trim_mean 函数，检查空输入时返回值是否为 NaN
        assert_equal(stats.trim_mean([], 0.6), np.nan)
class TestSigmaClip:
    # 定义测试类 TestSigmaClip，用于测试 sigmaclip 函数的各种情况

    def test_sigmaclip1(self):
        # 定义测试方法 test_sigmaclip1，测试默认参数下的 sigmaclip 函数行为
        a = np.concatenate((np.linspace(9.5, 10.5, 31), np.linspace(0, 20, 5)))
        # 创建一个数组 a，包含两部分均匀分布的数据
        fact = 4  # default
        # 设置参数 fact 为默认值 4
        c, low, upp = stats.sigmaclip(a)
        # 调用 sigmaclip 函数，对数组 a 进行剪切，并返回剪切后的数组 c，以及上下限 low 和 upp
        assert_(c.min() > low)
        # 断言剪切后的数组 c 的最小值大于下限 low
        assert_(c.max() < upp)
        # 断言剪切后的数组 c 的最大值小于上限 upp
        assert_equal(low, c.mean() - fact*c.std())
        # 断言下限 low 等于剪切数组 c 的均值减去 fact 倍标准差的结果
        assert_equal(upp, c.mean() + fact*c.std())
        # 断言上限 upp 等于剪切数组 c 的均值加上 fact 倍标准差的结果
        assert_equal(c.size, a.size)
        # 断言剪切后的数组 c 的大小等于原始数组 a 的大小

    def test_sigmaclip2(self):
        # 定义测试方法 test_sigmaclip2，测试指定参数下的 sigmaclip 函数行为
        a = np.concatenate((np.linspace(9.5, 10.5, 31), np.linspace(0, 20, 5)))
        # 创建一个数组 a，包含两部分均匀分布的数据
        fact = 1.5
        # 设置参数 fact 为 1.5
        c, low, upp = stats.sigmaclip(a, fact, fact)
        # 调用 sigmaclip 函数，使用指定参数对数组 a 进行剪切，并返回剪切后的数组 c，以及上下限 low 和 upp
        assert_(c.min() > low)
        # 断言剪切后的数组 c 的最小值大于下限 low
        assert_(c.max() < upp)
        # 断言剪切后的数组 c 的最大值小于上限 upp
        assert_equal(low, c.mean() - fact*c.std())
        # 断言下限 low 等于剪切数组 c 的均值减去 fact 倍标准差的结果
        assert_equal(upp, c.mean() + fact*c.std())
        # 断言上限 upp 等于剪切数组 c 的均值加上 fact 倍标准差的结果
        assert_equal(c.size, 4)
        # 断言剪切后的数组 c 的大小等于 4
        assert_equal(a.size, 36)
        # 断言原始数组 a 的大小仍然为 36，即原数组未被修改

    def test_sigmaclip3(self):
        # 定义测试方法 test_sigmaclip3，测试特定数据集下的 sigmaclip 函数行为
        a = np.concatenate((np.linspace(9.5, 10.5, 11),
                            np.linspace(-100, -50, 3)))
        # 创建一个数组 a，包含两部分数据：一部分均匀分布在 9.5 到 10.5 之间，另一部分分布在 -100 到 -50 之间
        fact = 1.8
        # 设置参数 fact 为 1.8
        c, low, upp = stats.sigmaclip(a, fact, fact)
        # 调用 sigmaclip 函数，使用指定参数对数组 a 进行剪切，并返回剪切后的数组 c，以及上下限 low 和 upp
        assert_(c.min() > low)
        # 断言剪切后的数组 c 的最小值大于下限 low
        assert_(c.max() < upp)
        # 断言剪切后的数组 c 的最大值小于上限 upp
        assert_equal(low, c.mean() - fact*c.std())
        # 断言下限 low 等于剪切数组 c 的均值减去 fact 倍标准差的结果
        assert_equal(upp, c.mean() + fact*c.std())
        # 断言上限 upp 等于剪切数组 c 的均值加上 fact 倍标准差的结果
        assert_equal(c, np.linspace(9.5, 10.5, 11))
        # 断言剪切后的数组 c 等于原始数据分布在 9.5 到 10.5 之间的均匀分布数据集

    def test_sigmaclip_result_attributes(self):
        # 定义测试方法 test_sigmaclip_result_attributes，测试 sigmaclip 函数返回结果的属性
        a = np.concatenate((np.linspace(9.5, 10.5, 11),
                            np.linspace(-100, -50, 3)))
        # 创建一个数组 a，包含两部分数据：一部分均匀分布在 9.5 到 10.5 之间，另一部分分布在 -100 到 -50 之间
        fact = 1.8
        # 设置参数 fact 为 1.8
        res = stats.sigmaclip(a, fact, fact)
        # 调用 sigmaclip 函数，使用指定参数对数组 a 进行剪切，并返回剪切后的结果 res
        attributes = ('clipped', 'lower', 'upper')
        # 定义属性列表 attributes，包括 'clipped', 'lower', 'upper'
        check_named_results(res, attributes)
        # 调用 check_named_results 函数，检查结果 res 是否包含指定的属性列表 attributes

    def test_std_zero(self):
        # 定义测试方法 test_std_zero，测试特定情况下的 sigmaclip 函数行为
        # regression test #8632
        x = np.ones(10)
        # 创建一个包含 10 个元素，全为 1 的数组 x
        assert_equal(stats.sigmaclip(x)[0], x)
        # 断言对数组 x 调用 sigmaclip 函数的结果的第一个元素等于 x 本身


class TestAlexanderGovern:
    # 定义测试类 TestAlexanderGovern，用于测试 alexandergovern 函数的行为

    def test_compare_dtypes(self):
        # 定义测试方法 test_compare_dtypes，测试不同数据类型下的 alexandergovern 函数行为
        args = [[13, 13, 13, 13, 13, 13, 13, 12, 12],
                [14, 13, 12, 12, 12, 12, 12, 11, 11],
                [14, 14, 13, 13, 13, 13, 13, 12, 12],
                [15, 14, 13, 13, 13, 12, 12, 12, 11]]
        # 创建一个包含多个数组的列表 args
        args_int16 = np.array(args, dtype=np.int16)
        # 将列表 args 转换为 NumPy 数组，数据类型为 np.int16
        args_int32 = np.array(args, dtype=np.int32)
        # 将列表 args 转换为 NumPy 数组，数据类型为 np.int32
        args_uint8 = np.array(args, dtype=np.uint8)
        # 将列表 args 转换为 NumPy 数组，数据类型为 np.uint8
        args_float64 = np.array(args, dtype=np.float64)
        # 将列表 args 转换为 NumPy 数组，数据类型为 np.float64

        res_int16 = stats.alexandergovern(*args_int16)
        # 调用 alexandergovern 函数，使用 np.int16 类型的参数，并获取返回结果
        res_int32 = stats.alexandergovern(*args_int32)
        # 调用 alexandergovern 函数，使用 np.int32 类型的参数，并获取返回结果
        res_unit8 = stats.alexandergovern(*args_uint8)
        # 调用 alexandergovern 函数，使用 np.uint8 类型
    # 定义一个测试函数，用于测试输入数组过小的情况
    def test_too_small_inputs(self, case):
        # 使用 pytest 的 warn 断言来检查是否会发出 SmallSampleWarning 警告，匹配特定的警告消息
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            # 调用 stats.alexandergovern 函数进行计算
            res = stats.alexandergovern(*case)
            # 断言计算结果的 statistic 属性为 NaN
            assert_equal(res.statistic, np.nan)
            # 断言计算结果的 pvalue 属性为 NaN
            assert_equal(res.pvalue, np.nan)

    # 定义一个测试函数，用于测试输入不合法（包含无穷大值）的情况
    def test_bad_inputs(self):
        # 使用 assert_raises 断言来检查是否会抛出 ValueError 异常，匹配特定的异常消息
        with assert_raises(ValueError, match="Input samples must be finite."):
            # 调用 stats.alexandergovern 函数并传入包含无穷大值的输入
            stats.alexandergovern([1, 2], [np.inf, np.inf])

    # 定义一个测试函数，用于测试与学者提供的数据进行比较
    def test_compare_scholar(self):
        '''
        从《The Modification and Evaluation of the
        Alexander-Govern Test in Terms of Power》一文中获取数据。
        '''
        # 定义三组数据：年轻、中年和老年
        young = [482.43, 484.36, 488.84, 495.15, 495.24, 502.69, 504.62,
                 518.29, 519.1, 524.1, 524.12, 531.18, 548.42, 572.1, 584.68,
                 609.09, 609.53, 666.63, 676.4]
        middle = [335.59, 338.43, 353.54, 404.27, 437.5, 469.01, 485.85,
                  487.3, 493.08, 494.31, 499.1, 886.41]
        old = [519.01, 528.5, 530.23, 536.03, 538.56, 538.83, 557.24, 558.61,
               558.95, 565.43, 586.39, 594.69, 629.22, 645.69, 691.84]
        # 调用 stats.alexandergovern 函数进行计算
        soln = stats.alexandergovern(young, middle, old)
        # 使用 assert_allclose 断言比较计算结果的 statistic 属性是否接近于给定值
        assert_allclose(soln.statistic, 5.3237, atol=1e-3)
        # 使用 assert_allclose 断言比较计算结果的 pvalue 属性是否接近于给定值
        assert_allclose(soln.pvalue, 0.06982, atol=1e-4)

        # 验证结果与 R 语言中 ag.test 的结果是否一致
        '''
        > library("onewaytests")
        > library("tibble")
        > young <- c(482.43, 484.36, 488.84, 495.15, 495.24, 502.69, 504.62,
        +                  518.29, 519.1, 524.1, 524.12, 531.18, 548.42, 572.1,
        +                  584.68, 609.09, 609.53, 666.63, 676.4)
        > middle <- c(335.59, 338.43, 353.54, 404.27, 437.5, 469.01, 485.85,
        +                   487.3, 493.08, 494.31, 499.1, 886.41)
        > old <- c(519.01, 528.5, 530.23, 536.03, 538.56, 538.83, 557.24,
        +                   558.61, 558.95, 565.43, 586.39, 594.69, 629.22,
        +                   645.69, 691.84)
        > young_fct <- c(rep("young", times=19))
        > middle_fct <-c(rep("middle", times=12))
        > old_fct <- c(rep("old", times=15))
        > ag.test(a ~ b, tibble(a=c(young, middle, old), b=factor(c(young_fct,
        +                                              middle_fct, old_fct))))

        Alexander-Govern Test (alpha = 0.05)
        -------------------------------------------------------------
        data : a and b

        statistic  : 5.324629
        parameter  : 2
        p.value    : 0.06978651

        Result     : Difference is not statistically significant.
        -------------------------------------------------------------
        '''
        # 使用 assert_allclose 断言比较计算结果的 statistic 属性是否接近于给定值
        assert_allclose(soln.statistic, 5.324629)
        # 使用 assert_allclose 断言比较计算结果的 pvalue 属性是否接近于给定值
        assert_allclose(soln.pvalue, 0.06978651)
    def test_compare_scholar3(self):
        '''
        Data taken from 'Robustness And Comparative Power Of WelchAspin,
        Alexander-Govern And Yuen Tests Under Non-Normality And Variance
        Heteroscedasticity', by Ayed A. Almoied. 2017. Page 34-37.
        https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=2775&context=oa_dissertations
        '''
        # 定义两组数据 x1 和 x2，用于进行统计分析
        x1 = [-1.77559, -1.4113, -0.69457, -0.54148, -0.18808, -0.07152,
              0.04696, 0.051183, 0.148695, 0.168052, 0.422561, 0.458555,
              0.616123, 0.709968, 0.839956, 0.857226, 0.929159, 0.981442,
              0.999554, 1.642958]
        x2 = [-1.47973, -1.2722, -0.91914, -0.80916, -0.75977, -0.72253,
              -0.3601, -0.33273, -0.28859, -0.09637, -0.08969, -0.01824,
              0.260131, 0.289278, 0.518254, 0.683003, 0.877618, 1.172475,
              1.33964, 1.576766]
        # 对数据 x1 和 x2 进行 Alexander-Govern 测试
        soln = stats.alexandergovern(x1, x2)
        # 断言测试结果的统计量接近指定值
        assert_allclose(soln.statistic, 0.713526, atol=1e-5)
        # 断言测试结果的 p 值接近指定值
        assert_allclose(soln.pvalue, 0.398276, atol=1e-5)

        '''
        tested in ag.test in R:
        > library("onewaytests")
        > library("tibble")
        > x1 <- c(-1.77559, -1.4113, -0.69457, -0.54148, -0.18808, -0.07152,
        +          0.04696, 0.051183, 0.148695, 0.168052, 0.422561, 0.458555,
        +          0.616123, 0.709968, 0.839956, 0.857226, 0.929159, 0.981442,
        +          0.999554, 1.642958)
        > x2 <- c(-1.47973, -1.2722, -0.91914, -0.80916, -0.75977, -0.72253,
        +         -0.3601, -0.33273, -0.28859, -0.09637, -0.08969, -0.01824,
        +         0.260131, 0.289278, 0.518254, 0.683003, 0.877618, 1.172475,
        +         1.33964, 1.576766)
        > x1_fact <- c(rep("x1", times=20))
        > x2_fact <- c(rep("x2", times=20))
        > a <- c(x1, x2)
        > b <- factor(c(x1_fact, x2_fact))
        > ag.test(a ~ b, tibble(a, b))
        Alexander-Govern Test (alpha = 0.05)
        -------------------------------------------------------------
        data : a and b

        statistic  : 0.7135182
        parameter  : 1
        p.value    : 0.3982783

        Result     : Difference is not statistically significant.
        -------------------------------------------------------------
        '''
        # 断言另一组测试结果的统计量接近指定值
        assert_allclose(soln.statistic, 0.7135182)
        # 断言另一组测试结果的 p 值接近指定值
        assert_allclose(soln.pvalue, 0.3982783)

    def test_nan_policy_propogate(self):
        # 定义包含 NaN 值的输入参数列表
        args = [[1, 2, 3, 4], [1, np.nan]]
        # 使用默认的 NaN 处理策略 'propagate' 进行 Alexander-Govern 测试
        res = stats.alexandergovern(*args)
        # 断言结果对象的 p 值为 NaN
        assert_equal(res.pvalue, np.nan)
        # 断言结果对象的统计量为 NaN
        assert_equal(res.statistic, np.nan)

    def test_nan_policy_raise(self):
        # 定义包含 NaN 值的输入参数列表
        args = [[1, 2, 3, 4], [1, np.nan]]
        # 使用 'raise' 策略期望抛出 ValueError 异常，匹配异常信息包含特定文本
        with assert_raises(ValueError, match="The input contains nan values"):
            stats.alexandergovern(*args, nan_policy='raise')
    # 定义测试函数，用于测试处理 NaN 的策略是否正确
    def test_nan_policy_omit(self):
        # 定义包含 NaN 的输入参数列表
        args_nan = [[1, 2, 3, np.nan, 4], [1, np.nan, 19, 25]]
        # 定义不包含 NaN 的输入参数列表
        args_no_nan = [[1, 2, 3, 4], [1, 19, 25]]
        
        # 调用 alexandergovern 函数处理含有 NaN 的参数列表，使用 'omit' 策略
        res_nan = stats.alexandergovern(*args_nan, nan_policy='omit')
        # 调用 alexandergovern 函数处理不含 NaN 的参数列表
        res_no_nan = stats.alexandergovern(*args_no_nan)
        
        # 断言处理后的结果 pvalue 相等
        assert_equal(res_nan.pvalue, res_no_nan.pvalue)
        # 断言处理后的结果 statistic 相等
        assert_equal(res_nan.statistic, res_no_nan.statistic)

    # 定义测试函数，测试常数输入情况下的处理
    def test_constant_input(self):
        # 提示消息，用于匹配常数输入警告
        msg = "An input array is constant; the statistic is not defined."
        
        # 使用 pytest 的 warn 函数捕获 ConstantInputWarning，并匹配特定消息
        with pytest.warns(stats.ConstantInputWarning, match=msg):
            # 调用 alexandergovern 函数，传入常数输入
            res = stats.alexandergovern([0.667, 0.667, 0.667],
                                        [0.123, 0.456, 0.789])
            # 断言结果 statistic 是 NaN
            assert_equal(res.statistic, np.nan)
            # 断言结果 pvalue 是 NaN
            assert_equal(res.pvalue, np.nan)
class TestFOneWay:

    def test_trivial(self):
        # A trivial test of stats.f_oneway, with F=0.
        F, p = stats.f_oneway([0, 2], [0, 2])
        assert_equal(F, 0.0)
        assert_equal(p, 1.0)

    def test_basic(self):
        # Despite being a floating point calculation, this data should
        # result in F being exactly 2.0.
        F, p = stats.f_oneway([0, 2], [2, 4])
        assert_equal(F, 2.0)
        assert_allclose(p, 1 - np.sqrt(0.5), rtol=1e-14)

    def test_known_exact(self):
        # Another trivial dataset for which the exact F and p can be
        # calculated.
        F, p = stats.f_oneway([2], [2], [2, 3, 4])
        # The use of assert_equal might be too optimistic, but the calculation
        # in this case is trivial enough that it is likely to go through with
        # no loss of precision.
        assert_equal(F, 3/5)
        assert_equal(p, 5/8)

    def test_large_integer_array(self):
        a = np.array([655, 788], dtype=np.uint16)
        b = np.array([789, 772], dtype=np.uint16)
        F, p = stats.f_oneway(a, b)
        # The expected value was verified by computing it with mpmath with
        # 40 digits of precision.
        assert_allclose(F, 0.77450216931805540, rtol=1e-14)

    def test_result_attributes(self):
        a = np.array([655, 788], dtype=np.uint16)
        b = np.array([789, 772], dtype=np.uint16)
        res = stats.f_oneway(a, b)
        attributes = ('statistic', 'pvalue')
        check_named_results(res, attributes)

    def test_nist(self):
        # These are the nist ANOVA files. They can be found at:
        # https://www.itl.nist.gov/div898/strd/anova/anova.html
        filenames = ['SiRstv.dat', 'SmLs01.dat', 'SmLs02.dat', 'SmLs03.dat',
                     'AtmWtAg.dat', 'SmLs04.dat', 'SmLs05.dat', 'SmLs06.dat',
                     'SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat']

        for test_case in filenames:
            rtol = 1e-7
            fname = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 'data/nist_anova', test_case))
            with open(fname) as f:
                content = f.read().split('\n')
            certified = [line.split() for line in content[40:48]
                         if line.strip()]
            dataf = np.loadtxt(fname, skiprows=60)
            y, x = dataf.T
            y = y.astype(int)
            caty = np.unique(y)
            f = float(certified[0][-1])

            xlist = [x[y == i] for i in caty]
            res = stats.f_oneway(*xlist)

            # With the hard test cases we relax the tolerance a bit.
            hard_tc = ('SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat')
            if test_case in hard_tc:
                rtol = 1e-4

            assert_allclose(res[0], f, rtol=rtol,
                            err_msg=f'Failing testcase: {test_case}')


注释：

        # 测试类 TestFOneWay，用于测试 stats.f_oneway 函数的各种情况
        class TestFOneWay:

            def test_trivial(self):
                # 使用 stats.f_oneway 进行简单测试，期望 F=0
                F, p = stats.f_oneway([0, 2], [0, 2])
                assert_equal(F, 0.0)
                assert_equal(p, 1.0)

            def test_basic(self):
                # 尽管是浮点数计算，但这组数据应该确切地导致 F=2.0
                F, p = stats.f_oneway([0, 2], [2, 4])
                assert_equal(F, 2.0)
                assert_allclose(p, 1 - np.sqrt(0.5), rtol=1e-14)

            def test_known_exact(self):
                # 另一个可以计算出确切 F 和 p 值的简单数据集
                F, p = stats.f_oneway([2], [2], [2, 3, 4])
                # 使用 assert_equal 可能有点过于乐观，但在这种情况下计算
                # 是足够简单的，应该不会丢失精度
                assert_equal(F, 3/5)
                assert_equal(p, 5/8)

            def test_large_integer_array(self):
                # 使用大整数数组进行测试
                a = np.array([655, 788], dtype=np.uint16)
                b = np.array([789, 772], dtype=np.uint16)
                F, p = stats.f_oneway(a, b)
                # 预期值通过使用 mpmath 计算得到，精确到 40 位小数
                assert_allclose(F, 0.77450216931805540, rtol=1e-14)

            def test_result_attributes(self):
                # 测试结果对象的属性
                a = np.array([655, 788], dtype=np.uint16)
                b = np.array([789, 772], dtype=np.uint16)
                res = stats.f_oneway(a, b)
                attributes = ('statistic', 'pvalue')
                check_named_results(res, attributes)

            def test_nist(self):
                # 这些是 nist ANOVA 文件，可以在以下网址找到：
                # https://www.itl.nist.gov/div898/strd/anova/anova.html
                filenames = ['SiRstv.dat', 'SmLs01.dat', 'SmLs02.dat', 'SmLs03.dat',
                             'AtmWtAg.dat', 'SmLs04.dat', 'SmLs05.dat', 'SmLs06.dat',
                             'SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat']

                for test_case in filenames:
                    rtol = 1e-7
                    fname = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                         'data/nist_anova', test_case))
                    with open(fname) as f:
                        content = f.read().split('\n')
                    certified = [line.split() for line in content[40:48]
                                 if line.strip()]
                    dataf = np.loadtxt(fname, skiprows=60)
                    y, x = dataf.T
                    y = y.astype(int)
                    caty = np.unique(y)
                    f = float(certified[0][-1])

                    xlist = [x[y == i] for i in caty]
                    res = stats.f_oneway(*xlist)

                    # 对于难度较大的测试案例，放宽容差值
                    hard_tc = ('SmLs07.dat', 'SmLs08.dat', 'SmLs09.dat')
                    if test_case in hard_tc:
                        rtol = 1e-4

                    assert_allclose(res[0], f, rtol=rtol,
                                    err_msg=f'Failing testcase: {test_case}')
    @pytest.mark.parametrize("a, b, expected", [
        # 参数化测试用例，分别测试不同的输入组合
        (np.array([42, 42, 42]), np.array([7, 7, 7]), (np.inf, 0)),
        # 第一个测试用例，输入数组都为常数，期望输出 (np.inf, 0)
        (np.array([42, 42, 42]), np.array([42, 42, 42]), (np.nan, np.nan))
        # 第二个测试用例，输入数组相同且为常数，期望输出 (np.nan, np.nan)
        ])
    # 测试类中的测试方法，用于验证输入数组都是常数时是否会触发警告
    def test_constant_input(self, a, b, expected):
        # 触发警告的消息内容
        msg = "Each of the input arrays is constant;"
        # 使用 pytest.warns 检查是否会触发指定类型的警告，并匹配特定消息
        with pytest.warns(stats.ConstantInputWarning, match=msg):
            # 调用 scipy.stats 中的 f_oneway 函数进行分析方差检验
            f, p = stats.f_oneway(a, b)
            # 验证检验结果是否符合预期
            assert f, p == expected

    @pytest.mark.parametrize('axis', [-2, -1, 0, 1])
    # 参数化测试，测试不同的轴向输入
    def test_2d_inputs(self, axis):
        # 构造多维数组 a, b, c 作为输入数据
        a = np.array([[1, 4, 3, 3],
                      [2, 5, 3, 3],
                      [3, 6, 3, 3],
                      [2, 3, 3, 3],
                      [1, 4, 3, 3]])
        b = np.array([[3, 1, 5, 3],
                      [4, 6, 5, 3],
                      [4, 3, 5, 3],
                      [1, 5, 5, 3],
                      [5, 5, 5, 3],
                      [2, 3, 5, 3],
                      [8, 2, 5, 3],
                      [2, 2, 5, 3]])
        c = np.array([[4, 3, 4, 3],
                      [4, 2, 4, 3],
                      [5, 4, 4, 3],
                      [5, 4, 4, 3]])

        # 根据不同的轴向重新定义数组的取值方式
        if axis in [-1, 1]:
            a = a.T
            b = b.T
            c = c.T
            take_axis = 0
        else:
            take_axis = 1

        # 警告消息内容
        warn_msg = "Each of the input arrays is constant;"
        # 使用 pytest.warns 检查是否会触发指定类型的警告，并匹配特定消息
        with pytest.warns(stats.ConstantInputWarning, match=warn_msg):
            # 调用 scipy.stats 中的 f_oneway 函数进行分析方差检验
            f, p = stats.f_oneway(a, b, c, axis=axis)

        # 验证使用 2D 数组计算的结果与单独切片计算的结果是否匹配
        for j in [0, 1]:
            # 分别对每个切片调用 f_oneway 进行检验
            fj, pj = stats.f_oneway(np.take(a, j, take_axis),
                                    np.take(b, j, take_axis),
                                    np.take(c, j, take_axis))
            # 使用 assert_allclose 进行数值近似比较
            assert_allclose(f[j], fj, rtol=1e-14)
            assert_allclose(p[j], pj, rtol=1e-14)
        for j in [2, 3]:
            # 再次检验切片结果，以验证警告是否仍然会被触发
            with pytest.warns(stats.ConstantInputWarning, match=warn_msg):
                fj, pj = stats.f_oneway(np.take(a, j, take_axis),
                                        np.take(b, j, take_axis),
                                        np.take(c, j, take_axis))
                # 使用 assert_equal 进行精确值比较
                assert_equal(f[j], fj)
                assert_equal(p[j], pj)
    def test_3d_inputs(self):
        # 创建三维数组 a, b, c，数组值并无特殊意义
        a = 1/np.arange(1.0, 4*5*7 + 1).reshape(4, 5, 7)
        b = 2/np.arange(1.0, 4*8*7 + 1).reshape(4, 8, 7)
        c = np.cos(1/np.arange(1.0, 4*4*7 + 1).reshape(4, 4, 7))

        # 对三个数组执行单因素方差分析，沿 axis=1 轴计算 F 值和 p 值
        f, p = stats.f_oneway(a, b, c, axis=1)

        # 断言结果的形状应为 (4, 7)
        assert f.shape == (4, 7)
        assert p.shape == (4, 7)

        # 遍历数组 a 的第一维和第三维，逐个计算单因素方差分析并断言结果与 f, p 的对应值接近
        for i in range(a.shape[0]):
            for j in range(a.shape[2]):
                fij, pij = stats.f_oneway(a[i, :, j], b[i, :, j], c[i, :, j])
                assert_allclose(fij, f[i, j])
                assert_allclose(pij, p[i, j])

    def test_length0_1d_error(self):
        # 断言当有空组时会触发 SmallSampleWarning，期望结果为 (NaN, NaN)
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            result = stats.f_oneway([1, 2, 3], [], [4, 5, 6, 7])
            assert_equal(result, (np.nan, np.nan))

    def test_length0_2d_error(self):
        # 断言当有空组时会触发 SmallSampleWarning，期望结果 f 和 p 均为 NaN 数组
        with pytest.warns(SmallSampleWarning, match=too_small_nd_not_omit):
            ncols = 3
            a = np.ones((4, ncols))
            b = np.ones((0, ncols))
            c = np.ones((5, ncols))
            f, p = stats.f_oneway(a, b, c)
            nans = np.full((ncols,), fill_value=np.nan)
            assert_equal(f, nans)
            assert_equal(p, nans)

    def test_all_length_one(self):
        # 断言所有输入样本长度为 1 时会触发 SmallSampleWarning，期望结果为 (NaN, NaN)
        with pytest.warns(SmallSampleWarning):
            result = stats.f_oneway([10], [11], [12], [13])
            assert_equal(result, (np.nan, np.nan))

    @pytest.mark.parametrize('args', [(), ([1, 2, 3],)])
    def test_too_few_inputs(self, args):
        # 断言输入参数数量不足时会触发 TypeError，期望匹配错误消息
        message = "At least two samples are required..."
        with assert_raises(TypeError, match=message):
            stats.f_oneway(*args)

    def test_axis_error(self):
        # 断言在指定错误的轴上进行计算时会触发 AxisError
        a = np.ones((3, 4))
        b = np.ones((5, 4))
        with assert_raises(AxisError):
            stats.f_oneway(a, b, axis=2)

    def test_bad_shapes(self):
        # 断言当输入数组形状不匹配时会触发 ValueError
        a = np.ones((3, 4))
        b = np.ones((5, 4))
        with assert_raises(ValueError):
            stats.f_oneway(a, b, axis=1)
class TestKruskal:
    # 测试简单情况下的 Kruskal-Wallis H 检验
    def test_simple(self):
        x = [1]
        y = [2]
        h, p = stats.kruskal(x, y)  # 计算 Kruskal-Wallis H 统计量和 p 值
        assert_equal(h, 1.0)  # 断言检验统计量 h 的值是否为 1.0
        assert_approx_equal(p, stats.distributions.chi2.sf(h, 1))  # 断言 p 值的近似值是否符合 Chi-squared 分布的累积分布函数
        h, p = stats.kruskal(np.array(x), np.array(y))  # 使用数组计算 Kruskal-Wallis H 统计量和 p 值
        assert_equal(h, 1.0)  # 断言检验统计量 h 的值是否为 1.0
        assert_approx_equal(p, stats.distributions.chi2.sf(h, 1))  # 断言 p 值的近似值是否符合 Chi-squared 分布的累积分布函数

    # 测试基本情况下的 Kruskal-Wallis H 检验
    def test_basic(self):
        x = [1, 3, 5, 7, 9]
        y = [2, 4, 6, 8, 10]
        h, p = stats.kruskal(x, y)  # 计算 Kruskal-Wallis H 统计量和 p 值
        assert_approx_equal(h, 3./11, significant=10)  # 断言检验统计量 h 的值近似等于 3/11
        assert_approx_equal(p, stats.distributions.chi2.sf(3./11, 1))  # 断言 p 值的近似值是否符合 Chi-squared 分布的累积分布函数
        h, p = stats.kruskal(np.array(x), np.array(y))  # 使用数组计算 Kruskal-Wallis H 统计量和 p 值
        assert_approx_equal(h, 3./11, significant=10)  # 断言检验统计量 h 的值近似等于 3/11
        assert_approx_equal(p, stats.distributions.chi2.sf(3./11, 1))  # 断言 p 值的近似值是否符合 Chi-squared 分布的累积分布函数

    # 测试简单情况下存在并列值的 Kruskal-Wallis H 检验
    def test_simple_tie(self):
        x = [1]
        y = [1, 2]
        h_uncorr = 1.5**2 + 2*2.25**2 - 12  # 计算未修正的 H 统计量
        corr = 0.75  # 修正系数
        expected = h_uncorr / corr   # 期望值
        h, p = stats.kruskal(x, y)  # 计算 Kruskal-Wallis H 统计量和 p 值
        # 由于表达式简单且确切答案为 0.5，使用 assert_equal() 是安全的。
        assert_equal(h, expected)  # 断言检验统计量 h 的值是否等于期望值

    # 测试另一种存在并列值的 Kruskal-Wallis H 检验
    def test_another_tie(self):
        x = [1, 1, 1, 2]
        y = [2, 2, 2, 2]
        h_uncorr = (12. / 8. / 9.) * 4 * (3**2 + 6**2) - 3 * 9  # 计算未修正的 H 统计量
        corr = 1 - float(3**3 - 3 + 5**3 - 5) / (8**3 - 8)  # 修正系数
        expected = h_uncorr / corr  # 期望值
        h, p = stats.kruskal(x, y)  # 计算 Kruskal-Wallis H 统计量和 p 值
        assert_approx_equal(h, expected)  # 断言检验统计量 h 的值是否近似等于期望值

    # 测试三组存在并列值的 Kruskal-Wallis H 检验
    def test_three_groups(self):
        # 一个测试包含三组并列值的 Kruskal-Wallis H 检验
        x = [1, 1, 1]
        y = [2, 2, 2]
        z = [2, 2]
        h_uncorr = (12. / 8. / 9.) * (3*2**2 + 3*6**2 + 2*6**2) - 3 * 9  # 计算未修正的 H 统计量
        corr = 1 - float(3**3 - 3 + 5**3 - 5) / (8**3 - 8)  # 修正系数
        expected = h_uncorr / corr  # 期望值
        h, p = stats.kruskal(x, y, z)  # 计算 Kruskal-Wallis H 统计量和 p 值
        assert_approx_equal(h, expected)  # 断言检验统计量 h 的值是否近似等于期望值
        assert_approx_equal(p, stats.distributions.chi2.sf(h, 2))  # 断言 p 值的近似值是否符合 Chi-squared 分布的累积分布函数

    # 测试空组的 Kruskal-Wallis H 检验
    def test_empty(self):
        # 一个测试包含空组的 Kruskal-Wallis H 检验
        x = [1, 1, 1]
        y = [2, 2, 2]
        z = []
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            assert_equal(stats.kruskal(x, y, z), (np.nan, np.nan))  # 断言空组的返回值是 (NaN, NaN)

    # 测试 Kruskal-Wallis 结果属性
    def test_kruskal_result_attributes(self):
        x = [1, 3, 5, 7, 9]
        y = [2, 4, 6, 8, 10]
        res = stats.kruskal(x, y)  # 计算 Kruskal-Wallis H 统计量和 p 值
        attributes = ('statistic', 'pvalue')  # 结果的属性
        check_named_results(res, attributes)  # 检查结果的命名属性

    # 测试 NaN 处理策略
    def test_nan_policy(self):
        x = np.arange(10.)
        x[9] = np.nan
        assert_equal(stats.kruskal(x, x), (np.nan, np.nan))  # 断言 NaN 处理策略 'omit' 后的返回值是 (0.0, 1.0)
        assert_almost_equal(stats.kruskal(x, x, nan_policy='omit'), (0.0, 1.0))  # 断言 NaN 处理策略 'raise' 会引发 ValueError
        assert_raises(ValueError, stats.kruskal, x, x, nan_policy='raise')  # 断言 NaN 处理策略 'foobar' 会引发 ValueError
    # 定义测试函数，用于测试处理大样本的情况
    def test_large_no_samples(self):
        # Test to see if large samples are handled correctly.
        # 设定样本数量
        n = 50000
        # 生成一个包含 n 个随机值的数组 x
        x = np.random.randn(n)
        # 生成一个包含 n 个随机值且每个值增加 50 的数组 y
        y = np.random.randn(n) + 50
        # 使用 Kruskal-Wallis 检验计算 x 和 y 之间的 H 统计量和 p 值
        h, p = stats.kruskal(x, y)
        # 预期 p 值为 0
        expected = 0
        # 断言 p 值与预期值相近
        assert_approx_equal(p, expected)

    # 定义测试函数，测试 stats.kruskal 函数在无参数传入时的行为
    def test_no_args_gh20661(self):
        # 期望引发 ValueError 异常，并指定异常消息
        message = r"Need at least two groups in stats.kruskal\(\)"
        # 使用 pytest 检测是否引发指定异常，并验证异常消息是否匹配
        with pytest.raises(ValueError, match=message):
            stats.kruskal()
# 使用装饰器标记此类兼容数组 API
@array_api_compatible
# 定义一个名为 TestCombinePvalues 的测试类
class TestCombinePvalues:
    # 设置参考值，这些值是使用 R 代码计算得出的：
    # options(digits=16)
    # library(metap)
    # x = c(0.01, 0.2, 0.3)
    # sumlog(x)  # fisher
    # sumz(x)  # stouffer
    # sumlog(1-x)  # pearson (negative statistic and complement of p-value)
    # minimump(x)  # tippett

    # 使用 pytest 的 parametrize 装饰器定义参数化测试
    @pytest.mark.parametrize(
        "method, expected_statistic, expected_pvalue",
        [("fisher", 14.83716180549625 , 0.02156175132483465),
         ("stouffer", 2.131790594240385, 0.01651203260896294),
         ("pearson", -1.179737662212887, 1-0.9778736999143087),
         ("tippett", 0.01, 0.02970100000000002),
         # mudholkar_george: library(transite); p_combine(x, method="MG")
         ("mudholkar_george", 6.828712071641684, 0.01654551838539527)])
    # 定义测试方法，验证计算的统计量和 p 值是否与预期相符
    def test_reference_values(self, xp, method, expected_statistic, expected_pvalue):
        # 定义输入数据 x
        x = [.01, .2, .3]
        # 调用 stats 模块的 combine_pvalues 函数计算结果
        res = stats.combine_pvalues(xp.asarray(x), method=method)
        # 使用 xp_assert_close 函数验证统计量是否接近预期值
        xp_assert_close(res.statistic, xp.asarray(expected_statistic))
        # 使用 xp_assert_close 函数验证 p 值是否接近预期值
        xp_assert_close(res.pvalue, xp.asarray(expected_pvalue))

    # 进行权重化的 Stouffer 方法的参数化测试
    @pytest.mark.parametrize(
        # 使用 R 的 metap 包中 sumz 计算的参考值：
        # options(digits=16)
        # library(metap)
        # x = c(0.01, 0.2, 0.3)
        # sumz(x, weights=c(1., 1., 1.))
        # sumz(x, weights=c(1., 4., 9.))
        "weights, expected_statistic, expected_pvalue",
        [([1., 1., 1.], 2.131790594240385, 0.01651203260896294),
         ([1., 4., 9.], 1.051815015753598, 0.1464422142261314)])
    # 定义测试方法，验证权重化 Stouffer 方法的计算结果
    def test_weighted_stouffer(self, xp, weights, expected_statistic, expected_pvalue):
        # 定义输入数据 x
        x = xp.asarray([.01, .2, .3])
        # 调用 stats 模块的 combine_pvalues 函数计算结果，使用指定的权重
        res = stats.combine_pvalues(x, method='stouffer', weights=xp.asarray(weights))
        # 使用 xp_assert_close 函数验证统计量是否接近预期值
        xp_assert_close(res.statistic, xp.asarray(expected_statistic))
        # 使用 xp_assert_close 函数验证 p 值是否接近预期值
        xp_assert_close(res.pvalue, xp.asarray(expected_pvalue))

    # 定义方法列表 methods，包含不同的合并方法
    methods = ["fisher", "pearson", "tippett", "stouffer", "mudholkar_george"]

    # 使用 pytest 的 parametrize 装饰器定义参数化测试，测试变体和方法组合
    @pytest.mark.parametrize("variant", ["single", "all", "random"])
    @pytest.mark.parametrize("method", methods)
    def test_monotonicity(self, variant, method, xp):
        xp_test = array_namespace(xp.asarray(1))
        # Test that result increases monotonically with respect to input.
        m, n = 10, 7
        rng = np.random.default_rng(278448169958891062669391462690811630763)

        # `pvaluess` is an m × n array of p values. Each row corresponds to
        # a set of p values to be combined with p values increasing
        # monotonically down one column (single), simultaneously down each
        # column (all), or independently down each column (random).
        if variant == "single":
            # Create an m × n array of p values where the first column
            # is linearly spaced from 0.1 to 0.9 and the remaining values
            # are randomly generated from a uniform distribution.
            pvaluess = xp.broadcast_to(xp.asarray(rng.random(n)), (m, n))
            pvaluess = xp_test.concat([xp.reshape(xp.linspace(0.1, 0.9, m), (-1, 1)),
                                       pvaluess[:, 1:]], axis=1)
        elif variant == "all":
            # Create an m × n array of p values where each column contains
            # values linearly spaced from 0.1 to 0.9.
            pvaluess = xp.broadcast_to(xp.linspace(0.1, 0.9, m), (n, m)).T
        elif variant == "random":
            # Create an m × n array of p values where each column is
            # sorted independently from a uniform distribution.
            pvaluess = xp_test.sort(xp.asarray(rng.uniform(0, 1, size=(m, n))), axis=0)

        # Calculate combined p values using the specified method for each row
        combined_pvalues = xp.asarray([
            stats.combine_pvalues(pvaluess[i, :], method=method)[1]
            for i in range(pvaluess.shape[0])
        ])
        # Assert that combined p values increase monotonically
        assert xp.all(combined_pvalues[1:] - combined_pvalues[:-1] >= 0)

    @pytest.mark.parametrize("method", methods)
    def test_result(self, method, xp):
        # Test result of combining p values with the specified method
        res = stats.combine_pvalues(xp.asarray([.01, .2, .3]), method=method)
        xp_assert_equal(res.statistic, res[0])
        xp_assert_equal(res.pvalue, res[1])

    @pytest.mark.parametrize("method", methods)
    # axis=None is currently broken for array API; will be handled when
    # axis_nan_policy decorator is updated
    @pytest.mark.parametrize("axis", [0, 1])
    def test_axis(self, method, axis, xp):
        rng = np.random.default_rng(234892349810482)
        x = xp.asarray(rng.random(size=(2, 10)))
        x = x.T if (axis == 0) else x
        # Combine p values along the specified axis
        res = stats.combine_pvalues(x, axis=axis)

        if axis is None:
            # Handle axis=None case separately due to current limitations
            x = xp.reshape(x, (-1,))
            ref = stats.combine_pvalues(x)
            xp_assert_close(res.statistic, ref.statistic)
            xp_assert_close(res.pvalue, ref.pvalue)
            return

        x = x.T if (axis == 0) else x
        x0, x1 = x[0, :], x[1, :]
        # Combine p values for each column independently
        ref0 = stats.combine_pvalues(x0)
        ref1 = stats.combine_pvalues(x1)

        xp_assert_close(res.statistic[0], ref0.statistic)
        xp_assert_close(res.statistic[1], ref1.statistic)
        xp_assert_close(res.pvalue[0], ref0.pvalue)
        xp_assert_close(res.pvalue[1], ref1.pvalue)
class TestCdfDistanceValidation:
    """
    Test that _cdf_distance() (via wasserstein_distance()) raises ValueErrors
    for bad inputs.
    """

    def test_distinct_value_and_weight_lengths(self):
        # 当权重的数量与数值的数量不匹配时，应该抛出 ValueError 异常。
        assert_raises(ValueError, stats.wasserstein_distance,
                      [1], [2], [4], [3, 1])
        assert_raises(ValueError, stats.wasserstein_distance, [1], [2], [1, 0])

    def test_zero_weight(self):
        # 当某个分布给定了零权重时，应该抛出 ValueError 异常。
        assert_raises(ValueError, stats.wasserstein_distance,
                      [0, 1], [2], [0, 0])
        assert_raises(ValueError, stats.wasserstein_distance,
                      [0, 1], [2], [3, 1], [0])

    def test_negative_weights(self):
        # 如果存在任何负权重，应该抛出 ValueError 异常。
        assert_raises(ValueError, stats.wasserstein_distance,
                      [0, 1], [2, 2], [1, 1], [3, -1])

    def test_empty_distribution(self):
        # 当尝试测量某物与空集之间的距离时，应该抛出 ValueError 异常。
        assert_raises(ValueError, stats.wasserstein_distance, [], [2, 2])
        assert_raises(ValueError, stats.wasserstein_distance, [1], [])

    def test_inf_weight(self):
        # 无穷大的权重是无效的。
        assert_raises(ValueError, stats.wasserstein_distance,
                      [1, 2, 1], [1, 1], [1, np.inf, 1], [1, 1])


class TestWassersteinDistanceND:
    """ Tests for wasserstein_distance_nd() output values.
    """

    def test_published_values(self):
        # 与发布的值和手动计算结果进行比较。
        # 这些值和计算结果发布在 James D. McCaffrey 的博客中，
        # https://jamesmccaffrey.wordpress.com/2018/03/05/earth-mover-distance
        # -wasserstein-metric-example-calculation/
        u = [(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1),
             (4,2), (6,1), (6,1)]
        v = [(2,1), (2,1), (3,2), (3,2), (3,2), (5,1), (5,1), (5,1), (5,1), (5,1),
             (5,1), (5,1), (7,1)]

        res = stats.wasserstein_distance_nd(u, v)
        # 在原始文章中，作者保留了两位小数以便计算。
        # 此测试使用更精确的距离值以获得精确的结果。
        # 请参阅原始博客文章中的表格和图表进行比较。
        flow = np.array([2., 3., 5., 1., 1., 1.])
        dist = np.array([1.00, 5**0.5, 4.00, 2**0.5, 1.00, 1.00])
        ref = np.sum(flow * dist)/np.sum(flow)
        assert_allclose(res, ref)

    @pytest.mark.parametrize('n_value', (4, 15, 35))
    @pytest.mark.parametrize('ndim', (3, 4, 7))
    @pytest.mark.parametrize('max_repeats', (5, 10))
    # 定义一个测试方法，用于验证相同分布的 n 维数据集的 Wasserstein 距离是否为零
    def test_same_distribution_nD(self, ndim, n_value, max_repeats):
        # 使用特定种子创建随机数生成器对象
        rng = np.random.default_rng(363836384995579937222333)
        # 生成指定范围内的随机整数数组，用于重复次数
        repeats = rng.integers(1, max_repeats, size=n_value, dtype=int)

        # 生成指定形状的随机数数组作为 u_values
        u_values = rng.random(size=(n_value, ndim))
        # 根据重复次数扩展 u_values，构造 v_values
        v_values = np.repeat(u_values, repeats, axis=0)
        # 生成与 v_values 长度相同的随机权重数组作为 v_weights
        v_weights = rng.random(np.sum(repeats))
        # 根据重复次数生成范围数组，用于计算 u_weights
        range_repeat = np.repeat(np.arange(len(repeats)), repeats)
        u_weights = np.bincount(range_repeat, weights=v_weights)
        # 随机排列 v_weights 和 v_values
        index = rng.permutation(len(v_weights))
        v_values, v_weights = v_values[index], v_weights[index]

        # 计算 u_values 和 v_values 之间的 n 维 Wasserstein 距离
        res = stats.wasserstein_distance_nd(u_values, v_values, u_weights, v_weights)
        # 断言计算结果与零的距离在给定的误差范围内
        assert_allclose(res, 0, atol=1e-15)

    # 使用不同参数化的方式定义测试方法，用于验证 n 维数据集的 Wasserstein 距离是否正确处理
    @pytest.mark.parametrize('nu', (8, 9, 38))
    @pytest.mark.parametrize('nv', (8, 12, 17))
    @pytest.mark.parametrize('ndim', (3, 5, 23))
    def test_collapse_nD(self, nu, nv, ndim):
        # 测试将 n 维分布折叠到零点分布的情况
        # 计算 u_values 和 v_values 之间的 n 维 Wasserstein 距离，参考使用 u_weights 和 v_weights
        rng = np.random.default_rng(38573488467338826109)
        u_values = rng.random(size=(nu, ndim))
        v_values = np.zeros((nv, ndim))
        u_weights = rng.random(size=nu)
        v_weights = rng.random(size=nv)
        # 计算参考值，作为对比基准
        ref = np.average(np.linalg.norm(u_values, axis=1), weights=u_weights)
        res = stats.wasserstein_distance_nd(u_values, v_values, u_weights, v_weights)
        # 断言计算结果与参考值的接近程度
        assert_allclose(res, ref)

    # 使用不同参数化的方式定义测试方法，验证处理包含零权重值的 n 维数据集时的行为
    @pytest.mark.parametrize('nu', (8, 16, 32))
    @pytest.mark.parametrize('nv', (8, 16, 32))
    @pytest.mark.parametrize('ndim', (1, 2, 6))
    def test_zero_weight_nD(self, nu, nv, ndim):
        # 具有零权重值的数据点对 Wasserstein 距离没有影响
        rng = np.random.default_rng(38573488467338826109)
        u_values = rng.random(size=(nu, ndim))
        v_values = rng.random(size=(nv, ndim))
        u_weights = rng.random(size=nu)
        v_weights = rng.random(size=nv)
        # 计算参考值，作为对比基准
        ref = stats.wasserstein_distance_nd(u_values, v_values, u_weights, v_weights)

        # 在 u_values 中插入包含零权重的行，并重新计算 Wasserstein 距离
        add_row, nrows = rng.integers(0, nu, size=2)
        add_value = rng.random(size=(nrows, ndim))
        u_values = np.insert(u_values, add_row, add_value, axis=0)
        u_weights = np.insert(u_weights, add_row, np.zeros(nrows), axis=0)
        res = stats.wasserstein_distance_nd(u_values, v_values, u_weights, v_weights)
        # 断言计算结果与参考值的接近程度
        assert_allclose(res, ref)
    def test_inf_values(self):
        # 测试处理无穷大值情况下的 Wasserstein 距离计算
        # 无穷大值可能导致距离为无穷大，或者触发 RuntimeWarning 并返回 NaN（距离未定义的情况）
        
        # 定义输入数据：uv, vv, uw
        uv, vv, uw = [[1, 1], [2, 1]], [[np.inf, -np.inf]], [1, 1]
        # 计算 Wasserstein 距离
        distance = stats.wasserstein_distance_nd(uv, vv, uw)
        # 断言计算结果为无穷大
        assert_equal(distance, np.inf)
        
        # 使用 np.errstate 忽略无效操作警告
        with np.errstate(invalid='ignore'):
            # 重新定义 uv, vv 以包含无穷大值
            uv, vv = [[np.inf, np.inf]], [[np.inf, -np.inf]]
            # 计算 Wasserstein 距离
            distance = stats.wasserstein_distance_nd(uv, vv)
            # 断言计算结果为 NaN
            assert_equal(distance, np.nan)

    @pytest.mark.parametrize('nu', (10, 15, 20))
    @pytest.mark.parametrize('nv', (10, 15, 20))
    @pytest.mark.parametrize('ndim', (1, 3, 5))
    def test_multi_dim_nD(self, nu, nv, ndim):
        # 测试多维情况下添加维度对结果的影响
        # 分别测试 nu, nv, ndim 参数组合
        
        # 创建随机数生成器对象 rng
        rng = np.random.default_rng(2736495738494849509)
        
        # 生成随机数据 u_values, v_values, u_weights, v_weights
        u_values = rng.random(size=(nu, ndim))
        v_values = rng.random(size=(nv, ndim))
        u_weights = rng.random(size=nu)
        v_weights = rng.random(size=nv)
        
        # 计算基准的 Wasserstein 距离
        ref = stats.wasserstein_distance_nd(u_values, v_values, u_weights, v_weights)

        # 随机选择一个维度和值进行添加
        add_dim = rng.integers(0, ndim)
        add_value = rng.random()

        # 在 u_values 和 v_values 中添加维度
        u_values = np.insert(u_values, add_dim, add_value, axis=1)
        v_values = np.insert(v_values, add_dim, add_value, axis=1)
        
        # 计算添加维度后的 Wasserstein 距离
        res = stats.wasserstein_distance_nd(u_values, v_values, u_weights, v_weights)
        
        # 断言添加维度后的计算结果与基准计算结果接近
        assert_allclose(res, ref)

    @pytest.mark.parametrize('nu', (7, 13, 19))
    @pytest.mark.parametrize('nv', (7, 13, 19))
    @pytest.mark.parametrize('ndim', (2, 4, 7))
    def test_orthogonal_nD(self, nu, nv, ndim):
        # 测试正交变换对 Wasserstein 距离计算结果的影响
        # 分别测试 nu, nv, ndim 参数组合
        
        # 创建随机数生成器对象 rng
        rng = np.random.default_rng(34746837464536)
        
        # 生成随机数据 u_values, v_values, u_weights, v_weights
        u_values = rng.random(size=(nu, ndim))
        v_values = rng.random(size=(nv, ndim))
        u_weights = rng.random(size=nu)
        v_weights = rng.random(size=nv)
        
        # 计算基准的 Wasserstein 距离
        ref = stats.wasserstein_distance_nd(u_values, v_values, u_weights, v_weights)

        # 生成一个随机的正交矩阵变换
        dist = stats.ortho_group(ndim)
        transform = dist.rvs(random_state=rng)
        shift = rng.random(size=ndim)
        
        # 应用正交变换和平移，计算变换后的 Wasserstein 距离
        res = stats.wasserstein_distance_nd(u_values @ transform + shift,
                                            v_values @ transform + shift,
                                            u_weights, v_weights)
        
        # 断言变换后的计算结果与基准计算结果接近
        assert_allclose(res, ref)
    # 定义一个测试方法，用于测试错误代码情况
    def test_error_code(self):
        # 使用指定的随机种子创建一个随机数生成器
        rng = np.random.default_rng(52473644737485644836320101)
        
        # 使用 pytest 的上下文管理器检查是否引发 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match='Invalid input values. The inputs'):
            # 创建两个不同尺寸的随机数组 u_values 和 v_values
            u_values = rng.random(size=(4, 10, 15))
            v_values = rng.random(size=(6, 2, 7))
            # 调用 wasserstein_distance_nd 函数计算 Wasserstein 距离
            _ = stats.wasserstein_distance_nd(u_values, v_values)
        
        # 再次使用 pytest 的上下文管理器检查 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError, match='Invalid input values. Dimensions'):
            # 创建两个不同尺寸的随机数组 u_values 和 v_values
            u_values = rng.random(size=(15,))
            v_values = rng.random(size=(3, 15))
            # 调用 wasserstein_distance_nd 函数计算 Wasserstein 距离
            _ = stats.wasserstein_distance_nd(u_values, v_values)
        
        # 最后使用 pytest 的上下文管理器检查 ValueError 异常，并验证异常消息
        with pytest.raises(ValueError,
                           match='Invalid input values. If two-dimensional'):
            # 创建两个不同尺寸的随机数组 u_values 和 v_values
            u_values = rng.random(size=(2, 10))
            v_values = rng.random(size=(2, 2))
            # 调用 wasserstein_distance_nd 函数计算 Wasserstein 距离
            _ = stats.wasserstein_distance_nd(u_values, v_values)

    # 使用 pytest 的参数化装饰器，指定多组测试参数 u_size 和 v_size
    @pytest.mark.parametrize('u_size', [1, 10, 50])
    @pytest.mark.parametrize('v_size', [1, 10, 50])
    # 定义一个测试方法，用于测试优化与分析结果的一致性
    def test_optimization_vs_analytical(self, u_size, v_size):
        # 使用指定的随机种子创建一个随机数生成器
        rng = np.random.default_rng(45634745675)
        
        # 测试 u_weights 和 v_weights 为空时的情况
        u_values = rng.random(size=(u_size, 1))
        v_values = rng.random(size=(v_size, 1))
        u_values_flat = u_values.ravel()
        v_values_flat = v_values.ravel()
        
        # 分别使用不同的后端计算 Wasserstein 距离
        d1 = stats.wasserstein_distance(u_values_flat, v_values_flat)
        d2 = stats.wasserstein_distance_nd(u_values, v_values)
        d3 = stats.wasserstein_distance_nd(u_values_flat, v_values_flat)
        
        # 断言 d2 和 d1 的值近似相等
        assert_allclose(d2, d1)
        # 断言 d3 和 d1 的值近似相等
        assert_allclose(d3, d1)
        
        # 测试指定了 u_weights 和 v_weights 的情况
        u_weights = rng.random(size=u_size)
        v_weights = rng.random(size=v_size)
        
        d1 = stats.wasserstein_distance(u_values_flat, v_values_flat,
                                        u_weights, v_weights)
        d2 = stats.wasserstein_distance_nd(u_values, v_values,
                                        u_weights, v_weights)
        d3 = stats.wasserstein_distance_nd(u_values_flat, v_values_flat,
                                        u_weights, v_weights)
        
        # 断言 d2 和 d1 的值近似相等
        assert_allclose(d2, d1)
        # 断言 d3 和 d1 的值近似相等
        assert_allclose(d3, d1)
class TestWassersteinDistance:
    """ Tests for wasserstein_distance() output values.
    """

    def test_simple(self):
        # For basic distributions, the value of the Wasserstein distance is
        # straightforward.
        assert_allclose(
            stats.wasserstein_distance([0, 1], [0], [1, 1], [1]),
            .5)
        assert_allclose(stats.wasserstein_distance(
            [0, 1], [0], [3, 1], [1]),
            .25)
        assert_allclose(stats.wasserstein_distance(
            [0, 2], [0], [1, 1], [1]),
            1)
        assert_allclose(stats.wasserstein_distance(
            [0, 1, 2], [1, 2, 3]),
            1)

    def test_same_distribution(self):
        # Any distribution moved to itself should have a Wasserstein distance
        # of zero.
        assert_equal(stats.wasserstein_distance([1, 2, 3], [2, 1, 3]), 0)
        assert_equal(
            stats.wasserstein_distance([1, 1, 1, 4], [4, 1],
                                       [1, 1, 1, 1], [1, 3]),
            0)

    def test_shift(self):
        # If the whole distribution is shifted by x, then the Wasserstein
        # distance should be the norm of x.
        assert_allclose(stats.wasserstein_distance([0], [1]), 1)
        assert_allclose(stats.wasserstein_distance([-5], [5]), 10)
        assert_allclose(
            stats.wasserstein_distance([1, 2, 3, 4, 5], [11, 12, 13, 14, 15]),
            10)
        assert_allclose(
            stats.wasserstein_distance([4.5, 6.7, 2.1], [4.6, 7, 9.2],
                                       [3, 1, 1], [1, 3, 1]),
            2.5)

    def test_combine_weights(self):
        # Assigning a weight w to a value is equivalent to including that value
        # w times in the value array with weight of 1.
        assert_allclose(
            stats.wasserstein_distance(
                [0, 0, 1, 1, 1, 1, 5], [0, 3, 3, 3, 3, 4, 4],
                [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]),
            stats.wasserstein_distance([5, 0, 1], [0, 4, 3],
                                       [1, 2, 4], [1, 2, 4]))

    def test_collapse(self):
        # Collapsing a distribution to a point distribution at zero is
        # equivalent to taking the average of the absolute values of the
        # values.
        u = np.arange(-10, 30, 0.3)
        v = np.zeros_like(u)
        assert_allclose(
            stats.wasserstein_distance(u, v),
            np.mean(np.abs(u)))

        u_weights = np.arange(len(u))
        v_weights = u_weights[::-1]
        assert_allclose(
            stats.wasserstein_distance(u, v, u_weights, v_weights),
            np.average(np.abs(u), weights=u_weights))

    def test_zero_weight(self):
        # Values with zero weight have no impact on the Wasserstein distance.
        assert_allclose(
            stats.wasserstein_distance([1, 2, 100000], [1, 1],
                                       [1, 1, 0], [1, 1]),
            stats.wasserstein_distance([1, 2], [1, 1], [1, 1], [1, 1]))



# 测试Wasserstein距离计算的各种情况

class TestWassersteinDistance:
    """对wasserstein_distance()输出值的测试。
    """

    def test_simple(self):
        # 对于基本分布，Wasserstein距离的值是直观的。
        assert_allclose(
            stats.wasserstein_distance([0, 1], [0], [1, 1], [1]),
            .5)
        assert_allclose(stats.wasserstein_distance(
            [0, 1], [0], [3, 1], [1]),
            .25)
        assert_allclose(stats.wasserstein_distance(
            [0, 2], [0], [1, 1], [1]),
            1)
        assert_allclose(stats.wasserstein_distance(
            [0, 1, 2], [1, 2, 3]),
            1)

    def test_same_distribution(self):
        # 任何分布移动到自身的Wasserstein距离应为零。
        assert_equal(stats.wasserstein_distance([1, 2, 3], [2, 1, 3]), 0)
        assert_equal(
            stats.wasserstein_distance([1, 1, 1, 4], [4, 1],
                                       [1, 1, 1, 1], [1, 3]),
            0)

    def test_shift(self):
        # 如果整个分布移动了x，那么Wasserstein距离应为x的范数。
        assert_allclose(stats.wasserstein_distance([0], [1]), 1)
        assert_allclose(stats.wasserstein_distance([-5], [5]), 10)
        assert_allclose(
            stats.wasserstein_distance([1, 2, 3, 4, 5], [11, 12, 13, 14, 15]),
            10)
        assert_allclose(
            stats.wasserstein_distance([4.5, 6.7, 2.1], [4.6, 7, 9.2],
                                       [3, 1, 1], [1, 3, 1]),
            2.5)

    def test_combine_weights(self):
        # 为值分配权重w相当于将该值以权重为1多次包含在值数组中。
        assert_allclose(
            stats.wasserstein_distance(
                [0, 0, 1, 1, 1, 1, 5], [0, 3, 3, 3, 3, 4, 4],
                [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]),
            stats.wasserstein_distance([5, 0, 1], [0, 4, 3],
                                       [1, 2, 4], [1, 2, 4]))

    def test_collapse(self):
        # 将分布折叠为以零为中心的点分布相当于取值的绝对值的平均值。
        u = np.arange(-10, 30, 0.3)
        v = np.zeros_like(u)
        assert_allclose(
            stats.wasserstein_distance(u, v),
            np.mean(np.abs(u)))

        u_weights = np.arange(len(u))
        v_weights = u_weights[::-1]
        assert_allclose(
            stats.wasserstein_distance(u, v, u_weights, v_weights),
            np.average(np.abs(u), weights=u_weights))

    def test_zero_weight(self):
        # 权重为零的值对Wasserstein距离没有影响。
        assert_allclose(
            stats.wasserstein_distance([1, 2, 100000], [1, 1],
                                       [1, 1, 0], [1, 1]),
            stats.wasserstein_distance([1, 2], [1, 1], [1, 1], [1, 1]))
    # 定义一个测试方法，用于测试无穷大值对 Wasserstein 距离的影响
    def test_inf_values(self):
        # Inf values can lead to an inf distance or trigger a RuntimeWarning
        # (and return NaN) if the distance is undefined.
        # 断言：计算两个分布之间的 Wasserstein 距离，其中包含无穷大值，预期结果是无穷大
        assert_equal(
            stats.wasserstein_distance([1, 2, np.inf], [1, 1]),
            np.inf)
        # 断言：计算两个分布之间的 Wasserstein 距离，其中一个分布包含正无穷大，预期结果是无穷大
        assert_equal(
            stats.wasserstein_distance([1, 2, np.inf], [-np.inf, 1]),
            np.inf)
        # 断言：计算两个分布之间的 Wasserstein 距离，其中一个分布包含负无穷大，预期结果是无穷大
        assert_equal(
            stats.wasserstein_distance([1, -np.inf, np.inf], [1, 1]),
            np.inf)
        # 使用上下文管理器抑制特定类型的警告信息
        with suppress_warnings() as sup:
            # 记录 RuntimeWarning 类型的警告信息，信息内容包含 "invalid value*"
            sup.record(RuntimeWarning, "invalid value*")
            # 断言：计算两个分布之间的 Wasserstein 距离，其中一个分布包含无穷大，预期结果是 NaN
            assert_equal(
                stats.wasserstein_distance([1, 2, np.inf], [np.inf, 1]),
                np.nan)
class TestEnergyDistance:
    """ Tests for energy_distance() output values.
    """

    def test_simple(self):
        # For basic distributions, the value of the energy distance is
        # straightforward.
        assert_almost_equal(
            stats.energy_distance([0, 1], [0], [1, 1], [1]),
            np.sqrt(2) * .5)
        assert_almost_equal(stats.energy_distance(
            [0, 1], [0], [3, 1], [1]),
            np.sqrt(2) * .25)
        assert_almost_equal(stats.energy_distance(
            [0, 2], [0], [1, 1], [1]),
            2 * .5)
        assert_almost_equal(
            stats.energy_distance([0, 1, 2], [1, 2, 3]),
            np.sqrt(2) * (3*(1./3**2))**.5)

    def test_same_distribution(self):
        # Any distribution moved to itself should have an energy distance of
        # zero.
        assert_equal(stats.energy_distance([1, 2, 3], [2, 1, 3]), 0)
        assert_equal(
            stats.energy_distance([1, 1, 1, 4], [4, 1], [1, 1, 1, 1], [1, 3]),
            0)

    def test_shift(self):
        # If a single-point distribution is shifted by x, then the energy
        # distance should be sqrt(2) * sqrt(x).
        assert_almost_equal(stats.energy_distance([0], [1]), np.sqrt(2))
        assert_almost_equal(
            stats.energy_distance([-5], [5]),
            np.sqrt(2) * 10**.5)

    def test_combine_weights(self):
        # Assigning a weight w to a value is equivalent to including that value
        # w times in the value array with weight of 1.
        assert_almost_equal(
            stats.energy_distance([0, 0, 1, 1, 1, 1, 5], [0, 3, 3, 3, 3, 4, 4],
                                  [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]),
            stats.energy_distance([5, 0, 1], [0, 4, 3], [1, 2, 4], [1, 2, 4]))

    def test_zero_weight(self):
        # Values with zero weight have no impact on the energy distance.
        assert_almost_equal(
            stats.energy_distance([1, 2, 100000], [1, 1], [1, 1, 0], [1, 1]),
            stats.energy_distance([1, 2], [1, 1], [1, 1], [1, 1]))

    def test_inf_values(self):
        # Inf values can lead to an inf distance or trigger a RuntimeWarning
        # (and return NaN) if the distance is undefined.
        assert_equal(stats.energy_distance([1, 2, np.inf], [1, 1]), np.inf)
        assert_equal(
            stats.energy_distance([1, 2, np.inf], [-np.inf, 1]),
            np.inf)
        assert_equal(
            stats.energy_distance([1, -np.inf, np.inf], [1, 1]),
            np.inf)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, "invalid value*")
            assert_equal(
                stats.energy_distance([1, 2, np.inf], [np.inf, 1]),
                np.nan)


class TestBrunnerMunzel:
    # Data from (Lumley, 1996)
    X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
    Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
    significant = 13


注释：


# 定义了一个测试类 TestEnergyDistance，用于测试 energy_distance() 函数的输出值
class TestEnergyDistance:
    """ Tests for energy_distance() output values.
    """

    def test_simple(self):
        # 对于基本分布，能量距离的值是直观的。
        assert_almost_equal(
            stats.energy_distance([0, 1], [0], [1, 1], [1]),
            np.sqrt(2) * .5)
        assert_almost_equal(stats.energy_distance(
            [0, 1], [0], [3, 1], [1]),
            np.sqrt(2) * .25)
        assert_almost_equal(stats.energy_distance(
            [0, 2], [0], [1, 1], [1]),
            2 * .5)
        assert_almost_equal(
            stats.energy_distance([0, 1, 2], [1, 2, 3]),
            np.sqrt(2) * (3*(1./3**2))**.5)

    def test_same_distribution(self):
        # 任何分布移到其本身应该具有能量距离为零。
        assert_equal(stats.energy_distance([1, 2, 3], [2, 1, 3]), 0)
        assert_equal(
            stats.energy_distance([1, 1, 1, 4], [4, 1], [1, 1, 1, 1], [1, 3]),
            0)

    def test_shift(self):
        # 如果单点分布移动了 x，那么能量距离应为 sqrt(2) * sqrt(x)。
        assert_almost_equal(stats.energy_distance([0], [1]), np.sqrt(2))
        assert_almost_equal(
            stats.energy_distance([-5], [5]),
            np.sqrt(2) * 10**.5)

    def test_combine_weights(self):
        # 将权重 w 赋予一个值，等效于将该值在值数组中以权重 1 包含 w 次。
        assert_almost_equal(
            stats.energy_distance([0, 0, 1, 1, 1, 1, 5], [0, 3, 3, 3, 3, 4, 4],
                                  [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]),
            stats.energy_distance([5, 0, 1], [0, 4, 3], [1, 2, 4], [1, 2, 4]))

    def test_zero_weight(self):
        # 权重为零的值对能量距离没有影响。
        assert_almost_equal(
            stats.energy_distance([1, 2, 100000], [1, 1], [1, 1, 0], [1, 1]),
            stats.energy_distance([1, 2], [1, 1], [1, 1], [1, 1]))

    def test_inf_values(self):
        # 无穷大的值可能导致无穷大的距离，或者如果距离未定义，则触发 RuntimeWarning（并返回 NaN）。
        assert_equal(stats.energy_distance([1, 2, np.inf], [1, 1]), np.inf)
        assert_equal(
            stats.energy_distance([1, 2, np.inf], [-np.inf, 1]),
            np.inf)
        assert_equal(
            stats.energy_distance([1, -np.inf, np.inf], [1, 1]),
            np.inf)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning, "invalid value*")
            assert_equal(
                stats.energy_distance([1, 2, np.inf], [np.inf, 1]),
                np.nan)


class TestBrunnerMunzel:
    # 数据来自 (Lumley, 1996)
    X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
    Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
    significant = 13
    def test_brunnermunzel_one_sided(self):
        # 使用 stats 模块中的 brunnermunzel 函数计算单侧检验的统计量和 p 值，与 R 的 lawstat 包结果进行比较
        u1, p1 = stats.brunnermunzel(self.X, self.Y, alternative='less')
        u2, p2 = stats.brunnermunzel(self.Y, self.X, alternative='greater')
        u3, p3 = stats.brunnermunzel(self.X, self.Y, alternative='greater')
        u4, p4 = stats.brunnermunzel(self.Y, self.X, alternative='less')

        # 断言检验两个单侧检验的 p 值是否近似相等
        assert_approx_equal(p1, p2, significant=self.significant)
        assert_approx_equal(p3, p4, significant=self.significant)
        # 断言检验不同的单侧检验的 p 值不相等
        assert_(p1 != p3)
        # 断言检验单侧检验的统计量是否近似等于特定值
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u3, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u4, -3.1374674823029505,
                            significant=self.significant)
        # 断言检验单侧检验的 p 值是否近似等于特定值
        assert_approx_equal(p1, 0.0028931043330757342,
                            significant=self.significant)
        assert_approx_equal(p3, 0.99710689566692423,
                            significant=self.significant)

    def test_brunnermunzel_two_sided(self):
        # 使用 stats 模块中的 brunnermunzel 函数计算双侧检验的统计量和 p 值，与 R 的 lawstat 包结果进行比较
        u1, p1 = stats.brunnermunzel(self.X, self.Y, alternative='two-sided')
        u2, p2 = stats.brunnermunzel(self.Y, self.X, alternative='two-sided')

        # 断言检验双侧检验的 p 值是否近似相等
        assert_approx_equal(p1, p2, significant=self.significant)
        # 断言检验双侧检验的统计量是否近似等于特定值
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        # 断言检验双侧检验的 p 值是否近似等于特定值
        assert_approx_equal(p1, 0.0057862086661515377,
                            significant=self.significant)

    def test_brunnermunzel_default(self):
        # 使用 stats 模块中的 brunnermunzel 函数默认进行双侧检验，与 R 的 lawstat 包结果进行比较
        u1, p1 = stats.brunnermunzel(self.X, self.Y)
        u2, p2 = stats.brunnermunzel(self.Y, self.X)

        # 断言检验默认双侧检验的 p 值是否近似相等
        assert_approx_equal(p1, p2, significant=self.significant)
        # 断言检验默认双侧检验的统计量是否近似等于特定值
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        # 断言检验默认双侧检验的 p 值是否近似等于特定值
        assert_approx_equal(p1, 0.0057862086661515377,
                            significant=self.significant)

    def test_brunnermunzel_alternative_error(self):
        # 准备参数用于检验非法的 alternative 参数输入是否会引发 ValueError 异常
        alternative = "error"
        distribution = "t"
        nan_policy = "propagate"
        # 断言检验 alternative 参数是否不在合法取值范围内
        assert_(alternative not in ["two-sided", "greater", "less"])
        # 断言检验调用 brunnermunzel 函数时传递非法 alternative 参数是否会引发 ValueError 异常
        assert_raises(ValueError,
                      stats.brunnermunzel,
                      self.X,
                      self.Y,
                      alternative,
                      distribution,
                      nan_policy)
    # 测试布伦纳-门策尔检验在正态分布假设下的情况
    def test_brunnermunzel_distribution_norm(self):
        # 计算布伦纳-门策尔检验的统计量和 p 值，基于正态分布假设
        u1, p1 = stats.brunnermunzel(self.X, self.Y, distribution="normal")
        u2, p2 = stats.brunnermunzel(self.Y, self.X, distribution="normal")
        # 断言两个检验的 p 值近似相等
        assert_approx_equal(p1, p2, significant=self.significant)
        # 断言统计量 u1 的近似值
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        # 断言统计量 u2 的近似值
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        # 断言 p1 的近似值
        assert_approx_equal(p1, 0.0017041417600383024,
                            significant=self.significant)

    # 测试布伦纳-门策尔检验在错误分布假设下引发异常的情况
    def test_brunnermunzel_distribution_error(self):
        alternative = "two-sided"
        distribution = "error"
        nan_policy = "propagate"
        # 断言 alternative 不是 "t" 或 "normal"
        assert_(alternative not in ["t", "normal"])
        # 断言调用布伦纳-门策尔检验时引发 ValueError 异常
        assert_raises(ValueError,
                      stats.brunnermunzel,
                      self.X,
                      self.Y,
                      alternative,
                      distribution,
                      nan_policy)

    # 测试布伦纳-门策尔检验处理空输入的情况
    @pytest.mark.parametrize("kwarg_update", [{'y': []}, {'x': []},
                                              {'x': [], 'y': []}])
    def test_brunnermunzel_empty_imput(self, kwarg_update):
        kwargs = {'x': self.X, 'y': self.Y}
        kwargs.update(kwarg_update)
        # 使用 pytest 的 warn 断言检查 SmallSampleWarning 是否被触发
        with pytest.warns(SmallSampleWarning, match=too_small_1d_not_omit):
            # 执行布伦纳-门策尔检验，并获取统计量和 p 值
            statistic, pvalue = stats.brunnermunzel(**kwargs)
        # 断言统计量为 NaN
        assert_equal(statistic, np.nan)
        # 断言 p 值为 NaN
        assert_equal(pvalue, np.nan)

    # 测试布伦纳-门策尔检验处理 NaN 输入且传播策略的情况
    def test_brunnermunzel_nan_input_propagate(self):
        X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, np.nan]
        Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
        # 计算布伦纳-门策尔检验的统计量和 p 值，传播 NaN 策略
        u1, p1 = stats.brunnermunzel(X, Y, nan_policy="propagate")
        u2, p2 = stats.brunnermunzel(Y, X, nan_policy="propagate")
        # 断言统计量 u1 为 NaN
        assert_equal(u1, np.nan)
        # 断言 p 值 p1 为 NaN
        assert_equal(p1, np.nan)
        # 断言统计量 u2 为 NaN
        assert_equal(u2, np.nan)
        # 断言 p 值 p2 为 NaN
        assert_equal(p2, np.nan)

    # 测试布伦纳-门策尔检验处理 NaN 输入且引发异常的情况
    def test_brunnermunzel_nan_input_raise(self):
        X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, np.nan]
        Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
        alternative = "two-sided"
        distribution = "t"
        nan_policy = "raise"
        # 断言调用布伦纳-门策尔检验时引发 ValueError 异常，传播 NaN 策略
        assert_raises(ValueError,
                      stats.brunnermunzel,
                      X,
                      Y,
                      alternative,
                      distribution,
                      nan_policy)
        # 断言调用布伦纳-门策尔检验时引发 ValueError 异常，传播 NaN 策略
        assert_raises(ValueError,
                      stats.brunnermunzel,
                      Y,
                      X,
                      alternative,
                      distribution,
                      nan_policy)
    # 定义测试函数，用于测试 brunnermunzel 方法在处理包含 NaN 的输入时的行为
    def test_brunnermunzel_nan_input_omit(self):
        # 定义输入数据列表 X，其中包含数字和一个 NaN 值
        X = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, np.nan]
        # 定义输入数据列表 Y
        Y = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
        # 调用 brunnermunzel 方法，忽略 NaN 值，计算统计量 u1 和 p1
        u1, p1 = stats.brunnermunzel(X, Y, nan_policy="omit")
        # 再次调用 brunnermunzel 方法，忽略 NaN 值，计算统计量 u2 和 p2
        u2, p2 = stats.brunnermunzel(Y, X, nan_policy="omit")

        # 断言 p1 和 p2 在指定精度下近似相等
        assert_approx_equal(p1, p2, significant=self.significant)
        # 断言 u1 的值近似为 3.1374674823029505
        assert_approx_equal(u1, 3.1374674823029505,
                            significant=self.significant)
        # 断言 u2 的值近似为 -3.1374674823029505
        assert_approx_equal(u2, -3.1374674823029505,
                            significant=self.significant)
        # 断言 p1 的值近似为 0.0057862086661515377
        assert_approx_equal(p1, 0.0057862086661515377,
                            significant=self.significant)

    # 定义测试函数，验证在使用 t 分布时，当 p 值为 NaN 时会发出警告信息
    def test_brunnermunzel_return_nan(self):
        """ tests that a warning is emitted when p is nan
        p-value with t-distributions can be nan (0/0) (see gh-15843)
        """
        # 定义输入数据列表 x 和 y
        x = [1, 2, 3]
        y = [5, 6, 7, 8, 9]

        # 定义警告信息的正则表达式匹配模式
        msg = "p-value cannot be estimated|divide by zero|invalid value encountered"
        # 使用 pytest 的 warns 上下文，验证是否会发出 RuntimeWarning 警告，并匹配警告信息
        with pytest.warns(RuntimeWarning, match=msg):
            # 调用 brunnermunzel 方法，使用 t 分布计算统计量和 p 值
            stats.brunnermunzel(x, y, distribution="t")

    # 定义测试函数，验证在使用正态分布时，当 p 值为 NaN 时会被设为 0，并发出警告信息
    def test_brunnermunzel_normal_dist(self):
        """ tests that a p is 0 for datasets that cause p->nan
        when t-distribution is used (see gh-15843)
        """
        # 定义输入数据列表 x 和 y
        x = [1, 2, 3]
        y = [5, 6, 7, 8, 9]

        # 使用 warns 上下文，验证是否会发出 RuntimeWarning 警告，并匹配警告信息
        with pytest.warns(RuntimeWarning, match='divide by zero'):
            # 调用 brunnermunzel 方法，使用正态分布计算统计量和 p 值，但 p 值会被设为 0
            _, p = stats.brunnermunzel(x, y, distribution="normal")
        # 断言 p 值等于 0
        assert_equal(p, 0)
class TestQuantileTest:
    r""" Test the non-parametric quantile test,
    including the computation of confidence intervals
    """

    def test_quantile_test_iv(self):
        # 准备一个包含三个数值的列表作为输入数据
        x = [1, 2, 3]

        # 测试输入是否为一维数值数组，若不是则抛出 ValueError 异常
        message = "`x` must be a one-dimensional array of numbers."
        with pytest.raises(ValueError, match=message):
            stats.quantile_test([x])

        # 测试 q 是否为标量，若不是则抛出 ValueError 异常
        message = "`q` must be a scalar."
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, q=[1, 2])

        # 测试 p 是否为 0 到 1 之间的浮点数，若不是则抛出 ValueError 异常
        message = "`p` must be a float strictly between 0 and 1."
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, p=[0.5, 0.75])
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, p=2)
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, p=-0.5)

        # 测试 alternative 是否为合法字符串之一，若不是则抛出 ValueError 异常
        message = "`alternative` must be one of..."
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x, alternative='one-sided')

        # 测试 confidence_level 是否为 0 到 1 之间的数值，若不是则抛出 ValueError 异常
        message = "`confidence_level` must be a number between 0 and 1."
        with pytest.raises(ValueError, match=message):
            stats.quantile_test(x).confidence_interval(1)

    @pytest.mark.parametrize(
        'p, alpha, lb, ub, alternative',
        [[0.3, 0.95, 1.221402758160170, 1.476980793882643, 'two-sided'],
         [0.5, 0.9, 1.506817785112854, 1.803988415397857, 'two-sided'],
         [0.25, 0.95, -np.inf, 1.39096812846378, 'less'],
         [0.8, 0.9, 2.117000016612675, np.inf, 'greater']]
    )
    def test_R_ci_quantile(self, p, alpha, lb, ub, alternative):
        # 对 R 库中的 `confintr` 函数 `ci_quantile` 进行测试
        # 示例代码如下：
        # library(confintr)
        # options(digits=16)
        # x <- exp(seq(0, 1, by = 0.01))
        # ci_quantile(x, q = 0.3)$interval
        # ci_quantile(x, q = 0.5, probs = c(0.05, 0.95))$interval
        # ci_quantile(x, q = 0.25, probs = c(0, 0.95))$interval
        # ci_quantile(x, q = 0.8, probs = c(0.1, 1))$interval
        x = np.exp(np.arange(0, 1.01, 0.01))
        
        # 调用 quantile_test 函数进行测试，并验证置信区间
        res = stats.quantile_test(x, p=p, alternative=alternative)
        assert_allclose(res.confidence_interval(alpha), [lb, ub], rtol=1e-15)

    @pytest.mark.parametrize(
        'q, p, alternative, ref',
        [[1.2, 0.3, 'two-sided', 0.01515567517648],
         [1.8, 0.5, 'two-sided', 0.1109183496606]]
    )
    def test_R_pvalue(self, q, p, alternative, ref):
        # 对 R 库中的 `snpar` 函数 `quant.test` 进行测试
        # 示例代码如下：
        # library(snpar)
        # options(digits=16)
        # x < - exp(seq(0, 1, by=0.01))
        # quant.test(x, q=1.2, p=0.3, exact=TRUE, alternative='t')
        x = np.exp(np.arange(0, 1.01, 0.01))
        
        # 调用 quantile_test 函数进行测试，并验证 p 值
        res = stats.quantile_test(x, q=q, p=p, alternative=alternative)
        assert_allclose(res.pvalue, ref, rtol=1e-12)

    @pytest.mark.parametrize('case', ['continuous', 'discrete'])
    @pytest.mark.parametrize('alternative', ['less', 'greater'])
    # 使用 pytest 的参数化装饰器，定义了多个测试参数 alpha，分别为 0.9 和 0.95
    @pytest.mark.parametrize('alpha', [0.9, 0.95])
    # 定义了一个测试方法 test_pval_ci_match，接受参数 case、alternative 和 alpha
    def test_pval_ci_match(self, case, alternative, alpha):
        # 验证以下语句的正确性：
    
        # 当 alternative='less' 时，95% 置信区间的下界为 -inf，
        # 上界 `xu` 是样本 `x` 中满足以下条件的最大元素：
        # `stats.quantile_test(x, q=xu, p=p, alternative='less').pvalue`
        # 大于 5%。
    
        # 并且对于 alternative='greater' 情况也有相应的语句。
    
        # 根据 case 和 alternative 计算种子值 seed
        seed = int((7**len(case) + len(alternative))*alpha)
        # 使用种子创建一个随机数生成器 rng
        rng = np.random.default_rng(seed)
        
        # 根据 case 类型生成随机变量 rvs 和 p、q 值
        if case == 'continuous':
            p, q = rng.random(size=2)
            rvs = rng.random(size=100)
        else:
            rvs = rng.integers(1, 11, size=100)
            p = rng.random()
            q = rng.integers(1, 11)
        
        # 对 rvs 进行 quantile_test 测试，使用指定的 q、p、alternative 参数
        res = stats.quantile_test(rvs, q=q, p=p, alternative=alternative)
        # 获取其置信区间 ci，置信水平为 alpha
        ci = res.confidence_interval(confidence_level=alpha)
    
        # 根据 alternative 选择落在置信区间内的元素索引
        if alternative == 'less':
            i_inside = rvs <= ci.high
        else:
            i_inside = rvs >= ci.low
        
        # 遍历落在置信区间内的元素，进行 quantile_test 测试，并断言其 p 值大于 1 - alpha
        for x in rvs[i_inside]:
            res = stats.quantile_test(rvs, q=x, p=p, alternative=alternative)
            assert res.pvalue > 1 - alpha
    
        # 遍历落在置信区间外的元素，进行 quantile_test 测试，并断言其 p 值小于 1 - alpha
        for x in rvs[~i_inside]:
            res = stats.quantile_test(rvs, q=x, p=p, alternative=alternative)
            assert res.pvalue < 1 - alpha
    def test_match_conover_examples(self):
        # 测试与 [1]（Conover Practical Nonparametric Statistics Third Edition）第139页中的示例对比

        # Example 1
        # 数据为 [189, 233, 195, 160, 212, 176, 231, 185, 199, 213, 202, 193,
        # 174, 166, 248]
        # 双侧检验是否上四分位数（p=0.75）等于193（q=193）。Conover 表示有7个观察值小于或等于193，
        # "对于二项随机变量 Y，P(Y<=7) = 0.0173"，因此双侧 p 值是其两倍，即0.0346。
        x = [189, 233, 195, 160, 212, 176, 231, 185, 199, 213, 202, 193,
             174, 166, 248]
        pvalue_expected = 0.0346
        res = stats.quantile_test(x, q=193, p=0.75, alternative='two-sided')
        assert_allclose(res.pvalue, pvalue_expected, rtol=1e-5)

        # Example 2
        # Conover 没有给出具体数据，只是说112个观察值中有8个是60或更少。这个测试是检验中位数是否等于60，
        # 备择假设是中位数大于60。p 值被计算为 P(Y<=8)，其中 Y 再次是二项分布随机变量，现在是 p=0.5，n=112。
        # Conover 使用正态近似，但我们可以轻松计算二项分布的累积分布函数。
        x = [59]*8 + [61]*(112-8)
        pvalue_expected = stats.binom.pmf(k=8, p=0.5, n=112)
        res = stats.quantile_test(x, q=60, p=0.5, alternative='greater')
        assert_allclose(res.pvalue, pvalue_expected, atol=1e-10)
class TestPageTrendTest:
    # 定义一个测试类 TestPageTrendTest

    # expected statistic and p-values generated using R at
    # https://rdrr.io/cran/cultevo/, e.g.
    # library(cultevo)
    # data = rbind(c(72, 47, 73, 35, 47, 96, 30, 59, 41, 36, 56, 49, 81, 43,
    #                   70, 47, 28, 28, 62, 20, 61, 20, 80, 24, 50),
    #              c(68, 52, 60, 34, 44, 20, 65, 88, 21, 81, 48, 31, 31, 67,
    #                69, 94, 30, 24, 40, 87, 70, 43, 50, 96, 43),
    #              c(81, 13, 85, 35, 79, 12, 92, 86, 21, 64, 16, 64, 68, 17,
    #                16, 89, 71, 43, 43, 36, 54, 13, 66, 51, 55))
    # 在 R 中生成的预期统计量和 p 值，用于参考和验证

    # result = page.test(data, verbose=FALSE)
    # 使用 page.test 函数对数据进行分析，verbose 参数设为 FALSE

    # Most test cases generated to achieve common critical p-values so that
    # results could be checked (to limited precision) against tables in
    # scipy.stats.page_trend_test reference [1]
    # 大多数测试案例生成以达到常见的关键 p 值，以便可以将结果（以有限的精度）与 scipy.stats.page_trend_test 参考表中的结果进行比较

    np.random.seed(0)
    # 使用种子值 0 初始化随机数生成器

    data_3_25 = np.random.rand(3, 25)
    # 生成一个 3x25 的随机数数组，元素值在 [0, 1) 范围内

    data_10_26 = np.random.rand(10, 26)
    # 生成一个 10x26 的随机数数组，元素值在 [0, 1) 范围内
    # 定义测试用例参数列表，包含多个元组，每个元组表示一个测试用例的参数
    ts = [
          (12805, 0.3886487053947608, False, 'asymptotic', data_3_25),
          (49140, 0.02888978556179862, False, 'asymptotic', data_10_26),
          (12332, 0.7722477197436702, False, 'asymptotic',
           [[72, 47, 73, 35, 47, 96, 30, 59, 41, 36, 56, 49, 81,
             43, 70, 47, 28, 28, 62, 20, 61, 20, 80, 24, 50],
            [68, 52, 60, 34, 44, 20, 65, 88, 21, 81, 48, 31, 31,
             67, 69, 94, 30, 24, 40, 87, 70, 43, 50, 96, 43],
            [81, 13, 85, 35, 79, 12, 92, 86, 21, 64, 16, 64, 68,
             17, 16, 89, 71, 43, 43, 36, 54, 13, 66, 51, 55]]),
          (266, 4.121656378600823e-05, False, 'exact',
           [[1.5, 4., 8.3, 5, 19, 11],
            [5, 4, 3.5, 10, 20, 21],
            [8.4, 3.2, 10, 12, 14, 15]]),
          (332, 0.9566400920502488, True, 'exact',
           [[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1],
            [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1],
            [3, 4, 1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
            [1, 2, 3, 4], [1, 2, 3, 4]]),
          (241, 0.9622210164861476, True, 'exact',
           [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [3, 2, 1], [2, 1, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],
            [1, 2, 3], [1, 2, 3], [1, 2, 3]]),
          (197, 0.9619432897162209, True, 'exact',
           [[6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1], [1, 3, 4, 5, 2, 6]]),
          (423, 0.9590458306880073, True, 'exact',
           [[5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1],
            [4, 1, 3, 2, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]]),
          (217, 0.9693058575034678, True, 'exact',
           [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [2, 1, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],
            [1, 2, 3]]),
          (395, 0.991530289351305, True, 'exact',
           [[7, 6, 5, 4, 3, 2, 1], [7, 6, 5, 4, 3, 2, 1],
            [6, 5, 7, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6, 7]]),
          (117, 0.9997817843373017, True, 'exact',
           [[3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1], [3, 2, 1],
            [3, 2, 1], [3, 2, 1], [3, 2, 1], [2, 1, 3], [1, 2, 3]]),
         ]
    
    # 使用 pytest 的参数化测试装饰器，定义测试方法 test_accuracy，接受参数 L, p, ranked, method, data
    @pytest.mark.parametrize("L, p, ranked, method, data", ts)
    def test_accuracy(self, L, p, ranked, method, data):
        # 设置随机种子为42
        np.random.seed(42)
        # 调用 stats.page_trend_test 函数，传入数据 data，并指定 ranked 和 method 参数
        res = stats.page_trend_test(data, ranked=ranked, method=method)
        # 断言检查统计量 L 是否等于 res.statistic
        assert_equal(L, res.statistic)
        # 断言检查 p 值是否接近于 res.pvalue
        assert_allclose(p, res.pvalue)
        # 断言检查 method 是否等于 res.method
        assert_equal(method, res.method)
    ts2 = [
           (542, 0.9481266260876332, True, 'exact',
            [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
             [1, 8, 4, 7, 6, 5, 9, 3, 2, 10]]),
           (1322, 0.9993113928199309, True, 'exact',
            [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [9, 2, 8, 7, 6, 5, 4, 3, 10, 1],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
           (2286, 0.9908688345484833, True, 'exact',
            [[8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1],
             [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1],
             [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1],
             [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1],
             [8, 7, 6, 5, 4, 3, 2, 1], [1, 3, 5, 6, 4, 7, 2, 8],
             [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
             [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
             [1, 2, 3, 4, 5, 6, 7, 8]]),
          ]

    # 定义一组参数化测试数据，每个元组包含统计量 L, p 值 p, 布尔值 ranked, 字符串 method, 二维列表 data
    @pytest.mark.parametrize("L, p, ranked, method, data", ts)
    # 标记为慢速测试
    @pytest.mark.slow()
    def test_accuracy2(self, L, p, ranked, method, data):
        # 设定随机种子
        np.random.seed(42)
        # 调用统计模块中的页面趋势检验函数，传入数据和相关参数
        res = stats.page_trend_test(data, ranked=ranked, method=method)
        # 断言检验结果的统计量等于预期的 L 值
        assert_equal(L, res.statistic)
        # 断言检验结果的 p 值近似等于预期的 p 值
        assert_allclose(p, res.pvalue)
        # 断言检验结果的方法等于预期的方法
        assert_equal(method, res.method)

    # 定义测试选项函数
    def test_options(self):
        # 设定随机种子
        np.random.seed(42)
        # 定义数据维度 m, n
        m, n = 10, 20
        # 预测的排名从 1 到 n
        predicted_ranks = np.arange(1, n+1)
        # 对排列 n 的随机排列
        perm = np.random.permutation(np.arange(n))
        # 生成 m 行 n 列的随机数据
        data = np.random.rand(m, n)
        # 计算每行数据的排名
        ranks = stats.rankdata(data, axis=1)
        # 不同的参数化调用页面趋势检验函数，并获取返回结果
        res1 = stats.page_trend_test(ranks)
        res2 = stats.page_trend_test(ranks, ranked=True)
        res3 = stats.page_trend_test(data, ranked=False)
        res4 = stats.page_trend_test(ranks, predicted_ranks=predicted_ranks)
        res5 = stats.page_trend_test(ranks[:, perm],
                                     predicted_ranks=predicted_ranks[perm])
        # 断言不同条件下的统计量结果应相等
        assert_equal(res1.statistic, res2.statistic)
        assert_equal(res1.statistic, res3.statistic)
        assert_equal(res1.statistic, res4.statistic)
        assert_equal(res1.statistic, res5.statistic)
    # 定义测试方法，用于测试页面趋势检验函数
    def test_Ames_assay(self):
        # 设置随机种子为42，以确保结果可复现性
        np.random.seed(42)

        # 创建包含观测数据的二维列表
        data = [[101, 117, 111], [91, 90, 107], [103, 133, 121],
                [136, 140, 144], [190, 161, 201], [146, 120, 116]]
        
        # 将二维列表转换为 NumPy 数组，并进行转置
        data = np.array(data).T
        
        # 创建预测排名数组，范围从1到6
        predicted_ranks = np.arange(1, 7)

        # 调用页面趋势检验函数，使用非排名数据进行检验，选择渐近法作为方法
        res = stats.page_trend_test(data, ranked=False,
                                    predicted_ranks=predicted_ranks,
                                    method="asymptotic")
        
        # 断言返回结果的统计量为257
        assert_equal(res.statistic, 257)
        
        # 断言返回结果的 p 值接近0.0035，精确到小数点后四位
        assert_almost_equal(res.pvalue, 0.0035, decimal=4)

        # 再次调用页面趋势检验函数，使用非排名数据进行检验，选择精确法作为方法
        res = stats.page_trend_test(data, ranked=False,
                                    predicted_ranks=predicted_ranks,
                                    method="exact")
        
        # 断言返回结果的统计量为257
        assert_equal(res.statistic, 257)
        
        # 断言返回结果的 p 值接近0.0023，精确到小数点后四位
        assert_almost_equal(res.pvalue, 0.0023, decimal=4)
    
# 使用指定的种子创建一个随机数生成器对象
rng = np.random.default_rng(902340982)
# 使用生成的随机数生成器对象生成长度为10的随机数组
x = rng.random(10)
# 使用相同的随机数生成器对象生成长度为10的另一个随机数组
y = rng.random(10)

# 使用pytest的@parametrize装饰器，为test_rename_mode_method方法参数化多个测试情况
@pytest.mark.parametrize("fun, args",
                         [(stats.wilcoxon, (x,)),  # 使用wilcoxon函数进行参数化测试
                          (stats.ks_1samp, (x, stats.norm.cdf)),  # 使用ks_1samp函数进行参数化测试，指定cdf为正态分布函数
                          (stats.ks_2samp, (x, y)),  # 使用ks_2samp函数进行参数化测试
                          (stats.kstest, (x, y)),  # 使用kstest函数进行参数化测试
                          ])
def test_rename_mode_method(fun, args):
    # 调用被参数化的函数并指定method='exact'，获取结果res和res2
    res = fun(*args, method='exact')
    res2 = fun(*args, mode='exact')
    # 断言两次调用结果相等
    assert_equal(res, res2)

    # 准备一个预期的TypeError异常消息，用于断言函数调用时会抛出此异常
    err = rf"{fun.__name__}() got multiple values for argument"
    # 使用pytest.raises断言调用函数时会抛出TypeError异常，且异常消息符合预期
    with pytest.raises(TypeError, match=re.escape(err)):
        fun(*args, method='exact', mode='exact')

# 定义TestExpectile测试类
class TestExpectile:
    # 定义测试方法，验证expectile函数的结果与np.mean(x)相等
    def test_same_as_mean(self):
        # 使用指定种子创建随机数生成器对象rng，生成长度为20的随机数组x
        rng = np.random.default_rng(42)
        x = rng.random(size=20)
        # 断言expectile函数计算结果与np.mean(x)的结果在数值上接近
        assert_allclose(stats.expectile(x, alpha=0.5), np.mean(x))

    # 定义测试方法，验证expectile函数的结果与np.amin(x)相等
    def test_minimum(self):
        # 使用指定种子创建随机数生成器对象rng，生成长度为20的随机数组x
        rng = np.random.default_rng(42)
        x = rng.random(size=20)
        # 断言expectile函数计算结果与np.amin(x)的结果在数值上接近
        assert_allclose(stats.expectile(x, alpha=0), np.amin(x))

    # 定义测试方法，验证expectile函数的结果与np.amax(x)相等
    def test_maximum(self):
        # 使用指定种子创建随机数生成器对象rng，生成长度为20的随机数组x
        rng = np.random.default_rng(42)
        x = rng.random(size=20)
        # 断言expectile函数计算结果与np.amax(x)的结果在数值上接近
        assert_allclose(stats.expectile(x, alpha=1), np.amax(x))

    # 定义测试方法，验证expectile函数对给定参数的期望值计算结果
    def test_weights(self):
        # 定义内部函数fun，用于expectile函数的最小化计算
        def fun(u, a, alpha, weights):
            w = np.full_like(a, fill_value=alpha)
            w[a <= u] = 1 - alpha
            return np.sum(w * weights * (a - u)**2)

        # 定义内部函数expectile2，使用optimize.minimize_scalar对a进行expectile计算
        def expectile2(a, alpha, weights):
            bracket = np.min(a), np.max(a)
            return optimize.minimize_scalar(fun, bracket=bracket,
                                            args=(a, alpha, weights)).x

        # 使用指定种子创建随机数生成器对象rng
        rng = np.random.default_rng(1856392524598679138)
        n = 10
        # 生成长度为n的随机数组a、随机数alpha和长度为n的随机权重数组weights
        a = rng.random(n)
        alpha = rng.random()
        weights = rng.random(n)

        # 调用stats.expectile函数计算结果
        res = stats.expectile(a, alpha, weights=weights)
        # 调用expectile2函数计算期望值的参考结果
        ref = expectile2(a, alpha, weights)
        # 断言stats.expectile计算结果与expectile2的参考结果在数值上接近
        assert_allclose(res, ref)

    # 使用pytest的@parametrize装饰器，为test_monotonicity_in_alpha方法参数化多个测试情况
    @pytest.mark.parametrize(
        "alpha", [0.2, 0.5 - 1e-12, 0.5, 0.5 + 1e-12, 0.8]
    )
    @pytest.mark.parametrize("n", [20, 2000])
    def test_monotonicity_in_alpha(self, n):
        # 使用指定种子创建随机数生成器对象rng
        rng = np.random.default_rng(42)
        # 生成服从参数为2的帕累托分布的长度为n的随机数组x
        x = rng.pareto(a=2, size=n)
        e_list = []
        # 生成一组以logspace间隔的alpha序列
        alpha_seq = np.logspace(-15, np.log10(0.5), 100)
        # 向e_list中添加一组排序后的唯一alpha值，以检查expectile函数的alpha单调性
        for alpha in np.r_[0, alpha_seq, 1 - alpha_seq[:-1:-1], 1]:
            e_list.append(stats.expectile(x, alpha=alpha))
        # 断言e_list中相邻元素之差大于0，即验证expectile函数在alpha上的单调性
        assert np.all(np.diff(e_list) > 0)

# 定义TestXP_Mean测试类，并标记为array_api_compatible
@array_api_compatible
class TestXP_Mean:
    # 使用pytest的@parametrize装饰器，为axis参数和weights参数进行参数化测试
    @pytest.mark.parametrize('axis', [None, 1, -1, (-2, 2)])
    @pytest.mark.parametrize('weights', [None, True])
    # 使用 pytest 的 parametrize 装饰器，为 test_xp_mean_basic 方法添加多个参数化的测试用例
    @pytest.mark.parametrize('keepdims', [False, True])
    # 定义了一个测试方法 test_xp_mean_basic，接受 xp、axis、weights 和 keepdims 参数
    def test_xp_mean_basic(self, xp, axis, weights, keepdims):
        # 创建一个随机数生成器 rng
        rng = np.random.default_rng(90359458245906)
        # 生成一个随机数组 x，形状为 (3, 4, 5)
        x = rng.random((3, 4, 5))
        # 将 x 转换为 xp 数组
        x_xp = xp.asarray(x)
        # 初始化权重 w 和 w_xp
        w = w_xp = None

        # 如果参数 weights 为真
        if weights:
            # 生成一个形状为 (1, 5) 的随机数组 w
            w = rng.random((1, 5))
            # 将 w 转换为 xp 数组
            w_xp = xp.asarray(w)
            # 广播 x 和 w，使其具有相同的形状
            x, w = np.broadcast_arrays(x, w)

        # 调用 _xp_mean 函数计算结果 res
        res = _xp_mean(x_xp, weights=w_xp, axis=axis, keepdims=keepdims)
        # 调用 numpy 的 average 函数计算参考结果 ref
        ref = np.average(x, weights=w, axis=axis, keepdims=keepdims)

        # 断言 res 和 ref 接近
        xp_assert_close(res, xp.asarray(ref))

    # 定义测试方法 test_non_broadcastable，测试非可广播的情况
    def test_non_broadcastable(self, xp):
        # 创建一个 xp 数组 x，范围是 0 到 9
        x, w = xp.arange(10.), xp.zeros(5)
        # 设置错误信息字符串
        message = "Array shapes are incompatible for broadcasting."
        # 使用 pytest.raises 断言引发 ValueError 异常，且异常消息符合指定的正则表达式
        with pytest.raises(ValueError, match=message):
            # 调用 _xp_mean 函数，传入 x 和 w 作为参数
            _xp_mean(x, weights=w)

    # 定义测试方法 test_special_cases，测试特殊情况
    def test_special_cases(self, xp):
        # 创建一个 xp 数组 weights，包含元素 [-1., 0., 1.]
        weights = xp.asarray([-1., 0., 1.])

        # 调用 _xp_mean 函数计算结果 res，输入为 [1., 1., 1.] 和 weights
        res = _xp_mean(xp.asarray([1., 1., 1.]), weights=weights)
        # 断言 res 接近 xp 数组中的 NaN
        xp_assert_close(res, xp.asarray(xp.nan))

        # 调用 _xp_mean 函数计算结果 res，输入为 [2., 1., 1.] 和 weights
        res = _xp_mean(xp.asarray([2., 1., 1.]), weights=weights)
        # 断言 res 接近 xp 数组中的负无穷大
        xp_assert_close(res, xp.asarray(-np.inf))

        # 调用 _xp_mean 函数计算结果 res，输入为 [1., 1., 2.] 和 weights
        res = _xp_mean(xp.asarray([1., 1., 2.]), weights=weights)
        # 断言 res 接近 xp 数组中的正无穷大
        xp_assert_close(res, xp.asarray(np.inf))

    # 定义测试方法 test_nan_policy，测试 NaN 策略
    def test_nan_policy(self, xp):
        # 创建一个 xp 数组 x，范围是 0 到 9
        x = xp.arange(10.)
        # 创建一个掩码 mask，标记 x 中等于 3 的位置为 True，其余为 False
        mask = (x == 3)
        # 在 x 中，将 mask 为 True 的位置设置为 NaN
        x = xp.where(mask, xp.asarray(xp.nan), x)

        # 设置错误信息字符串
        message = 'The input contains nan values'
        # 使用 pytest.raises 断言引发 ValueError 异常，且异常消息符合指定的正则表达式
        with pytest.raises(ValueError, match=message):
            # 调用 _xp_mean 函数，传入 x 和 nan_policy='raise' 作为参数
            _xp_mean(x, nan_policy='raise')

        # 使用默认的 nan_policy='propagate'，调用 _xp_mean 函数计算结果 res1 和 res2
        res1 = _xp_mean(x)
        res2 = _xp_mean(x, nan_policy='propagate')
        # 创建一个 xp 数组 ref，包含 NaN 值
        ref = xp.asarray(xp.nan)
        # 断言 res1 和 res2 等于 ref
        xp_assert_equal(res1, ref)
        xp_assert_equal(res2, ref)

        # 使用 nan_policy='omit'，调用 _xp_mean 函数计算结果 res
        res = _xp_mean(x, nan_policy='omit')
        # 计算 x 中非 NaN 值的均值 ref
        ref = xp.mean(x[~mask])
        # 断言 res 接近 ref
        xp_assert_close(res, ref)

        # 在 weights 中使用 nan_policy='omit'，调用 _xp_mean 函数计算结果 res
        weights = xp.ones(10)
        # 将 weights 中 mask 为 True 的位置设置为 NaN
        weights = xp.where(mask, xp.asarray(xp.nan), weights)
        # 调用 _xp_mean 函数，传入 arange(10.)、weights 和 nan_policy='omit' 作为参数
        res = _xp_mean(xp.arange(10.), weights=weights, nan_policy='omit')
        # 断言 res 接近 ref
        xp_assert_close(res, ref)

        # 检查是否会因忽略 NaN 值导致空切片而引发警告
        message = 'After omitting NaNs...'
        # 使用 pytest.warns 断言会引发 RuntimeWarning 警告，且警告消息符合指定的正则表达式
        with pytest.warns(RuntimeWarning, match=message):
            # 调用 _xp_mean 函数，传入 x * np.nan 和 nan_policy='omit' 作为参数
            res = _xp_mean(x * np.nan, nan_policy='omit')
            # 断言 res 等于 ref
            xp_assert_equal(res, ref)
    # 定义一个测试方法，测试空数组输入情况
    def test_empty(self, xp):
        # 设置警告信息，用于匹配触发的警告类和消息内容
        message = 'One or more sample arguments is too small...'
        # 断言触发特定警告并捕获结果，进行相应测试
        with pytest.warns(SmallSampleWarning, match=message):
            # 调用被测试函数 `_xp_mean` 处理空数组
            res = _xp_mean(xp.asarray([]))
            # 参考结果使用 NaN 构成的数组
            ref = xp.asarray(xp.nan)
            # 断言结果与参考结果相等
            xp_assert_equal(res, ref)

        # 设置另一个警告信息，用于匹配触发的警告类和消息内容
        message = "All axis-slices of one or more sample arguments..."
        # 断言触发特定警告并捕获结果，进行相应测试
        with pytest.warns(SmallSampleWarning, match=message):
            # 调用被测试函数 `_xp_mean` 处理空二维数组
            res = _xp_mean(xp.asarray([[]]), axis=1)
            # 参考结果使用包含 NaN 的数组
            ref = xp.asarray([xp.nan])
            # 断言结果与参考结果相等
            xp_assert_equal(res, ref)

        # 调用被测试函数 `_xp_mean` 处理空列数组
        res = _xp_mean(xp.asarray([[]]), axis=0)
        # 参考结果是空数组
        ref = xp.asarray([])
        # 断言结果与参考结果相等
        xp_assert_equal(res, ref)

    # 定义一个测试方法，测试数据类型转换情况
    def test_dtype(self, xp):
        # 获取浮点数类型的最大值
        max = xp.finfo(xp.float32).max
        # 创建一个 NumPy 数组，使用浮点32位类型
        x_np = np.asarray([max, max], dtype=np.float32)
        # 将 NumPy 数组转换为对应的测试框架数组
        x_xp = xp.asarray(x_np)

        # 在溢出情况下忽略浮点错误
        with np.errstate(over='ignore'):
            # 调用被测试函数 `_xp_mean` 处理浮点32位类型数组
            res = _xp_mean(x_xp)
            # 计算 NumPy 数组的均值作为参考结果
            ref = np.mean(x_np)
            # 使用 NumPy 测试工具断言结果与参考结果相等
            np.testing.assert_equal(ref, np.inf)
            # 使用自定义的断言函数检查测试框架结果与参考结果的接近程度
            xp_assert_close(res, xp.asarray(ref))

        # 使用浮点64位类型作为数据类型参数调用被测试函数 `_xp_mean`
        res = _xp_mean(x_xp, dtype=xp.float64)
        # 计算浮点64位类型数据的均值作为参考结果
        ref = xp.asarray(np.mean(np.asarray(x_np, dtype=np.float64)))
        # 使用自定义的断言函数检查测试框架结果与参考结果的接近程度
        xp_assert_close(res, ref)

    # 定义一个测试方法，测试整数转换情况
    def test_integer(self, xp):
        # 创建测试框架中的整数数组
        x = xp.arange(10)
        # 创建测试框架中的浮点数数组
        y = xp.arange(10.)
        # 断言处理整数数组和对应浮点数数组的均值相等
        xp_assert_equal(_xp_mean(x), _xp_mean(y))
        # 断言处理带权重的浮点数数组与自身的均值相等
        xp_assert_equal(_xp_mean(y, weights=x), _xp_mean(y, weights=y))
@array_api_compatible
# 将该类标记为兼容数组 API，用于不同的数组后端
@pytest.mark.usefixtures("skip_xp_backends")
# 使用 pytest 的 fixture 跳过特定的 XP 后端
@skip_xp_backends('jax.numpy', reasons=['JAX arrays do not support item assignment'])
# 如果是 JAX 后端，则跳过，因为它不支持项目赋值

class TestXP_Var:
    # 测试 XP_Var 类

    @pytest.mark.parametrize('axis', [None, 1, -1, (-2, 2)])
    # 参数化测试，测试不同的轴值
    @pytest.mark.parametrize('keepdims', [False, True])
    # 参数化测试，测试不同的 keepdims 值
    @pytest.mark.parametrize('correction', [0, 1])
    # 参数化测试，测试不同的 correction 值
    @pytest.mark.parametrize('nan_policy', ['propagate', 'omit'])
    # 参数化测试，测试不同的 nan_policy 值

    def test_xp_var_basic(self, xp, axis, keepdims, correction, nan_policy):
        # 基本的 XP 变量测试函数

        rng = np.random.default_rng(90359458245906)
        # 创建一个随机数生成器对象

        x = rng.random((3, 4, 5))
        # 生成一个 3x4x5 的随机数组 x

        var_ref = np.var
        # 将 numpy 的 var 函数赋给 var_ref

        if nan_policy == 'omit':
            # 如果 nan_policy 是 'omit'

            nan_mask = rng.random(size=x.shape) > 0.5
            # 创建一个与 x 形状相同的随机布尔掩码，用于确定哪些元素设置为 NaN
            x[nan_mask] = np.nan
            # 将 x 中根据掩码为 True 的位置设置为 NaN
            var_ref = np.nanvar
            # 将 var_ref 更新为 numpy 的 nanvar 函数

        x_xp = xp.asarray(x)
        # 将 x 转换为特定 XP 后端的数组表示形式

        res = _xp_var(x_xp, axis=axis, keepdims=keepdims, correction=correction,
                      nan_policy=nan_policy)
        # 调用 _xp_var 函数计算 XP 后端的方差

        with suppress_warnings() as sup:
            # 屏蔽特定类型的警告
            sup.filter(RuntimeWarning, "Degrees of freedom <= 0 for slice")
            # 设置过滤器以忽略特定的运行时警告消息

            ref = var_ref(x, axis=axis, keepdims=keepdims, ddof=correction)
            # 计算参考值，使用 var_ref 函数计算方差

        xp_assert_close(res, xp.asarray(ref))
        # 使用 xp_assert_close 函数断言 res 与 ref 在 XP 后端上的数组表示形式相等

    def test_special_cases(self, xp):
        # 特殊情况测试函数

        # correction 太大时的情况
        res = _xp_var(xp.asarray([1., 2.]), correction=3)
        # 计算 XP 后端数组 [1., 2.] 的方差，使用 ddof=3

        xp_assert_close(res, xp.asarray(xp.nan))
        # 使用 xp_assert_close 函数断言 res 是 XP 后端数组中的 NaN 值

    def test_nan_policy(self, xp):
        # NaN 策略测试函数

        x = xp.arange(10.)
        # 创建一个 XP 后端数组 x，包含从 0 到 9 的浮点数

        mask = (x == 3)
        # 创建一个掩码，标记 x 中值为 3 的位置

        x = xp.where(mask, xp.asarray(xp.nan), x)
        # 将 x 中值为 3 的位置设置为 NaN

        # nan_policy='raise' 会引发错误
        message = 'The input contains nan values'
        # 设置错误消息内容
        with pytest.raises(ValueError, match=message):
            _xp_var(x, nan_policy='raise')
            # 调用 _xp_var 函数，期望引发 ValueError 错误，且错误消息匹配特定的 message

        # `nan_policy='propagate'` 是默认值，结果为 NaN
        res1 = _xp_var(x)
        # 计算 XP 后端数组 x 的方差，使用默认的 nan_policy='propagate'
        res2 = _xp_var(x, nan_policy='propagate')
        # 再次计算 XP 后端数组 x 的方差，指定 nan_policy='propagate'

        ref = xp.asarray(xp.nan)
        # 创建一个 XP 后端数组表示 NaN

        xp_assert_equal(res1, ref)
        # 使用 xp_assert_equal 函数断言 res1 与 ref 在 XP 后端上的数组表示形式相等
        xp_assert_equal(res2, ref)
        # 使用 xp_assert_equal 函数断言 res2 与 ref 在 XP 后端上的数组表示形式相等

        # `nan_policy='omit'` 会忽略 x 中的 NaN 值
        res = _xp_var(x, nan_policy='omit')
        # 计算 XP 后端数组 x 的方差，使用 nan_policy='omit'

        xp_test = array_namespace(x)  # torch has different default correction
        # 获取特定 XP 后端的数组命名空间，用于确定默认的 correction

        ref = xp_test.var(x[~mask])
        # 计算 XP 后端数组 x 中未被掩码标记的部分的方差

        xp_assert_close(res, ref)
        # 使用 xp_assert_close 函数断言 res 与 ref 在 XP 后端上的数组表示形式相近

        # 检查是否因忽略 NaN 值导致空切片而引发警告
        message = 'After omitting NaNs...'
        # 设置警告消息内容
        with pytest.warns(RuntimeWarning, match=message):
            # 期望引发 RuntimeWarning 警告，且警告消息匹配特定的 message
            res = _xp_var(x * np.nan, nan_policy='omit')
            # 调用 _xp_var 函数计算 XP 后端数组 x 乘以 np.nan 的方差，使用 nan_policy='omit'

            ref = xp.asarray(xp.nan)
            # 创建一个 XP 后端数组表示 NaN

            xp_assert_equal(res, ref)
            # 使用 xp_assert_equal 函数断言 res 与 ref 在 XP 后端上的数组表示形式相等
    # 测试空数组作为输入的情况
    def test_empty(self, xp):
        # 定义警告消息，用于匹配的警告信息
        message = 'One or more sample arguments is too small...'
        # 捕获特定警告类型，并检查是否出现指定的消息
        with pytest.warns(SmallSampleWarning, match=message):
            # 调用被测试函数 _xp_var，传入空的数组作为参数
            res = _xp_var(xp.asarray([]))
            # 参考结果是一个包含 NaN 的数组
            ref = xp.asarray(xp.nan)
            # 使用自定义的断言函数检查结果与参考值是否相等
            xp_assert_equal(res, ref)

        # 定义另一个警告消息，用于匹配的警告信息
        message = "All axis-slices of one or more sample arguments..."
        # 捕获特定警告类型，并检查是否出现指定的消息
        with pytest.warns(SmallSampleWarning, match=message):
            # 调用被测试函数 _xp_var，传入空的二维数组及指定轴向参数
            res = _xp_var(xp.asarray([[]]), axis=1)
            # 参考结果是一个包含 NaN 的数组
            ref = xp.asarray([xp.nan])
            # 使用自定义的断言函数检查结果与参考值是否相等
            xp_assert_equal(res, ref)

        # 调用被测试函数 _xp_var，传入空的二维数组及默认轴向参数
        res = _xp_var(xp.asarray([[]]), axis=0)
        # 参考结果是一个空的数组
        ref = xp.asarray([])
        # 使用自定义的断言函数检查结果与参考值是否相等
        xp_assert_equal(res, ref)

    # 测试不同数据类型的情况
    def test_dtype(self, xp):
        # 获取指定类型的最大值
        max = xp.finfo(xp.float32).max
        # 使用 NumPy 创建一个数组，并指定数据类型为 float32
        x_np = np.asarray([max, max], dtype=np.float32)
        # 将 NumPy 数组转换为指定的 xp 数组
        x_xp = xp.asarray(x_np)

        # 当输入为 float32 类型时，可能会发生溢出
        with np.errstate(over='ignore'):
            # 调用被测试函数 _xp_var，计算结果
            res = _xp_var(x_xp)
            # 计算 NumPy 中相同输入的方差作为参考结果
            ref = np.var(x_np)
            # 使用 NumPy 提供的断言函数检查结果与参考值是否相等
            np.testing.assert_equal(ref, np.inf)
            # 使用自定义的断言函数检查结果与参考值在精度上的接近程度
            xp_assert_close(res, xp.asarray(ref))

        # 当使用 float64 类型时，应该返回正确的结果
        res = _xp_var(x_xp, dtype=xp.float64)
        # 计算 NumPy 中相同输入的方差，并将结果转换为 xp 数组
        ref = xp.asarray(np.var(np.asarray(x_np, dtype=np.float64)))
        # 使用自定义的断言函数检查结果与参考值是否相等
        xp_assert_close(res, ref)

    # 测试整数类型输入的情况
    def test_integer(self, xp):
        # 使用 xp.arange 创建一个整数类型的数组
        x = xp.arange(10)
        # 使用 xp.arange 创建一个浮点类型的数组
        y = xp.arange(10.)
        # 使用自定义的断言函数检查整数类型和浮点类型输入下的方差结果是否相等
        xp_assert_equal(_xp_var(x), _xp_var(y))
# 将函数标记为与数组API兼容的测试函数
@array_api_compatible
def test_chk_asarray(xp):
    # 使用指定种子创建随机数生成器对象
    rng = np.random.default_rng(2348923425434)
    # 生成一个形状为(2, 3, 4)的随机数组
    x0 = rng.random(size=(2, 3, 4))
    # 将数组x0转换为xp数组（根据数组API不同，可能是NumPy数组或其他）
    x = xp.asarray(x0)

    # 设定轴参数为1
    axis = 1
    # 调用_chk_asarray函数，将x和axis作为参数，返回转换后的数组和轴
    x_out, axis_out = _chk_asarray(x, axis=axis, xp=xp)
    # 断言转换后的数组x_out与x0相等（按xp的数组类型进行比较）
    xp_assert_equal(x_out, xp.asarray(x0))
    # 断言axis_out与设定的轴参数axis相等
    assert_equal(axis_out, axis)

    # 设定轴参数为None
    axis = None
    # 再次调用_chk_asarray函数，此时axis为None，将x的ravel结果作为参数传递
    x_out, axis_out = _chk_asarray(x, axis=axis, xp=xp)
    # 断言转换后的数组x_out与x0的ravel结果相等
    xp_assert_equal(x_out, xp.asarray(x0.ravel()))
    # 断言axis_out为0，因为当axis为None时，_chk_asarray应将其视为0
    assert_equal(axis_out, 0)

    # 设定轴参数为2
    axis = 2
    # 以x[0, 0, 0]作为参数调用_chk_asarray函数，axis为2
    x_out, axis_out = _chk_asarray(x[0, 0, 0], axis=axis, xp=xp)
    # 断言转换后的数组x_out与x0[0, 0, 0]的至少一维数组结果相等
    xp_assert_equal(x_out, xp.asarray(np.atleast_1d(x0[0, 0, 0])))
    # 断言axis_out与设定的轴参数axis相等
    assert_equal(axis_out, axis)


# 标记为跳过某些数组API后端的测试函数，原因是这些参数与NumPy兼容
@pytest.mark.skip_xp_backends('numpy',
                              reasons=['These parameters *are* compatible with NumPy'])
# 使用fixture跳过特定数组API后端的测试
@pytest.mark.usefixtures("skip_xp_backends")
# 将函数标记为与数组API兼容的测试函数
@array_api_compatible
def test_axis_nan_policy_keepdims_nanpolicy(xp):
    # 创建一个xp数组，包含元素[1, 2, 3, 4]
    x = xp.asarray([1, 2, 3, 4])
    # 指定错误消息字符串
    message = "Use of `nan_policy` and `keepdims`..."
    # 使用pytest的raises断言，验证调用stats.skew时使用nan_policy='omit'会抛出NotImplementedError异常
    with pytest.raises(NotImplementedError, match=message):
        stats.skew(x, nan_policy='omit')
    # 同样验证使用keepdims=True时会抛出NotImplementedError异常
    with pytest.raises(NotImplementedError, match=message):
        stats.skew(x, keepdims=True)
```