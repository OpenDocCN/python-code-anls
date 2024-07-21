# `.\pytorch\torch\_numpy\testing\utils.py`

```
"""
Utility function to facilitate testing.

"""
import contextlib  # 导入上下文管理模块
import gc  # 导入垃圾回收模块
import operator  # 导入操作符模块
import os  # 导入操作系统模块
import platform  # 导入平台信息模块
import pprint  # 导入漂亮打印模块
import re  # 导入正则表达式模块
import shutil  # 导入文件操作模块
import sys  # 导入系统相关模块
import warnings  # 导入警告模块
from functools import wraps  # 导入 wraps 装饰器
from io import StringIO  # 导入字符串IO模块
from tempfile import mkdtemp, mkstemp  # 导入临时文件和目录创建模块
from warnings import WarningMessage  # 导入警告消息类

import torch._numpy as np  # 导入torch._numpy模块作为np
from torch._numpy import (  # 从torch._numpy导入多个函数和类
    arange,
    asarray as asanyarray,
    empty,
    float32,
    intp,
    ndarray,
)

__all__ = [  # 定义模块的公开接口列表
    "assert_equal",
    "assert_almost_equal",
    "assert_approx_equal",
    "assert_array_equal",
    "assert_array_less",
    "assert_string_equal",
    "assert_",  # 函数名重复出现，保留一次
    "assert_array_almost_equal",
    "build_err_msg",
    "decorate_methods",
    "print_assert_equal",
    "verbose",
    "assert_array_almost_equal_nulp",
    "assert_raises_regex",
    "assert_array_max_ulp",
    "assert_warns",
    "assert_no_warnings",
    "assert_allclose",
    "IgnoreException",
    "clear_and_catch_warnings",
    "temppath",
    "tempdir",
    "IS_PYPY",
    "HAS_REFCOUNT",
    "IS_WASM",
    "suppress_warnings",
    "assert_array_compare",
    "assert_no_gc_cycles",
    "break_cycles",
    "IS_PYSTON",
]

verbose = 0  # 设置全局变量 verbose 为0

IS_WASM = platform.machine() in ["wasm32", "wasm64"]  # 检测当前平台是否为WASM
IS_PYPY = sys.implementation.name == "pypy"  # 检测是否运行在PyPy环境
IS_PYSTON = hasattr(sys, "pyston_version_info")  # 检测是否运行在Pyston环境
HAS_REFCOUNT = (
    getattr(sys, "getrefcount", None) is not None and not IS_PYSTON
)  # 检测系统是否支持引用计数

def assert_(val, msg=""):
    """
    Assert that works in release mode.
    Accepts callable msg to allow deferring evaluation until failure.

    The Python built-in ``assert`` does not work when executing code in
    optimized mode (the ``-O`` flag) - no byte-code is generated for it.

    For documentation on usage, refer to the Python documentation.

    """
    __tracebackhide__ = True  # 隐藏 traceback，用于pytest
    if not val:
        try:
            smsg = msg()  # 尝试调用msg函数获取详细错误信息
        except TypeError:
            smsg = msg  # 如果msg不是callable，则直接使用msg
        raise AssertionError(smsg)  # 抛出断言错误

def gisnan(x):
    return np.isnan(x)  # 判断x是否为NaN

def gisfinite(x):
    return np.isfinite(x)  # 判断x是否为有限数

def gisinf(x):
    return np.isinf(x)  # 判断x是否为无穷大数

def build_err_msg(
    arrays,
    err_msg,
    header="Items are not equal:",
    verbose=True,
    names=("ACTUAL", "DESIRED"),
    precision=8,
):
    msg = ["\n" + header]  # 构建错误消息头部
    if err_msg:
        if err_msg.find("\n") == -1 and len(err_msg) < 79 - len(header):
            msg = [msg[0] + " " + err_msg]  # 如果错误消息单行且长度不超过79字符，则添加到消息中
        else:
            msg.append(err_msg)  # 否则作为新行添加到消息中
    # 如果 verbose 参数为真，则执行以下代码块
    if verbose:
        # 使用 enumerate() 遍历数组列表 arrays，并返回索引 i 和数组 a
        for i, a in enumerate(arrays):
            # 检查数组 a 是否为 ndarray 类型
            if isinstance(a, ndarray):
                # 如果是 ndarray 类型，则使用 ndarray.__repr__ 方法生成表示形式
                r_func = ndarray.__repr__
            else:
                # 如果不是 ndarray 类型，则使用内置的 repr() 方法生成表示形式
                r_func = repr

            # 尝试调用 r_func 生成对象 a 的表示形式，捕获可能的异常
            try:
                r = r_func(a)
            except Exception as exc:
                # 如果生成表示形式时出现异常，则返回带有错误消息的特定格式字符串
                r = f"[repr failed for <{type(a).__name__}>: {exc}]"

            # 如果表示形式包含超过三行，则截取前三行并在末尾加上省略号
            if r.count("\n") > 3:
                r = "\n".join(r.splitlines()[:3])
                r += "..."

            # 将结果添加到消息列表中，格式为 "{names[i]}: {r}"
            msg.append(f" {names[i]}: {r}")

    # 将消息列表中的所有元素连接成一个以换行符分隔的字符串，并返回该字符串作为结果
    return "\n".join(msg)
# 定义了一个断言函数 assert_equal，用于比较两个对象是否相等
def assert_equal(actual, desired, err_msg="", verbose=True):
    """
    Raises an AssertionError if two objects are not equal.

    Given two objects (scalars, lists, tuples, dictionaries or numpy arrays),
    check that all elements of these objects are equal. An exception is raised
    at the first conflicting values.

    When one of `actual` and `desired` is a scalar and the other is array_like,
    the function checks that each element of the array_like object is equal to
    the scalar.

    This function handles NaN comparisons as if NaN was a "normal" number.
    That is, AssertionError is not raised if both objects have NaNs in the same
    positions.  This is in contrast to the IEEE standard on NaNs, which says
    that NaN compared to anything must return False.

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal.

    Examples
    --------
    >>> np.testing.assert_equal([4,5], [4,6])
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal:
    item=1
     ACTUAL: 5
     DESIRED: 6

    The following comparison does not raise an exception.  There are NaNs
    in the inputs, but they are in the same positions.

    >>> np.testing.assert_equal(np.array([1.0, 2.0, np.nan]), [1, 2, np.nan])

    """
    __tracebackhide__ = True  # Hide traceback for py.test

    # 计算 actual 和 desired 是否有一个为 None
    num_nones = sum([actual is None, desired is None])
    # 如果只有一个为 None，则抛出 AssertionError
    if num_nones == 1:
        raise AssertionError(f"Not equal: {actual} != {desired}")
    # 如果两个都是 None，则返回 True
    elif num_nones == 2:
        return True
    # 否则，继续比较

    # 如果 actual 或 desired 是 numpy 的数据类型对象，则直接比较
    if isinstance(actual, np.DType) or isinstance(desired, np.DType):
        result = actual == desired
        # 如果不相等，则抛出 AssertionError
        if not result:
            raise AssertionError(f"Not equal: {actual} != {desired}")
        else:
            return True

    # 如果 desired 和 actual 都是字符串，则直接比较
    if isinstance(desired, str) and isinstance(actual, str):
        assert actual == desired
        return

    # 如果 desired 是字典，则逐一比较其键值对
    if isinstance(desired, dict):
        # 如果 actual 不是字典，则抛出 AssertionError
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        # 比较字典的长度
        assert_equal(len(actual), len(desired), err_msg, verbose)
        # 逐一比较字典的键值对
        for k in desired.keys():
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], f"key={k!r}\n{err_msg}", verbose)
        return
    
    # 如果 desired 和 actual 都是列表或元组，则逐一比较其元素
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        # 比较列表或元组的长度
        assert_equal(len(actual), len(desired), err_msg, verbose)
        # 逐一比较列表或元组的元素
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], f"item={k!r}\n{err_msg}", verbose)
        return
    # 从 torch._numpy 模块中导入必要的函数和类
    from torch._numpy import imag, iscomplexobj, isscalar, ndarray, real, signbit

    # 如果 actual 或 desired 是 ndarray 类型，则调用 assert_array_equal 函数比较它们
    if isinstance(actual, ndarray) or isinstance(desired, ndarray):
        return assert_array_equal(actual, desired, err_msg, verbose)
    
    # 构建错误信息，用于比较失败时的异常处理
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)

    # 处理复数情况：将复数分解为实部和虚部，以正确处理 nan/inf/负零
    # XXX: 捕获 ValueError 异常，适用于 ndarray 的子类中 iscomplexobj 函数可能失败的情况
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except (ValueError, TypeError):
        usecomplex = False

    # 如果存在复数，则分别处理实部和虚部
    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            # 断言实部和虚部分别相等
            assert_equal(actualr, desiredr)
            assert_equal(actuali, desiredi)
        except AssertionError:
            raise AssertionError(msg)  # noqa: B904

    # isscalar 测试以检查如 [np.nan] != np.nan 的情况
    if isscalar(desired) != isscalar(actual):
        raise AssertionError(msg)

    # 处理 Inf/nan/负零情况
    try:
        isdesnan = gisnan(desired)
        isactnan = gisnan(actual)
        if isdesnan and isactnan:
            return  # 如果都是 nan，则认为相等

        # 对于浮点数，特别处理有符号零
        array_actual = np.asarray(actual)
        array_desired = np.asarray(desired)

        if desired == 0 and actual == 0:
            if not signbit(desired) == signbit(actual):
                raise AssertionError(msg)

    except (TypeError, ValueError, NotImplementedError):
        pass

    try:
        # 显式使用 __eq__ 方法进行比较，参考 gh-2552
        if not (desired == actual):
            raise AssertionError(msg)

    except (DeprecationWarning, FutureWarning) as e:
        # 处理两种类型无法比较的情况
        if "elementwise == comparison" in e.args[0]:
            raise AssertionError(msg)  # noqa: B904
        else:
            raise
# 定义一个函数，用于测试两个对象是否相等，如果测试失败则打印错误消息。
# 测试的条件是 `actual == desired`。
def print_assert_equal(test_string, actual, desired):
    """
    Test if two objects are equal, and print an error message if test fails.

    The test is performed with ``actual == desired``.

    Parameters
    ----------
    test_string : str
        The message supplied to AssertionError.
    actual : object
        The object to test for equality against `desired`.
    desired : object
        The expected result.

    Examples
    --------
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 1])  # doctest: +SKIP
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 2])  # doctest: +SKIP
    Traceback (most recent call last):
    ...
    AssertionError: Test XYZ of func xyz failed
    ACTUAL:
    [0, 1]
    DESIRED:
    [0, 2]

    """
    __tracebackhide__ = True  # py.test 中隐藏回溯信息
    import pprint  # 导入 pprint 模块

    # 如果 actual 和 desired 不相等
    if not (actual == desired):
        msg = StringIO()  # 创建一个 StringIO 对象 msg，用于构建错误消息
        msg.write(test_string)  # 写入测试字符串到 msg
        msg.write(" failed\nACTUAL: \n")  # 写入失败提示和换行
        pprint.pprint(actual, msg)  # 使用 pprint 格式化打印 actual 到 msg
        msg.write("DESIRED: \n")  # 写入期望值提示和换行
        pprint.pprint(desired, msg)  # 使用 pprint 格式化打印 desired 到 msg
        raise AssertionError(msg.getvalue())  # 抛出 AssertionError，并附带 msg 的内容


# 定义一个函数，用于在给定精度下比较两个项目是否相等，否则抛出 AssertionError
def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    """
    Raises an AssertionError if two items are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies that the elements of `actual` and `desired` satisfy.

        ``abs(desired-actual) < float64(1.5 * 10**(-decimal))``

    That is a looser test than originally documented, but agrees with what the
    actual implementation in `assert_array_almost_equal` did up to rounding
    vagaries. An exception is raised at conflicting values. For ndarrays this
    delegates to assert_array_almost_equal

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    decimal : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> from torch._numpy.testing import assert_almost_equal
    >>> assert_almost_equal(2.3333333333333, 2.33333334)
    >>> assert_almost_equal(2.3333333333333, 2.33333334, decimal=10)
    Traceback (most recent call last):
        ...

    """
    AssertionError:
    Arrays are not almost equal to 10 decimals
     ACTUAL: 2.3333333333333
     DESIRED: 2.33333334

    >>> assert_almost_equal(np.array([1.0,2.3333333333333]),
    ...                     np.array([1.0,2.33333334]), decimal=9)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 9 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 2 (50%)
    Max absolute difference: 6.666699636781459e-09
    Max relative difference: 2.8571569790287484e-09
     x: torch.ndarray([1.0000, 2.3333], dtype=float64)
     y: torch.ndarray([1.0000, 2.3333], dtype=float64)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    from torch._numpy import imag, iscomplexobj, ndarray, real

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except ValueError:
        usecomplex = False

    def _build_err_msg():
        # 构建错误消息的头部，指明数组在指定小数位数下不几乎相等
        header = "Arrays are not almost equal to %d decimals" % decimal
        return build_err_msg([actual, desired], err_msg, verbose=verbose, header=header)

    if usecomplex:
        # 如果涉及复数，分别处理实部和虚部，以正确处理 nan/inf/负零
        if iscomplexobj(actual):
            actualr = real(actual)  # 提取实部
            actuali = imag(actual)  # 提取虚部
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            # 分别对实部和虚部进行几乎相等断言
            assert_almost_equal(actualr, desiredr, decimal=decimal)
            assert_almost_equal(actuali, desiredi, decimal=decimal)
        except AssertionError:
            raise AssertionError(_build_err_msg())  # 引发自定义错误消息异常

    if isinstance(actual, (ndarray, tuple, list)) or isinstance(
        desired, (ndarray, tuple, list)
    ):
        # 如果涉及到数组、元组或列表，则调用 assert_array_almost_equal 函数进行断言
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    try:
        # 如果其中一个不是有限数，这里特殊处理：
        # 如果任何一个是 nan，则检查两者是否都是 nan，并测试它们是否相等
        if not (gisfinite(desired) and gisfinite(actual)):
            if gisnan(desired) or gisnan(actual):
                if not (gisnan(desired) and gisnan(actual)):
                    raise AssertionError(_build_err_msg())
            else:
                if not desired == actual:
                    raise AssertionError(_build_err_msg())
            return
    except (NotImplementedError, TypeError):
        pass
    # 对于其他情况，使用绝对值检查是否在指定小数位数下几乎相等
    if abs(desired - actual) >= np.float64(1.5 * 10.0 ** (-decimal)):
        raise AssertionError(_build_err_msg())
# 定义一个函数用于断言两个数值在指定有效数字范围内近似相等，否则引发 AssertionError
def assert_approx_equal(actual, desired, significant=7, err_msg="", verbose=True):
    """
    Raises an AssertionError if two items are not equal up to significant
    digits.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    Given two numbers, check that they are approximately equal.
    Approximately equal is defined as the number of significant digits
    that agree.

    Parameters
    ----------
    actual : scalar
        The object to check.
    desired : scalar
        The expected object.
    significant : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    >>> np.testing.assert_approx_equal(0.12345677777777e-20, 0.1234567e-20)  # doctest: +SKIP
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345671e-20,  # doctest: +SKIP
    ...                                significant=8)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345672e-20,  # doctest: +SKIP
    ...                                significant=8)
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal to 8 significant digits:
     ACTUAL: 1.234567e-21
     DESIRED: 1.2345672e-21

    the evaluated condition that raises the exception is

    >>> abs(0.12345670e-20/1e-21 - 0.12345672e-20/1e-21) >= 10**-(8-1)
    True

    """
    __tracebackhide__ = True  # 在 py.test 中隐藏回溯信息
    import numpy as np  # 导入 numpy 库

    (actual, desired) = map(float, (actual, desired))  # 将 actual 和 desired 转换为浮点数
    if desired == actual:  # 如果期望值和实际值相等，直接返回
        return
    # 计算一个比例因子，用于将数值标准化到 (-10.0, 10.0) 的范围内
    scale = 0.5 * (np.abs(desired) + np.abs(actual))  # 计算标准化的基数
    scale = np.power(10, np.floor(np.log10(scale)))  # 将基数转换为 10 的幂
    try:
        sc_desired = desired / scale  # 尝试计算期望值的标准化值
    except ZeroDivisionError:
        sc_desired = 0.0  # 如果出现除零错误，设定为 0
    try:
        sc_actual = actual / scale  # 尝试计算实际值的标准化值
    except ZeroDivisionError:
        sc_actual = 0.0  # 如果出现除零错误，设定为 0
    # 构建错误消息，说明两个数值不相等到指定有效数字
    msg = build_err_msg(
        [actual, desired],
        err_msg,
        header="Items are not equal to %d significant digits:" % significant,
        verbose=verbose,
    )
    try:
        # 尝试检查期望值和实际值是否都是有限的
        if not (gisfinite(desired) and gisfinite(actual)):
            # 如果期望值或实际值中有任何一个是NaN，则特殊处理：
            # 检查如果两者都是NaN，或者检查它们是否相等
            if gisnan(desired) or gisnan(actual):
                # 如果期望值或实际值是NaN但不都是NaN，则抛出断言错误
                if not (gisnan(desired) and gisnan(actual)):
                    raise AssertionError(msg)
            else:
                # 如果期望值和实际值都不是NaN但不相等，则抛出断言错误
                if not desired == actual:
                    raise AssertionError(msg)
            return
    except (TypeError, NotImplementedError):
        pass

    # 如果不满足特殊情况，检查两个数值的绝对差是否大于或等于指定精度的阈值
    if np.abs(sc_desired - sc_actual) >= np.power(10.0, -(significant - 1)):
        # 如果差值超出了指定的精度阈值，则抛出断言错误
        raise AssertionError(msg)
# 定义一个函数，用于比较两个数组或类数组对象是否相等
def assert_array_compare(
    comparison,
    x,
    y,
    err_msg="",
    verbose=True,
    header="",
    precision=6,
    equal_nan=True,
    equal_inf=True,
    *,
    strict=False,
):
    __tracebackhide__ = True  # 在 py.test 中隐藏回溯信息
    # 导入必要的模块和函数
    from torch._numpy import all, array, asarray, bool_, inf, isnan, max

    # 将输入的 x 和 y 转换为 numpy 数组
    x = asarray(x)
    y = asarray(y)

    # 定义一个函数，用于将数组转换为字符串形式
    def array2string(a):
        return str(a)

    # 保存原始的 x 和 y 数组用于输出格式化
    ox, oy = x, y

    # 定义一个函数，用于处理 nan 和 inf 的比较
    def func_assert_same_pos(x, y, func=isnan, hasval="nan"):
        """处理 nan 和 inf 的比较。

        结合运行 func 函数在 x 和 y 上的结果，检查它们在相同位置是否都为 True。
        """
        __tracebackhide__ = True  # 在 py.test 中隐藏回溯信息
        # 计算 x 和 y 中 func 函数的结果
        x_id = func(x)
        y_id = func(y)
        # 处理三种类型略有问题的 ndarray 子类的情况：
        # (1) 对于 `masked` 数组标量，all() 可能返回 masked 数组，所以使用 != True
        # (2) 一些 ndarray 子类的 __eq__ 方法返回 Python 布尔值而不是逐元素比较结果，
        #     因此需要转换为 bool_() 并使用 isinstance(..., bool) 检查
        # (3) 某些 ndarray 子类可能没有实现 np.all() 方法，所以优先使用 .all() 方法
        if (x_id == y_id).all().item() is not True:
            # 构建错误消息
            msg = build_err_msg(
                [x, y],
                err_msg + f"\nx and y {hasval} location mismatch:",
                verbose=verbose,
                header=header,
                names=("x", "y"),
                precision=precision,
            )
            raise AssertionError(msg)
        # 如果是标量，则这里我们知道数组在所有位置上都有相同的标志，所以应该返回标量的标志。
        if isinstance(x_id, bool) or x_id.ndim == 0:
            return bool_(x_id)
        elif isinstance(y_id, bool) or y_id.ndim == 0:
            return bool_(y_id)
        else:
            return y_id

    except ValueError:
        import traceback

        # 处理 ValueError 异常
        efmt = traceback.format_exc()
        header = f"error during assertion:\n\n{efmt}\n\n{header}"

        # 构建错误消息
        msg = build_err_msg(
            [x, y],
            err_msg,
            verbose=verbose,
            header=header,
            names=("x", "y"),
            precision=precision,
        )
        raise ValueError(msg)  # noqa: B904
    # 定义函数 np.testing.assert_array_equal，用于比较两个 array_like 对象是否相等。
    def assert_array_equal(x, y, err_msg=None, verbose=True, strict=False):
        # 如果 strict 参数为 True，当 array_like 对象的形状或数据类型不匹配时会引发 AssertionError。
        # 此外，当 x 或 y 中一个为标量而另一个为 array_like 时，如果 strict 为 False（默认），函数会检查每个数组元素是否等于标量。
        # 另外，如果 x 和 y 不相等，会根据 verbose 参数决定是否将冲突的值追加到错误消息中。
        # np.testing.assert_array_equal 函数在处理标量时具有特殊处理，详见 Notes 部分。
        pass  # 实际比较是在底层的 C 代码中完成的，该函数主要用于异常处理和错误消息的输出。
    """
    __tracebackhide__ = True  # 将此测试代码中的错误回溯隐藏，用于 py.test
    assert_array_compare(
        operator.__eq__,  # 使用 operator 模块中的相等比较函数
        x,  # 第一个数组或对象 x
        y,  # 第二个数组或对象 y
        err_msg=err_msg,  # 如果数组不相等时的错误消息
        verbose=verbose,  # 是否输出详细比较信息
        header="Arrays are not equal",  # 错误消息的头部信息
        strict=strict,  # 是否严格比较，例如对 NaN 的处理
    )
    """
# 自定义断言函数，用于比较两个数组或类似对象在指定精度下是否近似相等，否则抛出 AssertionError
def assert_array_almost_equal(x, y, decimal=6, err_msg="", verbose=True):
    """
    Raises an AssertionError if two objects are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies identical shapes and that the elements of ``actual`` and
    ``desired`` satisfy.

        ``abs(desired-actual) < 1.5 * 10**(-decimal)``

    That is a looser test than originally documented, but agrees with what the
    actual implementation did up to rounding vagaries. An exception is raised
    at shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if both
    objects have NaNs in the same positions.

    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    decimal : int, optional
        Desired precision, default is 6.
    err_msg : str, optional
      The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal

    Examples
    --------
    the first assert does not raise an exception

    >>> np.testing.assert_array_almost_equal([1.0,2.333,np.nan],
    ...                                      [1.0,2.333,np.nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33339,np.nan], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 5.999999999994898e-05
    Max relative difference: 2.5713661239633743e-05
     x: torch.ndarray([1.0000, 2.3333,    nan], dtype=float64)
     y: torch.ndarray([1.0000, 2.3334,    nan], dtype=float64)

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33333, 5], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    x and y nan location mismatch:
     x: torch.ndarray([1.0000, 2.3333,    nan], dtype=float64)
     y: torch.ndarray([1.0000, 2.3333, 5.0000], dtype=float64)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    # 从 torch._numpy 模块中导入指定函数和类
    from torch._numpy import any as npany, float_, issubdtype, number, result_type

    # 定义比较函数，用于比较两个数组 x 和 y
    def compare(x, y):
        try:
            # 检查 x 或 y 中是否包含正负无穷大的元素
            if npany(gisinf(x)) or npany(gisinf(y)):
                # 获取 x 和 y 中正负无穷大的标志
                xinfid = gisinf(x)
                yinfid = gisinf(y)
                # 如果 x 和 y 中正负无穷大的标志不完全相同，则返回 False
                if not (xinfid == yinfid).all():
                    return False
                # 如果 x 和 y 都只有一个元素且都是正负无穷大，则直接比较它们的值
                if x.size == y.size == 1:
                    return x == y
                # 从 x 和 y 中排除掉正负无穷大的元素
                x = x[~xinfid]
                y = y[~yinfid]
        except (TypeError, NotImplementedError):
            pass

        # 确保 y 是一个非精确类型，以避免处理 MIN_INT 的绝对值；这会导致后续对 x 的强制转换。
        # 确定 y 的数据类型
        dtype = result_type(y, 1.0)
        # 将 y 转换为指定数据类型的数组
        y = asanyarray(y, dtype)
        # 计算 x 和 y 的绝对差值
        z = abs(x - y)

        # 如果 z 的数据类型不是数字类型，则将其转换为 float_ 类型；用于处理对象数组
        if not issubdtype(z.dtype, number):
            z = z.astype(float_)
        
        # 返回比较结果，判断 z 是否小于指定精度下的阈值
        return z < 1.5 * 10.0 ** (-decimal)

    # 使用 assert_array_compare 函数对比 x 和 y 的数组
    assert_array_compare(
        compare,  # 比较函数
        x,        # 第一个数组
        y,        # 第二个数组
        err_msg=err_msg,            # 错误信息
        verbose=verbose,            # 是否详细输出
        header=("Arrays are not almost equal to %d decimals" % decimal),  # 输出头部信息
        precision=decimal,          # 比较的精度
    )
# 定义一个函数，用于断言两个数组对象是否按照严格的小于关系排序
def assert_array_less(x, y, err_msg="", verbose=True):
    """
    Raises an AssertionError if two array_like objects are not ordered by less
    than.

    Given two array_like objects, check that the shape is equal and all
    elements of the first object are strictly smaller than those of the
    second object. An exception is raised at shape mismatch or incorrectly
    ordered values. Shape mismatch does not raise if an object has zero
    dimension. In contrast to the standard usage in numpy, NaNs are
    compared, no assertion is raised if both objects have NaNs in the same
    positions.

    Parameters
    ----------
    x : array_like
      The smaller object to check.
    y : array_like
      The larger object to compare.
    err_msg : string
      The error message to be printed in case of failure.
    verbose : bool
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
      If actual and desired objects are not equal.

    See Also
    --------
    assert_array_equal: tests objects for equality
    assert_array_almost_equal: test objects for equality up to precision

    Examples
    --------
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1.1, 2.0, np.nan])
    >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1, 2.0, np.nan])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 1.0
    Max relative difference: 0.5
     x: torch.ndarray([1.,  1., nan], dtype=float64)
     y: torch.ndarray([1.,  2., nan], dtype=float64)

    >>> np.testing.assert_array_less([1.0, 4.0], 3)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    Mismatched elements: 1 / 2 (50%)
    Max absolute difference: 2.0
    Max relative difference: 0.6666666666666666
     x: torch.ndarray([1., 4.], dtype=float64)
     y: torch.ndarray(3)

    >>> np.testing.assert_array_less([1.0, 2.0, 3.0], [4])
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not less-ordered
    <BLANKLINE>
    (shapes (3,), (1,) mismatch)
     x: torch.ndarray([1., 2., 3.], dtype=float64)
     y: torch.ndarray([4])

    """
    __tracebackhide__ = True  # 在 py.test 中隐藏回溯信息
    # 调用 assert_array_compare 函数，比较两个对象的小于关系
    assert_array_compare(
        operator.__lt__,  # 比较操作为小于
        x,  # 较小的对象
        y,  # 较大的对象
        err_msg=err_msg,  # 错误消息
        verbose=verbose,  # 是否显示详细信息
        header="Arrays are not less-ordered",  # 错误消息的标题
        equal_inf=False,  # 不允许无穷大相等
    )


# 定义一个函数，用于断言两个字符串是否相等
def assert_string_equal(actual, desired):
    """
    Test if two strings are equal.

    If the given strings are equal, `assert_string_equal` does nothing.
    If they are not equal, an AssertionError is raised, and the diff
    between the strings is shown.

    Parameters
    ----------
    actual : str
        The string to test for equality against the expected string.
    desired : str
        The expected string to compare against.

    """
    desired : str
        The expected string.

    Examples
    --------
    >>> np.testing.assert_string_equal('abc', 'abc')  # doctest: +SKIP
    >>> np.testing.assert_string_equal('abc', 'abcd')  # doctest: +SKIP
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ...
    AssertionError: Differences in strings:
    - abc+ abcd?    +

    """
    # 延迟导入 difflib 以减少启动时间
    __tracebackhide__ = True  # 为 py.test 隐藏回溯信息
    import difflib

    # 如果 actual 不是字符串类型，抛出 AssertionError 异常
    if not isinstance(actual, str):
        raise AssertionError(repr(type(actual)))
    # 如果 desired 不是字符串类型，抛出 AssertionError 异常
    if not isinstance(desired, str):
        raise AssertionError(repr(type(desired)))
    # 如果 expected 和 actual 相等，则直接返回
    if desired == actual:
        return

    # 使用 difflib 模块比较 actual 和 desired 的行级差异
    diff = list(
        difflib.Differ().compare(actual.splitlines(True), desired.splitlines(True))
    )
    diff_list = []
    # 处理差异列表
    while diff:
        d1 = diff.pop(0)
        # 如果是相同行，则继续
        if d1.startswith("  "):
            continue
        # 如果 actual 多出一行，则处理删除的行和增加的行
        if d1.startswith("- "):
            l = [d1]
            d2 = diff.pop(0)
            if d2.startswith("? "):
                l.append(d2)
                d2 = diff.pop(0)
            if not d2.startswith("+ "):
                raise AssertionError(repr(d2))
            l.append(d2)
            # 继续处理可能的后续差异说明行
            if diff:
                d3 = diff.pop(0)
                if d3.startswith("? "):
                    l.append(d3)
                else:
                    diff.insert(0, d3)
            # 如果修改后的行与删除的行相同，则继续下一轮循环
            if d2[2:] == d1[2:]:
                continue
            # 将差异行添加到差异列表中
            diff_list.extend(l)
            continue
        # 如果不符合上述条件，则抛出 AssertionError 异常
        raise AssertionError(repr(d1))
    # 如果差异列表为空，则返回
    if not diff_list:
        return
    # 构建差异消息字符串
    msg = f"Differences in strings:\n{''.join(diff_list).rstrip()}"
    # 如果 actual 和 desired 不相等，则抛出 AssertionError 异常，显示差异信息
    if actual != desired:
        raise AssertionError(msg)
# 导入unittest模块，用于编写和运行测试
import unittest

# 定义一个名为_Dummy的测试用例类，继承自unittest.TestCase
class _Dummy(unittest.TestCase):
    
    # 定义一个空方法nop，用于在测试用例中占位
    def nop(self):
        pass

# 创建_Dummy类的实例_d，并调用其nop方法（虽然不执行任何操作）
_d = _Dummy("nop")

# 定义一个自定义的断言函数assert_raises_regex
def assert_raises_regex(exception_class, expected_regexp, *args, **kwargs):
    """
    assert_raises_regex(exception_class, expected_regexp, callable, *args,
                        **kwargs)
    assert_raises_regex(exception_class, expected_regexp)

    如果在调用callable时使用args和kwargs参数执行时，抛出的异常是exception_class类的，并且异常消息与expected_regexp匹配，则断言失败。

    或者可以像`assert_raises`一样作为上下文管理器使用。

    Notes
    -----
    .. versionadded:: 1.9.0

    """
    # 隐藏py.test的回溯信息
    __tracebackhide__ = True
    # 调用_d实例的assertRaisesRegex方法进行断言
    return _d.assertRaisesRegex(exception_class, expected_regexp, *args, **kwargs)

# 定义一个函数decorate_methods，用于给类中匹配正则表达式testmatch的公共方法应用decorator修饰器
def decorate_methods(cls, decorator, testmatch=None):
    """
    Apply a decorator to all methods in a class matching a regular expression.

    The given decorator is applied to all public methods of `cls` that are
    matched by the regular expression `testmatch`
    (``testmatch.search(methodname)``). Methods that are private, i.e. start
    with an underscore, are ignored.

    Parameters
    ----------
    cls : class
        Class whose methods to decorate.
    decorator : function
        Decorator to apply to methods
    testmatch : compiled regexp or str, optional
        The regular expression. Default value is None, in which case the
        nose default (``re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)``)
        is used.
        If `testmatch` is a string, it is compiled to a regular expression
        first.

    """
    # 如果testmatch为None，默认使用nose的默认正则表达式匹配测试方法名
    if testmatch is None:
        testmatch = re.compile(rf"(?:^|[\\b_\\.{os.sep}-])[Tt]est")
    else:
        testmatch = re.compile(testmatch)
    
    # 获取类的属性字典
    cls_attr = cls.__dict__

    # 延迟导入以减少启动时间
    from inspect import isfunction

    # 获取所有方法列表
    methods = [_m for _m in cls_attr.values() if isfunction(_m)]
    
    # 遍历方法列表，为匹配正则表达式的公共方法应用decorator修饰器
    for function in methods:
        try:
            # 获取方法的名称
            if hasattr(function, "compat_func_name"):
                funcname = function.compat_func_name
            else:
                funcname = function.__name__
        except AttributeError:
            # 如果不是函数，则跳过
            continue
        
        # 如果方法名称符合正则表达式testmatch，并且不是以下划线开头的私有方法，则应用decorator修饰器
        if testmatch.search(funcname) and not funcname.startswith("_"):
            setattr(cls, funcname, decorator(function))
    
    return

# 定义一个私有函数_assert_valid_refcount，用于检查ufuncs是否正确处理对象`1`的引用计数，用于一些回归测试
def _assert_valid_refcount(op):
    """
    Check that ufuncs don't mishandle refcount of object `1`.
    Used in a few regression tests.
    """
    # 如果没有引用计数功能，直接返回True
    if not HAS_REFCOUNT:
        return True

    # 导入必要的模块
    import gc
    import numpy as np

    # 创建一个100x100的数组b，并将c指向b
    b = np.arange(100 * 100).reshape(100, 100)
    c = b
    i = 1

    # 禁用垃圾回收
    gc.disable()
    try:
        # 获取对象`1`的初始引用计数
        rc = sys.getrefcount(i)
        
        # 执行op(b, c)操作多次，检查引用计数是否正常
        for j in range(15):
            d = op(b, c)
        
        # 断言对象`1`的引用计数至少与初始引用计数一样多
        assert_(sys.getrefcount(i) >= rc)
    
    finally:
        # 最终恢复垃圾回收功能
        gc.enable()
    
    # 删除变量d，用于静态代码检查
    del d  # for pyflakes

# 定义一个函数assert_allclose，用于断言两个对象的值在一定范围内接近
def assert_allclose(
    actual,
    desired,
    rtol=1e-7,  # 相对容差，用于比较浮点数的相等性
    atol=0,     # 绝对容差，用于比较浮点数的相等性
    equal_nan=True,  # 是否将 NaN 视为相等
    err_msg="",  # 错误消息，用于在断言失败时显示
    verbose=True,  # 是否显示详细信息
    check_dtype=False,  # 是否检查数据类型
# Raises an AssertionError if two objects are not equal up to desired tolerance.
def assert_allclose(
    actual,  # Array obtained.
    desired,  # Array desired.
    rtol=1e-7,  # Relative tolerance.
    atol=0,  # Absolute tolerance.
    equal_nan=False,  # If True, NaNs will compare equal.
    err_msg='',  # The error message to be printed in case of failure.
    verbose=True,  # If True, the conflicting values are appended to the error message.
):
    """
    Raises an AssertionError if two objects are not equal up to desired
    tolerance.

    Given two array_like objects, check that their shapes and all elements
    are equal (but see the Notes for the special handling of a scalar). An
    exception is raised if the shapes mismatch or any values conflict. In
    contrast to the standard usage in numpy, NaNs are compared like numbers,
    no assertion is raised if both objects have NaNs in the same positions.

    The test is equivalent to ``allclose(actual, desired, rtol, atol)`` (note
    that ``allclose`` has different default values). It compares the difference
    between `actual` and `desired` to ``atol + rtol * abs(desired)``.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    equal_nan : bool, optional
        If True, NaNs will compare equal.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    assert_array_almost_equal_nulp, assert_array_max_ulp

    Notes
    -----
    When one of `actual` and `desired` is a scalar and the other is
    array_like, the function checks that each element of the array_like
    object is equal to the scalar.

    Examples
    --------
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)

    """
    __tracebackhide__ = True  # Hide traceback for py.test

    def compare(x, y):
        return np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)

    actual, desired = asanyarray(actual), asanyarray(desired)
    header = f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}"

    if check_dtype:
        assert actual.dtype == desired.dtype

    assert_array_compare(
        compare,
        actual,
        desired,
        err_msg=str(err_msg),
        verbose=verbose,
        header=header,
        equal_nan=equal_nan,
    )
    # 检查数组中所有元素的最大单位最后一位（ULP）差异是否在给定的N单位范围内。
    # 如果不满足以下条件，将引发断言错误：
    # abs(x - y) <= nulp * spacing(maximum(abs(x), abs(y)))
    def assert_array_max_ulp(x, y, nulp):
        """
        Check that all items of arrays differ in at most
        N Units in the Last Place.
    
        Notes
        -----
        An assertion is raised if the following condition is not met::
    
            abs(x - y) <= nulp * spacing(maximum(abs(x), abs(y)))
    
        Examples
        --------
        >>> x = np.array([1., 1e-10, 1e-20])
        >>> eps = np.finfo(x.dtype).eps
        >>> np.testing.assert_array_almost_equal_nulp(x, x*eps/2 + x)  # doctest: +SKIP
    
        >>> np.testing.assert_array_almost_equal_nulp(x, x*eps + x)  # doctest: +SKIP
        Traceback (most recent call last):
          ...
        AssertionError: X and Y are not equal to 1 ULP (max is 2)
    
        """
        __tracebackhide__ = True  # Hide traceback for py.test
        import numpy as np
    
        # 计算数组 x 和 y 的绝对值
        ax = np.abs(x)
        ay = np.abs(y)
        # 选择 x 和 y 中绝对值较大的作为参考，计算单位最后一位的间距
        ref = nulp * np.spacing(np.where(ax > ay, ax, ay))
        # 如果不是所有元素满足 abs(x - y) <= ref，则引发断言错误
        if not np.all(np.abs(x - y) <= ref):
            # 如果 x 或 y 是复数对象，错误消息指示 x 和 y 不等于 nulp ULP
            if np.iscomplexobj(x) or np.iscomplexobj(y):
                msg = "X and Y are not equal to %d ULP" % nulp
            # 否则，显示最大的 ULP 差异值
            else:
                max_nulp = np.max(nulp_diff(x, y))
                msg = "X and Y are not equal to %d ULP (max is %g)" % (nulp, max_nulp)
            # 抛出断言错误，显示相应的错误消息
            raise AssertionError(msg)
# 导入必要的库函数 numpy
import numpy as np

# 定义函数 assert_array_max_ulp，用于比较两个数组的最大单位差距
def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    """
    Check that all items of arrays differ in at most N Units in the Last Place.

    Parameters
    ----------
    a, b : array_like
        Input arrays to be compared.
    maxulp : int, optional
        The maximum number of units in the last place that elements of `a` and
        `b` can differ. Default is 1.
    dtype : dtype, optional
        Data-type to convert `a` and `b` to if given. Default is None.

    Returns
    -------
    ret : ndarray
        Array containing number of representable floating point numbers between
        items in `a` and `b`.

    Raises
    ------
    AssertionError
        If one or more elements differ by more than `maxulp`.

    Notes
    -----
    For computing the ULP difference, this API does not differentiate between
    various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
    is zero).

    See Also
    --------
    assert_array_almost_equal_nulp : Compare two arrays relatively to their
        spacing.

    Examples
    --------
    >>> a = np.linspace(0., 1., 100)
    >>> res = np.testing.assert_array_max_ulp(a, np.arcsin(np.sin(a)))  # doctest: +SKIP

    """
    # 设置隐藏 traceback 的选项，用于 pytest
    __tracebackhide__ = True  # Hide traceback for py.test

    # 导入 numpy 库

    # 调用 nulp_diff 函数计算两个数组 a 和 b 的 ULP 差距
    ret = nulp_diff(a, b, dtype)

    # 检查是否所有的差距都小于等于 maxulp
    if not np.all(ret <= maxulp):
        # 如果有元素的差距大于 maxulp，则抛出 AssertionError 异常
        raise AssertionError(
            f"Arrays are not almost equal up to {maxulp:g} "
            f"ULP (max difference is {np.max(ret):g} ULP)"
        )

    # 返回计算结果 ret
    return ret


# 定义函数 nulp_diff，用于计算两个数组中每个元素的 ULP 差距
def nulp_diff(x, y, dtype=None):
    """For each item in x and y, return the number of representable floating
    points between them.

    Parameters
    ----------
    x : array_like
        first input array
    y : array_like
        second input array
    dtype : dtype, optional
        Data-type to convert `x` and `y` to if given. Default is None.

    Returns
    -------
    nulp : array_like
        number of representable floating point numbers between each item in x
        and y.

    Notes
    -----
    For computing the ULP difference, this API does not differentiate between
    various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
    is zero).

    Examples
    --------
    # By definition, epsilon is the smallest number such as 1 + eps != 1, so
    # there should be exactly one ULP between 1 and 1 + eps
    >>> nulp_diff(1, 1 + np.finfo(x.dtype).eps)  # doctest: +SKIP
    1.0
    """
    # 导入 numpy 库
    import numpy as np

    # 如果指定了 dtype，则将 x 和 y 转换为指定的数据类型
    if dtype:
        x = np.asarray(x, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
    else:
        x = np.asarray(x)
        y = np.asarray(y)

    # 确定 x 和 y 共同的数据类型
    t = np.common_type(x, y)

    # 如果 x 或 y 是复数数组，抛出 NotImplementedError
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise NotImplementedError("_nulp not implemented for complex array")

    # 将 x 和 y 转换为相同的数组类型 t
    x = np.array([x], dtype=t)
    y = np.array([y], dtype=t)

    # 将 x 和 y 中的 NaN 值替换为 NaN
    x[np.isnan(x)] = np.nan
    y[np.isnan(y)] = np.nan
    # 检查 x 和 y 的形状是否相同，若不同则抛出值错误异常
    if not x.shape == y.shape:
        raise ValueError(f"x and y do not have the same shape: {x.shape} - {y.shape}")

    # 定义一个内部函数 _diff，用于计算两个数组 rx 和 ry 的差异，并返回绝对值后的结果
    def _diff(rx, ry, vdt):
        diff = np.asarray(rx - ry, dtype=vdt)
        return np.abs(diff)

    # 对输入的数组 x 和 y 进行整数表示（假设 integer_repr 是一个函数来实现这个转换）
    rx = integer_repr(x)
    ry = integer_repr(y)

    # 调用 _diff 函数计算整数表示后的 rx 和 ry 的差异，返回计算结果
    return _diff(rx, ry, t)
# 定义一个函数，用于将浮点数的二进制表示重新解释为带符号的幅度表示
def _integer_repr(x, vdt, comp):
    # 将浮点数 x 视图转换为指定类型 vdt
    rx = x.view(vdt)
    # 如果不是单个元素，则处理每个元素
    if not (rx.size == 1):
        # 对负数进行补码操作
        rx[rx < 0] = comp - rx[rx < 0]
    else:
        # 对单个元素的负数进行补码操作
        if rx < 0:
            rx = comp - rx

    return rx

# 定义一个函数，返回浮点数 x 的带符号幅度表示的二进制表示的解释
def integer_repr(x):
    """Return the signed-magnitude interpretation of the binary representation
    of x."""
    import numpy as np

    # 根据 x 的数据类型选择不同的参数调用 _integer_repr 函数
    if x.dtype == np.float16:
        return _integer_repr(x, np.int16, np.int16(-(2**15)))
    elif x.dtype == np.float32:
        return _integer_repr(x, np.int32, np.int32(-(2**31)))
    elif x.dtype == np.float64:
        return _integer_repr(x, np.int64, np.int64(-(2**63)))
    else:
        # 如果数据类型不支持，则抛出错误
        raise ValueError(f"Unsupported dtype {x.dtype}")

# 定义一个上下文管理器，用于捕获特定类型的警告
@contextlib.contextmanager
def _assert_warns_context(warning_class, name=None):
    __tracebackhide__ = True  # 在 pytest 中隐藏回溯信息
    with suppress_warnings() as sup:
        # 记录特定类型的警告
        l = sup.record(warning_class)
        yield
        # 如果没有记录到该类型的警告，则抛出断言错误
        if not len(l) > 0:
            name_str = f" when calling {name}" if name is not None else ""
            raise AssertionError("No warning raised" + name_str)

# 定义一个函数，用于确保函数或代码块中产生特定类型的警告
def assert_warns(warning_class, *args, **kwargs):
    """
    Fail unless the given callable throws the specified warning.

    A warning of class warning_class should be thrown by the callable when
    invoked with arguments args and keyword arguments kwargs.
    If a different type of warning is thrown, it will not be caught.

    If called with all arguments other than the warning class omitted, may be
    used as a context manager:

        with assert_warns(SomeWarning):
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    .. versionadded:: 1.4.0

    Parameters
    ----------
    warning_class : class
        The class defining the warning that `func` is expected to throw.
    func : callable, optional
        Callable to test
    *args : Arguments
        Arguments for `func`.
    **kwargs : Kwargs
        Keyword arguments for `func`.

    Returns
    -------
    The value returned by `func`.

    Examples
    --------
    >>> import warnings
    >>> def deprecated_func(num):
    ...     warnings.warn("Please upgrade", DeprecationWarning)
    ...     return num*num
    >>> with np.testing.assert_warns(DeprecationWarning):
    ...     assert deprecated_func(4) == 16
    >>> # or passing a func
    >>> ret = np.testing.assert_warns(DeprecationWarning, deprecated_func, 4)
    >>> assert ret == 16
    """
    if not args:
        return _assert_warns_context(warning_class)

    func = args[0]
    args = args[1:]
    with _assert_warns_context(warning_class, name=func.__name__):
        return func(*args, **kwargs)

# 定义一个上下文管理器，用于确保在调用期间未产生任何警告
@contextlib.contextmanager
def _assert_no_warnings_context(name=None):
    # 省略具体实现
    pass
    # 设置 __tracebackhide__ 为 True，用于在 py.test 中隐藏回溯信息
    __tracebackhide__ = True  # Hide traceback for py.test
    
    # 使用 warnings 模块捕获所有警告信息
    with warnings.catch_warnings(record=True) as l:
        # 设置警告过滤器，始终显示警告
        warnings.simplefilter("always")
    
        # 执行 yield，允许测试框架执行其中的测试代码
        yield
    
        # 如果捕获到了警告
        if len(l) > 0:
            # 根据是否有名称，决定构造的名称字符串
            name_str = f" when calling {name}" if name is not None else ""
            # 抛出断言错误，显示捕获到的警告信息
            raise AssertionError(f"Got warnings{name_str}: {l}")
# 如果没有传入任何参数，则使用作为上下文管理器调用时，用于确保给定的可调用对象不会产生任何警告
def assert_no_warnings(*args, **kwargs):
    """
    Fail if the given callable produces any warnings.

    If called with all arguments omitted, may be used as a context manager:

        with assert_no_warnings():
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    func : callable
        The callable to test.
    \\*args : Arguments
        Arguments passed to `func`.
    \\*\\*kwargs : Kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    The value returned by `func`.
    """
    # 如果没有传入参数，则调用 _assert_no_warnings_context 函数
    if not args:
        return _assert_no_warnings_context()

    # 第一个参数为要测试的可调用对象
    func = args[0]
    # 剩余的参数作为 func 的参数
    args = args[1:]
    # 使用 _assert_no_warnings_context 作为上下文管理器，并传入 func 的名称作为参数
    with _assert_no_warnings_context(name=func.__name__):
        return func(*args, **kwargs)


# 生成具有不同对齐和偏移的数据以测试 simd 向量化
def _gen_alignment_data(dtype=float32, type="binary", max_size=24):
    """
    generator producing data with different alignment and offsets
    to test simd vectorization

    Parameters
    ----------
    dtype : dtype
        data type to produce
    type : string
        'unary': create data for unary operations, creates one input
                 and output array
        'binary': create data for unary operations, creates two input
                 and output array
    max_size : integer
        maximum size of data to produce

    Returns
    -------
    if type is 'unary' yields one output, one input array and a message
    containing information on the data
    if type is 'binary' yields one output array, two input array and a message
    containing information on the data

    """
    # 格式化字符串，用于输出 'unary' 类型数据的信息
    ufmt = "unary offset=(%d, %d), size=%d, dtype=%r, %s"
    # 格式化字符串，用于输出 'binary' 类型数据的信息
    bfmt = "binary offset=(%d, %d, %d), size=%d, dtype=%r, %s"
    # 循环变量 o 的取值范围为 [0, 1, 2]
    for o in range(3):
        # 循环变量 s 的取值范围从 o+2 到 max(o+3, max_size)
        for s in range(o + 2, max(o + 3, max_size)):
            # 如果 type 等于 "unary"，则执行以下操作
            if type == "unary":

                # 定义一个返回从 s 开始的数组的函数 inp
                def inp():
                    return arange(s, dtype=dtype)[o:]

                # 创建一个空的数组 out，长度为 s，数据类型为 dtype
                out = empty((s,), dtype=dtype)[o:]
                # 生成一个包含 out、inp()、格式化字符串的元组，并使用 yield 返回
                yield out, inp(), ufmt % (o, o, s, dtype, "out of place")
                # 将 inp() 的结果赋给 d
                d = inp()
                # 生成一个包含 d、d、格式化字符串的元组，并使用 yield 返回
                yield d, d, ufmt % (o, o, s, dtype, "in place")
                # 生成一个包含 out[1:], inp()[:-1]、格式化字符串的元组，并使用 yield 返回
                yield out[1:], inp()[:-1], ufmt % (
                    o + 1,
                    o,
                    s - 1,
                    dtype,
                    "out of place",
                )
                # 生成一个包含 out[:-1], inp()[1:]、格式化字符串的元组，并使用 yield 返回
                yield out[:-1], inp()[1:], ufmt % (
                    o,
                    o + 1,
                    s - 1,
                    dtype,
                    "out of place",
                )
                # 生成一个包含 inp()[:-1], inp()[1:]、格式化字符串的元组，并使用 yield 返回
                yield inp()[:-1], inp()[1:], ufmt % (o, o + 1, s - 1, dtype, "aliased")
                # 生成一个包含 inp()[1:], inp()[:-1]、格式化字符串的元组，并使用 yield 返回
                yield inp()[1:], inp()[:-1], ufmt % (o + 1, o, s - 1, dtype, "aliased")

            # 如果 type 等于 "binary"，则执行以下操作
            if type == "binary":

                # 定义一个返回从 s 开始的数组的函数 inp1
                def inp1():
                    return arange(s, dtype=dtype)[o:]

                # 将函数 inp1 赋给变量 inp2
                inp2 = inp1
                # 创建一个空的数组 out，长度为 s，数据类型为 dtype
                out = empty((s,), dtype=dtype)[o:]
                # 生成一个包含 out、inp1()、inp2()、格式化字符串的元组，并使用 yield 返回
                yield out, inp1(), inp2(), bfmt % (o, o, o, s, dtype, "out of place")
                # 将 inp1() 的结果赋给 d
                d = inp1()
                # 生成一个包含 d、d、inp2()、格式化字符串的元组，并使用 yield 返回
                yield d, d, inp2(), bfmt % (o, o, o, s, dtype, "in place1")
                # 将 inp2() 的结果赋给 d
                d = inp2()
                # 生成一个包含 d、inp1()、d、格式化字符串的元组，并使用 yield 返回
                yield d, inp1(), d, bfmt % (o, o, o, s, dtype, "in place2")
                # 生成一个包含 out[1:], inp1()[:-1]、inp2()[:-1]、格式化字符串的元组，并使用 yield 返回
                yield out[1:], inp1()[:-1], inp2()[:-1], bfmt % (
                    o + 1,
                    o,
                    o,
                    s - 1,
                    dtype,
                    "out of place",
                )
                # 生成一个包含 out[:-1], inp1()[1:], inp2()[:-1]、格式化字符串的元组，并使用 yield 返回
                yield out[:-1], inp1()[1:], inp2()[:-1], bfmt % (
                    o,
                    o + 1,
                    o,
                    s - 1,
                    dtype,
                    "out of place",
                )
                # 生成一个包含 out[:-1], inp1()[:-1], inp2()[1:]、格式化字符串的元组，并使用 yield 返回
                yield out[:-1], inp1()[:-1], inp2()[1:], bfmt % (
                    o,
                    o,
                    o + 1,
                    s - 1,
                    dtype,
                    "out of place",
                )
                # 生成一个包含 inp1()[1:], inp1()[:-1], inp2()[:-1]、格式化字符串的元组，并使用 yield 返回
                yield inp1()[1:], inp1()[:-1], inp2()[:-1], bfmt % (
                    o + 1,
                    o,
                    o,
                    s - 1,
                    dtype,
                    "aliased",
                )
                # 生成一个包含 inp1()[:-1], inp1()[1:], inp2()[:-1]、格式化字符串的元组，并使用 yield 返回
                yield inp1()[:-1], inp1()[1:], inp2()[:-1], bfmt % (
                    o,
                    o + 1,
                    o,
                    s - 1,
                    dtype,
                    "aliased",
                )
                # 生成一个包含 inp1()[:-1], inp1()[:-1], inp2()[1:]、格式化字符串的元组，并使用 yield 返回
                yield inp1()[:-1], inp1()[:-1], inp2()[1:], bfmt % (
                    o,
                    o,
                    o + 1,
                    s - 1,
                    dtype,
                    "aliased",
                )
class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"



# 定义一个名为 IgnoreException 的异常类，用于表示由于禁用功能而忽略的异常
class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"



@contextlib.contextmanager
def tempdir(*args, **kwargs):
    """Context manager to provide a temporary test folder.

    All arguments are passed as this to the underlying tempfile.mkdtemp
    function.

    """
    # 创建临时文件夹的上下文管理器
    tmpdir = mkdtemp(*args, **kwargs)
    try:
        yield tmpdir  # 返回临时文件夹路径
    finally:
        shutil.rmtree(tmpdir)  # 清理临时文件夹及其内容



@contextlib.contextmanager
def temppath(*args, **kwargs):
    """Context manager for temporary files.

    Context manager that returns the path to a closed temporary file. Its
    parameters are the same as for tempfile.mkstemp and are passed directly
    to that function. The underlying file is removed when the context is
    exited, so it should be closed at that time.

    Windows does not allow a temporary file to be opened if it is already
    open, so the underlying file must be closed after opening before it
    can be opened again.

    """
    # 创建临时文件的上下文管理器
    fd, path = mkstemp(*args, **kwargs)
    os.close(fd)  # 关闭临时文件
    try:
        yield path  # 返回临时文件路径
    finally:
        os.remove(path)  # 移除临时文件



class clear_and_catch_warnings(warnings.catch_warnings):
    """Context manager that resets warning registry for catching warnings

    Warnings can be slippery, because, whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module.  This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters.  This context manager accepts a sequence of `modules`
    as a keyword argument to its constructor and:

    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.

    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.

    For compatibility with Python 3.0, please consider all arguments to be
    keyword-only.

    Parameters
    ----------
    record : bool, optional
        Specifies whether warnings should be captured by a custom
        implementation of ``warnings.showwarning()`` and be appended to a list
        returned by the context manager. Otherwise None is returned by the
        context manager. The objects appended to the list are arguments whose
        attributes mirror the arguments to ``showwarning()``.
    modules : sequence, optional
        Sequence of modules for which to reset warnings registry on entry and
        restore on exit. To work correctly, all 'ignore' filters should
        filter by one of these modules.

    Examples
    --------
    >>> import warnings
    >>> with np.testing.clear_and_catch_warnings(  # doctest: +SKIP
    ...         modules=[np.core.fromnumeric]):
    ...     warnings.simplefilter('always')
    ...     warnings.filterwarnings('ignore', module='np.core.fromnumeric')
    ...     # do something that raises a warning but ignore those in

    """



# 定义一个名为 clear_and_catch_warnings 的上下文管理器类，用于重置警告注册表以捕获警告
class clear_and_catch_warnings(warnings.catch_warnings):
    """Context manager that resets warning registry for catching warnings

    Warnings can be slippery, because, whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module.  This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters.  This context manager accepts a sequence of `modules`
    as a keyword argument to its constructor and:

    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.

    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.

    For compatibility with Python 3.0, please consider all arguments to be
    keyword-only.

    Parameters
    ----------
    record : bool, optional
        Specifies whether warnings should be captured by a custom
        implementation of ``warnings.showwarning()`` and be appended to a list
        returned by the context manager. Otherwise None is returned by the
        context manager. The objects appended to the list are arguments whose
        attributes mirror the arguments to ``showwarning()``.
    modules : sequence, optional
        Sequence of modules for which to reset warnings registry on entry and
        restore on exit. To work correctly, all 'ignore' filters should
        filter by one of these modules.

    Examples
    --------
    >>> import warnings
    >>> with np.testing.clear_and_catch_warnings(  # doctest: +SKIP
    ...         modules=[np.core.fromnumeric]):
    ...     warnings.simplefilter('always')
    ...     warnings.filterwarnings('ignore', module='np.core.fromnumeric')
    ...     # do something that raises a warning but ignore those in

    """
    ...     # np.core.fromnumeric
    """
    
    class_modules = ()  # 定义一个空元组 class_modules，用于存储模块集合
    
    def __init__(self, record=False, modules=()):
        self.modules = set(modules).union(self.class_modules)  # 初始化 self.modules 为传入模块参数和类模块的并集
        self._warnreg_copies = {}  # 初始化一个空字典 _warnreg_copies，用于存储警告注册表的备份
        super().__init__(record=record)  # 调用父类的初始化方法，传入记录标志参数
        
    def __enter__(self):
        for mod in self.modules:  # 遍历 self.modules 中的每个模块
            if hasattr(mod, "__warningregistry__"):  # 检查模块是否有 "__warningregistry__" 属性
                mod_reg = mod.__warningregistry__  # 获取模块的警告注册表
                self._warnreg_copies[mod] = mod_reg.copy()  # 备份当前的警告注册表到 _warnreg_copies
                mod_reg.clear()  # 清空模块的警告注册表
        return super().__enter__()  # 返回父类的 __enter__() 方法的结果
    
    def __exit__(self, *exc_info):
        super().__exit__(*exc_info)  # 调用父类的 __exit__() 方法，传入异常信息参数
        for mod in self.modules:  # 遍历 self.modules 中的每个模块
            if hasattr(mod, "__warningregistry__"):  # 检查模块是否有 "__warningregistry__" 属性
                mod.__warningregistry__.clear()  # 清空模块的警告注册表
            if mod in self._warnreg_copies:  # 如果模块在 _warnreg_copies 中有备份
                mod.__warningregistry__.update(self._warnreg_copies[mod])  # 恢复模块的警告注册表数据
# 定义一个上下文管理器类 suppress_warnings，功能类似于 warnings.catch_warnings
class suppress_warnings:
    """
    Context manager and decorator doing much the same as
    ``warnings.catch_warnings``.
    
    However, it also provides a filter mechanism to work around
    https://bugs.python.org/issue4180.
    
    This bug causes Python before 3.4 to not reliably show warnings again
    after they have been ignored once (even within catch_warnings). It
    means that no "ignore" filter can be used easily, since following
    tests might need to see the warning. Additionally it allows easier
    specificity for testing warnings and can be nested.
    
    Parameters
    ----------
    forwarding_rule : str, optional
        One of "always", "once", "module", or "location". Analogous to
        the usual warnings module filter mode, it is useful to reduce
        noise mostly on the outmost level. Unsuppressed and unrecorded
        warnings will be forwarded based on this rule. Defaults to "always".
        "location" is equivalent to the warnings "default", match by exact
        location the warning warning originated from.
        
    Notes
    -----
    Filters added inside the context manager will be discarded again
    when leaving it. Upon entering all filters defined outside a
    context will be applied automatically.
    
    When a recording filter is added, matching warnings are stored in the
    ``log`` attribute as well as in the list returned by ``record``.
    
    If filters are added and the ``module`` keyword is given, the
    warning registry of this module will additionally be cleared when
    applying it, entering the context, or exiting it. This could cause
    warnings to appear a second time after leaving the context if they
    were configured to be printed once (default) and were already
    printed before the context was entered.
    
    Nesting this context manager will work as expected when the
    forwarding rule is "always" (default). Unfiltered and unrecorded
    warnings will be passed out and be matched by the outer level.
    On the outmost level they will be printed (or caught by another
    warnings context). The forwarding rule argument can modify this
    behaviour.
    
    Like ``catch_warnings`` this context manager is not threadsafe.
    
    Examples
    --------
    
    With a context manager::
    
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Some text")
            sup.filter(module=np.ma.core)
            log = sup.record(FutureWarning, "Does this occur?")
            command_giving_warnings()
            # The FutureWarning was given once, the filtered warnings were
            # ignored. All other warnings abide outside settings (may be
            # printed/error)
            assert_(len(log) == 1)
            assert_(len(sup.log) == 1)  # also stored in log attribute
    """
    
    # 初始化方法，设置默认的转发规则为 "always"
    def __init__(self, forwarding_rule="always"):
        self.forwarding_rule = forwarding_rule
        self.log = []  # 初始化警告记录列表
    
    # 进入上下文时的操作，返回自身实例
    def __enter__(self):
        return self
    
    # 退出上下文时的操作，清理过滤器
    def __exit__(self, *args):
        pass
    
    # 添加警告过滤器，支持参数是警告类和其他关键字参数
    def filter(self, *args, **kwargs):
        pass
    
    # 记录匹配的警告，返回记录的警告列表
    def record(self, *args, **kwargs):
        pass
    def __init__(self, forwarding_rule="always"):
        self._entered = False
        # 初始化一个空的抑制列表
        self._suppressions = []

        # 检查传入的转发规则是否合法，只能是 "always", "module", "once", "location" 中的一个
        if forwarding_rule not in {"always", "module", "once", "location"}:
            raise ValueError("unsupported forwarding rule.")
        # 设置转发规则
        self._forwarding_rule = forwarding_rule

    def _clear_registries(self):
        if hasattr(warnings, "_filters_mutated"):
            # 如果 warnings 模块有 _filters_mutated 属性，调用它以通知过滤器已经被修改
            warnings._filters_mutated()
            return
        # 否则直接清空警告注册表
        for module in self._tmp_modules:
            if hasattr(module, "__warningregistry__"):
                module.__warningregistry__.clear()

    def _filter(self, category=Warning, message="", module=None, record=False):
        if record:
            record = []  # 存储警告日志的列表
        else:
            record = None

        if self._entered:
            # 如果已经进入了上下文管理器
            if module is None:
                # 如果没有指定模块，按照类别和消息内容过滤警告
                warnings.filterwarnings("always", category=category, message=message)
            else:
                # 如果指定了模块，构建模块名的正则表达式
                module_regex = module.__name__.replace(".", r"\.") + "$"
                # 根据模块名、类别和消息内容过滤警告
                warnings.filterwarnings(
                    "always", category=category, message=message, module=module_regex
                )
                # 将该模块添加到临时模块集合中，并清除相关的警告注册表
                self._tmp_modules.add(module)
                self._clear_registries()

            # 将当前的抑制信息添加到临时抑制列表中
            self._tmp_suppressions.append(
                (category, message, re.compile(message, re.I), module, record)
            )
        else:
            # 如果尚未进入上下文管理器，则将抑制信息添加到主抑制列表中
            self._suppressions.append(
                (category, message, re.compile(message, re.I), module, record)
            )

        return record
    def filter(self, category=Warning, message="", module=None):
        """
        Add a new suppressing filter or apply it if the state is entered.

        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.

        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        # 调用内部方法 _filter，传递参数进行过滤操作，并设置不记录日志
        self._filter(category=category, message=message, module=module, record=False)

    def record(self, category=Warning, message="", module=None):
        """
        Append a new recording filter or apply it if the state is entered.

        All warnings matching will be appended to the ``log`` attribute.

        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.

        Returns
        -------
        log : list
            A list which will be filled with all matched warnings.

        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        # 调用内部方法 _filter，传递参数进行过滤操作，并设置记录日志到 self.log 属性
        return self._filter(
            category=category, message=message, module=module, record=True
        )

    def __enter__(self):
        if self._entered:
            raise RuntimeError("cannot enter suppress_warnings twice.")

        # 备份当前警告显示函数和过滤器
        self._orig_show = warnings.showwarning
        self._filters = warnings.filters
        warnings.filters = self._filters[:]  # 复制过滤器列表，避免直接引用

        self._entered = True  # 设置已进入上下文状态标志
        self._tmp_suppressions = []  # 临时存储的警告抑制列表
        self._tmp_modules = set()  # 临时存储的模块集合
        self._forwarded = set()  # 已转发的警告集合

        self.log = []  # 重置全局日志列表（不需要保留相同的列表）

        # 遍历已定义的警告抑制规则列表
        for cat, mess, _, mod, log in self._suppressions:
            if log is not None:
                del log[:]  # 清空日志列表
            if mod is None:
                warnings.filterwarnings("always", category=cat, message=mess)
            else:
                module_regex = mod.__name__.replace(".", r"\.") + "$"
                warnings.filterwarnings(
                    "always", category=cat, message=mess, module=module_regex
                )
                self._tmp_modules.add(mod)  # 将模块添加到临时集合中

        warnings.showwarning = self._showwarning  # 设置警告显示函数为自定义函数
        self._clear_registries()  # 清理注册表信息

        return self  # 返回当前对象作为上下文管理器的上下文
    # 在退出上下文时，恢复警告显示和过滤器，并清理注册表
    def __exit__(self, *exc_info):
        warnings.showwarning = self._orig_show  # 恢复原始警告显示函数
        warnings.filters = self._filters  # 恢复原始警告过滤器
        self._clear_registries()  # 清理注册表
        self._entered = False  # 标记上下文已退出
        del self._orig_show  # 删除保存的原始警告显示函数引用
        del self._filters  # 删除保存的原始警告过滤器引用

    # 处理警告消息的方法，根据规则记录或忽略特定类型的警告
    def _showwarning(
        self, message, category, filename, lineno, *args, use_warnmsg=None, **kwargs
    ):
        # 逆序遍历已记录和临时记录的抑制规则
        for cat, _, pattern, mod, rec in (self._suppressions + self._tmp_suppressions)[::-1]:
            # 检查消息类别和模式是否匹配抑制规则
            if issubclass(category, cat) and pattern.match(message.args[0]) is not None:
                if mod is None:
                    # 消息和类别匹配，记录或忽略
                    if rec is not None:
                        msg = WarningMessage(
                            message, category, filename, lineno, **kwargs
                        )
                        self.log.append(msg)  # 记录警告消息
                        rec.append(msg)  # 记录到规则的记录列表
                    return
                # 使用startswith，因为warnings会从文件名中去掉 .pyc/.pyo 的后缀
                elif mod.__file__.startswith(filename):
                    # 消息和模块（文件名）匹配
                    if rec is not None:
                        msg = WarningMessage(
                            message, category, filename, lineno, **kwargs
                        )
                        self.log.append(msg)  # 记录警告消息
                        rec.append(msg)  # 记录到规则的记录列表
                    return

        # 没有匹配的过滤规则，根据转发规则将警告传递给外部处理器
        if self._forwarding_rule == "always":
            if use_warnmsg is None:
                self._orig_show(message, category, filename, lineno, *args, **kwargs)
            else:
                self._orig_showmsg(use_warnmsg)
            return

        # 根据转发规则的不同标准判断是否转发警告
        if self._forwarding_rule == "once":
            signature = (message.args, category)
        elif self._forwarding_rule == "module":
            signature = (message.args, category, filename)
        elif self._forwarding_rule == "location":
            signature = (message.args, category, filename, lineno)

        if signature in self._forwarded:
            return
        self._forwarded.add(signature)  # 将警告签名加入已转发集合

        # 根据是否提供 use_warnmsg 参数选择转发警告
        if use_warnmsg is None:
            self._orig_show(message, category, filename, lineno, *args, **kwargs)
        else:
            self._orig_showmsg(use_warnmsg)

    # 创建函数装饰器，应用于函数以应用特定的警告抑制规则
    def __call__(self, func):
        """
        Function decorator to apply certain suppressions to a whole
        function.
        """

        @wraps(func)
        def new_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return new_func
@contextlib.contextmanager
def _assert_no_gc_cycles_context(name=None):
    __tracebackhide__ = True  # Hide traceback for py.test

    # 如果没有启用引用计数，测试就没有意义
    if not HAS_REFCOUNT:
        # 如果没有启用引用计数，直接结束上下文管理器
        yield
        return

    # 断言当前已经启用了垃圾回收
    assert_(gc.isenabled())
    # 禁用垃圾回收
    gc.disable()
    # 保存当前的垃圾回收调试状态
    gc_debug = gc.get_debug()
    try:
        # 尝试进行多次垃圾回收，最多100次
        for i in range(100):
            if gc.collect() == 0:
                break
        else:
            # 如果无法完全回收垃圾，抛出运行时错误
            raise RuntimeError(
                "Unable to fully collect garbage - perhaps a __del__ method "
                "is creating more reference cycles?"
            )

        # 设置垃圾回收调试模式为保存所有对象
        gc.set_debug(gc.DEBUG_SAVEALL)
        # 执行 yield 表达式
        yield
        # 检查在上下文管理器内是否创建了引用循环
        # gc.collect 返回找到的处于循环中不可达对象的数量
        n_objects_in_cycles = gc.collect()
        # 记录在循环中的对象列表
        objects_in_cycles = gc.garbage[:]
    finally:
        # 清空 gc.garbage 列表
        del gc.garbage[:]
        # 恢复之前的垃圾回收调试模式
        gc.set_debug(gc_debug)
        # 启用垃圾回收
        gc.enable()

    # 如果发现了引用循环，抛出断言错误
    if n_objects_in_cycles:
        name_str = f" when calling {name}" if name is not None else ""
        raise AssertionError(
            "Reference cycles were found{}: {} objects were collected, "
            "of which {} are shown below:{}".format(
                name_str,
                n_objects_in_cycles,
                len(objects_in_cycles),
                "".join(
                    "\n  {} object with id={}:\n    {}".format(
                        type(o).__name__,
                        id(o),
                        pprint.pformat(o).replace("\n", "\n    "),
                    )
                    for o in objects_in_cycles
                ),
            )
        )


def assert_no_gc_cycles(*args, **kwargs):
    """
    Fail if the given callable produces any reference cycles.

    If called with all arguments omitted, may be used as a context manager:

        with assert_no_gc_cycles():
            do_something()

    .. versionadded:: 1.15.0

    Parameters
    ----------
    func : callable
        The callable to test.
    \\*args : Arguments
        Arguments passed to `func`.
    \\*\\*kwargs : Kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    Nothing. The result is deliberately discarded to ensure that all cycles
    are found.

    """
    # 如果没有传入任何参数，直接调用 _assert_no_gc_cycles_context 函数
    if not args:
        return _assert_no_gc_cycles_context()

    # 否则，获取 func 参数和其余的 args、kwargs 参数
    func = args[0]
    args = args[1:]
    # 使用 _assert_no_gc_cycles_context 作为上下文管理器，传递 func 函数名作为 name 参数
    with _assert_no_gc_cycles_context(name=func.__name__):
        func(*args, **kwargs)


def break_cycles():
    """
    Break reference cycles by calling gc.collect
    Objects can call other objects' methods (for instance, another object's
     __del__) inside their own __del__. On PyPy, the interpreter only runs
    between calls to gc.collect, so multiple calls are needed to completely
    release all cycles.
    """
    # 执行垃圾回收
    gc.collect()
    # 如果运行环境是 PyPy，执行以下操作以确保所有的终结器都被调用
    if IS_PYPY:
        # 执行多次垃圾回收操作，以确保所有的对象都被适当地清理和释放
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()
def requires_memory(free_bytes):
    """Decorator to skip a test if not enough memory is available"""
    import pytest  # 导入 pytest 模块

    def decorator(func):
        @wraps(func)  # 保留原函数的元信息
        def wrapper(*a, **kw):
            msg = check_free_memory(free_bytes)  # 调用 check_free_memory 函数检查内存
            if msg is not None:
                pytest.skip(msg)  # 如果内存不足，则跳过测试

            try:
                return func(*a, **kw)  # 执行被装饰的函数
            except MemoryError:
                # 捕获内存错误，不将其视为失败
                pytest.xfail("MemoryError raised")

        return wrapper  # 返回包装后的函数

    return decorator  # 返回装饰器函数本身


def check_free_memory(free_bytes):
    """
    Check whether `free_bytes` amount of memory is currently free.
    Returns: None if enough memory available, otherwise error message
    """
    env_var = "NPY_AVAILABLE_MEM"  # 设置环境变量名
    env_value = os.environ.get(env_var)  # 获取环境变量的值
    if env_value is not None:
        try:
            mem_free = _parse_size(env_value)  # 尝试解析环境变量值为内存大小
        except ValueError as exc:
            raise ValueError(  # 抛出异常，说明环境变量格式无效
                f"Invalid environment variable {env_var}: {exc}"
            )

        msg = (
            f"{free_bytes/1e9} GB memory required, but environment variable "
            f"NPY_AVAILABLE_MEM={env_value} set"
        )
    else:
        mem_free = _get_mem_available()  # 获取系统可用内存大小

        if mem_free is None:
            msg = (
                "Could not determine available memory; set NPY_AVAILABLE_MEM "
                "environment variable (e.g. NPY_AVAILABLE_MEM=16GB) to run "
                "the test."
            )
            mem_free = -1
        else:
            msg = (
                f"{free_bytes/1e9} GB memory required, but {mem_free/1e9} GB available"
            )

    return msg if mem_free < free_bytes else None  # 如果可用内存小于所需内存，返回错误消息，否则返回 None


def _parse_size(size_str):
    """Convert memory size strings ('12 GB' etc.) to float"""
    suffixes = {  # 定义内存大小后缀及其对应的倍数关系
        "": 1,
        "b": 1,
        "k": 1000,
        "m": 1000**2,
        "g": 1000**3,
        "t": 1000**4,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }

    size_re = re.compile(
        r"^\s*(\d+|\d+\.\d+)\s*({})\s*$".format("|".join(suffixes.keys())), re.I  # 编译正则表达式，匹配内存大小字符串
    )

    m = size_re.match(size_str.lower())  # 匹配给定的内存大小字符串
    if not m or m.group(2) not in suffixes:
        raise ValueError(f"value {size_str!r} not a valid size")  # 如果匹配失败或者后缀无效，抛出异常
    return int(float(m.group(1)) * suffixes[m.group(2)])  # 计算内存大小并返回


def _get_mem_available():
    """Return available memory in bytes, or None if unknown."""
    try:
        import psutil  # 尝试导入 psutil 模块，用于获取系统内存信息

        return psutil.virtual_memory().available  # 返回系统可用内存大小
    except (ImportError, AttributeError):
        pass  # 如果导入失败或者属性错误，直接忽略
    # 如果运行环境是 Linux 系统
    if sys.platform.startswith("linux"):
        # 初始化一个空字典来存储内存信息
        info = {}
        
        # 打开并读取 /proc/meminfo 文件，该文件包含系统内存信息
        with open("/proc/meminfo") as f:
            # 逐行读取文件内容
            for line in f:
                # 将每行内容按空格分割为列表
                p = line.split()
                # 使用列表第一个元素（去除末尾的冒号并转换为小写）作为键，将第二个元素（单位为 KB）乘以 1024 后作为值存入字典
                info[p[0].strip(":").lower()] = int(p[1]) * 1024
            
        # 如果字典中有 "memavailable" 键，表示系统版本 >= 3.14
        if "memavailable" in info:
            # 返回 memavailable 的值，即可用内存量
            return info["memavailable"]
        else:
            # 返回 memfree 和 cached 两者之和，表示可用内存总量（包括空闲内存和缓存）
            return info["memfree"] + info["cached"]
    
    # 如果不是 Linux 系统，则返回空值 None
    return None
# 创建一个装饰器函数，用于在测试期间临时关闭跟踪功能。
# 这在检查引用计数的测试中很有必要，因为跟踪本身会影响引用计数。
def _no_tracing(func):
    # 检查当前系统是否支持获取跟踪对象的功能
    if not hasattr(sys, "gettrace"):
        return func
    else:
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取当前的跟踪对象，并保存为原始跟踪对象
            original_trace = sys.gettrace()
            try:
                # 设置跟踪对象为 None，以关闭跟踪
                sys.settrace(None)
                # 调用被装饰的函数，不再进行跟踪
                return func(*args, **kwargs)
            finally:
                # 恢复原始的跟踪对象
                sys.settrace(original_trace)

        return wrapper


def _get_glibc_version():
    try:
        # 获取当前 GNU libc 的版本信息
        ver = os.confstr("CS_GNU_LIBC_VERSION").rsplit(" ")[1]
    except Exception as inst:
        # 如果获取版本信息出错，则将版本号设置为 "0.0"
        ver = "0.0"

    return ver


# 获取当前系统的 GNU libc 版本信息
_glibcver = _get_glibc_version()


def _glibc_older_than(x):
    # 检查当前系统的 GNU libc 版本是否低于给定版本 x
    return _glibcver != "0.0" and _glibcver < x
```