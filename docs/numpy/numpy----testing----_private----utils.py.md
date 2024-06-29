# `.\numpy\numpy\testing\_private\utils.py`

```
"""
Utility function to facilitate testing.

"""
import os  # 导入操作系统功能模块
import sys  # 导入系统模块
import platform  # 导入平台信息模块
import re  # 导入正则表达式模块
import gc  # 导入垃圾回收模块
import operator  # 导入操作符模块
import warnings  # 导入警告模块
from functools import partial, wraps  # 导入函数工具模块中的partial和wraps函数
import shutil  # 导入文件操作模块
import contextlib  # 导入上下文管理模块
from tempfile import mkdtemp, mkstemp  # 从临时文件模块中导入创建临时目录和文件函数
from unittest.case import SkipTest  # 从单元测试模块导入跳过测试异常
from warnings import WarningMessage  # 从警告模块导入警告消息类型
import pprint  # 导入美化打印模块
import sysconfig  # 导入系统配置模块

import numpy as np  # 导入NumPy库
from numpy._core import (  # 导入NumPy核心模块的若干数据类型和函数
     intp, float32, empty, arange, array_repr, ndarray, isnat, array)
from numpy import isfinite, isnan, isinf  # 从NumPy导入判断函数

import numpy.linalg._umath_linalg  # 导入NumPy线性代数模块的基本运算
from numpy._utils import _rename_parameter  # 导入NumPy工具模块中的参数重命名函数

from io import StringIO  # 从IO模块中导入字符串IO类

__all__ = [  # 定义公共接口列表，包含若干函数名字符串
        'assert_equal', 'assert_almost_equal', 'assert_approx_equal',
        'assert_array_equal', 'assert_array_less', 'assert_string_equal',
        'assert_array_almost_equal', 'assert_raises', 'build_err_msg',
        'decorate_methods', 'jiffies', 'memusage', 'print_assert_equal',
        'rundocs', 'runstring', 'verbose', 'measure',
        'assert_', 'assert_array_almost_equal_nulp', 'assert_raises_regex',
        'assert_array_max_ulp', 'assert_warns', 'assert_no_warnings',
        'assert_allclose', 'IgnoreException', 'clear_and_catch_warnings',
        'SkipTest', 'KnownFailureException', 'temppath', 'tempdir', 'IS_PYPY',
        'HAS_REFCOUNT', "IS_WASM", 'suppress_warnings', 'assert_array_compare',
        'assert_no_gc_cycles', 'break_cycles', 'HAS_LAPACK64', 'IS_PYSTON',
        '_OLD_PROMOTION', 'IS_MUSL', '_SUPPORTS_SVE', 'NOGIL_BUILD',
        'IS_EDITABLE'
        ]


class KnownFailureException(Exception):
    '''Raise this exception to mark a test as a known failing test.'''
    pass


KnownFailureTest = KnownFailureException  # 为向后兼容而定义的已知失败测试异常

verbose = 0  # 设定初始的详细信息级别为0

IS_WASM = platform.machine() in ["wasm32", "wasm64"]  # 判断当前平台是否为WebAssembly

IS_PYPY = sys.implementation.name == 'pypy'  # 判断当前解释器是否为PyPy

IS_PYSTON = hasattr(sys, "pyston_version_info")  # 判断当前解释器是否为Pyston

IS_EDITABLE = not bool(np.__path__) or 'editable' in np.__path__[0]  # 判断NumPy是否处于可编辑状态

HAS_REFCOUNT = getattr(sys, 'getrefcount', None) is not None and not IS_PYSTON  # 判断当前环境是否支持引用计数

HAS_LAPACK64 = numpy.linalg._umath_linalg._ilp64  # 检查NumPy线性代数模块是否支持64位整型

_OLD_PROMOTION = lambda: np._get_promotion_state() == 'legacy'  # 定义检查旧式类型提升状态的匿名函数

IS_MUSL = False  # 初始化是否为Musl libc的标志位为假

# 使用另一种方式从sysconfig获取主机GNU类型配置，若包含'musl'则设定IS_MUSL为真
_v = sysconfig.get_config_var('HOST_GNU_TYPE') or ''
if 'musl' in _v:
    IS_MUSL = True

NOGIL_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))  # 检查当前构建是否不支持全局解释器锁（GIL）

def assert_(val, msg=''):
    """
    Assert that works in release mode.
    Accepts callable msg to allow deferring evaluation until failure.

    The Python built-in ``assert`` does not work when executing code in
    optimized mode (the ``-O`` flag) - no byte-code is generated for it.

    For documentation on usage, refer to the Python documentation.

    """
    __tracebackhide__ = True  # 隐藏异常的追踪信息，用于pytest
    if not val:
        try:
            smsg = msg()  # 尝试调用可调用的消息参数以延迟评估直到失败
        except TypeError:
            smsg = msg
        raise AssertionError(smsg)  # 抛出断言错误异常
if os.name == 'nt':
    # 当操作系统是 Windows 时，定义获取性能属性的函数 GetPerformanceAttributes
    # object: 对象名称，counter: 计数器名称，instance: 实例名，默认为 None
    # inum: 实例编号，默认为 -1，format: 格式，默认为 None，machine: 机器名，默认为 None
    def GetPerformanceAttributes(object, counter, instance=None,
                                 inum=-1, format=None, machine=None):
        # 许多计数器需要两次采样才能给出准确结果，
        # 包括 "% Processor Time"（因为按定义，线程的 CPU 使用率在任何时刻都是 0 或 100）。
        # 要读取这类计数器，应保持计数器打开状态，并在需要时调用 CollectQueryData()。
        # 参见 http://msdn.microsoft.com/library/en-us/dnperfmo/html/perfmonpt2.asp
        # "AddCounter" 过程将强制 CPU 达到 100%，这解释了这一现象 :)
        import win32pdh
        if format is None:
            format = win32pdh.PDH_FMT_LONG
        # 构建计数器路径
        path = win32pdh.MakeCounterPath( (machine, object, instance, None,
                                          inum, counter))
        # 打开查询
        hq = win32pdh.OpenQuery()
        try:
            # 添加计数器到查询
            hc = win32pdh.AddCounter(hq, path)
            try:
                # 收集查询数据
                win32pdh.CollectQueryData(hq)
                # 获取格式化的计数器值
                type, val = win32pdh.GetFormattedCounterValue(hc, format)
                return val
            finally:
                # 移除计数器
                win32pdh.RemoveCounter(hc)
        finally:
            # 关闭查询
            win32pdh.CloseQuery(hq)

    # 定义获取内存使用情况的函数 memusage
    # processName: 进程名称，默认为 "python"，instance: 实例编号，默认为 0
    def memusage(processName="python", instance=0):
        # 导入 win32pdh 模块
        import win32pdh
        # 调用 GetPerformanceAttributes 函数获取 "Process" 对象的 "Virtual Bytes" 计数器值
        return GetPerformanceAttributes("Process", "Virtual Bytes",
                                        processName, instance,
                                        win32pdh.PDH_FMT_LONG, None)
elif sys.platform[:5] == 'linux':
    # 当操作系统是 Linux 时，定义获取内存使用情况的函数 memusage
    def memusage(_proc_pid_stat=f'/proc/{os.getpid()}/stat'):
        """
        返回当前运行的 Python 进程的虚拟内存大小（字节）。
        """
        try:
            with open(_proc_pid_stat) as f:
                l = f.readline().split(' ')
            return int(l[22])
        except Exception:
            return
else:
    # 当操作系统不是 Windows 也不是 Linux 时，定义未实现的 memusage 函数
    def memusage():
        """
        返回当前运行的 Python 进程的内存使用情况。[未实现]
        """
        raise NotImplementedError

if sys.platform[:5] == 'linux':
    # 当操作系统是 Linux 时，定义获取 jiffies 数的函数 jiffies
    def jiffies(_proc_pid_stat=f'/proc/{os.getpid()}/stat', _load_time=[]):
        """
        返回已经消耗的 jiffies 数。

        返回该进程在用户模式下已经被调度的 jiffies 数（1/100 秒）。
        参见 man 5 proc。
        """
        import time
        if not _load_time:
            _load_time.append(time.time())
        try:
            with open(_proc_pid_stat) as f:
                l = f.readline().split(' ')
            return int(l[13])
        except Exception:
            return int(100*(time.time()-_load_time[0]))
else:
    # 当操作系统不是 Linux 时，注释说明 os.getpid 可能不可用，使用 time 会安全但不精确
    # 定义名为 jiffies 的函数，用于返回已经过的 jiffies 数量
    def jiffies(_load_time=[]):
        """
        Return number of jiffies elapsed.

        Return number of jiffies (1/100ths of a second) that this
        process has been scheduled in user mode. See man 5 proc.

        """
        # 导入时间模块
        import time
        # 如果 _load_time 列表为空，则将当前时间加入列表中
        if not _load_time:
            _load_time.append(time.time())
        # 返回从开始计时到现在经过的 jiffies 数量（1 jiffy = 1/100 秒）
        return int(100*(time.time()-_load_time[0]))
# 构建错误消息的函数，用于比较多个对象的内容是否相等，并生成错误信息
def build_err_msg(arrays, err_msg, header='Items are not equal:',
                  verbose=True, names=('ACTUAL', 'DESIRED'), precision=8):
    # 初始化错误消息列表，包含指定的标题
    msg = ['\n' + header]
    # 将错误消息转换为字符串类型
    err_msg = str(err_msg)
    # 如果存在错误消息且不包含换行符且长度小于79减去标题长度，则将其作为一行添加到消息中
    if err_msg:
        if err_msg.find('\n') == -1 and len(err_msg) < 79 - len(header):
            msg = [msg[0] + ' ' + err_msg]
        else:
            # 否则作为单独的行添加到消息中
            msg.append(err_msg)
    # 如果 verbose 参数为 True，则详细显示每个数组的内容
    if verbose:
        # 遍历给定的数组列表
        for i, a in enumerate(arrays):
            # 如果数组是 ndarray 类型，则使用指定精度生成其字符串表示
            if isinstance(a, ndarray):
                r_func = partial(array_repr, precision=precision)
            else:
                r_func = repr

            try:
                # 尝试生成数组或对象的字符串表示
                r = r_func(a)
            except Exception as exc:
                # 如果生成失败，记录异常信息
                r = f'[repr failed for <{type(a).__name__}>: {exc}]'
            # 如果字符串表示包含超过三行，则只保留前三行，并在末尾添加省略号
            if r.count('\n') > 3:
                r = '\n'.join(r.splitlines()[:3])
                r += '...'
            # 将生成的字符串表示和其名称添加到消息列表中
            msg.append(f' {names[i]}: {r}')
    # 返回所有消息内容的字符串形式
    return '\n'.join(msg)


def assert_equal(actual, desired, err_msg='', verbose=True, *, strict=False):
    """
    Raises an AssertionError if two objects are not equal.

    Given two objects (scalars, lists, tuples, dictionaries or numpy arrays),
    check that all elements of these objects are equal. An exception is raised
    at the first conflicting values.

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
    strict : bool, optional
        If True and either of the `actual` and `desired` arguments is an array,
        raise an ``AssertionError`` when either the shape or the data type of
        the arguments does not match. If neither argument is an array, this
        parameter has no effect.

        .. versionadded:: 2.0.0

    Raises
    ------
    AssertionError
        If actual and desired are not equal.

    See Also
    --------
    assert_allclose
    assert_array_almost_equal_nulp,
    assert_array_max_ulp,

    Notes
    -----
    By default, when one of `actual` and `desired` is a scalar and the other is
    an array, the function checks that each element of the array is equal to
    the scalar. This behaviour can be disabled by setting ``strict==True``.

    Examples
    --------
    >>> np.testing.assert_equal([4, 5], [4, 6])
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal:
    item=1
     ACTUAL: 5
     DESIRED: 6
    """
    """
    The following comparison does not raise an exception.  There are NaNs
    in the inputs, but they are in the same positions.

    >>> np.testing.assert_equal(np.array([1.0, 2.0, np.nan]), [1, 2, np.nan])

    As mentioned in the Notes section, `assert_equal` has special
    handling for scalars when one of the arguments is an array.
    Here, the test checks that each value in `x` is 3:

    >>> x = np.full((2, 5), fill_value=3)
    >>> np.testing.assert_equal(x, 3)

    Use `strict` to raise an AssertionError when comparing a scalar with an
    array of a different shape:

    >>> np.testing.assert_equal(x, 3, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (shapes (2, 5), () mismatch)
     ACTUAL: array([[3, 3, 3, 3, 3],
           [3, 3, 3, 3, 3]])
     DESIRED: array(3)

    The `strict` parameter also ensures that the array data types match:

    >>> x = np.array([2, 2, 2])
    >>> y = np.array([2., 2., 2.], dtype=np.float32)
    >>> np.testing.assert_equal(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not equal
    <BLANKLINE>
    (dtypes int64, float32 mismatch)
     ACTUAL: array([2, 2, 2])
     DESIRED: array([2., 2., 2.], dtype=float32)
    """
    __tracebackhide__ = True  # Hide traceback for py.test

    # If `desired` is a dictionary, compare it with `actual` as dictionaries
    if isinstance(desired, dict):
        # Ensure `actual` is also a dictionary; otherwise, raise an AssertionError
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        # Assert that the lengths of `actual` and `desired` dictionaries are equal
        assert_equal(len(actual), len(desired), err_msg, verbose)
        # Iterate through items in `desired` dictionary
        for k, i in desired.items():
            # Check if each key `k` in `desired` exists in `actual`; raise AssertionError if not
            if k not in actual:
                raise AssertionError(repr(k))
            # Recursively assert equality for the corresponding values
            assert_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}',
                         verbose)
        return

    # If both `desired` and `actual` are lists or tuples, compare them element-wise
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        # Assert that the lengths of `actual` and `desired` lists/tuples are equal
        assert_equal(len(actual), len(desired), err_msg, verbose)
        # Iterate through indices of `desired` list/tuple
        for k in range(len(desired)):
            # Recursively assert equality for each item at index `k`
            assert_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}',
                         verbose)
        return

    # Import necessary functions and handle comparison for numpy arrays (`ndarray`)
    from numpy._core import ndarray, isscalar, signbit
    from numpy import iscomplexobj, real, imag
    if isinstance(actual, ndarray) or isinstance(desired, ndarray):
        return assert_array_equal(actual, desired, err_msg, verbose,
                                  strict=strict)

    # If none of the above conditions match, build an error message and handle complex numbers
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)

    # Separate complex numbers into real and imaginary parts to handle special cases
    # like nan, inf, and negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except (ValueError, TypeError):
        usecomplex = False
    ```
    # 如果使用复数，进行复数对象的处理
    if usecomplex:
        # 检查 actual 是否为复数对象，分别提取实部和虚部
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual  # 如果不是复数对象，实部为 actual，虚部为 0
            actuali = 0
        # 检查 desired 是否为复数对象，分别提取实部和虚部
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired  # 如果不是复数对象，实部为 desired，虚部为 0
            desiredi = 0
        try:
            # 断言实部和虚部分别相等
            assert_equal(actualr, desiredr)
            assert_equal(actuali, desiredi)
        except AssertionError:
            raise AssertionError(msg)  # 抛出 AssertionError，指定错误消息

    # isscalar 测试，检查例如 [np.nan] != np.nan 的情况
    if isscalar(desired) != isscalar(actual):
        raise AssertionError(msg)  # 抛出 AssertionError，指定错误消息

    try:
        # 检查 desired 和 actual 是否为 NaT (Not a Time) 类型
        isdesnat = isnat(desired)
        isactnat = isnat(actual)
        # 检查 desired 和 actual 的数据类型是否匹配
        dtypes_match = (np.asarray(desired).dtype.type ==
                        np.asarray(actual).dtype.type)
        if isdesnat and isactnat:
            # 如果两者均为 NaT，并且数据类型相同（datetime 或 timedelta），则认为它们相等
            if dtypes_match:
                return
            else:
                raise AssertionError(msg)  # 抛出 AssertionError，指定错误消息

    except (TypeError, ValueError, NotImplementedError):
        pass

    # 处理 Inf/nan/negative zero 的情况
    try:
        # 检查 desired 和 actual 是否为 nan
        isdesnan = isnan(desired)
        isactnan = isnan(actual)
        if isdesnan and isactnan:
            return  # 如果两者均为 nan，则认为它们相等

        # 对于浮点数，特别处理 signed zero
        array_actual = np.asarray(actual)
        array_desired = np.asarray(desired)
        if (array_actual.dtype.char in 'Mm' or
                array_desired.dtype.char in 'Mm'):
            # 对于 datetime64 和 timedelta64，版本 1.18 之前的版本中，isnan 失败。
            # 现在它成功了，但与不同类型的标量进行比较会发出 DeprecationWarning。
            # 通过跳过下一个检查来避免这种情况
            raise NotImplementedError('cannot compare to a scalar '
                                      'with a different type')

        if desired == 0 and actual == 0:
            # 如果 desired 和 actual 均为 0，检查它们的符号位是否相同
            if not signbit(desired) == signbit(actual):
                raise AssertionError(msg)  # 抛出 AssertionError，指定错误消息

    except (TypeError, ValueError, NotImplementedError):
        pass

    try:
        # 显式使用 __eq__ 进行比较，解决 gh-2552
        if not (desired == actual):
            raise AssertionError(msg)  # 抛出 AssertionError，指定错误消息

    except (DeprecationWarning, FutureWarning) as e:
        # 处理两种类型无法进行比较的情况
        if 'elementwise == comparison' in e.args[0]:
            raise AssertionError(msg)  # 抛出 AssertionError，指定错误消息
        else:
            raise  # 抛出原始异常
# 定义一个函数，用于比较两个对象是否相等，如果测试失败则打印错误信息
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
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 1])
    >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 2])
    Traceback (most recent call last):
    ...
    AssertionError: Test XYZ of func xyz failed
    ACTUAL:
    [0, 1]
    DESIRED:
    [0, 2]

    """
    __tracebackhide__ = True  # 用于在 py.test 中隐藏回溯信息
    import pprint

    # 如果 actual 不等于 desired，则抛出 AssertionError
    if not (actual == desired):
        # 创建一个字符串流对象用于存储错误消息
        msg = StringIO()
        msg.write(test_string)  # 写入测试字符串信息
        msg.write(' failed\nACTUAL: \n')
        pprint.pprint(actual, msg)  # 使用 pprint 格式化输出 actual 到消息流
        msg.write('DESIRED: \n')
        pprint.pprint(desired, msg)  # 使用 pprint 格式化输出 desired 到消息流
        raise AssertionError(msg.getvalue())  # 抛出包含消息流内容的 AssertionError

# 装饰器函数，用于向用户警告不兼容 NEP50 的特性
@np._no_nep50_warning()
def assert_almost_equal(actual, desired, decimal=7, err_msg='', verbose=True):
    """
    Raises an AssertionError if two items are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies that the elements of `actual` and `desired` satisfy::

        abs(desired-actual) < float64(1.5 * 10**(-decimal))

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
    >>> from numpy.testing import assert_almost_equal
    >>> assert_almost_equal(2.3333333333333, 2.33333334)
    >>> assert_almost_equal(2.3333333333333, 2.33333334, decimal=10)
    Traceback (most recent call last):
        ...
    AssertionError:

    """
    # 使用__tracebackhide__参数来隐藏pytest测试框架中的异常回溯信息
    __tracebackhide__ = True
    
    # 导入所需的模块和函数
    from numpy._core import ndarray
    from numpy import iscomplexobj, real, imag
    
    # 如果actual或desired是复数对象，则分别提取其实部和虚部，以便正确处理nan、inf和负零值
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except ValueError:
        usecomplex = False
    
    # 构建错误消息的辅助函数
    def _build_err_msg():
        header = ('Arrays are not almost equal to %d decimals' % decimal)
        return build_err_msg([actual, desired], err_msg, verbose=verbose,
                             header=header)
    
    # 如果涉及到复数，分别检查实部和虚部是否几乎相等
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
            assert_almost_equal(actualr, desiredr, decimal=decimal)
            assert_almost_equal(actuali, desiredi, decimal=decimal)
        except AssertionError:
            raise AssertionError(_build_err_msg())
    
    # 如果涉及到数组、元组或列表，则调用assert_array_almost_equal函数进行比较
    if isinstance(actual, (ndarray, tuple, list)) \
            or isinstance(desired, (ndarray, tuple, list)):
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    
    # 如果actual和desired中的任何一个不是有限数，特殊处理：
    # 如果有一个是nan，那么两者都应该是nan；否则直接比较它们的值是否相等
    try:
        if not (isfinite(desired) and isfinite(actual)):
            if isnan(desired) or isnan(actual):
                if not (isnan(desired) and isnan(actual)):
                    raise AssertionError(_build_err_msg())
            else:
                if not desired == actual:
                    raise AssertionError(_build_err_msg())
            return
    except (NotImplementedError, TypeError):
        pass
    
    # 最后，如果actual和desired的差的绝对值大于或等于给定的精度阈值，抛出AssertionError
    if abs(desired - actual) >= np.float64(1.5 * 10.0**(-decimal)):
        raise AssertionError(_build_err_msg())
@np._no_nep50_warning()
def assert_approx_equal(actual, desired, significant=7, err_msg='',
                        verbose=True):
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
    >>> np.testing.assert_approx_equal(0.12345677777777e-20, 0.1234567e-20)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345671e-20,
    ...                                significant=8)
    >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345672e-20,
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
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np  # 导入 NumPy 库，用于处理数值运算

    (actual, desired) = map(float, (actual, desired))  # 将输入的 actual 和 desired 转换为浮点数
    if desired == actual:  # 如果 desired 和 actual 相等，则直接返回，表示通过检测
        return
    # 将数值归一化到 (-10.0, 10.0) 的范围内
    with np.errstate(invalid='ignore'):  # 忽略无效操作的错误
        scale = 0.5*(np.abs(desired) + np.abs(actual))  # 计算归一化所需的缩放因子
        scale = np.power(10, np.floor(np.log10(scale)))  # 将 scale 设置为 10 的整数次幂
    try:
        sc_desired = desired/scale  # 归一化后的 desired 值
    except ZeroDivisionError:
        sc_desired = 0.0  # 处理除以零的情况
    try:
        sc_actual = actual/scale  # 归一化后的 actual 值
    except ZeroDivisionError:
        sc_actual = 0.0  # 处理除以零的情况
    msg = build_err_msg(
        [actual, desired], err_msg,
        header='Items are not equal to %d significant digits:' % significant,
        verbose=verbose)  # 构建错误消息
    try:
        # 尝试执行以下代码块，处理异常情况
        # 如果 desired 或 actual 不是有限数，进行特殊处理：
        # 如果其中任何一个是 NaN，则检查两者是否都是 NaN，并测试它们是否相等
        # 否则
        if not (isfinite(desired) and isfinite(actual)):
            # 如果 desired 或 actual 是 NaN，则需要确保它们都是 NaN
            if isnan(desired) or isnan(actual):
                if not (isnan(desired) and isnan(actual)):
                    # 如果它们不都是 NaN，则抛出 AssertionError
                    raise AssertionError(msg)
            else:
                # 否则，检查它们是否相等，如果不相等则抛出 AssertionError
                if not desired == actual:
                    raise AssertionError(msg)
            return
    except (TypeError, NotImplementedError):
        # 捕获 TypeError 或 NotImplementedError 异常，不做处理，继续执行
        pass
    # 如果 sc_desired 和 sc_actual 的绝对差大于等于 10 的 -(significant-1) 次方
    # 则抛出 AssertionError
    if np.abs(sc_desired - sc_actual) >= np.power(10., -(significant-1)):
        raise AssertionError(msg)
# 添加注释说明
@np._no_nep50_warning()
# 禁用 NEP 50 警告的装饰器，防止出现不兼容的行为
def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='',
                         precision=6, equal_nan=True, equal_inf=True,
                         *, strict=False, names=('ACTUAL', 'DESIRED')):
    # 设置隐藏 traceback 以便在 pytest 中隐藏异常信息
    __tracebackhide__ = True  
    # 导入必要的模块和函数
    from numpy._core import (array2string, isnan, inf, errstate,
                            all, max, object_)

    # 将输入的 x 和 y 转换为 numpy 数组
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    # 用于输出格式的原始数组
    ox, oy = x, y

    # 检查是否为数字类型
    def isnumber(x):
        return x.dtype.char in '?bhilqpBHILQPefdgFDG'

    # 检查是否为时间类型
    def istime(x):
        return x.dtype.char in "Mm"

    # 检查是否为字符串向量类型
    def isvstring(x):
        return x.dtype.char == "T"

    # 比较函数，处理 NaN 和 Inf 的情况
    def func_assert_same_pos(x, y, func=isnan, hasval='nan'):
        """Handling nan/inf.

        Combine results of running func on x and y, checking that they are True
        at the same locations.

        """
        __tracebackhide__ = True  # 隐藏 traceback 以便在 pytest 中隐藏异常信息

        # 对 x 和 y 分别应用 func 函数，检查它们在相同位置是否为 True
        x_id = func(x)
        y_id = func(y)
        
        # 处理三种略微异常的子类 ndarray 的情况：
        # 1. masked array 的标量可能返回 masked arrays，因此使用 != True
        # 2. 某些 ndarray 子类的 __eq__ 方法返回 Python 布尔值而不是逐元素比较，
        #    所以需要强制转换为 np.bool() 并使用 isinstance(..., bool) 检查
        # 3. 具有简单 __array_function__ 实现的子类可能不实现 np.all()，所以
        #    更倾向于使用 .all() 方法
        if np.bool(x_id == y_id).all() != True:
            msg = build_err_msg(
                [x, y],
                err_msg + '\n%s location mismatch:'
                % (hasval), verbose=verbose, header=header,
                names=names,
                precision=precision)
            raise AssertionError(msg)
        
        # 如果其中一个是标量，则此处我们知道数组在每个地方都具有相同的标志，因此应返回标量标志。
        if isinstance(x_id, bool) or x_id.ndim == 0:
            return np.bool(x_id)
        elif isinstance(y_id, bool) or y_id.ndim == 0:
            return np.bool(y_id)
        else:
            return y_id

    except ValueError:
        import traceback
        # 处理异常时打印 traceback 信息
        efmt = traceback.format_exc()
        header = f'error during assertion:\n\n{efmt}\n\n{header}'

        # 构建错误消息，指示断言失败
        msg = build_err_msg([x, y], err_msg, verbose=verbose, header=header,
                            names=names, precision=precision)
        raise ValueError(msg)


@_rename_parameter(['x', 'y'], ['actual', 'desired'], dep_version='2.0.0')
# 将参数 x 和 y 重命名为 actual 和 desired，用于向后兼容性
def assert_array_equal(actual, desired, err_msg='', verbose=True, *,
                       strict=False):
    """
    Raises an AssertionError if two array_like objects are not equal.
    """
    # 给定两个类数组对象，检查它们的形状是否相等，并且所有元素是否相等（但参见注释中标量的特殊处理）。如果形状不匹配或存在冲突的值，则会引发异常。与 numpy 中的标准用法相比，NaN 会像数字一样进行比较，如果两个对象在相同位置都有 NaN，则不会引发断言错误。
    
    # 建议使用通常用于浮点数的相等性验证时要注意的问题。
    
    # 当 `actual` 或 `desired` 已经是 `numpy.ndarray` 实例且 `desired` 不是字典时，`assert_equal(actual, desired)` 的行为与此函数完全相同。否则，在比较之前，此函数会对输入执行 `np.asanyarray`，而 `assert_equal` 为常见的 Python 类型定义了特殊的比较规则。例如，只有 `assert_equal` 可以用于比较嵌套的 Python 列表。在新代码中，如果需要 `assert_array_equal` 的行为，考虑仅使用 `assert_equal`，并显式将 `actual` 或 `desired` 转换为数组。
    
    # 参数：
    # actual : array_like
    #     要检查的实际对象。
    # desired : array_like
    #     期望的对象。
    # err_msg : str, optional
    #     失败时打印的错误消息。
    # verbose : bool, optional
    #     如果为 True，则将冲突的值附加到错误消息中。
    # strict : bool, optional
    #     如果为 True，则当数组类对象的形状或数据类型不匹配时引发 AssertionError。禁用注释部分中提到的标量特殊处理。
    #     .. versionadded:: 1.24.0
    
    # 引发：
    # AssertionError
    #     如果实际和期望对象不相等。
    
    # 另请参阅：
    # --------
    # assert_allclose: 用期望的相对和/或绝对精度比较两个类数组对象的相等性。
    # assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal
    
    # 注释：
    # --------
    # 当 `actual` 和 `desired` 中的一个是标量，另一个是类数组对象时，函数会检查数组对象的每个元素是否等于标量。可以使用 `strict` 参数禁用此行为。
    
    # 示例：
    # --------
    # 第一个断言不会引发异常：
    # >>> np.testing.assert_array_equal([1.0,2.33333,np.nan],
    # ...                               [np.exp(0),2.33333, np.nan])
    
    # 使用浮点数时，由于数值精度问题，断言失败：
    # >>> np.testing.assert_array_equal([1.0,np.pi,np.nan],
    # ...                               [1, np.sqrt(np.pi)**2, np.nan])
    # Traceback (most recent call last):
    #     ...
    # AssertionError:
    # Arrays are not equal
    # <BLANKLINE>
    # Mismatched elements: 1 / 3 (33.3%)
    # Max absolute difference among violations: 4.4408921e-16
    # 最大违规中的相对差异：1.41357986e-16
    # 实际值：array([1.      , 3.141593,      nan])
    # 期望值：array([1.      , 3.141593,      nan])
    #
    # 对于这些情况，请使用 `assert_allclose` 或 nulp（浮点数值的数量）函数：
    #
    # >>> np.testing.assert_allclose([1.0,np.pi,np.nan],
    # ...                            [1, np.sqrt(np.pi)**2, np.nan],
    # ...                            rtol=1e-10, atol=0)
    #
    # 如在注释部分提到的，`assert_array_equal` 对标量有特殊处理。此处的测试检查 `x` 中的每个值是否为 3：
    #
    # >>> x = np.full((2, 5), fill_value=3)
    # >>> np.testing.assert_array_equal(x, 3)
    #
    # 使用 `strict` 来在将标量与数组进行比较时引发 AssertionError：
    #
    # >>> np.testing.assert_array_equal(x, 3, strict=True)
    # Traceback (most recent call last):
    #     ...
    # AssertionError:
    # Arrays are not equal
    #
    # (形状 (2, 5), () 不匹配)
    #  实际值：array([[3, 3, 3, 3, 3],
    #        [3, 3, 3, 3, 3]])
    #  期望值：array(3)
    #
    # `strict` 参数还确保数组的数据类型匹配：
    #
    # >>> x = np.array([2, 2, 2])
    # >>> y = np.array([2., 2., 2.], dtype=np.float32)
    # >>> np.testing.assert_array_equal(x, y, strict=True)
    # Traceback (most recent call last):
    #     ...
    # AssertionError:
    # Arrays are not equal
    #
    # (数据类型 int64, float32 不匹配)
    #  实际值：array([2, 2, 2])
    #  期望值：array([2., 2., 2.], dtype=float32)
    """
    __tracebackhide__ = True  # 在 py.test 中隐藏回溯信息
    assert_array_compare(operator.__eq__, actual, desired, err_msg=err_msg,
                         verbose=verbose, header='Arrays are not equal',
                         strict=strict)
# 应用修饰符，用于阻止 NEP 50 警告的显示
@np._no_nep50_warning()
# 重命名参数 x 和 y 为 actual 和 desired，并指定在版本 2.0.0 后弃用
@_rename_parameter(['x', 'y'], ['actual', 'desired'], dep_version='2.0.0')
# 定义一个函数，用于比较两个对象在指定精度下的近似程度，并在不符合预期时抛出 AssertionError
def assert_array_almost_equal(actual, desired, decimal=6, err_msg='',
                              verbose=True):
    """
    Raises an AssertionError if two objects are not equal up to desired
    precision.

    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.

    The test verifies identical shapes and that the elements of ``actual`` and
    ``desired`` satisfy::

        abs(desired-actual) < 1.5 * 10**(-decimal)

    That is a looser test than originally documented, but agrees with what the
    actual implementation did up to rounding vagaries. An exception is raised
    at shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if both
    objects have NaNs in the same positions.

    Parameters
    ----------
    actual : array_like
        The actual object to check.
    desired : array_like
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
    Max absolute difference among violations: 6.e-05
    Max relative difference among violations: 2.57136612e-05
     ACTUAL: array([1.     , 2.33333,     nan])
     DESIRED: array([1.     , 2.33339,     nan])

    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33333, 5], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    nan location mismatch:
     ACTUAL: array([1.     , 2.33333,     nan])
     DESIRED: array([1.     , 2.33333, 5.     ])

    """
    # 隐藏 py.test 的异常回溯信息
    __tracebackhide__ = True

    # 从 numpy._core 导入必要的模块和函数
    from numpy._core import number, result_type
    # 从 numpy._core.numerictypes 导入特定函数和类
    from numpy._core.numerictypes import issubdtype
    # 从 numpy._core.fromnumeric 导入 any 函数并重命名为 npany
    from numpy._core.fromnumeric import any as npany

    # 定义函数 compare，用于比较两个数组 x 和 y 的近似性
    def compare(x, y):
        try:
            # 检查 x 或 y 是否包含无穷大的元素，如果有则返回 True
            if npany(isinf(x)) or npany(isinf(y)):
                # 记录 x 和 y 中的无穷大元素的位置
                xinfid = isinf(x)
                yinfid = isinf(y)
                # 如果 xinfid 和 yinfid 的元素不完全相等，返回 False
                if not (xinfid == yinfid).all():
                    return False
                # 如果 x 和 y 都只有一个元素且为正负无穷大，则直接比较它们
                if x.size == y.size == 1:
                    return x == y
                # 从 x 和 y 中排除掉无穷大的元素，以便后续比较
                x = x[~xinfid]
                y = y[~yinfid]
        except (TypeError, NotImplementedError):
            pass

        # 确保 y 是一种非精确类型，以避免 abs(MIN_INT) 的问题；这会导致后续 x 的类型转换
        dtype = result_type(y, 1.)
        y = np.asanyarray(y, dtype)
        # 计算 x 和 y 的绝对差值
        z = abs(x - y)

        # 如果 z 的数据类型不是数字类型，则将其转换为 np.float64 类型，以处理对象数组
        if not issubdtype(z.dtype, number):
            z = z.astype(np.float64)

        # 返回比较结果，即 z 是否小于给定精度下的阈值
        return z < 1.5 * 10.0**(-decimal)

    # 使用 assert_array_compare 函数比较 actual 和 desired 两个数组，
    # 并设置错误信息、详细输出、比较的精度等参数
    assert_array_compare(compare, actual, desired, err_msg=err_msg,
                         verbose=verbose,
                         header=('Arrays are not almost equal to %d decimals' % decimal),
                         precision=decimal)
# 检查两个类数组对象是否按照严格的“小于”关系排序，否则引发 AssertionError
def assert_array_less(x, y, err_msg='', verbose=True, *, strict=False):
    """
    Raises an AssertionError if two array_like objects are not ordered by less
    than.

    Given two array_like objects `x` and `y`, check that the shape is equal and
    all elements of `x` are strictly less than the corresponding elements of
    `y` (but see the Notes for the special handling of a scalar). An exception
    is raised at shape mismatch or values that are not correctly ordered. In
    contrast to the  standard usage in NumPy, no assertion is raised if both
    objects have NaNs in the same positions.

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
    strict : bool, optional
        If True, raise an AssertionError when either the shape or the data
        type of the array_like objects does not match. The special
        handling for scalars mentioned in the Notes section is disabled.

        .. versionadded:: 2.0.0

    Raises
    ------
    AssertionError
      If x is not strictly smaller than y, element-wise.

    See Also
    --------
    assert_array_equal: tests objects for equality
    assert_array_almost_equal: test objects for equality up to precision

    Notes
    -----
    When one of `x` and `y` is a scalar and the other is array_like, the
    function performs the comparison as though the scalar were broadcasted
    to the shape of the array. This behaviour can be disabled with the `strict`
    parameter.

    Examples
    --------
    The following assertion passes because each finite element of `x` is
    strictly less than the corresponding element of `y`, and the NaNs are in
    corresponding locations.

    >>> x = [1.0, 1.0, np.nan]
    >>> y = [1.1, 2.0, np.nan]
    >>> np.testing.assert_array_less(x, y)

    The following assertion fails because the zeroth element of `x` is no
    longer strictly less than the zeroth element of `y`.

    >>> y[0] = 1
    >>> np.testing.assert_array_less(x, y)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not strictly ordered `x < y`
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference among violations: 0.
    Max relative difference among violations: 0.
     x: array([ 1.,  1., nan])
     y: array([ 1.,  2., nan])

    Here, `y` is a scalar, so each element of `x` is compared to `y`, and
    the assertion passes.

    >>> x = [1.0, 4.0]
    >>> y = 5.0
    >>> np.testing.assert_array_less(x, y)

    However, with ``strict=True``, the assertion will fail because the shapes
    do not match.

    >>> np.testing.assert_array_less(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not strictly ordered `x < y`
    """
    """
    (shapes (2,), () mismatch)
     x: array([1., 4.])
     y: array(5.)

    With ``strict=True``, the assertion also fails if the dtypes of the two
    arrays do not match.

    >>> y = [5, 5]
    >>> np.testing.assert_array_less(x, y, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not strictly ordered `x < y`
    <BLANKLINE>
    (dtypes float64, int64 mismatch)
     x: array([1., 4.])
     y: array([5, 5])
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    # 使用断言检查两个数组 x 和 y 是否按严格顺序排列，若不满足则抛出异常
    assert_array_compare(operator.__lt__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not strictly ordered `x < y`',
                         equal_inf=False,
                         strict=strict,
                         names=('x', 'y'))
# 定义一个函数 `runstring`，用于在给定的字典环境中执行字符串形式的代码
def runstring(astr, dict):
    exec(astr, dict)

# 定义一个函数 `assert_string_equal`，用于比较两个字符串是否相等，并在不相等时抛出异常
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
        The expected string.

    Examples
    --------
    >>> np.testing.assert_string_equal('abc', 'abc')
    >>> np.testing.assert_string_equal('abc', 'abcd')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ...
    AssertionError: Differences in strings:
    - abc+ abcd?    +

    """
    # 延迟导入 difflib 以减少启动时间
    __tracebackhide__ = True  # 隐藏 py.test 的回溯信息
    import difflib

    # 检查参数类型，如果不是字符串则抛出异常
    if not isinstance(actual, str):
        raise AssertionError(repr(type(actual)))
    if not isinstance(desired, str):
        raise AssertionError(repr(type(desired)))
    
    # 如果两个字符串相等，直接返回
    if desired == actual:
        return

    # 使用 difflib 模块比较两个字符串的差异
    diff = list(difflib.Differ().compare(actual.splitlines(True),
                                          desired.splitlines(True)))
    diff_list = []
    while diff:
        d1 = diff.pop(0)
        if d1.startswith('  '):
            continue
        if d1.startswith('- '):
            l = [d1]
            d2 = diff.pop(0)
            if d2.startswith('? '):
                l.append(d2)
                d2 = diff.pop(0)
            if not d2.startswith('+ '):
                raise AssertionError(repr(d2))
            l.append(d2)
            if diff:
                d3 = diff.pop(0)
                if d3.startswith('? '):
                    l.append(d3)
                else:
                    diff.insert(0, d3)
            if d2[2:] == d1[2:]:
                continue
            diff_list.extend(l)
            continue
        raise AssertionError(repr(d1))
    
    # 如果没有差异，则直接返回
    if not diff_list:
        return
    
    # 构造差异信息并抛出 AssertionError
    msg = f"Differences in strings:\n{''.join(diff_list).rstrip()}"
    if actual != desired:
        raise AssertionError(msg)


# 定义一个函数 `rundocs`，用于运行给定文件中的文档测试
def rundocs(filename=None, raise_on_error=True):
    """
    Run doctests found in the given file.

    By default `rundocs` raises an AssertionError on failure.

    Parameters
    ----------
    filename : str
        The path to the file for which the doctests are run.
    raise_on_error : bool
        Whether to raise an AssertionError when a doctest fails. Default is
        True.

    Notes
    -----
    The doctests can be run by the user/developer by adding the ``doctests``
    argument to the ``test()`` call. For example, to run all tests (including
    doctests) for ``numpy.lib``:

    >>> np.lib.test(doctests=True)  # doctest: +SKIP
    """
    # 导入必要的模块
    from numpy.distutils.misc_util import exec_mod_from_location
    import doctest
    
    # 如果未提供文件名，获取调用栈上一级的文件名
    if filename is None:
        f = sys._getframe(1)
        filename = f.f_globals['__file__']
    # 从文件路径中提取文件名并去除扩展名，得到文件名
    name = os.path.splitext(os.path.basename(filename))[0]
    # 根据文件名和路径加载执行模块，并返回模块对象
    m = exec_mod_from_location(name, filename)

    # 使用 doctest.DocTestFinder 查找模块中的所有文档测试
    tests = doctest.DocTestFinder().find(m)
    # 创建一个 doctest.DocTestRunner 对象用于运行文档测试，设定为非详细模式
    runner = doctest.DocTestRunner(verbose=False)

    # 初始化一个空列表 msg 用于存储测试结果消息
    msg = []
    # 根据 raise_on_error 参数决定如何设置输出函数 out
    if raise_on_error:
        # 定义一个 lambda 函数，将输出的消息添加到 msg 列表中
        out = lambda s: msg.append(s)
    else:
        # 如果不需要输出错误信息，则将 out 设置为 None
        out = None

    # 遍历所有找到的文档测试，并运行它们，将结果输出到 out 函数（或 None）
    for test in tests:
        runner.run(test, out=out)

    # 如果有测试失败且 raise_on_error 为 True，则抛出 AssertionError 异常，包含所有失败消息
    if runner.failures > 0 and raise_on_error:
        raise AssertionError("Some doctests failed:\n%s" % "\n".join(msg))
def check_support_sve():
    """
    gh-22982
    """

    # 导入 subprocess 模块用于执行外部命令
    import subprocess
    # 定义要执行的命令
    cmd = 'lscpu'
    try:
        # 运行命令并捕获输出，以文本形式获取输出结果
        output = subprocess.run(cmd, capture_output=True, text=True)
        # 检查输出中是否包含 'sve'
        return 'sve' in output.stdout
    except OSError:
        # 如果发生 OSError，返回 False
        return False


# 检查当前系统是否支持 SVE（Scalable Vector Extension）
_SUPPORTS_SVE = check_support_sve()

#
# assert_raises and assert_raises_regex are taken from unittest.
#
import unittest

# 创建一个虚拟的 TestCase 以便使用 assertRaises 方法
class _Dummy(unittest.TestCase):
    def nop(self):
        pass

# 实例化虚拟 TestCase
_d = _Dummy('nop')


def assert_raises(*args, **kwargs):
    """
    assert_raises(exception_class, callable, *args, **kwargs)
    assert_raises(exception_class)

    如果调用 callable 时抛出 exception_class 类型的异常（带有指定参数和关键字参数），则测试失败。
    如果抛出不同类型的异常，则不会捕获，测试用例会因未预期的异常而失败。

    也可以将 `assert_raises` 用作上下文管理器：

    >>> from numpy.testing import assert_raises
    >>> with assert_raises(ZeroDivisionError):
    ...     1 / 0

    等同于

    >>> def div(x, y):
    ...     return x / y
    >>> assert_raises(ZeroDivisionError, div, 1, 0)

    """
    # 隐藏 pytest 的异常回溯信息
    __tracebackhide__ = True
    return _d.assertRaises(*args, **kwargs)


def assert_raises_regex(exception_class, expected_regexp, *args, **kwargs):
    """
    assert_raises_regex(exception_class, expected_regexp, callable, *args,
                        **kwargs)
    assert_raises_regex(exception_class, expected_regexp)

    如果调用 callable 时抛出 exception_class 类型的异常，并且异常消息匹配 expected_regexp，则测试失败。

    可以像 `assert_raises` 一样使用上下文管理器。

    注意
    -----
    .. versionadded:: 1.9.0

    """
    # 隐藏 pytest 的异常回溯信息
    __tracebackhide__ = True
    return _d.assertRaisesRegex(exception_class, expected_regexp, *args, **kwargs)


def decorate_methods(cls, decorator, testmatch=None):
    """
    Apply a decorator to all methods in a class matching a regular expression.

    给匹配正则表达式 `testmatch` 的类中的所有公共方法应用装饰器。

    Parameters
    ----------
    cls : class
        要装饰方法的类。
    decorator : function
        要应用的装饰器函数。
    testmatch : compiled regexp or str, optional
        正则表达式。默认为 None，此时使用默认值：
        ``re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)``。
        如果 `testmatch` 是字符串，则首先编译为正则表达式。

    """
    # 如果 testmatch 为 None，则使用正则表达式创建一个匹配函数名的模式
    if testmatch is None:
        testmatch = re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)
    else:
        # 否则，将传入的 testmatch 编译成正则表达式对象
        testmatch = re.compile(testmatch)
    
    # 获取类的属性字典
    cls_attr = cls.__dict__

    # 延迟导入以减少启动时间，从 inspect 模块中导入 isfunction 函数
    from inspect import isfunction

    # 获取所有函数方法（即可调用对象）组成的列表
    methods = [_m for _m in cls_attr.values() if isfunction(_m)]
    
    # 遍历类的每个函数方法
    for function in methods:
        try:
            # 尝试获取函数的名称作为 funcname
            if hasattr(function, 'compat_func_name'):
                funcname = function.compat_func_name
            else:
                funcname = function.__name__
        except AttributeError:
            # 如果对象不是函数，则跳过
            continue
        
        # 检查 funcname 是否匹配 testmatch 的正则表达式，并且函数名称不以下划线开头
        if testmatch.search(funcname) and not funcname.startswith('_'):
            # 对匹配的函数应用 decorator 函数，并设置回原类的属性中
            setattr(cls, funcname, decorator(function))
    
    # 返回
    return
def measure(code_str, times=1, label=None):
    """
    Return elapsed time for executing code in the namespace of the caller.

    The supplied code string is compiled with the Python builtin ``compile``.
    The precision of the timing is 10 milli-seconds. If the code will execute
    fast on this timescale, it can be executed many times to get reasonable
    timing accuracy.

    Parameters
    ----------
    code_str : str
        The code to be timed.
    times : int, optional
        The number of times the code is executed. Default is 1. The code is
        only compiled once.
    label : str, optional
        A label to identify `code_str` with. This is passed into ``compile``
        as the second argument (for run-time error messages).

    Returns
    -------
    elapsed : float
        Total elapsed time in seconds for executing `code_str` `times` times.

    Examples
    --------
    >>> times = 10
    >>> etime = np.testing.measure('for i in range(1000): np.sqrt(i**2)', times=times)
    >>> print("Time for a single execution : ", etime / times, "s")  # doctest: +SKIP
    Time for a single execution :  0.005 s

    """
    # 获取调用者的栈帧
    frame = sys._getframe(1)
    locs, globs = frame.f_locals, frame.f_globals

    # 使用内置函数 compile 编译传入的代码字符串，创建可执行的代码对象
    code = compile(code_str, f'Test name: {label} ', 'exec')
    i = 0
    # 获取当前时间戳（以 jiffies 为单位）
    elapsed = jiffies()
    while i < times:
        i += 1
        # 执行编译后的代码，使用调用者的局部和全局命名空间
        exec(code, globs, locs)
    # 计算执行代码所用的总时间（以 jiffies 为单位），并转换为秒
    elapsed = jiffies() - elapsed
    return 0.01*elapsed


def _assert_valid_refcount(op):
    """
    Check that ufuncs don't mishandle refcount of object `1`.
    Used in a few regression tests.
    """
    # 如果没有引用计数相关的功能，直接返回 True
    if not HAS_REFCOUNT:
        return True

    import gc
    import numpy as np

    # 创建一个 100x100 的数组 b
    b = np.arange(100*100).reshape(100, 100)
    # 创建 c，指向数组 b
    c = b
    i = 1

    # 禁用垃圾回收器
    gc.disable()
    try:
        # 获取整数对象 1 的引用计数
        rc = sys.getrefcount(i)
        # 多次执行传入的操作 op，并检查对象 1 的引用计数
        for j in range(15):
            d = op(b, c)
        # 断言执行操作后对象 1 的引用计数是否不少于初始时的引用计数
        assert_(sys.getrefcount(i) >= rc)
    finally:
        # 恢复垃圾回收器的状态
        gc.enable()
    # 删除变量 d，以避免 pyflakes 报告未使用的变量
    del d  # for pyflakes


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True,
                    err_msg='', verbose=True, *, strict=False):
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

    """
    # 函数未实现代码部分，不需要添加注释
    pass
    rtol : float, optional
        相对容差。
    atol : float, optional
        绝对容差。
    equal_nan : bool, optional.
        如果为 True，则 NaN 将被视为相等。
    err_msg : str, optional
        失败时要打印的错误消息。
    verbose : bool, optional
        如果为 True，则将冲突的值附加到错误消息中。
    strict : bool, optional
        如果为 True，则当参数的形状或数据类型不匹配时会引发 ``AssertionError``。在“Notes”部分提到的标量特殊处理将被禁用。

        .. versionadded:: 2.0.0

    Raises
    ------
    AssertionError
        如果实际值和期望值在指定精度范围内不相等。

    See Also
    --------
    assert_array_almost_equal_nulp, assert_array_max_ulp

    Notes
    -----
    当 `actual` 和 `desired` 中的一个是标量，另一个是类似数组时，函数将执行比较，就好像标量被广播到数组的形状。
    可以使用 `strict` 参数禁用此行为。

    Examples
    --------
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)

    如“Notes”部分所述，`assert_allclose` 对标量有特殊处理。在这里，测试检查 `numpy.sin` 在整数倍π处几乎为零。

    >>> x = np.arange(3) * np.pi
    >>> np.testing.assert_allclose(np.sin(x), 0, atol=1e-15)

    使用 `strict` 来比较一个数组与一个或多个维度的标量时会引发 ``AssertionError``。

    >>> np.testing.assert_allclose(np.sin(x), 0, atol=1e-15, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Not equal to tolerance rtol=1e-07, atol=1e-15
    <BLANKLINE>
    (shapes (3,), () mismatch)
     ACTUAL: array([ 0.000000e+00,  1.224647e-16, -2.449294e-16])
     DESIRED: array(0)

    `strict` 参数还确保数组的数据类型匹配：

    >>> y = np.zeros(3, dtype=np.float32)
    >>> np.testing.assert_allclose(np.sin(x), y, atol=1e-15, strict=True)
    Traceback (most recent call last):
        ...
    AssertionError:
    Not equal to tolerance rtol=1e-07, atol=1e-15
    <BLANKLINE>
    (dtypes float64, float32 mismatch)
     ACTUAL: array([ 0.000000e+00,  1.224647e-16, -2.449294e-16])
     DESIRED: array([0., 0., 0.], dtype=float32)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np

    def compare(x, y):
        return np._core.numeric.isclose(x, y, rtol=rtol, atol=atol,
                                       equal_nan=equal_nan)

    actual, desired = np.asanyarray(actual), np.asanyarray(desired)
    header = f'Not equal to tolerance rtol={rtol:g}, atol={atol:g}'
    # 使用 assert_array_compare 函数进行数组比较，检查 actual 和 desired 是否满足 compare 函数的条件
    # 可选参数 err_msg 用于指定断言失败时显示的错误信息
    # 可选参数 verbose 控制是否输出详细信息
    # 可选参数 header 控制是否包含标题信息
    # 可选参数 equal_nan 控制 NaN 值的比较方式
    # 可选参数 strict 控制是否进行严格的比较
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
                         verbose=verbose, header=header, equal_nan=equal_nan,
                         strict=strict)
# 比较两个数组的每个元素是否在最后一位单位上相等
def assert_array_almost_equal_nulp(x, y, nulp=1):
    """
    Compare two arrays relatively to their spacing.

    This is a relatively robust method to compare two arrays whose amplitude
    is variable.

    Parameters
    ----------
    x, y : array_like
        Input arrays.
    nulp : int, optional
        The maximum number of unit in the last place for tolerance (see Notes).
        Default is 1.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the spacing between `x` and `y` for one or more elements is larger
        than `nulp`.

    See Also
    --------
    assert_array_max_ulp : Check that all items of arrays differ in at most
        N Units in the Last Place.
    spacing : Return the distance between x and the nearest adjacent number.

    Notes
    -----
    An assertion is raised if the following condition is not met::

        abs(x - y) <= nulp * spacing(maximum(abs(x), abs(y)))

    Examples
    --------
    >>> x = np.array([1., 1e-10, 1e-20])
    >>> eps = np.finfo(x.dtype).eps
    >>> np.testing.assert_array_almost_equal_nulp(x, x*eps/2 + x)

    >>> np.testing.assert_array_almost_equal_nulp(x, x*eps + x)
    Traceback (most recent call last):
      ...
    AssertionError: Arrays are not equal to 1 ULP (max is 2)

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import numpy as np  # 导入NumPy库
    # 计算x和y的绝对值
    ax = np.abs(x)
    ay = np.abs(y)
    # 选择ax和ay中较大的数，计算其间隔
    ref = nulp * np.spacing(np.where(ax > ay, ax, ay))
    # 检查所有元素的差异是否小于或等于间隔的nulp倍
    if not np.all(np.abs(x-y) <= ref):
        # 如果x或y是复数对象，则输出相应的错误信息
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            msg = f"Arrays are not equal to {nulp} ULP"
        else:
            # 否则计算最大的nulp差异，并输出错误信息
            max_nulp = np.max(nulp_diff(x, y))
            msg = f"Arrays are not equal to {nulp} ULP (max is {max_nulp:g})"
        # 抛出断言错误
        raise AssertionError(msg)


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
    >>> res = np.testing.assert_array_max_ulp(a, np.arcsin(np.sin(a)))

    """
    # 隐藏 pytest 中的异常回溯信息
    __tracebackhide__ = True  # Hide traceback for py.test
    
    # 导入 NumPy 库，用于数值计算
    import numpy as np
    
    # 调用 nulp_diff 函数计算两个数组 a 和 b 之间的差异
    ret = nulp_diff(a, b, dtype)
    
    # 检查 ret 中的所有元素是否都小于等于指定的 maxulp
    if not np.all(ret <= maxulp):
        # 如果不满足条件，抛出断言错误，指出数组在给定精度 maxulp 下不几乎相等，
        # 并显示实际差异中的最大单位误差 (ULP)
        raise AssertionError("Arrays are not almost equal up to %g "
                             "ULP (max difference is %g ULP)" %
                             (maxulp, np.max(ret)))
    
    # 返回计算出的差异 ret
    return ret
# 定义一个函数，用于计算两个数组中每个元素之间的可表示浮点数的数量差异
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
    >>> nulp_diff(1, 1 + np.finfo(x.dtype).eps)
    1.0
    """
    import numpy as np  # 导入 NumPy 库

    # 如果提供了 dtype 参数，则将 x 和 y 转换为指定的数据类型
    if dtype:
        x = np.asarray(x, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
    else:
        x = np.asarray(x)  # 否则将 x 转换为 NumPy 数组
        y = np.asarray(y)  # 同样将 y 转换为 NumPy 数组

    t = np.common_type(x, y)  # 获取 x 和 y 的公共数据类型
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise NotImplementedError("_nulp not implemented for complex array")  # 如果 x 或 y 是复数数组，则抛出未实现异常

    x = np.array([x], dtype=t)  # 创建数据类型为 t 的 x 数组
    y = np.array([y], dtype=t)  # 创建数据类型为 t 的 y 数组

    x[np.isnan(x)] = np.nan  # 将 x 中的 NaN 值保留为 NaN
    y[np.isnan(y)] = np.nan  # 将 y 中的 NaN 值保留为 NaN

    if not x.shape == y.shape:  # 如果 x 和 y 的形状不相同，则引发值错误异常
        raise ValueError("Arrays do not have the same shape: %s - %s" %
                         (x.shape, y.shape))

    def _diff(rx, ry, vdt):
        # 计算 rx 和 ry 之间的差异，并返回其绝对值
        diff = np.asarray(rx - ry, dtype=vdt)
        return np.abs(diff)

    rx = integer_repr(x)  # 调用 integer_repr 函数处理 x，获取其整数表示
    ry = integer_repr(y)  # 调用 integer_repr 函数处理 y，获取其整数表示
    return _diff(rx, ry, t)  # 返回计算得到的差异


def _integer_repr(x, vdt, comp):
    """Return the signed-magnitude interpretation of the binary representation
    of the float as sign-magnitude:
    take into account two-complement representation.
    See also https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    """
    # 重新解释浮点数的二进制表示为符号-幅度表示
    rx = x.view(vdt)
    if not (rx.size == 1):  # 如果 rx 的大小不为 1
        rx[rx < 0] = comp - rx[rx < 0]  # 将 rx 中小于 0 的元素转换为其补码表示
    else:
        if rx < 0:  # 如果 rx 小于 0
            rx = comp - rx  # 则计算其补码表示

    return rx  # 返回处理后的 rx


def integer_repr(x):
    """Return the signed-magnitude interpretation of the binary representation
    of x."""
    import numpy as np  # 导入 NumPy 库

    # 根据 x 的数据类型选择相应的 _integer_repr 函数
    if x.dtype == np.float16:
        return _integer_repr(x, np.int16, np.int16(-2**15))
    elif x.dtype == np.float32:
        return _integer_repr(x, np.int32, np.int32(-2**31))
    elif x.dtype == np.float64:
        return _integer_repr(x, np.int64, np.int64(-2**63))
    else:
        raise ValueError(f'Unsupported dtype {x.dtype}')  # 如果 x 的数据类型不支持，则引发值错误异常


@contextlib.contextmanager
def _assert_warns_context(warning_class, name=None):
    __tracebackhide__ = True  # 隐藏 pytest 的回溯信息
    # 使用 `suppress_warnings()` 上下文管理器来捕获特定类型的警告
    with suppress_warnings() as sup:
        # 记录在 `suppress_warnings()` 中捕获到的特定警告类的警告信息
        l = sup.record(warning_class)
        # 在此处生成器继续执行
        yield
        # 如果没有捕获到任何警告信息，则抛出断言错误
        if not len(l) > 0:
            # 准备异常消息字符串，包含调用函数名（如果有）
            name_str = f' when calling {name}' if name is not None else ''
            # 抛出断言错误，指示未触发期望的警告
            raise AssertionError("No warning raised" + name_str)
def assert_warns(warning_class, *args, **kwargs):
    """
    Fail unless the given callable throws the specified warning.

    A warning of class warning_class should be thrown by the callable when
    invoked with arguments args and keyword arguments kwargs.
    If a different type of warning is thrown, it will not be caught.

    If called with all arguments other than the warning class omitted, may be
    used as a context manager::

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
    # 如果未提供除了警告类以外的参数，则作为上下文管理器使用
    if not args and not kwargs:
        return _assert_warns_context(warning_class)
    # 如果参数少于1个且包含 "match" 关键字参数，则引发运行时错误
    elif len(args) < 1:
        if "match" in kwargs:
            raise RuntimeError(
                "assert_warns does not use 'match' kwarg, "
                "use pytest.warns instead"
                )
        raise RuntimeError("assert_warns(...) needs at least one arg")

    # 获取第一个参数作为 callable
    func = args[0]
    # 剩余参数重新赋值给 args
    args = args[1:]
    # 使用 _assert_warns_context 作为上下文管理器来检测警告
    with _assert_warns_context(warning_class, name=func.__name__):
        # 调用 func 并传入参数和关键字参数，返回其返回值
        return func(*args, **kwargs)


@contextlib.contextmanager
def _assert_no_warnings_context(name=None):
    __tracebackhide__ = True  # 在 pytest 中隐藏回溯信息
    # 捕获所有警告
    with warnings.catch_warnings(record=True) as l:
        # 设置警告过滤器，始终显示警告
        warnings.simplefilter('always')
        # 执行 yield 语句，即在 with 块内运行的代码
        yield
        # 如果捕获到了警告，则抛出 AssertionError
        if len(l) > 0:
            name_str = f' when calling {name}' if name is not None else ''
            raise AssertionError(f'Got warnings{name_str}: {l}')


def assert_no_warnings(*args, **kwargs):
    """
    Fail if the given callable produces any warnings.

    If called with all arguments omitted, may be used as a context manager::

        with assert_no_warnings():
            do_something()

    The ability to be used as a context manager is new in NumPy v1.11.0.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    func : callable
        The callable to test.
    *args : Arguments
        Arguments passed to `func`.
    **kwargs : Kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    The value returned by `func`.

    """
    # 如果未提供参数，则作为上下文管理器使用
    if not args:
        return _assert_no_warnings_context()
    # 从参数列表中获取第一个参数作为函数对象
    func = args[0]
    # 将参数列表中除第一个参数外的其余参数保存到 args 变量中
    args = args[1:]
    # 使用 _assert_no_warnings_context 上下文管理器，确保在执行函数时没有警告产生
    with _assert_no_warnings_context(name=func.__name__):
        # 调用函数 func，并传入剩余的位置参数和关键字参数 kwargs
        return func(*args, **kwargs)
# 定义一个生成对齐数据的生成器函数，用于测试 SIMD 向量化
def _gen_alignment_data(dtype=float32, type='binary', max_size=24):
    """
    generator producing data with different alignment and offsets
    to test simd vectorization

    Parameters
    ----------
    dtype : dtype
        数据类型，用于生成数据
    type : string
        数据类型，'unary': 用于一元操作的数据，创建一个输入和输出数组
                  'binary': 用于二元操作的数据，创建两个输入和输出数组
    max_size : integer
        要生成的数据的最大尺寸

    Returns
    -------
    如果 type 是 'unary'，产生一个输出数组、一个输入数组和一个包含数据信息的消息
    如果 type 是 'binary'，产生一个输出数组、两个输入数组和一个包含数据信息的消息

    """
    # 格式化字符串，用于描述一元操作的数据
    ufmt = 'unary offset=(%d, %d), size=%d, dtype=%r, %s'
    # 格式化字符串，用于描述二元操作的数据
    bfmt = 'binary offset=(%d, %d, %d), size=%d, dtype=%r, %s'
    # 循环变量 o 在范围 [0, 3) 内迭代
    for o in range(3):
        # 循环变量 s 在范围 [o+2, max(o+3, max_size)) 内迭代，其中 max_size 是一个预先定义的最大值
        for s in range(o + 2, max(o + 3, max_size)):
            # 如果 type 等于 'unary'，执行以下操作
            if type == 'unary':
                # 创建一个 lambda 函数 inp，返回一个以 s 为长度、dtype 为数据类型的 numpy 数组，从索引 o 开始
                inp = lambda: arange(s, dtype=dtype)[o:]
                # 创建一个空的 numpy 数组 out，长度为 s，dtype 为数据类型，从索引 o 开始
                out = empty((s,), dtype=dtype)[o:]
                # 生成一个生成器，返回 out 和 inp() 的结果，以及格式化的字符串
                yield out, inp(), ufmt % (o, o, s, dtype, 'out of place')
                # 将 inp() 的结果赋给变量 d
                d = inp()
                # 生成一个生成器，返回 d 和 d 的结果，以及格式化的字符串
                yield d, d, ufmt % (o, o, s, dtype, 'in place')
                # 生成一个生成器，返回 out[1:] 和 inp()[:-1] 的结果，以及格式化的字符串
                yield out[1:], inp()[:-1], ufmt % \
                    (o + 1, o, s - 1, dtype, 'out of place')
                # 生成一个生成器，返回 out[:-1] 和 inp()[1:] 的结果，以及格式化的字符串
                yield out[:-1], inp()[1:], ufmt % \
                    (o, o + 1, s - 1, dtype, 'out of place')
                # 生成一个生成器，返回 inp()[:-1] 和 inp()[1:] 的结果，以及格式化的字符串
                yield inp()[:-1], inp()[1:], ufmt % \
                    (o, o + 1, s - 1, dtype, 'aliased')
                # 生成一个生成器，返回 inp()[1:] 和 inp()[:-1] 的结果，以及格式化的字符串
                yield inp()[1:], inp()[:-1], ufmt % \
                    (o + 1, o, s - 1, dtype, 'aliased')
            # 如果 type 等于 'binary'，执行以下操作
            if type == 'binary':
                # 创建一个 lambda 函数 inp1，返回一个以 s 为长度、dtype 为数据类型的 numpy 数组，从索引 o 开始
                inp1 = lambda: arange(s, dtype=dtype)[o:]
                # 创建一个 lambda 函数 inp2，返回一个以 s 为长度、dtype 为数据类型的 numpy 数组，从索引 o 开始
                inp2 = lambda: arange(s, dtype=dtype)[o:]
                # 创建一个空的 numpy 数组 out，长度为 s，dtype 为数据类型，从索引 o 开始
                out = empty((s,), dtype=dtype)[o:]
                # 生成一个生成器，返回 out、inp1() 和 inp2() 的结果，以及格式化的字符串
                yield out, inp1(), inp2(),  bfmt % \
                    (o, o, o, s, dtype, 'out of place')
                # 将 inp1() 的结果赋给变量 d
                d = inp1()
                # 生成一个生成器，返回 d、d 和 inp2() 的结果，以及格式化的字符串
                yield d, d, inp2(), bfmt % \
                    (o, o, o, s, dtype, 'in place1')
                # 将 inp2() 的结果赋给变量 d
                d = inp2()
                # 生成一个生成器，返回 d、inp1() 和 d 的结果，以及格式化的字符串
                yield d, inp1(), d, bfmt % \
                    (o, o, o, s, dtype, 'in place2')
                # 生成一个生成器，返回 out[1:]、inp1()[:-1] 和 inp2()[:-1] 的结果，以及格式化的字符串
                yield out[1:], inp1()[:-1], inp2()[:-1], bfmt % \
                    (o + 1, o, o, s - 1, dtype, 'out of place')
                # 生成一个生成器，返回 out[:-1]、inp1()[1:] 和 inp2()[:-1] 的结果，以及格式化的字符串
                yield out[:-1], inp1()[1:], inp2()[:-1], bfmt % \
                    (o, o + 1, o, s - 1, dtype, 'out of place')
                # 生成一个生成器，返回 out[:-1]、inp1()[:-1] 和 inp2()[1:] 的结果，以及格式化的字符串
                yield out[:-1], inp1()[:-1], inp2()[1:], bfmt % \
                    (o, o, o + 1, s - 1, dtype, 'out of place')
                # 生成一个生成器，返回 inp1()[1:]、inp1()[:-1] 和 inp2()[:-1] 的结果，以及格式化的字符串
                yield inp1()[1:], inp1()[:-1], inp2()[:-1], bfmt % \
                    (o + 1, o, o, s - 1, dtype, 'aliased')
                # 生成一个生成器，返回 inp1()[:-1]、inp1()[1:] 和 inp2()[:-1] 的结果，以及格式化的字符串
                yield inp1()[:-1], inp1()[1:], inp2()[:-1], bfmt % \
                    (o, o + 1, o, s - 1, dtype, 'aliased')
                # 生成一个生成器，返回 inp1()[:-1]、inp1()[:-1] 和 inp2()[1:] 的结果，以及格式化的字符串
                yield inp1()[:-1], inp1()[:-1], inp2()[1:], bfmt % \
                    (o, o, o + 1, s - 1, dtype, 'aliased')
# 定义一个自定义异常类 IgnoreException，用于表示由于禁用的功能而忽略的异常
class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"
    pass


# 定义一个上下文管理器 tempdir，用于提供临时测试文件夹
@contextlib.contextmanager
def tempdir(*args, **kwargs):
    """Context manager to provide a temporary test folder.

    All arguments are passed as this to the underlying tempfile.mkdtemp
    function.

    """
    # 创建一个临时文件夹，参数通过到底层 tempfile.mkdtemp 函数
    tmpdir = mkdtemp(*args, **kwargs)
    try:
        yield tmpdir  # 返回临时文件夹路径
    finally:
        shutil.rmtree(tmpdir)  # 在退出上下文管理器时删除临时文件夹


# 定义一个上下文管理器 temppath，用于临时文件操作
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
    # 创建一个临时文件，参数直接传递给 tempfile.mkstemp 函数
    fd, path = mkstemp(*args, **kwargs)
    os.close(fd)  # 关闭文件描述符
    try:
        yield path  # 返回临时文件路径
    finally:
        os.remove(path)  # 在退出上下文管理器时删除临时文件


# 定义一个上下文管理器 clear_and_catch_warnings，用于重置警告注册表以捕获警告
class clear_and_catch_warnings(warnings.catch_warnings):
    """ Context manager that resets warning registry for catching warnings

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
    >>> with np.testing.clear_and_catch_warnings(
    ...         modules=[np._core.fromnumeric]):
    ...     warnings.simplefilter('always')
    ...     warnings.filterwarnings('ignore', module='np._core.fromnumeric')
    ...     # do something that raises a warning but ignore those in

    """
    # 初始化方法，重置警告注册表
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 进入上下文时的操作，备份并清除指定模块的警告注册表
    def __enter__(self):
        super().__enter__()
        # 在进入上下文管理器时处理警告
        # 存储和移除给定模块的 __warningregistry__ 条目
        # 这确保了可以在上下文管理器内重新触发任何警告
        pass

    # 退出上下文时的操作，恢复警告注册表的状态
    def __exit__(self, *exc_info):
        super().__exit__(*exc_info)
        # 在退出上下文管理器时恢复 __warningregistry__ 的状态
        pass
    ...     # np._core.fromnumeric
    """
    # 定义一个空元组来存放类模块信息
    class_modules = ()

    # 初始化方法，接受是否记录记录信息和模块信息作为参数
    def __init__(self, record=False, modules=()):
        # 将传入的模块信息与类模块信息合并，存储在实例变量中
        self.modules = set(modules).union(self.class_modules)
        # 初始化警告注册表备份字典
        self._warnreg_copies = {}
        # 调用父类的初始化方法
        super().__init__(record=record)

    # 进入上下文时调用的方法
    def __enter__(self):
        # 遍历所有模块
        for mod in self.modules:
            # 如果模块有 '__warningregistry__' 属性
            if hasattr(mod, '__warningregistry__'):
                # 备份模块的警告注册表，并清空原注册表
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        # 调用父类的 __enter__ 方法并返回
        return super().__enter__()

    # 退出上下文时调用的方法
    def __exit__(self, *exc_info):
        # 调用父类的 __exit__ 方法
        super().__exit__(*exc_info)
        # 还原警告注册表
        for mod in self.modules:
            # 如果模块有 '__warningregistry__' 属性，则清空当前注册表
            if hasattr(mod, '__warningregistry__'):
                mod.__warningregistry__.clear()
            # 如果模块在备份字典中，则恢复注册表为备份内容
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])
    """
# 定义一个上下文管理器和装饰器，功能与 warnings.catch_warnings 类似
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
    # 初始化函数，设置警告抑制器的初始状态和转发规则
    def __init__(self, forwarding_rule="always"):
        # 标记是否已进入警告抑制状态
        self._entered = False

        # 存储当前实例中的警告抑制列表
        self._suppressions = []

        # 检查转发规则是否合法，若不合法则抛出数值错误异常
        if forwarding_rule not in {"always", "module", "once", "location"}:
            raise ValueError("unsupported forwarding rule.")
        
        # 设置警告转发规则
        self._forwarding_rule = forwarding_rule

    # 清理警告过滤器注册表
    def _clear_registries(self):
        # 如果警告模块支持过滤器变化通知，则调用此方法
        if hasattr(warnings, "_filters_mutated"):
            warnings._filters_mutated()
            return
        
        # 否则，直接清空警告注册表，通常情况下不会造成问题
        for module in self._tmp_modules:
            if hasattr(module, "__warningregistry__"):
                module.__warningregistry__.clear()

    # 设置警告过滤条件，并记录警告信息
    def _filter(self, category=Warning, message="", module=None, record=False):
        # 如果需要记录警告信息，则创建一个列表用于存储
        if record:
            record = []  # 存储警告日志的列表
        else:
            record = None
        
        # 如果已进入警告抑制状态
        if self._entered:
            # 根据参数设置警告过滤条件
            if module is None:
                warnings.filterwarnings(
                    "always", category=category, message=message)
            else:
                # 将模块名称转换为正则表达式形式，用于模块匹配
                module_regex = module.__name__.replace('.', r'\.') + '$'
                warnings.filterwarnings(
                    "always", category=category, message=message,
                    module=module_regex)
                
                # 将匹配的模块添加到临时模块集合中，并清理相关注册表
                self._tmp_modules.add(module)
                self._clear_registries()

            # 将设置的警告抑制条件记录到临时抑制列表中
            self._tmp_suppressions.append(
                (category, message, re.compile(message, re.I), module, record))
        else:
            # 否则，将设置的警告抑制条件记录到实例抑制列表中
            self._suppressions.append(
                (category, message, re.compile(message, re.I), module, record))

        # 返回记录警告信息的列表
        return record
    # 定义一个名为 filter 的方法，用于添加或应用一个新的警告过滤器。
    # 可选参数：
    #   category: 警告类别，用于过滤特定类型的警告
    #   message: 匹配警告消息的正则表达式
    #   module: 要过滤的模块，注意必须完全匹配模块（及其文件），不能是子模块，因此可能不适用于外部模块。
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
        # 调用内部方法 _filter，传递参数 category、message、module，并设置 record 参数为 False
        self._filter(category=category, message=message, module=module,
                     record=False)

    # 定义一个名为 record 的方法，用于添加或应用一个新的警告记录过滤器。
    # 可选参数：
    #   category: 警告类别，用于过滤特定类型的警告
    #   message: 匹配警告消息的正则表达式
    #   module: 要过滤的模块，注意必须完全匹配模块（及其文件），不能是子模块，因此可能不适用于外部模块。
    # 返回：
    #   log: 一个列表，其中包含所有匹配的警告
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
        # 调用内部方法 _filter，传递参数 category、message、module，并设置 record 参数为 True
        return self._filter(category=category, message=message, module=module,
                            record=True)
    # 进入上下文管理器时调用的方法，用于设置警告处理器和过滤器
    def __enter__(self):
        # 如果已经进入过一次，抛出运行时错误
        if self._entered:
            raise RuntimeError("cannot enter suppress_warnings twice.")

        # 保存原始的警告显示函数和过滤器
        self._orig_show = warnings.showwarning
        self._filters = warnings.filters
        # 复制一份过滤器列表，以免直接修改原始过滤器
        warnings.filters = self._filters[:]

        self._entered = True
        self._tmp_suppressions = []
        self._tmp_modules = set()
        self._forwarded = set()

        self.log = []  # 重置全局日志列表（无需保持相同列表）

        # 对于每一个要抑制的警告，根据设置进行处理
        for cat, mess, _, mod, log in self._suppressions:
            if log is not None:
                del log[:]  # 清空日志
            if mod is None:
                # 如果模块为None，则按类别和消息抑制警告
                warnings.filterwarnings(
                    "always", category=cat, message=mess)
            else:
                # 如果指定了模块，则按照类别、消息和模块正则表达式抑制警告
                module_regex = mod.__name__.replace('.', r'\.') + '$'
                warnings.filterwarnings(
                    "always", category=cat, message=mess,
                    module=module_regex)
                self._tmp_modules.add(mod)
        
        # 设置警告显示函数为自定义的_showwarning函数
        warnings.showwarning = self._showwarning
        # 清理注册表
        self._clear_registries()

        # 返回当前实例，以便作为上下文管理器的对象
        return self

    # 退出上下文管理器时调用的方法，用于恢复原始的警告处理器和过滤器
    def __exit__(self, *exc_info):
        # 恢复原始的警告显示函数
        warnings.showwarning = self._orig_show
        # 恢复原始的警告过滤器
        warnings.filters = self._filters
        # 清理注册表
        self._clear_registries()
        self._entered = False
        # 删除临时保存的原始显示函数和过滤器变量
        del self._orig_show
        del self._filters
    def _showwarning(self, message, category, filename, lineno,
                     *args, use_warnmsg=None, **kwargs):
        # 反向遍历合并的警告抑制列表（_suppressions 和 _tmp_suppressions）
        for cat, _, pattern, mod, rec in (
                self._suppressions + self._tmp_suppressions)[::-1]:
            # 检查当前警告类是否是被抑制的类，并且消息匹配正则表达式模式
            if (issubclass(category, cat) and
                    pattern.match(message.args[0]) is not None):
                # 如果没有指定模块，记录或忽略警告消息
                if mod is None:
                    # 创建警告消息对象，记录到日志和记录列表中
                    if rec is not None:
                        msg = WarningMessage(message, category, filename,
                                             lineno, **kwargs)
                        self.log.append(msg)
                        rec.append(msg)
                    return
                # 使用 startswith 检查模块文件名是否匹配
                elif mod.__file__.startswith(filename):
                    # 如果消息和模块（文件名）匹配，记录到日志和记录列表中
                    if rec is not None:
                        msg = WarningMessage(message, category, filename,
                                             lineno, **kwargs)
                        self.log.append(msg)
                        rec.append(msg)
                    return

        # 如果没有过滤器，将警告消息传递给外部处理器
        # 除非设置了只传递一次的规则
        if self._forwarding_rule == "always":
            # 如果没有指定使用的警告消息对象，调用原始的警告显示方法
            if use_warnmsg is None:
                self._orig_show(message, category, filename, lineno,
                                *args, **kwargs)
            else:
                # 使用指定的警告消息对象进行显示
                self._orig_showmsg(use_warnmsg)
            return

        # 根据不同的转发规则生成唯一的签名，并检查是否已经转发过该警告消息
        if self._forwarding_rule == "once":
            signature = (message.args, category)
        elif self._forwarding_rule == "module":
            signature = (message.args, category, filename)
        elif self._forwarding_rule == "location":
            signature = (message.args, category, filename, lineno)

        if signature in self._forwarded:
            return
        # 将签名添加到已转发集合中
        self._forwarded.add(signature)
        # 如果没有指定使用的警告消息对象，调用原始的警告显示方法
        if use_warnmsg is None:
            self._orig_show(message, category, filename, lineno, *args,
                            **kwargs)
        else:
            # 使用指定的警告消息对象进行显示
            self._orig_showmsg(use_warnmsg)
@contextlib.contextmanager
def _assert_no_gc_cycles_context(name=None):
    __tracebackhide__ = True  # Hide traceback for py.test

    # 如果没有引用计数的话，测试就没有意义
    if not HAS_REFCOUNT:
        yield
        return

    # 禁用垃圾回收
    assert_(gc.isenabled())
    gc.disable()
    gc_debug = gc.get_debug()
    try:
        # 尝试进行垃圾回收，最多尝试100次
        for i in range(100):
            if gc.collect() == 0:
                break
        else:
            # 如果无法完全回收垃圾，则抛出运行时错误
            raise RuntimeError(
                "Unable to fully collect garbage - perhaps a __del__ method "
                "is creating more reference cycles?")

        # 设置垃圾回收的调试模式为DEBUG_SAVEALL
        gc.set_debug(gc.DEBUG_SAVEALL)
        yield
        # 检查在上下文中是否创建了引用循环，gc.collect 返回找到的循环中不可达对象的数量
        n_objects_in_cycles = gc.collect()
        objects_in_cycles = gc.garbage[:]
    finally:
        # 清理垃圾列表和恢复之前的垃圾回收调试模式及启用垃圾回收
        del gc.garbage[:]
        gc.set_debug(gc_debug)
        gc.enable()

    # 如果找到引用循环，则抛出断言错误
    if n_objects_in_cycles:
        name_str = f' when calling {name}' if name is not None else ''
        raise AssertionError(
            "Reference cycles were found{}: {} objects were collected, "
            "of which {} are shown below:{}"
            .format(
                name_str,
                n_objects_in_cycles,
                len(objects_in_cycles),
                ''.join(
                    "\n  {} object with id={}:\n    {}".format(
                        type(o).__name__,
                        id(o),
                        pprint.pformat(o).replace('\n', '\n    ')
                    ) for o in objects_in_cycles
                )
            )
        )


def assert_no_gc_cycles(*args, **kwargs):
    """
    如果给定的可调用对象产生任何引用循环，则失败。

    如果省略所有参数调用，则可以作为上下文管理器使用::

        with assert_no_gc_cycles():
            do_something()

    .. versionadded:: 1.15.0

    Parameters
    ----------
    func : callable
        要测试的可调用对象。
    \\*args : Arguments
        传递给 `func` 的位置参数。
    \\*\\*kwargs : Kwargs
        传递给 `func` 的关键字参数。

    Returns
    -------
    Nothing. 故意丢弃结果以确保找到所有循环。

    """
    if not args:
        return _assert_no_gc_cycles_context()

    func = args[0]
    args = args[1:]
    with _assert_no_gc_cycles_context(name=func.__name__):
        func(*args, **kwargs)


def break_cycles():
    """
    通过调用 gc.collect 打破引用循环
    对象可以调用其他对象的方法（例如，另一个对象的 __del__）在其自己的 __del__ 内部。在 PyPy 上，
    解释器只在调用 gc.collect 之间运行，因此需要多次调用才能完全释放所有循环。
    """

    gc.collect()
    # 如果运行环境是 PyPy（一个 Python 解释器的替代实现），则执行以下代码块
    if IS_PYPY:
        # 多次调用 gc.collect() 来确保所有的 finalizer 都被调用
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()
# 定义装饰器函数，用于检查可用内存是否足够，并根据情况跳过测试
def requires_memory(free_bytes):
    """Decorator to skip a test if not enough memory is available"""
    import pytest  # 导入 pytest 库

    def decorator(func):
        @wraps(func)
        def wrapper(*a, **kw):
            # 检查可用内存是否达到要求，如果不足则跳过测试
            msg = check_free_memory(free_bytes)
            if msg is not None:
                pytest.skip(msg)

            try:
                return func(*a, **kw)
            except MemoryError:
                # 如果发生内存错误，则标记测试为预期失败
                pytest.xfail("MemoryError raised")

        return wrapper

    return decorator


# 检查当前系统下是否有足够的可用内存
def check_free_memory(free_bytes):
    """
    Check whether `free_bytes` amount of memory is currently free.
    Returns: None if enough memory available, otherwise error message
    """
    env_var = 'NPY_AVAILABLE_MEM'
    env_value = os.environ.get(env_var)  # 获取环境变量 NPY_AVAILABLE_MEM 的值
    if env_value is not None:
        try:
            mem_free = _parse_size(env_value)  # 尝试解析环境变量值表示的内存大小
        except ValueError as exc:
            raise ValueError(f'Invalid environment variable {env_var}: {exc}')

        # 根据环境变量值和所需内存大小计算提示消息
        msg = (f'{free_bytes/1e9} GB memory required, but environment variable '
               f'NPY_AVAILABLE_MEM={env_value} set')
    else:
        mem_free = _get_mem_available()  # 获取系统可用内存大小

        if mem_free is None:
            # 如果无法获取可用内存大小，则提示设置环境变量 NPY_AVAILABLE_MEM
            msg = ("Could not determine available memory; set NPY_AVAILABLE_MEM "
                   "environment variable (e.g. NPY_AVAILABLE_MEM=16GB) to run "
                   "the test.")
            mem_free = -1
        else:
            # 计算提示消息，显示所需内存和当前可用内存大小
            msg = f'{free_bytes/1e9} GB memory required, but {mem_free/1e9} GB available'

    # 如果可用内存小于所需内存，则返回错误消息；否则返回 None
    return msg if mem_free < free_bytes else None


# 解析内存大小字符串（例如 '12 GB'）并转换为字节大小
def _parse_size(size_str):
    """Convert memory size strings ('12 GB' etc.) to float"""
    suffixes = {'': 1, 'b': 1,
                'k': 1000, 'm': 1000**2, 'g': 1000**3, 't': 1000**4,
                'kb': 1000, 'mb': 1000**2, 'gb': 1000**3, 'tb': 1000**4,
                'kib': 1024, 'mib': 1024**2, 'gib': 1024**3, 'tib': 1024**4}

    size_re = re.compile(r'^\s*(\d+|\d+\.\d+)\s*({0})\s*$'.format(
        '|'.join(suffixes.keys())), re.I)

    m = size_re.match(size_str.lower())
    if not m or m.group(2) not in suffixes:
        raise ValueError(f'value {size_str!r} not a valid size')
    return int(float(m.group(1)) * suffixes[m.group(2)])


# 获取系统当前可用内存大小（仅限 Linux 平台）
def _get_mem_available():
    """Return available memory in bytes, or None if unknown."""
    try:
        import psutil  # 尝试导入 psutil 库
        return psutil.virtual_memory().available  # 返回虚拟内存的可用大小
    except (ImportError, AttributeError):
        pass

    if sys.platform.startswith('linux'):
        # 如果是 Linux 系统，则从 /proc/meminfo 文件中获取内存信息
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                p = line.split()
                info[p[0].strip(':').lower()] = int(p[1]) * 1024

        if 'memavailable' in info:
            # 对于 Linux 版本 >= 3.14，返回可用内存大小
            return info['memavailable']
        else:
            # 对于较早版本的 Linux，估算可用内存大小
            return info['memfree'] + info['cached']

    return None


def _no_tracing(func):
    """
    Decorator to temporarily turn off tracing for the duration of a test.
    Needed in tests that check refcounting, otherwise the tracing itself
    influences the refcounts
    """
    如果当前环境不支持跟踪功能（sys 模块中没有 gettrace 方法），直接返回原函数。
    否则，定义一个装饰器函数 wrapper，用于临时关闭跟踪功能。
    @wraps(func)  # 使用 functools.wraps 来保留原始函数的元数据
    def wrapper(*args, **kwargs):
        # 保存当前的跟踪函数引用，以便恢复
        original_trace = sys.gettrace()
        try:
            # 设置当前跟踪函数为 None，即关闭跟踪功能
            sys.settrace(None)
            # 调用原始函数，此时跟踪功能已关闭
            return func(*args, **kwargs)
        finally:
            # 恢复原始的跟踪函数
            sys.settrace(original_trace)
    # 返回装饰后的函数 wrapper
    return wrapper
# 获取当前系统上的 GNU C 库版本信息
def _get_glibc_version():
    try:
        # 尝试获取 GNU C 库版本信息并从中提取具体版本号
        ver = os.confstr('CS_GNU_LIBC_VERSION').rsplit(' ')[1]
    except Exception:
        # 如果获取失败，将版本号设置为 '0.0'
        ver = '0.0'

    # 返回获取到的 GNU C 库版本号
    return ver

# 获取当前系统上的 GNU C 库版本号
_glibcver = _get_glibc_version()

# 检查当前 GNU C 库版本是否早于指定版本 x
_glibc_older_than = lambda x: (_glibcver != '0.0' and _glibcver < x)
```