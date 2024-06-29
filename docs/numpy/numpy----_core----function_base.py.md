# `.\numpy\numpy\_core\function_base.py`

```py
import functools  # 导入 functools 模块，用于创建偏函数
import warnings  # 导入 warnings 模块，用于警告处理
import operator  # 导入 operator 模块，用于操作符函数
import types  # 导入 types 模块，用于动态类型创建

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from . import numeric as _nx  # 从当前包中导入 numeric 模块，并使用 _nx 别名
from .numeric import result_type, nan, asanyarray, ndim  # 从 numeric 模块导入指定函数
from numpy._core.multiarray import add_docstring  # 从核心 multiarray 模块导入函数
from numpy._core._multiarray_umath import _array_converter  # 从核心 _multiarray_umath 模块导入函数
from numpy._core import overrides  # 从核心模块导入 overrides 函数

__all__ = ['logspace', 'linspace', 'geomspace']  # 定义模块的公共接口

# 创建一个偏函数 array_function_dispatch，通过 functools.partial 进行部分应用
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _linspace_dispatcher(start, stop, num=None, endpoint=None, retstep=None,
                         dtype=None, axis=None, *, device=None):
    return (start, stop)  # 返回 start 和 stop 参数的元组


# 使用 array_function_dispatch 装饰器，将 _linspace_dispatcher 函数注册为 array function
@array_function_dispatch(_linspace_dispatcher)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0, *, device=None):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    .. versionchanged:: 1.16.0
        Non-scalar `start` and `stop` are now supported.

    .. versionchanged:: 1.20.0
        Values are rounded towards ``-inf`` instead of ``0`` when an
        integer ``dtype`` is specified. The old behavior can
        still be obtained with ``np.linspace(start, stop, num).astype(int)``

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred dtype will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.

        .. versionadded:: 1.9.0
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

        .. versionadded:: 1.16.0
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    # 将 num 转换为整数索引
    num = operator.index(num)
    # 如果 num 小于 0，则抛出数值错误异常
    if num < 0:
        raise ValueError(
            "Number of samples, %s, must be non-negative." % num
        )
    # 如果 endpoint 为 False，则将 div 设置为 num，否则设置为 num - 1
    div = (num - 1) if endpoint else num

    # 转换起始点和结束点，获取转换后的起始点和结束点以及数据类型
    conv = _array_converter(start, stop)
    start, stop = conv.as_arrays()
    dt = conv.result_type(ensure_inexact=True)

    # 如果未指定 dtype，则将其设置为 dt，并将 integer_dtype 设置为 False
    if dtype is None:
        dtype = dt
        integer_dtype = False
    else:
        # 检查 dtype 是否为整数类型
        integer_dtype = _nx.issubdtype(dtype, _nx.integer)

    # 使用 dtype=type(dt) 强制浮点点评估：
    # 计算 delta，即停止点与起始点之差
    delta = np.subtract(stop, start, dtype=type(dt))
    
    # 创建一个数组 y，使用 arange 函数生成从 0 到 num-1 的一维数组，根据 delta 的维度进行形状调整
    y = _nx.arange(
        0, num, dtype=dt, device=device
    ).reshape((-1,) + (1,) * ndim(delta))

    # 对于 div 大于 0 的情况：
    if div > 0:
        # 判断是否为标量 delta，用于决定是否进行就地乘法优化
        _mult_inplace = _nx.isscalar(delta)
        # 计算步长 step，即 delta 除以 div
        step = delta / div
        # 判断是否存在步长为零的情况
        any_step_zero = (
            step == 0 if _mult_inplace else _nx.asanyarray(step == 0).any())
        # 如果存在步长为零的情况：
        if any_step_zero:
            # 将 y 除以 div，并根据 _mult_inplace 进行就地乘法或普通乘法
            y /= div
            if _mult_inplace:
                y *= delta
            else:
                y = y * delta
        else:
            # 如果步长不为零，根据 _mult_inplace 进行就地乘法或普通乘法
            if _mult_inplace:
                y *= step
            else:
                y = y * step
    # 如果不满足前述条件，则步长设为NaN，表示未定义的步长
    else:
        step = nan
        
    # 将y与delta相乘，允许对输出类进行可能的覆盖
    y = y * delta

    # 将起始值start加到y上
    y += start

    # 如果需要包含终点且序列长度大于1，则将最后一个元素设为停止值stop
    if endpoint and num > 1:
        y[-1, ...] = stop

    # 如果axis不等于0，则将y数组的轴移动到指定的axis位置
    if axis != 0:
        y = _nx.moveaxis(y, 0, axis)

    # 如果需要整数类型，则将y向下取整
    if integer_dtype:
        _nx.floor(y, out=y)

    # 将y数组转换为指定的dtype，并使用conv.wrap进行包装处理
    y = conv.wrap(y.astype(dtype, copy=False))
    
    # 如果需要返回步长，则返回y和step
    if retstep:
        return y, step
    # 否则只返回y
    else:
        return y
# 定义日志空间函数的调度器，用于分派参数到具体的实现函数
def _logspace_dispatcher(start, stop, num=None, endpoint=None, base=None,
                         dtype=None, axis=None):
    # 返回起始点、终止点和基数，作为元组
    return (start, stop, base)


# 使用装饰器将_logspace_dispatcher函数注册为logspace函数的分发函数
@array_function_dispatch(_logspace_dispatcher)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None,
             axis=0):
    """
    Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

    .. versionchanged:: 1.16.0
        Non-scalar `start` and `stop` are now supported.

    .. versionchanged:: 1.25.0
        Non-scalar 'base` is now supported

    Parameters
    ----------
    start : array_like
        ``base ** start`` is the starting value of the sequence.
    stop : array_like
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : array_like, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred type will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start,
        stop, or base are array-like.  By default (0), the samples will be
        along a new axis inserted at the beginning. Use -1 to get an axis at
        the end.

        .. versionadded:: 1.16.0


    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    arange : Similar to linspace, with the step size specified instead of the
             number of samples. Note that, when used with a float endpoint, the
             endpoint may or may not be included.
    linspace : Similar to logspace, but with the samples uniformly distributed
               in linear space, instead of log space.
    geomspace : Similar to logspace, but with endpoints specified directly.
    :ref:`how-to-partition`

    Notes
    -----
    If base is a scalar, logspace is equivalent to the code

    >>> y = np.linspace(start, stop, num=num, endpoint=endpoint)
    ... # doctest: +SKIP
    >>> power(base, y).astype(dtype)
    ... # doctest: +SKIP

    Examples
    --------
    >>> np.logspace(2.0, 3.0, num=4)
    """
    array([ 100.        ,  215.443469  ,  464.15888336, 1000.        ])
    # 创建一个包含四个元素的 NumPy 数组，这些元素是以对数刻度在 2.0 到 3.0 之间均匀分布的值

    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)
    # 使用对数刻度生成器创建一个包含四个元素的 NumPy 数组，基于 2.0 到 3.0 之间的对数刻度，但不包括终点值

    array([100.        ,  177.827941  ,  316.22776602,  562.34132519])
    # 上述生成器生成的 NumPy 数组，包含了四个对数刻度值，第一个元素为 100.0

    >>> np.logspace(2.0, 3.0, num=4, base=2.0)
    # 使用指定的基数 2.0，生成一个包含四个元素的 NumPy 数组，这些元素是以对数刻度在 2.0 到 3.0 之间均匀分布的值

    array([4.        ,  5.0396842 ,  6.34960421,  8.        ])
    # 使用基数 2.0 生成的 NumPy 数组，包含了四个对数刻度值

    >>> np.logspace(2.0, 3.0, num=4, base=[2.0, 3.0], axis=-1)
    # 使用不同的基数数组 [2.0, 3.0]，在最后一个轴上生成一个二维 NumPy 数组，包含了对数刻度值

    array([[ 4.        ,  5.0396842 ,  6.34960421,  8.        ],
           [ 9.        , 12.98024613, 18.72075441, 27.        ]])
    # 上述生成器生成的二维 NumPy 数组，每行代表一个基数，每列代表对应的对数刻度值

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    # 导入 matplotlib 的 pyplot 模块

    >>> N = 10
    # 定义变量 N，并赋值为 10

    >>> x1 = np.logspace(0.1, 1, N, endpoint=True)
    # 使用对数刻度生成器生成一个包含 N 个元素的 NumPy 数组 x1，包含了在 0.1 到 1 之间的对数刻度值，包括终点值

    >>> x2 = np.logspace(0.1, 1, N, endpoint=False)
    # 使用对数刻度生成器生成一个包含 N 个元素的 NumPy 数组 x2，包含了在 0.1 到 1 之间的对数刻度值，不包括终点值

    >>> y = np.zeros(N)
    # 创建一个长度为 N 的全零 NumPy 数组 y

    >>> plt.plot(x1, y, 'o')
    # 在图中绘制以 x1 为横坐标，y 为纵坐标的散点图，点形状为圆圈'o'

    [<matplotlib.lines.Line2D object at 0x...>]
    # 返回一个 matplotlib.lines.Line2D 对象的列表，表示绘制的散点图线条

    >>> plt.plot(x2, y + 0.5, 'o')
    # 在图中绘制以 x2 为横坐标，y + 0.5 为纵坐标的散点图，点形状为圆圈'o'

    [<matplotlib.lines.Line2D object at 0x...>]
    # 返回一个 matplotlib.lines.Line2D 对象的列表，表示绘制的散点图线条

    >>> plt.ylim([-0.5, 1])
    # 设置纵坐标轴的范围为 -0.5 到 1

    (-0.5, 1)
    # 返回设置的纵坐标轴范围的元组

    >>> plt.show()
    # 显示绘制的整个图形

    """
    if not isinstance(base, (float, int)) and np.ndim(base):
        # 如果 base 不是浮点数或整数，并且是一个多维数组，则对其进行广播，因为它可能影响轴的解释方式。
        # 计算 start、stop 和 base 的广播维度的最大值
        ndmax = np.broadcast(start, stop, base).ndim
        # 将 start、stop 和 base 转换为广播后的数组
        start, stop, base = (
            np.array(a, copy=None, subok=True, ndmin=ndmax)
            for a in (start, stop, base)
        )
        # 在指定轴上扩展 base 数组的维度
        base = np.expand_dims(base, axis=axis)
    # 使用 linspace 函数生成在指定范围内的均匀分布数组 y
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    if dtype is None:
        # 如果未指定 dtype，则返回 base 的 y 次幂作为结果
        return _nx.power(base, y)
    # 否则返回转换为指定 dtype 的 base 的 y 次幂作为结果
    return _nx.power(base, y).astype(dtype, copy=False)
# 定义一个分派函数 `_geomspace_dispatcher`，用于分发参数到具体的函数处理
def _geomspace_dispatcher(start, stop, num=None, endpoint=None, dtype=None,
                          axis=None):
    # 返回 start 和 stop 参数的元组
    return (start, stop)


# 使用装饰器 `array_function_dispatch` 对 `_geomspace_dispatcher` 进行装饰
@array_function_dispatch(_geomspace_dispatcher)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """
    Return numbers spaced evenly on a log scale (a geometric progression).

    This is similar to `logspace`, but with endpoints specified directly.
    Each output sample is a constant multiple of the previous.

    .. versionchanged:: 1.16.0
        Non-scalar `start` and `stop` are now supported.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The final value of the sequence, unless `endpoint` is False.
        In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred dtype will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

        .. versionadded:: 1.16.0

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    logspace : Similar to geomspace, but with endpoints specified using log
               and base.
    linspace : Similar to geomspace, but with arithmetic instead of geometric
               progression.
    arange : Similar to linspace, with the step size specified instead of the
             number of samples.
    :ref:`how-to-partition`

    Notes
    -----
    If the inputs or dtype are complex, the output will follow a logarithmic
    spiral in the complex plane.  (There are an infinite number of spirals
    passing through two points; the output will follow the shortest such path.)

    Examples
    --------
    >>> np.geomspace(1, 1000, num=4)
    array([    1.,    10.,   100.,  1000.])
    >>> np.geomspace(1, 1000, num=3, endpoint=False)
    array([   1.,   10.,  100.])
    >>> np.geomspace(1, 1000, num=4, endpoint=False)
    array([   1.        ,    5.62341325,   31.6227766 ,  177.827941  ])
    >>> np.geomspace(1, 256, num=9)
    array([   1.,    2.,    4.,    8.,   16.,   32.,   64.,  128.,  256.])

    Note that the above may not produce exact integers:

    >>> np.geomspace(1, 256, num=9, dtype=int)
    """
    # 函数文档字符串已经提供了对函数功能和参数的详细解释，无需额外注释
    pass  # 实际上，函数体未提供任何具体的实现，只是作为函数文档的容器存在
    """
    Compute a sequence of numbers in geometric progression.

    Parameters:
    - `start`: array_like
        The starting value of the sequence.
    - `stop`: array_like
        The end value of the sequence, unless `endpoint` is False.
    - `num`: int, optional
        Number of samples to generate. Default is 50.
    - `endpoint`: bool, optional
        If True, `stop` is the last sample. If False, it's not included.
    - `base`: float, optional
        The base of the geometric progression. Default is 10.0.
    - `dtype`: dtype, optional
        The data type of the output array. If not specified, it's determined from inputs.

    Returns:
    - `result`: ndarray
        The array of values in geometric progression.

    Examples:
    >>> np.geomspace(1, 256, num=9)
    array([  1,   2,   4,   8,  16,  32,  64, 128, 256])

    >>> np.geomspace(1000, 1, num=4)
    array([1000.,  100.,   10.,    1.])

    >>> np.geomspace(-1000, -1, num=4)
    array([-1000.,  -100.,   -10.,    -1.])

    >>> np.geomspace(1j, 1000j, num=4)
    array([0.   +1.j, 0.  +10.j, 0. +100.j, 0.+1000.j])

    >>> np.geomspace(-1+0j, 1+0j, num=5)
    array([-1.00000000e+00+1.22464680e-16j, -7.07106781e-01+7.07106781e-01j,
            6.12323400e-17+1.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
            1.00000000e+00+0.00000000e+00j])

    >>> import matplotlib.pyplot as plt
    >>> N = 10
    >>> y = np.zeros(N)
    >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint=True), y + 1, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint=False), y + 2, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.axis([0.5, 2000, 0, 3])
    [0.5, 2000, 0, 3]
    >>> plt.grid(True, color='0.7', linestyle='-', which='both', axis='both')
    >>> plt.show()
    """

    # Convert input arguments to numpy arrays
    start = asanyarray(start)
    stop = asanyarray(stop)

    # Check for zero values in start or stop, as geometric sequence cannot include zero
    if _nx.any(start == 0) or _nx.any(stop == 0):
        raise ValueError('Geometric sequence cannot include zero')

    # Determine the data type for the result array
    dt = result_type(start, stop, float(num), _nx.zeros((), dtype))

    # If `dtype` is provided, convert it to numpy dtype
    if dtype is None:
        dtype = dt
    else:
        dtype = _nx.dtype(dtype)

    # Ensure start and stop are of the same data type
    start = start.astype(dt, copy=True)
    stop = stop.astype(dt, copy=True)

    # Normalize start and stop for negative real and complex inputs
    out_sign = _nx.sign(start)
    start /= out_sign
    stop = stop / out_sign

    # Compute logarithms of start and stop values
    log_start = _nx.log10(start)
    log_stop = _nx.log10(stop)

    # Compute the geometric progression using logspace
    result = logspace(log_start, log_stop, num=num,
                      endpoint=endpoint, base=10.0, dtype=dt)

    # Adjust endpoints to match the start and stop arguments
    if num > 0:
        result[0] = start
        if num > 1 and endpoint:
            result[-1] = stop

    # Adjust the sign of the result array
    result *= out_sign

    # Move axis if necessary
    if axis != 0:
        result = _nx.moveaxis(result, 0, axis)

    # Return the final result array with the specified dtype
    return result.astype(dtype, copy=False)
def _needs_add_docstring(obj):
    """
    Returns true if the only way to set the docstring of `obj` from python is
    via add_docstring.

    This function errs on the side of being overly conservative.
    """
    # 定义 CPython 中的 Py_TPFLAGS_HEAPTYPE 标志
    Py_TPFLAGS_HEAPTYPE = 1 << 9

    # 如果 obj 是函数类型、方法类型或者属性类型，则返回 False
    if isinstance(obj, (types.FunctionType, types.MethodType, property)):
        return False

    # 如果 obj 是类型对象并且其 __flags__ 属性包含 Py_TPFLAGS_HEAPTYPE 标志，则返回 False
    if isinstance(obj, type) and obj.__flags__ & Py_TPFLAGS_HEAPTYPE:
        return False

    # 否则返回 True，即需要通过 add_docstring 来设置 obj 的文档字符串
    return True


def _add_docstring(obj, doc, warn_on_python):
    """
    Add a docstring `doc` to the object `obj`, optionally warn if attaching
    to a pure-python object.

    Parameters
    ----------
    obj : object
        The object to attach the docstring to.
    doc : str
        The docstring to attach.
    warn_on_python : bool
        Whether to emit a warning if attaching docstring to a pure-python object.

    Notes
    -----
    If `warn_on_python` is True and `_needs_add_docstring` returns False for `obj`,
    emit a UserWarning.

    Attempt to add `doc` as the docstring to `obj`. If an exception occurs during
    this operation, it is caught and ignored.
    """
    # 如果 warn_on_python 为 True 且 _needs_add_docstring 返回 False，则发出警告
    if warn_on_python and not _needs_add_docstring(obj):
        warnings.warn(
            "add_newdoc was used on a pure-python object {}. "
            "Prefer to attach it directly to the source."
            .format(obj),
            UserWarning,
            stacklevel=3)
    
    # 尝试将 doc 添加为 obj 的文档字符串
    try:
        add_docstring(obj, doc)
    except Exception:
        pass


def add_newdoc(place, obj, doc, warn_on_python=True):
    """
    Add documentation to an existing object, typically one defined in C

    The purpose is to allow easier editing of the docstrings without requiring
    a re-compile. This exists primarily for internal use within numpy itself.

    Parameters
    ----------
    place : str
        The absolute name of the module to import from
    obj : str or None
        The name of the object to add documentation to, typically a class or
        function name.
    doc : {str, Tuple[str, str], List[Tuple[str, str]]}
        If a string, the documentation to apply to `obj`

        If a tuple, then the first element is interpreted as an attribute
        of `obj` and the second as the docstring to apply -
        ``(method, docstring)``

        If a list, then each element of the list should be a tuple of length
        two - ``[(method1, docstring1), (method2, docstring2), ...]``
    warn_on_python : bool, optional
        If True, emit `UserWarning` if this is used to attach documentation
        to a pure-python object. Default is True.

    Notes
    -----
    This routine never raises an error if the docstring can't be written, but
    will raise an error if the object being documented does not exist.

    This routine cannot modify read-only docstrings, as appear
    in new-style classes or built-in functions. Because this
    routine never raises an error the caller must check manually
    that the docstrings were changed.

    Since this function grabs the ``char *`` from a c-level str object and puts
    it into the ``tp_doc`` slot of the type of `obj`, it violates a number of
    C-API best-practices, by:

    - modifying a `PyTypeObject` after calling `PyType_Ready`
    - calling `Py_INCREF` on the str and losing the reference, so the str
      will never be released

    If possible it should be avoided.
    """
    # 根据 place 和 obj 导入模块，并获取其中的 obj 对象
    new = getattr(__import__(place, globals(), {}, [obj]), obj)
    
    # 如果 doc 是字符串类型，调用 _add_docstring 添加文档字符串给 new 对象
    if isinstance(doc, str):
        _add_docstring(new, doc.strip(), warn_on_python)
    # 如果 doc 是一个元组（tuple）类型
    elif isinstance(doc, tuple):
        # 将元组拆解成两个变量 attr 和 docstring
        attr, docstring = doc
        # 获取 new 对象中名为 attr 的属性，并添加文档字符串
        _add_docstring(getattr(new, attr), docstring.strip(), warn_on_python)
    
    # 如果 doc 是一个列表（list）类型
    elif isinstance(doc, list):
        # 遍历列表中的每个元素，每个元素是一个元组 (attr, docstring)
        for attr, docstring in doc:
            # 获取 new 对象中名为 attr 的属性，并添加文档字符串
            _add_docstring(
                getattr(new, attr), docstring.strip(), warn_on_python
            )
```