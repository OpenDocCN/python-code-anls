# `.\numpy\numpy\lib\_scimath_impl.py`

```py
"""
Wrapper functions to more user-friendly calling of certain math functions
whose output data-type is different than the input data-type in certain
domains of the input.

For example, for functions like `log` with branch cuts, the versions in this
module provide the mathematically valid answers in the complex plane::

  >>> import math
  >>> np.emath.log(-math.exp(1)) == (1+1j*math.pi)
  True

Similarly, `sqrt`, other base logarithms, `power` and trig functions are
correctly handled.  See their respective docstrings for specific examples.

Functions
---------

.. autosummary::
   :toctree: generated/

   sqrt
   log
   log2
   logn
   log10
   power
   arccos
   arcsin
   arctanh

"""
import numpy._core.numeric as nx
import numpy._core.numerictypes as nt
from numpy._core.numeric import asarray, any
from numpy._core.overrides import array_function_dispatch
from numpy.lib._type_check_impl import isreal


__all__ = [
    'sqrt', 'log', 'log2', 'logn', 'log10', 'power', 'arccos', 'arcsin',
    'arctanh'
    ]


# Precompute the natural logarithm of 2
_ln2 = nx.log(2.0)


def _tocomplex(arr):
    """
    Convert its input `arr` to a complex array.

    The input is returned as a complex array of the smallest type that will fit
    the original data: types like single, byte, short, etc. become csingle,
    while others become cdouble.

    A copy of the input is always made.

    Parameters
    ----------
    arr : array

    Returns
    -------
    array
        An array with the same input data as the input but in complex form.

    Examples
    --------

    First, consider an input of type short:

    >>> a = np.array([1,2,3],np.short)

    >>> ac = np.lib.scimath._tocomplex(a); ac
    array([1.+0.j, 2.+0.j, 3.+0.j], dtype=complex64)

    >>> ac.dtype
    dtype('complex64')

    If the input is of type double, the output is correspondingly of the
    complex double type as well:

    >>> b = np.array([1,2,3],np.double)

    >>> bc = np.lib.scimath._tocomplex(b); bc
    array([1.+0.j, 2.+0.j, 3.+0.j])

    >>> bc.dtype
    dtype('complex128')

    Note that even if the input was complex to begin with, a copy is still
    made, since the astype() method always copies:

    >>> c = np.array([1,2,3],np.csingle)

    >>> cc = np.lib.scimath._tocomplex(c); cc
    array([1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)

    >>> c *= 2; c
    array([2.+0.j,  4.+0.j,  6.+0.j], dtype=complex64)

    >>> cc
    array([1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)
    """
    if issubclass(arr.dtype.type, (nt.single, nt.byte, nt.short, nt.ubyte,
                                   nt.ushort, nt.csingle)):
        return arr.astype(nt.csingle)
    else:
        return arr.astype(nt.cdouble)


def _fix_real_lt_zero(x):
    """
    Convert `x` to complex if it has real, negative components.

    Otherwise, output is just the array version of the input (via asarray).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array

    Examples
    --------
    """
    # 调用 numpy 库中 scimath 模块的 _fix_real_lt_zero 函数，处理输入的数组 x，返回结果数组
    >>> np.lib.scimath._fix_real_lt_zero([1,2])
    array([1, 2])
    
    # 调用 numpy 库中 scimath 模块的 _fix_real_lt_zero 函数，处理输入的数组 x，如果数组中有实数小于零的元素，将其转换为复数形式返回
    >>> np.lib.scimath._fix_real_lt_zero([-1,2])
    array([-1.+0.j,  2.+0.j])
    
    """
    # 将输入参数 x 转换为 numpy 数组
    x = asarray(x)
    # 如果数组 x 中存在实数且小于零的元素，则将数组转换为复数形式
    if any(isreal(x) & (x < 0)):
        x = _tocomplex(x)
    # 返回处理后的数组 x
    return x
# 将输入 `x` 转换为 numpy 数组
x = asarray(x)
# 如果 `x` 中存在实数且小于零的元素，则将 `x` 中所有元素转换为浮点数类型
if any(isreal(x) & (x < 0)):
    x = x * 1.0
# 返回处理后的数组 `x`
return x



# 将输入 `x` 转换为 numpy 数组
x = asarray(x)
# 如果 `x` 中存在实数且绝对值大于 1 的元素，则将 `x` 转换为复数类型
if any(isreal(x) & (abs(x) > 1)):
    x = _tocomplex(x)
# 返回处理后的数组 `x`
return x



# 定义一个函数 `_unary_dispatcher`，接收参数 `x`，返回元组 (x,)
def _unary_dispatcher(x):
    return (x,)



# 使用 `array_function_dispatch` 装饰器，将 `_unary_dispatcher` 函数注册为 `sqrt` 函数的分发器
@array_function_dispatch(_unary_dispatcher)
# 定义 `sqrt` 函数，计算输入 `x` 的平方根
def sqrt(x):
    # 调用 `_fix_real_lt_zero` 函数处理 `x`
    x = _fix_real_lt_zero(x)
    # 使用 `nx.sqrt` 计算 `x` 的平方根并返回结果
    return nx.sqrt(x)



# 使用 `array_function_dispatch` 装饰器，将 `_unary_dispatcher` 函数注册为 `log` 函数的分发器
@array_function_dispatch(_unary_dispatcher)
# 定义 `log` 函数，计算输入 `x` 的自然对数
def log(x):
    # 调用 `_fix_real_lt_zero` 函数处理 `x`
    x = _fix_real_lt_zero(x)
    # 返回 `nx.log` 计算后的对数值
    return nx.log(x)
    # 将输入参数 x 修正为非正实数，以处理复杂数的特定情况
    x = _fix_real_lt_zero(x)
    # 调用 numpy 中的特殊对数函数 nx.log 处理修正后的参数 x，返回结果
    return nx.log(x)
# 使用装饰器将函数log10分派给_unary_dispatcher处理
@array_function_dispatch(_unary_dispatcher)
# 定义计算以10为底对数的函数log10，接受参数x
def log10(x):
    """
    计算以10为底对数的值 `x`。

    返回对数的“主值”（关于此的描述，请参见`numpy.log10`）:math:`log_{10}(x)`。
    对于实数 `x > 0`，返回实数（`log10(0)` 返回 `-inf`，`log10(np.inf)` 返回 `inf`）。
    否则，返回复数的主值。

    Parameters
    ----------
    x : array_like or scalar
       需要计算对数的值。

    Returns
    -------
    out : ndarray or scalar
       `x` 值的以10为底的对数。如果 `x` 是标量，则 `out` 也是标量，否则返回数组对象。

    See Also
    --------
    numpy.log10

    Notes
    -----
    对于 `real x < 0` 返回 `NAN` 的log10()，请使用 `numpy.log10`
    （注意，除此之外 `numpy.log10` 和此 `log10` 是相同的，即对于 `x = 0` 都返回 `-inf`，
    对于 `x = inf` 都返回 `inf`，特别地，如果 `x.imag != 0` 返回复数的主值）。

    Examples
    --------

    (We set the printing precision so the example can be auto-tested)

    >>> np.set_printoptions(precision=4)

    >>> np.emath.log10(10**1)
    1.0

    >>> np.emath.log10([-10**1, -10**2, 10**2])
    array([1.+1.3644j, 2.+1.3644j, 2.+0.j    ])

    """
    # 修正实数小于零的情况
    x = _fix_real_lt_zero(x)
    # 调用nx.log10函数计算log10值并返回
    return nx.log10(x)


# 使用装饰器将函数logn分派给_logn_dispatcher处理
@array_function_dispatch(_logn_dispatcher)
# 定义计算以n为底对数的函数logn，接受参数n和x
def logn(n, x):
    """
    计算以 `n` 为底对 `x` 的对数。

    如果 `x` 包含负数输入，则在复数域中计算并返回结果。

    Parameters
    ----------
    n : array_like
       底数的整数基数。
    x : array_like
       需要计算对数的值。

    Returns
    -------
    out : ndarray or scalar
       `x` 值以 `n` 为底的对数。如果 `x` 是标量，则 `out` 也是标量，否则返回数组。

    Examples
    --------
    >>> np.set_printoptions(precision=4)

    >>> np.emath.logn(2, [4, 8])
    array([2., 3.])
    >>> np.emath.logn(2, [-4, -8, 8])
    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])

    """
    # 修正实数小于零的情况
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    # 计算log(x)/log(n)，返回以n为底对x的对数
    return nx.log(x)/nx.log(n)


# 使用装饰器将函数log2分派给_unary_dispatcher处理
@array_function_dispatch(_unary_dispatcher)
# 定义计算以2为底对数的函数log2，接受参数x
def log2(x):
    """
    计算以2为底对数的值 `x`。

    返回对数的“主值”（关于此的描述，请参见`numpy.log2`）:math:`log_2(x)`。
    对于实数 `x > 0`，返回实数（`log2(0)` 返回 `-inf`，`log2(np.inf)` 返回 `inf`）。
    否则，返回复数的主值。

    Parameters
    ----------
    x : array_like
       需要计算对数的值。

    Returns
    -------
    out : ndarray or scalar
       `x` 值的以2为底的对数。如果 `x` 是标量，则 `out` 也是标量，否则返回数组。

    See Also
    --------
    numpy.log2

    Notes
    -----

    """
    # 修正实数小于零的情况
    x = _fix_real_lt_zero(x)
    # 调用nx.log2函数计算log2值并返回
    return nx.log2(x)
    For a log2() that returns ``NAN`` when real `x < 0`, use `numpy.log2`
    (note, however, that otherwise `numpy.log2` and this `log2` are
    identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,
    and, notably, the complex principle value if ``x.imag != 0``).



    Examples
    --------
    We set the printing precision so the example can be auto-tested:

    >>> np.set_printoptions(precision=4)

    >>> np.emath.log2(8)
    3.0
    >>> np.emath.log2([-4, -8, 8])
    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])



    """
    x = _fix_real_lt_zero(x)
    调用修复负实数的函数 `_fix_real_lt_zero()`，确保 x >= 0
    return nx.log2(x)
    返回以自然对数为底的 x 的对数，使用 numpy 的 log2 函数 nx.log2()
# 定义一个用于分派的函数，返回元组 (x, p)
def _power_dispatcher(x, p):
    return (x, p)


# 使用 _power_dispatcher 注册为 array 函数分派的处理器
@array_function_dispatch(_power_dispatcher)
def power(x, p):
    """
    返回 x 的 p 次方，即 x**p。

    如果 x 包含负值，则输出将转换为复数域。

    Parameters
    ----------
    x : array_like
        输入值。
    p : array_like of ints
        x 被提升到的幂次数。如果 x 包含多个值，则 p 必须是标量或者包含与 x 相同数量的值。
        在后一种情况下，结果是 ``x[0]**p[0], x[1]**p[1], ...``。

    Returns
    -------
    out : ndarray or scalar
        ``x**p`` 的结果。如果 x 和 p 都是标量，则 out 也是标量，否则返回数组。

    See Also
    --------
    numpy.power

    Examples
    --------
    >>> np.set_printoptions(precision=4)

    >>> np.emath.power(2, 2)
    4

    >>> np.emath.power([2, 4], 2)
    array([ 4, 16])

    >>> np.emath.power([2, 4], -2)
    array([0.25  ,  0.0625])

    >>> np.emath.power([-2, 4], 2)
    array([ 4.-0.j, 16.+0.j])

    >>> np.emath.power([2, 4], [2, 4])
    array([ 4, 256])

    """
    # 修正 x 中小于零的实数部分为复数
    x = _fix_real_lt_zero(x)
    # 修正 p 中小于零的整数部分为零
    p = _fix_int_lt_zero(p)
    # 使用 nx.power 计算 x 的 p 次方
    return nx.power(x, p)


# 使用 _unary_dispatcher 注册为 array 函数分派的处理器
@array_function_dispatch(_unary_dispatcher)
def arccos(x):
    """
    计算 x 的反余弦值。

    返回 x 的反余弦值的“主值”（有关详细信息，请参阅 `numpy.arccos`）。
    对于实数 x，满足 `abs(x) <= 1`，返回值是闭区间 :math:`[0, \\pi]` 内的实数。
    否则，返回复数的主值。

    Parameters
    ----------
    x : array_like or scalar
       需要求反余弦的值（值）。

    Returns
    -------
    out : ndarray or scalar
       `x` 值的反余弦值。如果 x 是标量，则 out 也是标量，否则返回数组对象。

    See Also
    --------
    numpy.arccos

    Examples
    --------
    >>> np.set_printoptions(precision=4)

    >>> np.emath.arccos(1) # 返回一个标量
    0.0

    >>> np.emath.arccos([1,2])
    array([0.-0.j   , 0.-1.317j])

    """
    # 修正 x 中绝对值大于 1 的实数部分为复数
    x = _fix_real_abs_gt_1(x)
    return nx.arccos(x)


# 使用 _unary_dispatcher 注册为 array 函数分派的处理器
@array_function_dispatch(_unary_dispatcher)
def arcsin(x):
    """
    计算 x 的反正弦值。

    返回 x 的反正弦值的“主值”（有关详细信息，请参阅 `numpy.arcsin`）。
    对于实数 x，满足 `abs(x) <= 1`，返回值是闭区间 :math:`[-\\pi/2, \\pi/2]` 内的实数。
    否则，返回复数的主值。

    Parameters
    ----------
    x : array_like or scalar
       需要求反正弦的值（值）。

    Returns
    -------
    out : ndarray or scalar
       `x` 值的反正弦值。如果 x 是标量，则 out 也是标量，否则返回数组对象。

    """
    # 修正 x 中绝对值大于 1 的实数部分为复数
    x = _fix_real_abs_gt_1(x)
    return nx.arcsin(x)
    # `out` 是函数的返回值，包含了计算出的反正弦值，可以是一个 ndarray 或者标量
    # 如果输入 `x` 是标量，则 `out` 也是标量；如果 `x` 是数组，则返回一个数组对象
    out : ndarray or scalar
       The inverse sine(s) of the `x` value(s). If `x` was a scalar, so
       is `out`, otherwise an array object is returned.

    # 参见 `numpy.arcsin` 函数，用于计算正弦的反函数
    See Also
    --------
    numpy.arcsin

    # 注意事项部分指出，如果希望 `arcsin()` 在实数 `x` 不在区间 `[-1, 1]` 时返回 `NAN`，应使用 `numpy.arcsin`
    Notes
    -----
    For an arcsin() that returns ``NAN`` when real `x` is not in the
    interval ``[-1,1]``, use `numpy.arcsin`.

    # 示例部分演示了函数的使用方法

    # 设置打印精度为 4
    Examples
    --------
    >>> np.set_printoptions(precision=4)

    # 计算 `0` 的反正弦值
    >>> np.emath.arcsin(0)
    0.0

    # 计算数组 `[0, 1]` 中每个元素的反正弦值
    >>> np.emath.arcsin([0,1])
    array([0.    , 1.5708])

    """
    # 调用内部函数 `_fix_real_abs_gt_1(x)` 来确保 `x` 的绝对值不大于 `1`
    x = _fix_real_abs_gt_1(x)
    # 调用 `nx.arcsin(x)` 来计算 `x` 的反正弦值，并返回结果
    return nx.arcsin(x)
# 定义一个装饰器函数，用于分派一元操作的数组函数调度
@array_function_dispatch(_unary_dispatcher)
def arctanh(x):
    """
    Compute the inverse hyperbolic tangent of `x`.

    Return the "principal value" (for a description of this, see
    `numpy.arctanh`) of ``arctanh(x)``. For real `x` such that
    ``abs(x) < 1``, this is a real number.  If `abs(x) > 1`, or if `x` is
    complex, the result is complex. Finally, `x = 1` returns ``inf`` and
    ``x = -1`` returns ``-inf``.

    Parameters
    ----------
    x : array_like
       The value(s) whose arctanh is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The inverse hyperbolic tangent(s) of the `x` value(s). If `x` was
       a scalar so is `out`, otherwise an array is returned.

    See Also
    --------
    numpy.arctanh

    Notes
    -----
    For an arctanh() that returns ``NAN`` when real `x` is not in the
    interval ``(-1,1)``, use `numpy.arctanh` (this latter, however, does
    return +/-inf for ``x = +/-1``).

    Examples
    --------
    >>> np.set_printoptions(precision=4)

    >>> np.emath.arctanh(0.5)
    0.5493061443340549

    >>> from numpy.testing import suppress_warnings
    >>> with suppress_warnings() as sup:
    ...     sup.filter(RuntimeWarning)
    ...     np.emath.arctanh(np.eye(2))
    array([[inf,  0.],
           [ 0., inf]])
    >>> np.emath.arctanh([1j])
    array([0.+0.7854j])

    """
    # 调用内部函数 `_fix_real_abs_gt_1` 处理参数 `x`，确保其实数部分的绝对值大于 1 的情况得到修正
    x = _fix_real_abs_gt_1(x)
    # 返回修正后参数 `x` 的反双曲正切值，由 `nx.arctanh(x)` 计算得出
    return nx.arctanh(x)
```