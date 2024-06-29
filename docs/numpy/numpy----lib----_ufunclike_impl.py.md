# `.\numpy\numpy\lib\_ufunclike_impl.py`

```py
"""
Module of functions that are like ufuncs in acting on arrays and optionally
storing results in an output array.

"""
__all__ = ['fix', 'isneginf', 'isposinf']  # 将这些函数添加到模块的公开接口中

import numpy._core.numeric as nx  # 导入 NumPy 核心模块的 numeric 子模块，用别名 nx
from numpy._core.overrides import array_function_dispatch  # 导入数组函数分发装饰器
import warnings  # 导入警告模块
import functools  # 导入 functools 模块


def _dispatcher(x, out=None):
    """
    A simple dispatcher function returning input and output arrays.

    Parameters
    ----------
    x : array_like
        The input array to be processed.
    out : ndarray, optional
        A pre-allocated output array where results are stored if provided.

    Returns
    -------
    tuple
        A tuple containing input and output arrays.
    """
    return (x, out)


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def fix(x, out=None):
    """
    Round to nearest integer towards zero.

    Round an array of floats element-wise to nearest integer towards zero.
    The rounded values are returned as floats.

    Parameters
    ----------
    x : array_like
        An array of floats to be rounded.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that the input broadcasts to. If not provided or None, a
        freshly-allocated array is returned.

    Returns
    -------
    out : ndarray of floats
        A float array with the same dimensions as the input.
        If second argument is not supplied then a float array is returned
        with the rounded values.

        If a second argument is supplied the result is stored there.
        The return value `out` is then a reference to that array.

    See Also
    --------
    rint, trunc, floor, ceil
    around : Round to given number of decimals

    Examples
    --------
    >>> np.fix(3.14)
    3.0
    >>> np.fix(3)
    3.0
    >>> np.fix([2.1, 2.9, -2.1, -2.9])
    array([ 2.,  2., -2., -2.])

    """
    # promote back to an array if flattened
    res = nx.asanyarray(nx.ceil(x, out=out))  # 对输入数组向上取整并转换为任意数组类型
    res = nx.floor(x, out=res, where=nx.greater_equal(x, 0))  # 对非负数部分向下取整

    # when no out argument is passed and no subclasses are involved, flatten
    # scalars
    if out is None and type(res) is nx.ndarray:
        res = res[()]  # 如果没有传入 out 参数且结果不是子类，则将结果扁平化为标量
    return res


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
def isposinf(x, out=None):
    """
    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray of bools
        Boolean array indicating where the input is positive infinity.

    """
    # Function not fully implemented, placeholder for actual implementation
    pass
    # 使用 nx.isinf(x) 函数检查输入数组 x 中的元素是否为正无穷
    is_inf = nx.isinf(x)
    
    # 尝试计算 ~nx.signbit(x)，即 x 中元素的符号位是否为非负
    try:
        signbit = ~nx.signbit(x)
    # 如果出现 TypeError，说明输入 x 包含不支持的数据类型
    except TypeError as e:
        # 获取 x 转换为任意数组后的数据类型
        dtype = nx.asanyarray(x).dtype
        # 抛出详细的 TypeError，指出不支持该数据类型进行此操作
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e
    # 如果没有异常，则计算逻辑与结果并存储到参数 out 中
    else:
        # 返回逻辑与的结果，即正无穷且符号位为非负的元素
        return nx.logical_and(is_inf, signbit, out)
# 使用装饰器为函数添加分派机制，用于在numpy模块中的函数调用
@array_function_dispatch(_dispatcher, verify=False, module='numpy')
# 定义了一个函数isneginf，用于检测输入数组中的元素是否为负无穷，并返回布尔数组
def isneginf(x, out=None):
    """
    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a numpy boolean array is
        returned with values True where the corresponding element of the
        input is negative infinity and values False where the element of
        the input is not negative infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as
        zeros and ones, if the type is boolean then as False and True. The
        return value `out` is then a reference to that array.

    See Also
    --------
    isinf, isposinf, isnan, isfinite

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is also supplied when x is a scalar
    input, if first and second arguments have different shapes, or if the
    first argument has complex values.

    Examples
    --------
    >>> np.isneginf(-np.inf)
    True
    >>> np.isneginf(np.inf)
    False
    >>> np.isneginf([-np.inf, 0., np.inf])
    array([ True, False, False])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([2, 2, 2])
    >>> np.isneginf(x, y)
    array([1, 0, 0])
    >>> y
    array([1, 0, 0])

    """
    # 检测输入数组中的元素是否为正无穷，返回布尔数组
    is_inf = nx.isinf(x)
    try:
        # 检测输入数组中的元素符号位，如果无法比较则引发TypeError异常
        signbit = nx.signbit(x)
    except TypeError as e:
        # 获取输入数组的数据类型，并抛出错误以避免歧义的操作
        dtype = nx.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e
    else:
        # 返回元素级别的逻辑与运算结果，即元素同时为负无穷和带符号位时为True，否则为False
        return nx.logical_and(is_inf, signbit, out)
```