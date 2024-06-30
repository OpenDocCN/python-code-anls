# `D:\src\scipysrc\scipy\scipy\special\_logsumexp.py`

```
# 导入 numpy 库，并将其命名为 np
import numpy as np
# 从 scipy._lib._util 模块中导入 _asarray_validated 函数
from scipy._lib._util import _asarray_validated

# 定义 __all__ 变量，指定在使用 from ... import * 时被导入的名字列表
__all__ = ["logsumexp", "softmax", "log_softmax"]


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.

        .. versionadded:: 0.11.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.

        .. versionadded:: 0.12.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.

        .. versionadded:: 0.15.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

        .. versionadded:: 0.16.0

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned. If ``return_sign`` is True, ``res`` contains the log of
        the absolute value of the argument.
    sgn : ndarray
        If ``return_sign`` is True, this will be an array of floating-point
        numbers matching res containing +1, 0, -1 (for real-valued inputs)
        or a complex phase (for complex inputs). This gives the sign of the
        argument of the logarithm in ``res``.
        If ``return_sign`` is False, only one result is returned.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2

    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import logsumexp
    >>> a = np.arange(10)
    >>> logsumexp(a)
    9.4586297444267107
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107

    With weights

    >>> a = np.arange(10)
    >>> b = np.arange(10, 0, -1)
    >>> logsumexp(a, b=b)
    9.9170178533034665
    >>> np.log(np.sum(b*np.exp(a)))
    9.9170178533034647

    Returning a sign flag

    >>> logsumexp([1,2],b=[1,-1],return_sign=True)
    (1.5413248546129181, -1.0)

    Notice that `logsumexp` does not directly support masked arrays. To use it
    on a masked array, convert the mask into zero weights:

    >>> a = np.ma.array([np.log(2), 2, np.log(3)],
    """
    # 实现对输入元素的指数求和的对数计算
    # 如果给定 b，则计算带权重的指数求和的对数
    res = np.log(np.sum(b * np.exp(a))) if b is not None else np.log(np.sum(np.exp(a)))
    # 如果 return_sign 为 True，则返回结果及其符号信息
    if return_sign:
        # 计算结果的符号信息，匹配 res 的浮点数数组
        sgn = np.sign(res)
        return res, sgn
    else:
        return res
    """
    Calculate the log-sum-exp of array `a`, optionally adjusting for `b`.
    
    Parameters:
    a : array_like
        Input array.
    b : array_like, optional
        Another array to adjust `a` by.
    axis : int, optional
        Axis or axes over which the operation is performed.
    keepdims : bool, optional
        If True, retains reduced dimensions with length 1.
    return_sign : bool, optional
        If True, returns sign information for complex results.
    
    Returns:
    out : ndarray
        The result of log-sum-exp operation.
    
    """
    
    # Validate and convert `a` to a numpy array ensuring it's valid
    a = _asarray_validated(a, check_finite=False)
    
    # Adjust `a` if `b` is provided to handle broadcasting and zero elements
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf
    
    # Compute the maximum value along the real part of `a` to scale
    initial_value = -np.inf if np.size(a) == 0 else None
    a_max = np.amax(a.real, axis=axis, keepdims=True, initial=initial_value)
    
    # Handle non-finite values in `a_max` to prevent issues
    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0
    
    # Compute `tmp` based on whether `b` is provided
    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)
    
    # Compute the sum `s` of `tmp`, handling division warnings
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
    
        # Adjust `s` and optionally determine sign for complex results
        if return_sign:
            if np.issubdtype(s.dtype, np.complexfloating):
                sgn = s / np.where(s == 0, 1, abs(s))
            else:
                sgn = np.sign(s)
            s = abs(s)
    
    # Compute the logarithm of `s` and adjust dimensions if necessary
    out = np.log(s)
    
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max
    
    # Return the result, optionally with sign information for complex results
    if return_sign:
        return out, sgn
    else:
        return out
# 计算 softmax 函数
def softmax(x, axis=None):
    r"""Compute the softmax function.

    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::

        softmax(x) = np.exp(x)/sum(np.exp(x))

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.

    Notes
    -----
    The formula for the softmax function :math:`\sigma(x)` for a vector
    :math:`x = \{x_0, x_1, ..., x_{n-1}\}` is

    .. math:: \sigma(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}

    The `softmax` function is the gradient of `logsumexp`.

    The implementation uses shifting to avoid overflow. See [1]_ for more
    details.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] P. Blanchard, D.J. Higham, N.J. Higham, "Accurately computing the
       log-sum-exp and softmax functions", IMA Journal of Numerical Analysis,
       Vol.41(4), :doi:`10.1093/imanum/draa038`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import softmax
    >>> np.set_printoptions(precision=5)

    >>> x = np.array([[1, 0.5, 0.2, 3],
    ...               [1,  -1,   7, 3],
    ...               [2,  12,  13, 3]])
    ...

    Compute the softmax transformation over the entire array.

    >>> m = softmax(x)
    >>> m
    array([[  4.48309e-06,   2.71913e-06,   2.01438e-06,   3.31258e-05],
           [  4.48309e-06,   6.06720e-07,   1.80861e-03,   3.31258e-05],
           [  1.21863e-05,   2.68421e-01,   7.29644e-01,   3.31258e-05]])

    >>> m.sum()
    1.0

    Compute the softmax transformation along the first axis (i.e., the
    columns).

    >>> m = softmax(x, axis=0)

    >>> m
    array([[  2.11942e-01,   1.01300e-05,   2.75394e-06,   3.33333e-01],
           [  2.11942e-01,   2.26030e-06,   2.47262e-03,   3.33333e-01],
           [  5.76117e-01,   9.99988e-01,   9.97525e-01,   3.33333e-01]])

    >>> m.sum(axis=0)
    array([ 1.,  1.,  1.,  1.])

    Compute the softmax transformation along the second axis (i.e., the rows).

    >>> m = softmax(x, axis=1)
    >>> m
    array([[  1.05877e-01,   6.42177e-02,   4.75736e-02,   7.82332e-01],
           [  2.42746e-03,   3.28521e-04,   9.79307e-01,   1.79366e-02],
           [  1.22094e-05,   2.68929e-01,   7.31025e-01,   3.31885e-05]])

    >>> m.sum(axis=1)
    array([ 1.,  1.,  1.])

    """
    # 确保输入数据转换为合法数组
    x = _asarray_validated(x, check_finite=False)
    # 计算沿指定轴的最大值
    x_max = np.amax(x, axis=axis, keepdims=True)
    # 计算经过移位的指数函数值
    exp_x_shifted = np.exp(x - x_max)
    # 返回经过 softmax 变换的结果
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)
    r"""Compute the logarithm of the softmax function.

    In principle::

        log_softmax(x) = log(softmax(x))

    but using a more accurate implementation.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : ndarray or scalar
        An array with the same shape as `x`. Exponential of the result will
        sum to 1 along the specified axis. If `x` is a scalar, a scalar is
        returned.

    Notes
    -----
    `log_softmax` is more accurate than ``np.log(softmax(x))`` with inputs that
    make `softmax` saturate (see examples below).

    .. versionadded:: 1.5.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import log_softmax
    >>> from scipy.special import softmax
    >>> np.set_printoptions(precision=5)

    >>> x = np.array([1000.0, 1.0])

    >>> y = log_softmax(x)
    >>> y
    array([   0., -999.])

    >>> with np.errstate(divide='ignore'):
    ...   y = np.log(softmax(x))
    ...
    >>> y
    array([  0., -inf])

    """

    # 将输入 x 转换为合法的数组，确保其内容非有限
    x = _asarray_validated(x, check_finite=False)

    # 在指定的轴上计算 x 的最大值，并保持其维度不变
    x_max = np.amax(x, axis=axis, keepdims=True)

    # 处理 x_max 中的非有限值，将其置为 0
    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0

    # 计算 tmp = x - x_max 和 exp_tmp = exp(tmp)
    tmp = x - x_max
    exp_tmp = np.exp(tmp)

    # 屏蔽关于 log(0) 的警告
    with np.errstate(divide='ignore'):
        # 计算 exp_tmp 指定轴上的和，并保持其维度不变
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        # 计算 log(s) 并赋值给 out
        out = np.log(s)

    # 计算并返回最终结果 tmp - out
    out = tmp - out
    return out
```