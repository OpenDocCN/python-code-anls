# `D:\src\scipysrc\scipy\scipy\_lib\_finite_differences.py`

```
# 从 numpy 库导入所需函数和数组类型
from numpy import arange, newaxis, hstack, prod, array

# 定义一个函数 _central_diff_weights，用于计算 Np 点中心差分的权重
def _central_diff_weights(Np, ndiv=1):
    """
    Return weights for an Np-point central derivative.

    Assumes equally-spaced function points.

    If weights are in the vector w, then
    derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)

    Parameters
    ----------
    Np : int
        Number of points for the central derivative.
    ndiv : int, optional
        Number of divisions. Default is 1.

    Returns
    -------
    w : ndarray
        Weights for an Np-point central derivative. Its size is `Np`.

    Notes
    -----
    Can be inaccurate for a large number of points.

    Examples
    --------
    We can calculate a derivative value of a function.

    >>> def f(x):
    ...     return 2 * x**2 + 3
    >>> x = 3.0 # derivative point
    >>> h = 0.1 # differential step
    >>> Np = 3 # point number for central derivative
    >>> weights = _central_diff_weights(Np) # weights for first derivative
    >>> vals = [f(x + (i - Np/2) * h) for i in range(Np)]
    >>> sum(w * v for (w, v) in zip(weights, vals))/h
    11.79999999999998

    This value is close to the analytical solution:
    f'(x) = 4x, so f'(3) = 12

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Finite_difference

    """
    # 检查输入的 Np 是否小于 ndiv + 1，如果是则引发 ValueError 异常
    if Np < ndiv + 1:
        raise ValueError(
            "Number of points must be at least the derivative order + 1."
        )
    # 检查 Np 是否为偶数，如果是则引发 ValueError 异常
    if Np % 2 == 0:
        raise ValueError("The number of points must be odd.")
    
    # 从 scipy 库导入 linalg 模块
    from scipy import linalg
    
    # 计算 ho（Np 的一半），得到权重计算所需的 x 值范围
    ho = Np >> 1
    x = arange(-ho, ho + 1.0)
    x = x[:, newaxis]
    
    # 初始化 X 矩阵，用于存储 x 的各阶幂
    X = x**0.0
    for k in range(1, Np):
        X = hstack([X, x**k])
    
    # 计算权重向量 w，使用 linalg 模块中的逆函数计算 X 的逆矩阵的第 ndiv 行
    w = prod(arange(1, ndiv + 1), axis=0) * linalg.inv(X)[ndiv]
    
    return w


# 定义一个函数 _derivative，用于计算函数在某点处的第 n 阶导数
def _derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    """
    Find the nth derivative of a function at a point.

    Given a function, use a central difference formula with spacing `dx` to
    compute the nth derivative at `x0`.

    Parameters
    ----------
    func : function
        Input function.
    x0 : float
        The point at which the nth derivative is found.
    dx : float, optional
        Spacing.
    n : int, optional
        Order of the derivative. Default is 1.
    args : tuple, optional
        Arguments
    order : int, optional
        Number of points to use, must be odd.

    Notes
    -----
    Decreasing the step size too small can result in round-off error.

    Examples
    --------
    >>> def f(x):
    ...     return x**3 + x**2
    >>> _derivative(f, 1.0, dx=1e-6)
    4.9999999999217337

    """
    # 检查 order 是否小于 n + 1，如果是则引发 ValueError 异常
    if order < n + 1:
        raise ValueError(
            "'order' (the number of points used to compute the derivative), "
            "must be at least the derivative order 'n' + 1."
        )
    # 检查 order 是否为偶数，如果是则引发 ValueError 异常
    if order % 2 == 0:
        raise ValueError(
            "'order' (the number of points used to compute the derivative) "
            "must be odd."
        )
    # 如果 n 等于 1
    if n == 1:
        # 根据 order 的不同选择预先计算的权重数组
        if order == 3:
            weights = array([-1, 0, 1]) / 2.0
        elif order == 5:
            weights = array([1, -8, 0, 8, -1]) / 12.0
        elif order == 7:
            weights = array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
        elif order == 9:
            weights = array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
        else:
            # 如果 order 不在预定义的范围内，则调用 _central_diff_weights 函数计算权重
            weights = _central_diff_weights(order, 1)
    # 如果 n 等于 2
    elif n == 2:
        # 根据 order 的不同选择预先计算的权重数组
        if order == 3:
            weights = array([1, -2.0, 1])
        elif order == 5:
            weights = array([-1, 16, -30, 16, -1]) / 12.0
        elif order == 7:
            weights = array([2, -27, 270, -490, 270, -27, 2]) / 180.0
        elif order == 9:
            weights = (
                array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9])
                / 5040.0
            )
        else:
            # 如果 order 不在预定义的范围内，则调用 _central_diff_weights 函数计算权重
            weights = _central_diff_weights(order, 2)
    else:
        # 对于其他 n 的情况，调用 _central_diff_weights 函数计算权重
        weights = _central_diff_weights(order, n)
    
    # 初始化计算结果变量为 0
    val = 0.0
    # 计算 order 的一半
    ho = order >> 1
    # 遍历权重数组，计算加权和
    for k in range(order):
        val += weights[k] * func(x0 + (k - ho) * dx, *args)
    
    # 返回加权和除以 n 维度的步长乘积的结果
    return val / prod((dx,) * n, axis=0)
```