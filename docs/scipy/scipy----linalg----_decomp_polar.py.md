# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_polar.py`

```
# 导入 numpy 库，并将其命名为 np
import numpy as np
# 从 scipy.linalg 库中导入 svd 函数
from scipy.linalg import svd

# 定义 __all__ 变量，用于模块级别的公开接口声明，包括函数名 'polar'
__all__ = ['polar']

# 定义 polar 函数，实现极分解（polar decomposition）
def polar(a, side="right"):
    """
    计算极分解（polar decomposition）。

    返回极分解的因子 `u` 和 `p`，使得当 `side` 为 "right" 时 `a = up`，
    或者当 `side` 为 "left" 时 `a = pu`，其中 `p` 是正半定的。根据 `a` 的形状，
    `u` 的行或列是正交的。当 `a` 是方阵时，`u` 是方阵且是幺正的。当 `a` 不是方阵时，
    计算“标准极分解”。

    Parameters
    ----------
    a : (m, n) array_like
        要进行因子分解的数组。
    side : {'left', 'right'}，可选
        确定是计算右极分解还是左极分解。
        如果 `side` 为 "right"，则 `a = up`。
        如果 `side` 为 "left"，则 `a = pu`。
        默认为 "right"。

    Returns
    -------
    u : (m, n) ndarray
        如果 `a` 是方阵，则 `u` 是幺正的。如果 m > n，则 `a` 的列是正交的；
        如果 m < n，则 `u` 的行是正交的。
    p : ndarray
        `p` 是 Hermite 正半定的。如果 `a` 是非奇异的，`p` 是正定的。
        `p` 的形状为 (n, n) 或 (m, m)，取决于 `side` 是 "right" 还是 "left"。

    References
    ----------
    .. [1] R. A. Horn and C. R. Johnson, "Matrix Analysis", Cambridge
           University Press, 1985.
    .. [2] N. J. Higham, "Functions of Matrices: Theory and Computation",
           SIAM, 2008.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import polar
    >>> a = np.array([[1, -1], [2, 4]])
    >>> u, p = polar(a)
    >>> u
    array([[ 0.85749293, -0.51449576],
           [ 0.51449576,  0.85749293]])
    >>> p
    array([[ 1.88648444,  1.2004901 ],
           [ 1.2004901 ,  3.94446746]])

    一个非方阵的示例，当 m < n 时：

    >>> b = np.array([[0.5, 1, 2], [1.5, 3, 4]])
    >>> u, p = polar(b)
    >>> u
    array([[-0.21196618, -0.42393237,  0.88054056],
           [ 0.39378971,  0.78757942,  0.4739708 ]])
    >>> p
    array([[ 0.48470147,  0.96940295,  1.15122648],
           [ 0.96940295,  1.9388059 ,  2.30245295],
           [ 1.15122648,  2.30245295,  3.65696431]])
    >>> u.dot(p)   # 验证分解结果。
    array([[ 0.5,  1. ,  2. ],
           [ 1.5,  3. ,  4. ]])
    >>> u.dot(u.T)   # `u` 的行是正交的。
    array([[  1.00000000e+00,  -2.07353665e-17],
           [ -2.07353665e-17,   1.00000000e+00]])

    另一个非方阵的示例，当 m > n 时：

    >>> c = b.T
    >>> u, p = polar(c)
    >>> u
    array([[-0.21196618,  0.39378971],
           [-0.42393237,  0.78757942],
           [ 0.88054056,  0.4739708 ]])
    >>> p
    array([[ 1.23116567,  1.93241587],
           [ 1.93241587,  4.84930602]])
    >>> u.dot(p)   # 验证分解结果。
    array([[ 0.5,  1.5],
           [ 1. ,  3. ],
           [ 2. ,  4. ]])

    """

    # 根据 `side` 参数选择右极分解还是左极分解
    if side == "right":
        # 对输入数组 `a` 进行奇异值分解（SVD），得到 `u, s, vh`
        u, s, vh = svd(a, full_matrices=False)
        # 构建对角阵 `p`，根据 `s` 进行填充
        p = np.diag(s)
        # 返回计算得到的 `u` 和 `p`
        return u, p
    elif side == "left":
        # 对输入数组 `a` 的转置进行奇异值分解（SVD），得到 `vh, s, u`
        vh, s, u = svd(a.T, full_matrices=False)
        # 构建对角阵 `p`，根据 `s` 进行填充
        p = np.diag(s)
        # 返回计算得到的 `u` 和 `p`
        return u.T, p
    else:
        # 如果 `side` 参数既不是 "right" 也不是 "left"，则抛出 ValueError 异常
        raise ValueError("side parameter must be 'left' or 'right'")
    >>> u.T.dot(u)  # 计算矩阵 u 的转置乘以自身，用于验证 u 的列向量是正交归一的。
    array([[  1.00000000e+00,  -1.26363763e-16],
           [ -1.26363763e-16,   1.00000000e+00]])

    """
    # 检查参数 `side` 是否为 'right' 或 'left'，否则抛出数值错误
    if side not in ['right', 'left']:
        raise ValueError("`side` must be either 'right' or 'left'")
    
    # 将参数 `a` 转换为 NumPy 数组
    a = np.asarray(a)
    
    # 检查 `a` 是否为二维数组，否则抛出数值错误
    if a.ndim != 2:
        raise ValueError("`a` must be a 2-D array.")
    
    # 对数组 `a` 进行奇异值分解（SVD），返回奇异值 `s` 和左右奇异向量 `w`, `vh`
    w, s, vh = svd(a, full_matrices=False)
    
    # 计算矩阵 `u`，其中 `u = w.dot(vh)`
    u = w.dot(vh)
    
    # 根据参数 `side` 的取值不同，计算投影矩阵 `p`
    if side == 'right':
        # 如果 `side` 为 'right'，则计算 `p = vh.T.conj() * s.dot(vh)`
        p = (vh.T.conj() * s).dot(vh)
    else:
        # 如果 `side` 为 'left'，则计算 `p = w * s.dot(w.T.conj())`
        p = (w * s).dot(w.T.conj())
    
    # 返回计算得到的矩阵 `u` 和 `p`
    return u, p
```