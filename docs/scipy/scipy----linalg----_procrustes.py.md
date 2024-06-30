# `D:\src\scipysrc\scipy\scipy\linalg\_procrustes.py`

```
"""
Solve the orthogonal Procrustes problem.

"""
# 导入NumPy库，用于数值计算
import numpy as np
# 导入本地的_svd模块中的svd函数，用于奇异值分解
from ._decomp_svd import svd


__all__ = ['orthogonal_procrustes']

# 定义函数orthogonal_procrustes，解决正交Procrustes问题
def orthogonal_procrustes(A, B, check_finite=True):
    """
    Compute the matrix solution of the orthogonal Procrustes problem.

    Given matrices A and B of equal shape, find an orthogonal matrix R
    that most closely maps A to B using the algorithm given in [1]_.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix to be mapped.
    B : (M, N) array_like
        Target matrix.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    R : (N, N) ndarray
        The matrix solution of the orthogonal Procrustes problem.
        Minimizes the Frobenius norm of ``(A @ R) - B``, subject to
        ``R.T @ R = I``.
    scale : float
        Sum of the singular values of ``A.T @ B``.

    Raises
    ------
    ValueError
        If the input array shapes don't match or if check_finite is True and
        the arrays contain Inf or NaN.

    Notes
    -----
    Note that unlike higher level Procrustes analyses of spatial data, this
    function only uses orthogonal transformations like rotations and
    reflections, and it does not use scaling or translation.

    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal
           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1966.
           :doi:`10.1007/BF02289451`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import orthogonal_procrustes
    >>> A = np.array([[ 2,  0,  1], [-2,  0,  0]])

    Flip the order of columns and check for the anti-diagonal mapping

    >>> R, sca = orthogonal_procrustes(A, np.fliplr(A))
    >>> R
    array([[-5.34384992e-17,  0.00000000e+00,  1.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  0.00000000e+00, -7.85941422e-17]])
    >>> sca
    9.0

    """
    # 如果check_finite为True，则检查A和B中是否包含无穷大或NaN，将其转换为NumPy数组
    if check_finite:
        A = np.asarray_chkfinite(A)
        B = np.asarray_chkfinite(B)
    else:
        A = np.asanyarray(A)
        B = np.asanyarray(B)
    # 如果A的维度不是2，则引发ValueError异常
    if A.ndim != 2:
        raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
    # 如果A和B的形状不匹配，则引发ValueError异常
    if A.shape != B.shape:
        raise ValueError(f'the shapes of A and B differ ({A.shape} vs {B.shape})')
    
    # 使用奇异值分解求解正交Procrustes问题，保存内存
    u, w, vt = svd(B.T.dot(A).T)
    R = u.dot(vt)  # 计算正交矩阵R
    scale = w.sum()  # 计算奇异值的和作为比例尺度
    return R, scale
```