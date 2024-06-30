# `D:\src\scipysrc\scipy\scipy\spatial\_procrustes.py`

```
"""
This module provides functions to perform full Procrustes analysis.

This code was originally written by Justin Kucynski and ported over from
scikit-bio by Yoshiki Vazquez-Baeza.
"""

# 导入所需的库
import numpy as np
from scipy.linalg import orthogonal_procrustes

# 定义模块中可以导出的函数名
__all__ = ['procrustes']

# 定义 Procrustes 分析函数
def procrustes(data1, data2):
    r"""Procrustes analysis, a similarity test for two data sets.

    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:

    - :math:`tr(AA^{T}) = 1`.

    - Both sets of points are centered around the origin.

    Procrustes ([1]_, [2]_) then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
    pointwise differences between the two input datasets.

    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), simply add columns of zeros to the smaller
    of the two.

    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).

    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.

    Raises
    ------
    ValueError
        If the input arrays are not two-dimensional.
        If the shape of the input arrays is different.
        If the input arrays have zero columns or zero rows.

    See Also
    --------
    scipy.linalg.orthogonal_procrustes
    scipy.spatial.distance.directed_hausdorff : Another similarity test
      for two data sets

    Notes
    -----
    - The disparity should not depend on the order of the input matrices, but
      the output matrices will, as only the first output matrix is guaranteed
      to be scaled such that :math:`tr(AA^{T}) = 1`.

    - Duplicate data points are generally ok, duplicating a data point will
      increase its effect on the procrustes fit.

    - The disparity scales as the number of points per input matrix.

    References
    ----------
    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".

    Examples
    --------
    """
    # 导入必要的库：numpy用于数值计算，scipy.spatial中的procrustes函数用于对齐两个矩阵
    >>> import numpy as np
    >>> from scipy.spatial import procrustes

    # 创建示例矩阵a和b，b是a经过旋转、平移、缩放和镜像后的结果
    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')

    # 对矩阵a和b进行Procrustes分析，得到变换后的矩阵mtx1、mtx2和二者之间的差异度disparity
    >>> mtx1, mtx2, disparity = procrustes(a, b)

    # 对差异度进行四舍五入，结果为0.0
    >>> round(disparity)

    """
    # 将输入数据data1和data2分别转换为np.float64类型的矩阵mtx1和mtx2，并确保复制数据
    mtx1 = np.array(data1, dtype=np.float64, copy=True)
    mtx2 = np.array(data2, dtype=np.float64, copy=True)

    # 检查输入矩阵的维度必须是二维的
    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")

    # 检查输入矩阵的形状必须相同
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")

    # 检查输入矩阵的大小必须大于0
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # 将数据平移到原点
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    # 计算矩阵mtx1和mtx2的范数（模）
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    # 检查范数是否为0，如果是则抛出异常
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # 将数据的缩放调整为使得mtx*mtx'的迹(trace)等于1
    mtx1 /= norm1
    mtx2 /= norm2

    # 对mtx2进行变换，使得其与mtx1尽可能一致
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # 计算两个数据集之间的差异度（相似度度量）
    disparity = np.sum(np.square(mtx1 - mtx2))

    # 返回变换后的矩阵mtx1、mtx2和它们之间的差异度
    return mtx1, mtx2, disparity
```