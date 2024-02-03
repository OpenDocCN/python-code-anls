# `numpy-ml\numpy_ml\utils\distance_metrics.py`

```
import numpy as np

# 计算两个实向量之间的欧几里德（L2）距离
def euclidean(x, y):
    """
    Compute the Euclidean (`L2`) distance between two real vectors

    Notes
    -----
    The Euclidean distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \sqrt{ \sum_i (x_i - y_i)^2  }

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The L2 distance between **x** and **y.
    """
    return np.sqrt(np.sum((x - y) ** 2))


# 计算两个实向量之间的曼哈顿（L1）距离
def manhattan(x, y):
    """
    Compute the Manhattan (`L1`) distance between two real vectors

    Notes
    -----
    The Manhattan distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \sum_i |x_i - y_i|

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The L1 distance between **x** and **y.
    """
    return np.sum(np.abs(x - y))


# 计算两个实向量之间的切比雪夫（L∞）距离
def chebyshev(x, y):
    """
    Compute the Chebyshev (:math:`L_\infty`) distance between two real vectors

    Notes
    -----
    The Chebyshev distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \max_i |x_i - y_i|

    Parameters
    ----------
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between

    Returns
    -------
    d : float
        The Chebyshev distance between **x** and **y.
    """
    return np.max(np.abs(x - y))


# 计算两个实向量之间的闵可夫斯基（Minkowski）距离
def minkowski(x, y, p):
    """
    Compute the Minkowski-`p` distance between two real vectors.

    Notes
    -----
    The Minkowski-`p` distance between two vectors **x** and **y** is

    .. math::

        d(\mathbf{x}, \mathbf{y}) = \left( \sum_i |x_i - y_i|^p \\right)^{1/p}

    Parameters
    ----------
    # 定义两个形状为`(N,)`的ndarray类型的向量x和y，用于计算它们之间的距离
    x,y : :py:class:`ndarray <numpy.ndarray>` s of shape `(N,)`
        The two vectors to compute the distance between
    # 定义距离函数的参数p，当p = 1时，为L1距离，当p = 2时，为L2距离。当p < 1时，闵可夫斯基-p不满足三角不等式，因此不是有效的距离度量
    p : float > 1
        The parameter of the distance function. When `p = 1`, this is the `L1`
        distance, and when `p=2`, this is the `L2` distance. For `p < 1`,
        Minkowski-`p` does not satisfy the triangle inequality and hence is not
        a valid distance metric.

    # 返回值
    Returns
    -------
    # 返回向量x和y之间的闵可夫斯基-p距离
    d : float
        The Minkowski-`p` distance between **x** and **y**.
    """
    # 计算闵可夫斯基-p距离的公式
    return np.sum(np.abs(x - y) ** p) ** (1 / p)
# 计算两个整数向量之间的汉明距离

# 汉明距离是指两个向量 x 和 y 之间的距离
# 其计算方式为：d(x, y) = 1/N * Σ(1_{x_i ≠ y_i})

# 参数：
# x, y：形状为(N,)的numpy.ndarray数组
# 要计算距离的两个向量。这两个向量应为整数值。

# 返回值：
# d：浮点数
# x 和 y 之间的汉明距离。
def hamming(x, y):
    return np.sum(x != y) / len(x)
```